from __future__ import annotations

import hashlib
import json
import logging
import os
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Protocol

import httpx

from . import db
from .types import ForecastInput, ForecastOutput, ensure_utc, isoformat

logger = logging.getLogger(__name__)

FORBIDDEN_XAI_TOOLS = {
    "web_search",
    "code_execution",
    "code_interpreter",
    "attachment_search",
    "collections_search",
    "file_search",
}


def stable_hash(payload: Any) -> str:
    body = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def build_temporal_system_prompt(as_of: str) -> str:
    return (
        "You are running inside a historical replay.\n\n"
        f"Current replay time: {as_of}\n"
        "You must behave as if nothing after this timestamp exists.\n\n"
        "Hard rules:\n"
        "1. Use only the supplied replay context.\n"
        "2. Never assume live web, live X, future prices, or final outcomes.\n"
        "3. If evidence is insufficient as of the replay time, say so.\n"
        "4. Return calibrated probability_yes, confidence, thesis, reasoning, "
        "and evidence.\n"
        "5. Output strict JSON only."
    )


def _require_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        raise RuntimeError(f"{var_name} is not set")
    return value


def _extract_output_text(body: dict[str, Any]) -> str:
    output_text = body.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    text_chunks: list[str] = []
    for item in body.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                text_chunks.append(content.get("text", ""))
    text = "".join(text_chunks).strip()
    if not text:
        raise ValueError("xAI response did not include output_text content")
    return text


def _parse_forecast_output(body: dict[str, Any], *, model_release: str) -> dict[str, Any]:
    parsed = json.loads(_extract_output_text(body))
    if not isinstance(parsed, dict):
        raise ValueError("Forecast output must be a JSON object")
    try:
        probability = float(parsed.get("probability_yes", 0.5))
        confidence = float(parsed.get("confidence", 0.5))
        expected_edge_bps = float(parsed.get("expected_edge_bps", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("Forecast output numeric fields must contain valid numbers") from exc
    parsed["probability_yes"] = max(0.001, min(0.999, probability))
    parsed["confidence"] = max(0.0, min(1.0, confidence))
    parsed["expected_edge_bps"] = expected_edge_bps
    evidence = parsed.get("evidence", [])
    if evidence is None:
        evidence = []
    if not isinstance(evidence, list):
        raise ValueError("Forecast output evidence must be a JSON array")
    parsed["evidence"] = evidence
    parsed["thesis"] = str(parsed.get("thesis", ""))
    parsed["reasoning"] = str(parsed.get("reasoning", ""))
    parsed.setdefault("agent_name", "grok_replay")
    parsed.setdefault("model_id", "grok")
    parsed.setdefault("model_release", model_release)
    return parsed


def _x_search_date_bounds(as_of: str, *, lookback_days: int) -> tuple[str, str]:
    as_of_dt = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
    to_date = as_of_dt.date()
    from_date = to_date - timedelta(days=lookback_days)
    return from_date.isoformat(), to_date.isoformat()


class ForecastTransport(Protocol):
    is_live_safe: bool

    def complete(
        self,
        *,
        model_release: str,
        system_prompt: str,
        context_bundle: dict[str, Any],
    ) -> dict[str, Any]: ...


@dataclass
class DeterministicReplayTransport:
    """FAKE deterministic forecaster for backtesting only.

    This transport does NOT call any real LLM or forecast API.  It applies
    hard-coded heuristics to the market mid-price:

    * +0.35 when news title/content contains "suspend"/"suspending"
    * +0.02 when news title/content contains "no change"/"unchanged"

    Because the rules are trivial, backtest results produced with this
    transport are **meaningless for strategy selection**.  They only verify
    that the replay pipeline executes end-to-end without errors.

    For any production or strategy-evaluation decision you MUST use a real
    Grok API transport (``XAIResponsesTransport`` or ``XAISearchTransport``)
    by setting ``FORECAST_MODE=xai`` or ``FORECAST_MODE=xai_search`` and
    providing a valid ``XAI_API_KEY``.
    """

    agent_name: str = "grok_replay"
    model_id: str = "grok"
    is_live_safe: bool = False

    def complete(
        self,
        *,
        model_release: str,
        system_prompt: str,
        context_bundle: dict[str, Any],
    ) -> dict[str, Any]:
        market = context_bundle["market"]
        news = context_bundle["recent_news"]
        probability = float(market["mid"])
        confidence = 0.52
        evidence: list[dict[str, Any]] = []

        for item in news:
            metadata = item.get("metadata", {})
            impact = float(metadata.get("impact", 0.0))
            title_lower = item["title"].lower()
            content_lower = item["content"].lower()
            if "suspend" in title_lower or "suspending" in content_lower:
                probability += 0.35 + impact
                confidence += 0.22
            elif "no change" in title_lower or "unchanged" in content_lower:
                probability += 0.02 + impact
                confidence += 0.08
            evidence.append(
                {
                    "document_id": item["document_id"],
                    "source": item["source"],
                    "title": item["title"],
                }
            )

        probability = min(0.995, max(0.01, round(probability, 4)))
        confidence = min(0.99, max(0.05, round(confidence, 4)))
        best_ask = float(market["best_ask"])
        edge_bps = round((probability - best_ask) * 10_000.0, 2)
        thesis = "Buy YES" if edge_bps > 0 else "No trade"
        reasoning = (
            "The model adjusts the market-implied probability using only "
            "documents published and ingested by the replay clock."
        )
        return {
            "agent_name": self.agent_name,
            "model_id": self.model_id,
            "model_release": model_release,
            "probability_yes": probability,
            "confidence": confidence,
            "expected_edge_bps": edge_bps,
            "thesis": thesis,
            "reasoning": reasoning,
            "evidence": evidence,
        }


@dataclass
class XAIResponsesTransport:
    """Live transport that calls the xAI Responses API.

    Users must set the correct model ID for their xAI API access via
    the ``GROK_MODEL_RELEASE`` environment variable or the ``model_release``
    parameter passed to ``ReplayGrokClient``.
    """

    api_key_env: str = "XAI_API_KEY"
    api_url: str = "https://api.x.ai/v1/responses"
    is_live_safe: bool = True

    def complete(
        self,
        *,
        model_release: str,
        system_prompt: str,
        context_bundle: dict[str, Any],
    ) -> dict[str, Any]:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} is not set")
        payload = {
            "model": model_release,
            "temperature": 0,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(context_bundle, sort_keys=True)},
            ],
            "tools": [],
        }
        if any(tool.get("type") in FORBIDDEN_XAI_TOOLS for tool in payload["tools"]):
            raise ValueError("Built-in xAI tools are forbidden in replay mode")
        request = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
        return _parse_forecast_output(body, model_release=model_release)


@dataclass
class XAISearchTransport:
    """Live transport that calls the xAI Responses API with x_search tool.

    Users must set the correct model ID for their xAI API access via
    the ``GROK_MODEL_RELEASE`` environment variable or the ``model_release``
    parameter passed to ``ReplayGrokClient``.
    """

    api_key_env: str = "XAI_API_KEY"
    api_url: str = "https://api.x.ai/v1/responses"
    search_window_days: int = 30
    timeout_seconds: float = 30.0
    is_live_safe: bool = True

    def complete(
        self,
        *,
        model_release: str,
        system_prompt: str,
        context_bundle: dict[str, Any],
    ) -> dict[str, Any]:
        api_key = _require_env(self.api_key_env)
        from_date, to_date = _x_search_date_bounds(
            str(context_bundle["as_of"]),
            lookback_days=self.search_window_days,
        )
        payload = {
            "model": model_release,
            "temperature": 0,
            "store": False,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(context_bundle, sort_keys=True)},
            ],
            "tools": [
                {
                    "type": "x_search",
                    "x_search": {
                        "from_date": from_date,
                        "to_date": to_date,
                    },
                }
            ],
        }
        if any(tool.get("type") in FORBIDDEN_XAI_TOOLS for tool in payload["tools"]):
            raise ValueError("Built-in xAI tools are forbidden in replay mode")

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                self.api_url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            body = response.json()

        system_fingerprint = body.get("system_fingerprint")
        if system_fingerprint:
            logger.info(
                "xAI replay search fingerprint=%s model_release=%s from_date=%s to_date=%s",
                system_fingerprint,
                model_release,
                from_date,
                to_date,
            )
        return _parse_forecast_output(body, model_release=model_release)


@dataclass
class SmartRuleTransport:
    """Rule-based forecaster that uses price level, trend, and resolution proximity.

    NOT a real forecaster — but much smarter than DeterministicReplayTransport.
    Uses actual market signals to generate plausible probability estimates:

    1. Price level: extreme prices (>0.90 or <0.10) are reinforced
    2. Price trend: compares mid vs last_trade for directional signal
    3. Resolution proximity: closer to resolution = more confidence in extremes
    4. Volume: higher volume = more confidence in current price

    Still NOT safe for live trading — use real xAI transport for that.
    """

    agent_name: str = "smart_rules"
    model_id: str = "smart_rules"
    is_live_safe: bool = False

    def complete(
        self,
        *,
        model_release: str,
        system_prompt: str,
        context_bundle: dict[str, Any],
    ) -> dict[str, Any]:
        market = context_bundle["market"]
        mid = float(market["mid"])
        best_ask = float(market["best_ask"])
        best_bid = float(market["best_bid"])
        last_trade = float(market["last_trade"])
        volume_24h = float(market.get("volume_24h", 0.0))

        # --- Signal 1: Price level (extreme prices tend to resolve in that direction) ---
        # Markets at 0.92+ usually resolve YES; markets at 0.08- usually resolve NO
        if mid >= 0.92:
            level_signal = 0.03 + (mid - 0.92) * 0.5  # stronger as it gets closer to 1
        elif mid <= 0.08:
            level_signal = -0.03 - (0.08 - mid) * 0.5
        elif mid >= 0.75:
            level_signal = 0.015
        elif mid <= 0.25:
            level_signal = -0.015
        else:
            level_signal = 0.0

        # --- Signal 2: Price trend (mid vs last_trade) ---
        trend = mid - last_trade
        # Dampen trend signal to avoid chasing noise
        trend_signal = max(-0.03, min(0.03, trend * 0.5))

        # --- Signal 3: Resolution proximity ---
        resolution_ts_str = market.get("resolution_ts")
        proximity_boost = 1.0
        if resolution_ts_str:
            try:
                as_of_str = context_bundle.get("as_of", "")
                if as_of_str:
                    as_of_dt = datetime.fromisoformat(str(as_of_str).replace("Z", "+00:00"))
                    res_dt = datetime.fromisoformat(str(resolution_ts_str).replace("Z", "+00:00"))
                    hours_to_res = max(0, (res_dt - as_of_dt).total_seconds() / 3600)
                    if hours_to_res < 72:
                        # Near resolution: amplify the level signal
                        proximity_boost = 1.0 + max(0, (72 - hours_to_res) / 72) * 0.5
            except (ValueError, TypeError):
                pass

        # --- Signal 4: Volume confidence ---
        # Higher volume = more confidence in current price
        if volume_24h > 50_000:
            volume_confidence = 0.75
        elif volume_24h > 10_000:
            volume_confidence = 0.65
        elif volume_24h > 1_000:
            volume_confidence = 0.58
        else:
            volume_confidence = 0.52

        # --- Combine signals ---
        adjustment = level_signal * proximity_boost + trend_signal
        probability = mid + adjustment
        probability = min(0.995, max(0.005, round(probability, 4)))

        confidence = min(0.95, max(0.10, volume_confidence + abs(level_signal) * 2))

        edge_bps = round((probability - best_ask) * 10_000.0, 2)
        if probability < best_bid:
            thesis = "Sell YES / price likely to fall"
        elif edge_bps > 0:
            thesis = "Buy YES"
        else:
            thesis = "No trade"

        return {
            "agent_name": self.agent_name,
            "model_id": self.model_id,
            "model_release": model_release,
            "probability_yes": probability,
            "confidence": confidence,
            "expected_edge_bps": edge_bps,
            "thesis": thesis,
            "reasoning": (
                f"Rule-based: level_signal={level_signal:+.4f} "
                f"trend_signal={trend_signal:+.4f} "
                f"proximity_boost={proximity_boost:.2f}"
            ),
            "evidence": [],
        }


def create_transport(*, mode: str, model_release: str) -> ForecastTransport:
    """Factory that selects a forecast transport based on *mode*.

    Args:
        mode: One of ``"deterministic"``, ``"xai"``, or ``"xai_search"``.
        model_release: The xAI model release identifier (e.g. ``"grok-3"``).
            Ignored for deterministic mode but validated for live transports.
    """
    if mode == "deterministic":
        return DeterministicReplayTransport()
    if mode == "smart_rules":
        return SmartRuleTransport()
    if mode == "ml_model":
        from .ml_transport import MLModelTransport

        return MLModelTransport()
    if mode in {"xai", "xai_no_search"}:
        _require_env("XAI_API_KEY")
        return XAIResponsesTransport()
    if mode == "xai_search":
        _require_env("XAI_API_KEY")
        return XAISearchTransport()
    raise ValueError(
        f"Unknown transport mode {mode!r} for model_release={model_release!r}; "
        "expected one of: deterministic, xai, xai_search"
    )


def validate_transport_for_live(transport: ForecastTransport) -> None:
    """Raise if transport is not safe for live trading."""
    if not transport.is_live_safe:
        raise RuntimeError(
            "Cannot use DeterministicReplayTransport for live trading. Set FORECAST_MODE=xai and provide XAI_API_KEY."
        )


@dataclass
class ReplayContextBuilder:
    conn: Any
    experiment_id: int | None
    lookback_minutes: int

    def build(self, market_id: str, as_of: Any) -> ForecastInput:
        market = db.get_market_state_as_of(self.conn, market_id, as_of)
        if market is None:
            raise ValueError(f"No market snapshot for {market_id} at {as_of}")
        news = db.get_market_news_as_of(self.conn, market_id, as_of, self.lookback_minutes)
        related = db.get_related_markets_as_of(self.conn, market_id, as_of)

        db.record_audit(
            self.conn,
            experiment_id=self.experiment_id,
            market_id=market_id,
            ts=ensure_utc(as_of),
            tool_name="get_market_state_as_of",
            request_max_ts=ensure_utc(as_of),
            result_max_ts=market.ts,
            row_count=1,
            request_json={"market_id": market_id, "as_of": isoformat(as_of)},
            response_payload={"market_id": market.market_id, "snapshot_ts": isoformat(market.ts)},
        )
        latest_news_ts = max((item.ingested_ts for item in news), default=ensure_utc(as_of))
        db.record_audit(
            self.conn,
            experiment_id=self.experiment_id,
            market_id=market_id,
            ts=ensure_utc(as_of),
            tool_name="get_market_news_as_of",
            request_max_ts=ensure_utc(as_of),
            result_max_ts=min(latest_news_ts, ensure_utc(as_of)),
            row_count=len(news),
            request_json={
                "market_id": market_id,
                "as_of": isoformat(as_of),
                "lookback_minutes": self.lookback_minutes,
            },
            response_payload=[item.document_id for item in news],
        )
        return ForecastInput(
            as_of=ensure_utc(as_of),
            market=market,
            recent_news=news,
            related_markets=related,
        )


@dataclass
class ReplayGrokClient:
    conn: Any
    experiment_id: int | None
    model_id: str
    model_release: str
    transport: ForecastTransport
    lookback_minutes: int = 240

    def __post_init__(self) -> None:
        if self.model_release.endswith("-latest") or self.model_release == self.model_id:
            raise ValueError("Replay mode requires a release-pinned model version")
        self.context_builder = ReplayContextBuilder(
            conn=self.conn,
            experiment_id=self.experiment_id,
            lookback_minutes=self.lookback_minutes,
        )

    def prompt_hash(self, as_of: Any) -> str:
        return stable_hash(build_temporal_system_prompt(isoformat(as_of)))

    def _bundle(self, context: ForecastInput) -> dict[str, Any]:
        market = context.market
        return {
            "as_of": isoformat(context.as_of),
            "market": {
                "market_id": market.market_id,
                "title": market.title,
                "domain": market.domain,
                "status": market.status,
                "best_bid": market.best_bid,
                "best_ask": market.best_ask,
                "mid": market.mid,
                "last_trade": market.last_trade,
                "volume_1m": market.volume_1m,
                "volume_24h": market.volume_24h,
                "open_interest": market.open_interest,
                "tick_size": market.tick_size,
                "rules_text": market.rules_text,
                "additional_context": market.additional_context,
                "resolution_ts": isoformat(res_ts) if (res_ts := market.resolution_ts) is not None else None,
            },
            "recent_news": [
                {
                    "document_id": item.document_id,
                    "source": item.source,
                    "url": item.url,
                    "title": item.title,
                    "published_ts": isoformat(item.published_ts),
                    "first_seen_ts": isoformat(item.first_seen_ts),
                    "ingested_ts": isoformat(item.ingested_ts),
                    "content": item.content,
                    "metadata": item.metadata,
                }
                for item in context.recent_news
            ],
            "related_markets": context.related_markets,
        }

    def forecast(self, market_id: str, as_of: Any) -> tuple[ForecastOutput, str, str]:
        context = self.context_builder.build(market_id, as_of)
        context_bundle = self._bundle(context)
        system_prompt = build_temporal_system_prompt(context_bundle["as_of"])
        prompt_hash = stable_hash(system_prompt)
        context_hash = stable_hash(context_bundle)
        raw = self.transport.complete(
            model_release=self.model_release,
            system_prompt=system_prompt,
            context_bundle=context_bundle,
        )
        output = ForecastOutput(
            agent_name=str(raw["agent_name"]),
            model_id=self.model_id,
            model_release=self.model_release,
            as_of=context.as_of,
            market_id=market_id,
            domain=context.market.domain,
            probability_yes=float(raw["probability_yes"]),
            confidence=float(raw["confidence"]),
            expected_edge_bps=float(raw["expected_edge_bps"]),
            thesis=str(raw["thesis"]),
            reasoning=str(raw["reasoning"]),
            evidence=list(raw.get("evidence", [])),
            raw_response=raw,
        )
        return output, prompt_hash, context_hash
