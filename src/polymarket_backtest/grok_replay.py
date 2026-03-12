from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from . import db
from .types import ForecastInput, ForecastOutput, MarketState, NewsItem, ensure_utc, isoformat


FORBIDDEN_XAI_TOOLS = {
    "web_search",
    "x_search",
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


class ForecastTransport(Protocol):
    def complete(
        self,
        *,
        model_release: str,
        system_prompt: str,
        context_bundle: dict[str, Any],
    ) -> dict[str, Any]:
        ...


@dataclass
class DeterministicReplayTransport:
    agent_name: str = "grok_replay"
    model_id: str = "grok"

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
    api_key_env: str = "XAI_API_KEY"
    api_url: str = "https://api.x.ai/v1/responses"

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
        text_chunks: list[str] = []
        for item in body.get("output", []):
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    text_chunks.append(content.get("text", ""))
        parsed = json.loads("".join(text_chunks))
        parsed.setdefault("agent_name", "grok_replay")
        parsed.setdefault("model_id", "grok")
        parsed.setdefault("model_release", model_release)
        return parsed


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
                "resolution_ts": isoformat(market.resolution_ts) if market.resolution_ts else None,
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
