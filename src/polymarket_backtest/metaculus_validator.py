from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from statistics import mean
from typing import Any

import httpx

from .grok_replay import DeterministicReplayTransport, build_temporal_system_prompt
from .types import MarketState, NewsItem, OrderLevel, ensure_utc, isoformat

type JsonDict = dict[str, Any]
type QuestionDict = dict[str, Any]

BASE_URL = "https://www.metaculus.com/api"
POSTS_PATH = f"{BASE_URL}/posts/"
DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=30.0)
DEFAULT_PAGE_SIZE = 25
MAX_RETRIES = 4
BACKOFF_BASE_SECONDS = 1.0
PAGE_DELAY_SECONDS = 0.25
AUTH_ENV_VAR = "METACULUS_API_TOKEN"
COMMUNITY_METHODS = (
    "recency_weighted",
    "metaculus_prediction",
    "unweighted",
    "single_aggregation",
)
REQUEST_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (compatible; polymarket-backtest/0.1)",
}


def fetch_resolved_questions(*, limit: int = 50, category: str | None = None) -> list[QuestionDict]:
    """
    Fetch recently resolved binary Metaculus questions.

    Notes:
    - Metaculus now documents binary question listing under `/api/posts/`, not `/api/questions/`.
    - As of 2026-03-12, official API docs indicate authenticated access is required. If anonymous
      access is rejected, set `METACULUS_API_TOKEN`.
    """

    return _fetch_binary_questions(
        status="resolved",
        limit=limit,
        category=category,
        order_by="-scheduled_resolve_time",
        include_resolution=True,
    )


def fetch_open_questions(*, limit: int = 20) -> list[QuestionDict]:
    """Fetch open binary Metaculus questions for live validation."""

    return _fetch_binary_questions(
        status="open",
        limit=limit,
        category=None,
        order_by="-published_at",
        include_resolution=False,
    )


def evaluate_deterministic_forecasts(questions: list[QuestionDict]) -> dict[str, Any]:
    """Score deterministic replay forecasts against Metaculus community predictions."""

    transport = DeterministicReplayTransport()
    per_question: list[dict[str, Any]] = []

    for question in questions:
        community_prediction = _coerce_probability(question.get("community_prediction"))
        resolution = _normalize_resolution(question.get("resolution"))
        if community_prediction is None or resolution is None:
            continue

        context_bundle = _build_context_bundle(question)
        raw_forecast = transport.complete(
            model_release="grok-replay-deterministic",
            system_prompt=build_temporal_system_prompt(context_bundle["as_of"]),
            context_bundle=context_bundle,
        )
        agent_probability = _coerce_probability(raw_forecast.get("probability_yes"))
        if agent_probability is None:
            continue

        agent_brier = round(_brier_score(agent_probability, resolution), 6)
        community_brier = round(_brier_score(community_prediction, resolution), 6)

        per_question.append(
            {
                "id": question["id"],
                "title": question["title"],
                "resolve_time": question.get("resolve_time"),
                "resolution": resolution,
                "agent_probability": round(agent_probability, 6),
                "community_prediction": round(community_prediction, 6),
                "agent_brier": agent_brier,
                "community_brier": community_brier,
                "agent_better": agent_brier < community_brier,
                "confidence": _coerce_probability(raw_forecast.get("confidence")),
                "thesis": str(raw_forecast.get("thesis", "")),
            }
        )

    agent_scores = [item["agent_brier"] for item in per_question]
    community_scores = [item["community_brier"] for item in per_question]

    return {
        "n_questions": len(per_question),
        "agent_brier": round(mean(agent_scores), 6) if agent_scores else 0.0,
        "community_brier": round(mean(community_scores), 6) if community_scores else 0.0,
        "agent_better_count": sum(1 for item in per_question if item["agent_better"]),
        "per_question": per_question,
    }


def run_metaculus_validation(*, n_questions: int = 50) -> str:
    """Fetch resolved questions, evaluate deterministic forecasts, and return a markdown report."""

    try:
        summary = evaluate_deterministic_forecasts(fetch_resolved_questions(limit=n_questions))
    except (RuntimeError, ValueError) as exc:
        return "\n".join(
            [
                "# Metaculus Validation",
                "",
                "- Status: failed",
                f"- Error: {str(exc).replace(chr(10), ' ')}",
            ]
        )

    lines = [
        "# Metaculus Validation",
        "",
        f"- Questions evaluated: {summary['n_questions']}",
        f"- Deterministic agent Brier: {summary['agent_brier']:.4f}",
        f"- Community Brier: {summary['community_brier']:.4f}",
        f"- Agent better count: {summary['agent_better_count']}/{summary['n_questions']}",
    ]

    if not summary["per_question"]:
        lines.extend(
            [
                "",
                "No resolved binary questions with usable community predictions were evaluated.",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            "",
            "## Per Question",
            "",
            "| ID | Title | Agent | Community | Resolution | Agent Brier | Community Brier | Better |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: |",
        ]
    )

    for item in summary["per_question"]:
        lines.append(
            "| "
            f"{item['id']} | "
            f"{_markdown_cell(_truncate(str(item['title']), width=80))} | "
            f"{item['agent_probability']:.3f} | "
            f"{item['community_prediction']:.3f} | "
            f"{item['resolution']:.1f} | "
            f"{item['agent_brier']:.4f} | "
            f"{item['community_brier']:.4f} | "
            f"{'yes' if item['agent_better'] else 'no'} |"
        )

    return "\n".join(lines)


def _fetch_binary_questions(
    *,
    status: str,
    limit: int,
    category: str | None,
    order_by: str,
    include_resolution: bool,
) -> list[QuestionDict]:
    if limit <= 0:
        raise ValueError("limit must be positive")

    questions: list[QuestionDict] = []
    offset = 0
    next_url: str | None = None

    with _build_client() as client:
        while len(questions) < limit:
            page_size = min(50, max(DEFAULT_PAGE_SIZE, limit - len(questions)))
            if next_url is None:
                payload = _request_posts_page(
                    client,
                    params=_build_query_params(
                        status=status,
                        limit=page_size,
                        offset=offset,
                        category=category,
                        order_by=order_by,
                    ),
                )
            else:
                _validate_next_url(next_url)
                payload = _request_posts_page(client, url=next_url)

            results = _extract_results(payload)
            if not results:
                break

            for post in results:
                normalized = _normalize_post_question(post, include_resolution=include_resolution)
                if normalized is None:
                    continue
                questions.append(normalized)
                if len(questions) >= limit:
                    break

            next_url = _extract_next_url(payload)
            offset += len(results)
            if next_url is None and len(results) < page_size:
                break
            if len(questions) < limit:
                time.sleep(PAGE_DELAY_SECONDS)

    return questions[:limit]


def _build_query_params(
    *,
    status: str,
    limit: int,
    offset: int,
    category: str | None,
    order_by: str,
) -> list[tuple[str, str]]:
    params = [
        ("forecast_type", "binary"),
        ("statuses", status),
        ("with_cp", "true"),
        ("include_descriptions", "true"),
        ("limit", str(limit)),
        ("offset", str(offset)),
        ("order_by", order_by),
    ]
    if category:
        params.append(("categories", category))
    return params


def _build_client() -> httpx.Client:
    headers = dict(REQUEST_HEADERS)
    token = os.getenv(AUTH_ENV_VAR)
    if token:
        headers["Authorization"] = f"Token {token}"
    return httpx.Client(
        base_url=BASE_URL,
        headers=headers,
        follow_redirects=True,
        timeout=DEFAULT_TIMEOUT,
    )


def _request_posts_page(
    client: httpx.Client,
    *,
    params: list[tuple[str, str]] | None = None,
    url: str = POSTS_PATH,
) -> JsonDict:
    payload = _request_json(client, url=url, params=params)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Metaculus returned {type(payload).__name__}, expected dict")
    return payload


def _request_json(
    client: httpx.Client,
    *,
    url: str,
    params: list[tuple[str, str]] | None,
) -> Any:
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        response: httpx.Response | None = None
        try:
            response = client.get(url, params=params)
            if response.status_code in {401, 403}:
                raise RuntimeError(_permission_error_message(response))
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = exc
            if not _should_retry(exc, response=response, attempt=attempt):
                break
            time.sleep(_retry_delay_seconds(response=response, attempt=attempt))

    if isinstance(last_error, RuntimeError):
        raise last_error
    raise RuntimeError(f"Metaculus request failed after {MAX_RETRIES} attempts: {url}") from last_error


def _permission_error_message(response: httpx.Response) -> str:
    snippet = " ".join(response.text.split())[:200]
    if "authenticated users" in response.text.lower():
        return (
            "Metaculus API access requires authentication in the current API version. "
            f"Set {AUTH_ENV_VAR} and retry. Response: {snippet}"
        )
    return f"Metaculus request failed with HTTP {response.status_code}: {snippet}"


def _should_retry(
    exc: Exception,
    *,
    response: httpx.Response | None,
    attempt: int,
) -> bool:
    if attempt >= MAX_RETRIES - 1:
        return False
    if isinstance(exc, RuntimeError):
        return False
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {408, 429, 500, 502, 503, 504}
    if response is not None and response.status_code in {408, 429, 500, 502, 503, 504}:
        return True
    return isinstance(exc, httpx.RequestError | json.JSONDecodeError)


def _retry_delay_seconds(*, response: httpx.Response | None, attempt: int) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(1.0, min(float(retry_after), 60.0))
            except ValueError:
                pass
    return BACKOFF_BASE_SECONDS * (2**attempt)


def _extract_results(payload: JsonDict) -> list[JsonDict]:
    results = payload.get("results")
    if not isinstance(results, list):
        raise RuntimeError("Metaculus posts response did not include a results list")
    return [item for item in results if isinstance(item, dict)]


def _extract_next_url(payload: JsonDict) -> str | None:
    next_url = payload.get("next")
    return next_url if isinstance(next_url, str) and next_url else None


def _validate_next_url(next_url: str) -> None:
    if not next_url.startswith(BASE_URL):
        raise ValueError(f"Refusing to follow redirect to untrusted URL: {next_url}")


def _normalize_post_question(post: JsonDict, *, include_resolution: bool) -> QuestionDict | None:
    question = post.get("question")
    if not isinstance(question, dict):
        return None
    if str(question.get("type", "")).lower() != "binary":
        return None

    title = _first_text(question.get("title"), post.get("title"))
    community_prediction = _extract_community_prediction(question, post)
    resolve_time = _resolve_time(question)
    if not title or community_prediction is None:
        return None

    normalized: QuestionDict = {
        "id": _coerce_id(question.get("id"), fallback=post.get("id")),
        "title": title,
        "community_prediction": community_prediction,
        "resolve_time": resolve_time,
        "description": _build_question_description(question),
    }

    if include_resolution:
        resolution = _normalize_resolution(question.get("resolution"))
        if resolution is None:
            return None
        normalized["resolution"] = resolution

    return normalized


def _extract_community_prediction(question: JsonDict, post: JsonDict) -> float | None:
    candidates = (
        question.get("aggregations"),
        post.get("aggregations"),
        question.get("community_prediction"),
        post.get("community_prediction"),
    )
    for candidate in candidates:
        probability = _extract_probability_candidate(candidate)
        if probability is not None:
            return probability
    return None


def _extract_probability_candidate(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, int | float):
        return _coerce_probability(value)

    if isinstance(value, str) and value.strip():
        return _coerce_probability(value)

    if isinstance(value, list):
        if len(value) == 2 and isinstance(value[1], list):
            return _extract_probability_candidate(value[1])
        for item in reversed(value):
            probability = _extract_probability_candidate(item)
            if probability is not None:
                return probability
        return None

    if not isinstance(value, dict):
        return None

    for method in COMMUNITY_METHODS:
        probability = _extract_probability_candidate(value.get(method))
        if probability is not None:
            return probability

    latest = value.get("latest")
    if latest is not None:
        probability = _extract_probability_candidate(latest)
        if probability is not None:
            return probability

    forecast_values = value.get("forecast_values")
    if isinstance(forecast_values, list) and forecast_values:
        probability = _extract_probability_candidate(forecast_values[0])
        if probability is not None:
            return probability

    centers = value.get("centers")
    if isinstance(centers, list) and centers:
        probability = _extract_probability_candidate(centers[0])
        if probability is not None:
            return probability

    for key in ("probability_yes", "community_prediction", "mean", "median", "value"):
        probability = _extract_probability_candidate(value.get(key))
        if probability is not None:
            return probability

    history = value.get("history")
    if isinstance(history, list) and history:
        probability = _extract_probability_candidate(history[-1])
        if probability is not None:
            return probability

    return None


def _normalize_resolution(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        if 0.0 <= float(value) <= 1.0:
            return float(value)
        return None

    normalized = str(value).strip().lower()
    if normalized in {"yes", "positive", "true", "1", "resolved_yes"}:
        return 1.0
    if normalized in {"no", "negative", "false", "0", "resolved_no"}:
        return 0.0
    return None


def _coerce_probability(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return None
    if not 0.0 <= probability <= 1.0:
        return None
    return round(probability, 6)


def _resolve_time(question: JsonDict) -> str | None:
    timestamp = _parse_datetime(
        _first_text(
            question.get("actual_resolve_time"),
            question.get("scheduled_resolve_time"),
            question.get("resolution_set_time"),
            question.get("actual_close_time"),
            question.get("scheduled_close_time"),
        )
    )
    return isoformat(timestamp) if timestamp is not None else None


def _build_question_description(question: JsonDict) -> str:
    blocks: list[str] = []
    description = _first_text(question.get("description"))
    resolution_criteria = _first_text(question.get("resolution_criteria"))
    fine_print = _first_text(question.get("fine_print"))

    if description:
        blocks.append(description)
    if resolution_criteria:
        blocks.append(f"Resolution criteria:\n{resolution_criteria}")
    if fine_print:
        blocks.append(f"Fine print:\n{fine_print}")

    return "\n\n".join(blocks)


def _build_context_bundle(question: QuestionDict) -> JsonDict:
    market = _build_market_state(question)
    news = _build_synthetic_news(question, as_of=market.ts)
    return {
        "as_of": isoformat(market.ts),
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
            for item in news
        ],
        "related_markets": [],
    }


def _build_market_state(question: QuestionDict) -> MarketState:
    community_prediction = _coerce_probability(question.get("community_prediction"))
    if community_prediction is None:
        raise ValueError("question is missing community_prediction")

    as_of = _parse_datetime(question.get("resolve_time")) or datetime.now(tz=UTC)
    spread = 0.01
    best_bid = max(0.0, community_prediction - spread)
    best_ask = min(1.0, community_prediction + spread)

    return MarketState(
        market_id=str(question["id"]),
        title=str(question["title"]),
        domain="metaculus",
        market_type="binary",
        ts=ensure_utc(as_of),
        status="resolved" if _normalize_resolution(question.get("resolution")) is not None else "open",
        best_bid=round(best_bid, 6),
        best_ask=round(best_ask, 6),
        mid=community_prediction,
        last_trade=community_prediction,
        volume_1m=0.0,
        volume_24h=0.0,
        open_interest=0.0,
        tick_size=0.01,
        rules_text=str(question.get("description", "")),
        additional_context="Synthetic Metaculus validation context",
        resolution_ts=_parse_datetime(question.get("resolve_time")),
        fees_enabled=False,
        fee_rate=0.0,
        fee_exponent=1.0,
        maker_rebate_rate=0.0,
        orderbook=[
            OrderLevel(side="bid", price=round(best_bid, 6), quantity=100.0, level_no=1),
            OrderLevel(side="ask", price=round(best_ask, 6), quantity=100.0, level_no=1),
        ],
    )


def _build_synthetic_news(question: QuestionDict, *, as_of: datetime) -> list[NewsItem]:
    description = str(question.get("description", "")).strip()
    return [
        NewsItem(
            document_id=f"metaculus:{question['id']}:description",
            source="metaculus",
            url=f"https://www.metaculus.com/questions/{question['id']}/",
            title=str(question["title"]),
            published_ts=ensure_utc(as_of),
            first_seen_ts=ensure_utc(as_of),
            ingested_ts=ensure_utc(as_of),
            content=description or str(question["title"]),
            metadata={"impact": 0.0},
        )
    ]


def _parse_datetime(value: Any) -> datetime | None:
    text = _first_text(value)
    if not text:
        return None
    try:
        return ensure_utc(datetime.fromisoformat(text.replace("Z", "+00:00")))
    except ValueError:
        return None


def _coerce_id(value: Any, *, fallback: Any) -> int | str:
    candidate = value if value is not None else fallback
    if isinstance(candidate, int):
        return candidate
    if isinstance(candidate, str):
        stripped = candidate.strip()
        if stripped.isdigit():
            return int(stripped)
        return stripped
    return str(candidate)


def _first_text(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _brier_score(probability: float, outcome: float) -> float:
    return (probability - outcome) ** 2


def _truncate(value: str, *, width: int) -> str:
    if len(value) <= width:
        return value
    return value[: max(0, width - 3)].rstrip() + "..."


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


__all__ = [
    "evaluate_deterministic_forecasts",
    "fetch_open_questions",
    "fetch_resolved_questions",
    "run_metaculus_validation",
]
