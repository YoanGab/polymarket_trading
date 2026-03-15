from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from importlib import resources
from pathlib import Path
from typing import Any

from .types import MarketState, NewsItem, OrderLevel, ensure_utc, isoformat


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    # Run column migrations BEFORE schema script so that indexes on new columns succeed.
    columns = {row[1] for row in conn.execute("PRAGMA table_info(market_snapshots)").fetchall()}
    if columns and "is_synthetic" not in columns:
        conn.execute("ALTER TABLE market_snapshots ADD COLUMN is_synthetic INTEGER NOT NULL DEFAULT 0")
    market_columns = {row[1] for row in conn.execute("PRAGMA table_info(markets)").fetchall()}
    if market_columns:
        if "event_id" not in market_columns:
            conn.execute("ALTER TABLE markets ADD COLUMN event_id TEXT")
        if "tags_json" not in market_columns:
            conn.execute("ALTER TABLE markets ADD COLUMN tags_json TEXT NOT NULL DEFAULT '[]'")
    conn.commit()

    schema = resources.files("polymarket_backtest").joinpath("schema.sql").read_text()
    conn.executescript(schema)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_markets_event_id ON markets (event_id)")
    conn.commit()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True)


def _hash(payload: Any) -> str:
    return hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()


def _coerce_iso8601(value: datetime | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return isoformat(value)
    return value


def _coerce_tags_json(value: Any) -> str:
    if value is None:
        return "[]"
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return "[]"
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return _json([stripped])
        if isinstance(parsed, list):
            return _json([str(item).strip() for item in parsed if str(item).strip()])
        return "[]"
    if isinstance(value, list | tuple | set):
        return _json([str(item).strip() for item in value if str(item).strip()])
    return "[]"


def _parse_tags_json(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return []
    else:
        parsed = value
    if not isinstance(parsed, list):
        return []
    tags: list[str] = []
    seen: set[str] = set()
    for item in parsed:
        tag = str(item).strip()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        tags.append(tag)
    return tags


def _inserted_count(conn: sqlite3.Connection, total_changes_before: int) -> int:
    return conn.total_changes - total_changes_before


def get_event_outcome_tokens_map(
    conn: sqlite3.Connection,
    market_ids: list[str],
) -> dict[str, list[str]]:
    unique_market_ids = list(dict.fromkeys(market_ids))
    if not unique_market_ids:
        return {}

    outcome_tokens_by_market: dict[str, list[str]] = {}
    for start in range(0, len(unique_market_ids), 900):
        chunk = unique_market_ids[start : start + 900]
        placeholders = ", ".join("?" for _ in chunk)
        rows = conn.execute(
            f"""
            SELECT
                base.market_id AS base_market_id,
                sibling.market_id AS outcome_market_id
            FROM markets base
            JOIN markets sibling
              ON sibling.event_id = base.event_id
            WHERE base.market_id IN ({placeholders})
              AND base.event_id IS NOT NULL
            ORDER BY
                base.market_id,
                CASE WHEN sibling.market_id = base.market_id THEN 0 ELSE 1 END,
                sibling.title COLLATE NOCASE,
                sibling.market_id
            """,
            tuple(chunk),
        ).fetchall()
        for row in rows:
            base_market_id = str(row["base_market_id"])
            outcome_market_id = str(row["outcome_market_id"])
            outcome_tokens_by_market.setdefault(base_market_id, []).append(outcome_market_id)
    return outcome_tokens_by_market


def get_event_outcome_tokens(
    conn: sqlite3.Connection,
    market_id: str,
) -> list[str]:
    return list(get_event_outcome_tokens_map(conn, [market_id]).get(market_id, []))


def create_experiment(
    conn: sqlite3.Connection,
    *,
    name: str,
    model_id: str,
    model_release: str,
    system_prompt_hash: str,
    config: dict[str, Any],
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO experiments (
            name, model_id, model_release, system_prompt_hash, config_json
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (name, model_id, model_release, system_prompt_hash, _json(config)),
    )
    conn.commit()
    row_id = cursor.lastrowid
    if row_id is None:
        raise RuntimeError("INSERT did not return a lastrowid")
    return int(row_id)


def add_market(
    conn: sqlite3.Connection,
    *,
    market_id: str,
    title: str,
    domain: str,
    market_type: str,
    open_ts: datetime,
    close_ts: datetime | None,
    resolution_ts: datetime | None,
    status: str,
    event_id: str | None = None,
    tags: list[str] | None = None,
    fees_enabled: bool = False,
    fee_rate: float = 0.0,
    fee_exponent: float = 0.0,
    maker_rebate_rate: float = 0.0,
) -> None:
    conn.execute(
        """
        INSERT INTO markets (
            market_id, title, domain, market_type, event_id, tags_json, open_ts, close_ts,
            resolution_ts, status, fees_enabled, fee_rate, fee_exponent,
            maker_rebate_rate
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            market_id,
            title,
            domain,
            market_type,
            event_id,
            _coerce_tags_json(tags),
            isoformat(open_ts),
            isoformat(close_ts) if close_ts else None,
            isoformat(resolution_ts) if resolution_ts else None,
            status,
            int(fees_enabled),
            fee_rate,
            fee_exponent,
            maker_rebate_rate,
        ),
    )


def bulk_add_markets(conn: sqlite3.Connection, markets: list[dict[str, Any]]) -> int:
    rows = [
        (
            market["market_id"],
            market["title"],
            market["domain"],
            market["market_type"],
            market.get("event_id"),
            _coerce_tags_json(market.get("tags")),
            _coerce_iso8601(market["open_ts"]),
            _coerce_iso8601(market.get("close_ts")),
            _coerce_iso8601(market.get("resolution_ts")),
            market["status"],
            int(bool(market.get("fees_enabled", False))),
            market.get("fee_rate", 0.0),
            market.get("fee_exponent", 0.0),
            market.get("maker_rebate_rate", 0.0),
        )
        for market in markets
    ]
    total_changes_before = conn.total_changes
    conn.executemany(
        """
        INSERT OR IGNORE INTO markets (
            market_id, title, domain, market_type, event_id, tags_json, open_ts, close_ts,
            resolution_ts, status, fees_enabled, fee_rate, fee_exponent,
            maker_rebate_rate
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    inserted_count = _inserted_count(conn, total_changes_before)
    conn.commit()
    return inserted_count


def add_rule_revision(
    conn: sqlite3.Connection,
    *,
    market_id: str,
    effective_ts: datetime,
    rules_text: str,
    additional_context: str = "",
    bulletin_ref: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO market_rule_revisions (
            market_id, effective_ts, rules_text, additional_context, bulletin_ref
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (
            market_id,
            isoformat(effective_ts),
            rules_text,
            additional_context,
            bulletin_ref,
        ),
    )


def bulk_add_rule_revisions(conn: sqlite3.Connection, revisions: list[dict[str, Any]]) -> int:
    rows = [
        (
            revision["market_id"],
            _coerce_iso8601(revision["effective_ts"]),
            revision["rules_text"],
            revision["additional_context"],
        )
        for revision in revisions
    ]
    total_changes_before = conn.total_changes
    conn.executemany(
        """
        INSERT INTO market_rule_revisions (
            market_id, effective_ts, rules_text, additional_context
        ) VALUES (?, ?, ?, ?)
        """,
        rows,
    )
    inserted_count = _inserted_count(conn, total_changes_before)
    conn.commit()
    return inserted_count


def add_snapshot(
    conn: sqlite3.Connection,
    *,
    market_id: str,
    ts: datetime,
    status: str,
    best_bid: float,
    best_ask: float,
    last_trade: float,
    volume_1m: float,
    volume_24h: float,
    open_interest: float,
    tick_size: float,
    orderbook: list[tuple[str, int, float, float]],
) -> None:
    mid = round((best_bid + best_ask) / 2.0, 4)
    cursor = conn.execute(
        """
        INSERT INTO market_snapshots (
            market_id, ts, status, best_bid, best_ask, mid, last_trade,
            volume_1m, volume_24h, open_interest, tick_size, features_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            market_id,
            isoformat(ts),
            status,
            best_bid,
            best_ask,
            mid,
            last_trade,
            volume_1m,
            volume_24h,
            open_interest,
            tick_size,
            "{}",
        ),
    )
    if cursor.lastrowid is None:
        raise RuntimeError("INSERT did not return a lastrowid")
    snapshot_id = int(cursor.lastrowid)
    conn.executemany(
        """
        INSERT INTO orderbook_levels (
            snapshot_id, side, level_no, price, quantity
        ) VALUES (?, ?, ?, ?, ?)
        """,
        [(snapshot_id, side, level_no, price, quantity) for side, level_no, price, quantity in orderbook],
    )


def bulk_add_snapshots(conn: sqlite3.Connection, snapshots: list[dict[str, Any]]) -> int:
    rows = [
        (
            snapshot["market_id"],
            _coerce_iso8601(snapshot["ts"]),
            snapshot["status"],
            snapshot["best_bid"],
            snapshot["best_ask"],
            (snapshot["best_bid"] + snapshot["best_ask"]) / 2.0,
            snapshot["last_trade"],
            snapshot["volume_1m"],
            snapshot["volume_24h"],
            snapshot["open_interest"],
            snapshot["tick_size"],
            int(snapshot.get("interpolated", False) or snapshot.get("forward_filled", False)),
            "{}",
        )
        for snapshot in snapshots
    ]
    total_changes_before = conn.total_changes
    conn.executemany(
        """
        INSERT OR IGNORE INTO market_snapshots (
            market_id, ts, status, best_bid, best_ask, mid, last_trade,
            volume_1m, volume_24h, open_interest, tick_size, is_synthetic,
            features_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    inserted_count = _inserted_count(conn, total_changes_before)
    conn.commit()
    return inserted_count


def add_news(
    conn: sqlite3.Connection,
    *,
    document_id: str,
    source: str,
    url: str,
    title: str,
    published_ts: datetime,
    first_seen_ts: datetime,
    ingested_ts: datetime,
    content: str,
    metadata: dict[str, Any],
    market_ids: list[str],
) -> None:
    conn.execute(
        """
        INSERT INTO news_documents (
            document_id, source, url, title, published_ts,
            first_seen_ts, ingested_ts, content, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            document_id,
            source,
            url,
            title,
            isoformat(published_ts),
            isoformat(first_seen_ts),
            isoformat(ingested_ts),
            content,
            _json(metadata),
        ),
    )
    conn.executemany(
        """
        INSERT INTO market_news_links (market_id, document_id, relevance)
        VALUES (?, ?, ?)
        """,
        [(market_id, document_id, 1.0) for market_id in market_ids],
    )


def add_resolution(
    conn: sqlite3.Connection,
    *,
    market_id: str,
    resolution_ts: datetime,
    resolved_outcome: float,
    status: str,
    disputed: bool = False,
    clarification_issued: bool = False,
    resolution_note: str = "",
) -> None:
    conn.execute(
        """
        INSERT INTO market_resolutions (
            market_id, resolution_ts, resolved_outcome, status, disputed,
            clarification_issued, resolution_note
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            market_id,
            isoformat(resolution_ts),
            resolved_outcome,
            status,
            int(disputed),
            int(clarification_issued),
            resolution_note,
        ),
    )


def bulk_add_resolutions(conn: sqlite3.Connection, resolutions: list[dict[str, Any]]) -> int:
    rows = [
        (
            resolution["market_id"],
            _coerce_iso8601(resolution["resolution_ts"]),
            resolution["resolved_outcome"],
            resolution["status"],
        )
        for resolution in resolutions
    ]
    total_changes_before = conn.total_changes
    conn.executemany(
        """
        INSERT OR IGNORE INTO market_resolutions (
            market_id, resolution_ts, resolved_outcome, status
        ) VALUES (?, ?, ?, ?)
        """,
        rows,
    )
    inserted_count = _inserted_count(conn, total_changes_before)
    conn.commit()
    return inserted_count


def get_market_ids(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT market_id FROM markets ORDER BY market_id").fetchall()
    return [str(row["market_id"]) for row in rows]


def get_all_timestamps(conn: sqlite3.Connection) -> list[datetime]:
    rows = conn.execute(
        """
        SELECT ts AS replay_ts FROM market_snapshots
        UNION
        SELECT first_seen_ts AS replay_ts FROM news_documents
        UNION
        SELECT effective_ts AS replay_ts FROM market_rule_revisions
        UNION
        SELECT resolution_ts AS replay_ts FROM market_resolutions
        ORDER BY replay_ts
        """
    ).fetchall()
    return [datetime.fromisoformat(str(row["replay_ts"])) for row in rows if row["replay_ts"]]


def get_market_state_as_of(
    conn: sqlite3.Connection,
    market_id: str,
    as_of: datetime,
) -> MarketState | None:
    row = conn.execute(
        """
        SELECT
            m.market_id,
            m.title,
            m.domain,
            m.market_type,
            m.tags_json,
            m.resolution_ts,
            m.fees_enabled,
            m.fee_rate,
            m.fee_exponent,
            m.maker_rebate_rate,
            s.id AS snapshot_id,
            s.ts,
            s.status,
            s.best_bid,
            s.best_ask,
            s.mid,
            s.last_trade,
            s.volume_1m,
            s.volume_24h,
            s.open_interest,
            s.tick_size
        FROM markets m
        JOIN market_snapshots s ON s.market_id = m.market_id
        WHERE m.market_id = ? AND s.ts <= ?
        ORDER BY s.ts DESC
        LIMIT 1
        """,
        (market_id, isoformat(as_of)),
    ).fetchone()
    if row is None:
        return None

    event_outcome_tokens = get_event_outcome_tokens(conn, market_id)
    outcome_tokens = event_outcome_tokens if len(event_outcome_tokens) > 1 else []
    outcome_count = len(outcome_tokens) if outcome_tokens else 2

    rule_row = conn.execute(
        """
        SELECT rules_text, additional_context
        FROM market_rule_revisions
        WHERE market_id = ? AND effective_ts <= ?
        ORDER BY effective_ts DESC
        LIMIT 1
        """,
        (market_id, isoformat(as_of)),
    ).fetchone()
    levels = conn.execute(
        """
        SELECT side, level_no, price, quantity
        FROM orderbook_levels
        WHERE snapshot_id = ?
        ORDER BY side, level_no
        """,
        (row["snapshot_id"],),
    ).fetchall()
    return MarketState(
        market_id=str(row["market_id"]),
        title=str(row["title"]),
        domain=str(row["domain"]),
        market_type=str(row["market_type"]),
        ts=datetime.fromisoformat(str(row["ts"])),
        status=str(row["status"]),
        best_bid=float(row["best_bid"]),
        best_ask=float(row["best_ask"]),
        mid=float(row["mid"]),
        last_trade=float(row["last_trade"]),
        volume_1m=float(row["volume_1m"]),
        volume_24h=float(row["volume_24h"]),
        open_interest=float(row["open_interest"]),
        tick_size=float(row["tick_size"]),
        rules_text=str(rule_row["rules_text"]) if rule_row else "",
        additional_context=str(rule_row["additional_context"]) if rule_row else "",
        resolution_ts=datetime.fromisoformat(str(row["resolution_ts"])) if row["resolution_ts"] else None,
        fees_enabled=bool(row["fees_enabled"]),
        fee_rate=float(row["fee_rate"]),
        fee_exponent=float(row["fee_exponent"]),
        maker_rebate_rate=float(row["maker_rebate_rate"]),
        orderbook=[
            OrderLevel(
                side="bid" if str(level["side"]) == "bid" else "ask",
                level_no=int(level["level_no"]),
                price=float(level["price"]),
                quantity=float(level["quantity"]),
            )
            for level in levels
        ],
        tags=_parse_tags_json(row["tags_json"]),
        outcome_count=outcome_count,
        outcome_tokens=outcome_tokens,
    )


def get_next_market_state(
    conn: sqlite3.Connection,
    market_id: str,
    after_ts: datetime,
) -> MarketState | None:
    row = conn.execute(
        """
        SELECT ts
        FROM market_snapshots
        WHERE market_id = ? AND ts > ?
        ORDER BY ts ASC
        LIMIT 1
        """,
        (market_id, isoformat(after_ts)),
    ).fetchone()
    if row is None:
        return None
    return get_market_state_as_of(conn, market_id, datetime.fromisoformat(str(row["ts"])))


def get_market_news_as_of(
    conn: sqlite3.Connection,
    market_id: str,
    as_of: datetime,
    lookback_minutes: int,
) -> list[NewsItem]:
    lookback_start = ensure_utc(as_of) - timedelta(minutes=lookback_minutes)
    rows = conn.execute(
        """
        SELECT
            n.document_id,
            n.source,
            n.url,
            n.title,
            n.published_ts,
            n.first_seen_ts,
            n.ingested_ts,
            n.content,
            n.metadata_json
        FROM news_documents n
        JOIN market_news_links l ON l.document_id = n.document_id
        WHERE l.market_id = ?
          AND n.published_ts <= ?
          AND n.ingested_ts <= ?
          AND n.first_seen_ts >= ?
        ORDER BY n.first_seen_ts ASC
        """,
        (
            market_id,
            isoformat(as_of),
            isoformat(as_of),
            isoformat(lookback_start),
        ),
    ).fetchall()
    return [
        NewsItem(
            document_id=str(row["document_id"]),
            source=str(row["source"]),
            url=str(row["url"]),
            title=str(row["title"]),
            published_ts=datetime.fromisoformat(str(row["published_ts"])),
            first_seen_ts=datetime.fromisoformat(str(row["first_seen_ts"])),
            ingested_ts=datetime.fromisoformat(str(row["ingested_ts"])),
            content=str(row["content"]),
            metadata=json.loads(str(row["metadata_json"])),
        )
        for row in rows
    ]


def get_related_markets_as_of(
    conn: sqlite3.Connection,
    market_id: str,
    as_of: datetime,
) -> list[dict[str, Any]]:
    domain_row = conn.execute(
        "SELECT domain, event_id FROM markets WHERE market_id = ?",
        (market_id,),
    ).fetchone()
    if domain_row is None:
        return []
    event_id = str(domain_row["event_id"]).strip() if domain_row["event_id"] else None
    event_outcome_tokens = get_event_outcome_tokens(conn, market_id)
    outcome_tokens = event_outcome_tokens if len(event_outcome_tokens) > 1 else []
    outcome_count = len(outcome_tokens) if outcome_tokens else 2

    if outcome_tokens and event_id:
        rows = conn.execute(
            """
            SELECT m.market_id, m.title, s.mid, s.best_bid, s.best_ask, s.ts
            FROM markets m
            JOIN market_snapshots s
              ON s.market_id = m.market_id
             AND s.ts = (
                 SELECT MAX(s2.ts)
                 FROM market_snapshots s2
                 WHERE s2.market_id = m.market_id
                   AND s2.ts <= ?
             )
            WHERE m.market_id != ?
              AND m.event_id = ?
            ORDER BY m.title COLLATE NOCASE, m.market_id
            """,
            (isoformat(as_of), market_id, event_id),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT m.market_id, m.title, s.mid, s.best_bid, s.best_ask, s.ts
            FROM markets m
            JOIN market_snapshots s
              ON s.market_id = m.market_id
             AND s.ts = (
                 SELECT MAX(s2.ts)
                 FROM market_snapshots s2
                 WHERE s2.market_id = m.market_id
                   AND s2.ts <= ?
             )
            WHERE m.market_id != ?
              AND m.domain = ?
            ORDER BY m.market_id
            """,
            (isoformat(as_of), market_id, str(domain_row["domain"])),
        ).fetchall()
    related: list[dict[str, Any]] = []
    for row in rows:
        related.append(
            {
                "market_id": str(row["market_id"]),
                "title": str(row["title"]),
                "mid": float(row["mid"]),
                "best_bid": float(row["best_bid"]),
                "best_ask": float(row["best_ask"]),
                "ts": str(row["ts"]),
                "event_id": event_id,
                "outcome_count": outcome_count,
                "outcome_tokens": list(outcome_tokens),
            }
        )
    return related


def get_resolution(conn: sqlite3.Connection, market_id: str) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT market_id, resolution_ts, resolved_outcome, status, disputed,
               clarification_issued, resolution_note
        FROM market_resolutions
        WHERE market_id = ?
        """,
        (market_id,),
    ).fetchone()


def record_audit(
    conn: sqlite3.Connection,
    *,
    experiment_id: int | None,
    market_id: str | None,
    ts: datetime,
    tool_name: str,
    request_max_ts: datetime,
    result_max_ts: datetime,
    row_count: int,
    request_json: dict[str, Any],
    response_payload: Any,
) -> None:
    conn.execute(
        """
        INSERT INTO replay_audit (
            experiment_id, market_id, ts, tool_name, request_max_ts,
            result_max_ts, row_count, request_json, response_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            experiment_id,
            market_id,
            isoformat(ts),
            tool_name,
            isoformat(request_max_ts),
            isoformat(result_max_ts),
            row_count,
            _json(request_json),
            _hash(response_payload),
        ),
    )


def seed_demo_data(conn: sqlite3.Connection) -> None:
    # Guard: refuse to seed demo data if real market data already exists
    existing = conn.execute("SELECT COUNT(*) FROM markets WHERE market_id NOT LIKE 'pm_%'").fetchone()[0]
    if existing > 0:
        raise RuntimeError("Cannot seed demo data: database contains real market data")

    base = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)

    add_market(
        conn,
        market_id="pm_dropout_june",
        title="Will Candidate X drop out by June 30?",
        domain="politics",
        market_type="politics",
        open_ts=base - timedelta(days=7),
        close_ts=base + timedelta(hours=23),
        resolution_ts=base + timedelta(hours=24),
        status="active",
    )
    add_market(
        conn,
        market_id="pm_fed_hold_march",
        title="Will the Fed keep rates unchanged in March?",
        domain="macro",
        market_type="politics",
        open_ts=base - timedelta(days=14),
        close_ts=base + timedelta(hours=7, minutes=55),
        resolution_ts=base + timedelta(hours=8),
        status="active",
    )

    add_rule_revision(
        conn,
        market_id="pm_dropout_june",
        effective_ts=base - timedelta(days=7),
        rules_text="Resolves YES if Candidate X officially suspends or ends the campaign by June 30.",
    )
    add_rule_revision(
        conn,
        market_id="pm_fed_hold_march",
        effective_ts=base - timedelta(days=14),
        rules_text="Resolves YES if the March FOMC target range is unchanged from the previous meeting.",
    )

    dropout_series = [
        (0, 0.22, 0.24, 0.23, 120.0, 12000.0, 8000.0),
        (1, 0.23, 0.25, 0.24, 180.0, 12180.0, 8200.0),
        (5, 0.44, 0.48, 0.46, 950.0, 13130.0, 9200.0),
        (30, 0.68, 0.72, 0.70, 1300.0, 14430.0, 10300.0),
        (240, 0.97, 0.99, 0.98, 700.0, 15130.0, 10700.0),
    ]
    for minute, bid, ask, last_trade, vol_1m, vol_24h, oi in dropout_series:
        ts = base + timedelta(minutes=minute)
        add_snapshot(
            conn,
            market_id="pm_dropout_june",
            ts=ts,
            status="active" if minute < 240 else "pending_resolution",
            best_bid=bid,
            best_ask=ask,
            last_trade=last_trade,
            volume_1m=vol_1m,
            volume_24h=vol_24h,
            open_interest=oi,
            tick_size=0.01,
            orderbook=[
                ("bid", 1, bid, 400.0),
                ("bid", 2, round(bid - 0.01, 2), 550.0),
                ("bid", 3, round(bid - 0.02, 2), 700.0),
                ("ask", 1, ask, 350.0),
                ("ask", 2, round(ask + 0.01, 2), 500.0),
                ("ask", 3, round(ask + 0.02, 2), 650.0),
            ],
        )

    fed_series = [
        (0, 0.955, 0.965, 0.96, 80.0, 24000.0, 16000.0),
        (5, 0.958, 0.968, 0.963, 75.0, 24075.0, 16020.0),
        (30, 0.968, 0.978, 0.973, 90.0, 24165.0, 16100.0),
        (240, 0.992, 0.995, 0.994, 150.0, 24315.0, 16250.0),
    ]
    for minute, bid, ask, last_trade, vol_1m, vol_24h, oi in fed_series:
        ts = base + timedelta(minutes=minute)
        add_snapshot(
            conn,
            market_id="pm_fed_hold_march",
            ts=ts,
            status="active" if minute < 240 else "pending_resolution",
            best_bid=bid,
            best_ask=ask,
            last_trade=last_trade,
            volume_1m=vol_1m,
            volume_24h=vol_24h,
            open_interest=oi,
            tick_size=0.001,
            orderbook=[
                ("bid", 1, bid, 700.0),
                ("bid", 2, round(bid - 0.002, 3), 900.0),
                ("bid", 3, round(bid - 0.004, 3), 1200.0),
                ("ask", 1, ask, 650.0),
                ("ask", 2, round(ask + 0.002, 3), 850.0),
                ("ask", 3, round(ask + 0.004, 3), 1000.0),
            ],
        )

    add_news(
        conn,
        document_id="doc_official_suspend",
        source="official",
        url="https://example.org/candidate-statement",
        title="Campaign statement says Candidate X suspends campaign",
        published_ts=base + timedelta(minutes=1),
        first_seen_ts=base + timedelta(minutes=1, seconds=20),
        ingested_ts=base + timedelta(minutes=1, seconds=40),
        content="The campaign released an official statement that Candidate X is suspending the campaign immediately.",
        metadata={"impact": 0.38, "sentiment": "yes_positive"},
        market_ids=["pm_dropout_june"],
    )
    add_news(
        conn,
        document_id="doc_fed_preview",
        source="official",
        url="https://example.org/fed-preview",
        title="Fed preview implies no change expected",
        published_ts=base + timedelta(minutes=2),
        first_seen_ts=base + timedelta(minutes=2, seconds=10),
        ingested_ts=base + timedelta(minutes=2, seconds=35),
        content="Analysts and the official schedule imply no rate change in March.",
        metadata={"impact": 0.03, "sentiment": "yes_positive"},
        market_ids=["pm_fed_hold_march"],
    )

    add_resolution(
        conn,
        market_id="pm_dropout_june",
        resolution_ts=base + timedelta(hours=24),
        resolved_outcome=1.0,
        status="resolved",
        disputed=False,
        clarification_issued=False,
        resolution_note="Official campaign filing confirmed suspension.",
    )
    add_resolution(
        conn,
        market_id="pm_fed_hold_march",
        resolution_ts=base + timedelta(hours=8),
        resolved_outcome=1.0,
        status="resolved",
        disputed=False,
        clarification_issued=False,
        resolution_note="FOMC statement kept target range unchanged.",
    )
    conn.commit()
