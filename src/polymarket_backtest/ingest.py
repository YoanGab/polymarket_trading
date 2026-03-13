from __future__ import annotations

import sqlite3
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from .db import (
    bulk_add_markets,
    bulk_add_resolutions,
    bulk_add_rule_revisions,
    bulk_add_snapshots,
    connect,
    init_db,
)
from .downloaders.gamma import (
    fetch_market_rules,
    fetch_resolved_markets,
    parse_resolution,
)
from .downloaders.warproxxx import (
    ARCHIVE_FILENAME,
    EXTRACTION_SENTINEL,
    extract_archive,
    parse_markets,
    parse_trades,
)
from .types import ensure_utc

APPROX_OPEN_LOOKBACK = timedelta(days=30)
RULE_FETCH_DELAY_SECONDS = 0.05
DEFAULT_MARKET_DOMAIN = "general"
DEFAULT_MARKET_TYPE = "binary"
DEFAULT_SNAPSHOT_STATUS = "active"
DEFAULT_TICK_SIZE = 0.01
EPOCH_OPEN_TS = datetime(1970, 1, 1, tzinfo=UTC)

MARKET_ID_COLUMNS = ("market_id", "condition_id")
MARKET_TITLE_COLUMNS = ("question", "title")
MARKET_OPEN_COLUMNS = ("open_ts", "openTime", "open_time", "earliest_trade_ts", "first_trade_ts", "timestamp")
MARKET_CLOSE_COLUMNS = ("close_ts", "closedTime", "closed_time", "resolution_ts")
TRADE_TIMESTAMP_COLUMNS = ("timestamp", "ts", "time", "datetime")


def ingest_gamma_markets(conn: sqlite3.Connection, raw_markets: list[dict[str, Any]]) -> dict[str, int]:
    parsed_markets = _parse_gamma_markets(raw_markets)
    market_rows: list[dict[str, Any]] = []
    resolution_rows: list[dict[str, Any]] = []

    for parsed in parsed_markets:
        raw_ts = parsed["resolution_ts"]
        resolution_ts = ensure_utc(raw_ts if isinstance(raw_ts, datetime) else datetime.fromisoformat(str(raw_ts)))
        market_rows.append(
            {
                "market_id": parsed["market_id"],
                "title": parsed["title"],
                "domain": parsed["domain"],
                "market_type": "binary",
                "open_ts": resolution_ts - APPROX_OPEN_LOOKBACK,
                "close_ts": resolution_ts,
                "resolution_ts": resolution_ts,
                "status": "resolved",
            }
        )
        resolution_rows.append(
            {
                "market_id": parsed["market_id"],
                "resolution_ts": resolution_ts,
                "resolved_outcome": parsed["resolved_outcome"],
                "status": "resolved",
            }
        )

    return {
        "markets": bulk_add_markets(conn, market_rows) if market_rows else 0,
        "resolutions": bulk_add_resolutions(conn, resolution_rows) if resolution_rows else 0,
    }


def ingest_gamma_rules(conn: sqlite3.Connection, market_ids: list[str]) -> int:
    unique_market_ids = list(dict.fromkeys(market_ids))
    if not unique_market_ids:
        return 0

    market_open_ts = _fetch_market_open_timestamps(conn, unique_market_ids)
    revisions: list[dict[str, Any]] = []

    for index, market_id in enumerate(unique_market_ids):
        open_ts = market_open_ts.get(market_id)
        if open_ts is None:
            continue

        revisions.append(
            {
                "market_id": market_id,
                "effective_ts": open_ts,
                "rules_text": fetch_market_rules(market_id),
                "additional_context": "",
            }
        )

        if index < len(unique_market_ids) - 1:
            time.sleep(RULE_FETCH_DELAY_SECONDS)

    return bulk_add_rule_revisions(conn, revisions) if revisions else 0


def ingest_gamma(conn: sqlite3.Connection, *, max_markets: int = 500) -> dict[str, int]:
    raw_markets = fetch_resolved_markets(max_markets=max_markets)
    parsed_market_ids = [parsed["market_id"] for parsed in _parse_gamma_markets(raw_markets)]
    counts = ingest_gamma_markets(conn, raw_markets)
    counts["rules"] = ingest_gamma_rules(conn, parsed_market_ids)
    return counts


def _parse_gamma_markets(raw_markets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parsed_markets: list[dict[str, Any]] = []
    for market in raw_markets:
        parsed = parse_resolution(market)
        if parsed is not None:
            parsed_markets.append(parsed)
    return parsed_markets


def _fetch_market_open_timestamps(
    conn: sqlite3.Connection,
    market_ids: list[str],
) -> dict[str, datetime]:
    placeholders = ",".join("?" for _ in market_ids)
    rows = conn.execute(
        f"""
        SELECT market_id, open_ts
        FROM markets
        WHERE market_id IN ({placeholders})
        """,
        market_ids,
    ).fetchall()
    return {
        str(row["market_id"]): ensure_utc(datetime.fromisoformat(str(row["open_ts"]))) for row in rows if row["open_ts"]
    }


def ingest_warproxxx_markets(conn: sqlite3.Connection, markets_df: pl.DataFrame) -> int:
    if markets_df.is_empty():
        return 0

    normalized = _with_canonical_column(markets_df, "market_id", MARKET_ID_COLUMNS)
    title_source = _first_present_column(normalized, MARKET_TITLE_COLUMNS)
    if title_source is None:
        raise ValueError(f"markets_df must include one of {MARKET_TITLE_COLUMNS}")

    open_source = _first_present_column(normalized, MARKET_OPEN_COLUMNS)
    close_source = _first_present_column(normalized, MARKET_CLOSE_COLUMNS)
    if close_source is None:
        normalized = normalized.with_columns(pl.lit(None, dtype=pl.String).alias("closedTime"))
        close_source = "closedTime"

    expressions: list[pl.Expr] = [
        _clean_string_expr("market_id").alias("market_id"),
        _clean_string_expr(title_source).fill_null(pl.col("market_id")).alias("title"),
        _datetime_expr(normalized, close_source).alias("close_ts"),
    ]
    if open_source is None:
        expressions.append(pl.lit(None, dtype=pl.Datetime(time_zone="UTC")).alias("open_ts"))
    else:
        expressions.append(_datetime_expr(normalized, open_source).alias("open_ts"))

    prepared = (
        normalized.with_columns(expressions)
        .with_columns(
            pl.coalesce("open_ts", "close_ts", pl.lit(EPOCH_OPEN_TS)).alias("open_ts"),
            pl.when(pl.col("close_ts").is_not_null())
            .then(pl.lit("resolved"))
            .otherwise(pl.lit("active"))
            .alias("status"),
            pl.lit(DEFAULT_MARKET_DOMAIN).alias("domain"),
            pl.lit(DEFAULT_MARKET_TYPE).alias("market_type"),
            pl.col("close_ts").alias("resolution_ts"),
        )
        .filter(pl.col("market_id").is_not_null())
        .unique(subset=["market_id"], keep="last", maintain_order=True)
        .select(
            "market_id",
            "title",
            "domain",
            "market_type",
            "open_ts",
            "close_ts",
            "resolution_ts",
            "status",
        )
    )
    if prepared.is_empty():
        return 0

    return bulk_add_markets(conn, list(prepared.iter_rows(named=True)))


def ingest_warproxxx_trades_as_snapshots(
    conn: sqlite3.Connection,
    trades_df: pl.DataFrame,
    interval_minutes: int = 5,
) -> int:
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be greater than zero")
    if trades_df.is_empty():
        return 0

    normalized = _with_canonical_column(trades_df, "market_id", MARKET_ID_COLUMNS)
    normalized = _with_canonical_column(normalized, "timestamp", TRADE_TIMESTAMP_COLUMNS)
    if "price" not in normalized.columns:
        raise ValueError("trades_df must include a price column")

    close_source = _first_present_column(normalized, MARKET_CLOSE_COLUMNS)
    if close_source is None:
        normalized = normalized.with_columns(pl.lit(None, dtype=pl.String).alias("closedTime"))
        close_source = "closedTime"

    if "direction" not in normalized.columns:
        normalized = normalized.with_columns(pl.lit(None, dtype=pl.String).alias("direction"))

    if "usd_amount" not in normalized.columns:
        if "token_amount" in normalized.columns:
            normalized = normalized.with_columns(
                (
                    pl.col("token_amount").cast(pl.Float64, strict=False)
                    * pl.col("price").cast(pl.Float64, strict=False)
                ).alias("usd_amount")
            )
        else:
            normalized = normalized.with_columns(pl.lit(0.0).alias("usd_amount"))

    trades = (
        normalized.with_columns(
            _clean_string_expr("market_id").alias("market_id"),
            _datetime_expr(normalized, "timestamp").alias("trade_ts"),
            pl.col("price").cast(pl.Float64, strict=False).alias("price"),
            pl.col("usd_amount").cast(pl.Float64, strict=False).fill_null(0.0).alias("usd_amount"),
            _clean_string_expr("direction").str.to_lowercase().alias("direction"),
            _datetime_expr(normalized, close_source).alias("close_ts"),
        )
        .filter(pl.col("market_id").is_not_null() & pl.col("trade_ts").is_not_null() & pl.col("price").is_not_null())
        .sort(["market_id", "trade_ts"])
        .with_columns(pl.col("trade_ts").dt.truncate(f"{interval_minutes}m").alias("ts"))
    )
    if trades.is_empty():
        return 0

    rolling_buckets = max(1, -(-24 * 60 // interval_minutes))
    bucketed = (
        trades.group_by(["market_id", "ts"])
        .agg(
            pl.col("price").min().alias("min_price"),
            pl.col("price").max().alias("max_price"),
            pl.col("price").last().alias("last_trade"),
            pl.col("price").filter(pl.col("direction") == "sell").min().alias("sell_min"),
            pl.col("price").filter(pl.col("direction") == "buy").max().alias("buy_max"),
            pl.col("usd_amount").sum().alias("bucket_volume"),
            pl.col("close_ts").max().alias("close_ts"),
        )
        .sort(["market_id", "ts"])
        .with_columns(
            (pl.col("bucket_volume") / float(interval_minutes)).alias("volume_1m"),
            pl.col("bucket_volume")
            .rolling_sum(window_size=rolling_buckets, min_samples=1)
            .over("market_id")
            .alias("volume_24h"),
            (pl.col("sell_min").is_not_null() & pl.col("buy_max").is_not_null()).alias("has_split"),
        )
        .with_columns(
            pl.when(pl.col("has_split"))
            .then(pl.col("buy_max"))
            .otherwise(pl.col("min_price") - DEFAULT_TICK_SIZE)
            .clip(0.0, 1.0)
            .alias("best_bid"),
            pl.when(pl.col("has_split"))
            .then(pl.col("sell_min"))
            .otherwise(pl.col("max_price") + DEFAULT_TICK_SIZE)
            .clip(0.0, 1.0)
            .alias("best_ask"),
            pl.when(pl.col("close_ts").is_not_null() & (pl.col("ts") >= pl.col("close_ts")))
            .then(pl.lit("resolved"))
            .otherwise(pl.lit(DEFAULT_SNAPSHOT_STATUS))
            .alias("status"),
            pl.lit(0.0).alias("open_interest"),
            pl.lit(DEFAULT_TICK_SIZE).alias("tick_size"),
        )
        .select(
            "market_id",
            "ts",
            "status",
            "best_bid",
            "best_ask",
            "last_trade",
            "volume_1m",
            "volume_24h",
            "open_interest",
            "tick_size",
        )
    )
    if bucketed.is_empty():
        return 0

    return bulk_add_snapshots(conn, list(bucketed.iter_rows(named=True)))


def ingest_warproxxx(db_path: Path, data_dir: Path) -> dict[str, int]:
    extracted_dir = _resolve_warproxxx_data_dir(data_dir)
    trades_df = parse_trades(extracted_dir)
    markets_df = parse_markets(extracted_dir)

    trade_markets = _derive_trade_market_open_times(trades_df)
    market_inputs = _merge_warproxxx_market_metadata(markets_df, trade_markets)
    snapshot_inputs = _attach_market_close_times(trades_df, market_inputs)

    with connect(db_path) as conn:
        init_db(conn)
        market_count = ingest_warproxxx_markets(conn, market_inputs)
        snapshot_count = ingest_warproxxx_trades_as_snapshots(conn, snapshot_inputs)

    return {"markets": market_count, "snapshots": snapshot_count}


def _resolve_warproxxx_data_dir(data_dir: Path) -> Path:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory: {data_dir}")

    if (data_dir / EXTRACTION_SENTINEL).exists():
        return data_dir

    archive_path = data_dir / ARCHIVE_FILENAME
    if archive_path.exists():
        return extract_archive(archive_path, data_dir)

    if any(data_dir.rglob("*.csv")):
        return data_dir

    return data_dir


def _derive_trade_market_open_times(trades_df: pl.DataFrame) -> pl.DataFrame:
    if trades_df.is_empty():
        return pl.DataFrame(
            schema={
                "market_id": pl.String,
                "open_ts": pl.Datetime(time_zone="UTC"),
            }
        )

    normalized = _with_canonical_column(trades_df, "market_id", MARKET_ID_COLUMNS)
    normalized = _with_canonical_column(normalized, "timestamp", TRADE_TIMESTAMP_COLUMNS)
    return (
        normalized.with_columns(
            _clean_string_expr("market_id").alias("market_id"),
            _datetime_expr(normalized, "timestamp").alias("open_ts"),
        )
        .filter(pl.col("market_id").is_not_null() & pl.col("open_ts").is_not_null())
        .group_by("market_id")
        .agg(pl.col("open_ts").min().alias("open_ts"))
        .sort("market_id")
    )


def _merge_warproxxx_market_metadata(markets_df: pl.DataFrame, trade_markets: pl.DataFrame) -> pl.DataFrame:
    if markets_df.is_empty():
        if trade_markets.is_empty():
            return pl.DataFrame()
        return trade_markets.with_columns(
            pl.col("market_id").alias("question"),
            pl.lit(None, dtype=pl.Float64).alias("volume"),
            pl.lit(None, dtype=pl.String).alias("closedTime"),
        )

    normalized = _with_canonical_column(markets_df, "market_id", MARKET_ID_COLUMNS)
    merged = normalized.join(trade_markets, on="market_id", how="left") if not trade_markets.is_empty() else normalized
    if trade_markets.is_empty():
        return merged

    missing_market_rows = trade_markets.join(merged.select("market_id"), on="market_id", how="anti")
    if missing_market_rows.is_empty():
        return merged

    placeholders = missing_market_rows.with_columns(
        pl.col("market_id").alias("question"),
        pl.lit(None, dtype=pl.Float64).alias("volume"),
        pl.lit(None, dtype=pl.String).alias("closedTime"),
    )
    return pl.concat([merged, placeholders], how="diagonal_relaxed")


def _attach_market_close_times(trades_df: pl.DataFrame, markets_df: pl.DataFrame) -> pl.DataFrame:
    if trades_df.is_empty() or markets_df.is_empty():
        return trades_df

    market_id_source = _first_present_column(markets_df, MARKET_ID_COLUMNS)
    if market_id_source is None:
        return trades_df

    close_source = _first_present_column(markets_df, MARKET_CLOSE_COLUMNS)
    if close_source is None:
        return trades_df

    close_times = (
        _with_canonical_column(markets_df, "market_id", MARKET_ID_COLUMNS)
        .with_columns(
            _clean_string_expr("market_id").alias("market_id"),
            pl.col(close_source).alias("closedTime"),
        )
        .select("market_id", "closedTime")
        .unique(subset=["market_id"], keep="last", maintain_order=True)
    )
    return _with_canonical_column(trades_df, "market_id", MARKET_ID_COLUMNS).join(
        close_times,
        on="market_id",
        how="left",
    )


def _with_canonical_column(df: pl.DataFrame, target: str, candidates: tuple[str, ...]) -> pl.DataFrame:
    if target in df.columns:
        return df

    source = _first_present_column(df, candidates)
    if source is None:
        raise ValueError(f"DataFrame must include one of {candidates}")

    return df.with_columns(pl.col(source).alias(target))


def _first_present_column(df: pl.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _clean_string_expr(column_name: str) -> pl.Expr:
    return pl.col(column_name).cast(pl.String, strict=False).str.strip_chars().replace("", None)


def _datetime_expr(df: pl.DataFrame, column_name: str) -> pl.Expr:
    dtype = df.schema[column_name]
    if dtype == pl.Datetime:
        expr = pl.col(column_name)
        if getattr(dtype, "time_zone", None):
            return expr.dt.convert_time_zone("UTC")
        return expr.dt.replace_time_zone("UTC")

    if dtype == pl.Date:
        return pl.col(column_name).cast(pl.Datetime).dt.replace_time_zone("UTC")

    if dtype.is_numeric():
        return _epoch_expr(pl.col(column_name).cast(pl.Int64, strict=False))

    text = _clean_string_expr(column_name)
    digits = text.str.extract(r"^(-?\d+)$", 1).cast(pl.Int64, strict=False)
    return pl.coalesce(
        text.str.to_datetime(strict=False, time_zone="UTC"),
        text.str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False).dt.replace_time_zone("UTC"),
        text.str.strptime(pl.Datetime, "%Y-%m-%d", strict=False).dt.replace_time_zone("UTC"),
        _epoch_expr(digits),
    )


def _epoch_expr(raw: pl.Expr) -> pl.Expr:
    return (
        pl.when(raw.is_null())
        .then(None)
        .when(raw.abs() >= 1_000_000_000_000_000)
        .then(pl.from_epoch(raw, time_unit="us"))
        .when(raw.abs() >= 1_000_000_000_000)
        .then(pl.from_epoch(raw, time_unit="ms"))
        .otherwise(pl.from_epoch(raw, time_unit="s"))
        .dt.replace_time_zone("UTC")
    )
