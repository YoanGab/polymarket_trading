from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from polymarket_backtest import db
from polymarket_backtest.downloaders.gamma import parse_resolution
from polymarket_backtest.ingest import ingest_gamma_markets


def _raw_gamma_market(*, market_id: str = "market-1", event_id: int = 12345) -> dict[str, object]:
    return {
        "closed": True,
        "conditionId": market_id,
        "question": "Will the ingest pipeline keep event metadata?",
        "closedTime": datetime(2026, 1, 1, 12, 0, tzinfo=UTC).isoformat(),
        "outcomes": '["Yes", "No"]',
        "outcomePrices": "[1, 0]",
        "events": [{"id": event_id}],
        "tags": [{"slug": "politics"}],
        "volume": "1250.5",
    }


def test_parse_resolution_stringifies_numeric_event_id() -> None:
    parsed = parse_resolution(_raw_gamma_market())

    assert parsed is not None
    assert parsed["event_id"] == "12345"
    assert parsed["domain"] == "politics"
    assert parsed["tags"] == ["politics"]


def test_ingest_gamma_markets_backfills_existing_market_metadata(tmp_path) -> None:
    conn = db.connect(tmp_path / "ingest.sqlite")
    try:
        db.init_db(conn)
        resolution_ts = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        db.add_market(
            conn,
            market_id="market-1",
            title="Will the ingest pipeline keep event metadata?",
            domain="general",
            market_type="binary",
            open_ts=resolution_ts - timedelta(days=30),
            close_ts=resolution_ts,
            resolution_ts=resolution_ts,
            status="resolved",
            event_id=None,
            tags=None,
        )
        conn.commit()

        counts = ingest_gamma_markets(conn, [_raw_gamma_market()])
        row = conn.execute(
            "SELECT event_id, tags_json FROM markets WHERE market_id = ?",
            ("market-1",),
        ).fetchone()

        assert counts["markets"] == 0
        assert counts["resolutions"] == 1
        assert row is not None
        assert row["event_id"] == "12345"
        assert json.loads(str(row["tags_json"])) == ["politics"]
    finally:
        conn.close()
