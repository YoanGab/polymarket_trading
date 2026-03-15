PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    model_id TEXT NOT NULL,
    model_release TEXT NOT NULL,
    system_prompt_hash TEXT NOT NULL,
    config_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS markets (
    market_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    domain TEXT NOT NULL,
    market_type TEXT NOT NULL,
    event_id TEXT,
    tags_json TEXT NOT NULL DEFAULT '[]',
    open_ts TEXT NOT NULL,
    close_ts TEXT,
    resolution_ts TEXT,
    status TEXT NOT NULL,
    fees_enabled INTEGER NOT NULL DEFAULT 0,
    fee_rate REAL NOT NULL DEFAULT 0.0,
    fee_exponent REAL NOT NULL DEFAULT 0.0,
    maker_rebate_rate REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS market_rule_revisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT NOT NULL,
    effective_ts TEXT NOT NULL,
    rules_text TEXT NOT NULL,
    additional_context TEXT NOT NULL DEFAULT '',
    bulletin_ref TEXT,
    FOREIGN KEY (market_id) REFERENCES markets (market_id)
);

CREATE INDEX IF NOT EXISTS idx_rule_revisions_market_ts
    ON market_rule_revisions (market_id, effective_ts);

CREATE INDEX IF NOT EXISTS idx_markets_event_id
    ON markets (event_id);

CREATE TABLE IF NOT EXISTS market_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    status TEXT NOT NULL,
    best_bid REAL NOT NULL CHECK (best_bid >= 0 AND best_bid <= 1),
    best_ask REAL NOT NULL CHECK (best_ask >= 0 AND best_ask <= 1),
    mid REAL NOT NULL,
    last_trade REAL NOT NULL,
    volume_1m REAL NOT NULL CHECK (volume_1m >= 0),
    volume_24h REAL NOT NULL CHECK (volume_24h >= 0),
    open_interest REAL NOT NULL,
    tick_size REAL NOT NULL,
    is_synthetic INTEGER NOT NULL DEFAULT 0,
    features_json TEXT NOT NULL DEFAULT '{}',
    CHECK (best_bid <= best_ask),
    FOREIGN KEY (market_id) REFERENCES markets (market_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_market_snapshots_market_ts
    ON market_snapshots (market_id, ts);

CREATE TABLE IF NOT EXISTS orderbook_levels (
    snapshot_id INTEGER NOT NULL,
    side TEXT NOT NULL,
    level_no INTEGER NOT NULL,
    price REAL NOT NULL,
    quantity REAL NOT NULL,
    PRIMARY KEY (snapshot_id, side, level_no),
    FOREIGN KEY (snapshot_id) REFERENCES market_snapshots (id)
);

CREATE TABLE IF NOT EXISTS news_documents (
    document_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    url TEXT NOT NULL,
    title TEXT NOT NULL,
    published_ts TEXT NOT NULL,
    first_seen_ts TEXT NOT NULL,
    ingested_ts TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_news_documents_ts
    ON news_documents (published_ts, first_seen_ts, ingested_ts);

CREATE TABLE IF NOT EXISTS market_news_links (
    market_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    relevance REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (market_id, document_id),
    FOREIGN KEY (market_id) REFERENCES markets (market_id),
    FOREIGN KEY (document_id) REFERENCES news_documents (document_id)
);

CREATE TABLE IF NOT EXISTS model_outputs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    market_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    domain TEXT NOT NULL,
    model_id TEXT NOT NULL,
    model_release TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    context_hash TEXT NOT NULL,
    probability_yes REAL NOT NULL,
    confidence REAL NOT NULL,
    expected_edge_bps REAL NOT NULL,
    thesis TEXT NOT NULL,
    reasoning TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    raw_response_json TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments (id),
    FOREIGN KEY (market_id) REFERENCES markets (market_id)
);

CREATE INDEX IF NOT EXISTS idx_model_outputs_market_ts
    ON model_outputs (market_id, ts);

CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    experiment_id INTEGER NOT NULL,
    strategy_name TEXT NOT NULL,
    market_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    side TEXT NOT NULL,
    liquidity_intent TEXT NOT NULL,
    limit_price REAL NOT NULL,
    requested_quantity REAL NOT NULL,
    filled_quantity REAL NOT NULL DEFAULT 0.0,
    edge_bps REAL NOT NULL,
    kelly_fraction REAL NOT NULL,
    holding_period_minutes INTEGER,
    thesis TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments (id),
    FOREIGN KEY (market_id) REFERENCES markets (market_id)
);

CREATE TABLE IF NOT EXISTS fills (
    fill_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    experiment_id INTEGER NOT NULL,
    market_id TEXT NOT NULL,
    strategy_name TEXT NOT NULL,
    fill_ts TEXT NOT NULL,
    side TEXT NOT NULL,
    liquidity_role TEXT NOT NULL,
    price REAL NOT NULL,
    quantity REAL NOT NULL,
    notional_usdc REAL NOT NULL,
    fee_usdc REAL NOT NULL,
    rebate_usdc REAL NOT NULL,
    impact_bps REAL NOT NULL,
    fill_delay_seconds REAL NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders (order_id),
    FOREIGN KEY (experiment_id) REFERENCES experiments (id),
    FOREIGN KEY (market_id) REFERENCES markets (market_id)
);

CREATE INDEX IF NOT EXISTS idx_fills_market_ts
    ON fills (market_id, fill_ts);

CREATE TABLE IF NOT EXISTS positions (
    position_id TEXT PRIMARY KEY,
    experiment_id INTEGER NOT NULL,
    strategy_name TEXT NOT NULL,
    market_id TEXT NOT NULL,
    opened_ts TEXT NOT NULL,
    closed_ts TEXT,
    quantity REAL NOT NULL,
    avg_entry_price REAL NOT NULL,
    avg_exit_price REAL,
    status TEXT NOT NULL,
    realized_pnl_pre_resolution REAL NOT NULL DEFAULT 0.0,
    hold_to_resolution_pnl REAL NOT NULL DEFAULT 0.0,
    resolved_outcome REAL,
    resolution_ts TEXT,
    thesis TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments (id),
    FOREIGN KEY (market_id) REFERENCES markets (market_id)
);

CREATE TABLE IF NOT EXISTS pnl_marks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    strategy_name TEXT NOT NULL,
    market_id TEXT,
    ts TEXT NOT NULL,
    cash REAL NOT NULL,
    position_qty REAL NOT NULL,
    mark_price REAL NOT NULL,
    inventory_value REAL NOT NULL,
    equity REAL NOT NULL,
    realized_pnl REAL NOT NULL,
    unrealized_pnl REAL NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
);

CREATE INDEX IF NOT EXISTS idx_pnl_marks_strategy_ts
    ON pnl_marks (strategy_name, ts);

CREATE TABLE IF NOT EXISTS market_resolutions (
    market_id TEXT PRIMARY KEY,
    resolution_ts TEXT NOT NULL,
    resolved_outcome REAL NOT NULL,
    status TEXT NOT NULL,
    disputed INTEGER NOT NULL DEFAULT 0,
    clarification_issued INTEGER NOT NULL DEFAULT 0,
    resolution_note TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (market_id) REFERENCES markets (market_id)
);

CREATE TABLE IF NOT EXISTS metric_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    strategy_name TEXT,
    scope_type TEXT NOT NULL,
    scope_name TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    horizon_min INTEGER,
    metric_value REAL NOT NULL,
    sample_size INTEGER NOT NULL,
    extra_json TEXT NOT NULL DEFAULT '{}',
    computed_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
);

CREATE TABLE IF NOT EXISTS replay_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER,
    market_id TEXT,
    ts TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    request_max_ts TEXT NOT NULL,
    result_max_ts TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    request_json TEXT NOT NULL,
    response_hash TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
);
