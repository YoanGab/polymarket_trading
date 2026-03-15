import heapq
import json
import logging
import sqlite3
import uuid
from bisect import bisect_left
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from . import db
from .grok_replay import ReplayGrokClient
from .market_categories import category_fee_settings, normalize_market_tags
from .market_simulator import MarketSimulator, new_order_id
from .strategies import StrategyEngine
from .types import (
    FillResult,
    MarketState,
    OrderLevel,
    PositionState,
    ReplayConfig,
    StrategyConfig,
    dc_replace,
    ensure_utc,
    isoformat,
)

logger = logging.getLogger(__name__)

COMMIT_EVERY = 50
PROGRESS_EVERY = 50


@dataclass
class StrategyPortfolio:
    cash: float
    positions: dict[str, PositionState] = field(default_factory=dict)
    last_known_mids: dict[str, float] = field(default_factory=dict)
    realized_pnl: float = 0.0
    missed_trades: int = 0
    missed_edge_bps: float = 0.0


@dataclass
class ReplayEngine:
    conn: sqlite3.Connection
    config: ReplayConfig
    grok: ReplayGrokClient
    strategies: list[StrategyConfig]
    market_categories: dict[str, list[str]] = field(default_factory=dict)
    simulator: MarketSimulator = field(default_factory=MarketSimulator)

    def __post_init__(self) -> None:
        self.portfolios = {
            strategy.name: StrategyPortfolio(cash=self.config.starting_cash) for strategy in self.strategies
        }
        self.strategy_engine = StrategyEngine()
        self._strategy_by_name = {strategy.name: strategy for strategy in self.strategies}
        self._resolution_cache: dict[str, sqlite3.Row | None] = {}
        self._market_cache: dict[tuple[str, str], MarketState] = {}
        self._history_by_market: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        self._history_index_by_key: dict[tuple[str, str], int] = {}
        self._next_ts_cache: dict[tuple[str, str], tuple[str, str] | None] = {}
        self.experiment_id = self.grok.experiment_id
        if self.experiment_id is None:
            raise ValueError("ReplayEngine requires grok.experiment_id")

    def run(self) -> int:
        return self._run_replay()

    def run_markets(self, market_ids: list[str]) -> int:
        return self._run_replay(market_ids)

    def run_single_market(self, market_id: str) -> int:
        return self._run_replay([market_id])

    def _run_replay(self, market_ids: Sequence[str] | None = None) -> int:
        selected_market_ids = self._normalize_market_ids(market_ids)
        timelines = self._build_market_timelines(selected_market_ids)
        if not timelines:
            logger.warning("Replay skipped because no market snapshots were found")
            self.conn.commit()
            self._persist_positions()
            return self.experiment_id

        self._preload_market_data(timelines)
        self._warm_resolution_cache(list(timelines))

        total_events = sum(len(timestamps) for timestamps in timelines.values())
        processed_events = 0
        last_replay_ts: datetime | None = None

        for market_id, timestamp in self._iter_market_events(timelines):
            last_replay_ts = timestamp
            portfolio_snapshot = self._snapshot_portfolios()
            snapshot_started = False
            try:
                self.conn.execute("SAVEPOINT snapshot_sp")
                snapshot_started = True
                self._process_market_snapshot(market_id, timestamp)
            except Exception as exc:
                self.portfolios = portfolio_snapshot
                if snapshot_started:
                    self.conn.execute("ROLLBACK TO snapshot_sp")
                    self.conn.execute("RELEASE snapshot_sp")
                logger.warning(
                    "Snapshot error for %s at %s: %s",
                    market_id,
                    isoformat(timestamp),
                    exc,
                )
            else:
                if snapshot_started:
                    self.conn.execute("RELEASE snapshot_sp")
            finally:
                self._settle_resolved_positions(timestamp)
                processed_events += 1
                if processed_events % COMMIT_EVERY == 0:
                    self.conn.commit()
                if processed_events % PROGRESS_EVERY == 0 or processed_events == total_events:
                    print(f"[ReplayEngine] processed {processed_events}/{total_events} market snapshots")

        final_replay_ts = self._resolve_final_replay_ts(last_replay_ts, list(timelines))
        if final_replay_ts is not None:
            self._settle_resolved_positions(final_replay_ts)

        self.conn.commit()
        self._persist_positions()
        return self.experiment_id

    def _normalize_market_ids(self, market_ids: Sequence[str] | None) -> list[str] | None:
        if market_ids is None:
            return None
        return list(dict.fromkeys(market_ids))

    def _build_market_timelines(self, market_ids: Sequence[str] | None) -> dict[str, list[datetime]]:
        if market_ids is not None and not market_ids:
            return {}

        if market_ids is not None and len(market_ids) > 900:
            # Use temp table to avoid SQLite's 999-variable limit
            self.conn.execute("CREATE TEMP TABLE IF NOT EXISTS _selected_markets (market_id TEXT PRIMARY KEY)")
            self.conn.execute("DELETE FROM _selected_markets")
            self.conn.executemany(
                "INSERT OR IGNORE INTO _selected_markets VALUES (?)",
                [(m,) for m in market_ids],
            )
            self.conn.commit()
            query = (
                "SELECT ms.market_id, ms.ts FROM market_snapshots ms "
                "JOIN _selected_markets sm ON sm.market_id = ms.market_id "
                "ORDER BY ms.market_id, ms.ts"
            )
            rows = self.conn.execute(query).fetchall()
        else:
            params: tuple[str, ...] = ()
            query = "SELECT market_id, ts FROM market_snapshots"
            if market_ids is not None:
                placeholders = ", ".join("?" for _ in market_ids)
                query += f" WHERE market_id IN ({placeholders})"
                params = tuple(market_ids)
            query += " ORDER BY market_id, ts"
            rows = self.conn.execute(query, params).fetchall()
        timelines: dict[str, list[datetime]] = {}
        for row in rows:
            market_id = str(row["market_id"])
            timelines.setdefault(market_id, []).append(datetime.fromisoformat(str(row["ts"])))

        if market_ids is not None:
            missing_market_ids = sorted(set(market_ids) - set(timelines))
            if missing_market_ids:
                logger.warning("Requested markets missing snapshots: %s", ", ".join(missing_market_ids))

        if self.config.eval_stride > 1:
            for market_id in timelines:
                ts_list = timelines[market_id]
                strided = ts_list[:: self.config.eval_stride]
                if strided[-1] != ts_list[-1]:
                    strided.append(ts_list[-1])
                timelines[market_id] = strided

        return timelines

    def _iter_market_events(self, timelines: dict[str, list[datetime]]) -> Iterator[tuple[str, datetime]]:
        heap: list[tuple[datetime, str, int]] = []
        for market_id, timestamps in timelines.items():
            if timestamps:
                heapq.heappush(heap, (timestamps[0], market_id, 0))

        while heap:
            timestamp, market_id, index = heapq.heappop(heap)
            yield market_id, timestamp
            next_index = index + 1
            market_timestamps = timelines[market_id]
            if next_index < len(market_timestamps):
                heapq.heappush(heap, (market_timestamps[next_index], market_id, next_index))

    def _warm_resolution_cache(self, market_ids: Sequence[str]) -> None:
        for market_id in market_ids:
            self._get_resolution(market_id)

    def _preload_market_data(self, timelines: dict[str, list[datetime]]) -> None:
        """Batch-load market snapshots for timeline timestamps into memory."""
        self._market_cache = {}
        self._history_by_market = {}
        self._history_index_by_key = {}
        self._next_ts_cache = {}
        all_market_ids = list(timelines.keys())
        if not all_market_ids:
            return

        # Build next-timestamp cache from timelines (cheap, in-memory)
        for market_id, timestamps in timelines.items():
            for i, ts in enumerate(timestamps):
                ts_key = isoformat(ts)
                if i + 1 < len(timestamps):
                    next_ts = timestamps[i + 1]
                    self._next_ts_cache[(market_id, ts_key)] = (market_id, isoformat(next_ts))
                else:
                    self._next_ts_cache[(market_id, ts_key)] = None

        # Batch load market metadata (one row per market — fast)
        # Use temp table for large market sets to avoid SQLite's 999-variable limit
        use_temp = len(all_market_ids) > 900
        if use_temp:
            self.conn.execute("CREATE TEMP TABLE IF NOT EXISTS _selected_markets (market_id TEXT PRIMARY KEY)")
            self.conn.execute("DELETE FROM _selected_markets")
            self.conn.executemany(
                "INSERT OR IGNORE INTO _selected_markets VALUES (?)",
                [(m,) for m in all_market_ids],
            )
            meta_filter = "JOIN _selected_markets sm ON sm.market_id = markets.market_id"
            rule_filter = "JOIN _selected_markets sm ON sm.market_id = market_rule_revisions.market_id"
            meta_params: tuple[str, ...] = ()
        else:
            placeholders = ", ".join("?" for _ in all_market_ids)
            meta_filter = f"WHERE market_id IN ({placeholders})"
            rule_filter = f"WHERE market_id IN ({placeholders})"
            meta_params = tuple(all_market_ids)

        market_meta: dict[str, Any] = {}
        meta_rows = self.conn.execute(
            f"""
            SELECT market_id, title, domain, market_type, resolution_ts,
                   fees_enabled, fee_rate, fee_exponent, maker_rebate_rate
            FROM markets
            {meta_filter}
            """,
            meta_params,
        ).fetchall()
        for mr in meta_rows:
            market_meta[str(mr["market_id"])] = mr

        # Batch load rule revisions (latest per market)
        rules_by_market: dict[str, tuple[str, str]] = {}
        rule_rows = self.conn.execute(
            f"""
            SELECT market_id, rules_text, additional_context, effective_ts
            FROM market_rule_revisions
            {rule_filter}
            ORDER BY market_id, effective_ts DESC
            """,
            meta_params,
        ).fetchall()
        for rr in rule_rows:
            mid = str(rr["market_id"])
            if mid not in rules_by_market:
                rules_by_market[mid] = (str(rr["rules_text"]), str(rr["additional_context"]))

        # Build (market_id, ts) pairs to load — only timestamps in timelines
        ts_pairs: list[tuple[str, str]] = []
        for market_id, timestamps in timelines.items():
            for ts in timestamps:
                ts_pairs.append((market_id, isoformat(ts)))

        # Load snapshots in chunks using exact (market_id, ts) matching
        chunk_size = 500
        history_by_market: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        for chunk_start in range(0, len(ts_pairs), chunk_size):
            chunk = ts_pairs[chunk_start : chunk_start + chunk_size]
            # Use a VALUES-based approach for exact lookups
            conditions = " OR ".join("(s.market_id = ? AND s.ts = ?)" for _ in chunk)
            params: list[str] = []
            for mid, ts_str in chunk:
                params.extend([mid, ts_str])
            rows = self.conn.execute(
                f"""
                SELECT s.market_id, s.ts, s.status, s.best_bid, s.best_ask,
                       s.mid, s.last_trade, s.volume_1m, s.volume_24h,
                       s.open_interest, s.tick_size
                FROM market_snapshots s
                WHERE {conditions}
                """,
                tuple(params),
            ).fetchall()

            for row in rows:
                market_id = str(row["market_id"])
                ts = datetime.fromisoformat(str(row["ts"]))
                ts_str = isoformat(ts)
                meta = market_meta.get(market_id)
                rules = rules_by_market.get(market_id, ("", ""))

                if meta is None:
                    continue

                state = MarketState(
                    market_id=market_id,
                    title=str(meta["title"]),
                    domain=str(meta["domain"]),
                    market_type=str(meta["market_type"]),
                    ts=ts,
                    status=str(row["status"]),
                    best_bid=float(row["best_bid"]),
                    best_ask=float(row["best_ask"]),
                    mid=float(row["mid"]),
                    last_trade=float(row["last_trade"]),
                    volume_1m=float(row["volume_1m"]),
                    volume_24h=float(row["volume_24h"]),
                    open_interest=float(row["open_interest"]),
                    tick_size=float(row["tick_size"]),
                    rules_text=rules[0],
                    additional_context=rules[1],
                    resolution_ts=(
                        datetime.fromisoformat(str(meta["resolution_ts"])) if meta["resolution_ts"] else None
                    ),
                    fees_enabled=bool(meta["fees_enabled"]),
                    fee_rate=float(meta["fee_rate"]),
                    fee_exponent=float(meta["fee_exponent"]),
                    maker_rebate_rate=float(meta["maker_rebate_rate"]),
                    orderbook=[],
                )
                state = self._apply_market_category_metadata(state)
                self._market_cache[(market_id, ts_str)] = state
                history_by_market.setdefault(market_id, []).append(
                    (
                        ts_str,
                        {
                            "market_id": market_id,
                            "ts": ts_str,
                            "status": str(row["status"]),
                            "best_bid": float(row["best_bid"]),
                            "best_ask": float(row["best_ask"]),
                            "mid": float(row["mid"]),
                            "last_trade": float(row["last_trade"]),
                            "volume_1m": float(row["volume_1m"]),
                            "volume_24h": float(row["volume_24h"]),
                            "open_interest": float(row["open_interest"]),
                            "tick_size": float(row["tick_size"]),
                            "resolution_ts": (
                                isoformat(datetime.fromisoformat(str(meta["resolution_ts"])))
                                if meta["resolution_ts"]
                                else None
                            ),
                        },
                    )
                )

        for market_id, history in history_by_market.items():
            history.sort(key=lambda item: item[0])
            self._history_by_market[market_id] = history
            for index, (ts_str, _) in enumerate(history):
                self._history_index_by_key[(market_id, ts_str)] = index

        logger.info(
            "Preloaded %d market states and %d market histories for %d markets",
            len(self._market_cache),
            len(self._history_by_market),
            len(all_market_ids),
        )

    def _get_resolution(self, market_id: str) -> sqlite3.Row | None:
        if market_id not in self._resolution_cache:
            self._resolution_cache[market_id] = db.get_resolution(self.conn, market_id)
        return self._resolution_cache[market_id]

    def _apply_market_category_metadata(self, market: MarketState) -> MarketState:
        market_categories = getattr(self, "market_categories", {})
        tags = normalize_market_tags(market_categories.get(market.market_id, market.tags))
        fees_enabled, fee_rate = category_fee_settings(tags)
        return dc_replace(
            market,
            tags=tags,
            fees_enabled=fees_enabled,
            fee_rate=fee_rate,
        )

    def _get_cached_market(self, market_id: str, timestamp: datetime) -> MarketState | None:
        """Look up a market state from cache, falling back to DB if needed."""
        ts_key = isoformat(timestamp)
        cached = self._market_cache.get((market_id, ts_key))
        if cached is not None:
            return cached
        market = db.get_market_state_as_of(self.conn, market_id, timestamp)
        if market is None:
            return None
        return self._apply_market_category_metadata(market)

    def _get_cached_next_market(self, market_id: str, timestamp: datetime) -> MarketState | None:
        """Look up the next market state from cache, falling back to DB if needed."""
        ts_key = isoformat(timestamp)
        next_info = self._next_ts_cache.get((market_id, ts_key))
        if next_info is not None:
            next_market_id, next_ts_str = next_info
            cached = self._market_cache.get((next_market_id, next_ts_str))
            if cached is not None:
                return cached
        # Cache miss or not in next_ts_cache — fall back to DB
        if (market_id, ts_key) in self._next_ts_cache and next_info is None:
            # Explicitly known to have no next snapshot
            return None
        market = db.get_next_market_state(self.conn, market_id, timestamp)
        if market is None:
            return None
        return self._apply_market_category_metadata(market)

    def _get_market_history(
        self,
        market_id: str,
        timestamp: datetime,
        *,
        lookback: int = 24,
    ) -> list[dict[str, Any]]:
        history = self._history_by_market.get(market_id, [])
        if not history or lookback <= 0:
            return []

        ts_key = isoformat(timestamp)
        current_index = self._history_index_by_key.get((market_id, ts_key))
        if current_index is None:
            current_index = bisect_left([entry_ts for entry_ts, _ in history], ts_key)

        start_index = max(0, current_index - lookback)
        return [snapshot for _, snapshot in history[start_index:current_index]]

    def _process_market_snapshot(self, market_id: str, timestamp: datetime) -> None:
        if self._is_market_resolved_as_of(market_id, timestamp):
            logger.info(
                "Skipping market_id=%s at ts=%s because it is already resolved",
                market_id,
                isoformat(timestamp),
            )
            return

        market = self._get_cached_market(market_id, timestamp)
        if market is None:
            logger.warning(
                "Missing market state for market_id=%s at ts=%s",
                market_id,
                isoformat(timestamp),
            )
            return

        market = self._normalize_quotes(market)
        market = self._ensure_orderbook(market, reason="current_snapshot")
        history = self._get_market_history(market_id, timestamp, lookback=24)

        try:
            forecast, prompt_hash, context_hash = self.grok.forecast(
                market_id,
                timestamp,
                market_state=market,
                prev_snapshots=history,
            )
        except Exception:
            logger.exception(
                "Forecast failed for market_id=%s at ts=%s",
                market_id,
                isoformat(timestamp),
            )
            return

        self._persist_model_output(forecast, prompt_hash, context_hash)

        next_market = self._get_cached_next_market(market_id, timestamp)
        if next_market is not None:
            next_market = self._normalize_quotes(next_market)
            next_market = self._ensure_orderbook(next_market, reason="next_snapshot")

        for strategy in self.strategies:
            portfolio = self.portfolios[strategy.name]
            position = portfolio.positions.get(market_id)
            just_exited = False

            if self.strategy_engine.should_exit(
                config=strategy,
                market=market,
                forecast=forecast,
                position=position,
            ):
                exit_order = self.strategy_engine.exit_order(
                    config=strategy,
                    market=market,
                    position=position,
                )
                self._execute_order(
                    portfolio,
                    market,
                    next_market,
                    exit_order,
                    entry_probability=position.entry_probability if position is not None else market.mid,
                )
                just_exited = True

            if just_exited:
                self._mark_portfolio_from_market(
                    portfolio=portfolio,
                    strategy_name=strategy.name,
                    market=market,
                )
                continue

            orders = self.strategy_engine.decide(
                config=strategy,
                market=market,
                forecast=forecast,
                position=portfolio.positions.get(market_id),
                available_cash=self._available_cash_for_entries(portfolio, strategy),
                portfolio_cash=portfolio.cash,
                starting_cash=self.config.starting_cash,
                total_invested=self._total_invested_capital(portfolio),
                on_missed_trade=(
                    lambda edge_bps, reason, *, _portfolio=portfolio, _market=market, _strategy=strategy:
                    self._record_missed_trade(
                        portfolio=_portfolio,
                        strategy_name=_strategy.name,
                        market_id=_market.market_id,
                        ts=_market.ts,
                        edge_bps=edge_bps,
                        reason=reason,
                    )
                ),
            )
            for order in orders:
                self._execute_order(
                    portfolio,
                    market,
                    next_market,
                    order,
                    entry_probability=(
                        1.0 - forecast.probability_yes if order.is_no_bet else forecast.probability_yes
                    ),
                )

            self._mark_portfolio_from_market(
                portfolio=portfolio,
                strategy_name=strategy.name,
                market=market,
            )

    def _is_market_resolved_as_of(self, market_id: str, as_of: datetime) -> bool:
        resolution = self._get_resolution(market_id)
        if resolution is None or resolution["resolution_ts"] is None:
            return False
        resolution_ts = datetime.fromisoformat(str(resolution["resolution_ts"]))
        return ensure_utc(as_of) >= ensure_utc(resolution_ts)

    def _resolve_final_replay_ts(
        self,
        last_replay_ts: datetime | None,
        market_ids: Sequence[str],
    ) -> datetime | None:
        candidates: list[datetime] = []
        if last_replay_ts is not None:
            candidates.append(last_replay_ts)
        for market_id in market_ids:
            resolution = self._get_resolution(market_id)
            if resolution is not None and resolution["resolution_ts"] is not None:
                candidates.append(datetime.fromisoformat(str(resolution["resolution_ts"])))
        if not candidates:
            return None
        return max(candidates, key=ensure_utc)

    def _snapshot_portfolios(self) -> dict[str, StrategyPortfolio]:
        return {
            strategy_name: StrategyPortfolio(
                cash=portfolio.cash,
                last_known_mids=dict(portfolio.last_known_mids),
                realized_pnl=portfolio.realized_pnl,
                missed_trades=portfolio.missed_trades,
                missed_edge_bps=portfolio.missed_edge_bps,
                positions={market_id: dc_replace(position) for market_id, position in portfolio.positions.items()},
            )
            for strategy_name, portfolio in self.portfolios.items()
        }

    def _strategy_config(self, strategy_name: str) -> StrategyConfig | None:
        if hasattr(self, "_strategy_by_name"):
            return self._strategy_by_name.get(strategy_name)
        for strategy in getattr(self, "strategies", []):
            if strategy.name == strategy_name:
                return strategy
        return None

    def _total_invested_capital(self, portfolio: StrategyPortfolio) -> float:
        return max(0.0, self.config.starting_cash - portfolio.cash)

    def _available_cash_for_entries(self, portfolio: StrategyPortfolio, strategy: StrategyConfig) -> float:
        max_invested_capital = self.config.starting_cash * strategy.max_total_invested_pct
        remaining_investable = max(0.0, max_invested_capital - self._total_invested_capital(portfolio))
        return min(portfolio.cash, remaining_investable)

    def _record_missed_trade(
        self,
        *,
        portfolio: StrategyPortfolio,
        strategy_name: str,
        market_id: str,
        ts: datetime,
        edge_bps: float,
        reason: str,
    ) -> None:
        portfolio.missed_trades += 1
        portfolio.missed_edge_bps += max(0.0, edge_bps)
        logger.info(
            "Missed trade strategy=%s market_id=%s ts=%s reason=%s edge_bps=%.2f cash=%.4f",
            strategy_name,
            market_id,
            isoformat(ts),
            reason,
            edge_bps,
            portfolio.cash,
        )

    def _normalize_quotes(self, market: MarketState) -> MarketState:
        """Clamp quotes to a valid range and preserve a positive spread."""
        min_price = 0.001
        max_price = 0.999
        spread_floor = max(market.tick_size, 0.01)

        best_bid = max(min_price, min(max_price, market.best_bid))
        best_ask = max(min_price, min(max_price, market.best_ask))
        if best_bid >= best_ask:
            best_ask = min(max_price, best_bid + spread_floor)
        if best_bid >= best_ask:
            best_bid = max(min_price, best_ask - spread_floor)

        return dc_replace(
            market,
            best_bid=round(best_bid, 4),
            best_ask=round(best_ask, 4),
            mid=round((best_bid + best_ask) / 2.0, 4),
        )

    def _ensure_orderbook(self, market: MarketState, *, reason: str) -> MarketState:
        if market.orderbook:
            return market

        tick_size = max(market.tick_size, 0.001)
        # Synthetic liquidity: derive per-minute volume from volume_1m when
        # available, otherwise approximate from volume_24h (total daily volume
        # divided by 1440 minutes).  Scale conservatively at 1% of per-minute
        # volume so a $1M/day market gets ~7 contracts (capped at 50).
        per_minute_volume = market.volume_1m if market.volume_1m > 0 else market.volume_24h / 1440.0
        synthetic_quantity = min(50.0, max(1.0, per_minute_volume * 0.01))
        bid_price = market.best_bid if 0.0 < market.best_bid < 1.0 else max(0.001, market.mid - tick_size / 2.0)
        ask_price = market.best_ask if 0.0 < market.best_ask < 1.0 else min(0.999, market.mid + tick_size / 2.0)
        bid_price = round(max(0.001, min(0.999 - tick_size, bid_price)), 4)
        ask_price = round(min(0.999, max(bid_price + tick_size, ask_price)), 4)

        logger.warning(
            "Synthetic orderbook created for market_id=%s at ts=%s (%s) — "
            "fills against this book are synthetic and may not reflect real liquidity",
            market.market_id,
            isoformat(market.ts),
            reason,
        )
        return dc_replace(
            market,
            orderbook=[
                OrderLevel(side="bid", level_no=1, price=bid_price, quantity=synthetic_quantity),
                OrderLevel(side="ask", level_no=1, price=ask_price, quantity=synthetic_quantity),
            ],
        )

    def _build_degraded_next_market(self, market: MarketState, order: Any) -> MarketState:
        logger.warning(
            "Missing next snapshot for market_id=%s at ts=%s; using degraded execution assumptions",
            market.market_id,
            isoformat(market.ts),
        )
        normalized_market = self._ensure_orderbook(market, reason="degraded_next_market")
        effective_1m = (
            normalized_market.volume_1m if normalized_market.volume_1m > 0 else normalized_market.volume_24h / 1440.0
        )
        return dc_replace(
            normalized_market,
            volume_1m=max(effective_1m * 0.25, order.requested_quantity * 3.0, 1.0),
        )

    def _normalized_resolved_outcome(
        self,
        resolution: sqlite3.Row,
        market_id: str,
        *,
        log_ambiguity: bool,
    ) -> float | None:
        raw_outcome = resolution["resolved_outcome"]
        if raw_outcome is None:
            logger.warning("Resolution missing resolved_outcome for market_id=%s", market_id)
            return None

        outcome = float(raw_outcome)
        clamped_outcome = max(0.0, min(1.0, outcome))
        if clamped_outcome != outcome:
            logger.warning(
                "Clamped out-of-range resolution for market_id=%s from %.4f to %.4f",
                market_id,
                outcome,
                clamped_outcome,
            )

        is_ambiguous = 0.01 < clamped_outcome < 0.99
        if log_ambiguity and (is_ambiguous or bool(resolution["disputed"])):
            logger.warning(
                "Ambiguous settlement for market_id=%s outcome=%.4f status=%s "
                "disputed=%s clarification_issued=%s note=%s",
                market_id,
                clamped_outcome,
                str(resolution["status"]),
                bool(resolution["disputed"]),
                bool(resolution["clarification_issued"]),
                str(resolution["resolution_note"] or ""),
            )

        if clamped_outcome <= 0.01:
            return 0.0
        if clamped_outcome >= 0.99:
            return 1.0
        return clamped_outcome

    def _settlement_fee_rate(self, market_id: str) -> float:
        """Return the fee rate to apply on profitable settlement payouts.

        Polymarket does NOT charge fees on resolution/settlement as of 2025-2026.
        Trading fees only apply on certain market types (crypto, some sports).
        Settlement payouts are fee-free. Return 0.0 for all markets.
        """
        return 0.0

    def _persist_model_output(self, forecast: Any, prompt_hash: str, context_hash: str) -> None:
        self.conn.execute(
            """
            INSERT INTO model_outputs (
                experiment_id, market_id, ts, agent_name, domain, model_id,
                model_release, prompt_hash, context_hash, probability_yes,
                confidence, expected_edge_bps, thesis, reasoning, evidence_json,
                raw_response_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.experiment_id,
                forecast.market_id,
                isoformat(forecast.as_of),
                forecast.agent_name,
                forecast.domain,
                forecast.model_id,
                forecast.model_release,
                prompt_hash,
                context_hash,
                forecast.probability_yes,
                forecast.confidence,
                forecast.expected_edge_bps,
                forecast.thesis,
                forecast.reasoning,
                json.dumps(forecast.evidence, sort_keys=True),
                json.dumps(forecast.raw_response, sort_keys=True),
            ),
        )

    def _execute_order(
        self,
        portfolio: StrategyPortfolio,
        market: Any,
        next_market: Any,
        order: Any,
        entry_probability: float,
    ) -> None:
        market = self._ensure_orderbook(self._normalize_quotes(market), reason="execution_market")
        effective_next_market = (
            self._ensure_orderbook(
                self._normalize_quotes(next_market),
                reason="execution_next_market",
            )
            if next_market is not None
            else self._build_degraded_next_market(market, order)
        )
        effective_market = self.simulator._market_for_intent(market, order)
        effective_execution_next_market = self.simulator._market_for_intent(effective_next_market, order)
        fill_estimate = (
            self.simulator._simulate_aggressive(effective_market, order)
            if order.liquidity_intent == "aggressive"
            else self.simulator._simulate_passive(effective_market, effective_execution_next_market, order)
        )
        estimated_notional = fill_estimate.vwap_price * fill_estimate.quantity
        strategy = self._strategy_config(order.strategy_name)
        if order.side == "buy" and strategy is not None:
            reserve_cash = self.config.starting_cash * (1.0 - strategy.max_total_invested_pct)
            low_edge_cutoff = strategy.edge_threshold_bps * 1.5
            cash_after_fill = portfolio.cash - estimated_notional
            if cash_after_fill < reserve_cash and order.edge_bps < low_edge_cutoff:
                self._record_missed_trade(
                    portfolio=portfolio,
                    strategy_name=order.strategy_name,
                    market_id=order.market_id,
                    ts=order.ts,
                    edge_bps=order.edge_bps,
                    reason="cash_pressure_low_edge",
                )
                logger.warning(
                    "Skipping buy order for strategy=%s market_id=%s at ts=%s because "
                    "cash would fall below reserve %.4f on low edge %.2f bps",
                    order.strategy_name,
                    order.market_id,
                    isoformat(order.ts),
                    reserve_cash,
                    order.edge_bps,
                )
                return
        if order.side == "buy" and estimated_notional > portfolio.cash:
            self._record_missed_trade(
                portfolio=portfolio,
                strategy_name=order.strategy_name,
                market_id=order.market_id,
                ts=order.ts,
                edge_bps=order.edge_bps,
                reason="insufficient_cash",
            )
            logger.warning(
                "Skipping buy order for strategy=%s market_id=%s at ts=%s because "
                "estimated notional %.4f exceeds cash %.4f",
                order.strategy_name,
                order.market_id,
                isoformat(order.ts),
                estimated_notional,
                portfolio.cash,
            )
            return

        order_id = new_order_id()
        self.conn.execute(
            """
            INSERT INTO orders (
                order_id, experiment_id, strategy_name, market_id, ts, side,
                liquidity_intent, limit_price, requested_quantity, edge_bps,
                kelly_fraction, holding_period_minutes, thesis
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                order_id,
                self.experiment_id,
                order.strategy_name,
                order.market_id,
                isoformat(order.ts),
                order.side,
                order.liquidity_intent,
                order.limit_price,
                order.requested_quantity,
                order.edge_bps,
                order.kelly_fraction,
                order.holding_period_minutes,
                order.thesis,
            ),
        )
        fills = self.simulator.simulate(
            order_id=order_id,
            market=market,
            next_market=effective_next_market,
            intent=order,
        )
        total_filled = 0.0
        for fill in fills:
            total_filled += fill.quantity
            self._persist_fill(fill)
            self._apply_fill(
                portfolio,
                fill,
                order.thesis,
                entry_probability,
                is_no_bet=order.is_no_bet,
            )
        self.conn.execute(
            "UPDATE orders SET filled_quantity = ? WHERE order_id = ?",
            (total_filled, order_id),
        )

    def _persist_fill(self, fill: FillResult) -> None:
        self.conn.execute(
            """
            INSERT INTO fills (
                fill_id, order_id, experiment_id, market_id, strategy_name,
                fill_ts, side, liquidity_role, price, quantity, notional_usdc,
                fee_usdc, rebate_usdc, impact_bps, fill_delay_seconds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                fill.order_id,
                self.experiment_id,
                fill.market_id,
                fill.strategy_name,
                isoformat(fill.fill_ts),
                fill.side,
                fill.liquidity_role,
                fill.price,
                fill.quantity,
                round(fill.price * fill.quantity, 4),
                fill.fee_usdc,
                fill.rebate_usdc,
                fill.impact_bps,
                fill.fill_delay_seconds,
            ),
        )

    def _apply_fill(
        self,
        portfolio: StrategyPortfolio,
        fill: FillResult,
        thesis: str,
        entry_probability: float,
        *,
        is_no_bet: bool = False,
    ) -> None:
        position = portfolio.positions.setdefault(
            fill.market_id,
            PositionState(strategy_name=fill.strategy_name, market_id=fill.market_id),
        )
        notional = fill.price * fill.quantity
        should_evict = False
        if fill.side == "buy":
            fill_cost = notional + fill.fee_usdc - fill.rebate_usdc
            # Strict cash guard: never allow cash to go negative
            if portfolio.cash - fill_cost < 0:
                max_affordable_qty = portfolio.cash / (fill.price + 1e-12)
                if max_affordable_qty < self.simulator.minimum_fill_quantity:
                    logger.warning(
                        "Cash guard: rejecting buy fill for market_id=%s — cost %.4f exceeds cash %.4f",
                        fill.market_id,
                        fill_cost,
                        portfolio.cash,
                    )
                    return
                # Reduce fill to what we can afford
                logger.warning(
                    "Cash guard: reducing buy fill from %.4f to %.4f for market_id=%s",
                    fill.quantity,
                    max_affordable_qty,
                    fill.market_id,
                )
                fill = FillResult(
                    order_id=fill.order_id,
                    market_id=fill.market_id,
                    strategy_name=fill.strategy_name,
                    fill_ts=fill.fill_ts,
                    side=fill.side,
                    liquidity_role=fill.liquidity_role,
                    price=fill.price,
                    quantity=round(max_affordable_qty, 4),
                    fee_usdc=round(fill.fee_usdc * (max_affordable_qty / fill.quantity), 4),
                    rebate_usdc=round(fill.rebate_usdc * (max_affordable_qty / fill.quantity), 4),
                    impact_bps=fill.impact_bps,
                    fill_delay_seconds=fill.fill_delay_seconds,
                )
                notional = fill.price * fill.quantity
                fill_cost = notional + fill.fee_usdc - fill.rebate_usdc

            portfolio.cash -= fill_cost
            assert portfolio.cash >= -1e-9, f"Cash went negative: {portfolio.cash}"
            portfolio.cash = max(0.0, portfolio.cash)  # clamp floating point dust

            new_quantity = position.quantity + fill.quantity
            if new_quantity > 0:
                position.avg_entry_price = (
                    (position.avg_entry_price * position.quantity) + (fill.price * fill.quantity)
                ) / new_quantity
            position.quantity = new_quantity
            position.total_opened_quantity += fill.quantity
            position.total_opened_notional += notional
            if position.opened_ts is None:
                position.opened_ts = fill.fill_ts
                position.closed_ts = None
                position.entry_probability = entry_probability
                position.thesis = thesis
                position.is_no_bet = is_no_bet
        else:
            # Fix 6: warn when sell quantity exceeds position
            if fill.quantity > position.quantity:
                logger.warning(
                    "Fill quantity %f exceeds position %f, clamping",
                    fill.quantity,
                    position.quantity,
                )
            sold_qty = min(position.quantity, fill.quantity)
            portfolio.cash += fill.price * sold_qty - fill.fee_usdc + fill.rebate_usdc
            realized = (fill.price - position.avg_entry_price) * sold_qty
            portfolio.realized_pnl += realized
            position.realized_pnl += realized
            position.realized_pnl_pre_resolution += realized
            position.quantity -= sold_qty
            if position.quantity <= 0:
                position.quantity = 0.0
                position.closed_ts = fill.fill_ts
                should_evict = True
        position.fees_paid += fill.fee_usdc
        position.rebates_earned += fill.rebate_usdc
        assert portfolio.cash >= -1e-9, f"Cash went negative after fill: {portfolio.cash}"
        portfolio.cash = max(0.0, portfolio.cash)
        if should_evict:
            self._persist_and_evict_position(portfolio, fill.market_id)

    def _mark_portfolio_from_market(
        self,
        *,
        portfolio: StrategyPortfolio,
        strategy_name: str,
        market: MarketState,
        mark_ts: datetime | None = None,
    ) -> None:
        market = self._normalize_quotes(market)
        market_id = market.market_id
        position = portfolio.positions.get(market_id)
        mark_price = self._position_mark_price(market, position)
        portfolio.last_known_mids[market_id] = mark_price
        position_qty = position.quantity if position is not None else 0.0
        inventory_value = position_qty * mark_price
        total_inventory_value = sum(
            position_state.quantity * portfolio.last_known_mids.get(position_market_id, position_state.avg_entry_price)
            for position_market_id, position_state in portfolio.positions.items()
            if position_state.quantity > 0
        )
        unrealized = 0.0
        if position is not None and position_qty > 0:
            unrealized = (mark_price - position.avg_entry_price) * position_qty
        equity = portfolio.cash + total_inventory_value
        self.conn.execute(
            """
            INSERT INTO pnl_marks (
                experiment_id, strategy_name, market_id, ts, cash, position_qty,
                mark_price, inventory_value, equity, realized_pnl, unrealized_pnl
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.experiment_id,
                strategy_name,
                market_id,
                isoformat(mark_ts or market.ts),
                round(portfolio.cash, 4),
                round(position_qty, 4),
                round(mark_price, 4),
                round(inventory_value, 4),
                round(equity, 4),
                round(portfolio.realized_pnl, 4),
                round(unrealized, 4),
            ),
        )

    def _settle_resolved_positions(self, replay_ts: datetime) -> None:
        for strategy in self.strategies:
            portfolio = self.portfolios[strategy.name]
            for market_id, position in list(portfolio.positions.items()):
                if position.quantity <= 0:
                    continue
                resolution = self._get_resolution(market_id)
                if resolution is None or resolution["resolution_ts"] is None:
                    continue
                resolution_ts = datetime.fromisoformat(str(resolution["resolution_ts"]))
                if ensure_utc(replay_ts) < ensure_utc(resolution_ts):
                    continue

                resolved_outcome = self._normalized_resolved_outcome(
                    resolution,
                    market_id,
                    log_ambiguity=True,
                )
                if resolved_outcome is None:
                    continue
                payout_ratio = self._payout_ratio_for_position(position, resolved_outcome)

                gross_payout = payout_ratio * position.quantity
                # Polymarket fees: 2% on profit (winnings), not on total payout
                profit = gross_payout - position.total_opened_notional
                settlement_fee = max(0.0, profit) * self._settlement_fee_rate(market_id)
                net_payout = gross_payout - settlement_fee
                settlement_increment = net_payout - position.avg_entry_price * position.quantity
                logger.info(
                    "Settling strategy=%s market_id=%s qty=%.4f payout_ratio=%.4f "
                    "gross_payout=%.4f settlement_fee=%.4f net_payout=%.4f is_no_bet=%s "
                    "replay_ts=%s resolution_ts=%s",
                    strategy.name,
                    market_id,
                    position.quantity,
                    payout_ratio,
                    gross_payout,
                    settlement_fee,
                    net_payout,
                    position.is_no_bet,
                    isoformat(replay_ts),
                    isoformat(resolution_ts),
                )
                portfolio.cash += net_payout
                position.realized_pnl += settlement_increment
                portfolio.realized_pnl += settlement_increment
                position.fees_paid += settlement_fee
                position.quantity = 0.0
                position.closed_ts = resolution_ts
                self._persist_and_evict_position(portfolio, market_id)

    def _persist_and_evict_position(self, portfolio: StrategyPortfolio, market_id: str) -> None:
        position = portfolio.positions.get(market_id)
        if position is None:
            return
        self._persist_position(position)
        position.opened_ts = None
        position.entry_probability = 0.0
        position.thesis = ""
        position.avg_entry_price = 0.0
        position.is_no_bet = False
        del portfolio.positions[market_id]

    def _persist_position(self, position: PositionState) -> None:
        if position.opened_ts is None:
            return

        market_id = position.market_id
        resolution = self._get_resolution(market_id)
        resolved_outcome = None
        resolution_ts = None
        hold_to_resolution_pnl = 0.0
        if resolution is not None:
            resolved_outcome = self._normalized_resolved_outcome(
                resolution,
                market_id,
                log_ambiguity=False,
            )
            resolution_ts = str(resolution["resolution_ts"]) if resolution["resolution_ts"] else None
            if position.total_opened_quantity > 0 and resolved_outcome is not None:
                payout_ratio = self._payout_ratio_for_position(position, resolved_outcome)
                hold_to_resolution_pnl = (
                    (payout_ratio * position.total_opened_quantity)
                    - position.total_opened_notional
                    - position.fees_paid
                    + position.rebates_earned
                )
        status = "closed" if position.quantity <= 0 else "open"
        self.conn.execute(
            """
            INSERT INTO positions (
                position_id, experiment_id, strategy_name, market_id,
                opened_ts, closed_ts, quantity, avg_entry_price,
                avg_exit_price, status, realized_pnl_pre_resolution,
                hold_to_resolution_pnl, resolved_outcome, resolution_ts,
                thesis
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                self.experiment_id,
                position.strategy_name,
                market_id,
                isoformat(opened) if (opened := position.opened_ts) is not None else None,
                isoformat(closed) if (closed := position.closed_ts) is not None else resolution_ts,
                position.quantity,
                position.avg_entry_price,
                None,
                status,
                round(position.realized_pnl_pre_resolution, 4),
                round(hold_to_resolution_pnl, 4),
                resolved_outcome,
                resolution_ts,
                position.thesis,
            ),
        )

    def _position_mark_price(self, market: MarketState, position: PositionState | None) -> float:
        if position is not None and position.is_no_bet:
            return min(0.999, max(0.001, 1.0 - market.mid))
        return market.mid

    def _payout_ratio_for_position(self, position: PositionState, resolved_outcome: float) -> float:
        return 1.0 - resolved_outcome if position.is_no_bet else resolved_outcome

    def _persist_positions(self) -> None:
        for strategy in self.strategies:
            portfolio = self.portfolios[strategy.name]
            for position in portfolio.positions.values():
                self._persist_position(position)
        self.conn.commit()
