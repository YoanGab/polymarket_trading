"""Multi-market Gymnasium environment for portfolio-level RL training.

The agent sees N markets simultaneously with a shared cash pool and must
decide *where* and *when* to invest across the full opportunity set.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import db
from .trading_env import Action, _MarketEpisode, _TradingCore
from .types import MarketState, ensure_utc

# Per-market feature count extracted from TradingState.to_array()
_MARKET_FEATURE_DIM = 31
# Portfolio-level features: cash_frac, total_invested_frac, n_positions_frac,
# unrealized_pnl_frac, portfolio_value_frac
_PORTFOLIO_DIM = 5
# Core RL actions per market (keep simple for learning)
_N_ACTIONS = 5  # hold, buy_yes, buy_no, sell_yes, sell_no

_ACTION_MAP: dict[int, Action] = {
    0: Action.hold(),
    1: Action.buy_yes(),
    2: Action.buy_no(),
    3: Action.sell_yes(),
    4: Action.sell_no(),
}


class MLScreener(Protocol):
    """Protocol for the ML screener that ranks markets by predicted edge."""

    def rank_markets(
        self,
        market_ids: list[str],
        as_of: datetime,
    ) -> list[tuple[str, float, float]]:
        """Return top markets sorted by edge.

        Args:
            market_ids: Pool of candidate market IDs.
            as_of: Current timestamp.

        Returns:
            List of (market_id, probability_yes, edge_bps) sorted descending
            by absolute edge.
        """
        ...


class PolymarketMultiMarketGymEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    """Gymnasium wrapper for multi-market portfolio RL training.

    The agent observes N market slots plus portfolio-level features and
    outputs one discrete action per slot at each timestep.

    Observation space (Dict):
        - ``market_features``: (N, M) float32 array of per-market features
        - ``portfolio``: (P,) float32 array of portfolio-level features
        - ``market_mask``: (N,) float32 binary mask (1 = active slot)
        - ``ml_edges``: (N, 2) float32 array of (probability_yes, edge_bps)

    Action space:
        MultiDiscrete([5] * N) -- per-slot action in {hold, buy_yes,
        buy_no, sell_yes, sell_no}.

    Reward:
        Delta portfolio value (cash + all positions marked to market)
        between consecutive steps.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        db_path: str | Path,
        starting_cash: float = 1_000.0,
        n_markets: int = 20,
        split: str = "train",
        ml_screener: MLScreener | None = None,
        *,
        auto_order_cash_fraction: float = 0.10,
        feature_lookback: int = 24,
        enable_ml_predictions: bool = False,
        episode_hours: float = 168.0,
        rescreen_interval: int = 0,
        random_seed: int | None = None,
    ) -> None:
        """Initialize the multi-market gym environment.

        Args:
            db_path: Path to the SQLite database.
            starting_cash: Initial cash balance.
            n_markets: Number of simultaneous market slots.
            split: Data split to use (train/val/test).
            ml_screener: Optional callable that ranks markets by edge.
                If None, markets are selected randomly.
            auto_order_cash_fraction: Fraction of cash per auto-sized order.
            feature_lookback: Number of past snapshots for feature extraction.
            enable_ml_predictions: Whether to use ML transport for predictions.
            episode_hours: Length of one episode in hours (default 1 week).
            rescreen_interval: Re-run screener every N steps to swap markets.
                0 means never re-screen mid-episode.
            random_seed: Seed for reproducibility.
        """
        super().__init__()
        self.db_path = str(db_path)
        self.n_markets = n_markets
        self.starting_cash = starting_cash
        self.ml_screener = ml_screener
        self.episode_hours = episode_hours
        self.rescreen_interval = rescreen_interval

        self.conn = db.connect(db_path)
        db.init_db(self.conn)

        self._core = _TradingCore(
            conn=self.conn,
            starting_cash=starting_cash,
            auto_order_cash_fraction=auto_order_cash_fraction,
            top_related_markets=0,
            feature_lookback=feature_lookback,
            enable_ml_predictions=enable_ml_predictions,
            random_seed=random_seed,
        )

        self._rng = random.Random(random_seed)
        self._np_rng = np.random.default_rng(random_seed)
        self.split = split
        self._all_market_ids = self._market_ids_for_split(split)
        if not self._all_market_ids:
            raise ValueError(f"No markets available for split {split!r}")

        # Pre-compute market time ranges (ONE query at init, used for all
        # subsequent _candidates_at / _pick_episode_start calls).
        self._market_ranges: dict[str, tuple[str, str, str | None]] = {}
        self._global_min_ts: datetime | None = None
        self._global_max_ts: datetime | None = None
        self._precompute_market_ranges()

        # Active episode tracking
        self._episodes: dict[str, _MarketEpisode] = {}
        self._slot_market_ids: list[str | None] = [None] * n_markets
        self._episode_start_ts: datetime | None = None
        self._episode_end_ts: datetime | None = None
        self._step_count = 0

        # ML edge cache per slot
        self._ml_edges: np.ndarray = np.zeros((n_markets, 2), dtype=np.float32)

        # Define spaces
        self.observation_space = spaces.Dict(
            {
                "market_features": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(n_markets, _MARKET_FEATURE_DIM),
                    dtype=np.float32,
                ),
                "portfolio": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(_PORTFOLIO_DIM,),
                    dtype=np.float32,
                ),
                "market_mask": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(n_markets,),
                    dtype=np.float32,
                ),
                "ml_edges": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(n_markets, 2),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = spaces.MultiDiscrete([_N_ACTIONS] * n_markets)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)
            self._np_rng = np.random.default_rng(seed)
            self._core._rng.seed(seed)

        self._core.reset_portfolio()
        self._episodes.clear()
        self._slot_market_ids = [None] * self.n_markets
        self._step_count = 0
        self._ml_edges = np.zeros((self.n_markets, 2), dtype=np.float32)

        # Pick a random episode start time from the available data
        self._episode_start_ts = self._pick_episode_start(options)
        self._episode_end_ts = self._episode_start_ts + timedelta(hours=self.episode_hours)

        # Select initial markets
        candidates = self._candidates_at(self._episode_start_ts)
        self._fill_market_slots(candidates)

        obs = self._build_observation()
        info = self._build_info()
        return obs, info

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self._step_count += 1

        # Compute portfolio value before actions
        before_value = self._portfolio_value()

        # Execute one action per active slot
        for slot_idx in range(self.n_markets):
            market_id = self._slot_market_ids[slot_idx]
            if market_id is None:
                continue
            episode = self._episodes.get(market_id)
            if episode is None or episode.done:
                continue

            act = _ACTION_MAP.get(int(action[slot_idx]), Action.hold())
            self._core.step_episode(episode, act)

        # Handle resolved/done markets -- settle and replace
        self._handle_completed_markets()

        # Optionally re-screen
        if self.rescreen_interval > 0 and self._step_count % self.rescreen_interval == 0:
            self._rescreen_markets()

        # Check if episode is over
        current_ts = self._latest_timestamp()
        done = False
        if current_ts is not None and self._episode_end_ts is not None:
            done = ensure_utc(current_ts) >= ensure_utc(self._episode_end_ts)
        # Also done if all markets exhausted
        if all(ep.done for ep in self._episodes.values()) and not self._has_replacement_candidates():
            done = True

        after_value = self._portfolio_value()
        reward = after_value - before_value

        obs = self._build_observation()
        info = self._build_info()
        info["portfolio_value"] = after_value
        info["step_count"] = self._step_count
        return obs, reward, done, False, info

    def render(self) -> str:
        value = self._portfolio_value()
        active = sum(1 for m in self._slot_market_ids if m is not None)
        n_positions = len([p for p in self._core.portfolio.positions.values() if p.quantity > 0])
        return (
            f"step={self._step_count} active_markets={active}/{self.n_markets} "
            f"cash={self._core.portfolio.cash:.2f} value={value:.2f} "
            f"positions={n_positions}"
        )

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------
    # Observation building
    # ------------------------------------------------------------------

    def _build_observation(self) -> dict[str, np.ndarray]:
        market_features = np.zeros((self.n_markets, _MARKET_FEATURE_DIM), dtype=np.float32)
        market_mask = np.zeros(self.n_markets, dtype=np.float32)

        for slot_idx in range(self.n_markets):
            market_id = self._slot_market_ids[slot_idx]
            if market_id is None:
                continue
            episode = self._episodes.get(market_id)
            if episode is None or episode.done:
                continue

            market_mask[slot_idx] = 1.0
            state = self._core.build_state(episode)
            market_features[slot_idx] = state.to_array()

        portfolio = self._build_portfolio_features()

        return {
            "market_features": market_features,
            "portfolio": portfolio,
            "market_mask": market_mask,
            "ml_edges": self._ml_edges.copy(),
        }

    def _build_portfolio_features(self) -> np.ndarray:
        cash = self._core.portfolio.cash
        cash_frac = cash / max(self.starting_cash, 1.0)

        positions = [p for p in self._core.portfolio.positions.values() if p.quantity > 0]
        total_invested = sum(p.quantity * p.avg_entry_price for p in positions)
        invested_frac = total_invested / max(self.starting_cash, 1.0)

        n_positions = len(positions)
        n_positions_frac = n_positions / max(self.n_markets, 1.0)

        unrealized_pnl = 0.0
        for p in positions:
            key = f"{p.market_id}:NO" if p.is_no_bet else p.market_id
            mark = self._core.portfolio.last_known_mids.get(key, p.avg_entry_price)
            unrealized_pnl += (mark - p.avg_entry_price) * p.quantity
        unrealized_frac = unrealized_pnl / max(self.starting_cash, 1.0)

        value = self._portfolio_value()
        value_frac = value / max(self.starting_cash, 1.0)

        return np.array(
            [cash_frac, invested_frac, n_positions_frac, unrealized_frac, value_frac],
            dtype=np.float32,
        )

    def _build_info(self) -> dict[str, Any]:
        return {
            "active_markets": [m for m in self._slot_market_ids if m is not None],
            "n_active": sum(1 for m in self._slot_market_ids if m is not None),
            "cash": self._core.portfolio.cash,
            "step_count": self._step_count,
        }

    # ------------------------------------------------------------------
    # Market selection and slot management
    # ------------------------------------------------------------------

    def _fill_market_slots(self, candidates: list[str]) -> None:
        """Fill empty market slots from candidates."""
        used_markets = {m for m in self._slot_market_ids if m is not None}

        if self.ml_screener is not None and candidates:
            ts = self._episode_start_ts or datetime.now()
            ranked = self.ml_screener.rank_markets(candidates, ts)
            available = [(mid, prob, edge) for mid, prob, edge in ranked if mid not in used_markets]
        else:
            available_ids = [c for c in candidates if c not in used_markets]
            self._rng.shuffle(available_ids)
            available = [(mid, 0.0, 0.0) for mid in available_ids]

        fill_idx = 0
        for slot_idx in range(self.n_markets):
            if self._slot_market_ids[slot_idx] is not None:
                continue
            if fill_idx >= len(available):
                break

            market_id, prob_yes, edge_bps = available[fill_idx]
            fill_idx += 1

            episode = self._load_episode(market_id)
            if episode is None:
                continue

            self._episodes[market_id] = episode
            self._slot_market_ids[slot_idx] = market_id
            self._ml_edges[slot_idx] = [
                _safe_value(prob_yes),
                _safe_value(edge_bps, 1000.0),
            ]

    def _load_episode(self, market_id: str) -> _MarketEpisode | None:
        """Load snapshot rows for a market and create an episode."""
        rows = self.conn.execute(
            """
            SELECT ts, status, best_bid, best_ask, mid, last_trade, volume_1m,
                   volume_24h, open_interest, tick_size
            FROM market_snapshots
            WHERE market_id = ?
            ORDER BY ts ASC
            """,
            (market_id,),
        ).fetchall()
        if not rows:
            return None

        resolution = db.get_resolution(self.conn, market_id)
        resolution_ts = (
            datetime.fromisoformat(str(resolution["resolution_ts"]))
            if resolution is not None and resolution["resolution_ts"] is not None
            else None
        )
        return _MarketEpisode(
            market_id=market_id,
            snapshot_rows=list(rows),
            resolution_ts=resolution_ts,
            index=0,
            done=False,
        )

    def _handle_completed_markets(self) -> None:
        """Replace markets that have resolved or exhausted snapshots."""
        needs_fill = False
        for slot_idx in range(self.n_markets):
            market_id = self._slot_market_ids[slot_idx]
            if market_id is None:
                continue
            episode = self._episodes.get(market_id)
            if episode is not None and episode.done:
                self._slot_market_ids[slot_idx] = None
                self._ml_edges[slot_idx] = [0.0, 0.0]
                needs_fill = True

        if needs_fill:
            current_ts = self._latest_timestamp()
            if current_ts is not None:
                candidates = self._candidates_at(current_ts)
                self._fill_market_slots(candidates)

    def _rescreen_markets(self) -> None:
        """Re-rank markets and potentially swap low-edge slots."""
        if self.ml_screener is None:
            return
        current_ts = self._latest_timestamp()
        if current_ts is None:
            return

        candidates = self._candidates_at(current_ts)
        used = {m for m in self._slot_market_ids if m is not None}
        ranked = self.ml_screener.rank_markets(candidates, current_ts)

        # Find slots with no position (safe to swap)
        for slot_idx in range(self.n_markets):
            market_id = self._slot_market_ids[slot_idx]
            if market_id is None:
                continue
            # Only swap if no open position in this market
            has_position = any(
                p.quantity > 0 for key, p in self._core.portfolio.positions.items() if p.market_id == market_id
            )
            if has_position:
                continue

            # Check if there's a better candidate
            for new_mid, prob, edge in ranked:
                if new_mid in used:
                    continue
                if abs(edge) > abs(self._ml_edges[slot_idx][1] * 1000.0):
                    # Swap
                    self._slot_market_ids[slot_idx] = new_mid
                    episode = self._load_episode(new_mid)
                    if episode is not None:
                        self._episodes[new_mid] = episode
                    self._ml_edges[slot_idx] = [
                        _safe_value(prob),
                        _safe_value(edge, 1000.0),
                    ]
                    used.add(new_mid)
                    break

    def _candidates_at(self, ts: datetime) -> list[str]:
        """Return market IDs that have snapshots around the given timestamp.

        Uses the pre-computed ``_market_ranges`` dict for O(N) in-memory
        filtering instead of a full database scan.
        """
        ts_iso = ensure_utc(ts).isoformat()
        result: list[str] = []
        for mid, (min_ts, _max_ts, res_ts) in self._market_ranges.items():
            # Market must have at least one snapshot <= ts
            if min_ts > ts_iso:
                continue
            # Market must not have resolved before ts
            if res_ts is not None and res_ts <= ts_iso:
                continue
            result.append(mid)
        return result

    def _has_replacement_candidates(self) -> bool:
        """Check if there are any markets available to fill empty slots.

        Uses ``_candidates_at`` which is now an in-memory operation.
        """
        ts = self._latest_timestamp()
        if ts is None:
            return False
        used = {m for m in self._slot_market_ids if m is not None}
        return any(c not in used for c in self._candidates_at(ts))

    # ------------------------------------------------------------------
    # Episode timing
    # ------------------------------------------------------------------

    def _pick_episode_start(self, options: dict[str, Any] | None) -> datetime:
        """Pick a random start time within the split's data range.

        Uses the pre-computed ``_global_min_ts`` / ``_global_max_ts`` so this
        is a pure in-memory operation (no database access).
        """
        if options is not None and "start_ts" in options:
            return ensure_utc(options["start_ts"])

        if self._global_min_ts is None or self._global_max_ts is None:
            raise ValueError("No snapshot data available")

        min_ts = self._global_min_ts
        max_ts = self._global_max_ts

        # Ensure room for at least one episode
        latest_start = max_ts - timedelta(hours=self.episode_hours)
        if latest_start <= min_ts:
            return min_ts

        # Random point in range
        range_seconds = (latest_start - min_ts).total_seconds()
        offset = self._rng.random() * range_seconds
        return min_ts + timedelta(seconds=offset)

    def _latest_timestamp(self) -> datetime | None:
        """Get the latest timestamp across all active episodes."""
        latest: datetime | None = None
        for market_id in self._slot_market_ids:
            if market_id is None:
                continue
            episode = self._episodes.get(market_id)
            if episode is None:
                continue
            ts = ensure_utc(episode.current_ts)
            if latest is None or ts > latest:
                latest = ts
        return latest

    # ------------------------------------------------------------------
    # Portfolio valuation
    # ------------------------------------------------------------------

    def _portfolio_value(self) -> float:
        """Compute current portfolio value across all active markets."""
        active_markets: dict[str, MarketState] = {}
        for market_id in self._slot_market_ids:
            if market_id is None:
                continue
            episode = self._episodes.get(market_id)
            if episode is None or episode.done:
                continue
            try:
                market = self._core._market_state_for_episode(episode)
                active_markets[market_id] = market
            except (ValueError, IndexError):
                continue
        return self._core.portfolio_value(active_markets=active_markets)

    # ------------------------------------------------------------------
    # Data utilities
    # ------------------------------------------------------------------

    def _market_ids_for_split(self, split: str) -> list[str]:
        """Return all market IDs belonging to the given split."""
        from .splits import HOLDOUT_CUTOFF, TRAIN_CUTOFF, VAL_CUTOFF

        if split == "all":
            condition = ""
            params: tuple[str, ...] = ()
        elif split == "train":
            condition = "WHERE split_ts < ?"
            params = (TRAIN_CUTOFF,)
        elif split == "val":
            condition = "WHERE split_ts >= ? AND split_ts < ?"
            params = (TRAIN_CUTOFF, VAL_CUTOFF)
        elif split == "test":
            condition = "WHERE split_ts >= ? AND split_ts < ?"
            params = (VAL_CUTOFF, HOLDOUT_CUTOFF)
        else:
            raise ValueError(f"Unknown split {split!r}")

        rows = self.conn.execute(
            f"""
            WITH market_bounds AS (
                SELECT m.market_id, COALESCE(m.resolution_ts, MAX(s.ts)) AS split_ts
                FROM markets m
                JOIN market_snapshots s ON s.market_id = m.market_id
                GROUP BY m.market_id
            )
            SELECT market_id
            FROM market_bounds
            {condition}
            ORDER BY market_id
            """,
            params,
        ).fetchall()
        return [str(row["market_id"]) for row in rows]

    def _precompute_market_ranges(self) -> None:
        """Load (min_ts, max_ts, resolution_ts) for every split market in ONE query.

        Populates ``self._market_ranges`` and the global min/max timestamps so
        that ``_candidates_at`` and ``_pick_episode_start`` can work entirely
        in-memory without touching the database.
        """
        market_ids = self._all_market_ids
        if not market_ids:
            return

        # Use a temp table when the market set exceeds SQLite's variable limit.
        if len(market_ids) > 900:
            self.conn.execute("CREATE TEMP TABLE IF NOT EXISTS _mm_ranges_tmp (market_id TEXT PRIMARY KEY)")
            self.conn.execute("DELETE FROM _mm_ranges_tmp")
            self.conn.executemany(
                "INSERT OR IGNORE INTO _mm_ranges_tmp VALUES (?)",
                [(m,) for m in market_ids],
            )
            rows = self.conn.execute(
                """
                SELECT s.market_id,
                       MIN(s.ts) AS min_ts,
                       MAX(s.ts) AS max_ts,
                       m.resolution_ts
                FROM market_snapshots s
                JOIN markets m ON m.market_id = s.market_id
                JOIN _mm_ranges_tmp t ON t.market_id = s.market_id
                GROUP BY s.market_id
                """
            ).fetchall()
        else:
            placeholders = ",".join("?" * len(market_ids))
            rows = self.conn.execute(
                f"""
                SELECT s.market_id,
                       MIN(s.ts) AS min_ts,
                       MAX(s.ts) AS max_ts,
                       m.resolution_ts
                FROM market_snapshots s
                JOIN markets m ON m.market_id = s.market_id
                WHERE s.market_id IN ({placeholders})
                GROUP BY s.market_id
                """,
                tuple(market_ids),
            ).fetchall()

        global_min: str | None = None
        global_max: str | None = None
        for r in rows:
            mid = str(r["market_id"])
            min_ts_str = str(r["min_ts"])
            max_ts_str = str(r["max_ts"])
            res_ts_str = str(r["resolution_ts"]) if r["resolution_ts"] is not None else None
            self._market_ranges[mid] = (min_ts_str, max_ts_str, res_ts_str)
            if global_min is None or min_ts_str < global_min:
                global_min = min_ts_str
            if global_max is None or max_ts_str > global_max:
                global_max = max_ts_str

        if global_min is not None and global_max is not None:
            self._global_min_ts = ensure_utc(datetime.fromisoformat(global_min))
            self._global_max_ts = ensure_utc(datetime.fromisoformat(global_max))


def _safe_value(value: float | None, scale: float = 1.0) -> float:
    """Safely convert a value to float, returning 0.0 for None/NaN/Inf."""
    if value is None or not math.isfinite(value):
        return 0.0
    return float(value) / scale
