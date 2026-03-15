from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .trading_env import Action, TradingEnvironment


class PolymarketGymEnv(gym.Env[np.ndarray, int]):
    """Gymnasium wrapper around ``TradingEnvironment``."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        db_path: str | Path,
        starting_cash: float = 1_000.0,
        market_ids: list[str] | None = None,
        split: str = "val",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.env = TradingEnvironment(
            db_path=db_path,
            starting_cash=starting_cash,
            market_ids=market_ids,
            split=split,
            **kwargs,
        )
        initial_state = self.env.reset(market_id=market_ids[0] if market_ids else None)
        shape = initial_state.to_array().shape
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=shape,
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(12)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.env._core._rng.seed(seed)
        market_id = options.get("market_id") if options is not None else None
        state = self.env.reset(market_id=market_id)
        return state.to_array(), {"state": state}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        result = self.env.step(self._map_action(int(action)))
        info = dict(result.info)
        info["state"] = result.new_state
        return result.new_state.to_array(), result.reward, result.done, False, info

    def render(self) -> str:
        state = self.env.get_state()
        summary = (
            f"{state.market_id} ts={iso_ts(state.timestamp)} mid={state.mid:.4f} "
            f"cash={state.cash:.2f} equity={self.env.portfolio_value:.2f} "
            f"positions={state.n_open_positions}"
        )
        return summary

    def close(self) -> None:
        self.env.conn.close()

    def _map_action(self, action: int) -> Action:
        if action == 0:
            return Action.hold()
        if action == 1:
            return Action.buy_yes()
        if action == 2:
            return Action.buy_no()
        if action == 3:
            return Action.sell_yes()
        if action == 4:
            return Action.sell_no()
        if action == 5:
            return Action.buy_yes_limit()
        if action == 6:
            return Action.buy_no_limit()
        if action == 7:
            return Action.sell_yes_limit()
        if action == 8:
            return Action.sell_no_limit()
        if action == 9:
            return Action.mint_pair()
        if action == 10:
            return Action.redeem_pair()
        if action == 11:
            return Action.cancel_orders()
        raise ValueError(f"Unsupported discrete action {action}")


def iso_ts(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat(timespec="seconds")
    return str(value)
