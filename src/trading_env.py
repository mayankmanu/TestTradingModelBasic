from dataclasses import dataclass
from typing import Dict, Tuple, List, Any

import numpy as np

from .utils import SimpleLogger, normalized_features


@dataclass
class TradeEnvConfig:
    initial_capital: float = 100_000.0
    transaction_cost: float = 0.0005  # proportion per trade
    stop_loss: float = 0.02  # 2% from entry triggers stop-out
    long_only: bool = False  # if True, disallow shorts; sell means close long


class TradeEnv:
    """
    Minimal trading environment stepped by rows of OHLCV.
    Position: -1 (short), 0 (flat), +1 (long). Size is 1 unit for simplicity.
    Equity = cash + position * price
    Reward: delta equity per step (can scale later).
    """

    def __init__(self, config: TradeEnvConfig, logger: SimpleLogger | None = None):
        self.cfg = config
        self.logger = logger or SimpleLogger()
        self.reset()

    def reset(self) -> Dict:
        self.cash = float(self.cfg.initial_capital)
        self.position = 0  # -1,0,+1
        self.entry_price = 0.0
        self.last_price = 0.0
        self.equity = self.cash
        self.max_equity = self.equity
        self.min_equity = self.equity
        self.total_costs = 0.0
        self.trades = 0
        # Trade ledger for evaluation/reporting
        self.ledger: List[Dict[str, Any]] = []
        return {
            "cash": self.cash,
            "position": self.position,
            "equity": self.equity,
        }

    def _apply_cost(self, price: float, delta_pos: int) -> float:
        # cost proportional to notional traded; here 1 unit per position change
        notional = abs(delta_pos) * price
        fee = self.cfg.transaction_cost * notional
        self.cash -= fee
        self.total_costs += fee
        return fee

    def _mark_to_market(self, price: float) -> None:
        self.equity = self.cash + self.position * price
        self.max_equity = max(self.max_equity, self.equity)
        self.min_equity = min(self.min_equity, self.equity)

    def _stop_loss_triggered(self, price: float) -> bool:
        if self.position == 0 or self.cfg.stop_loss <= 0:
            return False
        if self.entry_price <= 0:
            return False
        move = (price - self.entry_price) / max(self.entry_price, 1e-6)
        if self.position == 1:
            hit = move <= -self.cfg.stop_loss
        else:
            # If long_only, we should never be short. For completeness, handle short stop.
            hit = move >= self.cfg.stop_loss
        return hit

    def step(self, row: Dict[str, float], action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        action: 0=hold, 1=buy(long), 2=sell(short)
        Returns: obs, reward, done, info
        """
        price = float(row["close"]) if row.get("close") is not None else float(row["open"])  # fallback
        dt = row.get("date")
        prev_equity = self.equity
        done = False
        realized_pnl_gross = 0.0
        realized_pnl_net = 0.0

        # Handle stop-loss before action effect to force exit
        if self._stop_loss_triggered(price):
            if self.position != 0:
                # Compute realized PnL for the closing leg
                realized_pnl_gross = (price - self.entry_price) * self.position
                self.logger.trade(
                    f"STOP-LOSS: close pos {self.position} @ {price:.4f} (entry {self.entry_price:.4f}) realized_gross={realized_pnl_gross:.4f} on {dt}"
                )
                # Close to flat: delta_pos to 0
                delta = -self.position
                self.cash += (-delta) * price  # if position was +1, delta -1 -> add +1*price to cash
                fee_sl = self._apply_cost(price, delta)
                realized_pnl_net = realized_pnl_gross - fee_sl
                self.logger.trade(
                    f"STOP-LOSS: fee={fee_sl:.4f} realized_net={realized_pnl_net:.4f}"
                )
                self.ledger.append({
                    "date": dt,
                    "action": "STOP_LOSS_CLOSE",
                    "from_pos": self.position,
                    "to_pos": 0,
                    "price": price,
                    "fee": fee_sl,
                    "realized_gross": realized_pnl_gross,
                    "realized_net": realized_pnl_net,
                    "cash": self.cash,
                    "equity": self.equity,
                })
                self.position = 0
                self.entry_price = 0.0

        # Translate action to desired target position
        if self.cfg.long_only:
            # 0: hold, 1: open/keep long, 2: close long (no shorts)
            if action == 1:
                target_pos = 1
            elif action == 2:
                target_pos = 0 if self.position == 1 else 0
            else:
                target_pos = self.position
        else:
            target_pos = {0: self.position, 1: 1, 2: -1}.get(action, self.position)

        # Execute trade if change
        delta_pos = target_pos - self.position
        if delta_pos != 0:
            # Cash impact of changing position by delta_pos units at current price
            # Increasing position (delta_pos>0) spends cash; decreasing adds cash
            self.cash -= delta_pos * price
            prev_pos = self.position
            # If we are closing or flipping, compute realized pnl on the closed leg
            if prev_pos != 0 and (target_pos == 0 or (prev_pos * target_pos) <= 0):
                # Closed leg is the entire previous unit
                realized_pnl_gross += (price - self.entry_price) * prev_pos
            fee = self._apply_cost(price, delta_pos)
            self.position = target_pos
            if self.position != 0:
                # If we flipped or opened, set/adjust entry price to current price
                self.entry_price = price
            else:
                self.entry_price = 0.0
            self.trades += 1
            realized_pnl_net = realized_pnl_gross - fee
            self.logger.trade(
                f"ACTION: {['HOLD','BUY','SELL'][action]} from pos {prev_pos} to {self.position} @ {price:.4f}, fee={fee:.4f}, realized_gross={realized_pnl_gross:.4f}, realized_net={realized_pnl_net:.4f} on {dt}"
            )
            self.ledger.append({
                "date": dt,
                "action": ["HOLD","BUY","SELL"][action],
                "from_pos": prev_pos,
                "to_pos": self.position,
                "price": price,
                "fee": fee,
                "realized_gross": realized_pnl_gross,
                "realized_net": realized_pnl_net,
                "cash": self.cash,
                "equity": self.equity,
            })

        # Mark-to-market and compute reward
        self._mark_to_market(price)
        reward = self.equity - prev_equity
        self.last_price = price

        obs = normalized_features(
            {
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0.0)),
            },
            ref_price=price,
        )

        info = {
            "price": price,
            "cash": self.cash,
            "equity": self.equity,
            "position": self.position,
            "trades": self.trades,
            "costs": self.total_costs,
            "reward": reward,
            "realized_pnl_gross": realized_pnl_gross,
            "realized_pnl_net": realized_pnl_net,
            "date": dt,
        }
        return obs, float(reward), done, info
