import os
import csv
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

from .data_loader import DataStream
from .model import PolicyValueLSTM
from .trading_env import TradeEnv, TradeEnvConfig
from .utils import (
    Checkpoint,
    SimpleLogger,
    device_auto,
    load_checkpoint,
    save_checkpoint,
)


@dataclass
class TrainConfig:
    data_dir: str
    save_dir: str = "checkpoints"
    chunk_size: int = 10000
    initial_capital: float = 100_000.0
    transaction_cost: float = 0.0005
    stop_loss: float = 0.02
    lr: float = 3e-4
    gamma: float = 0.99
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    epochs: int = 1
    seed: int = 42
    reset_loader_state: bool = False
    flat_at_end: bool = False
    train_long_only: bool = False


class Trainer:
    def __init__(self, cfg: TrainConfig, logger: SimpleLogger | None = None):
        self.cfg = cfg
        self.logger = logger or SimpleLogger()
        self.device = device_auto()
        self.model = PolicyValueLSTM(input_dim=5, hidden_dim=64).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.global_step = 0

    def _checkpoint_path(self) -> str:
        return f"{self.cfg.save_dir}/ckpt.pt"

    def _save(self, loader: DataStream):
        ckpt = Checkpoint(
            model_state=self.model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            trainer_state={"global_step": self.global_step},
            loader_state=loader.state_dict(),
        )
        save_checkpoint(self._checkpoint_path(), ckpt)

    def _load(self, loader: DataStream) -> bool:
        ckpt, ok = load_checkpoint(self._checkpoint_path())
        if not ok:
            return False
        self.model.load_state_dict(ckpt.model_state)
        self.optimizer.load_state_dict(ckpt.optimizer_state)
        self.global_step = int(ckpt.trainer_state.get("global_step", 0))
        if self.cfg.reset_loader_state:
            # Keep model/optimizer but restart the data stream from the beginning
            loader.reset()
            self.logger.info(
                f"Loaded checkpoint weights at step {self.global_step}; reset loader state as requested."
            )
        else:
            loader.load_state_dict(ckpt.loader_state)
            self.logger.info(
                f"Loaded checkpoint at step {self.global_step} from {self._checkpoint_path()}"
            )
        return True

    def _compute_returns_advantages(
        self, rewards: List[float], values: torch.Tensor, gamma: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        R = 0.0
        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.append(R)
        returns.reverse()
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages = returns_t - values.detach()
        return returns_t, advantages

    def train(self):
        # Data stream and resume state
        stream = DataStream(self.cfg.data_dir, chunk_size=self.cfg.chunk_size)
        self._load(stream)  # ok if not found

        env_cfg = TradeEnvConfig(
            initial_capital=self.cfg.initial_capital,
            transaction_cost=self.cfg.transaction_cost,
            stop_loss=self.cfg.stop_loss,
            long_only=self.cfg.train_long_only,
        )

        for epoch in range(self.cfg.epochs):
            self.logger.info(f"Epoch {epoch+1}/{self.cfg.epochs}")

            for symbol, df in stream.iter_chunks():
                env = TradeEnv(env_cfg, logger=self.logger)
                hidden = self.model.init_hidden(batch_size=1, device=self.device)

                log_probs: List[torch.Tensor] = []
                values: List[torch.Tensor] = []
                entropies: List[torch.Tensor] = []
                rewards: List[float] = []

                # Initialize observation from first row
                if len(df) == 0:
                    continue

                for i in range(len(df)):
                    row = df.iloc[i].to_dict()
                    # Build observation vector (5-dim)
                    obs_np = np.array(
                        [
                            (row["open"] - row["close"]) / max(row["close"], 1e-6),
                            (row["high"] - row["close"]) / max(row["close"], 1e-6),
                            (row["low"] - row["close"]) / max(row["close"], 1e-6),
                            0.0,
                            row.get("volume", 0.0) / 1e6,
                        ],
                        dtype=np.float32,
                    )
                    obs = torch.from_numpy(obs_np).to(self.device).unsqueeze(0)  # [1,F]

                    logits, value, hidden = self.model(obs, hidden)
                    probs = F.softmax(logits, dim=-1)
                    dist = Categorical(probs=probs)
                    action = dist.sample()  # [1]
                    log_prob = dist.log_prob(action).squeeze(0)
                    entropy = dist.entropy().mean()

                    # Execute environment step
                    obs_next, reward, done, info = env.step(row, int(action.item()))

                    log_probs.append(log_prob)
                    values.append(value.squeeze(0))
                    entropies.append(entropy)
                    rewards.append(float(reward))
                    self.global_step += 1

                if not rewards:
                    continue

                # Stack tensors
                values_t = torch.stack(values)  # [T]
                returns_t, advantages = self._compute_returns_advantages(
                    rewards, values_t, self.cfg.gamma
                )
                log_probs_t = torch.stack(log_probs)
                entropies_t = torch.stack(entropies)

                policy_loss = -(log_probs_t * advantages).mean()
                value_loss = F.mse_loss(values_t, returns_t)
                entropy_loss = -entropies_t.mean()
                loss = (
                    policy_loss
                    + self.cfg.value_coef * value_loss
                    + self.cfg.entropy_coef * (-entropy_loss)
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                self.logger.info(
                    f"[{symbol}] step={self.global_step} chunk_T={len(rewards)} equity={info['equity']:.2f} trades={info['trades']} loss={loss.item():.6f}"
                )

                # Save after each chunk
                self._save(stream)

        self.logger.info("Training finished.")

    def evaluate(self):
        # Load state
        stream = DataStream(self.cfg.data_dir, chunk_size=self.cfg.chunk_size)
        loaded = self._load(stream)
        if not loaded:
            self.logger.info("No checkpoint found. Evaluating with fresh model.")

        env_cfg = TradeEnvConfig(
            initial_capital=self.cfg.initial_capital,
            transaction_cost=self.cfg.transaction_cost,
            stop_loss=self.cfg.stop_loss,
            long_only=True,
        )

        self.model.eval()
        with torch.no_grad():
            total_equity_pnl = 0.0
            total_realized_net = 0.0
            out_dir = os.path.join(self.cfg.save_dir, "eval_trades")
            os.makedirs(out_dir, exist_ok=True)

            for symbol, df in stream.iter_chunks():
                env = TradeEnv(env_cfg, logger=self.logger)
                hidden = self.model.init_hidden(batch_size=1, device=self.device)

                for i in range(len(df)):
                    row = df.iloc[i].to_dict()
                    obs_np = np.array(
                        [
                            (row["open"] - row["close"]) / max(row["close"], 1e-6),
                            (row["high"] - row["close"]) / max(row["close"], 1e-6),
                            (row["low"] - row["close"]) / max(row["close"], 1e-6),
                            0.0,
                            row.get("volume", 0.0) / 1e6,
                        ],
                        dtype=np.float32,
                    )
                    obs = torch.from_numpy(obs_np).to(self.device).unsqueeze(0)

                    logits, value, hidden = self.model(obs, hidden)
                    action = torch.argmax(logits, dim=-1).item()
                    obs_next, reward, done, info = env.step(row, int(action))

                # Optionally flatten at the end to realize PnL
                if self.cfg.flat_at_end and env.position != 0 and len(df) > 0:
                    last_row = df.iloc[-1].to_dict()
                    # In long_only, only need to close long with SELL (2)
                    forced_action = 2 if env.position == 1 else 0
                    if forced_action != 0:
                        obs_next, reward, done, info = env.step(last_row, forced_action)

                # Write per-symbol ledger
                ledger_path = os.path.join(out_dir, f"{symbol}.csv")
                with open(ledger_path, "w", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "date","action","from_pos","to_pos","price","fee",
                            "realized_gross","realized_net","cash","equity",
                        ],
                    )
                    writer.writeheader()
                    for rec in env.ledger:
                        writer.writerow(rec)

                # Summaries
                symbol_realized = sum(r.get("realized_net", 0.0) for r in env.ledger)
                symbol_equity_pnl = env.equity - self.cfg.initial_capital
                total_realized_net += symbol_realized
                total_equity_pnl += symbol_equity_pnl
                self.logger.info(
                    f"[EVAL {symbol}] equity={env.equity:.2f} pnl={symbol_equity_pnl:.2f} realized_net={symbol_realized:.2f} trades={env.trades} costs={env.total_costs:.4f} -> ledger: {ledger_path}"
                )

            self.logger.info(
                f"[EVAL TOTAL] equity_pnl={total_equity_pnl:.2f} realized_net_total={total_realized_net:.2f}"
            )
