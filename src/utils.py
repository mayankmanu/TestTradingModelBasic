import os
import json
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, Tuple

import numpy as np
import torch


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def device_auto() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Checkpoint:
    model_state: Dict[str, Any]
    optimizer_state: Dict[str, Any]
    trainer_state: Dict[str, Any]
    loader_state: Dict[str, Any]


def save_checkpoint(path: str, ckpt: Checkpoint) -> None:
    ensure_dir(os.path.dirname(path))
    # Use torch for tensors and json for the rest
    torch.save(
        {
            "model_state": ckpt.model_state,
            "optimizer_state": ckpt.optimizer_state,
            "trainer_state": ckpt.trainer_state,
            "loader_state": ckpt.loader_state,
        },
        path,
    )


def load_checkpoint(path: str) -> Tuple[Checkpoint, bool]:
    if not os.path.exists(path):
        return Checkpoint({}, {}, {}, {}), False
    blob = torch.load(path, map_location="cpu")
    ckpt = Checkpoint(
        model_state=blob.get("model_state", {}),
        optimizer_state=blob.get("optimizer_state", {}),
        trainer_state=blob.get("trainer_state", {}),
        loader_state=blob.get("loader_state", {}),
    )
    return ckpt, True


class SimpleLogger:
    def __init__(self):
        pass

    def info(self, msg: str) -> None:
        print(msg, flush=True)

    def trade(self, msg: str) -> None:
        print(f"[TRADE] {msg}", flush=True)


def moving_std(x: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.std(x) + eps)


def normalized_features(price_row: Dict[str, float], ref_price: float) -> np.ndarray:
    # Create simple relative features vs close price
    c = price_row["close"]
    denom = max(ref_price, 1e-6)
    return np.array([
        (price_row["open"] - c) / denom,
        (price_row["high"] - c) / denom,
        (price_row["low"] - c) / denom,
        0.0 if c == 0 else (c - ref_price) / denom,
        price_row.get("volume", 0.0) / 1e6,
    ], dtype=np.float32)
