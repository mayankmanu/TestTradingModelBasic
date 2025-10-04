import os
import json
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
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


# ---- Technical indicators and feature building ----

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic technical indicators in-place and return the dataframe.
    Assumes columns: date, open, high, low, close, volume
    """
    df = df.copy()
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Returns
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_20"] = df["close"].pct_change(20)

    # Moving averages and crossover
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma5_rel"] = (df["ma5"] / df["close"]) - 1.0
    df["ma20_rel"] = (df["ma20"] / df["close"]) - 1.0
    df["ma_cross"] = (df["ma5"] > df["ma20"]).astype(float)

    # Volatility (rolling std of returns)
    df["vol_std10"] = df["ret_1"].rolling(10).std()

    # ATR(14)
    prev_close = df["close"].shift(1)
    tr1 = (df["high"] - df["low"]).abs()
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    df["atr14_rel"] = df["atr14"] / (df["close"].replace(0, np.nan))

    # RSI(14) simplified
    delta = df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    df["rsi14"] = rsi
    df["rsi14_norm"] = df["rsi14"] / 100.0  # 0..1

    # Volume ratio vs 20-period average
    vol_ma20 = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / (vol_ma20.replace(0, np.nan))

    return df


def feature_vector_from_row(row: Dict[str, float]) -> np.ndarray:
    """
    Build a 10-dim feature vector from a row that already contains the following fields:
    ret_1, ret_5, ret_20, ma5_rel, ma20_rel, ma_cross, vol_std10, atr14_rel, rsi14_norm, vol_ratio
    Returns np.nan for missing entries; caller should skip rows with NaNs.
    """
    feats = [
        row.get("ret_1", np.nan),
        row.get("ret_5", np.nan),
        row.get("ret_20", np.nan),
        row.get("ma5_rel", np.nan),
        row.get("ma20_rel", np.nan),
        row.get("ma_cross", np.nan),
        row.get("vol_std10", np.nan),
        row.get("atr14_rel", np.nan),
        row.get("rsi14_norm", np.nan),
        row.get("vol_ratio", np.nan),
    ]
    return np.array(feats, dtype=np.float32)
