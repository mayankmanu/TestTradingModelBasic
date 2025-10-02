# RL Trading Model (AlphaZero-like, Actor-Critic with LSTM)

This is a minimal algo-trading RL project that learns from your real 1-minute OHLCV data.

- Actions: `buy`, `sell`, `hold`
- Configurable: `initial_capital`, `stop_loss`, `transaction_cost`
- Processes data in chunks and saves training state to resume later
- Policy-Value LSTM model (actor-critic), optimized to maximize profit
- Simple CLI to train/evaluate

## Project Layout

- `src/app.py` — CLI entry (train/eval)
- `src/data_loader.py` — Chunked CSV loader for OHLCV data
- `src/trading_env.py` — Trading environment with PnL, logs, stop-loss, fees
- `src/model.py` — LSTM Policy-Value network (actor-critic)
- `src/trainer.py` — Training loop with checkpoint save/load
- `src/utils.py` — Helpers (config, seeds, logging)
- `requirements.txt`

## Data Format

Place CSVs under `data/` directory (or pass `--data_dir PATH`). Each CSV should have columns:

```
date,Open,high,low,close,volume
```

- `date` can be any parseable string; rows will be sorted by date per file.
- File name (without extension) will be used as `symbol`.

## Quick Start

1) Install dependencies

```
pip install -r requirements.txt
```

2) Train on your data directory

```
python -m src.app train --data_dir data --save_dir checkpoints \
  --chunk_size 10000 --initial_capital 100000 --transaction_cost 0.0005 \
  --stop_loss 0.02 --epochs 2
```

3) Resume training (automatically loads latest checkpoint if exists)

```
python -m src.app train --data_dir data --save_dir checkpoints
```

4) Evaluate (no learning, logs PnL)

```
python -m src.app eval --data_dir data --save_dir checkpoints --chunk_size 10000
```

## Notes

- Uses one shared agent across all symbols, trained sequentially over chunks.
- Position is one unit long, flat, or one unit short. Stop-loss is applied to open position based on price move from entry.
- Transaction cost is applied on each position change (enter/exit/flip).
- Training uses on-policy advantage (A2C-style) with entropy regularization.

## Disclaimer

This is for educational purposes only. No financial advice. Use at your own risk.
