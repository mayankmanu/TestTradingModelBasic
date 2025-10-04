import argparse
from dataclasses import asdict

from .trainer import TrainConfig, Trainer
from .utils import SimpleLogger


def parse_args():
    p = argparse.ArgumentParser(description="RL Trading Trainer")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_shared(sp):
        sp.add_argument("--data_dir", type=str, required=True, help="Directory with CSV files")
        sp.add_argument("--save_dir", type=str, default="checkpoints")
        sp.add_argument("--chunk_size", type=int, default=10000)
        sp.add_argument("--initial_capital", type=float, default=100000.0)
        sp.add_argument("--transaction_cost", type=float, default=0.0005)
        sp.add_argument("--stop_loss", type=float, default=0.02)
        sp.add_argument("--lr", type=float, default=3e-4)
        sp.add_argument("--gamma", type=float, default=0.99)
        sp.add_argument("--entropy_coef", type=float, default=0.01)
        sp.add_argument("--value_coef", type=float, default=0.5)
        sp.add_argument("--epochs", type=int, default=1, help="Epochs per iteration")
        sp.add_argument("--iterations", type=int, default=1, help="Number of AlphaZero-like training cycles")
        sp.add_argument("--eval_every", type=int, default=0, help="Evaluate every N iterations (0=never)")
        sp.add_argument("--reset_loader_state", action="store_true", help="Keep model weights from checkpoint but restart data stream from beginning")
        sp.add_argument("--flat_at_end", action="store_true", help="During eval, force-close any open position at the final bar to realize PnL")
        sp.add_argument("--train_long_only", action="store_true", help="During training, disallow shorts and only allow HOLD/BUY/SELL-to-close")
        # Exploration schedules and eval sampling
        sp.add_argument("--train_temperature_start", type=float, default=1.0, help="Initial temperature for action sampling during training")
        sp.add_argument("--train_temperature_end", type=float, default=1.0, help="Final temperature for training")
        sp.add_argument("--train_temperature_decay_iters", type=int, default=1, help="Iterations over which to anneal training temperature")
        sp.add_argument("--eval_sample", action="store_true", help="During eval, sample from policy instead of greedy argmax")
        sp.add_argument("--eval_temperature", type=float, default=1.0, help="Temperature used for eval sampling")

    sp_train = sub.add_parser("train")
    add_shared(sp_train)

    sp_eval = sub.add_parser("eval")
    add_shared(sp_eval)

    return p.parse_args()


def main():
    args = parse_args()
    logger = SimpleLogger()

    cfg = TrainConfig(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        chunk_size=args.chunk_size,
        initial_capital=args.initial_capital,
        transaction_cost=args.transaction_cost,
        stop_loss=args.stop_loss,
        lr=args.lr,
        gamma=args.gamma,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        epochs=args.epochs,
        iterations=args.iterations,
        eval_every=args.eval_every,
        reset_loader_state=args.reset_loader_state,
        flat_at_end=args.flat_at_end,
        train_long_only=args.train_long_only,
        train_temperature_start=args.train_temperature_start,
        train_temperature_end=args.train_temperature_end,
        train_temperature_decay_iters=args.train_temperature_decay_iters,
        eval_sample=args.eval_sample,
        eval_temperature=args.eval_temperature,
    )

    trainer = Trainer(cfg, logger=logger)

    if args.cmd == "train":
        logger.info(f"Starting training with config: {asdict(cfg)}")
        trainer.train()
    elif args.cmd == "eval":
        logger.info(f"Starting evaluation with config: {asdict(cfg)}")
        trainer.evaluate()


if __name__ == "__main__":
    main()
