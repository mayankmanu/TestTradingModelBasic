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
        sp.add_argument("--epochs", type=int, default=1)
        sp.add_argument("--reset_loader_state", action="store_true", help="Keep model weights from checkpoint but restart data stream from beginning")
        sp.add_argument("--flat_at_end", action="store_true", help="During eval, force-close any open position at the final bar to realize PnL")
        sp.add_argument("--train_long_only", action="store_true", help="During training, disallow shorts and only allow HOLD/BUY/SELL-to-close")

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
        reset_loader_state=args.reset_loader_state,
        flat_at_end=args.flat_at_end,
        train_long_only=args.train_long_only,
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
