"""Run: python -m rhm_training --preset smoke [--dry-run]."""
import argparse
from dataclasses import replace
import shlex
from .runner import TrainingConfig, run_training


def main():
    parser = argparse.ArgumentParser(description="Run existing Transformer training with reproducible presets")
    parser.add_argument("--preset", choices=("smoke", "paper"), default="smoke")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without training")
    for arg in ("v", "L", "m2", "m3", "train_size", "val_size", "ternary_val_size",
                "batch_size", "depth", "d_embedding", "n_heads", "ffwd_size", "epochs",
                "seed_rules", "seed_sample_train", "seed_sample_test", "seed_model"):
        parser.add_argument("--" + arg.replace("_", "-"), type=int, dest=arg)
    for arg in ("eta", "lr", "weight_decay"):
        parser.add_argument("--" + arg.replace("_", "-"), type=float, dest=arg)
    for arg in ("device", "objective", "output"):
        parser.add_argument("--" + arg, dest=arg)
    parser.add_argument("--checkpoints", action="store_true")
    parser.add_argument("--fresh-online-data", action="store_true")
    parser.add_argument("--rope", action="store_true")
    args = vars(parser.parse_args())
    config = TrainingConfig.preset(args.pop("preset"))
    dry_run = args.pop("dry_run")
    modifications = {k: v for k, v in args.items() if v is not None and v is not False}
    config = replace(config, **modifications)
    command = run_training(config, dry_run=dry_run)
    if dry_run:
        print(shlex.join(command))
    else:
        print(f"Training result prefix: {config.output}")


if __name__ == "__main__":
    main()
