"""Run with ``python -m rhm_inference`` from the repository root."""
import argparse
import csv
import json
from pathlib import Path
from .bayes import GrammarConfig, KNOWN_LAYERS, estimate_bayes_loss


def main():
    parser = argparse.ArgumentParser(description="Estimate RHM Bayes last-token loss")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--v", type=int, default=8)
    parser.add_argument("--L", type=int, default=3)
    parser.add_argument("--m2", type=int, default=2)
    parser.add_argument("--m3", type=int, default=16)
    parser.add_argument("--eta", type=float, default=None, help="default: v")
    parser.add_argument("--seed-rules", type=int, default=12345678)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--known-layers", choices=KNOWN_LAYERS, default="all")
    parser.add_argument("--fixed-length", action="store_true",
                        help="condition on the observed sentence length")
    parser.add_argument("--out-dir", default="runs/bayes_losses")
    args = parser.parse_args()
    config = GrammarConfig(v=args.v, L=args.L, m2=args.m2, m3=args.m3,
                           eta=args.v if args.eta is None else args.eta,
                           seed_rules=args.seed_rules)
    summary, rows = estimate_bayes_loss(
        config, n_samples=args.samples, known_layers=args.known_layers,
        sample_seed=args.sample_seed, unknown_length=not args.fixed_length)
    out_dir = Path(args.out_dir) / args.known_layers
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "samples.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("entropy", "length", "support", "optimal_accuracy"))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {"parameters": vars(args), "summary": summary}
    (out_dir / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
