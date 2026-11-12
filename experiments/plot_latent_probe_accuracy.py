"""Reproduce the three-panel latent probe accuracy plot from saved probe bundles.

Run from repository root:
    python -m experiments.plot_latent_probe_accuracy --depth 6 \
        --data-dir figure_data/latents --output figures/latent_probe_depth6

Only load research-owned .pkl files (pickle is not safe for untrusted input).
"""
import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from .common import save_figure
except ImportError:
    from experiments.common import save_figure

FILENAMES = {
    3: "second_third_last_probe_checkpoint_results_depth3.pkl",
    6: "second_third_last_probe_checkpoint_results.pkl",
    12: "second_third_last_probe_checkpoint_results_depth12.pkl",
}
STEPS = {
    3: (512, 16384, 262144, 5931648),
    6: (512, 16384, 262144, 1048576),
    12: (512, 16384, 262144, 2097152),
}
PANELS = (
    ("second_last_level1", r"(a) $\mu^{(2)}_8$"),
    ("third_last_level1", r"(b) $\mu^{(2)}_7$"),
    ("third_last_level2", r"(c) $\mu^{(1)}_1$"),
)
STAGE_LABELS = ("Stage 1a", "Stage 1b", "Stage 2", "Stage 3")
STAGE_COLORS = ("tab:purple", "tab:blue", "tab:orange", "tab:green")


def load_bundle(path):
    if not path.is_file():
        raise FileNotFoundError(f"Probe results bundle not found: {path}")
    with path.open("rb") as f:
        data = pickle.load(f)
    for key in ("results", "layer_labels", "chance_accuracy", "architecture"):
        if key not in data:
            raise ValueError(f"{path} missing required key {key!r}")
    return data


def add_stage2_override(data, path):
    """Follow original D=6 plotting notebook's corrected step-16384 override."""
    with path.open("rb") as f:
        override = pickle.load(f)
    for key in ("grammar", "dataset", "architecture", "probe_C", "layer_labels", "chance_accuracy"):
        if override.get(key) != data.get(key):
            raise ValueError(f"Stage-2 override differs in {key!r}: {path}")
    for name, metadata in override.get("probe_metadata", {}).items():
        if metadata != data.get("probe_metadata", {}).get(name):
            raise ValueError(f"Stage-2 override differs in probe {name!r}: {path}")
    if 16384 not in override["results"]:
        raise ValueError(f"Missing step 16384 in {path}")
    data["results"] = dict(data["results"])
    data["results"][16384] = override["results"][16384]


def plot(data, depth, output):
    arch = data["architecture"]
    if int(arch["num_layers"]) != depth:
        raise ValueError(f"Requested depth {depth}, but results have depth {arch['num_layers']}")
    if depth in (3, 12):
        expected_width = {3: 768, 12: 256}[depth]
        if int(arch["embedding_dim"]) != expected_width:
            raise ValueError(f"Expected D={depth}, width={expected_width}")
    labels = data["layer_labels"]
    if len(labels) != depth + 1:
        raise ValueError(f"Expected embedding plus {depth} layer labels")
    stages = STEPS[depth]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True, sharey=True)
    x = np.arange(depth + 1)
    for ax, (probe, title) in zip(axes, PANELS):
        for step, color, stage_label in zip(stages, STAGE_COLORS, STAGE_LABELS):
            if step not in data["results"]:
                raise ValueError(f"Missing checkpoint step {step} in bundle (depth={depth})")
            try:
                acc = np.asarray(data["results"][step]["probes"][probe]["test"], dtype=float)
            except KeyError as exc:
                raise ValueError(f"Missing probe {probe} at step {step}") from exc
            if acc.shape != (depth + 1,) or not np.all(np.isfinite(acc)):
                raise ValueError(f"Invalid accuracy curve: step={step}, probe={probe}")
            ax.plot(x, acc, "o-", color=color, linewidth=2, markersize=5, label=stage_label)
        ax.axhline(float(data["chance_accuracy"]), color="gray", ls=":", lw=1.3)
        ax.set_xticks(x, labels, rotation=45 if depth == 3 else 0)
        ax.set_xlabel("Probed layer", fontsize=15)
        ax.set_ylim(0, 1.02)
        ax.set_title(title, fontsize=20)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Held-out probe accuracy", fontsize=15)
    axes[0].legend(loc="lower right", frameon=False)
    fig.tight_layout(w_pad=1.6)
    save_figure(fig, output)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, choices=(3, 6, 12), default=6)
    parser.add_argument("--data-dir", type=Path, default=Path("figure_data/latents"))
    parser.add_argument("--bundle", type=Path, help="Override bundle path")
    parser.add_argument("--stage2-override", type=Path,
                        help="For D=6, use original corrected step-16384 .pkl")
    parser.add_argument("--output", type=Path, help="Output filename stem (both PDF and PNG)")
    args = parser.parse_args()
    bundle = args.bundle or args.data_dir / FILENAMES[args.depth]
    data = load_bundle(bundle)
    if args.stage2_override:
        add_stage2_override(data, args.stage2_override)
    elif args.depth == 6:
        override_path = args.data_dir / "second_third_last_probe_step_16384.pkl"
        if override_path.is_file():
            add_stage2_override(data, override_path)
            print(f"Applied corrected Stage 1b data from {override_path}")
        else:
            print("Note: corrected D=6 step-16384 file not found; using step in main bundle, if present")
    output = args.output or Path(f"figures/latent_probe_depth{args.depth}")
    plot(data, args.depth, output)


if __name__ == "__main__":
    main()
