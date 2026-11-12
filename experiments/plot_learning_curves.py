"""Original learning-curve figure extracted from Learning_curves_plot.ipynb."""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from experiments.common import save_figure

parser = argparse.ArgumentParser(description="Learning curves from original train-result .pt files")
parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing training results .pt")
parser.add_argument("--output", type=Path, default=Path("figures/learning_curves_fixed_vs_varying_tree"))
args = parser.parse_args()
runs_dir = args.data_dir

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 12, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

configs = [
    (6, 512, 0.01, False, "Varying-tree training", "tab:red"),
    (3, 768, 0.03, True, "Fixed-tree training", "tab:blue"),
]
results = {}
for depth, embedding, weight_decay, fixed, label, color in configs:
    path = runs_dir / (
        f"run_L3_v8_bin2_ter16_P16777216_lr0.001_l2{weight_decay}_bs128"
        f"_depth{depth}_demb_{embedding}_nheads_16_arr0.pt"
    )
    saved = torch.load(path, map_location="cpu", weights_only=False)
    config = vars(saved["config"])
    if config["depth"] != depth or config["ternary_train"] != fixed:
        raise ValueError(f"Unexpected training configuration: {path}")
    # Fixed-tree training and validation are both ternary, so val_loss is
    # the ternary loss. Mixed training records a separate ternary_val_loss.
    loss_key = "val_loss" if fixed else "ternary_val_loss"
    entries = [e for e in saved["dynamics"] if loss_key in e]
    steps = np.asarray([e["t"] for e in entries], dtype=float)
    losses = np.asarray([e[loss_key] for e in entries], dtype=float) / np.log(config["v"])
    valid = (steps > 0) & np.isfinite(steps) & np.isfinite(losses) & (losses > 0)
    steps, losses = steps[valid], losses[valid]
    order = np.argsort(steps)
    if not len(steps):
        raise ValueError(f"No valid learning-curve entries: {path}")
    results[depth] = {"steps": steps[order], "loss": losses[order]}

L_2_VT = 0.20574168745407478
L_2_FT = 0.1935

plateaus = [(0.59, r'$\mathcal{L}_{\mathrm{trig}}^{\mathrm{VT}}$'),
            (0.452838292701, r'$\mathcal{L}_{\mathrm{trig}}^{\mathrm{FT}}$'),
            (L_2_VT, r'$\mathcal{L}_2$'),
            (0.1159764843808, r'$\mathcal{L}_3$')]

# Double-column width, matching the companion paper figure.
fig, ax = plt.subplots(figsize=(7.2, 3.65))
for depth, embedding, weight_decay, fixed, label, color in configs:
    curve = results[depth]
    ax.plot(curve["steps"], curve["loss"], color=color, lw=2.5,
            label=label, zorder=4)

checkpoint_steps = np.array([1024, 16384, 262144, 2097152])
varying = results[6]
checkpoint_losses = []
for step in checkpoint_steps:
    matches = np.flatnonzero(varying["steps"] == step)
    if len(matches) != 1:
        raise ValueError(f"Expected one recorded varying-tree loss at t={step}")
    checkpoint_losses.append(varying["loss"][matches[0]])
ax.scatter(checkpoint_steps, checkpoint_losses, marker="*", s=150,
           color="tab:red", edgecolors="black", linewidths=0.65,
           zorder=6, label="_nolegend_")

# Point offsets keep the stage labels clear of the curves and plateau lines.
stage_labels = ["Stage 1a", "Stage 1b", "Stage 2", "Stage 3"]
stage_offsets = [(0, 22), (0, 22), (12, 20), (0, 32)]
for step, loss, label, offset in zip(
        checkpoint_steps, checkpoint_losses, stage_labels, stage_offsets):
    ax.annotate(label, xy=(step, loss), xytext=offset,
                textcoords="offset points", ha="center", va="bottom",
                fontsize=10, zorder=8,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.85),
                arrowprops=dict(arrowstyle="-", color="0.4", lw=0.7,
                                shrinkA=3, shrinkB=8))

for value, label in plateaus:
    ax.axhline(value, color="black", ls=":" if value == L_2_VT else "--",
               lw=1.3, alpha=0.65, zorder=2)
    # Larger plateau labels inside the right edge of the axes.
    below = label == plateaus[1][1]
    ax.text(0.98, value * (0.97 if below else 1.035), label,
            transform=ax.get_yaxis_transform(), fontsize=15, ha="right",
            va="top" if below else "bottom", zorder=7)
# The two L_2 references share the single label above.
ax.axhline(L_2_FT, color="black", ls="--", lw=1.3, alpha=0.65, zorder=2)
ax.axhline(1, color="grey", ls=":", lw=0.9)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(100, max(r["steps"].max() for r in results.values()) * 1.18)
visible_losses = np.concatenate([r["loss"][r["steps"] >= 100] for r in results.values()])
ax.set_ylim(min(visible_losses.min(), plateaus[-1][0]) * 0.82,
            max(1.08, visible_losses.max() * 1.12))
ax.set_xlabel(r"Training step $t$")
ax.set_ylabel(r"$\mathcal{L}/\ln(v)$")
ax.grid(which="both", alpha=0.14, lw=0.5)
ax.legend(loc="lower left", frameon=True, facecolor="white",
          edgecolor="none", framealpha=0.9, fontsize=11)
fig.subplots_adjust(left=0.095, right=0.98, bottom=0.16, top=0.96)

save_figure(fig, args.output)
plt.close(fig)
