# Standalone: load cached losses only; no checkpoint loading or probe fitting.
from pathlib import Path
import argparse
from experiments.common import load_probe_curve, save_figure, PLATEAUS, EXPECTED_METADATA
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser(description="Relative-depth probe comparison")
parser.add_argument("--data-dir", type=Path, required=True)
parser.add_argument("--output", type=Path, default=Path("figures/token_probe_relative_depth"))
parser.add_argument("--depth12-stage2", type=int, default=262144)
args = parser.parse_args()
WORKSPACE = args.data_dir
COMPARISON_STEPS = {6: (512, 16384, 262144, 2965824),
                    12: (512, 16384, args.depth12_stage2, 2097152)}

plt.rcParams.update({'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 10,
                     'legend.fontsize': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42})
stage_labels = ['Stage 1a', 'Stage 1b', 'Stage 2', 'Stage 3']
stage_colors = ['tab:purple', 'tab:blue', 'tab:orange', 'tab:green']

def plot_relative_depth_comparison(steps_by_depth, output_suffix=''):
    curves = {}
    for D, steps in steps_by_depth.items():
        for stage, step in enumerate(steps, 1):
            curve = load_probe_curve(WORKSPACE, step, D)
            curves[D, stage] = curve
            print(f'D={D}, {stage_labels[stage - 1]}, t={step}')

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for (D, stage), curve in curves.items():
        ax.plot(np.arange(D + 1) / D, curve, color=stage_colors[stage - 1],
                ls='--' if D == 6 else '-', marker='s' if D == 6 else 'o',
                markerfacecolor='white' if D == 6 else stage_colors[stage - 1],
                lw=1.6, ms=3.4, alpha=.95, zorder=4)
    for value, label in PLATEAUS:
        ax.axhline(value, color='black', ls='--', lw=.95, alpha=.65, zorder=2)
        ax.text(1.025, value * 1.025, label, fontsize=8.5, va='bottom')
    ax.axhline(1, color='grey', ls=':', lw=1)
    ax.axvline(.5, color='black', ls='--', lw=.9, alpha=.5)
    for x, label, ha in [(.48, 'Inside', 'right'), (.52, 'Outside', 'left')]:
        ax.text(x, .975, label, transform=ax.get_xaxis_transform(), ha=ha, va='top', fontsize=8)
    values = np.concatenate([*curves.values(), [value for value, _ in PLATEAUS]])
    ax.set(yscale='log', xlim=(-.025, 1.2),
           ylim=(values.min() * .78, max(1.08, values.max() * 1.12)),
           xticks=np.linspace(0, 1, 5), xlabel=r'Relative transformer layer $k/D$',
           ylabel=r'Probe loss $\mathcal{L}_{x_d}^{(k,t)}$',
           title='Varying-tree training')
    ax.grid(which='both', alpha=.16, lw=.55)
    stage_handles = [Line2D([0], [0], color=color, lw=1.8, label=stage_labels[i - 1])
                     for i, color in enumerate(stage_colors, 1)]
    depth_handles = [Line2D([0], [0], color='black', ls=ls, marker=marker,
                           markerfacecolor='white' if D == 6 else 'black',
                           lw=1.6, ms=3.4, label=f'$D={D}$')
                     for D, ls, marker in [(6, '--', 's'), (12, '-', 'o')]]
    fig.legend(handles=stage_handles, loc='upper center', ncol=4,
               frameon=False, bbox_to_anchor=(.5, 1.0))
    fig.legend(handles=depth_handles, loc='upper center', ncol=2,
               frameon=False, bbox_to_anchor=(.5, .94))
    fig.tight_layout(rect=(0, 0, 1, .84))
    save_figure(fig, args.output)
    plt.close(fig)
    return fig

plot_relative_depth_comparison(COMPARISON_STEPS)
