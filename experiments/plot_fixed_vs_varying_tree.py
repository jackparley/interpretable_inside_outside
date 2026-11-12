"""Paper figure: cached fixed-tree and varying-tree probe losses, shared log y-axis."""
from pathlib import Path
import argparse
from experiments.common import load_probe_curve, save_figure
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser(description="Fixed versus varying tree probe losses")
parser.add_argument("--data-dir", type=Path, required=True)
parser.add_argument("--output", type=Path, default=Path("figures/probe_loss_fixed_vs_varying_tree"))
parser.add_argument("--depth12-stage2", type=int, default=262144)
args = parser.parse_args()
WORKSPACE = args.data_dir
VARYING_D12_STAGE2 = args.depth12_stage2

plt.rcParams.update({'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 10,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
fixed_colors = ['tab:blue', 'tab:orange', 'tab:green']
varying_colors = ['tab:purple', 'tab:blue', 'tab:orange', 'tab:green']
varying_labels = ['Stage 1a', 'Stage 1b', 'Stage 2', 'Stage 3']
fixed_curves = {}
for stage, step in enumerate((1024, 262144, 1048576), 1):
    path = WORKSPACE / f'probe_results_step_{step}_ternary_train_ternary_test_last_token_depth3.pkl'
    fixed_curves[stage] = load_probe_curve(WORKSPACE, step, 3, fixed=True)
varying_curves = {}
for D, steps in {6: (1024, 16384, 262144, 2965824),
                 12: (1024, 16384, VARYING_D12_STAGE2, 2097152)}.items():
    for stage, step in enumerate(steps):
        suffix = '' if D == 6 else '_depth12'
        path = WORKSPACE / f'probe_results_step_{step}_mixed_train_ternary_test_last_token{suffix}.pkl'
        varying_curves[D, stage] = load_probe_curve(WORKSPACE, step, D)

fig, (ax_fixed, ax_varying) = plt.subplots(1, 2, sharey=True, figsize=(7.2, 3.65))
for stage, curve in fixed_curves.items():
    ax_fixed.plot(np.arange(4), curve, color=fixed_colors[stage - 1],
                  marker='o', lw=1.6, ms=3.3, zorder=4)
for (D, stage), curve in varying_curves.items():
    ax_varying.plot(np.arange(D + 1) / D, curve, color=varying_colors[stage],
                    ls='--' if D == 6 else '-', marker='s' if D == 6 else 'o',
                    markerfacecolor='white' if D == 6 else varying_colors[stage],
                    lw=1.5, ms=2.8, zorder=4)

plateaus = [(0.59, r'$\mathcal{L}_{\mathrm{trig}}^{\mathrm{VT}}$'),
            (0.452838292701, r'$\mathcal{L}_{\mathrm{trig}}^{\mathrm{FT}}$'),
            (0.20574168745407478, r'$\mathcal{L}_2$'),
            (0.1159764843808, r'$\mathcal{L}_3$')]
for ax, x_text, refs in [(ax_fixed, 3.12, plateaus[1:]), (ax_varying, 1.035, plateaus)]:
    for value, label in refs:
        ax.axhline(value, color='black', ls='--', lw=.85, alpha=.65, zorder=2)
        # Place the right-panel FT label below its line to clear nearby curves.
        below = ax is ax_varying and label == plateaus[1][1]
        ax.text(x_text, value * (.97 if below else 1.035), label,
                fontsize=13, va='top' if below else 'bottom')
    ax.axhline(1, color='grey', ls=':', lw=.9)
    ax.set_yscale('log')
    ax.grid(which='both', alpha=.14, lw=.5)
all_values = np.concatenate([*fixed_curves.values(), *varying_curves.values(),
                             [value for value, _ in plateaus]])
ax_fixed.set_ylim(all_values.min() * .78, max(1.08, all_values.max() * 1.12))
ax_fixed.set(xticks=np.arange(4), xticklabels=['emb', '$k=1$', '$k=2$', '$k=3$'],
             xlim=(-.15, 4.05), xlabel=r'Transformer layer $k$',
             ylabel=r'Probe loss $\mathcal{L}_{x_d}^{(k,t)}$', title='Fixed-tree training')
ax_varying.set(xticks=np.linspace(0, 1, 5), xlim=(-.025, 1.34),
               xlabel=r'Relative transformer layer $k/D$', title='Varying-tree training')
ax_varying.axvline(.5, color='black', ls='--', lw=.8, alpha=.5)
for x, label, ha in [(.48, 'Inside', 'right'), (.52, 'Outside', 'left')]:
    ax_varying.text(x, .975, label, transform=ax_varying.get_xaxis_transform(),
                    ha=ha, va='top', fontsize=7)
fixed_handles = [Line2D([0], [0], color=color, lw=1.6, label=f'Stage {i}')
                 for i, color in enumerate(fixed_colors, 1)]
varying_handles = [Line2D([0], [0], color=color, lw=1.6, label=label)
                   for color, label in zip(varying_colors, varying_labels)]
# Separate stage legends preserve Stage 1 (left) versus Stages 1a/1b (right).
fig.legend(handles=fixed_handles, loc='upper center', bbox_to_anchor=(.29, .995),
           ncol=3, frameon=False, handlelength=1.4, columnspacing=.9)
fig.legend(handles=varying_handles, loc='upper center', bbox_to_anchor=(.765, .995),
           ncol=2, frameon=False, handlelength=1.4, columnspacing=.9)
depth_handles = [Line2D([0], [0], color='black', ls=ls, marker=marker,
                       markerfacecolor='white' if D == 6 else 'black',
                       lw=1.4, ms=3, label=f'$D={D}$')
                 for D, ls, marker in [(6, '--', 's'), (12, '-', 'o')]]
ax_varying.legend(handles=depth_handles, loc='lower left', fontsize=7,
                   frameon=False, ncol=2, handlelength=1.6, columnspacing=1)
fig.subplots_adjust(left=.085, right=.99, bottom=.15, top=.77, wspace=.12)
save_figure(fig, args.output)
plt.close(fig)
