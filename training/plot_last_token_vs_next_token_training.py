"""Compare cached last-token probes of next-token and last-token trained models."""
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

WORKSPACE = Path(__file__).resolve().parent
D = 12
STAGES = [('Stage 1a', 4096, 512, 'tab:purple'),
          ('Stage 3', 1048576, 2097152, 'tab:green')]
EXPECTED = {'probe_C': .1, 'n_train_mixed': 32768, 'n_test_ternary': 32768,
            'train_seed': 42, 'test_seed': 43, 'probe_position': 'last input position'}

def load_test_loss(step, next_token):
    suffix = '_rope' if next_token else ''
    path = WORKSPACE / f'probe_results_step_{step}_mixed_train_ternary_test_last_token_depth12{suffix}.pkl'
    if not path.is_file():
        raise FileNotFoundError(f'Run the missing probe checkpoint first: {path}')
    with path.open('rb') as handle:
        saved = pickle.load(handle)
    if saved.get('checkpoint_step') != step or any(saved.get(k) != v for k, v in EXPECTED.items()):
        raise ValueError(f'Checkpoint or probe metadata mismatch: {path}')
    if next_token and (saved.get('experiment') != 'mixed_train_ternary_test_last_token_depth12_rope_v1'
                       or not saved.get('architecture', {}).get('rope')):
        raise ValueError(f'Expected dedicated next-token/RoPE results: {path}')
    curve = np.asarray(saved.get('token_probe', saved)['test_cross_entropy_normalized'], dtype=float)
    if curve.shape != (D + 1,) or not np.all(np.isfinite(curve)) or np.any(curve <= 0):
        raise ValueError(f'Expected {D + 1} positive finite losses: {path}')
    print(f'Loaded: {path.name}')
    return curve

plt.rcParams.update({'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 10,
                     'legend.fontsize': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42})
fig, ax = plt.subplots(figsize=(5.4, 3.8))
curves = {}
for stage, next_step, last_step, color in STAGES:
    for next_token, step in [(True, next_step), (False, last_step)]:
        curve = load_test_loss(step, next_token)
        curves[stage, next_token] = curve
        ax.plot(np.arange(D + 1) / D, curve, color=color,
                ls='-' if next_token else '--', marker='o' if next_token else 's',
                markerfacecolor=color if next_token else 'white',
                lw=1.7, ms=3.4, zorder=4)
plateaus = [(0.590867, r'$\mathcal{L}_{\mathrm{trig}}^{\mathrm{VT}}$'),
            (0.1159764843808, r'$\mathcal{L}_3$')]
for value, label in plateaus:
    ax.axhline(value, color='black', ls='--', lw=.9, alpha=.65)
    ax.text(1.025, value * 1.035, label, fontsize=8.5, va='bottom')
ax.axhline(1, color='grey', ls=':', lw=.9)
ax.axvline(.5, color='black', ls='--', lw=.8, alpha=.5)
for x, label, ha in [(.48, 'Inside', 'right'), (.52, 'Outside', 'left')]:
    ax.text(x, .975, label, transform=ax.get_xaxis_transform(), ha=ha, va='top', fontsize=8)
values = np.concatenate([*curves.values(), [v for v, _ in plateaus]])
ax.set(yscale='log', ylim=(values.min() * .78, max(1.08, values.max() * 1.12)),
       xlim=(-.025, 1.2), xticks=np.linspace(0, 1, 5),
       xlabel=r'Relative transformer layer $k/D$',
       ylabel=r'Probe loss $\mathcal{L}_{x_d}^{(k,t)}$',
       title='Last-token vs Next-token training')
ax.grid(which='both', alpha=.15, lw=.5)
method_handles = [
    Line2D([0], [0], color='black', ls='-', marker='o', lw=1.7, ms=3.4,
           label='Varying-tree next-token training'),
    Line2D([0], [0], color='black', ls='--', marker='s', markerfacecolor='white',
           lw=1.7, ms=3.4, label='Varying-tree last-token training'),
]
stage_handles = [Line2D([0], [0], color=color, lw=1.8, label=stage)
                 for stage, _, _, color in STAGES]
fig.legend(handles=method_handles, loc='upper center', frameon=False,
           bbox_to_anchor=(.5, 1.0), ncol=1)
fig.legend(handles=stage_handles, loc='upper center', frameon=False,
           bbox_to_anchor=(.5, .875), ncol=2)
fig.tight_layout(rect=(0, 0, 1, .80))
for extension in ('pdf', 'png'):
    path = WORKSPACE / f'token_probe_cross_entropy_last_vs_next_token_training_depth12.{extension}'
    fig.savefig(path, dpi=300)
    print('Saved:', path)
plt.show()
