#!/usr/bin/env bash
set -euo pipefail

# Run from the root of interpretable_inside_outside.
[ -f README.md ] && [ -d experiments ] && [ -d training ] || {
  echo 'Run this script from the repository root.' >&2; exit 1;
}

python -m unittest discover -s tests -v
python tests/test_grammar_consistency.py
python -m rhm_training --preset smoke --dry-run

for module in plot_learning_curves plot_fixed_vs_varying_tree plot_relative_depth plot_latent_probe_accuracy; do
  python -m "experiments.${module}" --help >/dev/null
done

echo 'Checking required figure-data files...'
python - <<'PY'
from pathlib import Path
p = Path('figure_data')
required = {
  'training': 2,
  'probes': 13,
  'latents': 4,
}
for folder, minimum in required.items():
    files = list((p / folder).glob('*'))
    files = [x for x in files if x.is_file()]
    assert len(files) >= minimum, f'{folder}: expected at least {minimum} files, found {len(files)}'
    print(f'{folder}: {len(files)} files')
PY

python -m experiments.plot_learning_curves --data-dir figure_data/training --output figures/learning_curves
python -m experiments.plot_fixed_vs_varying_tree --data-dir figure_data/probes --output figures/fixed_vs_varying
python -m experiments.plot_relative_depth --data-dir figure_data/probes --output figures/relative_depth
for depth in 3 6 12; do
  python -m experiments.plot_latent_probe_accuracy --depth "$depth" \
    --data-dir figure_data/latents --output "figures/latent_probe_depth${depth}"
done

echo 'All release checks passed.'
