# Plotting the paper figures

Run these commands **from the repository root** (e.g. `python -m experiments.plot_learning_curves`).
They read **saved numerical results**, not model checkpoints, and never run training or probing.

```bash
python -m experiments.plot_learning_curves --data-dir /path/to/training-results --output figures/learning_curves
python -m experiments.plot_fixed_vs_varying_tree --data-dir /path/to/probe-caches --output figures/probe_fixed_vs_varying
python -m experiments.plot_relative_depth --data-dir /path/to/probe-caches --output figures/probe_relative_depth
```

All commands generate `.pdf` and `.png` files. The original figures used a specific set of run configurations and validation metrics, so the filenames and metadata are intentionally validated; missing inputs produce explicit errors. To select the later D=12 Stage-2 probe cache, add `--depth12-stage2 741456` to either probe command.

## Expected input files

- `plot_learning_curves` uses two training-history `.pt` files: D=6 varying training, D=3 fixed training. It reads `dynamics` entries (`ternary_val_loss` or `val_loss`) and normalizes by `ln(v)`. The filename and hyperparameters follow the source notebook; see `configs` in the script.
- The probe plots expect files named `probe_results_step_<step>_...pkl` with the original `checkpoint_step`, `probe_C`, seeds, number of samples, and `token_probe.test_cross_entropy_normalized` fields. These are **probe result caches**, not `.pt` model weights.
- Step sets for the two probe plots are not identical: the original figures use step `1024` in the two-panel comparison and `512` in the relative-depth comparison.
- Plotting requires `numpy`, `matplotlib`, and `torch` (torch is only required by the training-history loader).

Load only your own trusted `.pt` and `.pkl` files; these formats can execute code when deserialized.

The original source scripts and notebooks in `training/` remain the reference. The plotting commands were extracted without changing metric definitions, plateau values, stages or figure styles.

## Integration

Copy the `experiments/` directory from this patch into the root of `interpretable_inside_outside/`. It does not overwrite the original plotting notebook or scripts under `training/`.

The plots were smoke-tested with synthetic files matching the documented result schemas. Actual paper curves still require the corresponding original `.pt` training histories and cached `.pkl` probe results. The `.pkl` probe caches are often much smaller than full model checkpoints and can be archived independently if appropriate.
