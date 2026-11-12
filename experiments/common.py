"""Shared, read-only loaders and validation for paper plots."""
from pathlib import Path
import pickle
import numpy as np

PLATEAUS = [
    (0.590867, r"$\mathcal{L}_{\mathrm{trig}}^{\mathrm{VT}}$"),
    (0.452838292701, r"$\mathcal{L}_{\mathrm{trig}}^{\mathrm{FT}}$"),
    (0.20574168745407478, r"$\mathcal{L}_2$"),
    (0.1159764843808, r"$\mathcal{L}_3$"),
]
EXPECTED_METADATA = {
    "probe_C": 0.1, "train_seed": 42, "test_seed": 43,
    "n_test_ternary": 32768, "probe_position": "last input position",
}

def probe_filename(step, depth, *, fixed=False):
    if fixed:
        return f"probe_results_step_{step}_ternary_train_ternary_test_last_token_depth3.pkl"
    suffix = "" if depth == 6 else f"_depth{depth}"
    return f"probe_results_step_{step}_mixed_train_ternary_test_last_token{suffix}.pkl"

def load_probe_curve(folder, step, depth, *, fixed=False):
    path = Path(folder) / probe_filename(step, depth, fixed=fixed)
    if not path.is_file():
        raise FileNotFoundError(f"Missing cached probe result: {path}")
    # Only load pickle files produced by your own experiments; unpickling is not safe for untrusted data.
    with path.open("rb") as fh:
        saved = pickle.load(fh)
    if saved.get("checkpoint_step") != step:
        raise ValueError(f"Checkpoint step mismatch: {path}")
    if fixed and saved.get("experiment") != "ternary_train_ternary_test_last_token_depth3_v1":
        raise ValueError(f"Wrong fixed-tree experiment: {path}")
    expected = {**EXPECTED_METADATA,
                "n_train_ternary" if fixed else "n_train_mixed": 32768}
    if any(saved.get(k) != v for k,v in expected.items()):
        raise ValueError(f"Probe metadata mismatch: {path}")
    values = saved.get("token_probe", saved)
    curve = np.asarray(values["test_cross_entropy_normalized"], dtype=float)
    if curve.shape != (depth+1,) or not np.all(np.isfinite(curve)) or np.any(curve <= 0):
        raise ValueError(f"Expected {depth+1} positive finite losses in {path}")
    return curve

def save_figure(fig, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = destination.with_suffix("." + ext)
        fig.savefig(path, dpi=300)
        print(f"Saved: {path}")
