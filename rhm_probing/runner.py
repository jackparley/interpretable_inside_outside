"""Run the original layerwise multinomial logistic probes through a reusable interface.

Requires a saved baseline probe bundle with the exact original probe_metadata,
dataset seeds, and grammar. Does not change targets, feature extraction or fitting.
"""
from __future__ import annotations

import gc
import pickle
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "training" / "scripts"
if str(LEGACY) not in sys.path:
    sys.path.insert(0, str(LEGACY))
import probe_stage2_16384 as shared  # noqa: E402

DEFAULT_PROBES = ("second_last_level1", "third_last_level1", "third_last_level2")


def load_reference_bundle(path):
    with Path(path).open("rb") as handle:
        bundle = pickle.load(handle)  # Trusted research-owned pickle files only.
    for key in ("grammar", "dataset", "probe_metadata", "probe_C"):
        if key not in bundle:
            raise ValueError(f"Reference bundle lacks {key!r}")
    return bundle


def infer_checkpoint(state, num_heads):
    # Strip the prefix from torch.compile checkpoints without changing weights.
    cleaned = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    arch = shared.infer_architecture(cleaned, num_heads)
    return cleaned, arch


def _build_data(bundle):
    grammar, dataset = bundle["grammar"], bundle["dataset"]
    shared.V, shared.L = int(grammar["V"]), int(grammar["L"])
    m2, m3 = int(grammar["BIN"]), int(grammar["BIN"]) * shared.V
    shared.rules = shared.sample_padded_rules(shared.V, shared.V, m2, m3, 2, 3, shared.L, int(grammar["SEED_RULES"]))
    shared.ternary_probs = {
        level: torch.cat((torch.zeros(m2), torch.full((m3,), 1. / m3)))
        for level in range(shared.L)
    }
    train, train_latents = shared.generate_fully_ternary(int(dataset["n_train"]), int(dataset["train_seed"]))
    test, test_latents = shared.generate_fully_ternary(int(dataset["n_test"]), int(dataset["test_seed"]))
    return train, train_latents, test, test_latents


def probe_checkpoints(reference_path, checkpoint_paths, output_path, *, num_heads=16,
                      probes=DEFAULT_PROBES, batch_size=64, device=None, resume=True):
    """Returns a compatible result bundle; saves after each checkpoint atomically.

    The reference bundle must come from the original token-level probing experiment.
    Checkpoint paths may have any architecture depth D (including 3, 6, 12),
    provided they use the same underlying grammar and positional task.
    """
    reference = load_reference_bundle(reference_path)
    metadata = {name: reference["probe_metadata"][name] for name in probes}
    output = Path(output_path)
    if not checkpoint_paths:
        raise ValueError("Specify at least one checkpoint")
    if device is not None:
        if device not in ("cpu", "cuda", "mps"):
            raise ValueError(f"Unsupported device {device}")
        shared.DEVICE = device
    existing = None
    if resume and output.exists():
        with output.open("rb") as handle:
            existing = pickle.load(handle)
        for key in ("grammar", "dataset", "probe_C"):
            if existing[key] != reference[key]:
                raise ValueError(f"Output bundle has different {key}; use a different --output")
        if existing["probe_metadata"] != metadata:
            raise ValueError("Output bundle has different probe definitions")

    # The original helper sets model import paths relative to training/scripts.
    train, train_latents, test, test_latents = _build_data(reference)
    target_length = int(reference["dataset"]["input_length"])
    checkpoint_paths = [Path(p) for p in checkpoint_paths]
    result_bundle = existing
    for path in checkpoint_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        saved = torch.load(path, map_location="cpu", weights_only=False)  # Trusted checkpoints only.
        if "model" not in saved:
            raise ValueError(f"{path}: missing model weights; use --checkpoints during training")
        state, arch = infer_checkpoint(saved["model"], num_heads)
        if int(arch["block_size"]) != target_length:
            raise ValueError(f"Checkpoint block_size={arch['block_size']}, expected {target_length}")
        step = saved.get("step", None)
        if step is None:
            raise ValueError(f"{path}: checkpoint has no step; cannot index results reproducibly")
        step = int(step)
        if result_bundle is not None and arch != result_bundle["architecture"]:
            raise ValueError("Output already contains a different Transformer architecture")
        if result_bundle is not None and step in result_bundle["results"]:
            print(f"Skipping step {step}: already in {output}", flush=True)
            continue
        if result_bundle is None:
            result_bundle = {key: reference[key] for key in ("grammar", "dataset", "probe_C", "chance_accuracy") if key in reference}
            result_bundle.update(architecture=arch,
                                 layer_labels=["emb"] + [f"L{i}" for i in range(1, arch["num_layers"] + 1)],
                                 probe_metadata=metadata, results={})
        model = shared.models.CLM(**arch)
        model.load_state_dict(state)
        model.to(shared.DEVICE).eval()
        positions = {name: int(spec["position"]) for name, spec in metadata.items()}
        train_acts = shared.extract_multiple_position_activations(model, train[:, :target_length], positions, batch_size)
        test_acts = shared.extract_multiple_position_activations(model, test[:, :target_length], positions, batch_size)
        result = {"checkpoint": str(path), "saved_step": step, "probes": {}}
        for name, spec in metadata.items():
            index = shared.L - int(spec["level"])
            result["probes"][name] = shared.fit_layerwise_probes(
                train_acts.pop(name), train_latents[index][:, int(spec["node"])].numpy(),
                test_acts.pop(name), test_latents[index][:, int(spec["node"])].numpy(),
                C=float(reference["probe_C"]))
        result_bundle["results"][step] = result
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_name(output.name + ".tmp")
        with temp.open("wb") as handle:
            pickle.dump(result_bundle, handle)
        temp.replace(output)
        print(f"Saved {path.name} step={step} to {output}", flush=True)
        del model, train_acts, test_acts, saved, state
        gc.collect()
        if shared.DEVICE == "mps":
            torch.mps.empty_cache()
        elif shared.DEVICE == "cuda":
            torch.cuda.empty_cache()
    return result_bundle
