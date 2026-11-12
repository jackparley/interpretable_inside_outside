"""Compute only the Stage 2 probes required by the final standalone plots.
Run: python scripts/probe_stage2_16384.py
Uses the original bundle's dataset, grammar, architecture and classifier settings.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
from pathlib import Path
import sys
import pickle
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "datasets" / "rhm"))
import models
from mixed_random_hierarchy_model import sample_padded_rules
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch.set_num_threads(4)

def generate_fully_ternary(n_examples, seed):
    """Return terminals (N, 27) and latents [root, level-2, level-1]."""
    generator = torch.Generator().manual_seed(seed)
    root = torch.randint(0, V, (n_examples,), generator=generator)
    features = root.clone()
    latents = [root.clone()]

    for level in range(L - 1):
        flat = features.reshape(-1)
        selected = torch.multinomial(
            ternary_probs[level].expand(flat.numel(), -1),
            num_samples=1,
            replacement=True,
            generator=generator,
        ).squeeze(-1).reshape(features.shape)
        features = rules[level][features, selected].flatten(start_dim=1)
        latents.append(features.clone())

    flat = features.reshape(-1)
    selected = torch.multinomial(
        ternary_probs[L - 1].expand(flat.numel(), -1),
        num_samples=1,
        replacement=True,
        generator=generator,
    ).squeeze(-1).reshape(features.shape)
    terminals = rules[L - 1][features, selected].flatten(start_dim=1)
    return terminals.long(), latents

def infer_architecture(state_dict, num_heads):
    embedding_dim = state_dict["token_embedding_table.weight"].shape[1]
    num_layers = max(
        int(key.split(".")[1]) for key in state_dict if key.startswith("blocks.")
    ) + 1
    return dict(
        vocab_size=state_dict["token_embedding_table.weight"].shape[0],
        block_size=state_dict["blocks.0.attn.tril"].shape[0],
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        ffwd_size=state_dict["blocks.0.ffwd.net.0.weight"].shape[0] // embedding_dim,
        num_layers=num_layers,
        dropout=0.0,
        rope="position_embedding_table.weight" not in state_dict,
        share_emb=False,
    )

def load_model(checkpoint_path, num_heads, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    architecture = infer_architecture(checkpoint["model"], num_heads)
    model = models.CLM(**architecture)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, architecture, checkpoint.get("step"), checkpoint.get("state", {})

def fit_layerwise_probes(train_activations, train_targets,
                         test_activations, test_targets, C=0.1):
    """Fit independent standardized multinomial linear probes at every layer."""
    if train_activations.shape[0] != test_activations.shape[0]:
        raise ValueError("Train and test activations have different layer counts")

    train_accuracy, test_accuracy = [], []
    for layer_index, (x_train, x_test) in enumerate(
        zip(train_activations, test_activations)
    ):
        probe = Pipeline([
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(
                C=C, max_iter=5000, solver="lbfgs", random_state=0
            )),
        ])
        probe.fit(x_train, train_targets)
        train_score = probe.score(x_train, train_targets)
        test_score = probe.score(x_test, test_targets)
        train_accuracy.append(train_score)
        test_accuracy.append(test_score)
        print(f"layer {layer_index:2d}: train={train_score:.3f}, test={test_score:.3f}")

    return {"train": np.asarray(train_accuracy), "test": np.asarray(test_accuracy)}

@torch.inference_mode()
def extract_multiple_position_activations(model, inputs, positions, batch_size=256):
    """Extract several causal positions in one forward pass.

    Returns a dict mapping each probe name to an array shaped
    (embedding + blocks, examples, hidden_dim).
    """
    for name, position in positions.items():
        if not 0 <= position < inputs.shape[1]:
            raise ValueError(f"{name}: position {position} outside length {inputs.shape[1]}")

    collected = {
        name: [[] for _ in range(model.num_layers + 1)]
        for name in positions
    }

    for start in range(0, len(inputs), batch_size):
        batch = inputs[start:start + batch_size].to(DEVICE)
        captured = {name: {} for name in positions}

        def capture_embedding(module, args):
            embedding = args[0]
            for name, position in positions.items():
                captured[name][0] = embedding[:, position].detach().cpu()

        pre_hook = model.blocks[0].register_forward_pre_hook(capture_embedding)

        def capture_block(layer_index):
            def hook(module, args, output):
                for name, position in positions.items():
                    captured[name][layer_index + 1] = output[:, position].detach().cpu()
            return hook

        block_hooks = [
            block.register_forward_hook(capture_block(layer_index))
            for layer_index, block in enumerate(model.blocks)
        ]
        try:
            model(batch)
        finally:
            pre_hook.remove()
            for hook in block_hooks:
                hook.remove()

        for name in positions:
            for layer_index in range(model.num_layers + 1):
                collected[name][layer_index].append(captured[name][layer_index])

    return {
        name: np.stack([
            torch.cat(layer_batches, dim=0).numpy()
            for layer_batches in collected[name]
        ])
        for name in positions
    }

def main():
    global V, L, rules, ternary_probs
    base_path = WORKSPACE / "second_third_last_probe_checkpoint_results.pkl"
    output = WORKSPACE / "second_third_last_probe_step_16384.pkl"
    with base_path.open("rb") as handle:
        base = pickle.load(handle)
    if output.exists():
        print(f"Already saved: {output}; remove or rename it to recompute.")
        return
    grammar, dataset = base["grammar"], base["dataset"]
    V, L = grammar["V"], grammar["L"]
    m2, m3 = grammar["BIN"], grammar["BIN"] * V
    rules = sample_padded_rules(V, V, m2, m3, 2, 3, L, grammar["SEED_RULES"])
    ternary_probs = {level: torch.cat([torch.zeros(m2), torch.full((m3,), 1.0 / m3)]) for level in range(L)}
    train, train_latents = generate_fully_ternary(dataset["n_train"], dataset["train_seed"])
    test, test_latents = generate_fully_ternary(dataset["n_test"], dataset["test_seed"])
    names = ["second_last_level1", "third_last_level1", "third_last_level2"]
    metadata = {name: base["probe_metadata"][name] for name in names}
    checkpoint = WORKSPACE / "run_L3_v8_bin2_P16777216_lr0.001_l20.03_bs128_arr0_t16384.pt"
    model, architecture, step, state = load_model(checkpoint, base["architecture"]["num_heads"], DEVICE)
    assert architecture == base["architecture"], "Architecture differs from the original probes"
    assert step is None or int(step) == 16384, f"Unexpected checkpoint step: {step}"
    positions = {name: config["position"] for name, config in metadata.items()}
    print(f"Extracting t=16384 train/test activations on {DEVICE}", flush=True)
    train_acts = extract_multiple_position_activations(model, train[:, :dataset["input_length"]], positions, 64)
    test_acts = extract_multiple_position_activations(model, test[:, :dataset["input_length"]], positions, 64)
    result = {"checkpoint": str(checkpoint), "saved_step": step, "val_loss": state.get("val_loss"), "probes": {}}
    for name, config in metadata.items():
        print(name, flush=True)
        latent_index = L - config["level"]
        result["probes"][name] = fit_layerwise_probes(
            train_acts.pop(name), train_latents[latent_index][:, config["node"]].numpy(),
            test_acts.pop(name), test_latents[latent_index][:, config["node"]].numpy(), C=base["probe_C"])
    bundle = {key: base[key] for key in ["grammar", "dataset", "architecture", "probe_C", "layer_labels", "chance_accuracy"]}
    bundle.update(results={16384: result}, probe_metadata=metadata)
    temporary = output.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(bundle, handle)
    temporary.replace(output)
    print(f"Saved {output}", flush=True)
    for name, values in result["probes"].items():
        print(name, dict(zip(base["layer_labels"], values["test"])), flush=True)

if __name__ == "__main__":
    main()
