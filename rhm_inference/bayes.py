"""Bayes-optimal last-token predictions for varying-tree RHM grammars.

This module keeps the original sum-product implementation in ``core.py``.
Its public functions provide a smaller, parameterized interface without
changing the historical ``inside_outside`` scripts.
"""
from dataclasses import dataclass
import math
import random

import numpy as np
import torch

from .core import (
    calculate_entropy,
    extract_binary_rules_as_tensors,
    extract_ternary_rules_as_tensors,
    generate_grammar,
    generate_grammar_2,
    generate_rules_double_new,
    parallel_inside_masked,
    parallel_outside,
    sample_from_grammar,
)

KNOWN_LAYERS = ("all", "bottom-intermediate", "bottom", "bottom-binary")


@dataclass(frozen=True)
class GrammarConfig:
    v: int = 8
    L: int = 3
    m2: int = 2
    m3: int = 16
    seed_rules: int = 12345678
    eta: float = 8.0

    def __post_init__(self):
        if self.v < 2 or self.L < 2 or not (0 < self.m2 <= self.v) or not (0 < self.m3 <= self.v**2):
            raise ValueError("Require v>=2, L>=2, 0<m2<=v, 0<m3<=v**2")
        if self.eta <= 0 or not math.isfinite(self.eta):
            raise ValueError("eta must be positive and finite")

    @property
    def p2(self):
        return self.eta / (self.m2 * self.eta + self.m3)

    @property
    def p3(self):
        return 1.0 / (self.m2 * self.eta + self.m3)


def build_grammar(config: GrammarConfig):
    """Sample one grammar, returning (sampling_grammar, binary, ternary).

    Rule tensors are in root-to-leaf order, with shapes
    [L,v,v,v] and [L,v,v,v,v], respectively.
    """
    v, L = config.v, config.L
    rules_1, rules_2 = generate_rules_double_new(
        v, v, config.m2, config.m3, config.p2, config.p3, L,
        config.seed_rules,
    )
    sampling_grammar = generate_grammar(rules_1)
    indexed_grammar = generate_grammar_2(rules_2)
    binary = extract_binary_rules_as_tensors(indexed_grammar, v)
    ternary = extract_ternary_rules_as_tensors(indexed_grammar, v)
    layers = [f"layer{i + 1}" for i in range(L)]
    return (sampling_grammar,
            torch.stack([binary[layer] for layer in layers]),
            torch.stack([ternary[layer] for layer in layers]))


def predictor_tensors(config: GrammarConfig, binary, ternary, known_layers="all"):
    """Replace unknown layers by ensemble-mean rule tensors.

    Uses precisely the approximation/convention of the historical script;
    this is not a posterior integration over the particular sampled grammar.
    """
    if known_layers not in KNOWN_LAYERS:
        raise ValueError(f"known_layers must be one of {KNOWN_LAYERS}")
    bin_t, ter_t = binary.clone(), ternary.clone()
    if known_layers == "all":
        bin_unknown, ter_unknown = [], []
    elif known_layers == "bottom-intermediate":
        bin_unknown, ter_unknown = [0], [0]
    elif known_layers == "bottom":
        bin_unknown = ter_unknown = list(range(config.L - 1))
    else:
        bin_unknown, ter_unknown = list(range(config.L - 1)), list(range(config.L))
    bin_mean = config.p2 * config.m2 / config.v**2
    ter_mean = config.p3 * config.m3 / config.v**3
    for idx in bin_unknown:
        bin_t[idx].fill_(bin_mean)
    for idx in ter_unknown:
        ter_t[idx].fill_(ter_mean)
    return bin_t, ter_t


def last_token_distribution(sequence, binary, ternary, config: GrammarConfig,
                            unknown_length=True):
    """P(last token | observed prefix), with the last token masked.

    As in the historical calculation, the last-token outside marginal is
    obtained with uniformly masked leaves. Returns None if normalization
    vanishes. ``unknown_length=True`` sets the chart size to 3**L.
    """
    d = len(sequence)
    if d < 2:
        raise ValueError("Sequence must have at least two tokens")
    padded = 3**config.L if unknown_length else None
    _, inside_list = parallel_inside_masked(
        sequence, binary, ternary, config.L, config.v, padded=padded)
    outside, _ = parallel_outside(
        d, binary, ternary, config.L, config.v, inside_list, padded=padded)
    weights = (outside[:, d - 1, d] / config.v).detach().cpu().double()
    total = weights.sum()
    if not bool(torch.isfinite(total)) or float(total) <= 0:
        return None
    return weights / total


def estimate_bayes_loss(config: GrammarConfig, n_samples=200,
                        known_layers="all", sample_seed=42,
                        unknown_length=True):
    """Compute per-example normalized conditional entropy and summary.

    Standard errors describe sentence-to-sentence sampling conditional on
    the fixed grammar; they do not include grammar-to-grammar fluctuations.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    sampling_grammar, binary, ternary = build_grammar(config)
    pred_bin, pred_ter = predictor_tensors(config, binary, ternary, known_layers)
    random.seed(sample_seed)
    np.random.seed(sample_seed)
    torch.manual_seed(sample_seed)
    rows = []
    failures = 0
    for _ in range(n_samples):
        root = str(random.randint(1, config.v))
        seq, _ = sample_from_grammar(
            sampling_grammar, root, config.L, config.p2, config.p3)
        pred = last_token_distribution(seq, pred_bin, pred_ter, config,
                                       unknown_length=unknown_length)
        if pred is None:
            failures += 1
            continue
        H = float(calculate_entropy(pred) / np.log(config.v))
        rows.append({"entropy": H, "length": len(seq),
                     "support": int((pred > 0).sum().item()),
                     "optimal_accuracy": float(pred.max())})
    h = np.asarray([r["entropy"] for r in rows], dtype=float)
    accuracies = np.asarray([r["optimal_accuracy"] for r in rows], dtype=float)
    summary = {
        "known_layers": known_layers,
        "n_requested": n_samples,
        "n_valid": len(rows),
        "zero_partition_count": failures,
        "mean_entropy": float(h.mean()) if len(h) else None,
        "se_entropy": float(h.std(ddof=1) / np.sqrt(len(h))) if len(h) > 1 else None,
        "mean_optimal_accuracy": float(accuracies.mean()) if len(accuracies) else None,
        "mean_support": float(np.mean([r["support"] for r in rows])) if rows else None,
        "normalized_by": "ln(v)",
        "unknown_length": unknown_length,
    }
    return summary, rows
