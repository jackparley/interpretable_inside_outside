import argparse
import os
import random
import time

import numpy as np
import torch

from Inside_outside import (
    calculate_entropy,
    extract_binary_rules_as_tensors,
    extract_ternary_rules_as_tensors,
    generate_grammar,
    generate_grammar_2,
    generate_rules_double_new,
    parallel_inside_masked,
    parallel_outside,
)


V = 8
F = 1 / 4
DEPTH = 3
RULE_SEED = 12345678
ETA = 0.0


def parse_eta(value):
    if value.lower() == "v":
        return float(V)
    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("eta must be a number or 'v'") from exc


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute inside-outside and CYK-support accuracy for the last "
            "token or a latent on the last branch of a fixed ternary grammar."
        )
    )
    parser.add_argument("N_samp", type=int, help="number of sampled ternary trees")
    parser.add_argument(
        "--latent",
        "--target",
        dest="latent",
        choices=("token", "root", "level2", "level1"),
        default="level1",
        help="predict the last token or the selected last-branch latent",
    )
    parser.add_argument(
        "--known-layers",
        nargs="+",
        choices=("all", "bottom-intermediate", "bottom", "bottom-binary"),
        default=("bottom", "bottom-intermediate", "all"),
        help=(
            "rule-knowledge settings to evaluate; by default evaluate bottom "
            "only, bottom+intermediate, and all rules"
        ),
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="optional seed for tree sampling",
    )
    parser.add_argument(
        "--eta",
        type=parse_eta,
        default=ETA,
        help=(
            "binary-to-ternary rule-probability ratio; use --eta v to set it "
            f"to the vocabulary size ({V}), or supply a number (default: {ETA:g})"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="directory for per-sample output",
    )
    parser.add_argument(
        "--span-readout",
        choices=("marginal", "fixed"),
        default="marginal",
        help=(
            "marginal sums over all last-branch spans compatible with the "
            "latent level; fixed uses the ground-truth ternary span"
        ),
    )
    parser.add_argument(
        "--predictor",
        choices=("both", "posterior", "cyk"),
        default="both",
        help=(
            "which readout(s) to report; posterior uses weighted "
            "inside-outside probabilities, while cyk floors the same chart to "
            "support and uses a uniform random guess among possible latent "
            "classes (default: both)"
        ),
    )
    return parser.parse_args()


def symbol_to_class(symbol, v):
    return (int(symbol) - 1) % v


def sample_ternary_with_latents(grammar, start_symbol, L):
    """Sample an all-ternary tree and keep latent symbols at each depth."""
    current_symbols = [start_symbol]
    latent_levels = [current_symbols.copy()]

    for layer in range(1, L + 1):
        layer_key = f"layer{layer}"
        next_symbols = []

        for symbol in current_symbols:
            productions = [
                rhs
                for rhs, prob in grammar[layer_key][symbol]
                if len(rhs) == 3 and prob > 0
            ]
            if not productions:
                raise RuntimeError(
                    f"No positive-probability ternary rules for symbol {symbol} "
                    f"in {layer_key}"
                )
            next_symbols.extend(random.choice(productions))

        current_symbols = next_symbols
        if layer < L:
            latent_levels.append(current_symbols.copy())

    terminals = current_symbols
    return terminals, latent_levels


def last_branch_target(latent_name, latent_levels, L, d, v, span_readout):
    target_leaf_pos = d - 1

    if latent_name == "root":
        top_depth = 0
    elif latent_name == "level2":
        top_depth = 1
    else:
        top_depth = L - 1

    span_size = 3 ** (L - top_depth)
    node_idx = target_leaf_pos // span_size
    i = node_idx * span_size
    j = i + span_size

    if span_readout == "fixed" or latent_name == "root":
        candidate_spans = [(i, j)]
    else:
        min_span = 2 ** (L - top_depth)
        max_span = 3 ** (L - top_depth)
        candidate_spans = [
            (d - span_len, d)
            for span_len in range(min_span, max_span + 1)
            if d - span_len >= 0
        ]

    true_symbol = latent_levels[top_depth][node_idx]
    true_class = symbol_to_class(true_symbol, v)

    inside_idx = L - top_depth
    outside_idx = top_depth

    return {
        "top_depth": top_depth,
        "node_idx": node_idx,
        "true_span_i": i,
        "true_span_j": j,
        "candidate_spans": candidate_spans,
        "inside_idx": inside_idx,
        "outside_idx": outside_idx,
        "true_class": true_class,
    }


def last_token_target(sequence, L, d, v):
    """Return the leaf chart location and class of the masked final token."""
    return {
        "top_depth": L,
        "node_idx": d - 1,
        "true_span_i": d - 1,
        "true_span_j": d,
        "candidate_spans": [(d - 1, d)],
        "inside_idx": 0,
        "outside_idx": L,
        "true_class": symbol_to_class(sequence[-1], v),
    }


def tensors_for_known_layers(
    known_layers,
    base_tensor_bin,
    base_tensor_ter,
    expected_bin,
    expected_ter,
    L,
):
    tensor_bin = base_tensor_bin.clone()
    tensor_ter = base_tensor_ter.clone()

    if known_layers == "all":
        expected_bin_layer_indices = []
        expected_ter_layer_indices = []
    elif known_layers == "bottom-intermediate":
        expected_bin_layer_indices = [0]
        expected_ter_layer_indices = [0]
    elif known_layers == "bottom":
        expected_bin_layer_indices = list(range(L - 1))
        expected_ter_layer_indices = list(range(L - 1))
    else:
        expected_bin_layer_indices = list(range(L - 1))
        expected_ter_layer_indices = list(range(L))

    for layer_idx in expected_bin_layer_indices:
        tensor_bin[layer_idx] = torch.full_like(tensor_bin[layer_idx], expected_bin)

    for layer_idx in expected_ter_layer_indices:
        tensor_ter[layer_idx] = torch.full_like(tensor_ter[layer_idx], expected_ter)

    return tensor_bin, tensor_ter


def latent_readout_distribution(inside, outside, candidate_spans, v, predictor):
    if predictor == "posterior":
        ZZ = torch.zeros(v, dtype=torch.float64, device=inside.device)
        for i, j in candidate_spans:
            ZZ += inside[:, i, j] * outside[:, i, j]

        ZZ = ZZ.detach().to(dtype=torch.float64, device="cpu")
        ZZ_sum = ZZ.sum()
        if ZZ_sum == 0:
            return None, 0
        return ZZ / ZZ_sum, int((ZZ > 0).sum().item())

    possible = torch.zeros(v, dtype=torch.bool, device=inside.device)
    for i, j in candidate_spans:
        possible |= (inside[:, i, j] * outside[:, i, j]) > 0

    n_possible = int(possible.sum().item())
    if n_possible == 0:
        return None, 0

    distribution = torch.zeros(v, dtype=torch.float64, device="cpu")
    distribution[possible.detach().cpu()] = 1.0 / n_possible
    return distribution, n_possible


def main():
    args = parse_args()
    predictors = ("posterior", "cyk") if args.predictor == "both" else (args.predictor,)

    v = V
    f = F
    L = DEPTH
    n = v
    eta = args.eta
    d_fixed = int(3**L)

    m_2 = int(f * v)
    m_3 = int(f * v**2)

    p_3 = 1 / (m_2 * eta + m_3)
    p_2 = eta * p_3

    if args.out_dir is None:
        settings = "-".join(args.known_layers)
        args.out_dir = (
            f"v_{v}_L_{L}_f_{f:g}_seed_{RULE_SEED}_latent_{args.latent}_"
            f"known_{settings}_span_{args.span_readout}_"
            f"predictor_{args.predictor}_eta_{eta:g}"
        )

    os.makedirs(args.out_dir, exist_ok=True)
    sample_filename = os.path.join(args.out_dir, "latent_samples.txt")
    avs_filename = (
        f"v_{v}_L_{L}_f_{f:g}_seed_{RULE_SEED}_latent_{args.latent}_"
        f"span_{args.span_readout}_predictor_{args.predictor}_"
        f"eta_{eta:g}_latent_avs.txt"
    )

    time_1 = time.time()

    rules_1, rules_2 = generate_rules_double_new(
        v, n, m_2, m_3, p_2, p_3, L, RULE_SEED
    )

    grammar_1 = generate_grammar(rules_1)
    grammar_2 = generate_grammar_2(rules_2)

    TT_bin = extract_binary_rules_as_tensors(grammar_2, v)
    TT_ter = extract_ternary_rules_as_tensors(grammar_2, v)
    sorted_layers = sorted(TT_bin.keys(), key=lambda x: int(x.replace("layer", "")))
    base_tensor_bin = torch.stack([TT_bin[layer] for layer in sorted_layers], dim=0)
    base_tensor_ter = torch.stack([TT_ter[layer] for layer in sorted_layers], dim=0)

    expected_bin = p_2 * m_2 / v**2
    expected_ter = p_3 * m_3 / v**3

    inference_tensors = {
        known_layers: tensors_for_known_layers(
            known_layers,
            base_tensor_bin,
            base_tensor_ter,
            expected_bin,
            expected_ter,
            L,
        )
        for known_layers in args.known_layers
    }

    if args.sample_seed is not None:
        random.seed(args.sample_seed)
        np.random.seed(args.sample_seed)
        torch.manual_seed(args.sample_seed)

    stats = {
        known_layers: {
            predictor: {
                "correct": [],
                "H": [],
                "opt_acc": [],
                "true_prob": [],
                "n_possible": [],
                "zero_Z": 0,
            }
            for predictor in predictors
        }
        for known_layers in args.known_layers
    }

    with open(sample_filename, "w", buffering=1) as sample_file:
        sample_file.write(
            "sample_idx\tpredictor\tknown_layers\tlatent\ttrue_span_i\t"
            "true_span_j\tcandidate_spans\ttrue_class\tpred_class\t"
            "correct\tH\topt_acc\ttrue_prob\tn_possible\n"
        )

        for sample_idx in range(args.N_samp):
            start_symbol = random.randint(1, n)
            seq, latent_levels = sample_ternary_with_latents(
                grammar_1, str(start_symbol), L
            )
            d = len(seq)
            if d != d_fixed:
                raise RuntimeError(f"Expected length {d_fixed}, got {d}")

            if args.latent == "token":
                target = last_token_target(seq, L, d, v)
            else:
                target = last_branch_target(
                    args.latent, latent_levels, L, d, v, args.span_readout
                )

            for known_layers, (tensor_bin, tensor_ter) in inference_tensors.items():
                _, inside_tensor_list = parallel_inside_masked(
                    seq,
                    tensor_bin,
                    tensor_ter,
                    L,
                    v,
                    mask_from=d,
                )
                _, outside_tensor_list = parallel_outside(
                    d,
                    tensor_bin,
                    tensor_ter,
                    L,
                    v,
                    inside_tensor_list,
                )

                inside = inside_tensor_list[target["inside_idx"]]
                outside = outside_tensor_list[target["outside_idx"]]
                for predictor in predictors:
                    distribution, n_possible = latent_readout_distribution(
                        inside,
                        outside,
                        target["candidate_spans"],
                        v,
                        predictor,
                    )

                    predictor_stats = stats[known_layers][predictor]
                    if distribution is None:
                        predictor_stats["zero_Z"] += 1
                        print(
                            "Warning: zero partition function for "
                            f"sample {sample_idx}, predictor={predictor}, "
                            f"known_layers={known_layers}"
                        )
                        continue

                    true_class = target["true_class"]
                    if predictor == "cyk":
                        pred_class = -1
                        correct = float(distribution[true_class].item())
                    else:
                        pred_class = int(torch.argmax(distribution).item())
                        correct = float(pred_class == true_class)
                    H = float(calculate_entropy(distribution) / np.log(v))
                    opt_acc = float(distribution.max().item())
                    true_prob = float(distribution[true_class].item())

                    predictor_stats["correct"].append(correct)
                    predictor_stats["H"].append(H)
                    predictor_stats["opt_acc"].append(opt_acc)
                    predictor_stats["true_prob"].append(true_prob)
                    predictor_stats["n_possible"].append(n_possible)

                    sample_file.write(
                        f"{sample_idx}\t{predictor}\t{known_layers}\t"
                        f"{args.latent}\t"
                        f"{target['true_span_i']}\t{target['true_span_j']}\t"
                        f"{';'.join(f'{i}:{j}' for i, j in target['candidate_spans'])}"
                        f"\t{true_class}\t{pred_class}\t{correct}\t{H}\t"
                        f"{opt_acc}\t{true_prob}\t{n_possible}\n"
                    )
                    sample_file.flush()

    time_2 = time.time()
    ct = (time_2 - time_1) / 60

    with open(avs_filename, "w") as avs_file:
        avs_file.write(
            "latent\tspan_readout\tpredictor\tknown_layers\taccuracy\tav_H\t"
            "av_opt_acc\tav_true_prob\tav_n_possible\tvalid_samples\t"
            "zero_partition_count\ttime_min\n"
        )

        for known_layers in args.known_layers:
            for predictor in predictors:
                predictor_stats = stats[known_layers][predictor]
                valid = len(predictor_stats["correct"])
                if valid:
                    accuracy = float(np.mean(predictor_stats["correct"]))
                    av_H = float(np.mean(predictor_stats["H"]))
                    av_opt_acc = float(np.mean(predictor_stats["opt_acc"]))
                    av_true_prob = float(np.mean(predictor_stats["true_prob"]))
                    av_n_possible = float(np.mean(predictor_stats["n_possible"]))
                else:
                    accuracy = av_H = av_opt_acc = av_true_prob = float("nan")
                    av_n_possible = float("nan")

                avs_file.write(
                    f"{args.latent}\t{args.span_readout}\t{predictor}\t"
                    f"{known_layers}\t{accuracy}\t{av_H}\t{av_opt_acc}\t"
                    f"{av_true_prob}\t{av_n_possible}\t{valid}\t"
                    f"{predictor_stats['zero_Z']}\t{ct}\n"
                )

                print(
                    f"latent={args.latent}, span_readout={args.span_readout}, "
                    f"predictor={predictor}, known_layers={known_layers}, "
                    f"accuracy={accuracy:.4f}, H={av_H:.4f}, "
                    f"opt_acc={av_opt_acc:.4f}, true_prob={av_true_prob:.4f}, "
                    f"n_possible={av_n_possible:.2f}, "
                    f"valid={valid}, zero_Z={predictor_stats['zero_Z']}"
                )

    print(f"time={ct:.2f} min")


if __name__ == "__main__":
    main()
