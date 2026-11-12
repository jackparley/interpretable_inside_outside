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
    sample_from_grammar,
)


V = 8
F = 1 / 4
DEPTH = 3
RULE_SEED = 12345678


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute conditional entropy of the last token for one fixed grammar "
            f"(seed={RULE_SEED}, v={V}, f={F}, L={DEPTH})."
        )
    )
    parser.add_argument("N_samp", type=int, help="number of sampled sentences")
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="optional seed for sentence sampling",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="directory for per-sentence output",
    )
    parser.add_argument(
        "--top-layer",
        choices=("original", "expected"),
        default="original",
        help=(
            "use the sampled top-layer rules, or replace the top layer by its "
            "expectation under the random grammar ensemble; kept for backward "
            "compatibility with previous runs"
        ),
    )
    parser.add_argument(
        "--known-layers",
        choices=("all", "bottom-intermediate", "bottom", "bottom-binary"),
        default=None,
        help=(
            "which rule layers are available to the predictor: all keeps every "
            "sampled layer, bottom-intermediate replaces only the top layer by "
            "its expectation, and bottom replaces the top and intermediate "
            "layers by their expectations; bottom-binary also replaces the "
            "bottom ternary rules by their expectation"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    v = V
    f = F
    L = DEPTH
    n = v
    eta = v
    #eta=1e-10
    padded = int(3**L)

    m_2 = int(f * v)
    m_3 = int(f * v**2)

    p_3 = 1 / (m_2 * eta + m_3)
    p_2 = eta * p_3

    if args.known_layers is None:
        args.known_layers = (
            "bottom-intermediate" if args.top_layer == "expected" else "all"
        )

    if args.out_dir is None:
        args.out_dir = (
            f"v_{v}_L_{L}_f_{f:g}_seed_{RULE_SEED}_known_{args.known_layers}_eta_{eta}"
        )

    os.makedirs(args.out_dir, exist_ok=True)
    sample_filename = os.path.join(args.out_dir, "last_token_samples.txt")
    avs_filename = (
        f"v_{v}_L_{L}_f_{f:g}_seed_{RULE_SEED}_known_{args.known_layers}_eta_{eta}"
        "_last_token_avs.txt"
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
    combined_tensor_bin = torch.stack([TT_bin[layer] for layer in sorted_layers], dim=0)
    combined_tensor_ter = torch.stack([TT_ter[layer] for layer in sorted_layers], dim=0)

    expected_bin = p_2 * m_2 / v**2
    expected_ter = p_3 * m_3 / v**3
    if args.known_layers == "all":
        expected_bin_layer_indices = []
        expected_ter_layer_indices = []
    elif args.known_layers == "bottom-intermediate":
        expected_bin_layer_indices = [0]
        expected_ter_layer_indices = [0]
    elif args.known_layers == "bottom":
        expected_bin_layer_indices = list(range(L - 1))
        expected_ter_layer_indices = list(range(L - 1))
    else:
        expected_bin_layer_indices = list(range(L - 1))
        expected_ter_layer_indices = list(range(L))

    for layer_idx in expected_bin_layer_indices:
        combined_tensor_bin[layer_idx] = torch.full_like(
            combined_tensor_bin[layer_idx], expected_bin
        )

    for layer_idx in expected_ter_layer_indices:
        combined_tensor_ter[layer_idx] = torch.full_like(
            combined_tensor_ter[layer_idx], expected_ter
        )

    if args.sample_seed is not None:
        random.seed(args.sample_seed)
        np.random.seed(args.sample_seed)
        torch.manual_seed(args.sample_seed)

    H_cls = []
    N_cls = []
    opt_accs = []
    zero_partition_count = 0

    with open(sample_filename, "w", buffering=1) as sample_file:
        sample_file.write("H_last\td\tN_cl\topt_acc\n")

        for sample_idx in range(args.N_samp):
            start_symbol = random.randint(1, n)
            seq, _ = sample_from_grammar(grammar_1, str(start_symbol), L, p_2, p_3)
            d = len(seq)

            _, inside_tensor_list = parallel_inside_masked(
                seq,
                combined_tensor_bin,
                combined_tensor_ter,
                L,
                v,
                padded=padded,
            )
            o_table_parallel, _ = parallel_outside(
                d,
                combined_tensor_bin,
                combined_tensor_ter,
                L,
                v,
                inside_tensor_list,
                padded=padded,
            )

            ZZ = ((1 / v) * o_table_parallel[:, d - 1, d]).detach().to(
                dtype=torch.float64, device="cpu"
            )
            ZZ_sum = ZZ.sum()

            if ZZ_sum == 0:
                zero_partition_count += 1
                print(f"Warning: zero partition function for sample {sample_idx}")
                continue

            ZZ_c = ZZ / ZZ_sum
            H = float(calculate_entropy(ZZ_c) / np.log(v))
            nonzero_count = int((ZZ != 0).sum().item())
            opt_acc = float(ZZ_c.max().item())

            H_cls.append(H)
            N_cls.append(nonzero_count)
            opt_accs.append(opt_acc)

            sample_file.write(f"{H}\t{d}\t{nonzero_count}\t{opt_acc}\n")
            sample_file.flush()

    time_2 = time.time()
    ct = (time_2 - time_1) / 60

    if H_cls:
        av_H = float(np.mean(H_cls))
        av_N_cl = float(np.mean(N_cls))
        av_opt_acc = float(np.mean(opt_accs))
    else:
        av_H = av_N_cl = av_opt_acc = float("nan")

    with open(avs_filename, "w") as avs_file:
        avs_file.write(
            "f\tknown_layers\tav_H_last\tav_N_cl\tav_opt_acc\ttime_min\t"
            "valid_samples\tzero_partition_count\n"
        )
        avs_file.write(
            f"{f}\t{args.known_layers}\t{av_H}\t{av_N_cl}\t{av_opt_acc}\t{ct}\t"
            f"{len(H_cls)}\t{zero_partition_count}\n"
        )

    print(
        f"seed={RULE_SEED}, v={v}, f={f:.4f}, L={L}, "
        f"known_layers={args.known_layers}, "
        f"H_last={av_H:.4f}, N_cl={av_N_cl:.2f}, "
        f"opt_acc={av_opt_acc:.4f}, valid={len(H_cls)}, "
        f"zero_Z={zero_partition_count}, time={ct:.2f} min"
    )


if __name__ == "__main__":
    main()
