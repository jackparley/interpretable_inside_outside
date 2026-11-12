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
            "Compute conditional entropy of the last token for one fixed grammar, "
            "stopping inside-outside at the highest known rule layer "
            f"(seed={RULE_SEED}, v={V}, f={F}, L={DEPTH})."
        )
    )
    parser.add_argument("N_samp", type=int, help="number of sampled sentences")
    parser.add_argument(
        "--eta-generation",
        type=float,
        default=0.0,
        help="eta used to sample sentences (default: 0, purely ternary)",
    )
    parser.add_argument(
        "--eta",
        "--eta-parsing",
        dest="eta_parsing",
        type=float,
        default=0.0,
        help="eta used on known rule layers (default: 0, purely ternary)",
    )
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
        "--known-layers",
        choices=("bottom", "bottom-intermediate"),
        default="bottom",
        help=(
            "highest available rule depth: bottom runs one inside-outside "
            "level; bottom-intermediate runs two (default: bottom)"
        ),
    )
    parser.add_argument(
        "--token-context",
        choices=("prefix", "patch"),
        default="prefix",
        help=(
            "token evidence available to the predictor: prefix observes the "
            "whole sentence except its last token; patch observes only the "
            "second-last and third-last tokens"
        ),
    )
    args = parser.parse_args()
    if args.eta_generation < 0 or args.eta_parsing < 0:
        parser.error("eta values must be non-negative")
    return args


def rule_probabilities(eta, m_2, m_3):
    p_3 = 1 / (m_2 * eta + m_3)
    p_2 = eta * p_3
    return p_2, p_3


def main():
    args = parse_args()

    v = V
    f = F
    L = DEPTH
    n = v
    eta_generation = args.eta_generation
    eta_parsing = args.eta_parsing

    m_2 = int(f * v)
    m_3 = int(f * v**2)
    p_2_generation, p_3_generation = rule_probabilities(
        eta_generation, m_2, m_3
    )
    p_2_parsing, p_3_parsing = rule_probabilities(eta_parsing, m_2, m_3)

    known_depth = 1 if args.known_layers == "bottom" else 2
    root_span_lengths = range(2**known_depth, 3**known_depth + 1)

    run_name = (
        f"v_{v}_L_{L}_f_{f:g}_seed_{RULE_SEED}_depth_restricted"
        f"_known_{args.known_layers}"
        f"_context_{args.token_context}"
        f"_eta_generation_{eta_generation:g}_eta_parsing_{eta_parsing:g}"
    )
    if args.out_dir is None:
        args.out_dir = run_name

    os.makedirs(args.out_dir, exist_ok=True)
    sample_filename = os.path.join(args.out_dir, "last_token_samples.txt")
    avs_filename = f"{run_name}_last_token_avs.txt"

    time_1 = time.time()

    # Generate the same seeded rule topology twice, but attach different
    # probabilities to it for data generation and parsing.
    generation_rules, _ = generate_rules_double_new(
        v,
        n,
        m_2,
        m_3,
        p_2_generation,
        p_3_generation,
        L,
        RULE_SEED,
    )
    _, parsing_rules = generate_rules_double_new(
        v, n, m_2, m_3, p_2_parsing, p_3_parsing, L, RULE_SEED
    )

    generation_grammar = generate_grammar(generation_rules)
    parsing_grammar = generate_grammar_2(parsing_rules)

    TT_bin = extract_binary_rules_as_tensors(parsing_grammar, v)
    TT_ter = extract_ternary_rules_as_tensors(parsing_grammar, v)
    sorted_layers = sorted(TT_bin.keys(), key=lambda x: int(x.replace("layer", "")))
    combined_tensor_bin = torch.stack([TT_bin[layer] for layer in sorted_layers])
    combined_tensor_ter = torch.stack([TT_ter[layer] for layer in sorted_layers])

    # Retain only the known suffix of the rule hierarchy.  The tensors remain
    # ordered from the highest retained layer to the bottom layer, which is the
    # ordering expected by the inside/outside routines at ``known_depth``.
    combined_tensor_bin = combined_tensor_bin[-known_depth:]
    combined_tensor_ter = combined_tensor_ter[-known_depth:]

    if args.sample_seed is not None:
        random.seed(args.sample_seed)
        np.random.seed(args.sample_seed)
        torch.manual_seed(args.sample_seed)

    H_cls = []
    N_cls = []
    opt_accs = []
    zero_partition_count = 0

    with open(sample_filename, "w", buffering=1) as sample_file:
        sample_file.write("H_last_nats\tH_last_normalized\td\tN_cl\topt_acc\n")

        for sample_idx in range(args.N_samp):
            start_symbol = random.randint(1, n)
            seq, _ = sample_from_grammar(
                generation_grammar,
                str(start_symbol),
                L,
                p_2_generation,
                p_3_generation,
            )
            d = len(seq)

            observed_positions = None
            if args.token_context == "patch":
                # With zero-based indexing these are 24 and 25 when d=27;
                # position 26 (the prediction target) remains masked.
                observed_positions = range(max(0, d - 3), d - 1)

            _, inside_tensor_list = parallel_inside_masked(
                seq,
                combined_tensor_bin,
                combined_tensor_ter,
                known_depth,
                v,
                observed_positions=observed_positions,
            )
            o_table_parallel, _ = parallel_outside(
                d,
                combined_tensor_bin,
                combined_tensor_ter,
                known_depth,
                v,
                inside_tensor_list,
                uniform_root_span_lengths=root_span_lengths,
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
            H_nats = float(calculate_entropy(ZZ_c))
            H = H_nats / np.log(v)
            nonzero_count = int((ZZ != 0).sum().item())
            opt_acc = float(ZZ_c.max().item())

            H_cls.append(H)
            N_cls.append(nonzero_count)
            opt_accs.append(opt_acc)
            sample_file.write(
                f"{H_nats}\t{H}\t{d}\t{nonzero_count}\t{opt_acc}\n"
            )

    ct = (time.time() - time_1) / 60
    if H_cls:
        av_H = float(np.mean(H_cls))
        av_N_cl = float(np.mean(N_cls))
        av_opt_acc = float(np.mean(opt_accs))
    else:
        av_H = av_N_cl = av_opt_acc = float("nan")
    av_H_nats = av_H * np.log(v)

    with open(avs_filename, "w") as avs_file:
        avs_file.write(
            "f\tknown_layers\tknown_depth\troot_span_lengths\ttoken_context\t"
            "eta_generation\teta_parsing\tav_H_last_nats\tav_H_last_normalized\t"
            "av_N_cl\tav_opt_acc\ttime_min\tvalid_samples\tzero_partition_count\n"
        )
        avs_file.write(
            f"{f}\t{args.known_layers}\t{known_depth}\t"
            f"{2**known_depth}-{3**known_depth}\t{args.token_context}\t{eta_generation}\t"
            f"{eta_parsing}\t{av_H_nats}\t{av_H}\t"
            f"{av_N_cl}\t{av_opt_acc}\t{ct}\t{len(H_cls)}\t{zero_partition_count}\n"
        )

    print(
        f"seed={RULE_SEED}, v={v}, f={f:.4f}, L={L}, "
        f"known_layers={args.known_layers}, known_depth={known_depth}, "
        f"root_spans={2**known_depth}-{3**known_depth}, "
        f"eta_generation={eta_generation:g}, "
        f"token_context={args.token_context}, eta_parsing={eta_parsing:g}, "
        f"H_last_nats={av_H_nats:.4f}, H_last_normalized={av_H:.4f}, "
        f"N_cl={av_N_cl:.2f}, "
        f"opt_acc={av_opt_acc:.4f}, valid={len(H_cls)}, "
        f"zero_Z={zero_partition_count}, time={ct:.2f} min"
    )


if __name__ == "__main__":
    main()
