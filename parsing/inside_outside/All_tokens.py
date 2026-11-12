import argparse
import torch
import numpy as np
import random
import time
import os

from Inside_outside import (
    generate_rules_double,
    generate_rules_double_new,
    generate_grammar,
    generate_grammar_2,
    calculate_entropy,
    sample_from_grammar,
    extract_binary_rules_as_tensors,
    extract_ternary_rules_as_tensors,
    inside_algorithm_2_masked,
    outside_algorithm_2,
    parallel_inside_masked,
    parallel_outside,
)

# ------------------------------------------------------------
# Arguments
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute normalized conditional entropy at every token position. "
            "By default, use the original f sweep and five grammar realizations."
        )
    )
    parser.add_argument("v", type=int, help="number of symbols per layer")
    parser.add_argument("L", type=int, help="grammar depth")
    parser.add_argument("N_samp", type=int, help="number of sentences per grammar")
    parser.add_argument(
        "--f",
        type=float,
        default=None,
        help="run only this rule density (for example, --f 0.25)",
    )
    parser.add_argument(
        "--seed-rules",
        type=int,
        default=None,
        help="seed for one fixed realization of the grammar rules",
    )
    parser.add_argument(
        "--n-reps",
        type=int,
        default=None,
        help=(
            "number of grammar repetitions; defaults to 1 with --seed-rules "
            "and 5 otherwise"
        ),
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="optional seed for sentence sampling",
    )
    return parser.parse_args()


args = parse_args()
v = args.v
L = args.L
N_samp = args.N_samp

if args.f is not None and not (0 < args.f <= 1):
    raise ValueError("--f must be in the interval (0, 1]")
if args.n_reps is not None and args.n_reps < 1:
    raise ValueError("--n-reps must be at least 1")

# Sweep f = 1/v, 2/v, ..., 1
l_bound = 1 / v
u_bound = 1
step_size = 1 / v
#fs = np.arange(l_bound, u_bound + step_size, step_size)
fs = np.array([args.f]) if args.f is not None else np.array([0.125, 0.25])

n = v
#eta = v
eta=1e-8
Nreps = args.n_reps if args.n_reps is not None else (1 if args.seed_rules is not None else 5)

if args.sample_seed is not None:
    random.seed(args.sample_seed)
    np.random.seed(args.sample_seed)
    torch.manual_seed(args.sample_seed)

seed_suffix = f"_seed_{args.seed_rules}" if args.seed_rules is not None else ""

# ------------------------------------------------------------
# Output folder
# ------------------------------------------------------------
out_dir = f"v_{v}_L_{L}{seed_suffix}_range"
os.makedirs(out_dir, exist_ok=True)

avs_filename = f"v_{v}_L_{L}{seed_suffix}_range_avs.txt"
with open(avs_filename, "w") as file:
    pass

# ------------------------------------------------------------
# Accumulators for grand average over grammar realizations
# grand_H_by_n[kk][n] = list of H values for f index kk, n steps from end
# ------------------------------------------------------------
from collections import defaultdict
grand_H_by_n = [defaultdict(list) for _ in range(len(fs))]

# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------
for ppp in range(Nreps):
    for kk, f in enumerate(fs):

        jj = kk + 1

        sample_filename = f"{out_dir}/file_{jj}.txt"
        with open(sample_filename, "w") as file:
            pass

        time_1 = time.time()

        m_2 = int(f * v)
        m_3 = int(f * v**2)

        p_3 = 1 / (m_2 * eta + m_3)
        p_2 = eta * p_3

        if args.seed_rules is None:
            rules_1, rules_2 = generate_rules_double(
                v, n, m_2, m_3, p_2, p_3, L
            )
        else:
            sampling_rng_state = random.getstate()
            rules_1, rules_2 = generate_rules_double_new(
                v, n, m_2, m_3, p_2, p_3, L, args.seed_rules
            )
            # The seeded rule generator uses Python's global RNG. Restore the
            # sampling stream so rule generation and sampling are independent.
            random.setstate(sampling_rng_state)

        grammar_1 = generate_grammar(rules_1)

        # Not needed for naive inside, but keeping these if you want
        # consistency with the old script / diagnostics later.
        grammar_2 = generate_grammar_2(rules_2)
        TT_bin = extract_binary_rules_as_tensors(grammar_2, v)
        TT_ter = extract_ternary_rules_as_tensors(grammar_2, v)

        sorted_layers = sorted(TT_bin.keys(), key=lambda x: int(x.replace("layer", "")))
        combined_tensor_bin = torch.stack([TT_bin[layer] for layer in sorted_layers], dim=0)
        combined_tensor_ter = torch.stack([TT_ter[layer] for layer in sorted_layers], dim=0)
        # --------------------------------------------------------
        # Sample sentences
        # --------------------------------------------------------
        results = []
        for _ in range(N_samp):
            start_symbol = random.randint(1, n)
            result, prob_tree = sample_from_grammar(
                grammar_1, str(start_symbol), L, p_2, p_3
            )
            results.append(result)

        H_cls = []
        N_cls = []
        opt_accs = []
        # H_by_pos[(n_pos, t)] = list of H values for this (f, rep)
        # n_pos: steps from end (0=last token), t: position from start (0=first token)
        H_by_pos = defaultdict(list)

        # --------------------------------------------------------
        # Naive inside computation
        # --------------------------------------------------------
        for result in results:
            seq = result
            d = len(seq)

            for n_pos in range(d):  # n_pos=0: last token, n_pos=d-1: first token
                t = d - 1 - n_pos   # t=0: first token, t=d-1: last token
                mask_from = d - n_pos  # 1-indexed: mask from here to end

                i_table_parallel, inside_tensor_list = parallel_inside_masked(
                    seq, combined_tensor_bin, combined_tensor_ter, L, v,
                    mask_from=mask_from
                )
                o_table_parallel, outside_tensor_list = parallel_outside(
                    d, combined_tensor_bin, combined_tensor_ter, L, v, inside_tensor_list
                )

                # Marginal at the first masked position: span (mask_from-1, mask_from)
                ZZ = (
                    (1 / v) * o_table_parallel[:, mask_from - 1, mask_from]
                ).clone().detach().to(dtype=torch.float64).cpu()

                if ZZ.sum() == 0:
                    continue

                ZZ_c = ZZ / ZZ.sum()
                H = calculate_entropy(ZZ_c) / np.log(v)
                H_by_pos[(n_pos, t)].append(H)

                # Keep the n_pos=0 (last token) stats for backward-compat output
                if n_pos == 0:
                    nonzero_count = (ZZ != 0).sum().item()
                    H_cls.append(H)
                    N_cls.append(nonzero_count)
                    opt_accs.append(ZZ_c.max().item())

                    with open(sample_filename, "a") as file:
                        file.write(f"{H}\t{d}\n")

        time_2 = time.time()
        ct = (time_2 - time_1) / 60

        if H_cls:
            av_H = sum(H_cls) / len(H_cls)
            av_N_cl = sum(N_cls) / len(N_cls)
            av_opt_acc = sum(opt_accs) / len(opt_accs)
        else:
            av_H = av_N_cl = av_opt_acc = float('nan')

        with open(avs_filename, "a") as file:
            file.write(f"{f}\t{av_H}\t{av_N_cl}\t{av_opt_acc}\t{ct}\n")

        # Accumulate per-position averages for grand average
        for (n_pos, t), h_vals in H_by_pos.items():
            grand_H_by_n[kk][(n_pos, t)].extend(h_vals)

        print(
            f"rep={ppp}, f={f:.4f}, "
            f"H_last={av_H:.4f}, time={ct:.2f} min"
        )

# ------------------------------------------------------------
# Grand average over grammar realizations: one file per f
# Rows: n_pos  mean_H  count
# Also write a combined file: f  n_pos  mean_H
# ------------------------------------------------------------
combined_filename = f"v_{v}_L_{L}{seed_suffix}_all_tokens_grand_avs.txt"
with open(combined_filename, "w") as combined_file:
    for kk, f in enumerate(fs):
        per_f_filename = (
            f"v_{v}_L_{L}_f{f:.4f}{seed_suffix}_all_tokens_avs.txt"
        )
        with open(per_f_filename, "w") as per_f_file:
            for (n_pos, t) in sorted(grand_H_by_n[kk].keys()):
                h_vals = grand_H_by_n[kk][(n_pos, t)]
                if h_vals:
                    mean_H = np.mean(h_vals)
                    count = len(h_vals)
                    per_f_file.write(f"{n_pos}\t{t}\t{mean_H}\t{count}\n")
                    combined_file.write(f"{f}\t{n_pos}\t{t}\t{mean_H}\t{count}\n")
