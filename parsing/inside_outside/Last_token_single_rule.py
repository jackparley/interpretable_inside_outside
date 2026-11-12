import argparse
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    parser.add_argument("N_samp", type=int, nargs="?", default=10000,
                        help="maximum sampled sentences (default: 10000)")
    parser.add_argument("--report-every", type=int, default=100,
                        help="refresh convergence data and plot every N attempts (default: 100)")
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="optional seed for sentence sampling",
    )
    parser.add_argument(
        "--out-dir",
        default=f"v_{V}_L_{DEPTH}_f_{F:g}_seed_{RULE_SEED}",
        help="directory for per-sentence output",
    )
    args = parser.parse_args()
    if args.N_samp < 1 or args.report_every < 1:
        parser.error("N_samp and --report-every must be positive")
    return args


def main():
    args = parse_args()

    v = V
    f = F
    L = DEPTH
    n = v
    eta = v
    padded = int(3**L)

    m_2 = int(f * v)
    m_3 = int(f * v**2)

    p_3 = 1 / (m_2 * eta + m_3)
    p_2 = eta * p_3

    os.makedirs(args.out_dir, exist_ok=True)
    sample_filename = os.path.join(args.out_dir, "last_token_samples.txt")
    avs_filename = os.path.join(args.out_dir, "last_token_avs.txt")

    convergence_filename = os.path.join(args.out_dir, "last_token_convergence.tsv")
    plot_filename = os.path.join(args.out_dir, "last_token_convergence.png")
    if os.path.exists(sample_filename):
        raise FileExistsError(f"{sample_filename} already exists; choose a new --out-dir")

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

    if args.sample_seed is not None:
        random.seed(args.sample_seed)
        np.random.seed(args.sample_seed)
        torch.manual_seed(args.sample_seed)

    H_cls = []
    N_cls = []
    opt_accs = []
    zero_partition_count = 0

    attempted = 0
    checkpoints = []

    def report():
        # Uncertainty is over sampled sentences, conditional on this fixed grammar.
        count = len(H_cls)
        mean = float(np.mean(H_cls)) if count else float("nan")
        se = float(np.std(H_cls, ddof=1) / np.sqrt(count)) if count > 1 else float("nan")
        half_width = 1.96 * se
        previous_count = checkpoints[-1][1] if checkpoints else 0
        batch = H_cls[previous_count:]
        batch_mean = float(np.mean(batch)) if batch else float("nan")
        row = (attempted, count, zero_partition_count, mean, se,
               mean - half_width, mean + half_width, batch_mean,
               (time.time() - time_1) / 60)
        checkpoints.append(row)
        temporary = convergence_filename + ".tmp"
        with open(temporary, "w") as output:
            output.write("attempted_samples\tvalid_samples\tzero_partition_count\tmean_H_last\t"
                         "standard_error\tci95_low\tci95_high\tbatch_mean_H_last\ttime_min\n")
            for checkpoint in checkpoints:
                output.write("\t".join(map(str, checkpoint)) + "\n")
        os.replace(temporary, convergence_filename)
        values = np.asarray(checkpoints)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(values[:, 1], values[:, 3], label="Running mean")
        ax.fill_between(values[:, 1], values[:, 5], values[:, 6], alpha=0.25,
                        label="Approx. 95% Monte Carlo interval")
        ax.plot(values[:, 1], values[:, 7], ".", alpha=0.5, label="New batch mean")
        ax.set(xlabel="Valid sampled sentences", ylabel="H(last token | prefix) / log(v)",
               title=f"Fixed grammar seed={RULE_SEED}; {count} valid samples")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_filename + ".tmp", format="png", dpi=150)
        plt.close(fig)
        os.replace(plot_filename + ".tmp", plot_filename)
        print(f"{attempted}/{args.N_samp} attempts, valid={count}, zero_Z={zero_partition_count}, "
              f"mean H={mean:.6f}, approx. 95% half-width={half_width:.6f}, "
              f"new batch mean={batch_mean:.6f}, elapsed={row[-1]:.2f} min", flush=True)

    with open(sample_filename, "x", buffering=1) as sample_file:
        sample_file.write("H_last\td\tN_cl\topt_acc\n")

        try:
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

                attempted = sample_idx + 1
                if ZZ_sum == 0:
                    zero_partition_count += 1
                    print(f"Warning: zero partition function for sample {sample_idx}")
                    if attempted % args.report_every == 0:
                        report()
                    continue

                ZZ_c = ZZ / ZZ_sum
                H = float(calculate_entropy(ZZ_c) / np.log(v))
                nonzero_count = int((ZZ != 0).sum().item())
                opt_acc = float(ZZ_c.max().item())

                H_cls.append(H)
                N_cls.append(nonzero_count)
                opt_accs.append(opt_acc)

                sample_file.write(f"{H}\t{d}\t{nonzero_count}\t{opt_acc}\n")

                if attempted % args.report_every == 0:
                    report()
        except KeyboardInterrupt:
            print("\nStopped by user; saving completed samples.", flush=True)
        finally:
            if not checkpoints or checkpoints[-1][0] != attempted:
                report()

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
            "f\tav_H_last\tav_N_cl\tav_opt_acc\ttime_min\t"
            "valid_samples\tzero_partition_count\n"
        )
        avs_file.write(
            f"{f}\t{av_H}\t{av_N_cl}\t{av_opt_acc}\t{ct}\t"
            f"{len(H_cls)}\t{zero_partition_count}\n"
        )

    print(
        f"seed={RULE_SEED}, v={v}, f={f:.4f}, L={L}, "
        f"H_last={av_H:.4f}, N_cl={av_N_cl:.2f}, "
        f"opt_acc={av_opt_acc:.4f}, valid={len(H_cls)}, "
        f"zero_Z={zero_partition_count}, time={ct:.2f} min"
    )


if __name__ == "__main__":
    main()
