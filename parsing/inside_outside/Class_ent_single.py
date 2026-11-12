import sys
import torch
import numpy as np
import random
import time
import os

from Inside_outside import (
    generate_rules_double,
    generate_grammar,
    generate_grammar_2,
    calculate_entropy,
    sample_from_grammar,
    extract_binary_rules_as_tensors,
    extract_ternary_rules_as_tensors,
    inside_algorithm_2,
)

# ------------------------------------------------------------
# Arguments
# ------------------------------------------------------------
v = int(sys.argv[1])
L = int(sys.argv[2])
N_samp = int(sys.argv[3])

# Sweep f = 1/v, 2/v, ..., 1
l_bound = 1 / v
u_bound = 1
step_size = 1 / v
fs = np.arange(l_bound, u_bound + step_size, step_size)

n = v
eta = v
Nreps = 1

# ------------------------------------------------------------
# Output folder
# ------------------------------------------------------------
out_dir = f"v_{v}_L_{L}_range"
os.makedirs(out_dir, exist_ok=True)

avs_filename = f"v_{v}_L_{L}_range_avs.txt"
with open(avs_filename, "w") as file:
    pass

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

        rules_1, rules_2 = generate_rules_double(
            v, n, m_2, m_3, p_2, p_3, L
        )

        grammar_1 = generate_grammar(rules_1)

        # Not needed for naive inside, but keeping these if you want
        # consistency with the old script / diagnostics later.
        grammar_2 = generate_grammar_2(rules_2)
        TT_bin = extract_binary_rules_as_tensors(grammar_2, v)
        TT_ter = extract_ternary_rules_as_tensors(grammar_2, v)

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

        # --------------------------------------------------------
        # Naive inside computation
        # --------------------------------------------------------
        for result in results:
            seq = result
            d = len(seq)

            i_table = inside_algorithm_2(seq, grammar_1, L, v)

            # Root layer assumed to be symbols 1, ..., v
            ZZ = torch.tensor(i_table[0, -1, :v], dtype=torch.float64)

            nonzero_count = (ZZ != 0).sum().item()

            if ZZ.sum() == 0:
                print("Warning: zero partition function")
                continue

            ZZ_c = ZZ / ZZ.sum()

            H = calculate_entropy(ZZ_c) / np.log(v)

            H_cls.append(H)
            N_cls.append(nonzero_count)
            opt_accs.append(ZZ_c.max().item())

            with open(sample_filename, "a") as file:
                file.write(f"{H}\t{d}\n")

        time_2 = time.time()
        ct = (time_2 - time_1) / 60

        av_H = sum(H_cls) / len(H_cls)
        av_N_cl = sum(N_cls) / len(N_cls)
        av_opt_acc = sum(opt_accs) / len(opt_accs)

        with open(avs_filename, "a") as file:
            file.write(f"{f}\t{av_H}\t{av_N_cl}\t{av_opt_acc}\t{ct}\n")

        print(
            f"rep={ppp}, f={f:.4f}, "
            f"H={av_H:.4f}, N_cl={av_N_cl:.2f}, "
            f"opt_acc={av_opt_acc:.4f}, time={ct:.2f} min"
        )