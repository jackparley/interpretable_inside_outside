import sys
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
v = int(sys.argv[1])
L = int(sys.argv[2])
N_samp = int(sys.argv[3])
padded=int(3**L)

# Sweep f = 1/v, 2/v, ..., 1
l_bound = 1 / v
u_bound = 1
step_size = 1 / v
fs = np.arange(l_bound, u_bound + step_size, step_size)

n = v
eta = v
#eta=1e-8
Nreps = 5

# ------------------------------------------------------------
# Output folder
# ------------------------------------------------------------
out_dir = f"v_{v}_L_{L}_range"
os.makedirs(out_dir, exist_ok=True)

avs_filename = f"v_{v}_L_{L}_range_avs_padded.txt"
with open(avs_filename, "w") as file:
    pass

# ------------------------------------------------------------
# Accumulators for grand average over grammar realizations
# ------------------------------------------------------------
grand_H    = [[] for _ in range(len(fs))]
grand_N_cl = [[] for _ in range(len(fs))]
grand_opt  = [[] for _ in range(len(fs))]

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

        rules_1, rules_2 = generate_rules_double_new(
            v, n, m_2, m_3, p_2, p_3, L
        )

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

        # --------------------------------------------------------
        # Naive inside computation
        # --------------------------------------------------------
        for result in results:
            seq = result
            d = len(seq)
            #Non-parallel version
            #i_table = inside_algorithm_2_masked(seq, grammar_1, L, v)
            #o_table = outside_algorithm_2(seq, grammar_1, L, v, i_table)

            #Parallel version
            i_table_parallel,inside_tensor_list = parallel_inside_masked(seq, combined_tensor_bin, combined_tensor_ter, L, v, padded=int(3**L))
            #Compare i_table and i_table_parallel
            # i_table: (n+1, n+1, (L+1)*v) numpy — first v entries in dim 2 are the root non-terminals
            # i_table_parallel: (v, n+1, n+1) torch — the same v root non-terminals
            #i_table_root = torch.tensor(i_table[:, :, :v], dtype=torch.float32).permute(2, 0, 1)
            #if not torch.allclose(i_table_root, i_table_parallel.cpu(), atol=1e-4):
             #   print("Warning: i_table and i_table_parallel do not match")
            #else:
             #   print("i_table and i_table_parallel match")

            o_table_parallel,outside_tensor_list = parallel_outside(d, combined_tensor_bin, combined_tensor_ter, L, v,inside_tensor_list, padded=int(3**L))

            # Root layer assumed to be symbols 1, ..., v
            #ZZ = torch.tensor((1/v)*o_table[d-1, d, L * v : (L + 1) * v], dtype=torch.float64)
            ZZ_parallel = ((1/v)*o_table_parallel[:,d-1,d]).clone().detach().to(dtype=torch.float64)
            #Compare ZZ and ZZ_parallel
            #if not torch.allclose(ZZ, ZZ_parallel.cpu(), atol=1e-6):
             #   print(f"Warning: ZZ and ZZ_parallel do not match | ZZ.sum()={ZZ.sum():.6e} ZZ_parallel.sum()={ZZ_parallel.sum():.6e}")
            ZZ=ZZ_parallel.cpu()
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

        grand_H[kk].append(av_H)
        grand_N_cl[kk].append(av_N_cl)
        grand_opt[kk].append(av_opt_acc)

        print(
            f"rep={ppp}, f={f:.4f}, "
            f"H={av_H:.4f}, N_cl={av_N_cl:.2f}, "
            f"opt_acc={av_opt_acc:.4f}, time={ct:.2f} min"
        )

# ------------------------------------------------------------
# Grand average over grammar realizations
# ------------------------------------------------------------
grand_avs_filename = f"v_{v}_L_{L}_range_grand_avs_padded.txt"
with open(grand_avs_filename, "w") as file:
    for kk, f in enumerate(fs):
        if grand_H[kk]:
            g_H   = np.mean(grand_H[kk])
            g_N   = np.mean(grand_N_cl[kk])
            g_opt = np.mean(grand_opt[kk])
            file.write(f"{f}\t{g_H}\t{g_N}\t{g_opt}\n")

