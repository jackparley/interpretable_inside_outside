#!/usr/bin/env python3
"""Re-evaluate autoregressive model checkpoints on ternary last-token completion.

Run from anywhere on the cluster, with the repository containing init.py on PYTHONPATH:
    python recompute_ternary_last_token.py --run-dir runs/example_experiment

Uses the training run's summary .pt for config and all matching _t<step>.pt
files for weights. Computes both regular cross-entropy and cross-entropy
conditioned on a non-EOS output (EOS/pad token excluded from softmax).
"""
import argparse
import copy
import math
import pickle
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--summary', type=Path, default=None,
                   help='Training summary .pt containing config; inferred if unique')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--eval-batch-size', type=int, default=None,
                   help='Defaults to training batch_size')
    p.add_argument('--output-prefix', type=Path, default=None)
    args = p.parse_args()

    # init.py should be importable from the working directory or repo root.
    import init

    run_dir = args.run_dir.expanduser().resolve()
    if args.summary is None:
        # Summaries have NO _t<integer> suffix. Do not confuse with checkpoints.
        summaries = sorted(f for f in run_dir.glob('*.pt')
                           if not re.search(r'_t\d+\.pt$', f.name))
        if len(summaries) != 1:
            raise RuntimeError(
                f'Found {len(summaries)} summary files in {run_dir}:\n'
                + '\n'.join(map(str, summaries))
                + '\nSupply --summary to choose the correct run.')
        summary_path = summaries[0]
    else:
        summary_path = args.summary.expanduser().resolve()

    summary = torch.load(summary_path, map_location='cpu', weights_only=False)
    if 'config' not in summary:
        raise ValueError(f'{summary_path} has no config; choose a training summary .pt')
    config = copy.deepcopy(summary['config'])
    config.device = args.device
    config.compile = False
    # Reestablish fields normally populated inside init_data_generated().
    config.vocab_size = config.v + 1
    config.pad_token_id = config.v
    config.block_size = 3 ** config.L - 1
    if args.eval_batch_size is not None:
        config.batch_size = args.eval_batch_size

    print('Using summary:', summary_path)
    print(f'Grammar L={config.L}, v={config.v}, bin={config.bin}, '
          f'ter={getattr(config, "ter", None)}, seed_rules={config.seed_rules}')
    print('Ternary validation examples:', config.ternary_val_size)

    # Produces EXACTLY the training code's fixed all-ternary validation set:
    # seed_sample_test+1, same grammar, same seed_rules, same rules.
    val_loader = init.init_ternary_val_loader_generated(config)
    if val_loader.num_batches <= 0:
        raise ValueError('Evaluation batch size is larger than validation set')

    # Construct once; overwrite model state at every checkpoint.
    model = init.init_model(config)
    model.eval()

    prefix = summary_path.stem
    pattern = re.compile(re.escape(prefix) + r'_t(\d+)\.pt$')
    checkpoints = []
    for f in run_dir.glob(prefix + '_t*.pt'):
        match = pattern.fullmatch(f.name)
        if match:
            checkpoints.append((int(match.group(1)), f))
    checkpoints.sort(key=lambda x: x[0])
    if not checkpoints:
        raise FileNotFoundError(f'No checkpoints matching {prefix}_t*.pt in {run_dir}')
    print(f'Found {len(checkpoints)} checkpoints')

    norm = math.log(config.v)
    output_rows = []
    for index, (step_from_name, path) in enumerate(checkpoints, 1):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        step = int(checkpoint.get('step', step_from_name))
        if step != step_from_name:
            raise ValueError(f'Filename step disagrees with checkpoint step: {path}')
        model.load_state_dict(checkpoint['model'], strict=True)
        model.eval()

        raw_sum = 0.0
        no_eos_sum = 0.0
        eos_mass_sum = 0.0
        num_examples = 0
        val_loader.reset()
        with torch.inference_mode():
            for _ in range(val_loader.num_batches):
                inputs, targets = val_loader.next_batch()
                inputs = inputs.to(args.device)
                targets = targets.to(args.device)
                if torch.any(targets == config.pad_token_id):
                    raise ValueError('Unexpected EOS in last-terminal targets')
                output = model(inputs)
                logits = output[:, -1, :].float()  # the 27th token prediction
                if logits.shape[1] != config.vocab_size:
                    raise ValueError('Unexpected model vocabulary size')

                # Cross-entropy with and without the EOS logit in denominator.
                target_logits = logits.gather(1, targets[:, None]).squeeze(1)
                lse_all = torch.logsumexp(logits, dim=-1)
                lse_real = torch.logsumexp(logits[:, :config.v], dim=-1)
                raw_sum += (lse_all - target_logits).sum().item()
                no_eos_sum += (lse_real - target_logits).sum().item()
                eos_mass_sum += torch.exp(logits[:, config.pad_token_id] - lse_all).sum().item()
                num_examples += len(targets)

        raw = raw_sum / num_examples
        no_eos = no_eos_sum / num_examples
        eos_correction = raw - no_eos
        result = {
            'step': step,
            'checkpoint': str(path),
            'n_examples': num_examples,
            'raw_loss': raw,
            'no_eos_loss': no_eos,
            'eos_correction': eos_correction,
            'mean_p_eos': eos_mass_sum / num_examples,
            'raw_loss_normalized': raw / norm,
            'no_eos_loss_normalized': no_eos / norm,
            'eos_correction_normalized': eos_correction / norm,
        }
        output_rows.append(result)
        print(f'[{index:>3}/{len(checkpoints)}] step {step:>9}: '
              f'raw/ln(v)={raw/norm:.6f}, '
              f'no-EOS/ln(v)={no_eos/norm:.6f}, '
              f'correction/ln(v)={eos_correction/norm:.6f}')
        del checkpoint

    out_prefix = args.output_prefix or (run_dir / (prefix + '_ternary_recomputed'))
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    with open(out_prefix.with_suffix('.pkl'), 'wb') as handle:
        pickle.dump({'config': vars(config), 'summary_path': str(summary_path),
                     'results': output_rows}, handle)

    steps = np.array([r['step'] for r in output_rows])
    raw_vals = np.array([r['raw_loss_normalized'] for r in output_rows])
    corrected_vals = np.array([r['no_eos_loss_normalized'] for r in output_rows])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(steps, raw_vals, 'o-', ms=3, label='Ternary last token (original)')
    ax.plot(steps, corrected_vals, 'o-', ms=3, label='Ternary last token (EOS excluded)')
    if np.any(steps > 0):
        ax.set_xscale('symlog', linthresh=1)  # includes t=0 checkpoint
    ax.set_xlabel('Training steps')
    ax.set_ylabel(r'Last-token loss / $\ln(v)$')
    ax.set_title('All-ternary validation')
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('\nSaved:', out_prefix.with_suffix('.pkl'))
    print('Saved:', out_prefix.with_suffix('.pdf'))


if __name__ == '__main__':
    main()
