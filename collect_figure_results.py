#!/usr/bin/env python3
"""Locate and stage the *cached numerical results* required by paper plots.

Run from any directory, e.g.
  python collect_figure_results.py --root /work/pcsl/parley/runs --root /work/pcsl/parley/Last_token --dest ./figure_data

No pickle/torch files are loaded; this only examines filenames and copies bytes.
Ambiguous filenames are not copied; manually resolve their source first.
"""
import argparse
import csv
import os
import shutil
from pathlib import Path

TRAINING = [
    'run_L3_v8_bin2_ter16_P16777216_lr0.001_l20.01_bs128_depth6_demb_512_nheads_16_arr0.pt',
    'run_L3_v8_bin2_ter16_P16777216_lr0.001_l20.03_bs128_depth3_demb_768_nheads_16_arr0.pt',
]
FIXED = [f'probe_results_step_{t}_ternary_train_ternary_test_last_token_depth3.pkl'
         for t in (1024, 262144, 1048576)]
D6 = [f'probe_results_step_{t}_mixed_train_ternary_test_last_token.pkl'
      for t in (512, 1024, 16384, 262144, 2965824)]
D12 = [f'probe_results_step_{t}_mixed_train_ternary_test_last_token_depth12.pkl'
       for t in (512, 1024, 16384, 262144, 2097152)]
GROUPS = {'training': TRAINING, 'probe_fixed_d3': FIXED,
          'probe_varying_d6': D6, 'probe_varying_d12': D12}
EXPECTED = {name: group for group, names in GROUPS.items() for name in names}
SKIP_DIRS = {'.git', '.venv', '__pycache__', '.ipynb_checkpoints'}

def scan(roots):
    found = {name: [] for name in EXPECTED}
    for root in roots:
        root = root.expanduser().resolve()
        if not root.is_dir():
            print(f'WARNING: missing search root: {root}')
            continue
        print(f'Scanning {root} ...', flush=True)
        for base, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for filename in files:
                if filename in found:
                    path = Path(base) / filename
                    if path.is_file() and path not in found[filename]:
                        found[filename].append(path)
    return found

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', action='append', required=True, type=Path,
                    help='Search root (repeat for several locations)')
    ap.add_argument('--dest', type=Path, default=Path('figure_data'),
                    help='Destination directory with training/ and probes/ subdirectories')
    ap.add_argument('--dry-run', action='store_true', help='Report only; no copying')
    ap.add_argument('--max-mb', type=float, default=250,
                    help='Maximum individual file size to copy (MiB); default 250')
    args = ap.parse_args()
    found = scan(args.root)
    args.dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for group, names in GROUPS.items():
        for name in names:
            sources = found[name]
            target_dir = 'training' if group == 'training' else 'probes'
            dest = args.dest / target_dir / name
            status = 'missing' if not sources else 'ambiguous' if len(sources) > 1 else 'found'
            src = sources[0] if len(sources) == 1 else None
            mb = src.stat().st_size / (1024 * 1024) if src else None
            if src and mb > args.max_mb:
                status = 'too_large'
            elif src and not args.dry_run:
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    if dest.stat().st_size == src.stat().st_size:
                        status = 'already_present'
                    else:
                        status = 'destination_conflict'
                else:
                    shutil.copy2(src, dest)
                    status = 'copied'
            print(f'{status:>20} [{group}] {name}')
            if status == 'ambiguous':
                for p in sources:
                    print(f'      candidate: {p}')
            rows.append({'group': group, 'filename': name, 'status': status,
                         'source': str(src) if src else ' | '.join(map(str, sources)),
                         'destination': str(dest), 'size_mib': f'{mb:.2f}' if mb is not None else ''})
    manifest = args.dest / 'manifest.csv'
    with manifest.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nManifest: {manifest}')
    print(f'Found exactly once: {sum(len(v)==1 for v in found.values())}/{len(EXPECTED)}')
    print('Run plotting commands with --data-dir figure_data/training or figure_data/probes.')
    print('NOTE: matching filenames are not proof of matching experiment metadata; plots validate their contents.')

if __name__ == '__main__':
    main()
