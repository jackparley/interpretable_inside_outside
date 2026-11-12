import os
import sys
import time
import copy
sys.path.append('~/SLM')

import torch

torch.set_float32_matmul_precision("high")
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import math
import random

import functools
import argparse

import models, init, measures, train


def _loader_state(loader):
    """Return the mutable iterator state needed to resume the next batch."""
    keys = ('current_batch', 'total_batches', '_sentence_idx', '_buffer')
    return {key: copy.deepcopy(getattr(loader, key))
            for key in keys if hasattr(loader, key)}


def _restore_loader_state(loader, state):
    for key, value in (state or {}).items():
        if hasattr(loader, key):
            setattr(loader, key, value)


def _rng_state():
    state = {
        'torch': torch.get_rng_state(),
        'numpy': np.random.get_state(),
        'python': random.getstate(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    if not state:
        return
    torch.set_rng_state(state['torch'].cpu())
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])
    if 'cuda' in state and torch.cuda.is_available():
        cuda_states = []
        for s in state['cuda']:
            if not isinstance(s, torch.Tensor):
                s = torch.as_tensor(s)
            # CUDA RNG restore expects CPU byte tensors.
            cuda_states.append(s.detach().to(device='cpu', dtype=torch.uint8))
        torch.cuda.set_rng_state_all(cuda_states)


def _save_checkpoint(path, model, optimizer, scheduler, train_loader, dynamics, best, step):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'train_loader': _loader_state(train_loader),
        'rng_state': _rng_state(),
        'state': dynamics[-1],
        'dynamics': dynamics,
        'best': best,
        'step': step,
    }, path)


def run( config):

    ternary_train = getattr(config, 'ternary_train', False)
    if ternary_train and not config.generate_data:
        raise ValueError("--ternary_train requires --generate_data")
    # In ternary-training mode the primary validation loader is already
    # all-ternary, so a second all-ternary loader/metric would be redundant.
    extra_ternary_val = config.ternary_val and not ternary_train
    if ternary_train and config.ternary_val:
        print("--ternary_val is redundant with --ternary_train; using val_loss for the all-ternary validation set")

    if config.generate_data:
        config.eta = config.v if getattr(config, 'eta', None) is None else config.eta
        if not math.isfinite(config.eta) or config.eta < 0:
            raise ValueError("--eta must be finite and non-negative")
        ter = config.ter if config.ter is not None else config.bin * config.v
        print(f"Generating RHM data (v={config.v}, bin={config.bin}, ter={ter}, eta={config.eta}, L={config.L}) ...")
        tokenizer, train_loader, val_loader = init.init_data_generated(config)
    else:
        dataset_id = config.dataset if config.dataset is not None else config.name_train_file
        print(f"Loading {config.path+dataset_id} ... ")
        tokenizer, train_loader, val_loader = init.init_data(config)

    if ternary_train and getattr(config, 'objective', 'last_token') == 'autoregressive':
        # The primary validation data are ternary, but init selected a packed
        # LM loader. Reuse those fixed sentences for last-symbol completion.
        val_loader = init.MySentenceDataLoader(
            config.batch_size, config.block_size,
            val_loader._data, config.pad_token_id,
        )

    NUM_TOPO = 12
    if config.topo_val and not config.generate_data:
        topo_val_loaders = init.init_topo_val_loaders(config, num_topo=NUM_TOPO)

    if extra_ternary_val:
        if config.generate_data:
            ternary_val_loader = init.init_ternary_val_loader_generated(config)
        else:
            ternary_val_loader = init.init_ternary_val_loader(config)

    if config.completion_val:
        if config.generate_data:
            completion_val_loader = init.init_completion_val_loader_generated(config)
        else:
            completion_val_loader = init.init_completion_val_loader(config)

    if config.measure_train:
        extrain_loader = copy.deepcopy(train_loader)

    model = init.init_model( config)
    #best_model = copy.deepcopy( model).to('cpu')

    criterion, optimizer, scheduler = init.init_training( model, config)

    def record_completion_losses(entry, loader, key):
        full, no_eos = measures.evaluate_completion_losses(
            model, loader, loader.num_batches, config.device,
        )
        entry[key] = full
        entry[key + '_no_eos'] = no_eos
        print(f"{key}: {full}, {key}_no_eos: {no_eos}")

    start_step = 0
    dynamics = []
    if config.resume_from is not None:
        checkpoint = torch.load(config.resume_from, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'])
        if 'optimizer' not in checkpoint:
            raise ValueError(
                f"Checkpoint {config.resume_from} has no optimizer state and cannot be resumed exactly"
            )
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        _restore_loader_state(train_loader, checkpoint.get('train_loader'))
        _restore_rng_state(checkpoint.get('rng_state'))
        start_step = checkpoint['step']
        dynamics = checkpoint.get('dynamics', [checkpoint['state']])
        val_loss = checkpoint['state']['val_loss']
        best = checkpoint.get('best', {'step': start_step, 'loss': val_loss})
        print(f"Resuming {config.resume_from} from step {start_step}")
    else:
        val_loss = measures.evaluate(model, criterion, val_loader, val_loader.num_batches, config.device)
        entry0 = {'t': 0, 'running': math.log(config.vocab_size), 'val_loss': val_loss}
        if ternary_train:
            record_completion_losses(entry0, val_loader, 'val_loss')
        if config.topo_val and not config.generate_data:
            topo_losses_0 = [measures.evaluate(model, criterion, tl, tl.num_batches, config.device)
                             for tl in topo_val_loaders]
            entry0['topo_val_losses'] = topo_losses_0
        if extra_ternary_val:
            record_completion_losses(entry0, ternary_val_loader, 'ternary_val_loss')
        if config.completion_val:
            entry0['completion_val_loss'] = measures.evaluate(model, criterion, completion_val_loader, completion_val_loader.num_batches, config.device)
        dynamics.append(entry0)
        if config.measure_train:
            train_loss = measures.evaluate(model, criterion, extrain_loader, extrain_loader.num_batches, config.device)
            dynamics[0]['train_loss'] = train_loss
        best = {'step':0, 'loss': val_loss}

        if config.checkpoints:
            _save_checkpoint(f"{config.outname}_t0.pt", model, optimizer, scheduler,
                             train_loader, dynamics, best, 0)

    if start_step >= config.max_steps:
        print(f"Checkpoint is already at step {start_step} (max_steps={config.max_steps}); nothing to do")
        return None

    print_ckpts, save_ckpts = init.init_loglinckpt( config.print_freq, config.max_steps, freq=config.save_freq)
    print_ckpt = next(print_ckpts)
    save_ckpt = next(save_ckpts)
    while print_ckpt <= start_step:
        print_ckpt = next(print_ckpts)
    while save_ckpt <= start_step:
        save_ckpt = next(save_ckpts)

    print("Validation loss at start: ", val_loss, f"(compare with {math.log(config.vocab_size)})")

    for step in range(start_step, config.max_steps):

        t0 = time.time()
        loss = train.train_step(model, train_loader, criterion, optimizer, scheduler, config)
        t1 = time.time()
        dt = (t1-t0)*1000

        if (step+1)==print_ckpt:

            val_loss = measures.evaluate(model, criterion, val_loader, val_loader.num_batches, config.device)
            if val_loss<best['loss']:   # update best model if loss is smaller
                best['step'] = step
                best['loss'] = val_loss
                #best_model = copy.deepcopy( model).to('cpu')
                #best['model'] = best_model.state_dict()

            print(f'step {step+1}, running loss {loss}, dt={dt:.2f}ms, validation loss {val_loss}')
            print_ckpt = next(print_ckpts)

            if (step+1)>=save_ckpt:

                print(f'Evaluating at {step+1} ...')
                entry = {'t': step+1, 'running': loss, 'val_loss': val_loss}
                if ternary_train:
                    record_completion_losses(entry, val_loader, 'val_loss')
                if config.topo_val and not config.generate_data:
                    topo_losses = [measures.evaluate(model, criterion, tl, tl.num_batches, config.device)
                                   for tl in topo_val_loaders]
                    entry['topo_val_losses'] = topo_losses
                if extra_ternary_val:
                    record_completion_losses(entry, ternary_val_loader, 'ternary_val_loss')
                if config.completion_val:
                    entry['completion_val_loss'] = measures.evaluate(model, criterion, completion_val_loader, completion_val_loader.num_batches, config.device)
                dynamics.append(entry)

                if config.measure_train:
                    train_loss = measures.evaluate(model, criterion, extrain_loader, extrain_loader.num_batches, config.device)
                    dynamics[-1]['train_loss'] = train_loss
                if config.checkpoints:
                    _save_checkpoint(f"{config.outname}_t{step+1}.pt", model,
                                     optimizer, scheduler, train_loader,
                                     dynamics, best, step+1)

                torch.save(
                    {'config': config, 'dynamics': dynamics, 'best':  best},
                    f"{config.outname}.pt"
                )

                save_ckpt = next(save_ckpts)

    if (step+1)<save_ckpt:

        print(f'Evaluating at {step+1} ...')
        entry = {'t': step+1, 'running': loss, 'val_loss': val_loss}
        if ternary_train:
            record_completion_losses(entry, val_loader, 'val_loss')
        if config.topo_val and not config.generate_data:
            topo_losses = [measures.evaluate(model, criterion, tl, tl.num_batches, config.device)
                           for tl in topo_val_loaders]
            entry['topo_val_losses'] = topo_losses
        if extra_ternary_val:
            record_completion_losses(entry, ternary_val_loader, 'ternary_val_loss')
        if config.completion_val:
            entry['completion_val_loss'] = measures.evaluate(model, criterion, completion_val_loader, completion_val_loader.num_batches, config.device)
        dynamics.append(entry)

        if config.measure_train:
            train_loss = measures.evaluate(model, criterion, extrain_loader, extrain_loader.num_batches, config.device)
            dynamics[-1]['train_loss'] = train_loss
        if config.checkpoints:
            _save_checkpoint(f"{config.outname}_t{step+1}.pt", model,
                             optimizer, scheduler, train_loader,
                             dynamics, best, step+1)

        torch.save(
            {'config': config, 'dynamics': dynamics, 'best':  best},
            f"{config.outname}.pt"
        )

    return None


torch.set_default_dtype(torch.float32)

parser = argparse.ArgumentParser(description='Training a Small Language Model on sequence data')
parser.add_argument('--device', type=str, default='cuda')
'''
	DATASET ARGS  (load from .npy files)
'''
parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--path', type=str, default="datasets/")
parser.add_argument('--tokenizer', type=str, default=None)

parser.add_argument('--block_size', metavar='T', type=int, help='sequence length in tokens')
parser.add_argument('--batch_size', metavar='B', type=int)
parser.add_argument('--train_size', metavar='P', type=int)
parser.add_argument('--val_size', metavar='Val', type=int, default=16384)
parser.add_argument('--vocab_size', metavar='V', type=int, help='vocabulary size (number of tokens) ')
parser.add_argument('--name_train_file', type=str, help='name of the training file (without .train.npy)')
parser.add_argument('--name_valid_file', type=str, help='name of the validation file (without .valid.npy)')
'''
	ON-THE-FLY RHM GENERATION ARGS  (alternative to loading .npy files)
'''
parser.add_argument('--generate_data', default=False, action='store_true',
                    help='generate RHM data on-the-fly instead of loading from .npy files')
parser.add_argument('--fresh_online_data', default=False, action='store_true',
                    help='sample fresh generated training batches after each loader reset')
parser.add_argument('--ternary_train', default=False, action='store_true',
                    help='with --generate_data, use only all-ternary trees for training and primary validation')
parser.add_argument('--v',   type=int, help='vocabulary size / num_features for RHM generation')
parser.add_argument('--bin', type=int, help='binary-rule multiplicity (m_2)')
parser.add_argument('--ter', type=int, default=None,
                    help='ternary-rule multiplicity (m_3; default: bin * v)')
parser.add_argument('--eta', type=float, default=None,
                    help='per-rule binary/ternary probability ratio (default: v; 0 uses only ternary rules)')
parser.add_argument('--L',   type=int, help='number of hierarchy levels')
parser.add_argument('--seed_rules',        type=int, default=12345678,
                    help='seed for grammar rule generation (default: 12345678)')
parser.add_argument('--seed_sample_train', type=int, default=11,
                    help='seed for training data generation (default: 11)')
parser.add_argument('--seed_sample_test',  type=int, default=22,
                    help='seed for test data generation (default: 22)')
parser.add_argument('--ternary_val_size',  type=int, default=16384,
                    help='number of all-ternary validation samples to generate (default: 16384)')
parser.add_argument('--completion_val_size', type=int, default=None,
                    help='mixed-tree sentences used for completion validation (default: val_size)')
'''
	ARCHITECTURE ARGS
'''
parser.add_argument('--n_heads', type=int, help='number of heads')
parser.add_argument('--d_embedding', type=int, help='embedding dimension')
parser.add_argument('--ffwd_size', type=int, help='multiplier for the MLP width')
parser.add_argument('--depth', type=int, help='depth of the network')
parser.add_argument('--rope', default=False, action='store_true')
parser.add_argument('--seed_model', type=int, help='seed for model initialization')
parser.add_argument('--compile', default=False, action='store_true',
                    help='torch.compile the model')
'''
	TRAINING ARGS
'''
parser.add_argument('--online', default=False, action='store_true')
parser.add_argument('--autocast', default=False, action='store_true',
                    help='cast forward and loss to lower precision')
parser.add_argument('--objective', choices=('last_token', 'autoregressive'),
                    default='last_token',
                    help='training objective (default: last_token)')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--lr', type=float, help='learning rate', default=0.1)
parser.add_argument('--scheduler', type=str, default='cosine')
parser.add_argument('--warmup_time', type=int, default=16)
parser.add_argument('--decay_time', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--l2', type=float, help='l2 regularisation', default=0.0)
parser.add_argument('--max_epochs', type=int, default=1)
'''
	OUTPUT ARGS
'''
parser.add_argument('--print_freq', type=int, help='frequency of prints', default=16)
parser.add_argument('--save_freq', type=int, help='frequency of saves', default=2)
parser.add_argument('--measure_train', default=False, action='store_true')
parser.add_argument('--topo_val', default=False, action='store_true', help='evaluate validation loss separately for each of the 12 tree topologies')
parser.add_argument('--ternary_val', default=False, action='store_true', help='evaluate validation loss on the all-ternary topology test set')
parser.add_argument('--completion_val', default=False, action='store_true',
                    help='evaluate last-token loss at mixed-tree packed sentence endings')
parser.add_argument('--checkpoints', default=False, action='store_true')
parser.add_argument('--resume_from', type=str, default=None,
                    help='resume model, optimizer, scheduler, loader, and RNG state from a checkpoint')
parser.add_argument('--outname', type=str, required=True, help='path of the output file')

config = parser.parse_args()
run( config)
