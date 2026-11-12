import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import math
import random
import json
import re
import models
import measures

class CharacterLevelTokenizer:
    """
    Tokenize text data at the level of the character.
    
    Args:
        data: The text corpus.
    """
    def __init__(self, data):
        self.data = data
        self.vocab = sorted(list(set(self.data)))
        self.vocab_size = len(self.vocab)

        self.i_to_s = { i:ch for i,ch in enumerate(self.vocab)}
        self.s_to_i = { ch:i for i,ch in self.i_to_s.items()}

    def encode(self,s):
        return [self.s_to_i[c] for c in s]

    def decode(self,s):
        return ''.join([self.i_to_s[i] for i in s])


class MySentenceDataLoader:
    """
    Data loader for last-token prediction.

    Expects data as a 2D array of shape [N, d_max], right-padded with pad_token_id.
    For each sentence the input is all tokens except the last real one, left-padded
    to length T = d_max - 1.  The target is the last real token (a scalar per sentence).

    Data is processed lazily (one batch at a time), so large memory-mapped arrays
    are never fully materialised in RAM.
    """

    def __init__(self, B, T, data, pad_token_id):

        self.B = B
        self.T = T
        self.pad_token_id = pad_token_id

        # Store raw array (may be a mmap'd numpy array); do NOT convert the full
        # dataset to a tensor here — that would defeat lazy loading.
        self._data = data

        N, d_max = data.shape
        assert T == d_max - 1, f"Expected T = d_max-1 = {d_max - 1}, got T = {T}"

        self.num_sentences = N
        self.num_batches   = N // B
        self.current_batch = 0
        print(f"loaded {N} sentences, split into {self.num_batches} batches of size {B}")

    def __len__(self):
        return self.num_batches

    def __deepcopy__(self, memo):
        """Share the underlying data array instead of copying it."""
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            if k == '_data':
                setattr(new, k, v)          # shared reference — safe for read-only data
            else:
                import copy
                setattr(new, k, copy.deepcopy(v, memo))
        return new

    def reset(self):
        self.current_batch = 0

    def next_batch(self):
        B   = self.B
        T   = self.T
        pad = self.pad_token_id
        idx = self.current_batch * B
        if idx + B > self.num_sentences:
            self.reset()
            idx = 0

        # np.array() copies the mmap slice into a plain array before wrapping as tensor.
        batch = torch.as_tensor(
            np.array(self._data[idx:idx + B]), dtype=torch.long
        )  # [B, d_max]

        # --- targets: last real (non-pad) token of each sentence ---
        real_mask     = (batch != pad)               # [B, d_max]
        lengths       = real_mask.sum(dim=1)         # [B]
        input_lengths = lengths - 1                  # [B]

        targets = batch[torch.arange(B), input_lengths.clamp(min=0)]  # [B]

        # --- inputs: real tokens (all but last) left-padded to T ---
        j_idx   = torch.arange(T).unsqueeze(0).expand(B, T)   # [B, T]
        src_col = j_idx - T + input_lengths.unsqueeze(1)       # [B, T]
        valid   = src_col >= 0

        gathered = torch.gather(batch[:, :T], 1, src_col.clamp(min=0))  # [B, T]
        inputs   = torch.where(valid, gathered,
                               torch.full_like(gathered, pad))           # [B, T]

        self.current_batch += 1
        return inputs, targets


def _padded_sentences_to_stream(batch, pad_token_id):
    """Remove right-padding and append one EOS token after every sentence.

    The padding id is reused as EOS in the packed LM representation.  This
    keeps real RHM token ids unchanged and does not increase the vocabulary.
    """
    batch = torch.as_tensor(batch, dtype=torch.long)
    real = batch != pad_token_id
    lengths = real.sum(dim=1)
    row_sizes = lengths + 1
    stream = torch.empty(
        int(row_sizes.sum().item()), dtype=torch.long, device=batch.device
    )
    eos_positions = torch.cumsum(row_sizes, dim=0) - 1
    stream[eos_positions] = pad_token_id
    non_eos = torch.ones(stream.numel(), dtype=torch.bool, device=batch.device)
    non_eos[eos_positions] = False
    stream[non_eos] = batch[real]
    return stream


class PackedSentenceDataLoader:
    """Pack padded sentences into fixed-size autoregressive stream batches.

    A returned batch has inputs and next-token targets of shape [B, T].
    Sentences are joined by EOS, represented by the existing padding id.
    """

    def __init__(self, B, T, data, pad_token_id):
        self.B = B
        self.T = T
        self.pad_token_id = pad_token_id
        self._data = data
        self.num_sentences = len(data)

        # Preserve the existing meaning of an epoch (N // B steps).  Sentences
        # are read lazily and wrap around if variable lengths require more data.
        self.num_batches = self.num_sentences // B
        self.current_batch = 0
        self._sentence_idx = 0
        self._buffer = torch.empty(0, dtype=torch.long)
        print(
            f"packed {self.num_sentences} sentences into {self.num_batches} "
            f"autoregressive batches of shape [{B}, {T}]"
        )

    def __len__(self):
        return self.num_batches

    def __deepcopy__(self, memo):
        import copy
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            setattr(new, k, v if k == '_data' else copy.deepcopy(v, memo))
        return new

    def reset(self):
        self.current_batch = 0
        self._sentence_idx = 0
        self._buffer = torch.empty(0, dtype=torch.long)

    def _append_sentences(self, required):
        while self._buffer.numel() < required:
            if self._sentence_idx >= self.num_sentences:
                self._sentence_idx = 0
            # Read a moderate slice at a time, while retaining mmap laziness.
            stop = min(self._sentence_idx + self.B, self.num_sentences)
            rows = np.array(self._data[self._sentence_idx:stop])
            self._sentence_idx = stop
            stream = _padded_sentences_to_stream(rows, self.pad_token_id)
            self._buffer = torch.cat((self._buffer, stream))

    def next_batch(self):
        if self.current_batch >= self.num_batches:
            self.reset()
        required = self.B * self.T + 1
        self._append_sentences(required)
        tokens = self._buffer[:required]
        # Retain the final token: it is the first input token of the next batch,
        # so next-token transitions remain continuous between calls.
        self._buffer = self._buffer[required - 1:]
        inputs = tokens[:-1].view(self.B, self.T)
        targets = tokens[1:].view(self.B, self.T)
        self.current_batch += 1
        return inputs, targets


class PackedSentenceCompletionDataLoader:
    """Last-token sentence completion examples selected from a packed stream.

    For every selected sentence boundary ``..., context, target, EOS``, the
    loader returns the preceding T packed tokens as input and ``target`` as a
    scalar label.  EOS is used only to identify sentence endings; it is never
    a target in this validation metric.
    """

    def __init__(self, B, T, data, pad_token_id):
        self.B = B
        self.T = T
        self.pad_token_id = pad_token_id

        stream = _padded_sentences_to_stream(data, pad_token_id).cpu()
        eos_positions = (stream == pad_token_id).nonzero(as_tuple=False).flatten()
        target_positions = eos_positions - 1
        # A full packed context of length T must precede the target.
        self._target_positions = target_positions[target_positions >= T]
        self._stream = stream
        self.num_examples = self._target_positions.numel()
        self.num_batches = self.num_examples // B
        self.current_batch = 0
        if self.num_batches == 0:
            raise ValueError(
                f"completion validation needs at least {B} eligible sentences, "
                f"but found {self.num_examples}"
            )
        print(
            f"packed completion validation: {self.num_examples} eligible "
            f"sentence endings, {self.num_batches} batches of size {B}"
        )

    def __len__(self):
        return self.num_batches

    def reset(self):
        self.current_batch = 0

    def next_batch(self):
        if self.current_batch >= self.num_batches:
            self.reset()
        start = self.current_batch * self.B
        positions = self._target_positions[start:start + self.B]
        offsets = torch.arange(-self.T, 0).unsqueeze(0)
        inputs = self._stream[positions.unsqueeze(1) + offsets]
        targets = self._stream[positions]

        # Construction invariant: every scalar target immediately precedes EOS.
        assert torch.all(self._stream[positions + 1] == self.pad_token_id)
        assert torch.all(targets != self.pad_token_id)
        self.current_batch += 1
        return inputs, targets


# ---------------------------------------------------------------------------
# Helpers for on-the-fly RHM generation
# ---------------------------------------------------------------------------

def _rhm_import():
    """Return the three generation functions from datasets/rhm, adding the
    directory to sys.path the first time this is called."""
    _dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', 'rhm')
    if _dir not in sys.path:
        sys.path.insert(0, _dir)
    from mixed_random_hierarchy_model import (          # noqa: PLC0415
        sample_padded_rules,
        create_probabilities_eta,
        sample_data_from_labels_varying_tree_tensorized,
    )
    return sample_padded_rules, create_probabilities_eta, sample_data_from_labels_varying_tree_tensorized


def _ternary_probabilities(m_2, m_3, L):
    """Return rule probabilities restricted to ternary productions."""
    return {
        level: torch.cat(
            (torch.zeros(m_2), torch.full((m_3,), 1.0 / m_3))
        )
        for level in range(L)
    }


def _rule_multiplicities(config):
    """Resolve binary/ternary rule counts, preserving the historical default."""
    m_2 = config.bin
    if m_2 is None or m_2 <= 0:
        raise ValueError("--bin must be a positive integer")
    m_3 = config.ter if getattr(config, 'ter', None) is not None else m_2 * config.v
    if m_3 <= 0:
        raise ValueError("--ter must be a positive integer")
    return m_2, m_3


class OnlineRHMDataLoader:
    """
    On-the-fly data loader for the Mixed Random Hierarchy Model.

    Generates each batch lazily from grammar rules; the full dataset of P
    sentences is never stored in memory.  Memory footprint is O(B * d_max)
    regardless of train_size P.

    By default, reproducibility is epoch-local: batch i always uses torch seed
    = seed + i, so results are identical across runs and epochs.  With
    fresh_batches=True, the seed advances with the total number of generated
    batches, so later epochs sample fresh data while remaining reproducible.
    """

    def __init__(self, B, num_batches, rules, probabilities, sample_fn,
                 num_features, num_classes, d_max, pad_token_id, seed,
                 fresh_batches=False):
        self.B             = B
        self.num_batches   = num_batches
        self.rules         = rules
        self.probabilities = probabilities
        self._sample_fn    = sample_fn
        self.num_features  = num_features
        self.num_classes   = num_classes
        self.d_max         = d_max
        self.T             = d_max - 1       # block_size
        self.pad_token_id  = pad_token_id
        self.seed          = seed
        self.fresh_batches = fresh_batches
        self.current_batch = 0
        self.total_batches = 0
        P = num_batches * B
        mode = "fresh batches" if fresh_batches else "epoch-repeated batches"
        print(
            f"OnlineRHMDataLoader: P={P} sentences, {num_batches} batches "
            f"of size {B} ({mode})"
        )

    def __len__(self):
        return self.num_batches

    def __deepcopy__(self, memo):
        """Share grammar/fn references; deep-copy lightweight scalar state."""
        import copy as _copy
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        shared = {'rules', 'probabilities', '_sample_fn'}
        for k, v in self.__dict__.items():
            setattr(new, k, v if k in shared else _copy.deepcopy(v, memo))
        return new

    def reset(self):
        self.current_batch = 0

    def next_batch(self):
        if self.current_batch >= self.num_batches:
            self.reset()

        B   = self.B
        T   = self.T
        pad = self.pad_token_id

        seed_offset = self.total_batches if self.fresh_batches else self.current_batch
        torch.manual_seed(self.seed + seed_offset)
        labels = torch.randint(0, self.num_classes, (B,))
        data, _ = self._sample_fn(
            labels, self.rules, self.probabilities, self.num_features, self.d_max,
        )  # [B, d_max]; pad symbol = num_features

        # --- targets: last real (non-pad) token of each sentence ---
        real_mask     = (data != pad)
        lengths       = real_mask.sum(dim=1)
        input_lengths = lengths - 1
        targets = data[torch.arange(B), input_lengths.clamp(min=0)]  # [B]

        # --- inputs: real tokens (all but last) left-padded to T ---
        j_idx    = torch.arange(T).unsqueeze(0).expand(B, T)
        src_col  = j_idx - T + input_lengths.unsqueeze(1)
        valid    = src_col >= 0
        gathered = torch.gather(data[:, :T], 1, src_col.clamp(min=0))
        inputs   = torch.where(valid, gathered, torch.full_like(gathered, pad))  # [B, T]

        self.current_batch += 1
        self.total_batches += 1
        return inputs, targets


class OnlineRHMAutoregressiveDataLoader(OnlineRHMDataLoader):
    """Generate fresh RHM sentences and pack them into next-token batches."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._buffer = torch.empty(0, dtype=torch.long)
        print(f"  autoregressive stream batches have shape [{self.B}, {self.T}]")

    def reset(self):
        self.current_batch = 0
        self._buffer = torch.empty(0, dtype=torch.long)

    def next_batch(self):
        if self.current_batch >= self.num_batches:
            self.reset()

        seed_offset = self.total_batches if self.fresh_batches else self.current_batch
        torch.manual_seed(self.seed + seed_offset)
        required = self.B * self.T + 1

        while self._buffer.numel() < required:
            labels = torch.randint(0, self.num_classes, (self.B,))
            data, _ = self._sample_fn(
                labels, self.rules, self.probabilities,
                self.num_features, self.d_max,
            )
            stream = _padded_sentences_to_stream(data, self.pad_token_id)
            self._buffer = torch.cat((self._buffer, stream))

        tokens = self._buffer[:required]
        self._buffer = self._buffer[required - 1:]
        inputs = tokens[:-1].view(self.B, self.T)
        targets = tokens[1:].view(self.B, self.T)
        self.current_batch += 1
        self.total_batches += 1
        return inputs, targets


def init_data(config):
    """
    Initialise dataset.
    
    Returns:
        Tokenizer,
        Two dataloaders for train and validation set.
    """

    if config.dataset is not None:
        # Single-dataset mode: same base name for train and valid
        train_file = config.path + config.dataset + ".train.npy"
        valid_file = config.path + config.dataset + ".test.npy"
        meta_file  = config.path + config.tokenizer
    else:
        # Separate-file mode: use name_train_file / name_valid_file
        train_file = config.path + config.name_train_file + ".train.npy"
        valid_file = config.path + config.name_valid_file + ".test.npy"
        # Derive meta.json from the shared base name (strip trailing .e<digits>)
        base = re.sub(r'\.e\d+$', '', config.name_train_file)
        meta_file = config.path + base + ".meta.json"

    with open(meta_file, 'r') as f:
        tokenizer = json.load(f)

    # vocab_size covers real tokens (0..vocab_size-1) plus the pad token at index vocab_size
    config.vocab_size    = tokenizer['vocab_size'] + 1
    config.pad_token_id  = tokenizer['pad_token_id']

    print("vocabulary size:", config.vocab_size)

    # mmap_mode='r' maps the file without loading it all into RAM; slicing
    # produces a view so only the requested rows are ever read from disk.
    train_data = np.load(train_file, mmap_mode='r')
    if config.train_size is not None:
        train_data = train_data[:config.train_size]
    print("number of training sentences:", len(train_data))

    # d_max is read directly from the data; T = d_max - 1
    config.block_size = train_data.shape[1] - 1

    objective = getattr(config, 'objective', 'last_token')
    loader_cls = (PackedSentenceDataLoader
                  if objective == 'autoregressive'
                  else MySentenceDataLoader)
    train_loader = loader_cls(
        config.batch_size, config.block_size, train_data, config.pad_token_id
    )

    valid_data = np.load(valid_file)
    if config.val_size is not None:
        valid_data = valid_data[:config.val_size]
    print("number of validation sentences:", len(valid_data))
    val_loader = loader_cls(
        config.batch_size, config.block_size, valid_data, config.pad_token_id
    )

    if config.online:
        config.max_steps = train_loader.num_batches
    else:
        assert config.max_epochs is not None, "max_epochs arg required unless training online!"
        config.max_steps = config.max_epochs * train_loader.num_batches
    print(f"Training for {config.max_steps} steps")

    return tokenizer, train_loader, val_loader


def init_topo_val_loaders(config, num_topo=12):
    """
    Load per-topology validation loaders.

    Expects files named  <base>.topo{i}.test.npy  for i in 0..num_topo-1,
    where base is derived from config.dataset or config.name_valid_file.
    """
    if config.dataset is not None:
        base = config.path + config.dataset
    else:
        base = config.path + re.sub(r'\.e\d+$', '', config.name_valid_file)

    loaders = []
    for i in range(num_topo):
        fpath = f"{base}.topo{i}.test.npy"
        data = np.load(fpath)
        loader = MySentenceDataLoader(
            config.batch_size, config.block_size, data, config.pad_token_id
        )
        loaders.append(loader)

    print(f"Loaded {num_topo} topology-specific validation sets")
    return loaders


def init_ternary_val_loader(config):
    """
    Load the all-ternary validation loader.

    Expects a file named  <base>.ternary.test.npy,
    where base is derived from config.dataset or config.name_valid_file.
    """
    if config.dataset is not None:
        base = config.path + config.dataset
    else:
        base = config.path + re.sub(r'\.e\d+$', '', config.name_valid_file)

    fpath = f"{base}.ternary.test.npy"
    data = np.load(fpath)
    loader = MySentenceDataLoader(
        config.batch_size, config.block_size, data, config.pad_token_id
    )
    print(f"Loaded all-ternary validation set: {data.shape}")
    return loader


def init_completion_val_loader(config):
    """Load mixed-tree sentences and select packed sentence completions."""
    if config.dataset is not None:
        valid_file = config.path + config.dataset + ".test.npy"
    else:
        valid_file = config.path + config.name_valid_file + ".test.npy"

    data = np.load(valid_file)
    size = getattr(config, 'completion_val_size', None)
    if size is None:
        size = getattr(config, 'val_size', None)
    if size is not None:
        data = data[:size]
    return PackedSentenceCompletionDataLoader(
        config.batch_size, config.block_size, data, config.pad_token_id
    )


def init_data_generated(config):
    """
    Generate train / validation data on-the-fly from a Mixed-RHM grammar.

    Required config fields:  v, bin, L, seed_rules, seed_sample_train,
                             seed_sample_test, train_size, batch_size.
    Optional:                ter (default bin * v), eta (default v), val_size (default 16384).

    Sets config.vocab_size, config.pad_token_id, config.block_size.
    Returns tokenizer dict, train_loader (OnlineRHMDataLoader), val_loader.
    """
    sample_padded_rules, create_probabilities_eta, sample_fn = _rhm_import()

    v   = config.v
    L   = config.L
    bin = config.bin      # binary-rule multiplicity (m_2)
    n   = v               # num_classes == num_features
    m_2, m_3 = _rule_multiplicities(config)
    config.eta = v if getattr(config, 'eta', None) is None else config.eta
    eta = config.eta
    s_2, s_3 = 2, 3
    d_max        = s_3 ** L
    pad_token_id = v      # fake/pad symbol lives at index v (above real vocab)

    config.vocab_size   = v + 1   # 0 .. v-1 real tokens + v = pad
    config.pad_token_id = pad_token_id
    config.block_size   = d_max - 1

    print(f"Grammar: v={v}, bin={bin}, ter={m_3}, eta={eta}, L={L}, d_max={d_max}, seed_rules={config.seed_rules}")
    print(f"vocabulary size: {config.vocab_size}")

    rules         = sample_padded_rules(v, n, m_2, m_3, s_2, s_3, L, config.seed_rules)
    ternary_train = getattr(config, 'ternary_train', False)
    probabilities = (_ternary_probabilities(m_2, m_3, L)
                     if ternary_train
                     else create_probabilities_eta(m_2, m_3, L, eta))
    if ternary_train:
        print("Data topology: all-ternary training and validation")
    else:
        print("Data topology: mixed training and validation")

    # --- training loader: fully on-the-fly, O(B * d_max) RAM ---
    assert config.train_size is not None, "--train_size is required with --generate_data"
    P             = config.train_size
    num_train_batches = P // config.batch_size
    assert num_train_batches > 0, "train_size must be >= batch_size"

    objective = getattr(config, 'objective', 'last_token')
    train_loader_cls = (OnlineRHMAutoregressiveDataLoader
                        if objective == 'autoregressive'
                        else OnlineRHMDataLoader)
    train_loader = train_loader_cls(
        B=config.batch_size,
        num_batches=num_train_batches,
        rules=rules,
        probabilities=probabilities,
        sample_fn=sample_fn,
        num_features=v,
        num_classes=n,
        d_max=d_max,
        pad_token_id=pad_token_id,
        seed=config.seed_sample_train,
        fresh_batches=getattr(config, 'fresh_online_data', False),
    )

    # --- validation loader: small, generate once ---
    val_size = config.val_size if config.val_size is not None else 16384
    torch.manual_seed(config.seed_sample_test)
    labels_val = torch.randint(0, n, (val_size,))
    val_data, _ = sample_fn(labels_val, rules, probabilities, v, d_max)
    val_loader_cls = (PackedSentenceDataLoader
                      if objective == 'autoregressive'
                      else MySentenceDataLoader)
    val_loader = val_loader_cls(
        config.batch_size, config.block_size,
        val_data.numpy().astype(np.int32), pad_token_id,
    )

    if config.online:
        config.max_steps = train_loader.num_batches
    else:
        assert config.max_epochs is not None, "max_epochs required unless --online"
        config.max_steps = config.max_epochs * train_loader.num_batches
    print(f"Training for {config.max_steps} steps")

    tokenizer = {'vocab_size': v, 'pad_token_id': pad_token_id}
    return tokenizer, train_loader, val_loader


def init_ternary_val_loader_generated(config):
    """
    Generate the all-ternary validation set on-the-fly (no .npy file needed).
    """
    sample_padded_rules, _create_probs, sample_fn = _rhm_import()

    v   = config.v
    L   = config.L
    n   = v
    m_2, m_3 = _rule_multiplicities(config)
    s_2, s_3 = 2, 3
    d_max = s_3 ** L

    rules = sample_padded_rules(v, n, m_2, m_3, s_2, s_3, L, config.seed_rules)

    ternary_probs = _ternary_probabilities(m_2, m_3, L)

    ternary_size = config.ternary_val_size if config.ternary_val_size is not None else 16384
    torch.manual_seed(config.seed_sample_test + 1)
    labels = torch.randint(0, n, (ternary_size,))
    data, _ = sample_fn(labels, rules, ternary_probs, v, d_max)

    print(f"Generated all-ternary validation set: {data.shape}")
    return MySentenceDataLoader(
        config.batch_size, config.block_size,
        data.numpy().astype(np.int32), config.pad_token_id,
    )


def init_completion_val_loader_generated(config):
    """Generate a fixed mixed-tree packed sentence-completion validation set."""
    sample_padded_rules, create_probabilities_eta, sample_fn = _rhm_import()

    v = config.v
    L = config.L
    m_2, m_3 = _rule_multiplicities(config)
    d_max = 3 ** L
    rules = sample_padded_rules(
        v, v, m_2, m_3, 2, 3, L, config.seed_rules
    )
    eta = v if getattr(config, 'eta', None) is None else config.eta
    probabilities = create_probabilities_eta(m_2, m_3, L, eta)

    size = getattr(config, 'completion_val_size', None)
    if size is None:
        size = config.val_size if config.val_size is not None else 16384
    # Reuse the standard validation seed/distribution, making the completion
    # metric a curated view of the same fixed mixed-tree validation sample.
    torch.manual_seed(config.seed_sample_test)
    labels = torch.randint(0, v, (size,))
    data, _ = sample_fn(labels, rules, probabilities, v, d_max)
    return PackedSentenceCompletionDataLoader(
        config.batch_size, config.block_size,
        data.numpy().astype(np.int32), config.pad_token_id,
    )


def init_model(config, seed=None):
    """
    Initialise machine-learning model. 
    """
    seed = config.seed_model if seed is None else seed
    torch.manual_seed(seed)

    model = models.CLM(
        vocab_size=config.vocab_size,
        block_size=config.block_size,
        embedding_dim=config.d_embedding,
        num_heads=config.n_heads,
        ffwd_size=config.ffwd_size,
        num_layers=config.depth,
        dropout=config.dropout,
        rope=config.rope,
        share_emb=False,
    )
    model.to(config.device)

    if config.compile:
        model = torch.compile(model)

    param_count = sum([p.numel() for p in model.parameters()])
    print("# parameters:", param_count)

    return model


def init_training( model, config):
    """
    Initialise training algorithm.
    """
    criterion = nn.CrossEntropyLoss( reduction='mean')
    
    if config.optim =='adam':
        optimizer = optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.l2
        )
    else:
        raise ValueError("optimizer is invalid (adamW)!")

    if config.scheduler =='cosine':
        scheduler = CosineWarmupLR(
            optimizer, config.warmup_time, config.decay_time, 0.1
        )
    else:
        raise ValueError("scheduler is invalid!")


    return criterion, optimizer, scheduler


def log2ckpt( end, freq):
    """
    Initialise log-spaced iterator.

    Returns:
        List with integer steps spaced multiplicatively by 2**(1/freq) until end.
    """
    current = 1.
    factor = 2**(1./freq)
    threshold = 2**(math.ceil(math.log(1./(factor-1)))+1)
    checkpoints = []

    while current < threshold:
        checkpoints.append( round( current))
        current += 1

    while round(current) < end:
        checkpoints.append( round( current))
        current *= factor

    checkpoints.append( round( end))

    return checkpoints

def init_loglinckpt( step, end, freq):
    """
    Initialise checkpoint iterator.

    Returns:
        Two iterators, one for linear and one for logscale. The iterators coincide upt to some multiple of step, 
        then one proceeds linearly in multiples of step and the other logarithmically in factors of 2**(1/freq).
    """
    # find the correct multiplier
    factor = 2**(1./freq)
    multiplier = 2**(math.ceil(math.log(1./(factor-1)))+1)

    # build log2ckpt lists until multiplier*step
    lin_ckpts = log2ckpt( multiplier*step, freq)
    log_ckpts = lin_ckpts.copy()

    # fill the linear list by adding steps until end
    current = lin_ckpts[-1] + step
    while current <= end:
        lin_ckpts.append(current)
        current += step
    lin_ckpts.append(0)

    # fill the log list by multiplying factors until end
    current = multiplier*factor
    while round(current)*step < end:
        log_ckpts.append( round(current)*step)
        current *= factor

    log_ckpts.append(round( end))
    log_ckpts.append(0)

    return iter(lin_ckpts), iter(log_ckpts)


class CosineWarmupLR(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup_time, decay_time, min_lr_factor):
        self.warmup = warmup_time
        self.decay = decay_time
        self.min_lr_factor = min_lr_factor
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(step=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, step):
        if step < self.warmup:
            # Linear warmup from 0 to 1
            return step / self.warmup
        elif step < self.decay:
            # Cosine decay to min_lr_factor
            decay_step = step - self.warmup
            total_decay = self.decay - self.warmup
            cosine_decay = 0.5 * (1 + np.cos(np.pi * decay_step / total_decay))
            return self.min_lr_factor + (1 - self.min_lr_factor) * cosine_decay
        else:
            # Stay constant after decay
            return self.min_lr_factor
