"""
random_hierarchy_model.py

Implementation of the Varying-Tree Random Hierarchy Model (RHM) dataset used in the
experiments of the paper

    "Deep Networks Learn to Parse Uniform-Depth Context-Free Languages
     from Local Statistics".

Purpose
-------
This module generates synthetic datasets based on hierarchical grammars
with both binary and ternary production rules. The model produces
sequences together with class labels corresponding to the root
non-terminal of the parse tree.

The dataset is designed to study how neural networks learn hierarchical
structure from local statistical correlations.

Main Components
---------------

Rule generation
    sample_padded_rules()

    Randomly samples grammar production rules for each layer of the
    hierarchy. Binary rules are padded with a special "fake symbol"
    so that both binary and ternary rules can be represented in a
    uniform tensor format.

Probability generation
    create_probabilities()

    Defines the probability distribution used to sample rules during
    sequence generation.

Data generation
    sample_data_from_labels_varying_tree_tensorized()

    Generates terminal sequences by recursively applying grammar rules
    starting from class labels.

Dataset class
-------------

MixedRandomHierarchyModel_varying_tree

A PyTorch Dataset implementing the Random Hierarchy Model with
variable tree structures.

Key features:

    • supports grammars with binary and ternary productions
    • generates sequences with varying tree shapes
    • supports one-hot or integer token representations
    • optional whitening and padding schemes
    • integrates directly with PyTorch DataLoader

The dataset returns:

    x : input sequence representation
    y : class label corresponding to the root non-terminal

These samples are used to train neural networks to predict the root
class from the generated sequences.

Paper
-----
Deep Networks Learn to Parse Uniform-Depth Context-Free Languages
from Local Statistics.
"""




from itertools import *
import warnings
import copy
import sys

import numpy as np
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import defaultdict



def sample_padded_rules(v, n, m_2, m_3, s_2, s_3, L, seed):
    random.seed(seed)

    # Define a callable for the defaultdict to return empty tensors
    def tensor_default():
        return torch.empty(0)

    tuples_2 = list(product(*[range(v) for _ in range(s_2)]))
    tuples_3 = list(product(*[range(v) for _ in range(s_3)]))

    # Define the defaultdict structure
    rules = defaultdict(tensor_default)

    # Initialize the grammar with sampled tensors
    binary_rules = torch.tensor(random.sample(tuples_2, n * m_2)).reshape(n, m_2, s_2)
    ternary_rules = torch.tensor(random.sample(tuples_3, n * m_3)).reshape(n, m_3, s_3)

    # Pad binary rules with the fake symbol (integer v)
    padding = torch.full((n, m_2, s_3 - s_2), v)
    padded_binary_rules = torch.cat((binary_rules, padding), dim=2)

    # Stack binary and ternary rules on top of each other
    rules[0] = torch.cat((padded_binary_rules, ternary_rules), dim=1)

    for i in range(1, L):
        binary_rules = torch.tensor(random.sample(tuples_2, v * m_2)).reshape(
            v, m_2, s_2
        )
        ternary_rules = torch.tensor(random.sample(tuples_3, v * m_3)).reshape(
            v, m_3, s_3
        )

        # Pad binary rules with the fake symbol (integer v)
        padding = torch.full((v, m_2, s_3 - s_2), v)
        padded_binary_rules = torch.cat((binary_rules, padding), dim=2)

        # Stack binary and ternary rules on top of each other
        rules[i] = torch.cat((padded_binary_rules, ternary_rules), dim=1)

    # Add the fake symbol with rules [v, v, v] at each layer
    for l in range(L):
        fake_rules = torch.full((1, m_2 + m_3, 3), v)
        rules[l] = torch.cat((rules[l], fake_rules), dim=0)

    return rules



def create_probabilities_eta(m_2, m_3, L, eta):
    """Assign each binary rule eta times the probability of a ternary rule."""
    if not np.isfinite(eta) or eta < 0:
        raise ValueError("eta must be finite and non-negative")
    probabilities = {}
    p_3 = 1 / (m_2 * eta + m_3)
    p_2 = eta * p_3
    for l in range(L):
        probabilities[l] = torch.cat(
            (torch.full((m_2,), p_2), torch.full((m_3,), p_3))
        )
    return probabilities


def create_probabilities(m_2, m_3, L):
    probabilities = {}
    for l in range(L):
        prob = torch.cat(
            (torch.full((m_2,), 1 / (2 * m_2)), torch.full((m_3,), 1 / (2 * m_3)))
        )
        probabilities[l] = prob
    return probabilities



def sample_data_from_labels_varying_tree_tensorized(
    labels, rules, probability, num_features, d_max
):
    """
    Create data of the Random Hierarchy Model starting from class labels and a set of rules. Rules are chosen according to probability.

    Args:
        labels: A tensor of size [batch_size, I], with I from 0 to num_classes-1 containing the class labels of the data to be sampled.
        rules: A dictionary containing the rules for each level of the hierarchy.
        probability: A dictionary containing the distribution of the rules for each level of the hierarchy.

    Returns:
        A tuple containing the inputs and outputs of the model.
    """
    L = len(rules)  # Number of levels in the hierarchy

    features = labels

    for l in range(L):
        chosen_rule = torch.multinomial(
            probability[l], features.numel(), replacement=True
        ).reshape(
            features.shape
        )  # Choose a rule for each variable in the current level according to probability[l]
        features = rules[l][features, chosen_rule].flatten(
            start_dim=1
        )  # Apply the chosen rule to each variable in the current level
    mask = features == num_features

    # Count the number of fake symbols in each row
    num_fake_symbols = mask.sum(dim=1)

    # Create a tensor to hold the final result
    result = torch.zeros_like(features)
    num_data = features.shape[0]  # Number of data points
    # Remove the fake symbols and keep the original order of the rest of the elements
    real_features = [features[i, ~mask[i]] for i in range(num_data)]
    # print(real_features)
    # Fill the result tensor with real features and append fake symbols at the end
    for i in range(num_data):
        num_real_symbols = d_max - num_fake_symbols[i]
        result[i, :num_real_symbols] = real_features[i]
        result[i, num_real_symbols:] = num_features
    return result, labels


def padded_batch_to_lm_stream(batch, pad_token):
    """
    Convert a padded batch of variable-length sequences into a flat LM stream
    with EOS=0 and real symbols shifted by +1.

    Input:
        batch:      Long tensor of shape [N, d_max]
        pad_token:  Integer token used for right-padding

    Output:
        stream:     1D Long tensor of shape [sum(lengths) + N]
                    Format: seq1_real+1, EOS, seq2_real+1, EOS, ...
    """
    # mask of real (non-padding) symbols
    real_mask = (batch != pad_token)                     # [N, d_max]

    # number of real symbols in each row
    lengths = real_mask.sum(dim=1)                      # [N]

    # all real symbols, already in correct row-major order
    real_tokens = batch[real_mask] + 1                  # shift so EOS can be 0

    # total output length = all real symbols + one EOS per sequence
    N = batch.shape[0]
    total_len = int(lengths.sum().item()) + N

    # allocate flat output
    stream = torch.empty(total_len, dtype=batch.dtype, device=batch.device)

    # positions of EOS tokens
    # each row contributes lengths[i] real tokens + 1 EOS
    row_sizes = lengths + 1
    eos_positions = torch.cumsum(row_sizes, dim=0) - 1  # [N]

    # fill EOS
    stream[eos_positions] = 0

    # fill all remaining positions with real tokens
    real_positions_mask = torch.ones(total_len, dtype=torch.bool, device=batch.device)
    real_positions_mask[eos_positions] = False
    stream[real_positions_mask] = real_tokens

    return stream



# ------------------------------------------------------------
# PyTorch Dataset implementing the Varying-Tree Random Hierarchy Model
# ------------------------------------------------------------

class MixedRandomHierarchyModel_varying_tree(Dataset):
    """
    Implement the Varying-Tree Random Hierarchy Model (RHM) as a PyTorch dataset.
    """

    def __init__(
        self,
        num_features=8,  # vocabulary size
        num_classes=2,  # number of classes
        binary_rules=1,  # number of synonymic low-level representations (multiplicity)
        s_2=2,
        s_3=3,  # size of the low-level representations
        num_layers=2,
        seed_rules=0,
        seed_sample=1,
        train_size=-1,
        test_size=0,
        padding_tail=0,
        padding_classification=0,
        input_format="onehot",
        whitening=0,
        transform=None,
        eta=None,  # per-rule binary/ternary probability ratio (default: num_features)
    ):

        v = num_features
        self.eta = v if eta is None else eta
        f = binary_rules/v
        m_2 = int(f * v)
        m_3 = int(f * v**2)
        self.num_features = num_features
        self.m_2 = m_2
        self.m_3 = m_3
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.s_2 = s_2
        self.s_3 = s_3
        self.binary_rules = binary_rules
        print(seed_rules, seed_sample)
        self.rules = sample_padded_rules(
            num_features, num_classes, m_2, m_3, s_2, s_3, num_layers, seed_rules
        )
        


        torch.manual_seed(seed_sample)
        labels = torch.randint(low=0, high=num_classes, size=(train_size + test_size,))
        d_max = int(s_3 ** num_layers)
        
        self.features, self.labels = (
            sample_data_from_labels_varying_tree_tensorized(
                labels,
                self.rules,
                create_probabilities_eta(m_2, m_3, num_layers, self.eta),
                num_features,
                d_max,
            )
        )

        if "onehot" not in input_format:
            assert not whitening, "Whitening only implemented for one-hot encoding"
        

        if "onehot" in input_format:

            self.features = F.one_hot(
                self.features.long(),
                num_classes=num_features + 1,  # Including the extra feature for padding
            ).float()

            self.features = self.features.permute(
                0, 2, 1
            )  # Shape: (batch_size, num_features+1, input_size)

            # Create a mask for positions with the fake token
            mask = self.features[:, -1, :] == 1  # Shape: (batch_size, input_size)

            # Remove the extra feature used for padding
            self.features = self.features[
                :, :-1, :
            ]  # Shape: (batch_size, num_features, input_size)

            if whitening:
                inv_sqrt_norm = (1.0 - 1.0 / num_features) ** -0.5
                self.features = (self.features - 1.0 / num_features) * inv_sqrt_norm

            # Apply the mask to set elements to zero
            self.features[mask.unsqueeze(1).expand_as(self.features)] = 0
            batch_size, num_features, input_size = self.features.shape
            print("done")
            # print(sum_of_squares)

            if padding_classification: 
                batch_size, num_features, input_size = self.features.shape

                # New number of features (add one for the dummy token's own one-hot position)
                new_num_features = num_features + 1
                new_input_size = input_size + 1

                # Create a zero tensor for the new features with the extra dimensions
                new_features = torch.zeros(batch_size, new_num_features, new_input_size, device=self.features.device, dtype=self.features.dtype)

                # Copy original features shifted by one position along the input axis
                new_features[:, :num_features, 1:] = self.features

                # Set the dummy token: for example, use the new index (num_features) in the first timestep (t=0)
                new_features[:, num_features, 0] = 1.0  # One-hot along the new feature dimension

                # Replace the original features
                self.features = new_features.permute(0, 2, 1)  # Shape: (batch_size, input_size, num_features+1) (B,T,C)



            if padding_tail:
                if num_layers==2:
                    pad_size = 1
                    pad_tensor = torch.zeros(
                        batch_size,
                        num_features,
                        pad_size,
                        device=self.features.device,
                        dtype=self.features.dtype,
                    )
                    self.features = torch.cat((self.features, pad_tensor), dim=2)
                elif num_layers==3:
                    pad_size = 3
                    pad_tensor = torch.zeros(
                        batch_size,
                        num_features,
                        pad_size,
                        device=self.features.device,
                        dtype=self.features.dtype,
                    )
                    self.features = torch.cat((self.features, pad_tensor), dim=2)
                elif num_layers==4:
                    self.features=self.features
        

        elif "long" in input_format:
            self.features = self.features.long()

        else:
            raise ValueError
        

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Args:
                idx: sample index

        Returns:
            Feature-label pairs at index
        """
        x, y = self.features[idx], self.labels[idx]

        if self.transform:
            x, y = self.transform(x, y)

        return x, y


