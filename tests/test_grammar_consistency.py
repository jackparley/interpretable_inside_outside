"""Check that the training and parsing grammars describe the same rules.

Run from the root of interpretable_inside_outside:
    python tests/test_grammar_consistency.py

Uses the standard L=3, v=8, m2=2, m3=16, seed=12345678 configuration.
"""
from pathlib import Path
import importlib.util
import torch

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / 'training/datasets/rhm/mixed_random_hierarchy_model.py'
PARSING = ROOT / 'parsing/inside_outside/For_training/Inside_outside.py'


def load_module(name, path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing module: {path}')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    train = load_module('training_grammar', TRAINING)
    parse = load_module('parsing_grammar', PARSING)
    v, n, m2, m3, L, seed = 8, 8, 2, 16, 3, 12345678
    eta = v

    padded = train.sample_padded_rules(v, n, m2, m3, 2, 3, L, seed)
    mixed = parse.sample_mixed_rules(v, n, m2, m3, 2, 3, L, seed)
    probs = train.create_probabilities_eta(m2, m3, L, eta)

    _, flat_rules = parse.generate_rules_double_new(
        v, n, m2, m3, float(probs[0][0]), float(probs[0][m2]), L, seed
    )
    grammar = parse.generate_grammar_2(flat_rules)
    binary = parse.extract_binary_rules_as_tensors(grammar, v)
    ternary = parse.extract_ternary_rules_as_tensors(grammar, v)

    for layer in range(L):
        count = n if layer == 0 else v
        torch.testing.assert_close(
            padded[layer][:count, :m2, :2], mixed[layer][0], rtol=0, atol=0
        )
        torch.testing.assert_close(
            padded[layer][:count, m2:m2 + m3, :], mixed[layer][1], rtol=0, atol=0
        )
        assert torch.all(padded[layer][:count, :m2, 2] == v)

        expected_binary = torch.zeros((v, v, v))
        expected_ternary = torch.zeros((v, v, v, v))
        for parent in range(v):
            for a, b in mixed[layer][0][parent].tolist():
                expected_binary[parent, a, b] = probs[layer][0]
            for a, b, c in mixed[layer][1][parent].tolist():
                expected_ternary[parent, a, b, c] = probs[layer][m2]

        key = f'layer{layer + 1}'
        torch.testing.assert_close(expected_binary, binary[key], rtol=0, atol=0)
        torch.testing.assert_close(expected_ternary, ternary[key], rtol=0, atol=0)
        print(f'Layer {layer + 1}: binary and ternary rules/probabilities match exactly')

    print('PASS: training and parsing grammars are identical for this configuration')


if __name__ == '__main__':
    main()
