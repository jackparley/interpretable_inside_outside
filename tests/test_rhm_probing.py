"""Unit tests for the public probe adapter (no paper checkpoint required)."""
import pickle
import tempfile
import unittest
from pathlib import Path

import torch

from rhm_probing.runner import load_reference_bundle, infer_checkpoint, shared


class TestProbingAdapter(unittest.TestCase):
    def test_reference_validation(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'bundle.pkl'
            path.write_bytes(pickle.dumps({'grammar': {}}))
            with self.assertRaisesRegex(ValueError, 'dataset'):
                load_reference_bundle(path)

    def test_infer_architecture_with_compiled_prefix(self):
        model = shared.models.CLM(vocab_size=5, block_size=8,
                                  embedding_dim=16, num_heads=4, ffwd_size=2,
                                  num_layers=2, dropout=0.0, rope=False, share_emb=False)
        state = {'_orig_mod.' + k: v for k, v in model.state_dict().items()}
        clean, arch = infer_checkpoint(state, 4)
        self.assertEqual(set(clean), set(model.state_dict()))
        self.assertEqual(arch['num_layers'], 2)
        self.assertEqual(arch['embedding_dim'], 16)
        self.assertEqual(arch['block_size'], 8)


if __name__ == '__main__':
    unittest.main()
