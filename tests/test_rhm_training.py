"""Tests command reproducibility without GPU or expensive training."""
import unittest
from dataclasses import replace
from rhm_training import TrainingConfig, build_command


class TrainingInterfaceTest(unittest.TestCase):
    def test_smoke_flags(self):
        cmd = build_command(TrainingConfig.preset("smoke"))
        self.assertIn("--generate_data", cmd)
        self.assertIn("--ternary_val", cmd)
        self.assertEqual(cmd[cmd.index("--bin") + 1], "2")
        self.assertEqual(cmd[cmd.index("--ter") + 1], "16")
        self.assertEqual(cmd[cmd.index("--objective") + 1], "last_token")
        self.assertEqual(cmd[cmd.index("--device") + 1], "cpu")

    def test_paper_preset(self):
        cmd = build_command(TrainingConfig.preset("paper"))
        self.assertEqual(cmd[cmd.index("--train_size") + 1], str(2**24))
        self.assertEqual(cmd[cmd.index("--depth") + 1], "6")

    def test_invalid_shape_rejected(self):
        with self.assertRaises(ValueError):
            build_command(replace(TrainingConfig(), d_embedding=127))


if __name__ == "__main__":
    unittest.main()
