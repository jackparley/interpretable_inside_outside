"""Non-invasive interface to the historical Transformer training entry point.

Does not change model architecture, data loaders, optimizer, or loss computation.
"""
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TrainingConfig:
    # Defaults reflect the primary L=3, v=8 varying-tree setting.
    v: int = 8
    L: int = 3
    m2: int = 2
    m3: int = 16
    eta: float = 8.0
    seed_rules: int = 12345678
    seed_sample_train: int = 11
    seed_sample_test: int = 22
    seed_model: int = 42
    train_size: int = 1024
    val_size: int = 128
    ternary_val_size: int = 128
    batch_size: int = 32
    depth: int = 3
    d_embedding: int = 128
    n_heads: int = 4
    ffwd_size: int = 4
    lr: float = 0.001
    weight_decay: float = 0.03
    epochs: int = 1
    device: str = "cpu"
    objective: str = "last_token"
    ternary_val: bool = True
    fresh_online_data: bool = False
    checkpoints: bool = False
    rope: bool = False
    output: str = "runs/training_smoke/test"

    def validate(self):
        if self.v <= 0 or self.L <= 0 or self.m2 <= 0 or self.m3 <= 0:
            raise ValueError("v, L, m2 and m3 must be positive")
        # The grammar sampler samples without replacement, including a root layer of v symbols.
        if self.v * self.m2 > self.v**2 or self.v * self.m3 > self.v**3:
            raise ValueError("Rule multiplicities exceed available unique child tuples")
        if self.eta < 0 or self.train_size < self.batch_size or self.batch_size <= 0:
            raise ValueError("Invalid eta or training/batch size")
        if self.val_size <= 0 or self.ternary_val_size <= 0 or self.epochs <= 0:
            raise ValueError("Validation sizes and epochs must be positive")
        if self.d_embedding % self.n_heads != 0:
            raise ValueError("d_embedding must be divisible by n_heads")
        if self.objective not in {"last_token", "autoregressive"}:
            raise ValueError("Unknown objective")
        if not self.output or Path(self.output).suffix == ".pt":
            raise ValueError("output must be a file prefix without .pt")
        return self

    @classmethod
    def preset(cls, name):
        if name == "smoke":
            return cls()
        if name == "paper":
            # This is a representative high-compute setting, not a claim that
            # every figure in the paper used these hyperparameters.
            return cls(train_size=2**24, val_size=16384, ternary_val_size=16384,
                       batch_size=128, depth=6, d_embedding=512, n_heads=16,
                       epochs=15, device="cuda", output="runs/paper_baseline/run")
        raise ValueError(f"Unknown preset {name!r}")


def build_command(config: TrainingConfig):
    config.validate()
    values = {
        "v": config.v, "L": config.L, "bin": config.m2, "ter": config.m3,
        "eta": config.eta, "seed_rules": config.seed_rules,
        "seed_sample_train": config.seed_sample_train,
        "seed_sample_test": config.seed_sample_test,
        "seed_model": config.seed_model, "train_size": config.train_size,
        "val_size": config.val_size, "ternary_val_size": config.ternary_val_size,
        "batch_size": config.batch_size, "depth": config.depth,
        "d_embedding": config.d_embedding, "n_heads": config.n_heads,
        "ffwd_size": config.ffwd_size, "lr": config.lr,
        "l2": config.weight_decay, "max_epochs": config.epochs,
        "device": config.device, "objective": config.objective,
        "outname": str((REPO_ROOT / config.output).resolve())
                   if not Path(config.output).is_absolute() else config.output,
    }
    command = [sys.executable, str(REPO_ROOT / "training" / "main.py"), "--generate_data"]
    for key, value in values.items():
        command.extend((f"--{key}", str(value)))
    for key in ("ternary_val", "fresh_online_data", "checkpoints", "rope"):
        if getattr(config, key):
            command.append(f"--{key}")
    return command


def run_training(config: TrainingConfig, dry_run=False):
    command = build_command(config)
    if dry_run:
        return command
    path = Path(config.output)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = path.with_name(path.name + "_manifest.json")
    manifest.write_text(json.dumps({"settings": asdict(config),
                                    "command": command}, indent=2) + "\n")
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    return command
