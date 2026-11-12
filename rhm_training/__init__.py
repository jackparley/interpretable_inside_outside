"""Reproducible CLI presets for the original RHM Transformer training code.

The implementation remains in training/{main,init,train,measures}.py.
"""
from .runner import TrainingConfig, build_command, run_training

__all__ = ["TrainingConfig", "build_command", "run_training"]
