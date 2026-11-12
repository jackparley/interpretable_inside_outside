import contextlib

import torch


def context_manager(config):
    """Return a bfloat16 autocast context when enabled, else a no-op context.

    Bfloat16 needs no GradScaler, so only the forward pass and loss are wrapped;
    backward and the optimizer step stay outside the autocast context.
    """
    if config.autocast:
        device_type = (
            "cuda" if str(config.device).startswith("cuda")
            else str(config.device)
        )
        return torch.autocast(device_type=device_type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def prediction_loss(model, inputs, targets, criterion):
    """Support scalar last-token and dense autoregressive targets."""
    logits = model(inputs)
    if targets.ndim == 1:
        return criterion(logits[:, -1, :], targets)
    return criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

def train_step(model, trainset, criterion, optimizer, scheduler, config):
    """
    One step of gradient descent.
    
    Returns:
        Current loss.
    """
    model.train()
    optimizer.zero_grad()

    inputs, targets = trainset.next_batch()
    inputs = inputs.to(config.device)
    targets = targets.to(config.device)

    with context_manager(config):
        loss = prediction_loss(model, inputs, targets, criterion)

    loss.backward()
    optimizer.step()
    scheduler.step()

    return loss.item()
