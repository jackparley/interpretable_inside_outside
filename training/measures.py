import torch
import torch.nn.functional as F


def prediction_loss(model, inputs, targets, criterion):
    """Evaluate scalar last-token or dense autoregressive targets."""
    logits = model(inputs)
    if targets.ndim == 1:
        return criterion(logits[:, -1, :], targets)
    return criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

def evaluate( model, criterion, dataset, iters, device):
    """
    Evaluate criterion by sampling iters batches from dataset.
    
    Returns: TODO explain text data and criterion

    """
    dataset.reset()
    model.eval()
    running = 0.
    
    with torch.no_grad():
        for _ in range(iters):

            inputs, targets = dataset.next_batch()
            inputs, targets = inputs.to(device), targets.to(device)

#             with torch.autocast(device_type=device, dtype=torch.bfloat16): #TODO: check that this is actually running faster on gpus
#                 logits = model(inputs)
#                 loss = criterion(logits.view(-1, logits.size(-1)),targets.view(-1))
            loss = prediction_loss(model, inputs, targets, criterion)

            running += loss.item()

    return running/iters


@torch.no_grad()
def evaluate_completion_losses(model, dataset, iters, device):
    """Return last-symbol CE with EOS included and with EOS renormalized away.

    The loader must supply scalar genuine-symbol targets. The padding/EOS id
    is the final vocabulary entry; removing its logit renormalizes the softmax.
    Both metrics use the same examples and forward passes, averaged by example.
    """
    if iters <= 0:
        raise ValueError("Completion validation requires at least one batch")
    dataset.reset()
    model.eval()
    full_sum = no_eos_sum = 0.0
    count = 0
    for _ in range(iters):
        inputs, targets = dataset.next_batch()
        inputs, targets = inputs.to(device), targets.to(device)
        if targets.ndim != 1:
            raise ValueError("Completion validation requires scalar last-symbol targets")
        logits = model(inputs)[:, -1, :].float()
        pad = dataset.pad_token_id
        if pad != logits.size(-1) - 1:
            raise ValueError("Expected padding/EOS to be the final vocabulary entry")
        if torch.any((targets < 0) | (targets >= pad)):
            raise ValueError("Completion targets must be genuine symbols")
        full_sum += F.cross_entropy(logits, targets, reduction='sum').item()
        no_eos_sum += F.cross_entropy(logits[:, :pad], targets, reduction='sum').item()
        count += targets.numel()
    return full_sum / count, no_eos_sum / count
