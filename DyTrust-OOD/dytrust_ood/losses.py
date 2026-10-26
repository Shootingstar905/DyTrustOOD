"""Paper Eqs. (10), (15), (16) on an identical set of intervention losses."""
import torch
from torch.nn import functional as F


def intervention_objective(cvae_loss, prediction_losses, alpha=0.01, beta=0.01):
    if not prediction_losses:
        raise ValueError("At least one supervised prediction loss is required")
    samples = torch.stack(prediction_losses)
    mean_loss = samples.mean()
    variance_loss = samples.var(unbiased=False)
    total = alpha * cvae_loss + beta * variance_loss + mean_loss
    if not torch.isfinite(total):
        raise FloatingPointError("Non-finite training objective")
    return total, {"cvae": cvae_loss, "variance": variance_loss, "prediction": mean_loss}


def edge_bce(z, edge_index, labels, decoder, batch_size=65536):
    if labels.numel() == 0:
        raise ValueError("Cannot train on an empty supervised edge batch")
    losses = []
    for offset in range(0, labels.numel(), batch_size):
        sl = slice(offset, offset + batch_size)
        logits = decoder(z, edge_index[:, sl], logits=True)
        losses.append(F.binary_cross_entropy_with_logits(logits, labels[sl], reduction="sum"))
    return torch.stack(losses).sum() / labels.numel()
