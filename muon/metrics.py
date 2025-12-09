"""
Metrics and diagnostics for optimizer stability and geometry analysis.

Implements:
    - Spectral norms of weight matrices
    - SAM-style sharpness proxy
    - Gradient noise scale (GNS) proxy
"""

from typing import Dict, Tuple, Callable
import math

import torch
import torch.nn as nn


def compute_spectral_norms(
    model: nn.Module,
    max_layers: int = 8,
    layer_types: Tuple[type, ...] = (nn.Linear, nn.Conv2d),
) -> Dict[str, float]:
    """
    Compute spectral norms of up to `max_layers` 2D weight matrices.

    Args:
        model: The neural network.
        max_layers: Maximum number of layers to compute.
        layer_types: Only consider these layer types.

    Returns:
        Dict mapping "layer_name" -> spectral_norm (σ_max).
    """
    norms: Dict[str, float] = {}
    count = 0

    for name, module in model.named_modules():
        if isinstance(module, layer_types):
            if hasattr(module, "weight") and module.weight is not None:
                w = module.weight.data
                if w.ndim > 2:
                    w = w.view(w.size(0), -1)
                if w.ndim == 2:
                    with torch.no_grad():
                        try:
                            sigma = torch.linalg.matrix_norm(w, ord=2).item()
                        except RuntimeError:
                            s = torch.linalg.svdvals(w)
                            sigma = s[0].item()
                    norms[name] = sigma
                    count += 1
                    if count >= max_layers:
                        break
    return norms


def estimate_sharpness(
    model: nn.Module,
    loss_fn: Callable,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float = 1e-3,
    normalize: bool = True,
) -> float:
    """
    SAM-style sharpness proxy:
        sharpness ≈ loss(w + ε * g/||g||) - loss(w)
    """
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    targets = targets.to(device)

    model.zero_grad(set_to_none=True)
    base_loss = loss_fn(model, inputs, targets)
    base_loss.backward()

    grad_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm_sq += p.grad.pow(2).sum().item()
    grad_norm = math.sqrt(grad_norm_sq) + 1e-12

    with torch.no_grad():
        for p in model.parameters():
            if p.grad is None:
                continue
            if normalize:
                p.add_(epsilon * p.grad / grad_norm)
            else:
                p.add_(epsilon * torch.sign(p.grad))

    model.zero_grad(set_to_none=True)
    perturbed_loss = loss_fn(model, inputs, targets)

    # Restore weights
    model.zero_grad(set_to_none=True)
    restore_loss = loss_fn(model, inputs, targets)
    restore_loss.backward()

    with torch.no_grad():
        for p in model.parameters():
            if p.grad is None:
                continue
            if normalize:
                p.sub_(epsilon * p.grad / grad_norm)
            else:
                p.sub_(epsilon * torch.sign(p.grad))

    return (perturbed_loss - base_loss).item()


def estimate_gradient_noise_scale(
    model: nn.Module,
    loss_fn: Callable,
    batch1: Tuple[torch.Tensor, torch.Tensor],
    batch2: Tuple[torch.Tensor, torch.Tensor],
) -> float:
    """
    Gradient noise scale proxy from two independent minibatches.

    GNS ≈ ||g₁ - g₂||² / 2
    """
    device = next(model.parameters()).device

    def grad_vec(batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        model.zero_grad(set_to_none=True)
        loss = loss_fn(model, x, y)
        loss.backward()
        grads = []
        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.detach().flatten())
        return torch.cat(grads)

    with torch.no_grad():
        g1 = grad_vec(batch1)
        g2 = grad_vec(batch2)
        diff = g1 - g2
        return torch.dot(diff, diff).item() / 2.0
