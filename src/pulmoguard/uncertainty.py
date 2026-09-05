"""
Monte Carlo Dropout uncertainty estimation.

Standard inference (`model.eval()`) disables dropout, producing a single
deterministic prediction with no notion of confidence beyond the softmax
score - which is well known to be poorly calibrated and overconfident,
especially out-of-distribution.

Monte Carlo Dropout (Gal & Ghahramani, 2016) keeps dropout ACTIVE at
inference time and runs multiple stochastic forward passes. The spread of
predictions across passes approximates the model's epistemic uncertainty:
tight agreement -> confident; wide disagreement -> uncertain, and a
candidate for abstention.

This module is intentionally decoupled from training/eval scripts so the
exact same uncertainty logic is used in evaluate.py (to build the
risk-coverage curve) and in infer.py / the serving app (for live
predictions) - avoiding train/serve skew.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def enable_mc_dropout(model: nn.Module) -> None:
    """Set the model to eval mode (freezes BatchNorm running stats) but
    force all Dropout layers back into train mode so they remain stochastic.

    This is the standard MC-Dropout trick: BatchNorm must stay in eval mode
    (we don't want batch statistics computed on a single inference batch),
    but Dropout must stay active.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


@dataclass
class UncertaintyResult:
    """Container for MC-Dropout inference output on a batch.

    mean_probs: (batch, num_classes) averaged softmax probability across passes.
    predictive_entropy: (batch,) entropy of the mean predicted distribution -
        the primary uncertainty score used for abstention decisions.
    predicted_class: (batch,) argmax of mean_probs.
    """
    mean_probs: torch.Tensor
    predictive_entropy: torch.Tensor
    predicted_class: torch.Tensor


@torch.no_grad()
def mc_dropout_predict(
    model: nn.Module,
    inputs: torch.Tensor,
    num_passes: int = 20,
) -> UncertaintyResult:
    """Run `num_passes` stochastic forward passes and aggregate.

    Predictive entropy H = -sum_c p_c * log(p_c) over the MEAN probability
    vector (not averaged per-pass entropy) - this is the standard formulation
    that captures genuine model disagreement rather than just per-pass
    softmax sharpness.
    """
    if num_passes < 1:
        raise ValueError("num_passes must be >= 1")

    enable_mc_dropout(model)

    all_probs = []
    for _ in range(num_passes):
        logits = model(inputs)
        probs = F.softmax(logits, dim=1)
        all_probs.append(probs)

    stacked = torch.stack(all_probs, dim=0)          # (passes, batch, classes)
    mean_probs = stacked.mean(dim=0)                 # (batch, classes)

    eps = 1e-12
    predictive_entropy = -(mean_probs * torch.log(mean_probs + eps)).sum(dim=1)
    predicted_class = mean_probs.argmax(dim=1)

    return UncertaintyResult(
        mean_probs=mean_probs,
        predictive_entropy=predictive_entropy,
        predicted_class=predicted_class,
    )


def normalized_entropy(entropy: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Normalize entropy to [0, 1] by dividing by max possible entropy
    (log(num_classes)), making the abstention threshold interpretable
    independent of class count."""
    max_entropy = torch.log(torch.tensor(float(num_classes)))
    return entropy / max_entropy
