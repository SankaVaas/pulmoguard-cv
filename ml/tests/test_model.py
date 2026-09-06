"""
Unit tests for PulmoGuard core modules.

These tests use synthetic tensors (no dataset download required) so they
run fast in CI or before pushing to Colab, and catch shape/API regressions
in model.py and uncertainty.py before they waste GPU time.

Run with:
    pytest tests/ -v
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pulmoguard.model import build_model, count_trainable_parameters
from pulmoguard.uncertainty import (
    enable_mc_dropout,
    mc_dropout_predict,
    normalized_entropy,
)


@pytest.fixture(scope="module")
def model():
    return build_model(backbone="efficientnet_b0", num_classes=2, pretrained=False, dropout_p=0.3)


def test_model_output_shape(model):
    """Model should output logits of shape (batch, num_classes)."""
    dummy_input = torch.randn(4, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        logits = model(dummy_input)
    assert logits.shape == (4, 2)


def test_model_has_trainable_parameters(model):
    """Sanity check that the backbone isn't accidentally fully frozen."""
    assert count_trainable_parameters(model) > 0


def test_invalid_backbone_raises():
    with pytest.raises(ValueError):
        build_model(backbone="not_a_real_backbone")


def test_enable_mc_dropout_keeps_dropout_active(model):
    """After enable_mc_dropout, Dropout layers must be in train() mode even
    though the overall model.training flag context is eval-like."""
    enable_mc_dropout(model)
    dropout_layers = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    assert len(dropout_layers) > 0
    assert all(layer.training for layer in dropout_layers)


def test_mc_dropout_predict_output_shapes(model):
    """MC-dropout inference should return correctly-shaped mean_probs,
    predictive_entropy, and predicted_class tensors."""
    dummy_input = torch.randn(3, 3, 224, 224)
    result = mc_dropout_predict(model, dummy_input, num_passes=5)

    assert result.mean_probs.shape == (3, 2)
    assert result.predictive_entropy.shape == (3,)
    assert result.predicted_class.shape == (3,)
    # Probabilities must sum to ~1 per sample
    sums = result.mean_probs.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_mc_dropout_rejects_invalid_num_passes(model):
    dummy_input = torch.randn(1, 3, 224, 224)
    with pytest.raises(ValueError):
        mc_dropout_predict(model, dummy_input, num_passes=0)


def test_normalized_entropy_bounds():
    """Normalized entropy must lie in [0, 1] for a binary classification problem."""
    raw_entropy = torch.tensor([0.0, 0.3465, 0.6931])  # 0, half-max, max for 2 classes
    norm = normalized_entropy(raw_entropy, num_classes=2)
    assert torch.all(norm >= -1e-6)
    assert torch.all(norm <= 1.0 + 1e-6)
