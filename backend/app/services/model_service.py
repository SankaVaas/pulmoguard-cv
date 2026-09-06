"""
Model service layer.

Wraps `pulmoguard.infer.PulmoGuardPredictor` (from the `pulmoguard-ml`
library) as an application-level singleton, loaded once at API startup.
This is the seam between the web layer (FastAPI routes, auth, HTTP
concerns) and the ML layer (model weights, uncertainty estimation) - routes
should never import torch or touch checkpoints directly.
"""
from __future__ import annotations

import logging
from pathlib import Path

from app.core.config import get_settings
from pulmoguard.infer import PulmoGuardPredictor

logger = logging.getLogger(__name__)
settings = get_settings()

_predictor: PulmoGuardPredictor | None = None


def load_model() -> None:
    """Load the model checkpoint once at application startup.

    Raises FileNotFoundError (surfaced as a fatal startup error, by design -
    an API server with no model loaded should not report itself healthy)
    if the checkpoint has not been placed at the configured path.
    """
    global _predictor
    checkpoint_path = Path(settings.ML_CHECKPOINT_PATH)
    config_path = Path(settings.ML_CONFIG_PATH)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at '{checkpoint_path}'. "
            f"Train a model via ml/notebooks/train_colab.ipynb, download the "
            f".pt file, and place it at this path (or set ML_CHECKPOINT_PATH)."
        )

    _predictor = PulmoGuardPredictor(checkpoint_path=checkpoint_path, config_path=config_path)
    logger.info("Model loaded successfully by model_service.")


def get_predictor() -> PulmoGuardPredictor:
    """Return the loaded predictor singleton. Raises RuntimeError if called
    before `load_model()` has succeeded (should be unreachable in normal
    operation since the health check gates traffic on model_loaded)."""
    if _predictor is None:
        raise RuntimeError("Model has not been loaded. Call load_model() at startup.")
    return _predictor


def is_model_loaded() -> bool:
    return _predictor is not None
