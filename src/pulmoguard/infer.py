"""
Production inference interface for PulmoGuard.

This is the module you use locally, after training in Colab, to actually
run the triage system. It wraps checkpoint loading, preprocessing,
MC-Dropout uncertainty estimation, and the abstention decision into a
single class with a stable API:

    from pulmoguard.infer import PulmoGuardPredictor
    predictor = PulmoGuardPredictor(checkpoint_path="outputs/checkpoints/pulmoguard_best.pt")
    result = predictor.predict("path/to/xray.jpeg")
    print(result)
    # PredictionResult(predicted_class='PNEUMONIA', confidence=0.94,
    #                   normalized_entropy=0.12, abstain=False, ...)

The abstention threshold is a deployment decision, not a training-time
constant - it is read from config at predictor construction time and can
be overridden per-call, so operators can tune the confidence/coverage
tradeoff (see outputs/plots/risk_coverage_curve.png) without retraining.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Union

import torch
from PIL import Image

from pulmoguard.data import build_transforms
from pulmoguard.model import build_model
from pulmoguard.uncertainty import mc_dropout_predict, normalized_entropy
from pulmoguard.utils import get_device, get_logger, load_config

logger = get_logger(__name__)


@dataclass
class PredictionResult:
    predicted_class: str
    confidence: float                 # max mean MC-dropout probability
    normalized_entropy: float         # 0 (certain) to 1 (maximally uncertain)
    abstain: bool                     # True if the system recommends deferring to a clinician
    class_probabilities: dict         # full probability distribution over classes
    mc_dropout_passes: int

    def to_dict(self) -> dict:
        return asdict(self)


class PulmoGuardPredictor:
    """Loads a trained PulmoGuard checkpoint and serves predictions with
    calibrated abstention. Instantiate once and reuse across many calls -
    model loading is the expensive part, inference is cheap."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        config_path: Union[str, Path] = "configs/config.yaml",
        device: Optional[torch.device] = None,
        abstain_entropy_threshold: Optional[float] = None,
    ):
        self.config = load_config(config_path)
        self.device = device or get_device()

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at '{checkpoint_path}'. "
                f"Train a model first (see notebooks/train_colab.ipynb) and "
                f"copy the .pt file locally, or check the path."
            )

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model = build_model(
            backbone=self.config["model"]["backbone"],
            num_classes=self.config["model"]["num_classes"],
            pretrained=False,
            dropout_p=self.config["model"]["dropout_p"],
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.class_names = ckpt.get("class_names", tuple(self.config["data"]["class_names"]))
        self.num_passes = self.config["uncertainty"]["mc_dropout_passes"]
        self.abstain_threshold = (
            abstain_entropy_threshold
            if abstain_entropy_threshold is not None
            else self.config["uncertainty"]["abstain_entropy_threshold"]
        )

        self.transform = build_transforms(self.config["data"]["image_size"])["eval"]

        logger.info(
            f"PulmoGuardPredictor ready | classes={self.class_names} | "
            f"device={self.device} | mc_passes={self.num_passes} | "
            f"abstain_threshold={self.abstain_threshold}"
        )

    def _load_image(self, image: Union[str, Path, bytes, Image.Image]) -> Image.Image:
        """Accept a filepath, raw bytes, or a PIL Image for flexibility
        across CLI, batch, and web-upload call sites."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (bytes, bytearray)):
            return Image.open(io.BytesIO(image)).convert("RGB")
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return Image.open(path).convert("RGB")

    @torch.no_grad()
    def predict(
        self, image: Union[str, Path, bytes, Image.Image], abstain_threshold: Optional[float] = None
    ) -> PredictionResult:
        """Run MC-Dropout inference on a single image and return a
        PredictionResult including the abstention decision.

        `abstain_threshold` overrides the instance default for this call
        only - useful for experimenting with different operating points
        without reconstructing the predictor.
        """
        threshold = abstain_threshold if abstain_threshold is not None else self.abstain_threshold

        pil_image = self._load_image(image)
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        result = mc_dropout_predict(self.model, tensor, num_passes=self.num_passes)
        norm_entropy = normalized_entropy(
            result.predictive_entropy, num_classes=len(self.class_names)
        ).item()

        predicted_idx = result.predicted_class.item()
        predicted_label = self.class_names[predicted_idx]
        confidence = result.mean_probs.max(dim=1).values.item()
        probs_dict = {
            cls: round(result.mean_probs[0, i].item(), 4)
            for i, cls in enumerate(self.class_names)
        }

        return PredictionResult(
            predicted_class=predicted_label,
            confidence=round(confidence, 4),
            normalized_entropy=round(norm_entropy, 4),
            abstain=norm_entropy > threshold,
            class_probabilities=probs_dict,
            mc_dropout_passes=self.num_passes,
        )
