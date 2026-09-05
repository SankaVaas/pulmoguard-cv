"""
Evaluation entrypoint for PulmoGuard.

Produces the project's headline artifact: a risk-coverage curve showing
model accuracy as a function of how much of the test set the model is
allowed to abstain from, using MC-Dropout predictive entropy as the
uncertainty signal. This is the metric that actually matters for a triage
tool ("if we let the model defer its least-confident 20% of cases to a
radiologist, how accurate is it on the rest?") - as opposed to a single
headline accuracy number on 100% forced coverage.

Also computes Expected Calibration Error (ECE) as a secondary diagnostic.

Usage:
    python -m pulmoguard.evaluate --config configs/config.yaml \
        --checkpoint outputs/checkpoints/pulmoguard_best.pt \
        --data-root data/chest_xray
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # headless-safe backend for Colab / servers
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix

from pulmoguard.data import build_dataloaders
from pulmoguard.model import build_model
from pulmoguard.uncertainty import mc_dropout_predict
from pulmoguard.utils import get_device, get_logger, load_config

logger = get_logger(__name__)


def load_checkpoint(checkpoint_path: str, config: dict, device: torch.device):
    """Load a trained model from a checkpoint saved by train.py."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = build_model(
        backbone=config["model"]["backbone"],
        num_classes=config["model"]["num_classes"],
        pretrained=False,  # weights are overwritten immediately below
        dropout_p=config["model"]["dropout_p"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    class_names = ckpt.get("class_names", tuple(config["data"]["class_names"]))
    return model, class_names, ckpt


def collect_predictions(
    model, loader, device: torch.device, num_passes: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run MC-Dropout inference over an entire loader.

    Returns:
        labels: (N,) ground truth class indices
        preds: (N,) predicted class indices (argmax of mean MC-dropout probs)
        entropies: (N,) predictive entropy per sample
        confidences: (N,) max mean-probability per sample (for calibration)
    """
    all_labels, all_preds, all_entropies, all_confidences = [], [], [], []

    for images, labels in loader:
        images = images.to(device)
        result = mc_dropout_predict(model, images, num_passes=num_passes)

        all_labels.extend(labels.tolist())
        all_preds.extend(result.predicted_class.cpu().tolist())
        all_entropies.extend(result.predictive_entropy.cpu().tolist())
        all_confidences.extend(result.mean_probs.max(dim=1).values.cpu().tolist())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_entropies),
        np.array(all_confidences),
    )


def compute_risk_coverage_curve(
    labels: np.ndarray, preds: np.ndarray, entropies: np.ndarray, num_steps: int = 21
) -> Dict[str, List[float]]:
    """Compute accuracy at each coverage level by progressively abstaining
    on the highest-entropy samples first.

    Coverage = fraction of the test set the model chooses to answer.
    At coverage=1.0, accuracy equals standard forced-choice accuracy.
    At lower coverage, only the most-confident predictions are counted,
    so accuracy should rise monotonically (or near-monotonically) as
    coverage decreases, if uncertainty is a meaningful signal.
    """
    order = np.argsort(entropies)  # ascending: most confident first
    n = len(labels)

    coverages = np.linspace(0.05, 1.0, num_steps)  # avoid coverage=0 (undefined accuracy)
    accuracies = []
    thresholds = []

    for cov in coverages:
        k = max(1, int(round(cov * n)))
        selected_idx = order[:k]
        acc = accuracy_score(labels[selected_idx], preds[selected_idx])
        accuracies.append(acc)
        # entropy threshold at this coverage level (for reporting/deployment)
        thresholds.append(float(entropies[order[k - 1]]))

    return {
        "coverage": coverages.tolist(),
        "accuracy": accuracies,
        "entropy_threshold": thresholds,
    }


def compute_ece(confidences: np.ndarray, correct: np.ndarray, num_bins: int = 10) -> float:
    """Expected Calibration Error: weighted average gap between confidence
    and actual accuracy across confidence bins. Lower is better (0 = perfectly
    calibrated). Reported as a secondary diagnostic alongside risk-coverage."""
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    n = len(confidences)

    for i in range(num_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if in_bin.sum() == 0:
            continue
        bin_confidence = confidences[in_bin].mean()
        bin_accuracy = correct[in_bin].mean()
        ece += (in_bin.sum() / n) * abs(bin_confidence - bin_accuracy)

    return float(ece)


def plot_risk_coverage_curve(curve: Dict[str, List[float]], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(curve["coverage"], curve["accuracy"], marker="o", linewidth=2, color="#1f6feb")
    ax.set_xlabel("Coverage (fraction of cases the model answers)")
    ax.set_ylabel("Accuracy on answered cases")
    ax.set_title("PulmoGuard: Risk-Coverage Curve\n(higher-entropy cases abstained first)")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=min(curve["accuracy"]) - 0.02, top=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved risk-coverage plot to {save_path}")


def plot_confusion_matrix(labels, preds, class_names, save_path: Path) -> None:
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (100% coverage, no abstention)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved confusion matrix plot to {save_path}")


def evaluate(config: dict, checkpoint_path: str, data_root: str, output_dir: str) -> dict:
    device = get_device()
    model, class_names, ckpt = load_checkpoint(checkpoint_path, config, device)
    logger.info(f"Loaded checkpoint from epoch {ckpt.get('epoch')} | classes={class_names}")

    loaders, _ = build_dataloaders(
        data_root=data_root,
        image_size=config["data"]["image_size"],
        batch_size=config["train"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        seed=config["project"]["seed"],
    )

    logger.info(f"Running MC-Dropout inference ({config['uncertainty']['mc_dropout_passes']} passes) on test set...")
    labels, preds, entropies, confidences = collect_predictions(
        model, loaders["test"], device, num_passes=config["uncertainty"]["mc_dropout_passes"]
    )

    forced_accuracy = accuracy_score(labels, preds)
    correct = (labels == preds).astype(int)
    ece = compute_ece(confidences, correct)

    curve = compute_risk_coverage_curve(
        labels, preds, entropies, num_steps=config["evaluate"]["coverage_steps"]
    )

    metrics_dir = Path(output_dir) / "metrics"
    plots_dir = Path(output_dir) / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_risk_coverage_curve(curve, plots_dir / "risk_coverage_curve.png")
    plot_confusion_matrix(labels, preds, class_names, plots_dir / "confusion_matrix.png")

    results = {
        "forced_accuracy_100pct_coverage": forced_accuracy,
        "expected_calibration_error": ece,
        "mc_dropout_passes": config["uncertainty"]["mc_dropout_passes"],
        "risk_coverage_curve": curve,
        "class_names": list(class_names),
    }

    metrics_path = metrics_dir / "evaluation_results.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved evaluation metrics to {metrics_path}")

    logger.info(f"Forced (100% coverage) accuracy: {forced_accuracy:.4f}")
    logger.info(f"Expected Calibration Error: {ece:.4f}")
    for cov, acc in zip(curve["coverage"], curve["accuracy"]):
        logger.info(f"  coverage={cov:.2f} -> accuracy={acc:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate PulmoGuard: risk-coverage curve + calibration.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    config = load_config(args.config)
    data_root = args.data_root or config["data"]["data_root"]
    evaluate(config=config, checkpoint_path=args.checkpoint, data_root=data_root, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
