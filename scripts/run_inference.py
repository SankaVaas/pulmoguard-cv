#!/usr/bin/env python
"""
Batch inference CLI for PulmoGuard.

Run predictions over a folder of chest X-ray images and write results to CSV.
Intended as the "make it work locally with trained weights" entrypoint.

Usage:
    python scripts/run_inference.py \
        --checkpoint outputs/checkpoints/pulmoguard_best.pt \
        --image-dir /path/to/xrays \
        --output predictions.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Allow running this script directly without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pulmoguard.infer import PulmoGuardPredictor  # noqa: E402
from pulmoguard.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    parser = argparse.ArgumentParser(description="Batch-predict pneumonia triage over a directory of X-rays.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained .pt checkpoint")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--image-dir", type=str, required=True, help="Directory of X-ray images")
    parser.add_argument("--output", type=str, default="predictions.csv")
    parser.add_argument("--abstain-threshold", type=float, default=None,
                          help="Override the configured abstention entropy threshold")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        logger.error(f"Image directory does not exist: {image_dir}")
        sys.exit(1)

    image_paths = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in VALID_EXTENSIONS)
    if not image_paths:
        logger.error(f"No images with extensions {VALID_EXTENSIONS} found in {image_dir}")
        sys.exit(1)

    logger.info(f"Found {len(image_paths)} images. Loading model...")
    predictor = PulmoGuardPredictor(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        abstain_entropy_threshold=args.abstain_threshold,
    )

    rows = []
    for i, path in enumerate(image_paths, start=1):
        try:
            result = predictor.predict(path)
            row = {"image_path": str(path), **result.to_dict()}
            row["class_probabilities"] = str(row["class_probabilities"])  # flatten for CSV
            rows.append(row)
        except Exception as exc:  # noqa: BLE001 - log and continue batch processing
            logger.error(f"Failed on {path}: {exc}")
            continue

        if i % 25 == 0 or i == len(image_paths):
            logger.info(f"Processed {i}/{len(image_paths)}")

    output_path = Path(args.output)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_abstained = sum(1 for r in rows if r["abstain"])
    logger.info(f"Wrote {len(rows)} predictions to {output_path}")
    logger.info(f"Abstained on {n_abstained}/{len(rows)} cases ({n_abstained / len(rows):.1%})")


if __name__ == "__main__":
    main()
