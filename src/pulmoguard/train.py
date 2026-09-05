"""
Training entrypoint for PulmoGuard.

Designed to run standalone in Colab (T4, free tier) as well as locally.
Usage:
    python -m pulmoguard.train --config configs/config.yaml --data-root data/chest_xray

Key production-hygiene choices:
  - All hyperparameters come from config.yaml, not CLI flags scattered ad hoc.
  - Mixed precision (torch.cuda.amp) to fit comfortably in T4 memory/time budget.
  - Checkpointing on best validation metric (not final epoch) with early stopping.
  - Deterministic seeding for reproducibility.
  - Structured logging instead of bare prints.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score

from pulmoguard.data import build_dataloaders
from pulmoguard.model import build_model, count_trainable_parameters
from pulmoguard.utils import get_device, get_logger, load_config, seed_everything

logger = get_logger(__name__)


def evaluate_epoch(model: nn.Module, loader, device: torch.device) -> dict:
    """Run a standard (non-MC-dropout) evaluation pass for model selection
    during training. MC-dropout evaluation for the risk-coverage curve
    happens separately in evaluate.py, after training is complete."""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return {"loss": avg_loss, "accuracy": acc, "macro_f1": f1}


def train(config: dict, data_root: str, output_dir: str) -> Path:
    """Run the full training loop and return the path to the best checkpoint."""
    seed_everything(config["project"]["seed"])
    device = get_device()
    logger.info(f"Using device: {device}")

    loaders, class_names = build_dataloaders(
        data_root=data_root,
        image_size=config["data"]["image_size"],
        batch_size=config["train"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        seed=config["project"]["seed"],
    )
    logger.info(f"Classes (index-aligned): {class_names}")
    logger.info(
        f"Dataset sizes -> train: {len(loaders['train'].dataset)}, "
        f"val: {len(loaders['val'].dataset)}, test: {len(loaders['test'].dataset)}"
    )

    model = build_model(
        backbone=config["model"]["backbone"],
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
        dropout_p=config["model"]["dropout_p"],
    ).to(device)
    logger.info(f"Trainable parameters: {count_trainable_parameters(model):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=config["train"]["label_smoothing"])
    optimizer = AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )
    use_amp = config["train"]["use_amp"] and device.type == "cuda"
    scaler = GradScaler(device.type, enabled=use_amp)

    checkpoint_dir = Path(output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = checkpoint_dir / config["train"]["best_checkpoint_name"]

    best_val_f1 = -1.0
    epochs_without_improvement = 0
    patience = config["train"]["early_stopping_patience"]

    for epoch in range(1, config["train"]["epochs"] + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0

        for step, (images, labels) in enumerate(loaders["train"], start=1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)

            if step % config["train"]["log_every_n_steps"] == 0:
                logger.info(
                    f"Epoch {epoch} | step {step}/{len(loaders['train'])} | "
                    f"batch_loss={loss.item():.4f}"
                )

        train_loss = running_loss / len(loaders["train"].dataset)
        val_metrics = evaluate_epoch(model, loaders["val"], device)
        epoch_time = time.time() - epoch_start

        logger.info(
            f"[Epoch {epoch}/{config['train']['epochs']}] "
            f"train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | val_macro_f1={val_metrics['macro_f1']:.4f} | "
            f"time={epoch_time:.1f}s"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "class_names": class_names,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                best_ckpt_path,
            )
            logger.info(f"New best model saved to {best_ckpt_path} (val_macro_f1={best_val_f1:.4f})")
        else:
            epochs_without_improvement += 1
            logger.info(f"No improvement for {epochs_without_improvement} epoch(s).")
            if epochs_without_improvement >= patience:
                logger.info("Early stopping triggered.")
                break

    logger.info(f"Training complete. Best val_macro_f1={best_val_f1:.4f}. Checkpoint: {best_ckpt_path}")
    return best_ckpt_path


def main():
    parser = argparse.ArgumentParser(description="Train PulmoGuard pneumonia classifier.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--data-root", type=str, default=None, help="Overrides config data.data_root")
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    config = load_config(args.config)
    data_root = args.data_root or config["data"]["data_root"]
    train(config=config, data_root=data_root, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
