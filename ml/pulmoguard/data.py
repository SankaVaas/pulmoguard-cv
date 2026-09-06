"""
Data loading for the chest X-ray pneumonia dataset.

Expects the standard Kaggle "Chest X-Ray Images (Pneumonia)" layout, which
torchvision.datasets.ImageFolder can consume directly:

    data_root/
        train/NORMAL/*.jpeg
        train/PNEUMONIA/*.jpeg
        val/NORMAL/*.jpeg
        val/PNEUMONIA/*.jpeg
        test/NORMAL/*.jpeg
        test/PNEUMONIA/*.jpeg

Note: the original Kaggle release ships only 16 images in `val/`, which is
too small to trust for model selection or threshold tuning. `build_dataloaders`
optionally carves a larger validation split out of `train/` to fix this -
this is a deliberate, documented deviation, not an oversight.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int) -> dict[str, transforms.Compose]:
    """Return train/eval transform pipelines.

    Train transforms include mild, clinically-plausible augmentation
    (small rotation, slight brightness/contrast jitter) - aggressive
    augmentation like horizontal flips is intentionally avoided since chest
    X-ray laterality and orientation can be diagnostically meaningful.
    """
    train_tf = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=7),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return {"train": train_tf, "eval": eval_tf}


def build_datasets(
    data_root: str | Path,
    image_size: int,
    carve_val_from_train: bool = True,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, torch.utils.data.Dataset]:
    """Construct train/val/test datasets.

    If `carve_val_from_train` is True, a stratified-by-index random split of
    `train/` is used for validation instead of the (too-small) shipped
    `val/` folder, and the shipped `val/` folder is ignored. Set to False to
    use the original folders as-is.
    """
    data_root = Path(data_root)
    tfs = build_transforms(image_size)

    train_dir = data_root / "train"
    test_dir = data_root / "test"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Expected '{train_dir}' and '{test_dir}' to exist. "
            f"Check DATA_SETUP.md / scripts/download_data.sh."
        )

    test_dataset = datasets.ImageFolder(str(test_dir), transform=tfs["eval"])

    if carve_val_from_train:
        full_train = datasets.ImageFolder(str(train_dir), transform=tfs["train"])
        # Build a version with eval transforms for the val subset (no augmentation at eval time).
        full_train_eval = datasets.ImageFolder(str(train_dir), transform=tfs["eval"])

        n_total = len(full_train)
        n_val = int(n_total * val_fraction)
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n_total, generator=generator).tolist()
        val_indices = perm[:n_val]
        train_indices = perm[n_val:]

        train_dataset = Subset(full_train, train_indices)
        val_dataset = Subset(full_train_eval, val_indices)
    else:
        val_dir = data_root / "val"
        train_dataset = datasets.ImageFolder(str(train_dir), transform=tfs["train"])
        val_dataset = datasets.ImageFolder(str(val_dir), transform=tfs["eval"])

    return {"train": train_dataset, "val": val_dataset, "test": test_dataset}


def build_dataloaders(
    data_root: str | Path,
    image_size: int,
    batch_size: int,
    num_workers: int = 2,
    seed: int = 42,
) -> tuple[dict[str, DataLoader], tuple[str, ...]]:
    """Construct train/val/test DataLoaders and return the class name tuple
    (index-aligned with model output logits) for downstream use."""
    datasets_dict = build_datasets(data_root, image_size, seed=seed)

    # class_to_idx is only available on the underlying ImageFolder, not Subset.
    train_ds = datasets_dict["train"]
    underlying = train_ds.dataset if isinstance(train_ds, Subset) else train_ds
    class_names = tuple(sorted(underlying.class_to_idx, key=underlying.class_to_idx.get))

    loaders = {
        "train": DataLoader(
            datasets_dict["train"], batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        ),
        "val": DataLoader(
            datasets_dict["val"], batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        ),
        "test": DataLoader(
            datasets_dict["test"], batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        ),
    }
    return loaders, class_names
