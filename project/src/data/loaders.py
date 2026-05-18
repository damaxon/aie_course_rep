from pathlib import Path
import random
from typing import Dict, Tuple

import pandas as pd
from torch.utils.data import DataLoader

from .datasets import DetectionDataset
from .transforms import DetectionTrainTransform, DetectionValTransform

def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

def build_label_mappings(annotations_df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
    class_names = sorted(annotations_df["label"].unique().tolist())
    label_to_id = {label: idx + 1 for idx, label in enumerate(class_names)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return label_to_id, id_to_label

def split_images_by_seed(
    image_names: list[str],
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[list[str], list[str]]:
    if not 0 < val_split < 1:
        raise ValueError("val_split должен быть в диапазоне (0, 1)")

    image_names = sorted(image_names)
    rng = random.Random(seed)
    rng.shuffle(image_names)

    val_size = int(len(image_names) * val_split)
    if val_size == 0:
        raise ValueError("Слишком маленький dataset: val_split даёт пустой validation set")

    val_images = image_names[:val_size]
    train_images = image_names[val_size:]

    if len(train_images) == 0:
        raise ValueError("Слишком маленький dataset: train set получился пустым")

    return train_images, val_images

def filter_annotations_by_images(
    annotations_df: pd.DataFrame,
    image_names: list[str],
) -> pd.DataFrame:
    return annotations_df[annotations_df["image"].isin(image_names)].copy().reset_index(drop=True)

def create_detection_dataloaders(
    images_dir: str | Path,
    annotations_csv: str | Path,
    batch_size: int = 4,
    val_split: float = 0.2,
    num_workers: int = 0,
    seed: int = 42,
):
    images_dir = Path(images_dir)
    annotations_csv = Path(annotations_csv)

    if not images_dir.exists():
        raise FileNotFoundError(f"{images_dir} not found")

    if not annotations_csv.exists():
        raise FileNotFoundError(f"{annotations_csv} not found")

    annotations_df = pd.read_csv(annotations_csv)

    required_columns = {
        "image",
        "label",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "image_width",
        "image_height",
    }
    missing_columns = required_columns - set(annotations_df.columns)
    if missing_columns:
        raise ValueError(
            f"В annotations.csv отсутствуют обязательные колонки: {sorted(missing_columns)}"
        )

    label_to_id, id_to_label = build_label_mappings(annotations_df)

    image_names = sorted(annotations_df["image"].unique().tolist())
    train_images, val_images = split_images_by_seed(
        image_names=image_names,
        val_split=val_split,
        seed=seed,
    )

    train_df = filter_annotations_by_images(annotations_df, train_images)
    val_df = filter_annotations_by_images(annotations_df, val_images)

    train_dataset = DetectionDataset(
        images_dir=images_dir,
        annotations_df=train_df,
        label_to_id=label_to_id,
        transforms=DetectionTrainTransform(),
    )

    val_dataset = DetectionDataset(
        images_dir=images_dir,
        annotations_df=val_df,
        label_to_id=label_to_id,
        transforms=DetectionValTransform(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, label_to_id, id_to_label

