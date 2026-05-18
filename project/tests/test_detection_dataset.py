import pandas as pd
import torch

from src.data.datasets import DetectionDataset
from src.data.paths import DATA_PROCESSED_DIR
from src.data.loaders import build_label_mappings


def test_detection_dataset_returns_valid_sample():
    annotations_path = DATA_PROCESSED_DIR / "detection" / "annotations.csv"
    images_dir = DATA_PROCESSED_DIR / "detection" / "images"

    df = pd.read_csv(annotations_path).head(20)
    label_to_id, _ = build_label_mappings(df)

    dataset = DetectionDataset(
        images_dir=images_dir,
        annotations_df=df,
        label_to_id=label_to_id,
        transforms=None,
    )

    image, target = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert image.ndim == 3
    assert "boxes" in target
    assert "labels" in target
    assert target["boxes"].shape[1] == 4
    assert target["labels"].dtype == torch.int64