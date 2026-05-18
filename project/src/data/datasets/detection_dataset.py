from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

class DetectionDataset(Dataset):

    def __init__(
            self,
            images_dir: str | Path,
            annotations_df: pd.DataFrame,
            label_to_id: Dict[str, int],
            transforms=None,
    ):
        self.images_dir = Path(images_dir)
        self.annotations_df = annotations_df.copy().reset_index(drop=True)
        self.label_to_id = label_to_id
        self.transforms = transforms

        if self.annotations_df.empty:
            raise ValueError("annotations_df пустой")
        
        required_columns = {
            "image",
            "label",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
        }

        missing_columns = required_columns - set(self.annotations_df.columns)
        if missing_columns:
            raise ValueError(
                f"В annotations_df отсутствуют обязательные колонки: {sorted(missing_columns)}"
            )
        
        self.image_names = sorted(self.annotations_df["image"].unique().tolist())

        if not self.image_names:
            raise ValueError("В annotations_df не найдено ни одного изображения")

    def __len__(self) -> int:
        return len(self.image_names)
    
    def __getitem__(self, idx: int):
        image_name = self.image_names[idx]
        rows = self.annotations_df[self.annotations_df["image"].eq(image_name)].reset_index(drop=True)

        image_path = self.images_dir / image_name
        if not image_path.exists():
            raise FileNotFoundError(f"{image_path} not found")

        image = Image.open(image_path).convert("RGB")

        boxes = torch.tensor(
            rows[["xmin", "ymin", "xmax", "ymax"]].values,
            dtype=torch.float32,
        )

        labels = torch.tensor(
            [self.label_to_id[label] for label in rows["label"].tolist()],
            dtype=torch.int64,
        )

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        valid = (widths > 1) & (heights > 1)

        boxes = boxes[valid]
        labels = labels[valid]

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            "image_name": image_name,
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        else:
            image = F.to_tensor(image)

        return image, target