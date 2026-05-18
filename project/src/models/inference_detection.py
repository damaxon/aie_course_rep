from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .detection_model import build_detection_model

def load_detector_metadata(meta_path: str | Path) -> dict[str, Any]:
    meta_path = Path(meta_path)
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found")

    with meta_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    return metadata


def build_label_mappings_from_meta(classes: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    label_to_id = {label: idx + 1 for idx, label in enumerate(classes)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return label_to_id, id_to_label


def load_detector_from_artifacts(
    weights_path: str | Path,
    meta_path: str | Path,
    device: str | torch.device = "cpu",
):
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"{weights_path} not found")

    metadata = load_detector_metadata(meta_path)

    model_name = metadata["best_model_name"]
    class_names = metadata["classes"]
    num_classes = len(class_names)

    label_to_id, id_to_label = build_label_mappings_from_meta(class_names)

    model = build_detection_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
    )

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    return {
        "model": model,
        "metadata": metadata,
        "class_names": class_names,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "model_name": model_name,
        "num_classes": num_classes,
        "image_size": metadata.get("image_size"),
    }