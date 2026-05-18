from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from fastapi import HTTPException, UploadFile
from PIL import Image, ImageDraw
from torchvision.transforms import functional as F


def read_image_from_upload(upload_file: UploadFile) -> Image.Image:
    try:
        content = upload_file.file.read()
        image = Image.open(BytesIO(content)).convert("RGB")
        return image
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Не удалось прочитать изображение: {exc}",
        ) from exc


def read_image_from_path(image_path: str | Path) -> Image.Image:

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"{image_path} not found")

    image = Image.open(image_path).convert("RGB")
    return image


def pil_image_to_tensor(
    image: Image.Image,
    device: str | torch.device = "cpu",
) -> torch.Tensor:

    return F.to_tensor(image).to(device)


@torch.no_grad()
def predict_single_image(
    model,
    image: Image.Image,
    id_to_label: dict[int, str],
    device: str | torch.device = "cpu",
    score_threshold: float = 0.3,
) -> list[dict[str, Any]]:

    model.eval()

    image_tensor = pil_image_to_tensor(image=image, device=device)

    outputs = model([image_tensor])
    output = outputs[0]

    boxes = output["boxes"].detach().cpu()
    labels = output["labels"].detach().cpu()
    scores = output["scores"].detach().cpu()

    keep = scores >= score_threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    detections = []
    for box, label, score in zip(boxes, labels, scores):
        label_id = int(label.item())
        label_name = id_to_label.get(label_id, f"unknown_{label_id}")

        detections.append(
            {
                "label_id": label_id,
                "label_name": label_name,
                "score": float(score.item()),
                "bbox": {
                    "xmin": float(box[0].item()),
                    "ymin": float(box[1].item()),
                    "xmax": float(box[2].item()),
                    "ymax": float(box[3].item()),
                },
            }
        )

    return detections


def build_prediction_response(
    image: Image.Image,
    detections: list[dict[str, Any]],
    model_name: str,
    score_threshold: float,
) -> dict[str, Any]:

    return {
        "model_name": model_name,
        "score_threshold": score_threshold,
        "image_width": image.size[0],
        "image_height": image.size[1],
        "num_detections": len(detections),
        "detections": detections,
    }

def draw_detections(image: Image.Image, detections: list[dict]) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)

    for det in detections:
        bbox = det["bbox"]
        text = f"{det['label_name']}: {det['score']:.2f}"

        x1, y1 = bbox["xmin"], bbox["ymin"]
        x2, y2 = bbox["xmax"], bbox["ymax"]

        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
        draw.text((x1, max(0, y1 - 14)), text, fill="red")

    return image
