from __future__ import annotations

import json
from pathlib import Path
from io import BytesIO

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Any, Dict, List

from src.data.paths import PROJECT_DIR
from src.models import load_detector_from_artifacts
from src.models.predict_detection import (
    build_prediction_response,
    predict_single_image,
    read_image_from_upload,
    draw_detections,
)

ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "models" / "best_detector.pt"
META_PATH = ARTIFACTS_DIR / "metrics" / "best_detector_meta.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(
    title="Vehicle Detection API",
    version="1.0.0",
    description=(
        "HTTP-сервис для детекции транспортных объектов на изображениях. "
        "Использует заранее обученную detection-модель из artifacts."
    ),
    docs_url="/docs",
    redoc_url=None,
)


class BoundingBoxResponse(BaseModel):
    xmin: float = Field(..., description="Левая граница bbox")
    ymin: float = Field(..., description="Верхняя граница bbox")
    xmax: float = Field(..., description="Правая граница bbox")
    ymax: float = Field(..., description="Нижняя граница bbox")


class DetectionResponse(BaseModel):
    label_id: int = Field(..., description="Идентификатор класса")
    label_name: str = Field(..., description="Название класса")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    bbox: BoundingBoxResponse


class PredictResponse(BaseModel):
    model_name: str = Field(..., description="Имя модели")
    score_threshold: float = Field(..., ge=0.0, le=1.0, description="Порог confidence")
    image_width: int = Field(..., ge=1, description="Ширина изображения")
    image_height: int = Field(..., ge=1, description="Высота изображения")
    num_detections: int = Field(..., ge=0, description="Количество найденных объектов")
    detections: List[DetectionResponse]


class HealthResponse(BaseModel):
    status: str
    service: str
    model_name: str
    device: str
    num_classes: int

class InfoResponse(BaseModel):
    service: str
    version: str
    task: str
    dataset: str
    model_name: str
    device: str
    num_classes: int
    classes: List[str]
    artifacts: Dict[str, str]
    metrics: Dict[str, float]


def _load_service_bundle() -> dict[str, Any]:
    return load_detector_from_artifacts(
        weights_path=MODEL_PATH,
        meta_path=META_PATH,
        device=DEVICE,
    )


SERVICE_BUNDLE = _load_service_bundle()
MODEL = SERVICE_BUNDLE["model"]
ID_TO_LABEL = SERVICE_BUNDLE["id_to_label"]
MODEL_NAME = SERVICE_BUNDLE["model_name"]
CLASS_NAMES = SERVICE_BUNDLE["class_names"]


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="vehicle-detection",
        model_name=MODEL_NAME,
        device=str(DEVICE),
        num_classes=len(CLASS_NAMES),
    )

@app.get("/info", response_model=InfoResponse, tags=["system"])
def info() -> InfoResponse:
    metadata = SERVICE_BUNDLE["metadata"]

    metrics = {
        "best_val_precision@0.5": metadata.get("best_val_precision@0.5"),
        "best_val_recall@0.5": metadata.get("best_val_recall@0.5"),
        "best_val_f1@0.5": metadata.get("best_val_f1@0.5"),
        "best_val_map": metadata.get("best_val_map"),
        "best_val_map@0.5": metadata.get("best_val_map@0.5"),
        "best_val_map@0.75": metadata.get("best_val_map@0.75"),
    }

    metrics = {
        key: float(value)
        for key, value in metrics.items()
        if value is not None
    }

    return InfoResponse(
        service="vehicle-detection",
        version="1.0.0",
        task=metadata.get("task", "detection"),
        dataset=metadata.get("dataset", "Udacity Self Driving Car Dataset"),
        model_name=MODEL_NAME,
        device=str(DEVICE),
        num_classes=len(CLASS_NAMES),
        classes=CLASS_NAMES,
        artifacts={
            "model_path": str(MODEL_PATH),
            "meta_path": str(META_PATH),
        },
        metrics=metrics,
    )

@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["prediction"],
    summary="Детекция объектов на изображении",
)
def predict(
    file: UploadFile = File(...),
    score_threshold: float = 0.3,
) -> PredictResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Файл не был передан")

    if not 0.0 <= score_threshold <= 1.0:
        raise HTTPException(status_code=400, detail="score_threshold должен быть в диапазоне [0, 1]")

    image = read_image_from_upload(file)

    detections = predict_single_image(
        model=MODEL,
        image=image,
        id_to_label=ID_TO_LABEL,
        device=DEVICE,
        score_threshold=score_threshold,
    )

    response_dict = build_prediction_response(
        image=image,
        detections=detections,
        model_name=MODEL_NAME,
        score_threshold=score_threshold,
    )

    return PredictResponse(**response_dict)



def _load_prediction_payload(
    prediction_json_text: str | None,
    prediction_json_file: UploadFile | None,
) -> dict:
    if prediction_json_text:
        try:
            return json.loads(prediction_json_text)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Не удалось распарсить JSON из текста: {exc}",
            ) from exc

    if prediction_json_file is not None:
        try:
            content = prediction_json_file.file.read()
            return json.loads(content.decode("utf-8"))
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Не удалось распарсить JSON-файл: {exc}",
            ) from exc

    raise HTTPException(
        status_code=400,
        detail="Передайте prediction_json_text или prediction_json_file",
    )

@app.post(
    "/visualize-detections",
    tags=["visualization"],
    summary="Нарисовать bbox на изображении по JSON-ответу /predict",
)
def visualize_detections(
    image_file: UploadFile = File(...),
    prediction_json_text: str | None = Form(None),
    prediction_json_file: UploadFile | None = File(None),
):
    if not image_file.filename:
        raise HTTPException(status_code=400, detail="Изображение не было передано")

    image = read_image_from_upload(image_file)

    prediction_payload = _load_prediction_payload(
        prediction_json_text=prediction_json_text,
        prediction_json_file=prediction_json_file,
    )

    detections = prediction_payload.get("detections")
    if not isinstance(detections, list):
        raise HTTPException(
            status_code=400,
            detail="В JSON отсутствует поле detections или оно не является списком",
        )

    result_image = draw_detections(image, detections)

    buffer = BytesIO()
    result_image.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")