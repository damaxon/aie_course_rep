from __future__ import annotations

from time import perf_counter
from uuid import uuid4
from src.logger import logger
import json
from pathlib import Path
from io import BytesIO

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, Form, Request
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

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid4())
    start = perf_counter()

    client_host = request.client.host if request.client else "unknown"

    logger.info(
        "request_started | request_id=%s | method=%s | path=%s | client=%s",
        request_id,
        request.method,
        request.url.path,
        client_host,
    )

    try:
        response = await call_next(request)
    except Exception as exc:
        latency_ms = (perf_counter() - start) * 1000

        logger.exception(
            "request_failed | request_id=%s | method=%s | path=%s | "
            "client=%s | latency_ms=%.2f | error=%s",
            request_id,
            request.method,
            request.url.path,
            client_host,
            latency_ms,
            exc,
        )

        raise

    latency_ms = (perf_counter() - start) * 1000

    logger.info(
        "request_finished | request_id=%s | method=%s | path=%s | "
        "status_code=%s | latency_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
    )

    response.headers["X-Request-ID"] = request_id

    return response


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
    logger.info(
        "loading_detector_artifacts | model_path=%s | meta_path=%s | device=%s",
        MODEL_PATH,
        META_PATH,
        DEVICE,
    )

    bundle = load_detector_from_artifacts(
        weights_path=MODEL_PATH,
        meta_path=META_PATH,
        device=DEVICE,
    )

    logger.info("detector_artifacts_loaded_successfully")

    return bundle


SERVICE_BUNDLE = _load_service_bundle()
MODEL = SERVICE_BUNDLE["model"]
ID_TO_LABEL = SERVICE_BUNDLE["id_to_label"]
MODEL_NAME = SERVICE_BUNDLE["model_name"]
CLASS_NAMES = SERVICE_BUNDLE["class_names"]

logger.info(
    "service_initialized | model_name=%s | device=%s | num_classes=%s | classes=%s",
    MODEL_NAME,
    DEVICE,
    len(CLASS_NAMES),
    CLASS_NAMES,
)

logger.info(
    "artifacts_loaded | model_path=%s | meta_path=%s",
    MODEL_PATH,
    META_PATH,
)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    logger.info("health_check_requested")

    return HealthResponse(
        status="ok",
        service="vehicle-detection",
        model_name=MODEL_NAME,
        device=str(DEVICE),
        num_classes=len(CLASS_NAMES),
    )

@app.get("/info", response_model=InfoResponse, tags=["system"])
def info() -> InfoResponse:
    logger.info("model_info_requested | model_name=%s", MODEL_NAME)
    
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
    start = perf_counter()

    logger.info(
        "predict_started | filename=%s | content_type=%s | score_threshold=%.3f",
        file.filename,
        file.content_type,
        score_threshold,
    )

    if not file.filename:
        logger.warning("predict_failed | reason=empty_filename")
        raise HTTPException(status_code=400, detail="Файл не был передан")

    if not 0.0 <= score_threshold <= 1.0:
        logger.warning(
            "predict_failed | reason=invalid_score_threshold | score_threshold=%.3f",
            score_threshold,
        )
        raise HTTPException(
            status_code=400,
            detail="score_threshold должен быть в диапазоне [0, 1]",
        )

    try:
        image = read_image_from_upload(file)
    except Exception as exc:
        logger.exception(
            "predict_failed | reason=image_read_error | filename=%s | error=%s",
            file.filename,
            exc,
        )
        raise

    logger.info(
        "predict_image_loaded | filename=%s | width=%s | height=%s",
        file.filename,
        image.size[0],
        image.size[1],
    )

    try:
        detections = predict_single_image(
            model=MODEL,
            image=image,
            id_to_label=ID_TO_LABEL,
            device=DEVICE,
            score_threshold=score_threshold,
        )
    except Exception as exc:
        logger.exception(
            "predict_failed | reason=model_inference_error | filename=%s | error=%s",
            file.filename,
            exc,
        )
        raise

    response_dict = build_prediction_response(
        image=image,
        detections=detections,
        model_name=MODEL_NAME,
        score_threshold=score_threshold,
    )

    latency_ms = (perf_counter() - start) * 1000

    labels_count = {}
    for det in detections:
        label_name = det["label_name"]
        labels_count[label_name] = labels_count.get(label_name, 0) + 1

    logger.info(
        "predict_finished | filename=%s | detections=%s | labels=%s | latency_ms=%.2f",
        file.filename,
        len(detections),
        labels_count,
        latency_ms,
    )

    return PredictResponse(**response_dict)

def _load_prediction_payload(
    prediction_json_text: str | None,
    prediction_json_file: UploadFile | None,
) -> dict:
    if prediction_json_text:
        logger.info(
            "prediction_payload_loading | source=text | text_length=%s",
            len(prediction_json_text),
        )

        try:
            payload = json.loads(prediction_json_text)
            logger.info("prediction_payload_loaded | source=text")
            return payload
        except Exception as exc:
            logger.warning(
                "prediction_payload_failed | source=text | error=%s",
                exc,
            )
            raise HTTPException(
                status_code=400,
                detail=f"Не удалось распарсить JSON из текста: {exc}",
            ) from exc

    if prediction_json_file is not None:
        logger.info(
            "prediction_payload_loading | source=file | filename=%s | content_type=%s",
            prediction_json_file.filename,
            prediction_json_file.content_type,
        )

        try:
            content = prediction_json_file.file.read()
            payload = json.loads(content.decode("utf-8"))

            logger.info(
                "prediction_payload_loaded | source=file | filename=%s | bytes=%s",
                prediction_json_file.filename,
                len(content),
            )

            return payload
        except Exception as exc:
            logger.warning(
                "prediction_payload_failed | source=file | filename=%s | error=%s",
                prediction_json_file.filename,
                exc,
            )
            raise HTTPException(
                status_code=400,
                detail=f"Не удалось распарсить JSON-файл: {exc}",
            ) from exc

    logger.warning("prediction_payload_failed | reason=missing_prediction_json")

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
    start = perf_counter()

    logger.info(
        "visualize_started | image_filename=%s | image_content_type=%s | "
        "has_json_text=%s | has_json_file=%s",
        image_file.filename,
        image_file.content_type,
        prediction_json_text is not None,
        prediction_json_file is not None,
    )

    if not image_file.filename:
        logger.warning("visualize_failed | reason=empty_image_filename")
        raise HTTPException(status_code=400, detail="Изображение не было передано")

    try:
        image = read_image_from_upload(image_file)
    except Exception as exc:
        logger.exception(
            "visualize_failed | reason=image_read_error | filename=%s | error=%s",
            image_file.filename,
            exc,
        )
        raise

    logger.info(
        "visualize_image_loaded | filename=%s | width=%s | height=%s",
        image_file.filename,
        image.size[0],
        image.size[1],
    )

    prediction_payload = _load_prediction_payload(
        prediction_json_text=prediction_json_text,
        prediction_json_file=prediction_json_file,
    )

    detections = prediction_payload.get("detections")
    if not isinstance(detections, list):
        logger.warning(
            "visualize_failed | reason=invalid_detections_field | payload_keys=%s",
            list(prediction_payload.keys()),
        )
        raise HTTPException(
            status_code=400,
            detail="В JSON отсутствует поле detections или оно не является списком",
        )

    logger.info(
        "visualize_detections_loaded | detections=%s",
        len(detections),
    )

    try:
        result_image = draw_detections(image, detections)
    except Exception as exc:
        logger.exception(
            "visualize_failed | reason=draw_detections_error | error=%s",
            exc,
        )
        raise

    buffer = BytesIO()
    result_image.save(buffer, format="PNG")
    buffer.seek(0)

    latency_ms = (perf_counter() - start) * 1000

    logger.info(
        "visualize_finished | image_filename=%s | detections=%s | latency_ms=%.2f",
        image_file.filename,
        len(detections),
        latency_ms,
    )

    return StreamingResponse(buffer, media_type="image/png")