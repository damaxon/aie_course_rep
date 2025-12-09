from __future__ import annotations

from time import perf_counter

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Dict,List,Any

from .core import compute_quality_flags, missing_table, summarize_dataset
from .logger import log_request

app = FastAPI(
    title="AIE Dataset Quality API",
    version="0.2.0",
    description=(
        "HTTP-сервис-заглушка для оценки готовности датасета к обучению модели. "
        "Использует простые эвристики качества данных вместо настоящей ML-модели."
    ),
    docs_url="/docs",
    redoc_url=None,
)


# ---------- Модели запросов/ответов ----------


class QualityRequest(BaseModel):
    """Агрегированные признаки датасета – 'фичи' для заглушки модели."""

    n_rows: int = Field(..., ge=0, description="Число строк в датасете")
    n_cols: int = Field(..., ge=0, description="Число колонок")
    max_missing_share: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Максимальная доля пропусков среди всех колонок (0..1)",
    )
    numeric_cols: int = Field(
        ...,
        ge=0,
        description="Количество числовых колонок",
    )
    categorical_cols: int = Field(
        ...,
        ge=0,
        description="Количество категориальных колонок",
    )


class QualityResponse(BaseModel):
    """Ответ заглушки модели качества датасета."""

    ok_for_model: bool = Field(
        ...,
        description="True, если датасет считается достаточно качественным для обучения модели",
    )
    quality_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Интегральная оценка качества данных (0..1)",
    )
    message: str = Field(
        ...,
        description="Человекочитаемое пояснение решения",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Время обработки запроса на сервере, миллисекунды",
    )
    flags: dict[str, bool] | None = Field(
        default=None,
        description="Булевы флаги с подробностями (например, too_few_rows, too_many_missing)",
    )
    dataset_shape: dict[str, int] | None = Field(
        default=None,
        description="Размеры датасета: {'n_rows': ..., 'n_cols': ...}, если известны",
    )


class QualityFlagsResponse(BaseModel):
    """Ответ заглушки модели флагов качества датасета."""
    flags: dict[str, bool] | None = Field(
        default=None,
        description="Булевы флаги с подробностями (например, too_few_rows, too_many_missing)",
    )


class HeadResponse(BaseModel):
    """Модель ответа для head запроса"""
    n_rows: int = Field(
        ..., 
        ge=0, 
        description="Число строк в датасете")
    n_cols: int = Field(
        ..., 
        ge=0, 
        description="Число колонок")
    rows_displayed: int = Field(
        ...,
        ge=0,
        description= "Число показаных строк")
    data: List[Dict[str, Any]] = Field(
        default=None,
        description="Информация из CSV",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Время обработки запроса на сервере, миллисекунды",
    )

# ---------- Системный эндпоинт ----------


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Простейший health-check сервиса."""
    return {
        "status": "ok",
        "service": "dataset-quality",
        "version": "0.2.0",
    }


# ---------- Заглушка /quality по агрегированным признакам ----------


@app.post("/quality", response_model=QualityResponse, tags=["quality"])
def quality(req: QualityRequest) -> QualityResponse:
    """
    Эндпоинт-заглушка, который принимает агрегированные признаки датасета
    и возвращает эвристическую оценку качества.
    """

    start = perf_counter()

    # Базовый скор от 0 до 1
    score = 1.0

    # Чем больше пропусков, тем хуже
    score -= req.max_missing_share

    # Штраф за слишком маленький датасет
    if req.n_rows < 1000:
        score -= 0.2

    # Штраф за слишком широкий датасет
    if req.n_cols > 100:
        score -= 0.1

    # Штрафы за перекос по типам признаков (если есть числовые и категориальные)
    if req.numeric_cols == 0 and req.categorical_cols > 0:
        score -= 0.1
    if req.categorical_cols == 0 and req.numeric_cols > 0:
        score -= 0.05

    # Нормируем скор в диапазон [0, 1]
    score = max(0.0, min(1.0, score))

    # Простое решение "ок / не ок"
    ok_for_model = score >= 0.7
    if ok_for_model:
        message = "Данных достаточно, модель можно обучать (по текущим эвристикам)."
    else:
        message = "Качество данных недостаточно, требуется доработка (по текущим эвристикам)."

    latency_ms = (perf_counter() - start) * 1000.0

    # Флаги, которые могут быть полезны для последующего логирования/аналитики
    flags = {
        "too_few_rows": req.n_rows < 1000,
        "too_many_columns": req.n_cols > 100,
        "too_many_missing": req.max_missing_share > 0.5,
        "no_numeric_columns": req.numeric_cols == 0,
        "no_categorical_columns": req.categorical_cols == 0,
    }

    # logger
    log_request(
        endpoint="/quality",
        status="success" if ok_for_model else "warning",
        latency_ms=latency_ms,
        ok_for_model=ok_for_model,
        n_rows=req.n_rows,
        n_cols=req.n_cols,
        message=f"score={score:.3f} max_missing_share={req.max_missing_share:.3f}",
    )

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags,
        dataset_shape={"n_rows": req.n_rows, "n_cols": req.n_cols},
    )


# ---------- /quality-from-csv: реальный CSV через нашу EDA-логику ----------


@app.post(
    "/quality-from-csv",
    response_model=QualityResponse,
    tags=["quality"],
    summary="Оценка качества по CSV-файлу с использованием EDA-ядра",
)
async def quality_from_csv(file: UploadFile = File(...)) -> QualityResponse:
    """
    Эндпоинт, который принимает CSV-файл, запускает EDA-ядро
    (summarize_dataset + missing_table + compute_quality_flags)
    и возвращает оценку качества данных.

    Именно это по сути связывает S03 (CLI EDA) и S04 (HTTP-сервис).
    """

    start = perf_counter()

    if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/octet-stream"):
        # content_type от браузера может быть разным, поэтому проверка мягкая
        # но для демонстрации оставим простую ветку 400
        latency_ms = (perf_counter() - start) * 1000.0
        log_request(
            endpoint="/quality-from-csv",
            status="error",
            latency_ms=latency_ms,
            ok_for_model=False,
            message=f"Invalid content type: {file.content_type}",
        )
        raise HTTPException(status_code=400, detail="Ожидается CSV-файл (content-type text/csv).")

    try:
        # FastAPI даёт file.file как file-like объект, который можно читать pandas'ом
        df = pd.read_csv(file.file)
    except Exception as exc:  # noqa: BLE001
        latency_ms = (perf_counter() - start) * 1000.0
        log_request(
            endpoint="/quality-from-csv",
            status="error",
            latency_ms=latency_ms,
            ok_for_model=False,
            message=f"Failed to read CSV: {exc}",
        )
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")

    if df.empty:
        latency_ms = (perf_counter() - start) * 1000.0
        log_request(
            endpoint="/quality-from-csv",
            status="error",
            latency_ms=latency_ms,
            ok_for_model=False,
            message="Empty CSV file",
        )
        raise HTTPException(status_code=400, detail="CSV-файл не содержит данных (пустой DataFrame).")

    # Используем EDA-ядро из S03
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags_all = compute_quality_flags(summary, missing_df)

    # Ожидаем, что compute_quality_flags вернёт quality_score в [0,1]
    score = float(flags_all.get("quality_score", 0.0))
    score = max(0.0, min(1.0, score))
    ok_for_model = score >= 0.7

    if ok_for_model:
        message = "CSV выглядит достаточно качественным для обучения модели (по текущим эвристикам)."
    else:
        message = "CSV требует доработки перед обучением модели (по текущим эвристикам)."

    latency_ms = (perf_counter() - start) * 1000.0

    # Оставляем только булевы флаги для компактности
    flags_bool: dict[str, bool] = {
        key: bool(value)
        for key, value in flags_all.items()
        if isinstance(value, bool)
    }

    # Размеры датасета берём из summary (если там есть поля n_rows/n_cols),
    # иначе — напрямую из DataFrame.
    try:
        n_rows = int(getattr(summary, "n_rows"))
        n_cols = int(getattr(summary, "n_cols"))
    except AttributeError:
        n_rows = int(df.shape[0])
        n_cols = int(df.shape[1])

    log_request(
        endpoint="/quality-from-csv",
        status="success" if ok_for_model else "warning",
        latency_ms=latency_ms,
        ok_for_model=ok_for_model,
        n_rows=n_rows,
        n_cols=n_cols,
        message=f"filename={file.filename!r} score={score:.3f}",
    )

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags_bool,
        dataset_shape={"n_rows": n_rows, "n_cols": n_cols},
    )


# ---------- /quality-flags-from-csv: реальный CSV через нашу EDA-логику для вывода флагов качества ----------


@app.post(
    "/quality-flags-from-csv",
    response_model=QualityFlagsResponse,
    tags=["quality"],
    summary="Оценка флагов качества по CSV-файлу с использованием EDA-ядра",
)
async def quality_flags_from_csv(file: UploadFile = File(...)) -> QualityFlagsResponse:
    """
    Эндпоинт, который принимает CSV-файл, запускает EDA-ядро
    (summarize_dataset + missing_table + compute_quality_flags)
    и возвращает оценку флагов качества данных.
    """

    start = perf_counter()

    if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/octet-stream"):
        # content_type от браузера может быть разным, поэтому проверка мягкая
        # но для демонстрации оставим простую ветку 400
        latency_ms = (perf_counter() - start) * 1000.0
        log_request(
            endpoint="/quality-flags-from-csv",
            status="error",
            latency_ms=latency_ms,
            ok_for_model=False,
            message=f"Invalid content type: {file.content_type}",
        )
        raise HTTPException(status_code=400, detail="Ожидается CSV-файл (content-type text/csv).")

    try:
        # FastAPI даёт file.file как file-like объект, который можно читать pandas'ом
        df = pd.read_csv(file.file)
    except Exception as exc:  # noqa: BLE001
        latency_ms = (perf_counter() - start) * 1000.0
        log_request(
            endpoint="/quality-flags-from-csv",
            status="error",
            latency_ms=latency_ms,
            ok_for_model=False,
            message=f"Failed to read CSV: {exc}",
        )
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")

    if df.empty:
        latency_ms = (perf_counter() - start) * 1000.0
        log_request(
            endpoint="/quality-flags-from-csv",
            status="error",
            latency_ms=latency_ms,
            ok_for_model=False,
            message="Empty CSV file",
        )
        raise HTTPException(status_code=400, detail="CSV-файл не содержит данных (пустой DataFrame).")

    # Используем EDA-ядро из S03
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags_all = compute_quality_flags(summary, missing_df)

    # Оставляем только булевы флаги для компактности
    flags_bool: dict[str, bool] = {
        key: bool(value)
        for key, value in flags_all.items()
        if isinstance(value, bool)
    }

    latency_ms = (perf_counter() - start) * 1000.0

    log_request(
        endpoint="/quality-flags-from-csv",
        status="success",
        latency_ms=latency_ms,
        ok_for_model=True,
        n_rows=len(df),
        n_cols=len(df.columns),
        message=f"filename={file.filename!r} flags_count={len(flags_bool)}",
    )

    return QualityFlagsResponse(
        flags=flags_bool,
    )


# ---------- /head-from-csv: вывод первых n-строк реального CSV ----------


@app.post(
    "/head-from-csv",
    response_model=HeadResponse,
    tags=["eda"],
    summary="Вывод первых n-строк из CSV-файла с использованием EDA-ядра",
)
async def head_from_csv(file: UploadFile = File(...),n: int=5) -> HeadResponse:
    """
    Эндпоинт, который принимает CSV-файл, запускает EDA-ядро
    и возвращает первые n-строк файла.
    """

    start = perf_counter()

    if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/octet-stream"):
        # content_type от браузера может быть разным, поэтому проверка мягкая
        # но для демонстрации оставим простую ветку 400
        latency_ms = (perf_counter() - start) * 1000.0
        log_request(
            endpoint="/head-from-csv",
            status="error",
            latency_ms=latency_ms,
            ok_for_model=False,
            message=f"Invalid content type: {file.content_type}",
        )
        raise HTTPException(status_code=400, detail="Ожидается CSV-файл (content-type text/csv).")

    try:
        # FastAPI даёт file.file как file-like объект, который можно читать pandas'ом
        df = pd.read_csv(file.file)
    except Exception as exc:  # noqa: BLE001
        latency_ms = (perf_counter() - start) * 1000.0
        log_request(
            endpoint="/head-from-csv",
            status="error",
            latency_ms=latency_ms,
            ok_for_model=False,
            message=f"Failed to read CSV: {exc}",
        )
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")

    if df.empty:
        latency_ms = (perf_counter() - start) * 1000.0
        log_request(
            endpoint="/head-from-csv",
            status="error",
            latency_ms=latency_ms,
            ok_for_model=False,
            message="Empty CSV file",
        )
        raise HTTPException(status_code=400, detail="CSV-файл не содержит данных (пустой DataFrame).")


    head_df = df.head(n)
    data_list = head_df.to_dict(orient='records')

    for row in data_list:
        for key, value in row.items():
            if pd.isna(value):
                row[key] = None
            elif isinstance(value, (pd.Timestamp, datetime)):
                row[key] = value.isoformat()

    n_rows = int(df.shape[0])
    n_cols = int(df.shape[1])
    rows_displayed=min(n, len(df))

    latency_ms = (perf_counter() - start) * 1000.0

    log_request(
        endpoint="/head-from-csv",
        status="success",
        latency_ms=latency_ms,
        ok_for_model=True,
        n_rows=n_rows,
        n_cols=n_cols,
        message=f"filename={file.filename!r} rows_displayed={rows_displayed}",
    )

    return HeadResponse(
        n_rows=n_rows,
        n_cols=n_cols,
        rows_displayed=rows_displayed,
        data=data_list,
        latency_ms = latency_ms,
    )