from __future__ import annotations

import json
from pathlib import Path
import typer
import uvicorn
import torch
from PIL import Image
from urllib.request import urlretrieve

from src.data.paths import PROJECT_DIR
from src.models import DetectionTrainConfig, fit_detection_model, load_detector_from_artifacts
from src.models.predict_detection import(
    build_prediction_response,
    draw_detections,
    predict_single_image,
    read_image_from_path,
)

app = typer.Typer(help="CLI для управления detection-сервисом")


ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "models" / "best_detector.pt"
META_PATH = ARTIFACTS_DIR / "metrics" / "best_detector_meta.json"

DEFAULT_WEIGHTS_URL = (
    "https://github.com/damaxon/aie_course_rep/"
    "releases/download/v0.7/best_detector.pt"
)

@app.command()
def run_api(
    host: str = typer.Option("127.0.0.1", help="Хост для запуска API"),
    port: int = typer.Option(8000, help="Порт для запуска API"),
    reload: bool = typer.Option(True, help="Автоперезагрузка при изменениях кода"),
) -> None:
    """
    Запустить FastAPI detection-сервис через uvicorn.
    """
    uvicorn.run(
        "src.api:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def check_artifacts() -> None:
    """
    Проверить наличие основных артефактов модели.
    """
    typer.echo(f"Model path: {MODEL_PATH}")
    typer.echo(f"Exists: {MODEL_PATH.exists()}")

    typer.echo(f"\nMeta path: {META_PATH}")
    typer.echo(f"Exists: {META_PATH.exists()}")

    if MODEL_PATH.exists() and META_PATH.exists():
        typer.echo("\nArtifacts look OK")
    else:
        typer.echo("\nArtifacts are missing")
        raise typer.Exit(code=1)
    

@app.command()
def train_detector(
    epochs: int = typer.Option(10, help="Количество эпох обучения"),
    batch_size: int = typer.Option(4, help="Batch size"),
    lr: float = typer.Option(1e-4, help="Learning rate"),
    weight_decay: float = typer.Option(1e-4, help="Weight decay"),
    patience: int = typer.Option(3, help="Early stopping patience"),
    val_split: float = typer.Option(0.2, help="Доля validation-выборки"),
    num_workers: int = typer.Option(0, help="Количество DataLoader workers"),
    seed: int = typer.Option(42, help="Seed для воспроизводимости"),
    score_threshold: float = typer.Option(0.3, help="Score threshold для evaluation"),
    iou_threshold: float = typer.Option(0.5, help="IoU threshold для evaluation"),
    pretrained: bool = typer.Option(True, help="Использовать pretrained weights"),
    use_amp: bool = typer.Option(True, help="Использовать AMP при доступной CUDA"),
    promote_to_best: bool = typer.Option(
        False,
        help="Заменить текущую best-модель результатом обучения",
    ),
    confirm_retrain: bool = typer.Option(
        False,
        help="Подтверждение запуска переобучения",
    ),
    confirm_long_run: bool = typer.Option(
        False,
        help="Подтверждение длительного и ресурсоёмкого обучения",
    ),
    confirm_overwrite_best: bool = typer.Option(
        False,
        help="Подтверждение возможной замены текущей best-модели",
    ),
) -> None:
    """
    Безопасно запустить повторное обучение detection-модели.

    По умолчанию обучение не запускается без явных подтверждений.
    """
    typer.echo("=== WARNING ===")
    typer.echo("Вы собираетесь запустить обучение detection-модели.")
    typer.echo("Это может занять несколько часов и использовать значительные ресурсы GPU/CPU.")
    typer.echo("По умолчанию текущая best-модель НЕ будет заменена.")
    typer.echo("")

    if not confirm_retrain:
        typer.echo("Запуск остановлен: не передан флаг --confirm-retrain")
        raise typer.Exit(code=1)

    if not confirm_long_run:
        typer.echo("Запуск остановлен: не передан флаг --confirm-long-run")
        raise typer.Exit(code=1)

    if promote_to_best and not confirm_overwrite_best:
        typer.echo(
            "Запуск остановлен: для замены best-модели требуется "
            "--confirm-overwrite-best"
        )
        raise typer.Exit(code=1)

    config = DetectionTrainConfig(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        val_split=val_split,
        num_workers=num_workers,
        seed=seed,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
        pretrained=pretrained,
        use_amp=use_amp,
        promote_to_best=promote_to_best,
        confirm_retrain=confirm_retrain,
        confirm_long_run=confirm_long_run,
        confirm_overwrite_best=confirm_overwrite_best,
    )

    typer.echo("=== TRAIN CONFIG ===")
    typer.echo(f"epochs: {config.epochs}")
    typer.echo(f"batch_size: {config.batch_size}")
    typer.echo(f"lr: {config.lr}")
    typer.echo(f"weight_decay: {config.weight_decay}")
    typer.echo(f"patience: {config.patience}")
    typer.echo(f"val_split: {config.val_split}")
    typer.echo(f"seed: {config.seed}")
    typer.echo(f"device: {config.device}")
    typer.echo(f"promote_to_best: {config.promote_to_best}")
    typer.echo("")

    result = fit_detection_model(config)

    typer.echo("=== TRAINING FINISHED ===")
    typer.echo(f"Run dir: {result['run_dir']}")
    typer.echo(f"Best model path: {result['best_model_path']}")
    typer.echo(f"Best meta path: {result['best_meta_path']}")
    typer.echo(f"Best epoch: {result['best_epoch']}")
    typer.echo(f"Best val mAP@0.5: {result['best_val_map_50']}")

    if promote_to_best:
        typer.echo("")
        typer.echo("Текущая best-модель была заменена новым результатом обучения.")
    else:
        typer.echo("")
        typer.echo("Текущая best-модель НЕ заменялась.")
        typer.echo("Новая модель сохранена только в run-директории.")


@app.command()
def paths() -> None:
    """
    Показать ключевые пути проекта.
    """
    typer.echo(f"PROJECT_DIR: {PROJECT_DIR}")
    typer.echo(f"ARTIFACTS_DIR: {ARTIFACTS_DIR}")
    typer.echo(f"MODEL_PATH: {MODEL_PATH}")
    typer.echo(f"META_PATH: {META_PATH}")


@app.command()
def predict_image(
    image: Path = typer.Option(
        ...,
        "--image",
        "-i",
        help="Путь к изображению для inference",
    ),
    score_threshold: float = typer.Option(
        0.3,
        "--score-threshold",
        "-t",
        help="Порог confidence score",
    ),
    output_json: Path | None = typer.Option(
        None,
        "--output-json",
        help="Путь для сохранения JSON-результата",
    ),
    output_image: Path | None = typer.Option(
        None,
        "--output-image",
        help="Путь для сохранения изображения с bbox",
    ),
) -> None:
    """
    Выполнить inference для одного изображения через CLI.
    """
    if not image.exists():
        typer.echo(f"Image not found: {image}")
        raise typer.Exit(code=1)

    if not 0.0 <= score_threshold <= 1.0:
        typer.echo("score_threshold должен быть в диапазоне [0, 1]")
        raise typer.Exit(code=1)

    if not MODEL_PATH.exists() or not META_PATH.exists():
        typer.echo("Model artifacts are missing.")
        typer.echo(f"MODEL_PATH: {MODEL_PATH}")
        typer.echo(f"META_PATH: {META_PATH}")
        typer.echo("Скачайте best_detector.pt из GitHub Releases и поместите его в artifacts/models/")
        raise typer.Exit(code=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    typer.echo("=== LOADING MODEL ===")
    typer.echo(f"Device: {device}")
    typer.echo(f"Model path: {MODEL_PATH}")
    typer.echo(f"Meta path: {META_PATH}")

    bundle = load_detector_from_artifacts(
        weights_path=MODEL_PATH,
        meta_path=META_PATH,
        device=device,
    )

    model = bundle["model"]
    id_to_label = bundle["id_to_label"]
    model_name = bundle["model_name"]

    typer.echo("=== RUNNING PREDICTION ===")
    typer.echo(f"Image: {image}")
    typer.echo(f"Score threshold: {score_threshold}")

    pil_image = read_image_from_path(image)

    detections = predict_single_image(
        model=model,
        image=pil_image,
        id_to_label=id_to_label,
        device=device,
        score_threshold=score_threshold,
    )

    response_dict = build_prediction_response(
        image=pil_image,
        detections=detections,
        model_name=model_name,
        score_threshold=score_threshold,
    )

    json_text = json.dumps(response_dict, ensure_ascii=False, indent=2)

    typer.echo("=== PREDICTION RESULT ===")
    typer.echo(json_text)

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)

        with output_json.open("w", encoding="utf-8") as f:
            f.write(json_text)

        typer.echo(f"JSON saved to: {output_json}")

    if output_image is not None:
        output_image.parent.mkdir(parents=True, exist_ok=True)

        result_image: Image.Image = draw_detections(
            image=pil_image,
            detections=detections,
        )
        result_image.save(output_image)

        typer.echo(f"Image with bbox saved to: {output_image}")


@app.command()
def download_weights(
    url: str = typer.Option(
        DEFAULT_WEIGHTS_URL,
        "--url",
        help="URL для скачивания весов модели",
    ),
    output_path: Path = typer.Option(
        MODEL_PATH,
        "--output-path",
        help="Куда сохранить файл весов модели",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Перезаписать существующий файл весов",
    ),
) -> None:
    """
    Скачать веса модели из GitHub Releases.
    """
    output_path = Path(output_path)

    if output_path.exists() and not force:
        typer.echo("Model weights already exist.")
        typer.echo(f"Path: {output_path}")
        typer.echo("Use --force to overwrite.")
        raise typer.Exit(code=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    typer.echo("=== DOWNLOADING MODEL WEIGHTS ===")
    typer.echo(f"URL: {url}")
    typer.echo(f"Output path: {output_path}")

    try:
        urlretrieve(url, output_path)
    except Exception as exc:
        typer.echo(f"Failed to download model weights: {exc}")
        raise typer.Exit(code=1) from exc

    if not output_path.exists():
        typer.echo("Download finished, but output file was not found.")
        raise typer.Exit(code=1)

    size_mb = output_path.stat().st_size / (1024 * 1024)

    typer.echo("=== DOWNLOAD FINISHED ===")
    typer.echo(f"Saved to: {output_path}")
    typer.echo(f"Size: {size_mb:.2f} MB")


if __name__ == "__main__":
    app()