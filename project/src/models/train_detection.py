from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.cuda.amp import GradScaler, autocast

from src.data import DATA_PROCESSED_DIR, PROJECT_DIR, create_detection_dataloaders
from .detection_model import build_detection_model
from .evaluate_detection import run_full_detection_evaluation


ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
RUNS_DIR = ARTIFACTS_DIR / "runs" / "detection"
ARCHIVE_DIR = MODELS_DIR / "archive"

BEST_MODEL_PATH = MODELS_DIR / "best_detector.pt"
BEST_META_PATH = METRICS_DIR / "best_detector_meta.json"


@dataclass
class DetectionTrainConfig:
    model_name: str = "fasterrcnn_resnet50_fpn"
    annotations_csv: str = str(DATA_PROCESSED_DIR / "detection" / "annotations.csv")
    images_dir: str = str(DATA_PROCESSED_DIR / "detection" / "images")
    batch_size: int = 4
    val_split: float = 0.2
    num_workers: int = 0
    seed: int = 42

    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 10
    patience: int = 3

    score_threshold: float = 0.3
    iou_threshold: float = 0.5

    use_amp: bool = True
    pretrained: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    promote_to_best: bool = False
    confirm_retrain: bool = False
    confirm_long_run: bool = False
    confirm_overwrite_best: bool = False


def _ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _make_run_dir() -> Path:
    run_dir = RUNS_DIR / _timestamp()
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _check_retrain_guard(config: DetectionTrainConfig) -> None:
    if not config.confirm_retrain:
        raise RuntimeError(
            "Переобучение модели отключено по умолчанию. "
            "Установите confirm_retrain=True, если действительно хотите запускать обучение."
        )

    if not config.confirm_long_run:
        raise RuntimeError(
            "Обучение detection-модели — длительный и ресурсоёмкий процесс. "
            "Установите confirm_long_run=True, чтобы подтвердить это."
        )

    if config.promote_to_best and not config.confirm_overwrite_best:
        raise RuntimeError(
            "Вы пытаетесь заменить текущую лучшую модель. "
            "Установите confirm_overwrite_best=True, чтобы явно подтвердить это."
        )


def _archive_existing_best() -> None:
    if BEST_MODEL_PATH.exists():
        backup_model = ARCHIVE_DIR / f"{_timestamp()}_best_detector.pt"
        shutil.copy2(BEST_MODEL_PATH, backup_model)

    if BEST_META_PATH.exists():
        backup_meta = ARCHIVE_DIR / f"{_timestamp()}_best_detector_meta.json"
        shutil.copy2(BEST_META_PATH, backup_meta)


def _move_to_device(images, targets, device: str | torch.device):
    images = [img.to(device) for img in images]
    moved_targets = []

    for target in targets:
        moved_target = {}
        for key, value in target.items():
            if torch.is_tensor(value):
                moved_target[key] = value.to(device)
            else:
                moved_target[key] = value
        moved_targets.append(moved_target)

    return images, moved_targets


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device: str | torch.device = "cpu",
    use_amp: bool = True,
    scaler: GradScaler | None = None,
) -> float:
    model.train()

    total_loss = 0.0
    total_batches = 0

    for images, targets in dataloader:
        images, targets = _move_to_device(images, targets, device)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None and str(device).startswith("cuda"):
            with autocast():
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1

    return total_loss / total_batches if total_batches > 0 else float("nan")


@torch.no_grad()
def evaluate_one_epoch(
    model,
    dataloader,
    device: str | torch.device = "cpu",
    score_threshold: float = 0.3,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    metrics = run_full_detection_evaluation(
        model=model,
        dataloader=dataloader,
        device=device,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
    )
    return metrics


def _save_run_artifacts(
    run_dir: Path,
    model,
    metadata: dict[str, Any],
    history: dict[str, list[Any]],
) -> tuple[Path, Path, Path]:
    model_path = run_dir / "model.pt"
    meta_path = run_dir / "meta.json"
    history_path = run_dir / "history.json"

    torch.save(model.state_dict(), model_path)

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    return model_path, meta_path, history_path


def _promote_run_to_best(run_model_path: Path, run_meta_path: Path) -> None:
    _archive_existing_best()
    shutil.copy2(run_model_path, BEST_MODEL_PATH)
    shutil.copy2(run_meta_path, BEST_META_PATH)


def fit_detection_model(config: DetectionTrainConfig) -> dict[str, Any]:
    _ensure_dirs()
    _check_retrain_guard(config)

    run_dir = _make_run_dir()

    train_loader, val_loader, label_to_id, id_to_label = create_detection_dataloaders(
        images_dir=config.images_dir,
        annotations_csv=config.annotations_csv,
        batch_size=config.batch_size,
        val_split=config.val_split,
        num_workers=config.num_workers,
        seed=config.seed,
    )

    num_classes = len(label_to_id)

    model = build_detection_model(
        model_name=config.model_name,
        num_classes=num_classes,
        pretrained=config.pretrained,
    )
    model = model.to(config.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    scaler = GradScaler(enabled=config.use_amp and str(config.device).startswith("cuda"))

    history = {
        "train_loss": [],
        "val_precision@0.5": [],
        "val_recall@0.5": [],
        "val_f1@0.5": [],
        "val_map": [],
        "val_map_50": [],
        "val_map_75": [],
        "val_mar_1": [],
        "val_mar_10": [],
        "val_mar_100": [],
        "epoch_time_sec": [],
    }

    best_metric = float("-inf")
    best_epoch = -1
    best_model_path = None
    best_meta_path = None
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=config.device,
            use_amp=config.use_amp,
            scaler=scaler,
        )

        eval_metrics = evaluate_one_epoch(
            model=model,
            dataloader=val_loader,
            device=config.device,
            score_threshold=config.score_threshold,
            iou_threshold=config.iou_threshold,
        )

        dt = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_precision@0.5"].append(eval_metrics["precision@0.5"])
        history["val_recall@0.5"].append(eval_metrics["recall@0.5"])
        history["val_f1@0.5"].append(eval_metrics["f1@0.5"])
        history["val_map"].append(eval_metrics["map"])
        history["val_map_50"].append(eval_metrics["map_50"])
        history["val_map_75"].append(eval_metrics["map_75"])
        history["val_mar_1"].append(eval_metrics["mar_1"])
        history["val_mar_10"].append(eval_metrics["mar_10"])
        history["val_mar_100"].append(eval_metrics["mar_100"])
        history["epoch_time_sec"].append(dt)

        current_metric = eval_metrics["map_50"]

        metadata = {
            "task": "detection",
            "dataset": "Udacity Self Driving Car Dataset",
            "best_experiment_id": "manual_retrain",
            "best_model_name": config.model_name,
            "num_classes_including_background": num_classes + 1,
            "classes": [id_to_label[i] for i in sorted(id_to_label)],
            "image_size": 512,
            "batch_size": config.batch_size,
            "seed": config.seed,
            "device": config.device,
            "use_amp": config.use_amp,
            "fast_mode": False,
            "annotations_csv": str(config.annotations_csv),
            "images_dir": str(config.images_dir),
            "best_val_precision@0.5": eval_metrics["precision@0.5"],
            "best_val_recall@0.5": eval_metrics["recall@0.5"],
            "best_val_f1@0.5": eval_metrics["f1@0.5"],
            "best_val_map": eval_metrics["map"],
            "best_val_map@0.5": eval_metrics["map_50"],
            "best_val_map@0.75": eval_metrics["map_75"],
            "best_val_mar_1": eval_metrics["mar_1"],
            "best_val_mar_10": eval_metrics["mar_10"],
            "best_val_mar_100": eval_metrics["mar_100"],
            "label_to_id": label_to_id,
            "id_to_label": {str(k): v for k, v in id_to_label.items()},
            "config": asdict(config),
        }

        print(
            f"Epoch {epoch:02d}/{config.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_map@0.5={eval_metrics['map_50']:.4f} | "
            f"val_f1@0.5={eval_metrics['f1@0.5']:.4f} | "
            f"time={dt:.1f}s"
        )

        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            epochs_without_improvement = 0

            best_model_path, best_meta_path, _ = _save_run_artifacts(
                run_dir=run_dir,
                model=model,
                metadata=metadata,
                history=history,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    if best_model_path is None or best_meta_path is None:
        raise RuntimeError("Не удалось сохранить лучшую модель")

    if config.promote_to_best:
        _promote_run_to_best(best_model_path, best_meta_path)

    return {
        "run_dir": str(run_dir),
        "best_model_path": str(best_model_path),
        "best_meta_path": str(best_meta_path),
        "best_epoch": best_epoch,
        "best_val_map_50": best_metric,
        "history": history,
    }