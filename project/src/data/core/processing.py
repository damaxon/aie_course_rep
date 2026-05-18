import pandas as pd
import shutil

from pathlib import Path
from ..paths import DATA_RAW_DIR, DATA_PROCESSED_DIR

def prepare_detection() -> dict:
    raw_dir = DATA_RAW_DIR / "Udacity Self Driving Car Dataset"

    src_images = raw_dir / "images"
    src_labels = raw_dir / "labels"

    dst_dir = DATA_PROCESSED_DIR / "detection"
    dst_images = dst_dir / "images"
    
    dst_images.mkdir(parents=True, exist_ok=True)

    copied_images = 0
    for img in src_images.glob("*"):
        dst_path = dst_images / img.name
        if not dst_path.exists():
            shutil.copy2(img, dst_path)
            copied_images += 1
    
    csv_files = list(src_labels.glob("*.csv"))
    if len(csv_files) != 1:
        raise ValueError(
            f"Ожидался ровно один CSV-файл с аннотациями в {src_labels}, найдено: {len(csv_files)}"
        )
    
    annotations_path = csv_files[0]
    annotations_df = pd.read_csv(annotations_path)

    expected_columns = {
        "filename": "image",
        "class": "label",
        "xmin": "xmin",
        "ymin": "ymin",
        "xmax": "xmax",
        "ymax": "ymax",
        "width": "image_width",
        "height": "image_height",
    }

    missing_columns = [col for col in expected_columns if col not in annotations_df.columns]
    if missing_columns:
        raise ValueError(
            f"В CSV отсутствуют ожидаемые колонки: {missing_columns}. "
            f"Фактические колонки: {annotations_df.columns.tolist()}"
        )
    
    processed_df = annotations_df.rename(columns=expected_columns)[
        ["image", "label", "xmin", "ymin", "xmax", "ymax", "image_width", "image_height"]
    ].copy()

    processed_df.to_csv(dst_dir / "annotations.csv", index=False)

    return {
        "copied_images": copied_images,
        "annotations_created": True,
        "num_annotations": len(processed_df),
    }

def prepare_all() -> dict:
    return {
         "detection": prepare_detection()
    }