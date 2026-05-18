from pathlib import Path

import pandas as pd

from src.data.paths import DATA_PROCESSED_DIR


def test_detection_processed_files_exist():
    detection_dir = DATA_PROCESSED_DIR / "detection"

    assert detection_dir.exists()
    assert (detection_dir / "images").exists()
    assert (detection_dir / "annotations.csv").exists()


def test_detection_annotations_have_required_columns():
    annotations_path = DATA_PROCESSED_DIR / "detection" / "annotations.csv"
    df = pd.read_csv(annotations_path)

    required_columns = {
        "image",
        "label",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "image_width",
        "image_height",
    }

    assert required_columns.issubset(df.columns)
    assert len(df) > 0


def test_detection_bbox_coordinates_are_valid():
    annotations_path = DATA_PROCESSED_DIR / "detection" / "annotations.csv"
    df = pd.read_csv(annotations_path)

    assert (df["xmax"] > df["xmin"]).all()
    assert (df["ymax"] > df["ymin"]).all()