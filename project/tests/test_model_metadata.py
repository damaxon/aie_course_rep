import json

from src.data.paths import PROJECT_DIR


def test_best_detector_metadata_contract():
    meta_path = PROJECT_DIR / "artifacts" / "metrics" / "best_detector_meta.json"

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    required_keys = {
        "task",
        "dataset",
        "best_model_name",
        "num_classes_including_background",
        "classes",
        "best_val_map@0.5",
    }

    assert required_keys.issubset(meta.keys())
    assert meta["task"] == "detection"
    assert meta["num_classes_including_background"] == len(meta["classes"]) + 1