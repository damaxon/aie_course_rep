from src.data.loaders import create_detection_dataloaders
from src.data.paths import DATA_PROCESSED_DIR


def test_create_detection_dataloaders():
    train_loader, val_loader, label_to_id, id_to_label = create_detection_dataloaders(
        images_dir=DATA_PROCESSED_DIR / "detection" / "images",
        annotations_csv=DATA_PROCESSED_DIR / "detection" / "annotations.csv",
        batch_size=2,
        val_split=0.2,
        num_workers=0,
        seed=42,
    )

    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(label_to_id) > 0
    assert len(id_to_label) > 0

    images, targets = next(iter(train_loader))

    assert isinstance(images, list)
    assert isinstance(targets, list)
    assert "boxes" in targets[0]
    assert "labels" in targets[0]