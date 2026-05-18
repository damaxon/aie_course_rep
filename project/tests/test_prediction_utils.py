from PIL import Image

from src.models.predict_detection import draw_detections


def test_draw_detections_returns_image():
    image = Image.new("RGB", (300, 200), color="white")

    detections = [
        {
            "label_id": 1,
            "label_name": "car",
            "score": 0.95,
            "bbox": {
                "xmin": 10,
                "ymin": 20,
                "xmax": 100,
                "ymax": 120,
            },
        }
    ]

    result = draw_detections(image, detections)

    assert isinstance(result, Image.Image)
    assert result.size == image.size