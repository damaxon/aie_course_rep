from dataclasses import dataclass
import random

import torch
from PIL import Image
from torchvision.transforms import functional as F

@dataclass
class DetectionTrainTransform:
    
    flip_prob: float = 0.5

    def __call__(self, image: Image.Image, target: dict):
        image = F.to_tensor(image)

        if random.random() < self.flip_prob:
            _, height, width = image.shape
            image = F.hflip(image)

            boxes = target["boxes"].clone()
            x_min = boxes[:, 0].clone()
            x_max = boxes[:, 2].clone()

            boxes[:, 0] = width - x_max
            boxes[:, 2] = width - x_min

            target = target.copy()
            target["boxes"] = boxes

        return image, target


@dataclass
class DetectionValTransform:
    def __call__(self, image: Image.Image, target: dict):
        image = F.to_tensor(image)
        return image, target