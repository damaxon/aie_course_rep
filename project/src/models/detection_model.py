from __future__ import annotations

import torch.nn as nn
import torchvision

def build_detection_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    
    if model_name == "fasterrcnn_resnet50_fpn":
        weights = "DEFAULT" if pretrained else None

        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=weights,
            weights_backbone=None,
        )

        in_features = model.roi_heads.box_predictor.cls_score.in_features

        model.roi_heads.box_predictor = (
            torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                in_features,
                num_classes + 1,  # + background
            )
        )

        return model

    if model_name == "fasterrcnn_mobilenet_v3_large_fpn":
        weights = "DEFAULT" if pretrained else None

        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            weights=weights,
            weights_backbone=None,
        )

        in_features = model.roi_heads.box_predictor.cls_score.in_features

        model.roi_heads.box_predictor = (
            torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                in_features,
                num_classes + 1,
            )
        )

        return model

    if model_name == "retinanet_resnet50_fpn":
        weights = "DEFAULT" if pretrained else None

        model = torchvision.models.detection.retinanet_resnet50_fpn(
            weights=weights,
            weights_backbone=None,
        )

        num_anchors = model.head.classification_head.num_anchors
        in_channels = model.head.classification_head.conv[0][0].in_channels

        model.head.classification_head = (
            torchvision.models.detection.retinanet.RetinaNetClassificationHead(
                in_channels=in_channels,
                num_anchors=num_anchors,
                num_classes=num_classes,
            )
        )

        return model

    raise ValueError(f"Unknown model_name: {model_name}")