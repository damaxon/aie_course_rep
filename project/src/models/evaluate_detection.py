from __future__ import annotations

from typing import Any

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def box_iou_single(box_a: torch.Tensor, box_b: torch.Tensor) -> float:

    x_left = max(float(box_a[0]), float(box_b[0]))
    y_top = max(float(box_a[1]), float(box_b[1]))
    x_right = min(float(box_a[2]), float(box_b[2]))
    y_bottom = min(float(box_a[3]), float(box_b[3]))

    inter_w = max(0.0, x_right - x_left)
    inter_h = max(0.0, y_bottom - y_top)
    inter_area = inter_w * inter_h

    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0

    return inter_area / union


def match_predictions_to_targets(
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    pred_scores: torch.Tensor,
    true_boxes: torch.Tensor,
    true_labels: torch.Tensor,
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:

    if len(pred_boxes) == 0 and len(true_boxes) == 0:
        return 0, 0, 0

    if len(pred_boxes) == 0:
        return 0, 0, len(true_boxes)

    if len(true_boxes) == 0:
        return 0, len(pred_boxes), 0

    order = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[order]
    pred_labels = pred_labels[order]

    matched_gt = set()
    tp = 0
    fp = 0

    for pred_idx in range(len(pred_boxes)):
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx in range(len(true_boxes)):
            if gt_idx in matched_gt:
                continue

            if int(pred_labels[pred_idx]) != int(true_labels[gt_idx]):
                continue

            iou = box_iou_single(pred_boxes[pred_idx], true_boxes[gt_idx])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(true_boxes) - len(matched_gt)
    return tp, fp, fn


@torch.no_grad()
def evaluate_detection(
    model,
    dataloader,
    device: str | torch.device = "cpu",
    score_threshold: float = 0.3,
    iou_threshold: float = 0.5,
) -> dict[str, float]:

    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for output, target in zip(outputs, targets):
            pred_boxes = output["boxes"].detach().cpu()
            pred_labels = output["labels"].detach().cpu()
            pred_scores = output["scores"].detach().cpu()

            keep = pred_scores >= score_threshold
            pred_boxes = pred_boxes[keep]
            pred_labels = pred_labels[keep]
            pred_scores = pred_scores[keep]

            true_boxes = target["boxes"].detach().cpu()
            true_labels = target["labels"].detach().cpu()

            tp, fp, fn = match_predictions_to_targets(
                pred_boxes=pred_boxes,
                pred_labels=pred_labels,
                pred_scores=pred_scores,
                true_boxes=true_boxes,
                true_labels=true_labels,
                iou_threshold=iou_threshold,
            )

            total_tp += tp
            total_fp += fp
            total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision@0.5": precision,
        "recall@0.5": recall,
        "f1@0.5": f1,
    }


def prepare_for_map(
    outputs: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    score_threshold: float = 0.0,
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]]:

    preds_for_map = []
    targets_for_map = []

    for output, target in zip(outputs, targets):
        pred_boxes = output["boxes"].detach().cpu()
        pred_scores = output["scores"].detach().cpu()
        pred_labels = output["labels"].detach().cpu()

        keep = pred_scores >= score_threshold

        preds_for_map.append(
            {
                "boxes": pred_boxes[keep],
                "scores": pred_scores[keep],
                "labels": pred_labels[keep],
            }
        )

        targets_for_map.append(
            {
                "boxes": target["boxes"].detach().cpu(),
                "labels": target["labels"].detach().cpu(),
            }
        )

    return preds_for_map, targets_for_map


@torch.no_grad()
def evaluate_detection_map(
    model,
    dataloader,
    device: str | torch.device = "cpu",
    score_threshold: float = 0.0,
) -> dict[str, float]:

    model.eval()

    metric = MeanAveragePrecision()

    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        preds_for_map, targets_for_map = prepare_for_map(
            outputs=outputs,
            targets=targets,
            score_threshold=score_threshold,
        )

        metric.update(preds_for_map, targets_for_map)

    results = metric.compute()

    return {
        "map": float(results["map"]),
        "map_50": float(results["map_50"]),
        "map_75": float(results["map_75"]),
        "mar_1": float(results["mar_1"]),
        "mar_10": float(results["mar_10"]),
        "mar_100": float(results["mar_100"]),
    }


@torch.no_grad()
def run_full_detection_evaluation(
    model,
    dataloader,
    device: str | torch.device = "cpu",
    score_threshold: float = 0.3,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:

    prf_metrics = evaluate_detection(
        model=model,
        dataloader=dataloader,
        device=device,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
    )

    map_metrics = evaluate_detection_map(
        model=model,
        dataloader=dataloader,
        device=device,
        score_threshold=0.0,
    )

    return {
        **prf_metrics,
        **map_metrics,
    }