#!/usr/bin/env python3
"""Compute AI2D grounding metrics (IoU/Recall/Precision) from saved predictions.

This script expects JSONL prediction files produced by infer_ai2d_grounding_lora.py.
Each line should contain at least:
- sample_id: str
- response: str (raw model output)
- gt_boxes: [[x1,y1,x2,y2], ...]

It will parse predicted boxes from the response (if you later add a parser that
saves explicit pred_boxes, you can extend this script easily) and compute:
- mean IoU over matched boxes
- Recall@0.5
- Precision@0.5
- average number of predicted boxes per sample
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

Box = Tuple[float, float, float, float]


# Legacy pattern for ID/x,y,w,h style boxes (kept for backward compatibility).
BOX_PATTERN = re.compile(
    r"(?P<id>[^/\s]+)/\s*(?P<x>-?\d+(?:\.\d*)?)\s*,\s*"  # ID/x,
    r"(?P<y>-?\d+(?:\.\d*)?)\s*,\s*"                      # y,
    r"(?P<w>-?\d+(?:\.\d*)?)\s*,\s*"                      # w,
    r"(?P<h>-?\d+(?:\.\d*)?)"                               # h
)

# New pattern for InternVL-style responses produced by infer_ai2d_grounding_lora.py,
# e.g. "<box>[[x1, y1, x2, y2]]</box>" (optionally with whitespace/newlines).
BOX_XYXY_PATTERN = re.compile(
    r"<box>\s*\[\[\s*"
    r"([+-]?\d+(?:\.\d*)?)\s*,\s*"   # x1
    r"([+-]?\d+(?:\.\d*)?)\s*,\s*"   # y1
    r"([+-]?\d+(?:\.\d*)?)\s*,\s*"   # x2
    r"([+-]?\d+(?:\.\d*)?)\s*"        # y2
    r"\]\]\s*</box>",
    re.IGNORECASE,
)

# Canonical box format used by GRPO target:
# "<box> x1 y1 x2 y2 </box>" (supports commas/whitespace separators).
BOX_XYXY_CANONICAL_PATTERN = re.compile(
    r"<box>\s*"
    r"([+-]?\d+(?:\.\d*)?)\s*[, ]+\s*"   # x1
    r"([+-]?\d+(?:\.\d*)?)\s*[, ]+\s*"   # y1
    r"([+-]?\d+(?:\.\d*)?)\s*[, ]+\s*"   # x2
    r"([+-]?\d+(?:\.\d*)?)\s*"
    r"</box>",
    re.IGNORECASE,
)

# Some outputs use <rect>x1 y1 x2 y2</rect>.
RECT_XYXY_PATTERN = re.compile(
    r"<rect>\s*"
    r"([+-]?\d+(?:\.\d*)?)\s*[, ]+\s*"
    r"([+-]?\d+(?:\.\d*)?)\s*[, ]+\s*"
    r"([+-]?\d+(?:\.\d*)?)\s*[, ]+\s*"
    r"([+-]?\d+(?:\.\d*)?)\s*"
    r"</rect>",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="JSONL predictions file")
    parser.add_argument("--bbox-threshold", type=float, default=0.5, help="IoU threshold for recall/precision")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON file to save metrics")
    return parser.parse_args()


def parse_pred_boxes_from_response(text: str) -> List[Box]:
    """Parse predicted boxes from the model response.

    Supports formats:
    1) Current InternVL grounding outputs: "<box>[[x1, y1, x2, y2]]</box>".
    2) Canonical GRPO outputs: "<box> x1 y1 x2 y2 </box>".
    3) Alternative "<rect> x1 y1 x2 y2 </rect>".
    4) Legacy ID/x,y,w,h entries (still supported via BOX_PATTERN).
    Returns a list of boxes in xyxy coordinates.
    """
    boxes: List[Box] = []

    # Prefer the explicit xyxy format used in infer_ai2d_grounding_lora.py.
    for m in BOX_XYXY_PATTERN.finditer(text):
        try:
            x1 = float(m.group(1))
            y1 = float(m.group(2))
            x2 = float(m.group(3))
            y2 = float(m.group(4))
        except ValueError:
            continue
        boxes.append((x1, y1, x2, y2))

    # Canonical <box> x1 y1 x2 y2 </box> format.
    for m in BOX_XYXY_CANONICAL_PATTERN.finditer(text):
        try:
            x1 = float(m.group(1))
            y1 = float(m.group(2))
            x2 = float(m.group(3))
            y2 = float(m.group(4))
        except ValueError:
            continue
        boxes.append((x1, y1, x2, y2))

    # <rect> x1 y1 x2 y2 </rect> fallback.
    for m in RECT_XYXY_PATTERN.finditer(text):
        try:
            x1 = float(m.group(1))
            y1 = float(m.group(2))
            x2 = float(m.group(3))
            y2 = float(m.group(4))
        except ValueError:
            continue
        boxes.append((x1, y1, x2, y2))

    if boxes:
        return boxes

    # Fallback: parse legacy ID/x,y,w,h style boxes (convert to xyxy).
    for m in BOX_PATTERN.finditer(text):
        try:
            x = float(m.group("x"))
            y = float(m.group("y"))
            w = float(m.group("w"))
            h = float(m.group("h"))
        except ValueError:
            continue
        x1, y1, x2, y2 = x, y, x + w, y + h
        boxes.append((x1, y1, x2, y2))
    return boxes


def iou(box1: Box, box2: Box) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    denom = area1 + area2 - inter
    if denom <= 0:
        return 0.0
    return inter / denom


def match_boxes(pred_boxes: Sequence[Box], gt_boxes: Sequence[Box]) -> List[Tuple[int, int, float]]:
    """Greedy one-to-one matching by IoU.

    Returns list of (pred_idx, gt_idx, iou).
    """
    matches: List[Tuple[int, int, float]] = []
    if not pred_boxes or not gt_boxes:
        return matches
    used_pred = set()
    used_gt = set()
    all_pairs: List[Tuple[float, int, int]] = []
    for pi, pb in enumerate(pred_boxes):
        for gi, gb in enumerate(gt_boxes):
            all_pairs.append((iou(pb, gb), pi, gi))
    all_pairs.sort(reverse=True, key=lambda t: t[0])
    for score, pi, gi in all_pairs:
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(pi)
        used_gt.add(gi)
        matches.append((pi, gi, score))
    return matches


@dataclass
class MetricAccumulator:
    total_samples: int = 0
    total_pred_boxes: int = 0
    total_gt_boxes: int = 0
    tp_at_thresh: int = 0
    total_pred_with_boxes: int = 0
    total_gt_with_boxes: int = 0
    sum_iou_over_matches: float = 0.0
    num_matches: int = 0

    def update(self, pred_boxes: Sequence[Box], gt_boxes: Sequence[Box], thresh: float) -> None:
        self.total_samples += 1
        self.total_pred_boxes += len(pred_boxes)
        self.total_gt_boxes += len(gt_boxes)
        if pred_boxes:
            self.total_pred_with_boxes += 1
        if gt_boxes:
            self.total_gt_with_boxes += 1
        matches = match_boxes(pred_boxes, gt_boxes)
        for _, _, score in matches:
            self.sum_iou_over_matches += score
            self.num_matches += 1
        for _, _, score in matches:
            if score >= thresh:
                self.tp_at_thresh += 1

    def finalize(self, thresh: float) -> Dict[str, Any]:
        mean_iou = self.sum_iou_over_matches / self.num_matches if self.num_matches > 0 else 0.0
        recall = self.tp_at_thresh / self.total_gt_boxes if self.total_gt_boxes > 0 else 0.0
        precision = self.tp_at_thresh / self.total_pred_boxes if self.total_pred_boxes > 0 else 0.0
        avg_pred_boxes = self.total_pred_boxes / self.total_samples if self.total_samples > 0 else 0.0
        return {
            "iou_threshold": thresh,
            "num_samples": self.total_samples,
            "num_matches": self.num_matches,
            "mean_iou_over_matches": mean_iou,
            "recall_at_thresh": recall,
            "precision_at_thresh": precision,
            "avg_predicted_boxes_per_sample": avg_pred_boxes,
            "total_pred_boxes": self.total_pred_boxes,
            "total_gt_boxes": self.total_gt_boxes,
        }


def load_predictions(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    args = parse_args()
    if not args.predictions.exists():
        raise FileNotFoundError(args.predictions)

    acc = MetricAccumulator()
    for rec in load_predictions(args.predictions):
        gt_raw = rec.get("gt_boxes") or []
        gt_boxes: List[Box] = []
        for b in gt_raw:
            if not isinstance(b, (list, tuple)) or len(b) != 4:
                continue
            x1, y1, x2, y2 = map(float, b)
            gt_boxes.append((x1, y1, x2, y2))
        response = rec.get("response", "")
        pred_boxes = parse_pred_boxes_from_response(response)
        acc.update(pred_boxes, gt_boxes, args.bbox_threshold)

    metrics = acc.finalize(args.bbox_threshold)
    print(json.dumps(metrics, indent=2))
    if args.output is not None:
        args.output.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
