#!/usr/bin/env python3
import re
from typing import List, Tuple

import torch

Box = Tuple[float, float, float, float]

BOX_PATTERN = re.compile(
    r"<box>\s*([+-]?\d*\.?\d+)\s+([+-]?\d*\.?\d+)\s+([+-]?\d*\.?\d+)\s+([+-]?\d*\.?\d+)\s*</box>",
    re.IGNORECASE,
)


def parse_boxes(text: str) -> List[Box]:
    boxes = []
    for m in BOX_PATTERN.findall(text or ""):
        x1, y1, x2, y2 = map(float, m)
        boxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
    return boxes


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def greedy_match(pred: List[Box], gt: List[Box]):
    pairs = []
    for pi, pb in enumerate(pred):
        for gi, gb in enumerate(gt):
            s = iou(pb, gb)
            if s > 0:
                pairs.append((s, pi, gi))
    pairs.sort(reverse=True)
    used_p, used_g = set(), set()
    matches = []
    for s, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        matches.append((pi, gi, s))
    return matches


def run_validation(trainer):
    if trainer.eval_dataset is None:
        return {}

    model = trainer.model
    tokenizer = trainer.processing_class
    model.eval()

    sum_iou = 0.0
    sum_recall = 0.0
    sum_precision = 0.0
    n = 0

    loader = trainer.get_eval_dataloader(trainer.eval_dataset)

    with torch.no_grad():
        for batch in loader:
            prompt_inputs = trainer._build_internvl_inputs(batch)
            prompt_inputs = trainer._prepare_input(prompt_inputs)

            gen_cfg = trainer.generation_config
            gen_cfg = gen_cfg.clone() if hasattr(gen_cfg, "clone") else gen_cfg
            gen_cfg.do_sample = False
            gen_cfg.temperature = 0.0
            gen_cfg.top_p = 1.0

            outputs = model.generate(
                input_ids=prompt_inputs["input_ids"],
                attention_mask=prompt_inputs["attention_mask"],
                pixel_values=prompt_inputs["pixel_values"],
                generation_config=gen_cfg,
            )

            prompt_len = prompt_inputs["input_ids"].size(1)
            if outputs.size(1) > prompt_len:
                completion_ids = outputs[:, prompt_len:]
            else:
                completion_ids = outputs

            texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

            for text, item in zip(texts, batch):
                pred = parse_boxes(text)
                gt = [tuple(map(float, x)) for x in item.get("gt_boxes", [])]

                matches = greedy_match(pred, gt)
                ious = [m[2] for m in matches]
                mean_iou = sum(ious) / len(ious) if ious else 0.0
                tp = sum(1 for _, _, s in matches if s >= 0.5)
                recall = tp / len(gt) if len(gt) > 0 else 0.0
                precision = tp / len(pred) if len(pred) > 0 else 0.0

                sum_iou += mean_iou
                sum_recall += recall
                sum_precision += precision
                n += 1

    model.train()

    if n == 0:
        return {}

    score = 0.45 * (sum_recall / n) + 0.35 * (sum_iou / n) + 0.20 * (sum_precision / n)

    return {
        "eval_mean_iou": sum_iou / n,
        "eval_recall_at_0.5": sum_recall / n,
        "eval_precision_at_0.5": sum_precision / n,
        "eval_score": score,
    }
