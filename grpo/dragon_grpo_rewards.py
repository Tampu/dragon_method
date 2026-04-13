import re
from typing import List, Sequence, Tuple

Box = Tuple[float, float, float, float]

BOX_PATTERN = re.compile(
    r"<box>\s*([+-]?\d*\.?\d+)\s+([+-]?\d*\.?\d+)\s+([+-]?\d*\.?\d+)\s+([+-]?\d*\.?\d+)\s*</box>",
    re.IGNORECASE,
)
RECT_PATTERN = re.compile(
    r"<rect>\s*([+-]?\d*\.?\d+)[,\s]+([+-]?\d*\.?\d+)[,\s]+([+-]?\d*\.?\d+)[,\s]+([+-]?\d*\.?\d+)\s*</rect>",
    re.IGNORECASE,
)
BOX_OPEN_PATTERN = re.compile(r"<box>", re.IGNORECASE)
RECT_OR_REF_PATTERN = re.compile(r"</?(rect|ref)>", re.IGNORECASE)
PLACEHOLDER_COORD_PATTERN = re.compile(r"\bx1\b|\by1\b|\bx2\b|\by2\b", re.IGNORECASE)
TAG_PATTERN = re.compile(r"</?boxes>|</?box>", re.IGNORECASE)


def parse_boxes_from_text(text: str) -> List[Box]:
    boxes: List[Box] = []
    t = text or ""
    # Normalize common malformed variants before parsing.
    t = re.sub(r"</?ref>", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t)

    for m in BOX_PATTERN.findall(t):
        x1, y1, x2, y2 = map(float, m)
        boxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
    for m in RECT_PATTERN.findall(t):
        x1, y1, x2, y2 = map(float, m)
        boxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
    return boxes


def has_valid_boxes_wrapper(text: str) -> bool:
    t = (text or "").lower()
    return "<boxes>" in t and "</boxes>" in t


def iou(box_a: Box, box_b: Box) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

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


def greedy_match(pred: Sequence[Box], gt: Sequence[Box]):
    pairs = []
    for pi, pb in enumerate(pred):
        for gi, gb in enumerate(gt):
            s = iou(pb, gb)
            if s > 0:
                pairs.append((s, pi, gi))
    pairs.sort(reverse=True)

    used_p = set()
    used_g = set()
    matches = []
    for s, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        matches.append((pi, gi, s))
    return matches


def _coerce_gold_boxes(gold):
    out: List[Box] = []
    for b in gold or []:
        if isinstance(b, (list, tuple)) and len(b) == 4:
            x1, y1, x2, y2 = map(float, b)
            out.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
    return out


def _non_box_text_penalty(text: str) -> float:
    t = text or ""
    stripped = TAG_PATTERN.sub(" ", t)
    # Remove numbers and punctuation; remaining alphabetic prose is discouraged.
    stripped = re.sub(r"[0-9\.\,\-\+\s]", " ", stripped)
    has_letters = bool(re.search(r"[A-Za-z]", stripped))
    # Extra hard penalty for obvious explanatory language.
    low = t.lower()
    explanation_markers = any(k in low for k in ["explanation", "because", "therefore", "answer is"])
    penalty = 0.0
    if has_letters:
        penalty += 0.20
    if explanation_markers:
        penalty += 0.20
    return penalty


def _format_target_penalty(text: str) -> float:
    t = text or ""
    penalty = 0.0
    if RECT_OR_REF_PATTERN.search(t):
        penalty += 0.25
    if PLACEHOLDER_COORD_PATTERN.search(t):
        penalty += 0.35
    return penalty


def dragon_box_reward(prompts, completions, gt_boxes, **kwargs):
    rewards = []
    iou_threshold = kwargs.get("iou_threshold", 0.5)

    for completion, gold in zip(completions, gt_boxes):
        pred = parse_boxes_from_text(completion)
        gold_boxes = _coerce_gold_boxes(gold)
        format_reward = 1.0 if has_valid_boxes_wrapper(completion) else 0.0
        parse_reward = 1.0 if len(pred) > 0 else 0.0
        text_penalty = _non_box_text_penalty(completion)
        target_penalty = _format_target_penalty(completion)
        box_tag_count = len(BOX_OPEN_PATTERN.findall(completion or ""))
        box_tag_reward = 1.0 if box_tag_count > 0 else 0.0

        # Dense early-phase shaping:
        # - keep a penalty for no boxes, but make it softer than a hard constant collapse
        # - still reward format compliance so the policy can bootstrap toward parseable outputs
        if len(pred) == 0:
            reward = -0.15 + 0.10 * format_reward + 0.10 * box_tag_reward - text_penalty - target_penalty
            rewards.append(float(reward))
            continue

        matches = greedy_match(pred, gold_boxes)
        matched_ious = [m[2] for m in matches]

        mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0
        tp = sum(1 for _, _, s in matches if s >= iou_threshold)

        recall = tp / len(gold_boxes) if len(gold_boxes) > 0 else 0.0
        precision = tp / len(pred) if len(pred) > 0 else 0.0
        extra_penalty = max(0, len(pred) - len(gold_boxes)) / max(1, len(gold_boxes))

        reward = (
            0.08 * format_reward
            + 0.08 * parse_reward
            + 0.14 * box_tag_reward
            + 0.35 * mean_iou
            + 0.30 * recall
            + 0.15 * precision
            - 0.10 * extra_penalty
            - text_penalty
            - target_penalty
        )
        rewards.append(float(reward))

    return rewards
