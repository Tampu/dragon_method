#!/usr/bin/env python3
"""
Convert AI2D grounding samples into InternVL conversation JSONL format.

Goal (QA-conditioned grounding):
Input  (human):   <image>\nQuestion ...\n[Options...]\nAnswer: <answer>
Target (gpt):     bounding boxes needed to answer, in a stable, label-agnostic format.

This script assumes each sample JSONL line contains (at minimum):
  - id
  - image_path  (relative to some root you will provide later in meta.json)
  - question / question_id / question_path
  - answer
  - annotation (path to annotation json)  [kept as metadata]
  - bbox (list of boxes)  <-- this must be present for grounding supervision
      each bbox item: { "x":..., "y":..., "w":..., "h":..., ... }
  - optionally bbox items have "source" (gt/pred/etc). We keep all by default,
    or you can filter via --bbox-source.

Output format per line:
{
  "id": "...",
  "image": "ai2d_samples/images/62.png",
  "conversations": [
    {"from":"human","value":"<image>\\n...\\nAnswer: ..."},
    {"from":"gpt","value":"<boxes>\\n<box> x1 y1 x2 y2 </box>\\n...</boxes>"}
  ],
  "metadata": {...}
}

Notes:
- Label-agnostic: we do NOT include object labels/ids in the target.
- Determinism: boxes are sorted (top-to-bottom, left-to-right) for stable targets.
- Optional: normalize coords to [0,1] with --normalize (requires reading image size).
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image


def load_question_entry(question_file: Path, question_id: str) -> Tuple[str, Dict[str, Any]]:
    """Return (question_text, meta) for the matching question id."""
    if not question_file.exists():
        return "", {}
    payload = json.loads(question_file.read_text())
    questions = payload.get("questions", {})
    for q_text, meta in questions.items():
        if meta.get("questionId") == question_id:
            return q_text, meta
    return "", {}


def build_prompt(question_text: str, choices: List[str]) -> str:
    prompt = (question_text or "").strip()
    if choices:
        lines = [f"({chr(65 + idx)}) {choice}" for idx, choice in enumerate(choices)]
        prompt = f"{prompt}\nOptions:\n" + "\n".join(lines)
    return prompt


def extract_choices(
    sample: Dict[str, Any],
    question_meta: Dict[str, Any],
    annotation_payload: Optional[Dict[str, Any]],
) -> List[str]:
    """Best-effort choice extraction from all known sample/annotation schemas."""
    candidates: List[Any] = []
    if question_meta:
        candidates.append(question_meta.get("answerTexts"))
    candidates.append(sample.get("choices"))
    candidates.append(sample.get("options"))
    candidates.append((sample.get("meta") or {}).get("choices"))
    if annotation_payload:
        candidates.append(annotation_payload.get("choices"))
        candidates.append((annotation_payload.get("answers") or {}).get("choices"))

    for item in candidates:
        if isinstance(item, list) and item:
            cleaned = [str(x).strip() for x in item if str(x).strip()]
            if cleaned:
                return cleaned
    return []


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def sort_boxes_xyxy(boxes: List[Tuple[float, float, float, float]]) -> List[Tuple[float, float, float, float]]:
    """Sort boxes top-to-bottom, then left-to-right, based on centers."""

    def key(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return (cy, cx)

    return sorted(boxes, key=key)


def dedup_boxes(boxes: List[Tuple[float, float, float, float]], tol: float = 1.0) -> List[Tuple[float, float, float, float]]:
    """Deduplicate near-identical boxes using a simple tolerance check."""
    deduped: List[Tuple[float, float, float, float]] = []
    for candidate in boxes:
        keep = True
        for existing in deduped:
            if (
                abs(candidate[0] - existing[0]) <= tol and
                abs(candidate[1] - existing[1]) <= tol and
                abs(candidate[2] - existing[2]) <= tol and
                abs(candidate[3] - existing[3]) <= tol
            ):
                keep = False
                break
        if keep:
            deduped.append(candidate)
    return deduped


def extract_xyxy_from_bbox_item(item: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    """Support common bbox formats -> (x1, y1, x2, y2)."""
    if all(k in item for k in ("x", "y", "w", "h")):
        x1 = float(item["x"])
        y1 = float(item["y"])
        x2 = x1 + float(item["w"])
        y2 = y1 + float(item["h"])
        return (x1, y1, x2, y2)
    if isinstance(item.get("bbox"), dict):
        bbox = item["bbox"]
        if all(k in bbox for k in ("x", "y", "w", "h")):
            x1 = float(bbox["x"])
            y1 = float(bbox["y"])
            x2 = x1 + float(bbox["w"])
            y2 = y1 + float(bbox["h"])
            return (x1, y1, x2, y2)
    rect = item.get("rectangle")
    if isinstance(rect, list) and len(rect) == 2:
        (x1, y1), (x2, y2) = rect
        return (float(x1), float(y1), float(x2), float(y2))
    return None


def load_image_size(image_abs_path: Path) -> Tuple[int, int]:
    with Image.open(image_abs_path) as img:
        width, height = img.size
    return width, height


def load_annotation_payload(sample: Dict[str, Any], annotations_root: Path) -> Optional[Dict[str, Any]]:
    """Load the annotation JSON referenced by the sample, if available."""
    ann_rel = sample.get("annotation") or sample.get("meta", {}).get("annotation")
    if not ann_rel:
        return None
    ann_path = Path(ann_rel)
    if not ann_path.is_absolute():
        ann_path = annotations_root / ann_path
    ann_path = ann_path.resolve()
    if not ann_path.exists():
        return None
    try:
        return json.loads(ann_path.read_text())
    except json.JSONDecodeError:
        return None


def collect_bbox_items(
    sample: Dict[str, Any],
    annotations_root: Path,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Return bbox list for the sample, loading the annotation file if needed."""
    bbox_items = sample.get("bbox")
    annotation_payload: Optional[Dict[str, Any]] = None
    if not isinstance(bbox_items, list) or len(bbox_items) == 0:
        annotation_payload = load_annotation_payload(sample, annotations_root)
        if annotation_payload is not None:
            bbox_items = annotation_payload.get("bbox", [])
        else:
            bbox_items = []
    return bbox_items, annotation_payload


def build_boxes_string(
    bbox_items: List[Dict[str, Any]],
    normalize: bool,
    image_size: Optional[Tuple[int, int]],
    bbox_source: str,
    dedup_tol: float,
    round_ndigits: int = 4,
) -> str:
    """Convert bbox list to InternVL-friendly <boxes> serialization."""
    if bbox_source == "reviewed":
        bbox_source = "all"

    boxes_xyxy: List[Tuple[float, float, float, float]] = []
    for box in bbox_items or []:
        src = (box.get("source") or "").lower()
        if bbox_source != "all" and src != bbox_source:
            continue
        xyxy = extract_xyxy_from_bbox_item(box)
        if xyxy is None:
            continue
        boxes_xyxy.append(xyxy)

    if not boxes_xyxy:
        return "<boxes>\n</boxes>"

    if normalize:
        if image_size is None:
            raise ValueError("normalize=True requires image_size=(width, height)")
        width, height = image_size
        normed: List[Tuple[float, float, float, float]] = []
        for x1, y1, x2, y2 in boxes_xyxy:
            normed.append(
                (
                    clamp(x1 / width, 0.0, 1.0),
                    clamp(y1 / height, 0.0, 1.0),
                    clamp(x2 / width, 0.0, 1.0),
                    clamp(y2 / height, 0.0, 1.0),
                )
            )
        boxes_xyxy = normed

    boxes_xyxy = dedup_boxes(sort_boxes_xyxy(boxes_xyxy), tol=dedup_tol)

    lines = ["<boxes>"]
    for x1, y1, x2, y2 in boxes_xyxy:
        if normalize:
            coords = (
                round(x1, round_ndigits),
                round(y1, round_ndigits),
                round(x2, round_ndigits),
                round(y2, round_ndigits),
            )
        else:
            coords = (
                int(round(x1)),
                int(round(y1)),
                int(round(x2)),
                int(round(y2)),
            )
        lines.append(f"<box> {coords[0]} {coords[1]} {coords[2]} {coords[3]} </box>")
    lines.append("</boxes>")
    return "\n".join(lines)


def convert_sample(
    sample: Dict[str, Any],
    question_text: str,
    question_meta: Dict[str, Any],
    include_answer_in_prompt: bool,
    normalize_boxes: bool,
    image_root: Path,
    bbox_source: str,
    dedup_tol: float,
    bbox_items: List[Dict[str, Any]],
    annotation_payload: Optional[Dict[str, Any]],
    require_choices: bool,
) -> Dict[str, Any]:
    choices = extract_choices(sample, question_meta, annotation_payload)
    if require_choices and not choices:
        raise ValueError(f"missing choices for sample id={sample.get('id')}")
    prompt = build_prompt(question_text or sample.get("question", ""), choices)

    answer = (sample.get("answer") or "").strip()
    if include_answer_in_prompt:
        human_value = f"<image>\n{prompt}\nAnswer: {answer}"
    else:
        human_value = f"<image>\n{prompt}"
    human_turn = {"from": "user", "value": human_value}

    image_rel = sample.get("image_path") or sample.get("image")
    if image_rel is None:
        raise KeyError("Sample is missing image_path/image field")

    image_abs = (image_root / image_rel).resolve()
    image_size = load_image_size(image_abs) if normalize_boxes else None

    assistant_turn = {
        "from": "assistant",
        "value": build_boxes_string(
            bbox_items=bbox_items,
            normalize=normalize_boxes,
            image_size=image_size,
            bbox_source=bbox_source,
            dedup_tol=dedup_tol,
        ),
    }

    metadata = {
        "question_id": sample.get("question_id") or sample.get("meta", {}).get("question_id"),
        "question_path": sample.get("question_path") or sample.get("meta", {}).get("question_path"),
        "annotation_path": sample.get("annotation") or sample.get("meta", {}).get("annotation"),
        "answer": answer,
    }
    if choices:
        metadata["choices"] = choices
        metadata["correct_choice_index"] = question_meta.get("correctAnswer")

    return {
        "id": sample.get("id"),
        "image": image_rel,
        "conversations": [human_turn, assistant_turn],
        "metadata": metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare InternVL SFT data for QA-conditioned grounding from AI2D samples."
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=Path("ai2d_samples/samples.jsonl"),
        help="Path to the samples JSONL file.",
    )
    parser.add_argument(
        "--questions-root",
        type=Path,
        default=Path("ai2d_samples"),
        help="Directory that contains the questions/ folder.",
    )
    parser.add_argument(
        "--annotations-root",
        type=Path,
        default=Path("ai2d_samples"),
        help="Root directory used to resolve annotation JSON paths.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("ai2d_samples"),
        help="Root directory to resolve sample['image_path'].",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ai2d_samples/ai2d_internvl_grounding.jsonl"),
        help="Destination JSONL file for InternVL conversations.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of samples to convert.",
    )
    parser.add_argument(
        "--include-answer",
        action="store_true",
        default=True,
        help="Include the answer in the human prompt (QA-conditioned grounding).",
    )
    parser.add_argument(
        "--no-include-answer",
        action="store_false",
        dest="include_answer",
        help="Disable including answer in the prompt (Q-only grounding).",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize box coordinates to [0, 1] (requires reading image sizes).",
    )
    parser.add_argument(
        "--bbox-source",
        type=str,
        default="all",
        choices=["all", "gt", "pred", "reviewed"],
        help="Filter boxes by their 'source' tag.",
    )
    parser.add_argument(
        "--dedup-tol",
        type=float,
        default=1.0,
        help="Tolerance for box deduplication (pixels or normalized units).",
    )
    parser.add_argument(
        "--require-choices",
        action="store_true",
        help="Skip samples that do not have choices resolved from question/meta/annotation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples_path = args.samples.resolve()
    questions_root = args.questions_root.resolve()
    annotations_root = args.annotations_root.resolve()
    image_root = args.image_root.resolve()
    output_path = args.output.resolve()

    question_dir = questions_root / "questions"

    out_lines: List[str] = []
    skipped_missing_choices = 0
    with samples_path.open("r", encoding="utf-8") as reader:
        for idx, line in enumerate(reader):
            if args.limit is not None and idx >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)

            qpath = sample.get("question_path") or sample.get("meta", {}).get("question_path")
            qid = sample.get("question_id") or sample.get("meta", {}).get("question_id")
            if qpath and qid:
                question_file = question_dir / Path(qpath).name
                question_text, question_meta = load_question_entry(question_file, qid)
            else:
                question_text, question_meta = sample.get("question", ""), {}

            bbox_items, annotation_payload = collect_bbox_items(sample, annotations_root)
            if annotation_payload and not question_text:
                question_text = annotation_payload.get("question_text", question_text)

            if not isinstance(bbox_items, list) or len(bbox_items) == 0:
                continue

            try:
                record = convert_sample(
                    sample=sample,
                    question_text=question_text,
                    question_meta=question_meta,
                    include_answer_in_prompt=args.include_answer,
                    normalize_boxes=args.normalize,
                    image_root=image_root,
                    bbox_source=args.bbox_source,
                    dedup_tol=args.dedup_tol,
                    bbox_items=bbox_items,
                    annotation_payload=annotation_payload,
                    require_choices=args.require_choices,
                )
            except ValueError:
                skipped_missing_choices += 1
                continue
            out_lines.append(json.dumps(record, ensure_ascii=False))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(out_lines)} samples to {output_path}")
    if args.require_choices:
        print(f"Skipped {skipped_missing_choices} samples without choices")


if __name__ == "__main__":
    main()
