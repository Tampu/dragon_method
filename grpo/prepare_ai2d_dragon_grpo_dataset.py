#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def xywh_to_xyxy(item: Dict[str, Any]) -> Optional[List[float]]:
    if not all(k in item for k in ("x", "y", "w", "h")):
        return None
    x1 = float(item["x"])
    y1 = float(item["y"])
    x2 = x1 + float(item["w"])
    y2 = y1 + float(item["h"])
    return [x1, y1, x2, y2]


def dedup_boxes(boxes: List[List[float]], tol: float = 1.0) -> List[List[float]]:
    out: List[List[float]] = []
    for b in boxes:
        keep = True
        for e in out:
            if all(abs(b[i] - e[i]) <= tol for i in range(4)):
                keep = False
                break
        if keep:
            out.append(b)
    return out


def build_prompt(question_text: str, choices: List[str], answer: str) -> str:
    lines = [(question_text or "").strip()]
    if choices:
        lines.append("Options:")
        for i, choice in enumerate(choices):
            lines.append(f"({chr(65+i)}) {choice}")
    lines.append(f"Answer: {(answer or '').strip()}")
    return "\n".join([x for x in lines if x])


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_question_entry(questions_root: Path, question_path: str, question_id: str) -> Tuple[str, List[str]]:
    if not question_path or not question_id:
        return "", []
    qfile = questions_root / Path(question_path).name
    if not qfile.exists():
        return "", []
    payload = json.loads(qfile.read_text(encoding="utf-8"))
    for q_text, meta in payload.get("questions", {}).items():
        if meta.get("questionId") == question_id:
            choices = meta.get("answerTexts") or []
            return q_text, choices
    return "", []


def load_annotation(annotations_root: Path, ann_path: str) -> Dict[str, Any]:
    p = annotations_root / ann_path
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def convert_record(
    obj: Dict[str, Any],
    questions_root: Path,
    annotations_root: Path,
    image_root: Path,
    include_answer: bool,
) -> Optional[Dict[str, Any]]:
    image_path = obj.get("image_path") or obj.get("image")
    if image_path is None:
        return None

    ann = load_annotation(annotations_root, obj.get("annotation", "")) if obj.get("annotation") else {}

    boxes_raw = obj.get("bbox")
    if not isinstance(boxes_raw, list):
        boxes_raw = ann.get("bbox", []) if isinstance(ann, dict) else []

    boxes: List[List[float]] = []
    for b in boxes_raw:
        xyxy = xywh_to_xyxy(b) if isinstance(b, dict) else None
        if xyxy is not None:
            boxes.append(xyxy)
    boxes = dedup_boxes(boxes)
    if not boxes:
        return None

    question_text, choices = load_question_entry(
        questions_root,
        obj.get("question_path", ""),
        obj.get("question_id", ""),
    )
    if not question_text:
        question_text = obj.get("question") or ann.get("question_text", "")

    if not choices:
        choices = obj.get("choices") or ann.get("choices") or (ann.get("answers") or {}).get("choices") or []

    answer = (obj.get("answer") or (ann.get("answers") or {}).get("correct") or "").strip()
    prompt = build_prompt(question_text, choices, answer if include_answer else "")

    image_abs = (image_root / image_path).resolve()
    if not image_abs.exists():
        return None

    return {
        "message": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_abs)},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "gt_boxes": boxes,
        "gt_answer": answer,
        "image": str(image_abs),
        "dataset": "ai2d",
        "sample_id": obj.get("id", ""),
        "question_id": obj.get("question_id", ""),
        "question_text": question_text,
        "choices": choices,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare AI2D GRPO dataset JSON from sample JSONL/JSON.")
    parser.add_argument("--input", type=Path, required=True, help="Input AI2D samples file (json or jsonl).")
    parser.add_argument("--output", type=Path, required=True, help="Output GRPO JSON file.")
    parser.add_argument("--questions-root", type=Path, default=Path("ai2d_samples/questions"))
    parser.add_argument("--annotations-root", type=Path, default=Path("ai2d_samples"))
    parser.add_argument("--image-root", type=Path, default=Path("ai2d_samples"))
    parser.add_argument("--include-answer", action="store_true", default=True)
    parser.add_argument("--no-include-answer", action="store_false", dest="include_answer")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    questions_root = args.questions_root.resolve()
    annotations_root = args.annotations_root.resolve()
    image_root = args.image_root.resolve()

    raw = load_json_or_jsonl(args.input.resolve())
    if args.limit is not None:
        raw = raw[: args.limit]

    out: List[Dict[str, Any]] = []
    skipped = 0
    for x in raw:
        rec = convert_record(
            x,
            questions_root=questions_root,
            annotations_root=annotations_root,
            image_root=image_root,
            include_answer=args.include_answer,
        )
        if rec is None:
            skipped += 1
            continue
        out.append(rec)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {len(out)} records to {args.output}")
    if skipped:
        print(f"Skipped {skipped} records (missing boxes/image/question data).")


if __name__ == "__main__":
    main()
