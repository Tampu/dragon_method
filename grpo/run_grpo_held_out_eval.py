#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GRPO checkpoint inference on AI2D heldout split, then compute same metrics as SFT eval."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="GRPO (or SFT) checkpoint directory with model shards + tokenizer.",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=Path("ai2d_samples/samples_infer_20.jsonl"),
        help="Heldout samples JSONL.",
    )
    parser.add_argument("--image-root", type=Path, default=Path("ai2d_samples"))
    parser.add_argument("--annotations-root", type=Path, default=Path("ai2d_samples"))
    parser.add_argument("--questions-root", type=Path, default=Path("ai2d_samples"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--max-tiles", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=0, help="0 => all heldout samples.")
    parser.add_argument("--bbox-threshold", type=float, default=0.5)
    parser.add_argument(
        "--predictions-out",
        type=Path,
        default=None,
        help="Predictions JSONL output path. Default: outputs/<checkpoint_name>_heldout_predictions_grpo.jsonl",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="Metrics JSON output path. Default: outputs/<checkpoint_name>_heldout_metrics_grpo.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    ckpt_name = args.checkpoint_dir.name
    predictions_out = args.predictions_out or (root / "outputs" / f"{ckpt_name}_heldout_predictions_grpo.jsonl")
    metrics_out = args.metrics_out or (root / "outputs" / f"{ckpt_name}_heldout_metrics_grpo.json")

    predict_script = root / "scripts" / "predict_ai2d_holdout_sft_v2b.py"
    eval_script = root / "scripts" / "eval_ai2d_metrics_from_predictions.py"

    predict_cmd = [
        sys.executable,
        str(predict_script),
        "--checkpoint-dir",
        str(args.checkpoint_dir),
        "--samples",
        str(args.samples),
        "--image-root",
        str(args.image_root),
        "--annotations-root",
        str(args.annotations_root),
        "--questions-root",
        str(args.questions_root),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--input-size",
        str(args.input_size),
        "--max-tiles",
        str(args.max_tiles),
        "--num-samples",
        str(args.num_samples),
        "--output",
        str(predictions_out),
    ]

    eval_cmd = [
        sys.executable,
        str(eval_script),
        "--predictions",
        str(predictions_out),
        "--bbox-threshold",
        str(args.bbox_threshold),
        "--output",
        str(metrics_out),
    ]

    print("[run] Inference command:")
    print(" ".join(predict_cmd))
    subprocess.run(predict_cmd, check=True, cwd=str(root))

    print("[run] Eval command:")
    print(" ".join(eval_cmd))
    subprocess.run(eval_cmd, check=True, cwd=str(root))

    print(f"[done] Predictions: {predictions_out}")
    print(f"[done] Metrics: {metrics_out}")


if __name__ == "__main__":
    main()

