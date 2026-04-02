#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast SFT launcher for InternVL3-8B on AI2D 2k.")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs for torchrun.")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/internvl3_8b_ai2d_sft.json"),
        help="Base config JSON to merge overrides into.",
    )
    parser.add_argument(
        "--meta-path",
        type=Path,
        default=Path("configs/ai2d_2k_meta.json"),
        help="Meta JSON used for 2k training samples.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("outputs/ai2d_2k_fast_sft"),
        help="Directory for generated config + checkpoints.",
    )
    parser.add_argument("--master-port", type=str, default=os.environ.get("MASTER_PORT", "29600"))
    parser.add_argument("--epochs", type=float, default=3.0, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate.")
    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default=os.environ.get("CUDA_VISIBLE_DEVICES"),
        help="Comma-separated GPU ids to expose, e.g. '7' or '4,6'.",
    )
    parser.add_argument(
        "--check-choices-lines",
        type=int,
        default=200,
        help="How many lines to sample from each annotation for verifying 'Options:' in user prompt.",
    )
    parser.add_argument(
        "--require-choices",
        action="store_true",
        default=True,
        help="Fail fast if sampled prompts do not contain options/choices.",
    )
    parser.add_argument(
        "--no-require-choices",
        action="store_false",
        dest="require_choices",
        help="Disable the choices presence check.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    base_config = (root / args.base_config).resolve() if not args.base_config.is_absolute() else args.base_config
    meta_path = (root / args.meta_path).resolve() if not args.meta_path.is_absolute() else args.meta_path
    work_dir = (root / args.work_dir).resolve() if not args.work_dir.is_absolute() else args.work_dir
    train_script = root / "scripts" / "internvl_chat_finetune.py"
    torchrun = Path(sys.executable).with_name("torchrun")

    if not base_config.exists():
        raise FileNotFoundError(f"Base config not found: {base_config}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    if not train_script.exists():
        raise FileNotFoundError(f"Training script not found: {train_script}")

    meta = json.loads(meta_path.read_text())
    total_samples = 0
    for ds_name, ds_meta in meta.items():
        ann = Path(ds_meta["annotation"])
        if not ann.is_absolute():
            ann = root / ann
        if not ann.exists():
            raise FileNotFoundError(f"Annotation file not found for dataset '{ds_name}': {ann}")
        with ann.open("r", encoding="utf-8") as f:
            count = sum(1 for _ in f)
        total_samples += count
        print(f"Dataset '{ds_name}': {count} samples from {ann}")
        if args.require_choices:
            checked = 0
            bad = 0
            with ann.open("r", encoding="utf-8") as f:
                for line in f:
                    if checked >= args.check_choices_lines:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    checked += 1
                    try:
                        obj = json.loads(line)
                        convs = obj.get("conversations", [])
                        user_text = convs[0].get("value", "") if convs else ""
                    except Exception:
                        bad += 1
                        continue
                    if "Options:" not in user_text:
                        bad += 1
            print(f"Choices check '{ds_name}': {checked - bad}/{checked} lines include 'Options:'")
            if checked > 0 and bad > 0:
                raise RuntimeError(
                    f"Choices check failed for dataset '{ds_name}' ({bad}/{checked} sampled lines missing 'Options:'). "
                    "Regenerate annotation with prepare_ai2d_internvl_sft_v2b.py."
                )
    print(f"Total samples from meta: {total_samples}")
    if total_samples < 2000:
        print("WARNING: Fewer than 2000 samples detected. This run is not true 2k SFT.")

    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir = work_dir / "ckpts"
    config_out = work_dir / "internvl3_8b_ai2d_2k_fast.json"

    cfg = json.loads(base_config.read_text())
    cfg.update(
        {
            "model_name_or_path": "OpenGVLab/InternVL3-8B",
            "freeze_llm": False,
            "freeze_backbone": False,
            "use_llm_lora": 16,
            "use_backbone_lora": 16,
            "max_seq_length": 1024,
            "force_image_size": 448,
            "conv_style": "internlm2-chat",
            "meta_path": str(meta_path),
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 1,
            "learning_rate": args.lr,
            "num_train_epochs": args.epochs,
            "gradient_checkpointing": False,
            "grad_checkpoint": False,
            "bf16": True,
            "tf32": True,
            "output_dir": str(output_dir),
            "overwrite_output_dir": True,
            "report_to": "none",
            "logging_steps": 20,
            "dataloader_num_workers": 8,
            "dataloader_persistent_workers": True,
            "save_strategy": "no",
        }
    )
    config_out.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    print(f"Saved config: {config_out}")

    env = dict(
        os.environ,
        LAUNCHER="pytorch",
        MASTER_ADDR=os.environ.get("MASTER_ADDR", "127.0.0.1"),
        MASTER_PORT=args.master_port,
    )
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        print(f"Using CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}")

    cmd = [
        str(torchrun),
        f"--nproc_per_node={args.gpus}",
        f"--master_port={args.master_port}",
        str(train_script),
        str(config_out),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env, cwd=str(root))


if __name__ == "__main__":
    main()
