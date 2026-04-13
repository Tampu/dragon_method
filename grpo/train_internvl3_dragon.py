#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from datasets import Dataset

from dragon_grpo_rewards import dragon_box_reward
from dragon_grpo_trainer import DragonEvalCallback, DragonGRPOConfig, DragonGRPOTrainer
from eval_dragon_rl import run_validation


def load_dataset(path: Path) -> Dataset:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list at {path}, got {type(data)}")
    return Dataset.from_list(data)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to finetuned SFT checkpoint directory.")
    parser.add_argument("--train-json", type=Path, required=True, help="Prepared GRPO train JSON.")
    parser.add_argument("--eval-json", type=Path, default=None, help="Prepared GRPO eval JSON.")
    parser.add_argument("--image-root", type=str, default="", help="Optional root for relative image paths.")

    parser.add_argument("--output-dir", type=str, default="outputs/dragon_grpo")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    parser.add_argument("--beta", type=float, default=0.0, help="KL coefficient. Use 0 to disable ref-model KL.")
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)

    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-completion-length", type=int, default=128)

    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=3)

    parser.add_argument("--llm-lora-rank", type=int, default=16)
    parser.add_argument("--vision-lora-rank", type=int, default=16)

    parser.add_argument("--force-image-size", type=int, default=448)
    parser.add_argument("--down-sample-ratio", type=float, default=0.5)
    parser.add_argument("--conv-style", type=str, default="internlm2-chat")
    parser.add_argument("--max-tiles", type=int, default=12)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip-eps", type=float, default=0.28)
    parser.add_argument("--use-ref-model", action="store_true", help="Enable frozen reference model for KL term.")

    return parser.parse_args()


def main():
    args = parse_args()

    train_ds = load_dataset(args.train_json)
    eval_ds = load_dataset(args.eval_json) if args.eval_json is not None else None

    cfg = DragonGRPOConfig(
        output_dir=args.output_dir,
        remove_unused_columns=False,

        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,

        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,

        bf16=True,
        gradient_checkpointing=True,
        tf32=True,

        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        evaluation_strategy="no",
        report_to="none",

        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        beta=args.beta,

        conv_style=args.conv_style,
        force_image_size=args.force_image_size,
        down_sample_ratio=args.down_sample_ratio,
        max_tiles=args.max_tiles,
        use_llm_lora=args.llm_lora_rank,
        use_backbone_lora=args.vision_lora_rank,

        image_root=args.image_root,
        disable_ref_model=(not args.use_ref_model),
        clip_eps=args.clip_eps,
        seed=args.seed,
    )

    callbacks = []
    if eval_ds is not None:
        callbacks.append(DragonEvalCallback(run_validation, eval_every_steps=args.eval_steps))

    trainer = DragonGRPOTrainer(
        model=args.model_path,
        reward_funcs=[dragon_box_reward],
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_model()
    trainer.tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
