# GRPO After SFT (AI2D / InternVL3-8B)

## 1) Required Scripts

- `/prepare_ai2d_dragon_grpo_dataset.py`
- `/dragon_grpo_rewards.py`
- `/dragon_grpo_trainer.py`
- `/train_internvl3_dragon_grpo.py`
- `/eval_ai2d_metrics_from_predictions.py`
- `/predict_ai2d_holdout_sft_v2b.py`
- `/run_grpo_holdout_eval_v2b.py`


## 2) Build GRPO Training JSON

Input is your AI2D sample JSONL (question + answer + image refs + annotations).

```bash
python scripts/prepare_ai2d_dragon_grpo_dataset.py \
  --input ai2d_samples/samples_train_80.jsonl \
  --output ai2d_samples/ai2d_dragon_grpo_train.json \
  --questions-root ai2d_samples/questions \
  --annotations-root ai2d_samples \
  --image-root ai2d_samples
```

## 3) Run GRPO Training

Example from your setup:

```bash
python scripts/train_internvl3_dragon_grpo.py \
  --model-path outputs/ai2d_2k_fast_sft/ckpts_lr3e-5_ep5_20260312_021631 \
  --train-json ai2d_samples/ai2d_dragon_grpo_train.json \
  --output-dir outputs/dragon_grpo_ai2d \
  --epochs 1 \
  --lr 3e-6 \
  --beta 0.0 \
  --num-generations 4 \
  --per-device-batch-size 1 \
  --grad-accum 4 \
  --max-tiles 1
```

Recommended for stronger signal:

```bash
--num-generations 8 --temperature 1.0 --top-p 1.0 --save-steps 50
```

## 4) Run Heldout Inference + Eval (Same Style as SFT)

One command:

```bash
python scripts/run_grpo_holdout_eval_v2b.py \
  --checkpoint-dir outputs/dragon_grpo_ai2d \
  --samples ai2d_samples/samples_infer_20.jsonl \
  --device cuda:0 \
  --dtype bf16 \
  --max-tiles 1
```

Outputs:

- `outputs/dragon_grpo_ai2d_heldout_predictions_grpo.jsonl`
- `outputs/dragon_grpo_ai2d_heldout_metrics_grpo.json`

## 5) What "Good" Looks Like During Training

Watch these logs:

- `ai2d/wrapper_rate`: should trend to `1.0` early.
- `ai2d/parsed_boxes_mean`: should rise over time (not stay near 0).
- `ai2d/reward`: should improve (less negative / more positive).
- `loss`: should not stay exactly `0.0` for the whole run.

## 6) Why It Was Not Working Well in Your Previous Run

Observed issues:

- Model learned wrapper format (`<boxes>...</boxes>`) but produced malformed tags (`<ref>`, `<rect>`, placeholders like `x1 y1 x2 y2`).
- Reward stayed mostly negative and unstable.
- `loss` stayed near `0.0`, indicating weak per-group advantage signal.
- Heldout eval reported zero parseable predicted boxes for canonical metric extraction.

Root causes:

- Output-format drift from pretrained prior.
- Weak discriminative reward between malformed vs canonical outputs.
- Low effective reward variance in generation groups.

## 7) Practical Improvement Suggestions

1. Keep strict output shaping:
- force assistant prefix with `<boxes>\n`
- keep "no explanation" instruction

2. Canonicalize before scoring:
- map `<rect>` and `[[x,y,x2,y2]]` variants to canonical `<box> x y x2 y2 </box>`
- strip `<ref>` content

3. Reward tightening:
- hard-penalize placeholders (`x1 y1 x2 y2`)
- heavily reward canonical numeric `<box>` tags

4. Increase exploration for GRPO:
- `num_generations=8`
- `temperature=1.0`
- `top_p=1.0`

5. Save often for resume:
- `--save-steps 50` (or 25)



