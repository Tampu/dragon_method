# diagram_attribution

# AI2D SFT and GRPO in This Repo

This note explains how the AI2D grounding pipeline works in this repo, using the actual scripts, configs, and saved outputs that were used.

## Goal

The task here is not answer prediction directly.

The task is:

- Input: image + question + options + correct answer
- Output: the bounding boxes needed to justify that answer

So both SFT and GRPO are training a QA-conditioned grounding model.

## Data Source

The core data lives under `ai2d_samples/`.

Important files:

- `ai2d_samples/samples*.jsonl`: raw/filtered AI2D samples
- `ai2d_samples/questions/`: original question JSON files
- `ai2d_samples/*.jsonl`: converted training files for InternVL
- `configs/ai2d_internvl_meta.json` and `configs/ai2d_2k_meta.json`: dataset wiring for training

Each AI2D sample contains at least:

- image path
- question id / question path
- answer
- one or more ground-truth boxes

Those boxes are the supervision target.

## SFT

### What SFT was trying to learn

SFT teaches InternVL to map:

- image
- question
- multiple-choice options
- correct answer

to:

- a structured `<boxes> ... </boxes>` output containing the relevant grounding boxes

This means the model is being taught where it should look in the diagram, conditioned on the answer being known.

### SFT data preparation

The SFT conversion happens in [prepare_ai2d_internvl_sft_v2b.py](/mnt/data2/traviku2/scripts/prepare_ai2d_internvl_sft_v2b.py).

That script builds InternVL chat-format records like:

```json
{
  "image": "ai2d_samples/images/62.png",
  "conversations": [
    {
      "from": "user",
      "value": "<image>\n<Question>\nOptions:\n(A) ...\n(B) ...\nAnswer: <correct answer>"
    },
    {
      "from": "assistant",
      "value": "<boxes>\n<box> x1 y1 x2 y2 </box>\n...</boxes>"
    }
  ]
}
```

So the prompt given to the model during SFT is:

```text
<image>
Question text
Options:
(A) choice A
(B) choice B
...
Answer: correct answer
```

The target is:

```text
<boxes>
<box> x1 y1 x2 y2 </box>
...
</boxes>
```

Boxes are:

- deduplicated
- sorted top-to-bottom / left-to-right
- stored as absolute xyxy coordinates by default

### SFT training script

The actual training loop is in [internvl_chat_finetune.py](/mnt/data2/traviku2/scripts/internvl_chat_finetune.py).

The fast launcher is [train_ai2d_2k_fast_v2b.py](/mnt/data2/traviku2/scripts/train_ai2d_2k_fast_v2b.py).

The fast launcher generated a run config at:

- [internvl3_8b_ai2d_2k_fast.json](/mnt/data2/traviku2/outputs/ai2d_2k_fast_sft/internvl3_8b_ai2d_2k_fast.json)

and the SFT checkpoint later used for GRPO was:

- [ckpts_lr3e-5_ep5_20260312_021631](/mnt/data2/traviku2/outputs/ai2d_2k_fast_sft/ckpts_lr3e-5_ep5_20260312_021631)

### What layers were finetuned in SFT

The SFT setup used InternVL3-8B with LoRA adapters.

From the actual fast SFT config / launcher:

- `freeze_llm = false`
- `freeze_backbone = false`
- `use_llm_lora = 16`
- `use_backbone_lora = 16`

InternVL applies LoRA to:

- Vision backbone attention / MLP projections 
- LLM attention / MLP projections

The local implementation is in [modeling_internvl_chat.py](/mnt/data2/traviku2/InternVL/internvl_chat/internvl/model/internvl_chat/modeling_internvl_chat.py).

In practice, this run should be understood as:

- LoRA finetuning on both the vision backbone and language model
- projector left trainable unless explicitly frozen

### SFT hyperparameters

For the main fast run used later for GRPO:

- base model: `OpenGVLab/InternVL3-8B`
- image size: `448`
- max sequence length: `1024`
- conversation style: `internlm2-chat`
- train batch size: `4`
- gradient accumulation: `1`
- learning rate: `3e-5` for the final chosen run
- epochs: `5`
- optimizer: `adamw_torch`
- scheduler: cosine
- warmup ratio: `0.03`
- weight decay: `0.01`
- bf16: `true`
- tf32: `true`
- logging steps: `20`
- save strategy during the fast run: `no`
- seed: `42`

The training set size recorded in the final SFT results was:

- `1872` samples

### SFT loss

Saved in:

- [train_results.json](/mnt/data2/traviku2/outputs/ai2d_2k_fast_sft/ckpts_lr3e-5_ep5_20260312_021631/train_results.json)
- [trainer_state.json](/mnt/data2/traviku2/outputs/ai2d_2k_fast_sft/ckpts_lr3e-5_ep5_20260312_021631/trainer_state.json)

Main numbers:

- final average training loss: `0.9316`
- early loss: about `2.04`
- late-step loss: mostly in the `0.80` to `0.95` range

So the model clearly learned the supervised target format better over time.

### How eval worked during SFT training

For the main fast SFT run, there was no real validation during training.

Why:

- `eval_dataset=None` in the SFT trainer
- the run logged training loss only

So "eval during SFT" was effectively:

- no heldout validation
- just train loss logging

### How eval worked after SFT

Post-training evaluation happened in two steps:

1. [predict_ai2d_holdout_sft_v2b.py](/mnt/data2/traviku2/scripts/predict_ai2d_holdout_sft_v2b.py)
2. [eval_ai2d_metrics_from_predictions.py](/mnt/data2/traviku2/scripts/eval_ai2d_metrics_from_predictions.py)

Inference used the same training-style prompt:

```text
<image>
Question ...
Options:
...
Answer: correct answer
```

Then evaluation parsed predicted boxes from the model response and matched them to GT boxes using greedy IoU matching.

Metrics computed:

- mean IoU over matched boxes
- Recall@0.5
- Precision@0.5
- average predicted boxes per sample

### SFT post-training performance

Saved in:

- [ai2d_eval_metrics_sft.json](/mnt/data2/traviku2/outputs/ai2d_eval_metrics_sft.json)

That eval gave:

- mean IoU: `0.0`
- recall@0.5: `0.0`
- precision@0.5: `0.0`
- avg predicted boxes: `0.0`

So the specific SFT checkpoint later used for GRPO was not producing useful heldout grounding outputs at that evaluation point.

There are also later exploratory SFT eval files on other checkpoints:

- [heldout_metrics_fast100.json](/mnt/data2/traviku2/outputs/ai2d_2k_fast_sft_ep100_lr3e-5_gpu7/heldout_metrics_fast100.json)
- [heldout_metrics_quality100.json](/mnt/data2/traviku2/outputs/ai2d_2k_fast_sft_ep10_lr3e-5_gpu0/heldout_metrics_quality100.json)

Those showed weak but non-zero grounding:

- mean IoU around `0.24` to `0.27`
- recall@0.5 around `3.5%` to `4.1%`
- precision@0.5 around `8.6%` to `11.7%`

So the best way to read SFT here is:

- training loss improved
- output formatting partially improved
- heldout grounding quality remained weak

## GRPO

### What GRPO was trying to do

GRPO was meant to refine the SFT model by rewarding better box outputs directly, instead of imitating gold outputs token-by-token.

The model starts from the SFT checkpoint, samples multiple candidate box outputs per prompt, scores them with a reward function, and then updates itself to prefer the better ones.

### Did GRPO start from the finetuned model?

Yes.

The launcher [train_internvl3_dragon_grpo.py](/mnt/data2/traviku2/scripts/train_internvl3_dragon_grpo.py) takes:

- `--model-path`

and in your run this pointed to the SFT checkpoint:

- [ckpts_lr3e-5_ep5_20260312_021631](/mnt/data2/traviku2/outputs/ai2d_2k_fast_sft/ckpts_lr3e-5_ep5_20260312_021631)

So GRPO was not starting from base InternVL. It started from the SFT-finetuned model.

### GRPO data preparation

The GRPO dataset is built by [prepare_ai2d_dragon_grpo_dataset.py](/mnt/data2/traviku2/scripts/prepare_ai2d_dragon_grpo_dataset.py).

Each record contains:

- user message with image + text
- `gt_boxes`
- `gt_answer`
- metadata

The prompt text is:

```text
Question text
Options:
(A) ...
(B) ...
...
Answer: correct answer
```

So again, this is QA-conditioned grounding, not answer prediction.

### GRPO prompt actually used during training

Inside [dragon_grpo_trainer.py](/mnt/data2/traviku2/scripts/dragon_grpo_trainer.py), the trainer appends an extra instruction:

```text
Return ONLY bounding boxes needed to justify the answer.
Format strictly as:
<boxes>
<box> x1 y1 x2 y2 </box>
...
</boxes>
No explanation.
```

It also seeds the assistant output with:

```text
<boxes>
```

So the effective GRPO prompt is:

- image
- question
- options
- answer
- strict box-only formatting instruction

### What GRPO output was supposed to be

Only:

```text
<boxes>
<box> x1 y1 x2 y2 </box>
...
</boxes>
```

No natural language explanation.

### What reward function was used

The reward is in [dragon_grpo_rewards.py](/mnt/data2/traviku2/scripts/dragon_grpo_rewards.py).

It rewards:

- valid outer `<boxes> ... </boxes>` wrapper
- parseable box tags
- presence of `<box>` tags
- higher mean IoU with GT boxes
- higher Recall@0.5
- higher Precision@0.5

It penalizes:

- extra predicted boxes
- non-box explanatory text
- wrong tags like `<rect>` or `<ref>`
- placeholder coordinates like `x1 y1 x2 y2`

So the reward mixes:

- formatting reward
- parsing reward
- localization reward
- precision/recall reward
- anti-cheating penalties

### Was GRPO offline or online?

It was online RL.

Why:

- the current model generates multiple fresh completions during training
- those fresh completions are rewarded immediately
- policy updates are computed from those sampled outputs

So it is not offline supervised replay. It is on-policy-style sampling with grouped reward normalization.

### How GRPO worked here

The trainer in [dragon_grpo_trainer.py](/mnt/data2/traviku2/scripts/dragon_grpo_trainer.py):

1. takes one prompt
2. samples `num_generations=4` box completions
3. scores each completion with the reward function
4. normalizes rewards within the group
5. computes advantages
6. updates the model to increase probability of the relatively better samples

This is the "group relative" part of GRPO: samples compete against each other within the same prompt.

### What layers were updated in GRPO

GRPO explicitly freezes the base model first, then applies LoRA.

So GRPO trains:

- LLM LoRA adapters
- vision LoRA adapters

and leaves the base weights frozen.

That is stricter than your SFT setup.

### GRPO policy / model setup

Main run settings from [train_internvl3_dragon_grpo.py](/mnt/data2/traviku2/scripts/train_internvl3_dragon_grpo.py):

- starting model: SFT checkpoint
- epochs: `1`
- lr: `3e-6`
- weight decay: `0.0`
- beta: `0.0`
- num generations: `4`
- temperature: `0.7`
- top-p: `0.9`
- batch size: `1`
- grad accumulation: `4`
- max prompt length: `1024`
- max completion length: `128`
- clip epsilon: `0.28`
- image size: `448`
- max tiles: `1` in the stabilized runs

`beta=0.0` means the KL/reference-model term was effectively disabled in your main run.

### How GRPO training behaved

The training logs are in:

- [checkpoint-1800/trainer_state.json](/mnt/data2/traviku2/outputs/dragon_grpo_ai2d/checkpoint-1800/trainer_state.json)

What happened:

- logged `loss` stayed at `0.0`
- reward stayed negative throughout training
- wrapper rate was usually `1.0`
- parsed box count improved somewhat over time

Examples from the log:

- early reward around `-0.50`
- later reward often around `-0.28` to `-0.32`

So GRPO mostly improved:

- format compliance
- ability to emit something that looked like box output

But it did not improve heldout localization quality.

### How eval worked during GRPO training

There is an optional validation callback in [eval_dragon_rl.py](/mnt/data2/traviku2/scripts/eval_dragon_rl.py).

If an eval dataset is provided, it computes:

- `eval_mean_iou`
- `eval_recall_at_0.5`
- `eval_precision_at_0.5`
- `eval_score`

But in your actual run, you did not pass an eval dataset.

So during GRPO training, you mainly watched:

- train-time reward
- wrapper rate
- parsed box count

not heldout validation.

### How eval worked after GRPO

Post-GRPO evaluation was done by:

- [run_grpo_holdout_eval_v2b.py](/mnt/data2/traviku2/scripts/run_grpo_holdout_eval_v2b.py)

That script:

1. runs inference on heldout AI2D
2. calls the same metric script used for SFT

So SFT and GRPO post-training evaluation used the same heldout metric family.

### GRPO post-training performance

Saved in:

- [dragon_grpo_ai2d_heldout_metrics_grpo.json](/mnt/data2/traviku2/outputs/dragon_grpo_ai2d_heldout_metrics_grpo.json)

Final heldout metrics were:

- num samples: `469`
- num matches: `0`
- mean IoU over matches: `0.0`
- recall@0.5: `0.0`
- precision@0.5: `0.0`
- avg predicted boxes per sample: `0.0`

So the final GRPO checkpoint did not improve heldout grounding in the saved evaluation.

## Why GRPO did not help yet

From the saved artifacts, the main problem is that the starting SFT model was already weak on heldout grounding.

That means GRPO started from a policy that:

- often failed to emit useful boxes
- often collapsed into empty or malformed outputs
- improved formatting more than localization

What the logs show:

- SFT train loss improved, but heldout grounding remained poor
- GRPO reward became less negative, but that mostly reflected format shaping
- heldout box quality stayed at zero in the final GRPO evaluation

So in this repo, GRPO did not fail because the implementation was meaningless. It failed because the policy it started from was not yet producing enough semantically correct box candidates for reward optimization to build on.

## Short Summary

SFT in this repo:

- taught the model to generate support boxes from image + question + options + answer
- used supervised gold `<boxes>` targets
- reduced training loss to about `0.93`
- did not run real validation during training
- achieved weak to zero heldout grounding depending on checkpoint

GRPO in this repo:

- started from the SFT checkpoint
- used online sampling with grouped relative rewards
- rewarded good box formatting and box overlap with GT
- trained LoRA adapters only
- improved train-time format compliance
- did not improve final heldout grounding in the saved run

