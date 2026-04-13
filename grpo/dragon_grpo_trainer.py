#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import sys
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Sized, Tuple, Union

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Sampler
from torchvision.transforms.functional import InterpolationMode
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from dragon_grpo_rewards import has_valid_boxes_wrapper, parse_boxes_from_text


def _ensure_local_internvl_importable() -> Path:
    root = Path(__file__).resolve().parents[1]
    local_pkg = root / "InternVL" / "internvl_chat"
    if str(local_pkg) not in sys.path:
        sys.path.insert(0, str(local_pkg))
    return local_pkg


RewardFunc = Callable[..., List[float]]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
QUAD_START_TOKEN = "<quad>"
QUAD_END_TOKEN = "</quad>"
REF_START_TOKEN = "<ref>"
REF_END_TOKEN = "</ref>"
BOX_START_TOKEN = "<box>"
BOX_END_TOKEN = "</box>"


@dataclass
class DragonGRPOConfig(TrainingArguments):
    max_prompt_length: int = field(default=1024)
    max_completion_length: int = field(default=128)
    num_generations: int = field(default=4)
    beta: float = field(default=0.0)
    temperature: float = field(default=0.7)
    top_p: float = field(default=0.9)

    conv_style: str = field(default="internlm2-chat")
    force_image_size: int = field(default=448)
    down_sample_ratio: float = field(default=0.5)
    use_llm_lora: int = field(default=16)
    use_backbone_lora: int = field(default=16)
    max_tiles: int = field(default=12)

    image_root: str = field(default="")
    disable_ref_model: bool = field(default=True)
    clip_eps: float = field(default=0.28)
    debug_print_every: int = field(default=50)
    force_box_output_instruction: bool = field(default=True)
    response_prefix: str = field(default="<boxes>\n")


class RepeatRandomSampler(Sampler[int]):
    def __init__(self, data_source: Sized, repeat_count: int):
        self.data_source = data_source
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)

    def __iter__(self):
        idxs = [idx for idx in torch.randperm(self.num_samples).tolist() for _ in range(self.repeat_count)]
        return iter(idxs)

    def __len__(self):
        return self.num_samples * self.repeat_count


class DragonEvalCallback(TrainerCallback):
    def __init__(self, eval_fn, eval_every_steps: int = 200):
        self.eval_fn = eval_fn
        self.eval_every_steps = eval_every_steps

    def on_step_end(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer")
        if trainer is None:
            return control
        if state.global_step > 0 and state.global_step % self.eval_every_steps == 0:
            metrics = self.eval_fn(trainer)
            if metrics:
                trainer.log(metrics)
        return control


def _build_transform(input_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> List[Image.Image]:
    width, height = image.size
    aspect_ratio = width / max(height, 1)
    targets = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda x: x[0] * x[1],
    )
    best = targets[0]
    best_diff = float("inf")
    for ratio in targets:
        val = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - val)
        if diff < best_diff:
            best_diff = diff
            best = ratio

    target_w = image_size * best[0]
    target_h = image_size * best[1]
    blocks = best[0] * best[1]
    resized = image.resize((target_w, target_h))

    tiles: List[Image.Image] = []
    tiles_per_row = max(target_w // image_size, 1)
    for idx in range(blocks):
        col = idx % tiles_per_row
        row = idx // tiles_per_row
        left = col * image_size
        top = row * image_size
        right = left + image_size
        bottom = top + image_size
        tiles.append(resized.crop((left, top, right, bottom)))

    if use_thumbnail and blocks > 1:
        tiles.append(image.resize((image_size, image_size)))

    return tiles


def _load_image_tensor(image_path: str, input_size: int, max_tiles: int, is_train: bool) -> torch.Tensor:
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
        # Keep one image tile by default for GRPO stability and to match your SFT pipeline
        # (`dynamic_image_size=false` in current configs). Multi-tile prompts can overflow
        # max prompt length and desync image tokens vs visual features.
        tiles = [image.resize((input_size, input_size))]
    transform = _build_transform(input_size)
    return torch.stack([transform(tile) for tile in tiles])


def selective_log_softmax(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return logits.log_softmax(dim=-1).gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


class DragonGRPOTrainer(Trainer):
    def __init__(
        self,
        model: str,
        reward_funcs: Union[RewardFunc, List[RewardFunc]],
        args: DragonGRPOConfig,
        train_dataset=None,
        eval_dataset=None,
        callbacks=None,
        optimizers=(None, None),
        attn_implementation: str = "flash_attention_2",
    ):
        if not isinstance(model, str):
            raise TypeError("model must be a checkpoint directory/path string")

        _ensure_local_internvl_importable()
        from internvl.conversation import get_conv_template  # type: ignore
        from internvl.model.internvl_chat import InternVLChatConfig, InternVLChatModel  # type: ignore

        self._get_conv_template = get_conv_template
        self.model_id = model
        self.args: DragonGRPOConfig = args

        config = InternVLChatConfig.from_pretrained(model)
        if config.llm_config.model_type == "internlm2":
            config.llm_config.attn_implementation = attn_implementation
        else:
            config.llm_config._attn_implementation = attn_implementation
        config.template = args.conv_style
        config.force_image_size = args.force_image_size

        model_obj = InternVLChatModel.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            config=config,
        )

        patch_size = model_obj.config.vision_config.patch_size
        if model_obj.config.vision_config.image_size != args.force_image_size:
            model_obj.vision_model.resize_pos_embeddings(
                old_size=model_obj.config.vision_config.image_size,
                new_size=args.force_image_size,
                patch_size=patch_size,
            )
            model_obj.config.vision_config.image_size = args.force_image_size

        model_obj.config.force_image_size = args.force_image_size
        model_obj.num_image_token = int((args.force_image_size // patch_size) ** 2 * (args.down_sample_ratio ** 2))

        tokenizer = AutoTokenizer.from_pretrained(
            model,
            add_eos_token=False,
            trust_remote_code=True,
            use_fast=False,
        )
        tokenizer.padding_side = "left"
        tokenizer.model_max_length = args.max_prompt_length

        token_list = [
            IMG_START_TOKEN,
            IMG_END_TOKEN,
            IMG_CONTEXT_TOKEN,
            QUAD_START_TOKEN,
            QUAD_END_TOKEN,
            REF_START_TOKEN,
            REF_END_TOKEN,
            BOX_START_TOKEN,
            BOX_END_TOKEN,
        ]
        num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        model_obj.img_context_token_id = img_context_token_id

        if num_new_tokens > 0:
            model_obj.language_model.resize_token_embeddings(len(tokenizer))
            output_embeddings = model_obj.language_model.get_output_embeddings().weight.data
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings[-num_new_tokens:] = output_embeddings_avg
            model_obj.config.llm_config.vocab_size = len(tokenizer)
            model_obj.language_model.config.vocab_size = len(tokenizer)

        model_obj.language_model.config.use_cache = False
        model_obj.vision_model.gradient_checkpointing = True
        model_obj.vision_model.encoder.gradient_checkpointing = True

        self._apply_lora(model_obj, args)

        if args.beta > 0.0 and not args.disable_ref_model:
            ref_model = deepcopy(model_obj).eval()
            for p in ref_model.parameters():
                p.requires_grad = False
        else:
            ref_model = None

        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_funcs = reward_funcs
        self.ref_model = ref_model
        self.processing_class = tokenizer

        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.beta = args.beta

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.pad_token_id = int(pad_id) if pad_id is not None else 0
        self.eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else self.pad_token_id

        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
        )

        self._metrics = defaultdict(list)

        def data_collator(features):
            return features

        super().__init__(
            model=model_obj,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

    def _apply_lora(self, model_obj, args: DragonGRPOConfig):
        for p in model_obj.parameters():
            p.requires_grad = False

        if args.use_backbone_lora:
            model_obj.wrap_backbone_lora(r=args.use_backbone_lora, lora_alpha=2 * args.use_backbone_lora)
            model_obj.config.use_backbone_lora = args.use_backbone_lora

        if args.use_llm_lora:
            model_obj.wrap_llm_lora(r=args.use_llm_lora, lora_alpha=2 * args.use_llm_lora)
            model_obj.config.use_llm_lora = args.use_llm_lora

        trainable = 0
        total = 0
        for _, p in model_obj.named_parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        print(f"Trainable params: {trainable:,} / {total:,} ({100.0 * trainable / max(total,1):.4f}%)")

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["message", "gt_boxes"]

    def _get_train_sampler(self) -> Sampler:
        return RepeatRandomSampler(self.train_dataset, self.num_generations)

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        return RepeatRandomSampler(eval_dataset, 1)

    def _extract_image_and_text(self, item: Dict) -> Tuple[str, str]:
        msg = item["message"][0]
        content = msg.get("content", [])
        image_path = ""
        text = ""
        for block in content:
            if block.get("type") == "image":
                image_path = block.get("image", "")
            elif block.get("type") == "text":
                text = block.get("text", "")

        if image_path and self.args.image_root and not os.path.isabs(image_path):
            image_path = str((Path(self.args.image_root) / image_path).resolve())

        return image_path, text

    def _box_output_instruction(self) -> str:
        return (
            "Return ONLY bounding boxes needed to justify the answer.\n"
            "Format strictly as:\n"
            "<boxes>\n"
            "<box> x1 y1 x2 y2 </box>\n"
            "...\n"
            "</boxes>\n"
            "No explanation."
        )

    @staticmethod
    def _clip_to_boxes_block(text: str) -> str:
        if not text:
            return "<boxes>\n</boxes>"
        m = re.search(r"(<boxes>.*?</boxes>)", text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
        # If model emits loose <box> tags without wrapper, wrap them.
        box_tags = re.findall(r"<box>.*?</box>", text, flags=re.IGNORECASE | re.DOTALL)
        if box_tags:
            joined = "\n".join(tag.strip() for tag in box_tags)
            return f"<boxes>\n{joined}\n</boxes>"
        return "<boxes>\n</boxes>"

    def _build_internvl_inputs(self, data_items: Sequence[Dict[str, Any]]):
        pixel_values_list = []
        num_patches_list = []
        rendered_prompts = []

        for data_item in data_items:
            image_path, user_text = self._extract_image_and_text(data_item)
            if not image_path:
                raise ValueError("Missing image path in data item message")

            pixel_values = _load_image_tensor(
                image_path=image_path,
                input_size=self.args.force_image_size,
                max_tiles=self.args.max_tiles,
                is_train=self.model.training,
            )
            pixel_values_list.append(pixel_values)
            num_patches = pixel_values.size(0)
            num_patches_list.append(num_patches)

            prompt = user_text
            if self.args.force_box_output_instruction:
                instr = self._box_output_instruction()
                if instr not in prompt:
                    prompt = (prompt.rstrip() + "\n\n" + instr).strip()
            if "<image>" not in prompt:
                prompt = "<image>\n" + prompt

            conv = self._get_conv_template(self.args.conv_style)
            system_prompt = conv.system_message

            image_tokens = f"{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches}{IMG_END_TOKEN}"
            prompt = prompt.replace("<image>", image_tokens, 1)

            turns: List[str] = []
            if system_prompt is not None:
                turns.append(f"<|im_start|>system\n{system_prompt}<|im_end|>\n")
            turns.append(f"<|im_start|>user\n{prompt}<|im_end|>\n")
            rendered_prompts.append("".join(turns))

        enc = self.processing_class(
            rendered_prompts,
            padding=True,
            return_tensors="pt",
            max_length=self.max_prompt_length,
            truncation=True,
        )

        flat_pixel_values = torch.cat(pixel_values_list, dim=0)
        image_flags = torch.ones(flat_pixel_values.size(0), dtype=torch.long)

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "pixel_values": flat_pixel_values,
            "image_flags": image_flags,
        }

    def _get_per_token_logps(
        self,
        model,
        input_ids,
        attention_mask,
        pixel_values,
        image_flags,
        logits_to_keep,
    ):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_flags=image_flags,
        )
        logits = outputs.logits[:, -logits_to_keep - 1 : -1, :]
        target_ids = input_ids[:, -logits_to_keep:]
        return selective_log_softmax(logits, target_ids)

    def _generate_with_vision(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        generation_config: GenerationConfig,
    ) -> torch.Tensor:
        model = self.model
        do_sample = bool(getattr(generation_config, "do_sample", True))
        temperature = float(getattr(generation_config, "temperature", 1.0) or 1.0)
        top_p = float(getattr(generation_config, "top_p", 1.0) or 1.0)
        max_new_tokens = int(getattr(generation_config, "max_new_tokens", 64))

        cur_ids = prompt_ids
        cur_mask = prompt_mask
        bsz = cur_ids.size(0)
        finished = torch.zeros(bsz, dtype=torch.bool, device=cur_ids.device)

        generated: List[torch.Tensor] = []

        for _ in range(max_new_tokens):
            outputs = model(
                input_ids=cur_ids,
                attention_mask=cur_mask,
                pixel_values=pixel_values,
                image_flags=torch.ones(pixel_values.size(0), dtype=torch.long, device=cur_ids.device),
            )
            next_logits = outputs.logits[:, -1, :]

            if do_sample:
                logits = next_logits / max(temperature, 1e-6)
                probs = F.softmax(logits, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                    cdf = torch.cumsum(sorted_probs, dim=-1)
                    keep = cdf <= top_p
                    keep[:, 0] = True
                    filtered = torch.zeros_like(probs)
                    filtered.scatter_(1, sorted_idx, sorted_probs * keep)
                    probs = filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_token = torch.argmax(next_logits, dim=-1)

            next_token = torch.where(
                finished,
                torch.full_like(next_token, self.pad_token_id),
                next_token,
            )
            generated.append(next_token)

            finished = finished | (next_token == self.eos_token_id) | (next_token == self.pad_token_id)
            if finished.all():
                break

            cur_ids = torch.cat([cur_ids, next_token.unsqueeze(1)], dim=1)
            cur_mask = torch.cat([cur_mask, torch.ones((bsz, 1), dtype=cur_mask.dtype, device=cur_mask.device)], dim=1)

        if not generated:
            return torch.full(
                (bsz, 1),
                fill_value=self.pad_token_id,
                dtype=prompt_ids.dtype,
                device=prompt_ids.device,
            )
        return torch.stack(generated, dim=1)

    def _compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        n = rewards.numel()
        g = self.num_generations
        usable = (n // g) * g
        if usable == 0:
            return torch.zeros_like(rewards)

        trimmed = rewards[:usable]
        grouped = trimmed.view(-1, g)
        means = grouped.mean(dim=1, keepdim=True)
        stds = grouped.std(dim=1, keepdim=True)
        adv = ((grouped - means) / (stds + 1e-4)).view(-1)

        if usable < n:
            tail = torch.zeros(n - usable, device=rewards.device, dtype=rewards.dtype)
            adv = torch.cat([adv, tail], dim=0)
        return adv

    def _prepare_inputs(self, inputs, is_train=True):
        prompt_inputs = self._build_internvl_inputs(inputs)
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        prompt_ids = prompt_inputs["input_ids"]
        prompt_mask = prompt_inputs["attention_mask"]
        pixel_values = prompt_inputs["pixel_values"]
        image_flags = prompt_inputs["image_flags"]

        model_dtype = next(self.model.parameters()).dtype
        if pixel_values.dtype != model_dtype:
            pixel_values = pixel_values.to(model_dtype)
            prompt_inputs["pixel_values"] = pixel_values

        # Seed assistant completion to start in box format mode.
        prefix_ids = None
        prefix_text = self.args.response_prefix or ""
        if prefix_text:
            prefix_ids = self.processing_class(
                prefix_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].to(prompt_ids.device)
            if prefix_ids.size(1) > 0:
                prefix_batch = prefix_ids.expand(prompt_ids.size(0), -1)
                prompt_ids = torch.cat([prompt_ids, prefix_batch], dim=1)
                prompt_mask = torch.cat(
                    [prompt_mask, torch.ones_like(prefix_batch, dtype=prompt_mask.dtype)],
                    dim=1,
                )

        with torch.no_grad():
            gen_cfg = deepcopy(self.generation_config)
            if not is_train:
                gen_cfg.do_sample = False
                gen_cfg.temperature = 0.0
                gen_cfg.top_p = 1.0

            completion_ids = self._generate_with_vision(
                prompt_ids=prompt_ids,
                prompt_mask=prompt_mask,
                pixel_values=pixel_values,
                generation_config=gen_cfg,
            )
            if prefix_ids is not None and prefix_ids.size(1) > 0:
                prefix_batch = prefix_ids.expand(completion_ids.size(0), -1)
                completion_ids = torch.cat([prefix_batch, completion_ids], dim=1)
            completion_mask = torch.ones_like(completion_ids, dtype=torch.int, device=completion_ids.device)

        is_eos = (completion_ids == self.eos_token_id) | (completion_ids == self.pad_token_id)
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1) - 1, dtype=torch.long, device=is_eos.device)
        has_eos = is_eos.any(dim=1)
        eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
        seq_idx = torch.arange(is_eos.size(1), device=is_eos.device).expand(is_eos.size(0), -1)
        completion_mask = completion_mask * (seq_idx <= eos_idx.unsqueeze(1)).int()

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        raw_completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        completions_text = [self._clip_to_boxes_block(t) for t in raw_completions_text]
        parsed_counts = [len(parse_boxes_from_text(t)) for t in completions_text]
        wrapper_hits = [1.0 if has_valid_boxes_wrapper(t) else 0.0 for t in completions_text]

        if is_train:
            with torch.no_grad():
                old_per_token_logps = self._get_per_token_logps(
                    self.model,
                    input_ids,
                    attention_mask,
                    pixel_values,
                    image_flags,
                    logits_to_keep,
                )
        else:
            old_per_token_logps = None

        rewards_per_func = torch.zeros(len(inputs), len(self.reward_funcs), device=prompt_ids.device)
        for i, reward_func in enumerate(self.reward_funcs):
            reward_kwargs = {}
            for k in inputs[0].keys():
                if k not in ["message", "completion"]:
                    reward_kwargs[k] = [x.get(k) for x in inputs]
            out = reward_func(
                prompts=[x["message"] for x in inputs],
                completions=completions_text,
                completion_ids=completion_ids,
                **reward_kwargs,
            )
            rewards_per_func[:, i] = torch.tensor(out, dtype=torch.float32, device=prompt_ids.device)

        rewards = rewards_per_func.sum(dim=1)
        advantages = self._compute_advantages(rewards)

        reward_names = [getattr(f, "__name__", str(f)) for f in self.reward_funcs]
        reward_per_func = rewards_per_func.mean(0)
        ds_name = inputs[0].get("dataset", "dataset")
        for i, name in enumerate(reward_names):
            self._metrics[f"{ds_name}/rewards/{name}"].append(reward_per_func[i].item())
        self._metrics[f"{ds_name}/reward"].append(rewards.mean().item())
        self._metrics[f"{ds_name}/parsed_boxes_mean"].append(float(sum(parsed_counts) / max(len(parsed_counts), 1)))
        self._metrics[f"{ds_name}/wrapper_rate"].append(float(sum(wrapper_hits) / max(len(wrapper_hits), 1)))

        if is_train and self.accelerator.is_main_process and self.args.debug_print_every > 0:
            step = int(getattr(self.state, "global_step", 0))
            if step % int(self.args.debug_print_every) == 0:
                sample_text = completions_text[0] if completions_text else ""
                sample_text = sample_text.replace("\n", "\\n")
                if len(sample_text) > 240:
                    sample_text = sample_text[:240] + "..."
                print(
                    f"[GRPO DEBUG] step={step} parsed_mean={self._metrics[f'{ds_name}/parsed_boxes_mean'][-1]:.3f} "
                    f"wrapper_rate={self._metrics[f'{ds_name}/wrapper_rate'][-1]:.3f} sample='{sample_text}'",
                    flush=True,
                )

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "pixel_values": pixel_values,
            "image_flags": image_flags,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "advantages": advantages,
            "completions_text": completions_text,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("return_outputs not supported")

        prompt_ids = inputs["prompt_ids"]
        prompt_mask = inputs["prompt_mask"]
        pixel_values = inputs["pixel_values"]
        image_flags = inputs["image_flags"]
        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps = self._get_per_token_logps(
            model,
            input_ids,
            attention_mask,
            pixel_values,
            image_flags,
            logits_to_keep,
        )

        if self.beta != 0.0 and self.ref_model is not None:
            with torch.no_grad():
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model,
                    input_ids,
                    attention_mask,
                    pixel_values,
                    image_flags,
                    logits_to_keep,
                )
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )
        else:
            per_token_kl = None

        advantages = inputs["advantages"]
        old_per_token_logps = inputs["old_per_token_logps"]

        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.args.clip_eps, 1 + self.args.clip_eps)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        if per_token_kl is not None:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        denom = completion_mask.sum(dim=1).clamp_min(1)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / denom).mean()

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)
        if per_token_kl is not None:
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / denom).mean()
            self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        _ = self._prepare_inputs(inputs, is_train=False)
        return torch.tensor(0.0, device=self.accelerator.device), None, None

    def log(self, logs, start_time=None):
        metrics = {k: sum(v) / len(v) for k, v in self._metrics.items() if len(v) > 0}
        if len(logs) > 0 and next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{k}": v for k, v in metrics.items()}
        logs = {**logs, **metrics}
        # Transformers versions differ on Trainer.log signature; this is compatible across both.
        super().log(logs)
        self._metrics.clear()
