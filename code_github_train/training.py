# -*- coding: utf-8 -*-
"""Safety LoRA training implementation, launched through train.py.

The frozen teacher runs to a randomly sampled denoising step. The LoRA UNet
then performs one differentiable update, whose predicted clean image and
after-step image supply the safety and quality objectives.

The default preset combines absolute predicted-image safety trend, an
after-latent delta term using a second checker, prompt CLIP alignment, TV,
an image variance floor, and weak noise regularization. Optional fixed phrase
embeddings augment the safety trend. Alternative safety modes are retained.
See training_config.py for the default experiment settings.
"""
from __future__ import annotations

import csv
import json
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import CLIPTokenizer, CLIPTextModel, CLIPModel, CLIPTokenizer as HFCLIPTokenizer
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel, StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from peft import LoraConfig, get_peft_model_state_dict
from diffusers.utils import convert_state_dict_to_diffusers

from training_config import split_sizes

# Paper: Jailbreak Vaccine, Eqs. (1)-(4), (9)-(22).
# Main Image Checker -> post-denoising safety; Vision Calibrator -> in-denoising delta.
# Auxiliary networks are used during training; inference loads the UNet LoRA only.




class PromptDataset(Dataset):
    def __init__(self, prompts: List[str]):
        self.prompts = prompts
    def __len__(self):
        return len(self.prompts)
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx]}


def collate_fn(batch):
    return {"prompts": [x["prompt"] for x in batch]}


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])



def get_weight_dtype(mixed_precision: str):
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def load_stable_diffusion_safety_checker(model_dir: str, weight_dtype, local_files_only: bool, device):
    return StableDiffusionSafetyChecker.from_pretrained(
        model_dir,
        torch_dtype=weight_dtype,
        local_files_only=local_files_only,
    ).to(device)


def read_and_split_prompts(csv_path: str, prompt_column: str, train_ratio: float, val_ratio: float, test_ratio: float, split_seed: int, max_prompts: Optional[int]):
    df = pd.read_csv(csv_path)
    if prompt_column not in df.columns:
        raise ValueError(f"Column {prompt_column} not found. Columns: {list(df.columns)}")

    prompts = []
    for _, row in df.iterrows():
        prompt = str(row[prompt_column]).strip()
        if prompt and prompt.lower() != "nan":
            prompts.append(prompt)

    if max_prompts is not None:
        prompts = prompts[:max_prompts]

    if len(prompts) < 10:
        raise RuntimeError(f"Too few prompts for 7:2:1 split: {len(prompts)}")

    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum <= 0:
        raise ValueError("split ratios must sum to a positive value")
    train_ratio = train_ratio / ratio_sum
    val_ratio = val_ratio / ratio_sum
    test_ratio = test_ratio / ratio_sum

    rng = random.Random(split_seed)
    idxs = list(range(len(prompts)))
    rng.shuffle(idxs)

    n = len(idxs)
    train_n, val_n, test_n = split_sizes(n, train_ratio, val_ratio, test_ratio)

    train_idx = set(idxs[:train_n])
    val_idx = set(idxs[train_n:train_n + val_n])
    test_idx = set(idxs[train_n + val_n:])

    train, val, test = [], [], []
    for i, prompt in enumerate(prompts):
        if i in train_idx:
            train.append(prompt)
        elif i in val_idx:
            val.append(prompt)
        elif i in test_idx:
            test.append(prompt)

    if not train or not val or not test:
        raise RuntimeError(f"Bad split: train={len(train)}, val={len(val)}, test={len(test)}")

    print(f"[Info] Split prompts: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def freeze_module(m: torch.nn.Module):
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)


def cast_trainable_parameters_to_fp32(module: torch.nn.Module):
    n = 0
    numel = 0
    for _, p in module.named_parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
            n += 1
            numel += p.numel()
    print(f"[Info] Cast trainable params to fp32: {n} tensors, {numel:,} params")


def print_trainable_parameters(module: torch.nn.Module):
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    print(f"[Info] trainable params: {trainable:,} / {total:,} ({100 * trainable / max(total, 1):.4f}%)")


def encode_prompts(tokenizer, text_encoder, prompts: List[str], negative_prompt: str, device):
    text_inputs = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    neg_inputs = tokenizer([negative_prompt] * len(prompts), padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        prompt_embeds = text_encoder(text_inputs.input_ids.to(device))[0]
        uncond_embeds = text_encoder(neg_inputs.input_ids.to(device))[0]
    return prompt_embeds, uncond_embeds


def autocast_ctx(device, enabled, dtype):
    if device.type == "cuda":
        try:
            return torch.amp.autocast(device_type="cuda", enabled=enabled, dtype=dtype)
        except TypeError:
            return torch.cuda.amp.autocast(enabled=enabled, dtype=dtype)
    return nullcontext()


def make_grad_scaler(device, enabled):
    if device.type != "cuda":
        enabled = False
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def guided_unet_noise(unet, latents, timestep, prompt_embeds, uncond_embeds, guidance_scale, scheduler, autocast_enabled, autocast_dtype, device):
    latent_input = latents
    if guidance_scale > 1.0:
        latent_input = torch.cat([latents] * 2)
        enc = torch.cat([uncond_embeds, prompt_embeds], dim=0)
    else:
        enc = prompt_embeds
    if hasattr(scheduler, "scale_model_input"):
        latent_input = scheduler.scale_model_input(latent_input, timestep)
    with autocast_ctx(device, autocast_enabled, autocast_dtype):
        noise = unet(latent_input, timestep, encoder_hidden_states=enc).sample
        if guidance_scale > 1.0:
            noise_u, noise_c = noise.chunk(2)
            noise = noise_u + guidance_scale * (noise_c - noise_u)
    return noise


def predict_x0_from_noise_pred(latents, noise_pred, timesteps, scheduler):
    alphas_cumprod = scheduler.alphas_cumprod.to(device=latents.device, dtype=latents.dtype)
    alpha_prod_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    beta_prod_t = 1.0 - alpha_prod_t
    pred_type = getattr(scheduler.config, "prediction_type", "epsilon")
    if pred_type == "epsilon":
        return (latents - beta_prod_t.sqrt() * noise_pred) / alpha_prod_t.sqrt()
    if pred_type == "v_prediction":
        return alpha_prod_t.sqrt() * latents - beta_prod_t.sqrt() * noise_pred
    raise ValueError(f"Unsupported prediction_type: {pred_type}")


def decode_latents_to_image01_chunked(vae, latents, chunk_size=1, clamp=False):
    scale = getattr(vae.config, "scaling_factor", 0.18215)
    vae_dtype = next(vae.parameters()).dtype
    outs = []
    for i in range(0, latents.shape[0], chunk_size):
        image = vae.decode(latents[i:i + chunk_size].to(dtype=vae_dtype) / scale).sample
        image01 = image / 2.0 + 0.5
        if clamp:
            image01 = image01.clamp(0, 1)
        outs.append(image01)
    return torch.cat(outs, dim=0)


def prepare_for_safety_checker(image01, size=224):
    image = image01.clamp(0, 1)
    if image.shape[-1] != size or image.shape[-2] != size:
        image = F.interpolate(image, size=(size, size), mode="bicubic", align_corners=False)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=image.device, dtype=image.dtype)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=image.device, dtype=image.dtype)
    return (image - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)


def safety_checker_nsfw_trend_chunked(safety_checker, image01, trend_mode="relu_square", softplus_beta=10.0, include_special_adjustment=False, chunk_size=1, special_adjustment_value=0.005):
    trends, raw_maxs, pos_counts = [], [], []
    for i in range(0, image01.shape[0], chunk_size):
        chunk = image01[i:i + chunk_size]
        pixel_values = prepare_for_safety_checker(chunk, size=224)
        pooled_output = safety_checker.vision_model(pixel_values)[1]
        image_embeds = safety_checker.visual_projection(pooled_output)
        image_embeds_norm = F.normalize(image_embeds.float(), dim=-1)
        concept_embeds_norm = F.normalize(safety_checker.concept_embeds.float(), dim=-1)
        special_embeds_norm = F.normalize(safety_checker.special_care_embeds.float(), dim=-1)
        concept_cos = image_embeds_norm @ concept_embeds_norm.t()
        special_cos = image_embeds_norm @ special_embeds_norm.t()
        concept_thresholds = safety_checker.concept_embeds_weights.float().view(1, -1)
        special_thresholds = safety_checker.special_care_embeds_weights.float().view(1, -1)
        special_raw = special_cos - special_thresholds
        if include_special_adjustment:
            special_active = (special_raw > 0).float().max(dim=1, keepdim=True).values.detach()
            adjustment = float(special_adjustment_value) * special_active
        else:
            adjustment = 0.0
        raw_scores = concept_cos - concept_thresholds + adjustment
        if trend_mode == "relu_square":
            positive = F.relu(raw_scores)
        else:
            positive = F.softplus(softplus_beta * raw_scores) / softplus_beta
        trends.append(torch.sum(positive ** 2, dim=-1))
        raw_maxs.append(raw_scores.max(dim=1).values)
        pos_counts.append((raw_scores > 0).float().sum(dim=1))
    return torch.cat(trends), torch.cat(raw_maxs), torch.cat(pos_counts)



def clip_preprocess_image(image01):
    image = image01.clamp(0, 1)
    image = F.interpolate(image, size=(224, 224), mode="bicubic", align_corners=False)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    return (image - mean) / std


def _clip_output_pooler(output):
    """Return pooled tensor from CLIP text/vision output across transformers versions."""
    if torch.is_tensor(output):
        return output
    if hasattr(output, "pooler_output") and torch.is_tensor(output.pooler_output):
        return output.pooler_output
    if hasattr(output, "last_hidden_state") and torch.is_tensor(output.last_hidden_state):
        return output.last_hidden_state[:, 0]
    if isinstance(output, (tuple, list)):
        if len(output) > 1 and torch.is_tensor(output[1]):
            return output[1]
        for item in output:
            if torch.is_tensor(item) and item.ndim == 2:
                return item
        for item in output:
            if torch.is_tensor(item) and item.ndim == 3:
                return item[:, 0]
    raise TypeError(f"Cannot extract pooled CLIP tensor from output type: {type(output)}")


def clip_text_features(clip_model, clip_tokenizer, prompts, device):
    text_inputs = clip_tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

    with torch.no_grad():
        if hasattr(clip_model, "text_model"):
            out = clip_model.text_model(**text_inputs)
            pooled = _clip_output_pooler(out)
            if hasattr(clip_model, "text_projection"):
                feats = clip_model.text_projection(pooled)
            else:
                feats = pooled
        else:
            out = clip_model(**text_inputs)
            feats = _clip_output_pooler(out)

        feats = F.normalize(feats.float(), dim=-1)

    return feats


def clip_image_features_from_image01(clip_model, image01):
    pixel_values = clip_preprocess_image(image01).to(dtype=next(clip_model.parameters()).dtype)

    if hasattr(clip_model, "vision_model"):
        out = clip_model.vision_model(pixel_values=pixel_values)
        pooled = _clip_output_pooler(out)
        if hasattr(clip_model, "visual_projection"):
            img_feats = clip_model.visual_projection(pooled)
        else:
            img_feats = pooled
    else:
        out = clip_model(pixel_values=pixel_values)
        img_feats = _clip_output_pooler(out)

    return F.normalize(img_feats.float(), dim=-1)


def clip_image_text_loss(clip_model, image01, text_feats):
    img_feats = clip_image_features_from_image01(clip_model, image01)
    sim = torch.sum(img_feats * text_feats, dim=-1)
    return (1.0 - sim).mean(), sim.detach().mean()


def parse_phrase_trend_phrases(s: str):
    return [x.strip() for x in str(s).split(",") if x.strip()]


def clip_phrase_nsfw_trend_chunked(clip_model, image01, phrase_text_feats, threshold=0.18, trend_mode="relu_square", softplus_beta=10.0, chunk_size=1):
    if phrase_text_feats is None or phrase_text_feats.numel() == 0:
        z = torch.zeros((image01.shape[0],), device=image01.device, dtype=torch.float32)
        return z, z, z

    trends, raw_maxs, pos_counts = [], [], []
    phrase_text_feats = F.normalize(phrase_text_feats.float(), dim=-1).to(image01.device)
    threshold = float(threshold)

    for i in range(0, image01.shape[0], chunk_size):
        chunk = image01[i:i + chunk_size]
        img_feats = clip_image_features_from_image01(clip_model, chunk)
        cos = img_feats @ phrase_text_feats.t()
        raw_scores = cos - threshold
        if trend_mode == "relu_square":
            positive = F.relu(raw_scores)
        else:
            positive = F.softplus(softplus_beta * raw_scores) / softplus_beta
        trends.append(torch.sum(positive ** 2, dim=-1))
        raw_maxs.append(raw_scores.max(dim=1).values)
        pos_counts.append((raw_scores > 0).float().sum(dim=1))

    return torch.cat(trends), torch.cat(raw_maxs), torch.cat(pos_counts)


def total_variation_loss(image01):
    x = image01.clamp(0, 1)
    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w


def variance_loss(image01, min_var):
    x = image01.clamp(0, 1)
    var = x.var(dim=(1, 2, 3), unbiased=False)
    return F.relu(min_var - var).mean(), var.detach().mean()


def append_log_row(path: Path, row: Dict[str, object]):
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_lora_weights(unet, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionPipeline.save_lora_weights(save_directory=str(output_dir), unet_lora_layers=state, safe_serialization=True)


def choose_step_index(args):
    n = args.num_inference_steps
    lo = max(1, int(np.ceil(n * args.step_min_ratio)))
    hi = min(n, int(np.floor(n * args.step_max_ratio)))
    # after_prev needs before_(t-1), so current step must be at least 2.
    if getattr(args, "safety_variant", "base") == "after_prev":
        lo = max(2, lo)
    if lo > hi:
        raise ValueError(f"Invalid step range: {lo}>{hi}")
    return random.randint(lo, hi)


def compute_after_delta_weight(args, step_idx: int):
    """Paper Eq. (15): ramp the in-denoising delta weight for later denoising steps.

    step_idx is the human-readable denoising step index in [1, num_inference_steps].
    Larger step_idx means closer to the end of the denoising trajectory.
    """
    n = args.num_inference_steps
    lo = max(1, int(np.ceil(n * args.step_min_ratio)))
    hi = min(n, int(np.floor(n * args.step_max_ratio)))
    if hi <= lo:
        progress = 1.0
    else:
        progress = (float(step_idx) - float(lo)) / max(float(hi - lo), 1.0)
        progress = max(0.0, min(1.0, progress))
    power = max(float(getattr(args, "after_delta_ramp_power", 1.0)), 1e-8)
    progress = progress ** power
    w_min = float(getattr(args, "lambda_after_delta_min", 0.0))
    w_max = float(getattr(args, "lambda_after_delta_max", 0.5))
    return w_min + (w_max - w_min) * progress


def compute_batch(prompts, tokenizer, text_encoder, vae, unet, teacher_unet, scheduler, safety_checker, after_delta_safety_checker, clip_model, clip_tokenizer, args, device, weight_dtype, autocast_enabled, autocast_dtype, require_grad: bool):
    """Compute one training/validation micro-batch.

    safety_checker:
        Main checker for mean(NSFWTrend(pred_x0_t)). In these experiments this is
        the normal / original stable-diffusion-safety-checker.
    after_delta_safety_checker:
        Checker used only for the after-latent delta auxiliary term. In scheme 2
        this should be the finetuned safety checker.
    """
    bsz = len(prompts)
    with torch.no_grad():
        prompt_embeds, uncond_embeds = encode_prompts(tokenizer, text_encoder, prompts, args.negative_prompt, device)
        clip_txt = clip_text_features(clip_model, clip_tokenizer, prompts, device) if args.lambda_clip > 0 else None
        phrase_texts = parse_phrase_trend_phrases(args.phrase_trend_phrases)
        phrase_trend_text_feats = clip_text_features(clip_model, clip_tokenizer, phrase_texts, device) if args.lambda_phrase_trend > 0 and len(phrase_texts) > 0 else None

    scheduler.set_timesteps(args.num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    step_idx = choose_step_index(args)
    i = step_idx - 1
    t = timesteps[i]

    latents = torch.randn((bsz, unet.config.in_channels, args.resolution // 8, args.resolution // 8), device=device, dtype=weight_dtype)
    latents = latents * scheduler.init_noise_sigma

    with torch.no_grad():
        running = latents
        latents_before_prev = None
        for j in range(i):
            if args.safety_variant == "after_prev" and j == i - 1:
                latents_before_prev = running.detach().clone()
            tj = timesteps[j]
            noise_j = guided_unet_noise(teacher_unet, running, tj, prompt_embeds, uncond_embeds, args.guidance_scale, scheduler, autocast_enabled, autocast_dtype, device)
            running = scheduler.step(noise_j, tj, running).prev_sample

        latents_before = running.detach().clone()
        if latents_before_prev is None:
            latents_before_prev = latents_before.detach().clone()

        image_before = decode_latents_to_image01_chunked(vae, latents_before, args.vae_decode_chunk_size, clamp=False)

        # Main checker, used for original delta mode and for logging.
        trend_before_main, _, _ = safety_checker_nsfw_trend_chunked(
            safety_checker,
            image_before,
            args.trend_mode,
            args.softplus_beta,
            args.include_special_adjustment,
            args.safety_checker_chunk_size,
            args.special_adjustment_value,
        )
        phrase_trend_before_main = torch.zeros_like(trend_before_main)
        if args.lambda_phrase_trend > 0 and phrase_trend_text_feats is not None:
            phrase_trend_before_main, _, _ = clip_phrase_nsfw_trend_chunked(
                clip_model,
                image_before,
                phrase_trend_text_feats,
                args.phrase_trend_threshold,
                args.trend_mode,
                args.softplus_beta,
                args.safety_checker_chunk_size,
            )
            trend_before_main = trend_before_main + float(args.lambda_phrase_trend) * phrase_trend_before_main

        # The after-delta term can use a different checker, e.g. finetuned checker.
        if args.safety_loss_mode == "absolute_predx0_plus_after_delta" and after_delta_safety_checker is not safety_checker:
            trend_before_after_delta, _, _ = safety_checker_nsfw_trend_chunked(
                after_delta_safety_checker,
                image_before,
                args.trend_mode,
                args.softplus_beta,
                args.include_special_adjustment,
                args.safety_checker_chunk_size,
                args.special_adjustment_value,
            )
            if args.lambda_phrase_trend > 0 and phrase_trend_text_feats is not None:
                trend_before_after_delta = trend_before_after_delta + float(args.lambda_phrase_trend) * phrase_trend_before_main
        else:
            trend_before_after_delta = trend_before_main.detach()

        if args.safety_variant == "after_prev":
            image_before_prev = decode_latents_to_image01_chunked(vae, latents_before_prev, args.vae_decode_chunk_size, clamp=False)
            trend_before_prev, _, _ = safety_checker_nsfw_trend_chunked(
                safety_checker,
                image_before_prev,
                args.trend_mode,
                args.softplus_beta,
                args.include_special_adjustment,
                args.safety_checker_chunk_size,
                args.special_adjustment_value,
            )
            if args.lambda_phrase_trend > 0 and phrase_trend_text_feats is not None:
                phrase_trend_before_prev, _, _ = clip_phrase_nsfw_trend_chunked(
                    clip_model,
                    image_before_prev,
                    phrase_trend_text_feats,
                    args.phrase_trend_threshold,
                    args.trend_mode,
                    args.softplus_beta,
                    args.safety_checker_chunk_size,
                )
                trend_before_prev = trend_before_prev + float(args.lambda_phrase_trend) * phrase_trend_before_prev
        else:
            trend_before_prev = trend_before_main.detach()

        noise_teacher = guided_unet_noise(teacher_unet, latents_before, t, prompt_embeds, uncond_embeds, args.guidance_scale, scheduler, autocast_enabled, autocast_dtype, device)

    ctx = nullcontext() if require_grad else torch.no_grad()
    with ctx:
        noise_lora = guided_unet_noise(unet, latents_before, t, prompt_embeds, uncond_embeds, args.guidance_scale, scheduler, autocast_enabled, autocast_dtype, device)
        loss_noise_delta = F.mse_loss(noise_lora.float(), noise_teacher.float(), reduction="mean")

        t_tensor = torch.full((bsz,), int(t.item()), device=device, dtype=torch.long)
        pred_x0_latents = predict_x0_from_noise_pred(latents_before, noise_lora, t_tensor, scheduler)
        pred_x0_img = decode_latents_to_image01_chunked(vae, pred_x0_latents, args.vae_decode_chunk_size, clamp=False)

        after_latents = scheduler.step(noise_lora, t, latents_before).prev_sample
        after_img = decode_latents_to_image01_chunked(vae, after_latents, args.vae_decode_chunk_size, clamp=False)

        # Main absolute loss: always computed by the main checker.
        # For these two experiments, main checker = normal stable-diffusion-safety-checker.
        trend_pred_x0, raw_pred_x0, pos_pred_x0 = safety_checker_nsfw_trend_chunked(
            safety_checker,
            pred_x0_img,
            args.trend_mode,
            args.softplus_beta,
            args.include_special_adjustment,
            args.safety_checker_chunk_size,
            args.special_adjustment_value,
        )
        phrase_trend_pred_x0 = torch.zeros_like(trend_pred_x0)
        phrase_raw_pred_x0 = torch.zeros_like(raw_pred_x0)
        phrase_pos_pred_x0 = torch.zeros_like(pos_pred_x0)
        if args.lambda_phrase_trend > 0 and phrase_trend_text_feats is not None:
            phrase_trend_pred_x0, phrase_raw_pred_x0, phrase_pos_pred_x0 = clip_phrase_nsfw_trend_chunked(
                clip_model,
                pred_x0_img,
                phrase_trend_text_feats,
                args.phrase_trend_threshold,
                args.trend_mode,
                args.softplus_beta,
                args.safety_checker_chunk_size,
            )
            trend_pred_x0 = trend_pred_x0 + float(args.lambda_phrase_trend) * phrase_trend_pred_x0

        if args.safety_loss_mode == "delta":
            # Original behavior, all computed by the main checker.
            trend_after_main, raw_after_main, pos_after_main = safety_checker_nsfw_trend_chunked(
                safety_checker,
                after_img,
                args.trend_mode,
                args.softplus_beta,
                args.include_special_adjustment,
                args.safety_checker_chunk_size,
                args.special_adjustment_value,
            )
            if args.lambda_phrase_trend > 0 and phrase_trend_text_feats is not None:
                phrase_trend_after_main, _, _ = clip_phrase_nsfw_trend_chunked(
                    clip_model,
                    after_img,
                    phrase_trend_text_feats,
                    args.phrase_trend_threshold,
                    args.trend_mode,
                    args.softplus_beta,
                    args.safety_checker_chunk_size,
                )
                trend_after_main = trend_after_main + float(args.lambda_phrase_trend) * phrase_trend_after_main
            if args.safety_on == "pred_x0":
                trend_safety, raw_safety, pos_safety = trend_pred_x0, raw_pred_x0, pos_pred_x0
            else:
                trend_safety, raw_safety, pos_safety = trend_after_main, raw_after_main, pos_after_main
            loss_safety_base = F.relu(trend_safety - trend_before_main).mean()

            if args.safety_variant == "after_prev":
                loss_after_prev = F.relu(trend_after_main - trend_before_prev).mean()
                after_prev_delta = (trend_after_main - trend_before_prev).detach().mean()
            else:
                loss_after_prev = torch.zeros((), device=device)
                after_prev_delta = torch.zeros((), device=device)
            after_delta_weight = 0.0
            trend_after_for_log = trend_after_main
            loss_safety = loss_safety_base + args.lambda_after_prev * loss_after_prev

        elif args.safety_loss_mode == "absolute_predx0":
            # Scheme 1: only mean(NSFWTrend(pred_x0_t)), computed by normal checker.
            trend_safety, raw_safety, pos_safety = trend_pred_x0, raw_pred_x0, pos_pred_x0
            loss_safety_base = trend_pred_x0.mean()  # Paper Eq. (16): L_post.
            loss_after_prev = torch.zeros((), device=device)
            after_prev_delta = torch.zeros((), device=device)
            after_delta_weight = 0.0
            # Log after-image trend under the same normal checker without adding its gradient.
            with torch.no_grad():
                trend_after_for_log, _, _ = safety_checker_nsfw_trend_chunked(
                    safety_checker,
                    after_img.detach(),
                    args.trend_mode,
                    args.softplus_beta,
                    args.include_special_adjustment,
                    args.safety_checker_chunk_size,
                    args.special_adjustment_value,
                )
                if args.lambda_phrase_trend > 0 and phrase_trend_text_feats is not None:
                    phrase_trend_after_for_log, _, _ = clip_phrase_nsfw_trend_chunked(
                        clip_model,
                        after_img.detach(),
                        phrase_trend_text_feats,
                        args.phrase_trend_threshold,
                        args.trend_mode,
                        args.softplus_beta,
                        args.safety_checker_chunk_size,
                    )
                    trend_after_for_log = trend_after_for_log + float(args.lambda_phrase_trend) * phrase_trend_after_for_log
            loss_safety = loss_safety_base

        elif args.safety_loss_mode == "absolute_predx0_plus_after_delta":
            # Scheme 2:
            #   main term: mean(NSFWTrend(pred_x0_t)) with normal checker.
            #   auxiliary term: ReLU(NSFWTrend(after_img)-NSFWTrend(before_t)) with finetuned checker.
            trend_safety, raw_safety, pos_safety = trend_pred_x0, raw_pred_x0, pos_pred_x0
            loss_safety_base = trend_pred_x0.mean()

            trend_after_delta, _, _ = safety_checker_nsfw_trend_chunked(
                after_delta_safety_checker,
                after_img,
                args.trend_mode,
                args.softplus_beta,
                args.include_special_adjustment,
                args.safety_checker_chunk_size,
                args.special_adjustment_value,
            )
            if args.lambda_phrase_trend > 0 and phrase_trend_text_feats is not None:
                phrase_trend_after_delta, _, _ = clip_phrase_nsfw_trend_chunked(
                    clip_model,
                    after_img,
                    phrase_trend_text_feats,
                    args.phrase_trend_threshold,
                    args.trend_mode,
                    args.softplus_beta,
                    args.safety_checker_chunk_size,
                )
                trend_after_delta = trend_after_delta + float(args.lambda_phrase_trend) * phrase_trend_after_delta
            loss_after_prev = F.relu(trend_after_delta - trend_before_after_delta).mean()  # Eq. (14): L_in.
            after_prev_delta = (trend_after_delta - trend_before_after_delta).detach().mean()
            after_delta_weight = compute_after_delta_weight(args, step_idx)
            trend_after_for_log = trend_after_delta
            loss_safety = loss_safety_base + float(after_delta_weight) * loss_after_prev

        else:
            raise ValueError(f"Unknown safety_loss_mode: {args.safety_loss_mode}")

        clip_start = int(np.ceil(args.num_inference_steps * args.clip_start_ratio))
        if args.lambda_clip > 0 and step_idx >= clip_start:
            loss_clip, clip_sim = clip_image_text_loss(clip_model, pred_x0_img, clip_txt)
            if args.clip_loss_weight_ramp:
                ramp = (step_idx - clip_start + 1) / max(args.num_inference_steps - clip_start + 1, 1)
                loss_clip = loss_clip * float(ramp)
        else:
            loss_clip = torch.zeros((), device=device)
            clip_sim = torch.zeros((), device=device)

        loss_tv = total_variation_loss(pred_x0_img)
        loss_var, image_var = variance_loss(pred_x0_img, args.min_image_var)

        loss_total = (
            args.lambda_safety * loss_safety
            + args.lambda_clip * loss_clip
            + args.lambda_tv * loss_tv
            + args.lambda_var * loss_var
            + args.lambda_noise_delta * loss_noise_delta
        )

    return {
        "loss_total": loss_total,
        "loss_safety": loss_safety,
        "loss_safety_base": loss_safety_base,
        "loss_after_prev": loss_after_prev,
        "loss_clip": loss_clip,
        "loss_tv": loss_tv,
        "loss_var": loss_var,
        "loss_noise_delta": loss_noise_delta,
        "trend_before_mean": trend_before_main.detach().mean(),
        "trend_before_prev_mean": trend_before_prev.detach().mean(),
        "trend_safety_mean": trend_safety.detach().mean(),
        "trend_after_mean": trend_after_for_log.detach().mean(),
        "after_prev_delta_mean": after_prev_delta.detach(),
        "after_delta_weight_mean": torch.tensor(float(after_delta_weight), device=device),
        "raw_safety_max_mean": raw_safety.detach().mean(),
        "pos_safety_count_mean": pos_safety.detach().mean(),
        "phrase_trend_pred_x0_mean": phrase_trend_pred_x0.detach().mean(),
        "phrase_raw_max_mean": phrase_raw_pred_x0.detach().mean(),
        "phrase_pos_count_mean": phrase_pos_pred_x0.detach().mean(),
        "clip_sim_mean": clip_sim.detach(),
        "image_var_mean": image_var.detach(),
        "chosen_step_index_mean": torch.tensor(float(step_idx), device=device),
        "chosen_scheduler_timestep_mean": torch.tensor(float(int(t.item())), device=device),
    }


def run_validation(val_loader, tokenizer, text_encoder, vae, unet, teacher_unet, scheduler, safety_checker, after_delta_safety_checker, clip_model, clip_tokenizer, args, device, weight_dtype, autocast_enabled, autocast_dtype):
    unet.eval()
    rng_state = None
    if getattr(args, "fixed_validation_conditions", False):
        rng_state = capture_rng_state()
        seed_everything(int(getattr(args, "validation_seed", 20260723)))
    keys = ["loss_total", "loss_safety", "loss_safety_base", "loss_after_prev", "loss_clip", "loss_tv", "loss_var", "loss_noise_delta", "trend_before_mean", "trend_before_prev_mean", "trend_safety_mean", "trend_after_mean", "after_prev_delta_mean", "after_delta_weight_mean", "raw_safety_max_mean", "pos_safety_count_mean", "phrase_trend_pred_x0_mean", "phrase_raw_max_mean", "phrase_pos_count_mean", "clip_sim_mean", "image_var_mean", "chosen_step_index_mean", "chosen_scheduler_timestep_mean"]
    sums = {k: 0.0 for k in keys}
    count = 0
    try:
        with torch.no_grad():
            for batch in val_loader:
                out = compute_batch(batch["prompts"], tokenizer, text_encoder, vae, unet, teacher_unet, scheduler, safety_checker, after_delta_safety_checker, clip_model, clip_tokenizer, args, device, weight_dtype, autocast_enabled, autocast_dtype, require_grad=False)
                bsz = len(batch["prompts"])
                for k in keys:
                    sums[k] += float(out[k].detach().cpu()) * bsz
                count += bsz
    finally:
        if rng_state is not None:
            restore_rng_state(rng_state)
    unet.train()
    row = {f"val_{k}": sums[k] / max(count, 1) for k in keys}
    row["val_num_prompts"] = count
    # Three model-selection scores:
    #   1) val_loss_total: strict total objective
    #   2) val_loss_safety: safety-only objective
    #   3) val_score_balanced: halfway between total and safety
    row["val_score_total"] = row["val_loss_total"]
    row["val_score_safety"] = row["val_loss_safety"]
    row["val_score_balanced"] = 0.5 * row["val_loss_total"] + 0.5 * row["val_loss_safety"]
    row["val_score"] = row["val_score_total"]
    return row


def train(args):
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Install a CUDA-enabled PyTorch build or select --device cpu --mixed_precision no.")
    if args.device == "cpu" and args.mixed_precision != "no":
        raise ValueError("CPU training requires --mixed_precision no.")
    seed_everything(args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_log_path = output_dir / "training_log.csv"
    val_log_path = output_dir / "validation_log.csv"
    with open(output_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    device = torch.device(args.device)
    weight_dtype = get_weight_dtype(args.mixed_precision)
    if device.type == "cuda" and args.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    writer = None
    if args.use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = Path(args.tensorboard_log_dir).resolve() if args.tensorboard_log_dir else output_dir / "runs"
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))
        print(f"[Info] TensorBoard log dir: {tb_dir}")

    print("=" * 100)
    print("[Config]")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print(f"device: {device}")
    print(f"weight_dtype: {weight_dtype}")
    print("=" * 100)

    train_prompts, val_prompts_all, test_prompts = read_and_split_prompts(
        args.csv_path,
        args.prompt_column,
        args.split_train_ratio,
        args.split_val_ratio,
        args.split_test_ratio,
        args.split_seed,
        args.max_prompts,
    )
    val_prompts = val_prompts_all[:args.val_max_prompts] if args.val_max_prompts else val_prompts_all
    pd.DataFrame({"prompt": train_prompts}).to_csv(output_dir / "train_prompts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"prompt": val_prompts_all}).to_csv(output_dir / "val_prompts_all.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"prompt": val_prompts}).to_csv(output_dir / "val_prompts_used.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"prompt": test_prompts}).to_csv(output_dir / "test_prompts.csv", index=False, encoding="utf-8-sig")

    train_loader = DataLoader(PromptDataset(train_prompts), batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True, pin_memory=(device.type == "cuda"))
    if len(train_loader) == 0:
        raise ValueError("train_batch_size exceeds the training split size; lower --train_batch_size or provide more prompts.")
    val_loader = DataLoader(PromptDataset(val_prompts), batch_size=args.val_batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn, drop_last=False, pin_memory=(device.type == "cuda"))

    print("[Info] Loading SD1.4 / teacher / safety checkers / CLIP...")
    tokenizer = CLIPTokenizer.from_pretrained(args.base_model_id, subfolder="tokenizer", local_files_only=args.local_files_only)
    text_encoder = CLIPTextModel.from_pretrained(args.base_model_id, subfolder="text_encoder", torch_dtype=weight_dtype, local_files_only=args.local_files_only).to(device)
    vae = AutoencoderKL.from_pretrained(args.base_model_id, subfolder="vae", torch_dtype=weight_dtype, local_files_only=args.local_files_only).to(device)
    unet = UNet2DConditionModel.from_pretrained(args.base_model_id, subfolder="unet", torch_dtype=weight_dtype, local_files_only=args.local_files_only).to(device)
    teacher_unet = UNet2DConditionModel.from_pretrained(args.base_model_id, subfolder="unet", torch_dtype=weight_dtype, local_files_only=args.local_files_only).to(device)
    scheduler = DDIMScheduler.from_pretrained(args.base_model_id, subfolder="scheduler", local_files_only=args.local_files_only)
    safety_checker = load_stable_diffusion_safety_checker(args.safety_checker_id, weight_dtype, args.local_files_only, device)
    after_delta_checker_id = args.after_delta_safety_checker_id.strip() if args.after_delta_safety_checker_id else ""
    if after_delta_checker_id and Path(after_delta_checker_id).resolve() != Path(args.safety_checker_id).resolve():
        print(f"[Info] Loading after-delta safety checker: {after_delta_checker_id}")
        after_delta_safety_checker = load_stable_diffusion_safety_checker(after_delta_checker_id, weight_dtype, args.local_files_only, device)
    else:
        after_delta_safety_checker = safety_checker

    clip_model = CLIPModel.from_pretrained(args.clip_model_id, torch_dtype=weight_dtype, local_files_only=args.clip_local_files_only).to(device)
    clip_tokenizer = HFCLIPTokenizer.from_pretrained(args.clip_model_id, local_files_only=args.clip_local_files_only)

    freeze_module(text_encoder); freeze_module(vae); freeze_module(teacher_unet); freeze_module(safety_checker); freeze_module(after_delta_safety_checker); freeze_module(clip_model)

    if args.channels_last and device.type == "cuda":
        unet.to(memory_format=torch.channels_last); teacher_unet.to(memory_format=torch.channels_last); vae.to(memory_format=torch.channels_last); safety_checker.vision_model.to(memory_format=torch.channels_last); after_delta_safety_checker.vision_model.to(memory_format=torch.channels_last)
        print("[Info] channels_last enabled.")
    if args.enable_vae_slicing and hasattr(vae, "enable_slicing"):
        vae.enable_slicing(); print("[Info] VAE slicing enabled.")

    unet.requires_grad_(False)
    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    unet.add_adapter(LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha, init_lora_weights="gaussian", target_modules=target_modules, lora_dropout=args.lora_dropout))
    cast_trainable_parameters_to_fp32(unet)
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing(); print("[Info] UNet gradient checkpointing enabled.")
    if args.enable_xformers:
        try:
            unet.enable_xformers_memory_efficient_attention(); teacher_unet.enable_xformers_memory_efficient_attention(); print("[Info] xFormers enabled.")
        except Exception as e:
            print(f"[Warning] xFormers enable failed: {repr(e)}")
    unet.train(); print_trainable_parameters(unet)

    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    fp16_trainable = [name for name, p in unet.named_parameters() if p.requires_grad and p.dtype == torch.float16]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
    autocast_enabled = (device.type == "cuda" and args.mixed_precision in ["fp16", "bf16"])
    autocast_dtype = weight_dtype if args.mixed_precision in ["fp16", "bf16"] else torch.float32
    scaler = make_grad_scaler(device, enabled=(device.type == "cuda" and args.mixed_precision == "fp16" and len(fp16_trainable) == 0))
    print(f"[Info] GradScaler enabled: {scaler.is_enabled()}")

    keys = ["loss_total", "loss_safety", "loss_safety_base", "loss_after_prev", "loss_clip", "loss_tv", "loss_var", "loss_noise_delta", "trend_before_mean", "trend_before_prev_mean", "trend_safety_mean", "trend_after_mean", "after_prev_delta_mean", "after_delta_weight_mean", "raw_safety_max_mean", "pos_safety_count_mean", "phrase_trend_pred_x0_mean", "phrase_raw_max_mean", "phrase_pos_count_mean", "clip_sim_mean", "image_var_mean", "chosen_step_index_mean", "chosen_scheduler_timestep_mean"]
    best_metrics = {
        "val_loss_total": {"best": float("inf"), "dir": output_dir / "best_val_loss_total_lora"},
        "val_loss_safety": {"best": float("inf"), "dir": output_dir / "best_val_loss_safety_lora"},
        "val_score_balanced": {"best": float("inf"), "dir": output_dir / "best_val_score_balanced_lora"},
    }
    data_iter = iter(train_loader)
    global_step = 0
    pbar = tqdm(total=args.max_train_steps, desc="safety-lora")

    while global_step < args.max_train_steps:
        optimizer.zero_grad(set_to_none=True)
        sums = {k: 0.0 for k in keys}
        for _ in range(args.gradient_accumulation_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader); batch = next(data_iter)
            out = compute_batch(batch["prompts"], tokenizer, text_encoder, vae, unet, teacher_unet, scheduler, safety_checker, after_delta_safety_checker, clip_model, clip_tokenizer, args, device, weight_dtype, autocast_enabled, autocast_dtype, require_grad=True)
            loss = out["loss_total"] / args.gradient_accumulation_steps
            if scaler.is_enabled(): scaler.scale(loss).backward()
            else: loss.backward()
            for k in keys:
                sums[k] += float(out[k].detach().cpu())
        if scaler.is_enabled():
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm); scaler.step(optimizer); scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm); optimizer.step()
        global_step += 1; pbar.update(1)
        n = float(args.gradient_accumulation_steps)
        train_row = {"step": global_step, "split": "train", "scheme": f"mid_predx0_clip_quality_{args.safety_loss_mode}", "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha, **{k: sums[k] / n for k in keys}, "lr": optimizer.param_groups[0]["lr"], "batch_size": args.train_batch_size, "gradient_accumulation_steps": args.gradient_accumulation_steps, "effective_batch_size": args.train_batch_size * args.gradient_accumulation_steps}
        if global_step % args.log_every_steps == 0 or global_step == 1:
            append_log_row(train_log_path, train_row)
            if writer is not None:
                for k, v in train_row.items():
                    if isinstance(v, (int, float)): writer.add_scalar(f"train/{k}", v, global_step)
            pbar.set_postfix({"loss": f"{train_row['loss_total']:.4f}", "safe": f"{train_row['loss_safety']:.4f}", "clip": f"{train_row['loss_clip']:.4f}", "best_total": f"{best_metrics['val_loss_total']['best']:.4f}" if best_metrics['val_loss_total']['best'] < float('inf') else "NA"})
        if args.val_every_steps > 0 and (global_step % args.val_every_steps == 0 or global_step == 1):
            print(f"\n[Info] Validation at step {global_step}...")
            val_row = run_validation(val_loader, tokenizer, text_encoder, vae, unet, teacher_unet, scheduler, safety_checker, after_delta_safety_checker, clip_model, clip_tokenizer, args, device, weight_dtype, autocast_enabled, autocast_dtype)
            val_row = {"step": global_step, "split": "val", "scheme": f"mid_predx0_clip_quality_{args.safety_loss_mode}", "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha, **val_row}
            append_log_row(val_log_path, val_row)
            if writer is not None:
                for k, v in val_row.items():
                    if isinstance(v, (int, float)): writer.add_scalar(f"val/{k}", v, global_step)
            for metric_name, metric_state in best_metrics.items():
                current_score = float(val_row[metric_name])
                if current_score < metric_state["best"]:
                    metric_state["best"] = current_score
                    best_dir = metric_state["dir"]
                    print(f"[Info] New best {metric_name}={current_score:.6f}; saving {best_dir}")
                    save_lora_weights(unet, best_dir)
                    info = dict(val_row)
                    info["best_metric"] = metric_name
                    info["best_score"] = current_score
                    with open(output_dir / f"best_{metric_name}_info.json", "w", encoding="utf-8") as f:
                        json.dump(info, f, ensure_ascii=False, indent=2)
                    # Backward-compatible alias: keep best_val_lora as total-loss best.
                    if metric_name == "val_loss_total":
                        alias_dir = output_dir / "best_val_lora"
                        save_lora_weights(unet, alias_dir)
                        with open(output_dir / "best_val_info.json", "w", encoding="utf-8") as f:
                            json.dump(info, f, ensure_ascii=False, indent=2)
        if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
            ckpt_dir = output_dir / f"checkpoint_step_{global_step:06d}"; print(f"\n[Info] Saving checkpoint: {ckpt_dir}"); save_lora_weights(unet, ckpt_dir)
        if args.empty_cache_every_steps > 0 and global_step % args.empty_cache_every_steps == 0 and torch.cuda.is_available(): torch.cuda.empty_cache()

    pbar.close()
    final_dir = output_dir / "final_lora"; print(f"[Info] Saving final LoRA: {final_dir}"); save_lora_weights(unet, final_dir); save_lora_weights(unet, output_dir)
    if writer is not None: writer.flush(); writer.close()
    print("=" * 100); print("[Done] Safety LoRA training finished."); print(f"[Done] Output dir: {output_dir}"); print(f"[Done] Best total LoRA: {output_dir / 'best_val_loss_total_lora'}"); print(f"[Done] Best safety LoRA: {output_dir / 'best_val_loss_safety_lora'}"); print(f"[Done] Best balanced LoRA: {output_dir / 'best_val_score_balanced_lora'}"); print("=" * 100)

