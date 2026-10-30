"""Portable CLI configuration; no machine-learning dependencies are imported here."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Jailbreak Vaccine: SD1.4 LoRA training with multi-state NSFW suppression and semantic-fidelity regularization", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--csv_path", type=str, default=None)
    p.add_argument("--prompt_column", type=str, default="perturbed_text")
    p.add_argument("--max_prompts", type=int, default=None)
    # Dataset split. Default 7:2:1 = train:val:test.
    p.add_argument("--split_train_ratio", type=float, default=0.7)
    p.add_argument("--split_val_ratio", type=float, default=0.2)
    p.add_argument("--split_test_ratio", type=float, default=0.1)
    # Kept for backward compatibility. If provided by old scripts, it is ignored by the 7:2:1 splitter.
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--val_max_prompts", type=int, default=64)
    p.add_argument("--split_seed", type=int, default=42)

    p.add_argument("--base_model_id", type=str, default=None)
    p.add_argument("--safety_checker_id", type=str, default=None, help="Main safety checker. For these experiments, use the normal stable-diffusion-safety-checker for mean(NSFWTrend(pred_x0_t)).")
    p.add_argument("--after_delta_safety_checker_id", type=str, default="", help="Vision Calibrator checkpoint for the in-denoising safety term (paper Eq. 14).")
    p.add_argument("--clip_model_id", type=str, default="openai/clip-vit-large-patch14")
    p.add_argument("--clip_local_files_only", action=argparse.BooleanOptionalAction)
    p.add_argument("--local_files_only", action=argparse.BooleanOptionalAction)

    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--save_every_steps", type=int, default=250)
    p.add_argument("--log_every_steps", type=int, default=10)
    p.add_argument("--val_every_steps", type=int, default=100)
    p.add_argument("--use_tensorboard", action=argparse.BooleanOptionalAction)
    p.add_argument("--tensorboard_log_dir", type=str, default=None)

    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--val_batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_train_steps", type=int, default=2000)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)

    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--adam_weight_decay", type=float, default=1e-2)
    p.add_argument("--adam_epsilon", type=float, default=1e-8)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=8)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--target_modules", type=str, default="to_q,to_k,to_v,to_out.0")

    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--negative_prompt", type=str, default="")
    p.add_argument("--step_min_ratio", type=float, default=0.35)
    p.add_argument("--step_max_ratio", type=float, default=1.0)
    p.add_argument("--clip_start_ratio", type=float, default=0.60)

    p.add_argument("--lambda_safety", type=float, default=1.0)
    p.add_argument("--lambda_clip", type=float, default=0.10)
    p.add_argument("--lambda_tv", type=float, default=0.001)
    p.add_argument("--lambda_var", type=float, default=0.05)
    p.add_argument("--lambda_noise_delta", type=float, default=0.01)
    p.add_argument("--min_image_var", type=float, default=0.02)
    p.add_argument("--safety_on", type=str, default="pred_x0", choices=["pred_x0", "after_latent"])
    # Safety loss modes:
    #   delta:
    #       original behavior. loss_safety = ReLU(NSFWTrend(safety_img) - NSFWTrend(before_t))
    #       where safety_img is selected by --safety_on.
    #   absolute_predx0:
    #       scheme 1. loss_safety = mean(NSFWTrend(pred_x0_t)).
    #   absolute_predx0_plus_after_delta:
    #       scheme 2. loss_safety = mean(NSFWTrend(pred_x0_t))
    #                     + w(step) * mean(ReLU(NSFWTrend(after_img) - NSFWTrend(before_t))).
    #       w(step) increases for later denoising steps.
    p.add_argument(
        "--safety_loss_mode",
        type=str,
        default="delta",
        choices=["delta", "absolute_predx0", "absolute_predx0_plus_after_delta"],
    )
    p.add_argument("--lambda_after_delta_min", type=float, default=0.0)
    p.add_argument("--lambda_after_delta_max", type=float, default=0.5)
    p.add_argument("--after_delta_ramp_power", type=float, default=1.0)
    p.add_argument("--clip_loss_weight_ramp", action=argparse.BooleanOptionalAction)

    # Safety variants:
    #   base:
    #       loss_safety = ReLU(NSFWTrend(safety_img) - NSFWTrend(before_t))
    #   after_prev:
    #       loss_safety = base + lambda_after_prev * ReLU(NSFWTrend(after_t_img) - NSFWTrend(before_(t-1)))
    p.add_argument("--safety_variant", type=str, default="base", choices=["base", "after_prev"])
    p.add_argument("--lambda_after_prev", type=float, default=1.0)

    p.add_argument("--trend_mode", type=str, default="relu_square", choices=["relu_square", "softplus_square"])
    p.add_argument("--softplus_beta", type=float, default=10.0)
    p.add_argument("--include_special_adjustment", action=argparse.BooleanOptionalAction)
    p.add_argument("--special_adjustment_value", type=float, default=0.005, help="Special-care adjustment added to concept scores when --include_special_adjustment is enabled.")
    p.add_argument("--fixed_validation_conditions", action=argparse.BooleanOptionalAction, help="Use the same random step choices and latent noise for every validation run, without changing the training RNG state.")
    p.add_argument("--validation_seed", type=int, default=20260723, help="Seed used only when --fixed_validation_conditions is enabled.")

    # Optional phrase-based NSFW trend.
    # Formula per image:
    #   phrase_raw_j = cosine(CLIP_image(image), CLIP_text(phrase_j)) - phrase_trend_threshold
    #   phrase_trend = sum_j ReLU(phrase_raw_j)^2
    # Total trend used by safety loss becomes:
    #   NSFWTrend_total = NSFWTrend_safety_checker + lambda_phrase_trend * phrase_trend
    p.add_argument("--lambda_phrase_trend", type=float, default=0.0)
    p.add_argument("--phrase_trend_threshold", type=float, default=0.18)
    p.add_argument(
        "--phrase_trend_phrases",
        type=str,
        default="penis,lewd,dick,genitalia,porn,orgasm,sexual,obscene",
        help="Comma-separated CLIP text phrases used as extra NSFW trend embeddings.",
    )

    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction)
    p.add_argument("--enable_xformers", action=argparse.BooleanOptionalAction)
    p.add_argument("--enable_tf32", action=argparse.BooleanOptionalAction)
    p.add_argument("--channels_last", action=argparse.BooleanOptionalAction)
    p.add_argument("--enable_vae_slicing", action=argparse.BooleanOptionalAction)
    p.add_argument("--vae_decode_chunk_size", type=int, default=1)
    p.add_argument("--safety_checker_chunk_size", type=int, default=1)
    p.add_argument("--empty_cache_every_steps", type=int, default=200)
    p.add_argument("--project_root", type=Path, default=Path(__file__).resolve().parent.parent,
                   help="Root containing data/, models/, and output/. Relative paths are resolved here.")
    p.add_argument("--check_only", action="store_true", help="Check local inputs without loading models or training dependencies.")
    p.add_argument("--dry_run", action="store_true", help="Print resolved configuration without checking inputs or starting training.")
    # Match the original 6000-step, effective-batch-32 launch configuration.
    p.set_defaults(
        csv_path="data/I2P_sexual_931.csv",
        base_model_id="models/stable-diffusion-v1-4",
        safety_checker_id="models/stable-diffusion-safety-checker",
        after_delta_safety_checker_id="models/clip_noise_finetuned_safety_checker/best",
        clip_model_id="models/clip-vit-large-patch14",
        output_dir="output/safety_lora",
        train_batch_size=2, gradient_accumulation_steps=16, max_train_steps=6000,
        val_max_prompts=186, save_every_steps=300, lora_rank=16, lora_alpha=16,
        step_min_ratio=0.40, clip_start_ratio=0.70, lambda_after_prev=0.0,
        safety_loss_mode="absolute_predx0_plus_after_delta",
        lambda_safety=3.0, lambda_clip=0.06, lambda_tv=0.0008,
        lambda_var=0.035, lambda_noise_delta=0.008, min_image_var=0.015,
        lambda_phrase_trend=1.0, include_special_adjustment=True,
        fixed_validation_conditions=True, clip_loss_weight_ramp=True,
        gradient_checkpointing=True, enable_tf32=True, enable_vae_slicing=True,
        enable_xformers=False, channels_last=False,
        local_files_only=True, clip_local_files_only=True, use_tensorboard=True,
    )
    args = p.parse_args(argv)
    try:
        return resolve_and_validate_args(args)
    except ValueError as exc:
        p.error(str(exc))


def resolve_path(value, root):
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else root / path).resolve())


def resolve_model_source(value, root):
    """Keep Hub IDs intact while anchoring local model paths at the project root.

    Use ./ for a custom local directory that does not exist yet, or hf:// to
    explicitly select a Hub ID that would otherwise look like a local path.
    """
    if not value:
        return ""
    if value.startswith("hf://"):
        return value[5:]
    path = Path(value).expanduser()
    if (path.is_absolute() or value.startswith((".", "~", "models/", "models\\"))
            or (root / path).exists()):
        return resolve_path(value, root)
    return value


def resolve_and_validate_args(args):
    args.project_root = str(args.project_root.expanduser().resolve())
    root = Path(args.project_root)
    for name in ("csv_path", "output_dir", "tensorboard_log_dir"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, resolve_path(value, root))
    for name in ("base_model_id", "safety_checker_id", "after_delta_safety_checker_id", "clip_model_id"):
        setattr(args, name, resolve_model_source(getattr(args, name), root))

    for name, value in vars(args).items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"--{name} must be finite")
    positive = ("resolution", "train_batch_size", "val_batch_size", "max_train_steps",
                "gradient_accumulation_steps", "learning_rate", "lora_rank", "lora_alpha",
                "num_inference_steps", "log_every_steps", "vae_decode_chunk_size",
                "safety_checker_chunk_size", "softplus_beta", "after_delta_ramp_power",
                "adam_epsilon", "max_grad_norm")
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive")
    for name in ("num_workers", "val_max_prompts", "save_every_steps", "val_every_steps",
                 "empty_cache_every_steps", "lambda_safety", "lambda_clip", "lambda_tv",
                 "lambda_var", "lambda_noise_delta", "lambda_phrase_trend", "lambda_after_prev",
                 "lambda_after_delta_min", "lambda_after_delta_max", "min_image_var"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name} must be nonnegative")
    if args.max_prompts is not None and args.max_prompts < 10:
        raise ValueError("--max_prompts must be at least 10 for the train/validation/test split")
    if args.resolution % 8:
        raise ValueError("--resolution must be divisible by 8")
    ratios = (args.split_train_ratio, args.split_val_ratio, args.split_test_ratio)
    if any(value <= 0 for value in ratios):
        raise ValueError("All three split ratios must be positive (they are normalized automatically)")
    if not 0 <= args.step_min_ratio <= args.step_max_ratio <= 1:
        raise ValueError("Require 0 <= step_min_ratio <= step_max_ratio <= 1")
    lo = max(2 if args.safety_variant == "after_prev" else 1,
             math.ceil(args.num_inference_steps * args.step_min_ratio))
    hi = min(args.num_inference_steps, math.floor(args.num_inference_steps * args.step_max_ratio))
    if lo > hi:
        raise ValueError("The step ratio range contains no usable denoising step")
    if not 0 <= args.clip_start_ratio <= 1:
        raise ValueError("--clip_start_ratio must lie in [0, 1]")
    if args.lambda_after_delta_min > args.lambda_after_delta_max:
        raise ValueError("lambda_after_delta_min must not exceed lambda_after_delta_max")
    if not 0 <= args.lora_dropout < 1:
        raise ValueError("--lora_dropout must lie in [0, 1)")
    if not all(0 <= value < 1 for value in (args.adam_beta1, args.adam_beta2)):
        raise ValueError("Adam beta values must lie in [0, 1)")
    if not any(part.strip() for part in args.target_modules.split(",")):
        raise ValueError("--target_modules must contain at least one module")
    if args.lambda_phrase_trend > 0 and not any(part.strip() for part in args.phrase_trend_phrases.split(",")):
        raise ValueError("--phrase_trend_phrases must be nonempty when lambda_phrase_trend > 0")
    if args.check_only and args.dry_run:
        raise ValueError("Choose either --check_only or --dry_run")
    return args


def check_inputs(args):
    """Inspect local assets without importing torch or modifying any files.

    Cached/remote Hub IDs are reported as deferred checks, since validating them
    requires the training libraries and possibly network access.
    """
    errors, messages = [], []

    def require_file(path):
        if not path.is_file():
            errors.append(f"Missing file: {path}")

    def require_weights(folder, prefix):
        safe_prefix = "model" if prefix == "pytorch_model" else prefix
        candidates = [folder / name for name in
                      (safe_prefix + ".safetensors", prefix + ".bin",
                       safe_prefix + ".safetensors.index.json", prefix + ".bin.index.json")]
        if not any(path.is_file() for path in candidates):
            errors.append(f"Missing {prefix} weights (.safetensors or .bin): {folder}")

    csv_path = Path(args.csv_path)
    require_file(csv_path)
    if csv_path.is_file():
        try:
            with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if args.prompt_column not in (reader.fieldnames or []):
                    errors.append(f"CSV column {args.prompt_column!r} is missing")
                else:
                    count = sum(1 for row in reader if (row.get(args.prompt_column) or "").strip()
                                and (row.get(args.prompt_column) or "").strip().lower() != "nan")
                    if args.max_prompts is not None:
                        count = min(count, args.max_prompts)
                    if count < 10:
                        errors.append(f"At least 10 nonempty prompts are required; found {count}")
                    if count >= 3:
                        train_size, _, _ = split_sizes(count, args.split_train_ratio, args.split_val_ratio, args.split_test_ratio)
                        if args.train_batch_size > train_size:
                            errors.append(f"train_batch_size={args.train_batch_size} exceeds the training split ({train_size})")
                    messages.append(f"Prompt CSV: {count} usable prompts")
        except (OSError, UnicodeError, csv.Error) as exc:
            errors.append(f"Cannot read CSV: {exc}")

    for name in ("base_model_id", "safety_checker_id", "after_delta_safety_checker_id", "clip_model_id"):
        value = getattr(args, name)
        if not value:
            continue
        folder = Path(value)
        if not folder.is_absolute():
            messages.append(f"{name}: {value} (Hub/cache validation deferred until model loading)")
            continue
        if not folder.is_dir():
            errors.append(f"Missing model directory for {name}: {folder}")
            continue
        if name == "base_model_id":
            for subfolder in ("text_encoder", "unet", "vae"):
                require_file(folder / subfolder / "config.json")
                require_weights(folder / subfolder, "pytorch_model" if subfolder == "text_encoder" else "diffusion_pytorch_model")
            require_file(folder / "scheduler" / "scheduler_config.json")
            tokenizer = folder / "tokenizer"
        else:
            require_file(folder / "config.json")
            require_weights(folder, "pytorch_model")
            tokenizer = folder if name == "clip_model_id" else None
        if tokenizer is not None:
            for filename in ("tokenizer_config.json", "vocab.json", "merges.txt"):
                require_file(tokenizer / filename)
        messages.append(f"{name}: {folder}")

    output = Path(args.output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        errors.append(f"Output already exists and is not an empty directory: {output}. Choose another --output_dir.")
    if args.device == "cpu" and args.mixed_precision != "no":
        errors.append("CPU training requires --mixed_precision no")
    return messages, errors


def format_config(args):
    return json.dumps(vars(args), indent=2, ensure_ascii=False)


def split_sizes(count, train_ratio, val_ratio, test_ratio):
    """Use the original rounded split, reserving one example per held-out set."""
    if count < 3:
        raise ValueError("At least three prompts are needed for three nonempty splits")
    total = train_ratio + val_ratio + test_ratio
    train_n = min(max(1, round(count * train_ratio / total)), count - 2)
    val_n = min(max(1, round(count * val_ratio / total)), count - train_n - 1)
    return train_n, val_n, count - train_n - val_n

