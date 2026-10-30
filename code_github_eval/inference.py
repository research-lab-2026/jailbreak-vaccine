"""Local Stable Diffusion + LoRA inference, adapted from the supplied evaluation code."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from inference_config import SCHEDULER_NAMES


def component_load_kwargs(model_path):
    # A pipeline-wide variant must exist for every weight-bearing component.
    base = Path(model_path)
    required = ("text_encoder/model.fp16.safetensors",
                "unet/diffusion_pytorch_model.fp16.safetensors",
                "vae/diffusion_pytorch_model.fp16.safetensors")
    return {"variant": "fp16", "use_safetensors": True} if all((base / name).is_file() for name in required) else {}


def load_scheduler(model_path):
    import diffusers

    folder = Path(model_path) / "scheduler"
    config = folder / "scheduler_config.json"
    if not config.is_file():
        config = folder / "config.json"
    settings = json.loads(config.read_text(encoding="utf-8"))
    class_name = settings.get("_class_name", "PNDMScheduler")
    if class_name not in SCHEDULER_NAMES:
        raise ValueError(f"Unsupported scheduler class: {class_name}")
    return getattr(diffusers, class_name).from_config(settings)


def load_base_pipe(model_path: str, device: str, dtype, enable_xformers: bool = False):
    import torch
    from transformers import CLIPTextModel, CLIPTokenizer
    from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel

    model_path = str(model_path)
    model_index = Path(model_path) / "model_index.json"
    load_kwargs = component_load_kwargs(model_path) if dtype == torch.float16 else {}

    if model_index.exists():
        print(f"[Info] Found model_index.json, loading full pipeline from {model_path}")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            local_files_only=True,
            **load_kwargs,
        )
    else:
        print(f"[Warn] model_index.json not found in {model_path}")
        print("[Info] Loading StableDiffusionPipeline manually from subfolders.")
        print(f"[Info] component load kwargs: {load_kwargs}")
        tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", local_files_only=True)
        text_encoder = CLIPTextModel.from_pretrained(
            model_path, subfolder="text_encoder", torch_dtype=dtype, local_files_only=True, **load_kwargs
        )
        vae = AutoencoderKL.from_pretrained(
            model_path, subfolder="vae", torch_dtype=dtype, local_files_only=True, **load_kwargs
        )
        unet = UNet2DConditionModel.from_pretrained(
            model_path, subfolder="unet", torch_dtype=dtype, local_files_only=True, **load_kwargs
        )
        scheduler = load_scheduler(model_path)
        pipe = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )

    pipe.safety_checker = None
    pipe.feature_extractor = None
    pipe.requires_safety_checker = False
    pipe.set_progress_bar_config(disable=True)
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if enable_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("[Info] xFormers enabled.")
        except Exception as e:
            print(f"[Warn] Could not enable xFormers: {e}")
    pipe = pipe.to(device)
    return pipe


def load_lora_into_pipe(pipe, lora_weight_path: str):
    weight_path = Path(lora_weight_path)
    if not weight_path.exists():
        raise FileNotFoundError(f"LoRA weight not found: {weight_path}")
    lora_dir = str(weight_path.parent)
    weight_name = weight_path.name
    print(f"[Info] Loading LoRA weights from {weight_path}")
    pipe.load_lora_weights(lora_dir, weight_name=weight_name)
    return pipe


def generate_for_task(pipe, job, args):
    import torch
    from tqdm.auto import tqdm

    job.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = job.output_dir / "generation_config.json"
    metadata_path.write_text(json.dumps(job.metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (job.output_dir / "prompts.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("prompt_idx", "case", "prompt", "seed", "file_name"))
        writer.writeheader()
        for record in job.records:
            writer.writerow(dict(prompt_idx=record.prompt_idx, case=record.case_value, prompt=record.prompt,
                                 seed=record.seed, file_name=record.file_name))

    pending = [record for record in job.records
               if not (args.skip_existing and (job.output_dir / record.file_name).is_file())]
    print(f"[Info] {args.model_alias} | {job.task.name}: total={len(job.records)}, pending={len(pending)}")
    with tqdm(total=len(pending), desc=f"{args.model_alias}:{job.task.name}", dynamic_ncols=True) as progress:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]
            generators = [torch.Generator(device=pipe.device).manual_seed(record.seed) for record in batch]
            with torch.inference_mode():
                result = pipe(prompt=[record.prompt for record in batch], generator=generators,
                              num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale,
                              height=args.height, width=args.width)
            if len(result.images) != len(batch):
                raise RuntimeError("The pipeline returned a different number of images than prompts")
            for record, image in zip(batch, result.images):
                target = job.output_dir / record.file_name
                temporary = target.with_suffix(".tmp")
                try:
                    image.save(temporary, format="PNG")
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
                progress.update(1)


def run_inference(args, jobs):
    import torch

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this PyTorch environment")
    precision = ("fp16" if device.startswith("cuda") else "fp32") if args.precision == "auto" else args.precision
    if device == "cpu" and precision != "fp32":
        raise ValueError("CPU inference requires --precision auto or fp32")
    if precision == "bf16" and device.startswith("cuda"):
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise ValueError("The selected CUDA device does not support bf16")
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    print(f"[Info] device={device}, precision={precision}; seed = seed_base + zero-based CSV row index")
    print("[Info] The final pipeline safety checker is disabled, matching the supplied evaluation protocol.")
    pipe = load_base_pipe(args.base_model_path, device=device, dtype=dtype, enable_xformers=args.enable_xformers)
    try:
        if args.lora_path:
            load_lora_into_pipe(pipe, args.lora_path)
        for job in jobs:
            generate_for_task(pipe, job, args)
    finally:
        try:
            if args.lora_path and hasattr(pipe, "unload_lora_weights"):
                pipe.unload_lora_weights()
        finally:
            del pipe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print("[Done] All requested benchmark tasks finished.")
