"""Portable benchmark configuration and input/output planning (standard library only)."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path


TASK_DEFINITIONS = (
    ("sneakyprompt_meaningful_perturbed", "SneakyPrompt200_meaningful_seeded_123.csv", "perturbed_text"),
    ("sneakyprompt_nonsense_perturbed", "SneakyPrompt200_nonsense_seeded_1234.csv", "perturbed_text"),
    ("nudity_eta_3_k_77", "Nudity_eta_3_K_77.csv", "prompt"),
    ("p4dn_16_prompt", "p4dn_16_prompt.csv", "prompt"),
    ("clean", "clean.csv", "prompt"),
)
PAPER_DATASETS = {
    "sneakyprompt_meaningful_perturbed": "SP-M",
    "sneakyprompt_nonsense_perturbed": "SP-N",
    "nudity_eta_3_k_77": "RAB",
    "p4dn_16_prompt": "P4D",
    "clean": "SixCD",
}
SCHEDULER_NAMES = (
    "DDIMScheduler", "DDPMScheduler", "DEISMultistepScheduler", "DPMSolverMultistepScheduler",
    "EulerAncestralDiscreteScheduler", "EulerDiscreteScheduler", "HeunDiscreteScheduler",
    "KDPM2AncestralDiscreteScheduler", "KDPM2DiscreteScheduler", "LMSDiscreteScheduler",
    "LCMScheduler", "PNDMScheduler", "UniPCMultistepScheduler",
)


@dataclass(frozen=True)
class Task:
    name: str
    csv_path: Path
    prompt_column: str


@dataclass(frozen=True)
class PromptRecord:
    prompt_idx: int
    case_value: str
    prompt: str
    seed: int
    file_name: str


@dataclass
class TaskJob:
    task: Task
    records: list[PromptRecord]
    output_dir: Path
    metadata: dict


def rooted_path(value, root):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def default_lora_path(root):
    # The supplied paper artifact takes precedence over unrelated local retraining.
    packaged = root / "checkpoints/safety_lora/best_val_loss_safety_lora/pytorch_lora_weights.safetensors"
    if packaged.is_file():
        return packaged
    preferred = root / "output/safety_lora/best_val_loss_safety_lora/pytorch_lora_weights.safetensors"
    if preferred.is_file():
        return preferred
    # Support existing local experiments without embedding their directory names.
    candidates = sorted((root / "output").glob("**/best_val_loss_safety_lora/pytorch_lora_weights.safetensors"))
    candidates = [path for path in candidates if path.is_file()]
    if len(candidates) > 1:
        raise ValueError("Multiple best-safety LoRA checkpoints found; select one with --lora_path")
    return candidates[0] if candidates else preferred


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Jailbreak Vaccine: SD1.4 image generation on SP-M, SP-N, P4D, RAB, and SixCD.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--project_root", type=Path, default=Path(__file__).resolve().parent.parent,
                        help="Root containing data/, models/, and output/; relative paths are anchored here.")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--base_model_path", default="models/stable-diffusion-v1-4", help="Local Diffusers SD1.4 or SD2.1 model directory.")
    parser.add_argument("--lora_path", default=None, help="LoRA weight file or directory; otherwise use the training output or a unique existing best-safety checkpoint.")
    parser.add_argument("--no_lora", action="store_true", help="Run the unmodified base model as a baseline.")
    parser.add_argument("--model_alias", default=None, help="One directory name for this model's outputs.")
    parser.add_argument("--output_root", default="output/generated_benchmarks")
    parser.add_argument("--tasks", default="all", help="Comma-separated task names, or all (the five documented tasks).")
    parser.add_argument("--seed_base", type=int, default=12345)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--max_prompts", type=int, default=None, help="Limit rows per task for a smoke test, preserving row indices and seeds.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or an indexed CUDA device.")
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--enable_xformers", action="store_true")
    parser.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check_only", action="store_true", help="Inspect files and run configuration without loading ML libraries.")
    parser.add_argument("--dry_run", action="store_true", help="Print resolved configuration without checking assets or generating images.")
    args = parser.parse_args(argv)
    try:
        root = args.project_root.expanduser().resolve()
        args.project_root = str(root)
        for name in ("data_dir", "base_model_path", "output_root"):
            setattr(args, name, str(rooted_path(getattr(args, name), root)))
        if args.no_lora and args.lora_path:
            raise ValueError("--no_lora and --lora_path cannot be used together")
        if args.no_lora:
            args.lora_path = None
        else:
            path = rooted_path(args.lora_path, root) if args.lora_path else default_lora_path(root)
            if path.is_dir():
                path /= "pytorch_lora_weights.safetensors"
            args.lora_path = str(path)
        args.model_alias = args.model_alias or ("base" if args.no_lora else "safety_lora")
        if (args.model_alias in (".", "..") or
                any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in args.model_alias)):
            raise ValueError("--model_alias may contain only letters, digits, underscores, and hyphens")
        known = [item[0] for item in TASK_DEFINITIONS]
        tasks = known if args.tasks == "all" else [name.strip() for name in args.tasks.split(",") if name.strip()]
        if not tasks or any(name not in known for name in tasks):
            raise ValueError(f"--tasks must select from: {', '.join(known)}")
        args.tasks = list(dict.fromkeys(tasks))
        for name in ("num_inference_steps", "batch_size", "height", "width"):
            if getattr(args, name) <= 0:
                raise ValueError(f"--{name} must be positive")
        if args.height % 8 or args.width % 8:
            raise ValueError("--height and --width must be divisible by 8")
        if args.max_prompts is not None and args.max_prompts <= 0:
            raise ValueError("--max_prompts must be positive")
        if not math.isfinite(args.guidance_scale) or args.guidance_scale < 0:
            raise ValueError("--guidance_scale must be finite and nonnegative")
        if not 0 <= args.seed_base <= 2**63 - 1:
            raise ValueError("--seed_base must lie in [0, 2**63 - 1]")
        if args.device not in ("auto", "cpu", "cuda") and not (args.device.startswith("cuda:") and args.device[5:].isdigit()):
            raise ValueError("--device must be auto, cpu, cuda, or cuda:<index>")
        if args.device == "cpu" and args.precision in ("fp16", "bf16"):
            raise ValueError("CPU inference requires --precision auto or fp32")
        if args.check_only and args.dry_run:
            raise ValueError("Choose either --check_only or --dry_run")
    except ValueError as exc:
        parser.error(str(exc))
    return args


def build_tasks(args):
    definitions = {name: (filename, column) for name, filename, column in TASK_DEFINITIONS}
    return [Task(name, Path(args.data_dir) / definitions[name][0], definitions[name][1]) for name in args.tasks]


def slugify_filename_component(value):
    return "".join(ch for ch in str(value) if ch.isalnum() or ch in ("-", "_")) or "na"


def read_prompt_records(task, seed_base, max_prompts=None):
    with task.csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if task.prompt_column not in (reader.fieldnames or []):
            raise ValueError(f"{task.name}: missing column {task.prompt_column!r}")
        records = []
        for index, row in enumerate(reader):
            if max_prompts is not None and index >= max_prompts:
                break
            prompt = row.get(task.prompt_column) or ""
            if not prompt.strip():
                raise ValueError(f"{task.name}: empty prompt at CSV row {index + 2}")
            case = str(index + 1)
            for name in ("case_number", "case", "id", "ID"):
                value = (row.get(name) or "").strip()
                if value and value.lower() not in ("nan", "null", "none", "na", "n/a"):
                    try:
                        numeric = float(value)
                        case = str(int(numeric)) if numeric.is_integer() else value
                    except ValueError:
                        case = value
                    break
            seed = seed_base + index
            if seed > 2**63 - 1:
                raise ValueError(f"{task.name}: row seed exceeds 2**63 - 1")
            filename = f"{index:04d}_case{slugify_filename_component(case)}_seed{seed}.png"
            records.append(PromptRecord(index, case, prompt, seed, filename))
    if not records:
        raise ValueError(f"{task.name}: no prompts found")
    return records


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(value, root):
    if value is None:
        return None
    path = Path(value)
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        # Avoid recording a machine's user/home directory in a shareable artifact.
        return "external/" + path.name


def prepare_jobs(args):
    """Check selected inputs and output provenance without creating files."""
    errors, jobs = [], []
    base = Path(args.base_model_path)
    for folder, prefix in (("text_encoder", "model"), ("unet", "diffusion_pytorch_model"), ("vae", "diffusion_pytorch_model")):
        if not (base / folder / "config.json").is_file():
            errors.append(f"Missing model config: {base / folder / 'config.json'}")
        bin_prefix = "pytorch_model" if folder == "text_encoder" else prefix
        names = (prefix + ".safetensors", prefix + ".safetensors.index.json", bin_prefix + ".bin", bin_prefix + ".bin.index.json")
        if not any((base / folder / name).is_file() for name in names):
            errors.append(f"Missing model weights in {base / folder}")
    for filename in ("tokenizer_config.json", "vocab.json", "merges.txt"):
        if not (base / "tokenizer" / filename).is_file():
            errors.append(f"Missing tokenizer file: {base / 'tokenizer' / filename}")
    scheduler_file = base / "scheduler/scheduler_config.json"
    if not scheduler_file.is_file():
        scheduler_file = base / "scheduler/config.json"
    try:
        scheduler_config = json.loads(scheduler_file.read_text(encoding="utf-8"))
        scheduler_name = scheduler_config.get("_class_name", "PNDMScheduler")
        if scheduler_name not in SCHEDULER_NAMES:
            errors.append(f"Unsupported scheduler: {scheduler_name}")
    except (OSError, ValueError) as exc:
        errors.append(f"Cannot read scheduler configuration: {exc}")
        scheduler_name = None
    lora_hash = None
    if args.lora_path:
        if not Path(args.lora_path).is_file():
            errors.append(f"Missing LoRA weights: {args.lora_path}. Train an adapter or set --lora_path explicitly.")
        else:
            lora_hash = file_digest(args.lora_path)
    common = dict(
        format_version=2, model_alias=args.model_alias,
        base_model_path=portable_path(args.base_model_path, args.project_root),
        lora_path=portable_path(args.lora_path, args.project_root), lora_sha256=lora_hash,
        seed_base=args.seed_base, num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale, batch_size=args.batch_size,
        height=args.height, width=args.width, scheduler=scheduler_name,
        device=args.device, precision=args.precision, enable_xformers=args.enable_xformers,
        safety_checker_enabled=False,
    )
    output_root = Path(args.output_root)
    if any(path.exists() and not path.is_dir() for path in (output_root, output_root / args.model_alias)):
        errors.append("The output root and model output path must be directories")
    for task in build_tasks(args):
        try:
            records = read_prompt_records(task, args.seed_base, args.max_prompts)
            metadata = dict(common, task=task.name, paper_dataset=PAPER_DATASETS[task.name], csv_path=portable_path(task.csv_path, args.project_root),
                            prompt_column=task.prompt_column, csv_sha256=file_digest(task.csv_path))
            out_dir = output_root / args.model_alias / task.name
            metadata_file = out_dir / "generation_config.json"
            if out_dir.exists():
                if not out_dir.is_dir():
                    raise ValueError(f"Output path is not a directory: {out_dir}")
                if metadata_file.is_file():
                    previous = json.loads(metadata_file.read_text(encoding="utf-8"))
                    if previous != metadata:
                        raise ValueError(f"Existing run configuration differs: {out_dir}. Select another --output_root or --model_alias.")
                elif any(out_dir.iterdir()):
                    raise ValueError(f"Existing output has no generation_config.json: {out_dir}. Select another output directory.")
            jobs.append(TaskJob(task, records, out_dir, metadata))
        except (OSError, ValueError, csv.Error) as exc:
            errors.append(str(exc))
    return jobs, errors


def dry_run_config(args):
    config = vars(args).copy()
    config["task_files"] = [dict(name=task.name, csv_path=str(task.csv_path), prompt_column=task.prompt_column) for task in build_tasks(args)]
    return json.dumps(config, indent=2, ensure_ascii=False)
