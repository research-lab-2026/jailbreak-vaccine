#!/usr/bin/env python3
"""Run the five requested benchmarks or inspect their inputs without loading models."""
from __future__ import annotations

import os
from pathlib import Path
import sys

from inference_config import PAPER_DATASETS, dry_run_config, parse_args, prepare_jobs


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        print(dry_run_config(args))
        return 0
    jobs, errors = prepare_jobs(args)
    for job in jobs:
        print(f"[Check] {PAPER_DATASETS[job.task.name]} | {job.task.name}: {len(job.records)} prompts, column={job.task.prompt_column}")
    if args.lora_path:
        print(f"[Check] LoRA weights: {args.lora_path}")
    if errors:
        for error in errors:
            print(f"[Error] {error}", file=sys.stderr)
        return 1
    if args.check_only:
        print(f"[OK] Input checks passed: {len(jobs)} tasks, {sum(len(job.records) for job in jobs)} images planned.")
        print("[Info] Model loading, dependency compatibility, and GPU execution were not tested.")
        return 0
    os.environ.setdefault("HF_HOME", str(Path(args.project_root) / ".cache/huggingface"))
    try:
        from inference import run_inference
        run_inference(args, jobs)
    except ModuleNotFoundError as exc:
        print(f"[Error] Missing dependency {exc.name!r}. Install requirements.txt with this Python interpreter.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
