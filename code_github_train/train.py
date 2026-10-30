#!/usr/bin/env python3
"""Run safety LoRA training, or check the configuration without GPU dependencies."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from training_config import check_inputs, format_config, parse_args


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        print(format_config(args))
        return 0

    messages, errors = check_inputs(args)
    for message in messages:
        print(f"[Check] {message}")
    if errors:
        for error in errors:
            print(f"[Error] {error}", file=sys.stderr)
        return 1
    if args.check_only:
        print("[OK] Input checks passed. Model loading, dependency compatibility, and GPU execution were not tested.")
        return 0

    # Keep caches portable; respect an explicitly configured cache location.
    os.environ.setdefault("HF_HOME", str(Path(args.project_root) / ".cache" / "huggingface"))
    try:
        from training import train
        train(args)
    except ModuleNotFoundError as exc:
        print(f"[Error] Missing dependency {exc.name!r}. Install requirements.txt with this Python interpreter.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
