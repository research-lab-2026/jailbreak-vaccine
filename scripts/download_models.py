#!/usr/bin/env python3
"""Fetch the three public model dependencies at the recorded upstream revisions."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--models", nargs="+", default=["all"],
                        choices=("all", "stable-diffusion-v1-4", "stable-diffusion-safety-checker", "clip-vit-large-patch14"))
    parser.add_argument("--dry_run", action="store_true", help="Show the download plan without network access or model downloads.")
    args = parser.parse_args(argv)
    root = args.project_root.expanduser().resolve()
    manifest = json.loads((root / "models/public_models.json").read_text(encoding="utf-8"))
    selected = [model for model in manifest if "all" in args.models or model["name"] in args.models]
    if args.dry_run:
        print(json.dumps(selected, indent=2))
        return 0
    os.environ.setdefault("HF_HOME", str(root / ".cache/huggingface"))
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError:
        parser.error("Install huggingface-hub before downloading model dependencies")
    for model in selected:
        destination = root / "models" / model["name"]
        print(f"Downloading {model['repo_id']} @ {model['revision']} -> {destination}")
        snapshot_download(repo_id=model["repo_id"], revision=model["revision"],
                          local_dir=str(destination), allow_patterns=model["allow_patterns"])
        missing = [name for name in model["required_files"] if not (destination / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Required files missing after download: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
