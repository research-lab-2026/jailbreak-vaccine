#!/usr/bin/env python3
"""Verify SHA-256 integrity of the supplied code, datasets, and trained weights."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--code_only", action="store_true", help="Verify the code/data archive before trained weights are extracted.")
    args = parser.parse_args(argv)
    root = args.project_root.expanduser().resolve()
    checked = failed = 0
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if args.code_only and relative.startswith(("checkpoints/", "models/clip_noise_finetuned_safety_checker/")):
            continue
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Manifest path escapes the project root: {relative}")
        if not path.is_file() or sha256(path) != expected:
            print(f"[FAIL] {relative}")
            failed += 1
        checked += 1
    print(f"Checked {checked} files; failures: {failed}.")
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
