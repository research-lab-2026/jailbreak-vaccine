#!/usr/bin/env python3
"""Rebuild the anonymous code/data archive, selected-model assets, and checksums."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pack(output, root, paths, compression):
    with zipfile.ZipFile(output, "w", compression=compression, allowZip64=True) as archive:
        for path in sorted(paths):
            relative = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = compression
            info.external_attr = 0o644 << 16
            with path.open("rb") as source, archive.open(info, "w", force_zip64=path.stat().st_size > 2**30) as target:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    target.write(chunk)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"ZIP integrity verification failed: {output.name}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args(argv)
    root = args.project_root.expanduser().resolve()
    destination = args.output_dir.expanduser().resolve() if args.output_dir else root.parent
    if destination.is_relative_to(root):
        parser.error("--output_dir must be outside the repository to avoid bundling archive outputs")
    destination.mkdir(parents=True, exist_ok=True)
    lora_relative = [
        "checkpoints/safety_lora/best_val_loss_safety_lora/pytorch_lora_weights.safetensors",
        "checkpoints/safety_lora/best_val_loss_safety_lora/README.md",
        "checkpoints/safety_lora/best_val_loss_safety_info.json",
        "checkpoints/safety_lora/training_config.json",
    ]
    checker_relative = ["models/clip_noise_finetuned_safety_checker/best/" + name for name in
                        ("model.safetensors", "config.json", "preprocessor_config.json", "metrics.json")]
    specs = {
        "lora": ("jailbreak_vaccine_sd14_best_safety.zip", lora_relative),
        "vision_calibrator": ("jailbreak_vaccine_vision_calibrator.zip", checker_relative),
    }
    publication = dict(status="not_published", repository=None, release="latest")
    manifest_file = root / "artifacts.json"
    if manifest_file.is_file():
        publication.update(json.loads(manifest_file.read_text(encoding="utf-8")).get("publication", {}))
    manifest = dict(schema_version=1, publication=publication, artifacts={})
    archives = []
    trained_paths = []
    for key, (filename, relative_paths) in specs.items():
        paths = [root / relative for relative in relative_paths]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        output = destination / filename
        pack(output, root, paths, zipfile.ZIP_STORED)
        manifest["artifacts"][key] = dict(filename=filename, bytes=output.stat().st_size,
                                         sha256=sha256(output), files={relative: sha256(root / relative) for relative in relative_paths})
        trained_paths.extend(paths)
        archives.append(output)
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")

    trees = {"code_github_train", "code_github_eval", "data", "docs", "scripts", "tests", "results"}
    root_files = {"README.md", "requirements.txt", "CITATION.cff", ".gitignore", ".gitattributes", "artifacts.json"}
    code_paths = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in {"__pycache__", ".git", ".cache", ".venv", "venv"} for part in relative.parts):
            continue
        if path.suffix in {".pyc", ".zip"} or path.name.startswith(".env"):
            continue
        if ((len(relative.parts) == 1 and path.name in root_files) or relative.parts[0] in trees
                or relative.as_posix() in {"models/README.md", "models/public_models.json"}):
            code_paths.append(path)
    supplied = sorted(set(code_paths + trained_paths))
    sums = root / "SHA256SUMS"
    sums.write_text("".join(sha256(path) + "  " + path.relative_to(root).as_posix() + "\n" for path in supplied),
                    encoding="utf-8", newline="\n")
    code_archive = destination / "anonymous_code_data.zip"
    pack(code_archive, root, code_paths + [sums], zipfile.ZIP_DEFLATED)
    combined = destination / "anonymous_weights.zip"
    pack(combined, root, trained_paths, zipfile.ZIP_STORED)
    archives += [code_archive, combined]
    (destination / "anonymous_archives_SHA256SUMS.txt").write_text(
        "".join(sha256(path) + "  " + path.name + "\n" for path in archives), encoding="utf-8", newline="\n")
    for path in archives:
        print(f"{path.name}: {path.stat().st_size:,} bytes")
    print(f"Verified {len(supplied)} supplied-file checksums; publication status: {publication['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
