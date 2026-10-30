#!/usr/bin/env python3
"""Download/install the selected Jailbreak Vaccine LoRA or Vision Calibrator artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import urllib.parse
import urllib.request
import zipfile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_asset_url(repository, release, filename):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Repository must have the form OWNER/REPOSITORY")
    base = f"https://github.com/{repository}/releases/"
    suffix = "latest/download/" if release == "latest" else "download/" + urllib.parse.quote(release, safe="") + "/"
    return base + suffix + urllib.parse.quote(filename, safe="")


def install_archive(archive, artifact, root):
    """Verify all bytes and destinations before installing any artifact file."""
    root = Path(root).resolve()
    if sha256(archive) != artifact["sha256"]:
        raise ValueError("Archive SHA-256 mismatch; no artifact files were installed")
    expected = artifact["files"]
    with tempfile.TemporaryDirectory(prefix=".artifact_extract_", dir=root) as temp:
        staging = Path(temp)
        with zipfile.ZipFile(archive) as packed:
            names = packed.namelist()
            if len(names) != len(set(names)) or set(names) != set(expected):
                raise ValueError("Archive contents differ from the artifact manifest")
            for name in names:
                target = (root / name).resolve()
                if ("\\" in name or name.startswith("/") or ".." in Path(name).parts
                        or not target.is_relative_to(root)):
                    raise ValueError("Archive contains an unsafe path")
                staged = staging / name
                staged.parent.mkdir(parents=True, exist_ok=True)
                with packed.open(name) as source, staged.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                if sha256(staged) != expected[name]:
                    raise ValueError(f"Artifact file checksum mismatch: {name}")
                if target.exists() and (not target.is_file() or sha256(target) != expected[name]):
                    raise ValueError(f"Existing file differs; choose another --project_root: {name}")
            for name in names:
                target = root / name
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(staging / name, target)
    return len(expected)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifact", choices=("lora", "vision_calibrator"), default="lora")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--repo", help="Public GitHub repository OWNER/REPO after the assets have been published.")
    source.add_argument("--url", help="Published HTTPS URL for this exact archive.")
    source.add_argument("--archive", type=Path, help="Install the local archive without network access.")
    parser.add_argument("--release", default="latest", help="GitHub Release tag or latest.")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)
    root = args.project_root.expanduser().resolve()
    manifest = json.loads((root / "artifacts.json").read_text(encoding="utf-8"))
    artifact = manifest["artifacts"][args.artifact]
    publication = manifest["publication"]
    repository = args.repo or publication.get("repository")
    try:
        url = args.url or (release_asset_url(repository, args.release, artifact["filename"]) if repository else None)
        if not args.archive and not url:
            raise ValueError("No hosted download is configured yet. After publishing the Release asset, pass --repo OWNER/REPO; alternatively use --archive PATH.")
        if url and urllib.parse.urlparse(url).scheme != "https":
            raise ValueError("Artifact URLs must use HTTPS")
        if args.dry_run:
            print(json.dumps(dict(artifact=args.artifact, url=url, archive=str(args.archive) if args.archive else None,
                                  sha256=artifact["sha256"], installed_files=list(artifact["files"])), indent=2))
            return 0
        if args.archive:
            count = install_archive(args.archive.expanduser().resolve(), artifact, root)
        else:
            with tempfile.TemporaryDirectory(prefix=".artifact_download_", dir=root) as temp:
                local = Path(temp) / artifact["filename"]
                print(f"Downloading {url}")
                request = urllib.request.Request(url, headers={"User-Agent": "Jailbreak-Vaccine-Artifact"})
                with urllib.request.urlopen(request, timeout=60) as response, local.open("wb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
                count = install_archive(local, artifact, root)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"[Error] {exc}\n")
    print(f"[OK] Verified and installed {count} files for {args.artifact}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
