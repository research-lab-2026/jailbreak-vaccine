# Publishing the Anonymous Review Artifact

## Current publication status

The public repository is
[research-lab-2026/jailbreak-vaccine](https://github.com/research-lab-2026/jailbreak-vaccine).
[Release v1.0.0](https://github.com/research-lab-2026/jailbreak-vaccine/releases/tag/v1.0.0) contains both verified
trained artifacts:

- [Selected best-safety SD1.4 LoRA](https://github.com/research-lab-2026/jailbreak-vaccine/releases/download/v1.0.0/jailbreak_vaccine_sd14_best_safety.zip)
- [Vision Calibrator](https://github.com/research-lab-2026/jailbreak-vaccine/releases/download/v1.0.0/jailbreak_vaccine_vision_calibrator.zip)

`artifacts.json` records `published` for both artifacts and the overall model
publication status. The tensor files and model ZIP bytes are unchanged from
the supplied artifacts; the archive and per-file SHA-256 values are recorded
in `artifacts.json` and `SHA256SUMS`.

## Code snapshots

[The refreshed code/data bundle](https://github.com/research-lab-2026/jailbreak-vaccine/releases/download/v1.0.0/anonymous_code_data_published.zip) contains the code,
six CSV inputs, and publication documentation matching the update on `main`.
The original `anonymous_code_data.zip` is retained in the Release as an earlier
review snapshot. GitHub's automatically generated source ZIP/tar files for
tag `v1.0.0` contain the initial tagged source, before the publication
documentation updates. The existing tag and all existing assets are retained.

The root `.gitignore` excludes weights, generated output, caches, and local
ZIPs. The six prompt CSVs and their documentation are included in ordinary
commits. The large trained checker is distributed as a Release attachment.
The three public model dependencies are downloaded from their upstream sources.
`anonymous_weights.zip` is an optional local installation bundle containing
the same selected LoRA and checker, with no other LoRA tensor file.

## Download and verify

Readers can install both artifacts with archive and per-file verification:

```bash
python scripts/download_artifacts.py --artifact lora --repo research-lab-2026/jailbreak-vaccine --release v1.0.0
python scripts/download_artifacts.py --artifact vision_calibrator --repo research-lab-2026/jailbreak-vaccine --release v1.0.0
python scripts/verify_release.py
```

The installer protects different existing files. Use `--dry_run` to inspect a
download plan, or `--archive PATH` to install an already-downloaded ZIP.
See [downloads](DOWNLOADS.md) for exact asset sizes and hashes.

## Rebuilding

From a complete local artifact directory, rebuild archives and checksums with:

```bash
python scripts/build_release.py --output_dir /path/to/new-build-directory
```

The builder requires the trained files locally and never downloads the public
model dependencies. It retains artifact publication status/download URLs when
the rebuilt archive identity and hashes match the verified published asset.
Changed model archives are marked `not_published` and their download URLs are
cleared until the new bytes have actually been uploaded and verified.

Build into a separate directory to preserve existing local archives. Publish
updated code bundles under a distinct filename, as done with
`anonymous_code_data_published.zip`, rather than replacing an earlier review snapshot. A changed
code/documentation file requires refreshing `SHA256SUMS` before publishing.

See [GitHub's release-link documentation](https://docs.github.com/en/repositories/releasing-projects-on-github/linking-to-releases)
for the distinction between fixed-tag and latest-release asset URLs.

Use the review venue's permitted anonymous hosting/identity settings.
Repository ownership, commit author identities, uploader accounts, and public
profiles can reveal authorship independently of file contents. No previous
Git history or account credentials are included. No source-code license is
selected on behalf of the authors; retain applicable upstream notices.
