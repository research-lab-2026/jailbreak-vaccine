# Publishing the Anonymous Review Artifact

## Current publication status

The public repository is
[research-lab-2026/jailbreak-vaccine](https://github.com/research-lab-2026/jailbreak-vaccine).
[Release v1.0.0](https://github.com/research-lab-2026/jailbreak-vaccine/releases/tag/v1.0.0)
is published and contains the verified
[`jailbreak_vaccine_sd14_best_safety.zip`](https://github.com/research-lab-2026/jailbreak-vaccine/releases/download/v1.0.0/jailbreak_vaccine_sd14_best_safety.zip).

The Vision Calibrator upload failed in the browser. Its failed upload slot may
remain in the release editor, but the public Release does not contain that
asset. `artifacts.json` therefore records `partially_published`, with the LoRA
`published` and the checker `pending_manual_upload`. No working online checker
link is claimed.

## Complete the remaining checker upload manually

1. Open the existing [v1.0.0 Release](https://github.com/research-lab-2026/jailbreak-vaccine/releases/tag/v1.0.0)
   and choose **Edit**. Reuse this Release; the verified LoRA does not need to be
   uploaded again.
2. Inspect the attachment list. If a failed placeholder for
   `jailbreak_vaccine_vision_calibrator.zip` is shown, remove only that failed
   entry before retrying. Keep the published LoRA attachment unchanged.
3. Attach the local `jailbreak_vaccine_vision_calibrator.zip` file
   (1,215,983,626 bytes). If the browser reports another upload error, retry the
   manual upload from a suitable browser/network session; do not mark it as
   published merely because a filename appears in the editor.
4. Wait for the attachment to show a completed upload, then save with
   **Update release**. Reopen the public Release page and confirm the checker
   filename and size appear as a downloadable asset.
5. Verify that the actual checker download matches its recorded SHA-256 in
   `artifacts.json`. Only then change its artifact status to `published`, add
   its fixed v1.0.0 download URL, and update the README/download instructions.
   Until then, retain the pending status and local-archive installation method.

The ordinary Git file limit is 100 MiB, so the checker belongs in a Release
attachment rather than a normal source-code commit. The three public base
model dependencies are obtained from their upstream sources, not uploaded here.

## Repository contents and rebuilding

The contents of `anonymous_code_data.zip` form the repository. The root
`.gitignore` excludes weights, generated output, caches, and local ZIPs; the
six prompt CSVs and their documentation are included in ordinary commits.
`anonymous_weights.zip` is an optional local-installation bundle containing the
same selected LoRA and checker, with no other LoRA tensor file.

Readers can use `scripts/download_artifacts.py` for the published LoRA; it
checks archive/file hashes and protects different existing files. The current
fixed command is:

```bash
python scripts/download_artifacts.py --artifact lora --repo research-lab-2026/jailbreak-vaccine --release v1.0.0
```

From the complete local artifact directory, rebuild archives and checksums with:

```bash
python scripts/build_release.py
```

The builder requires the trained files locally and never downloads the public
model dependencies. Its generated artifact records must retain the verified
publication status/URL fields before the final checksum manifest and code ZIP
are republished. Do not modify a published model archive without updating its
recorded checksum and verifying the replacement asset.

The LoRA's fixed download URL is:

```text
https://github.com/research-lab-2026/jailbreak-vaccine/releases/download/v1.0.0/jailbreak_vaccine_sd14_best_safety.zip
```

See [GitHub's release-link documentation](https://docs.github.com/en/repositories/releasing-projects-on-github/linking-to-releases)
for the distinction between fixed-tag and latest-release asset URLs.

Use the review venue's permitted anonymous hosting/identity settings. Repository
ownership, commit author identities, uploader accounts, and public profiles can
reveal authorship independently of file contents. No previous Git history or
account credentials are included. No source-code license is selected on behalf
of the authors; retain applicable upstream notices.
