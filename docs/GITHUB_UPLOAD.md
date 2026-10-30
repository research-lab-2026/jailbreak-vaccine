# Publishing the Anonymous Review Artifact

Publish the contents of `anonymous_code_data.zip` as the repository. The root
`.gitignore` excludes model weights, generated output, caches, and local ZIPs.
The six prompt CSVs and their documentation are included in ordinary commits.

Create a GitHub Release and upload these two independent assets using their
exact names:

1. `jailbreak_vaccine_sd14_best_safety.zip` - only the requested best-safety LoRA.
2. `jailbreak_vaccine_vision_calibrator.zip` - the trained Vision Calibrator.

`anonymous_weights.zip` is an optional combined local-installation bundle with
the same selected LoRA and checker. It is not needed if the two independent
assets are uploaded. It contains no other LoRA tensor file.

Readers can extract the assets into their clone or use `scripts/download_artifacts.py`.
The script accepts `--repo OWNER/REPO --release TAG`, checks the recorded archive
and file hashes, and refuses to overwrite different existing files.

The current `artifacts.json` explicitly records that hosting has not yet been
configured. Once a real Release has been published, set its `publication.repository`
to the actual `OWNER/REPO`, change `publication.status` to `published`, and
replace the README hosting-status paragraph with the real download URLs.
Changing this metadata requires refreshing `SHA256SUMS`. Rebuild all archives
and checksums from the complete local artifact directory with:

```bash
python scripts/build_release.py
```

Do not modify a published model archive without updating its recorded archive
checksum. The builder requires the supplied trained files locally and never
downloads the three public model dependencies.

The official asset URL formats are:

```text
https://github.com/OWNER/REPO/releases/latest/download/jailbreak_vaccine_sd14_best_safety.zip
https://github.com/OWNER/REPO/releases/download/TAG/jailbreak_vaccine_sd14_best_safety.zip
```

`OWNER`, `REPO`, and `TAG` are templates, not live download addresses. See
[GitHub's release-link documentation](https://docs.github.com/en/repositories/releasing-projects-on-github/linking-to-releases).
The checker exceeds the ordinary 100 MiB Git file limit; a Release attachment
keeps it outside normal Git history.

Use the review venue's permitted anonymous hosting/identity settings. Repository
ownership, commit author identities, uploader accounts, and public profiles can
reveal authorship independently of file contents. No old Git history or account
credentials are included here. No source-code license is selected on behalf
of the authors; retain applicable upstream notices.
