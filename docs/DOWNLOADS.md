# Downloading the Trained Artifacts

## Publication status

The public repository has been created at
[research-lab-2026/jailbreak-vaccine](https://github.com/research-lab-2026/jailbreak-vaccine).
**Model asset publication is pending: there is no live public LoRA download
yet.** The ZIP files are prepared locally and have not been uploaded as GitHub
Release assets. The assets must be published before a reviewer can download
these trained weights over the internet. Repository creation alone does not
publish the model ZIPs.

## Best-safety LoRA only

The independent asset is `jailbreak_vaccine_sd14_best_safety.zip` (about 12.8 MB).
It contains only the selected adapter plus its README, selection record, and
anonymized training configuration. It does not include final/intermediate or
other best-metric LoRA weights.

Extract it at the repository root, or install from a local path:

```bash
python scripts/download_artifacts.py --artifact lora --archive /path/to/jailbreak_vaccine_sd14_best_safety.zip
```

After it is published on a public GitHub Release:

```bash
python scripts/download_artifacts.py --artifact lora --repo OWNER/REPO --release TAG
```

For the latest Release, omit `--release TAG`. Add `--dry_run` to inspect the
constructed URL before downloading. `OWNER/REPO` and `TAG` must be replaced by
the actual published repository/tag; the script does not create or publish a
repository or Release.

For another HTTPS host, pass `--url` pointing to the exact published ZIP. The
archive must match the recorded SHA-256; the script also verifies every file
before installation. An installation already containing identical files is
accepted, while different existing files are protected.

The adapter installs at:

```text
checkpoints/safety_lora/best_val_loss_safety_lora/pytorch_lora_weights.safetensors
```

Inference uses this supplied checkpoint by default when present. Select it
explicitly with `--lora_path checkpoints/safety_lora/best_val_loss_safety_lora`.

## Vision Calibrator

The checker is a separate asset so inference users need not download it:

```bash
python scripts/download_artifacts.py --artifact vision_calibrator --archive /path/to/jailbreak_vaccine_vision_calibrator.zip
python scripts/download_artifacts.py --artifact vision_calibrator --repo OWNER/REPO --release TAG
```

It installs under `models/clip_noise_finetuned_safety_checker/best/` and is
needed for LoRA training, not ordinary LoRA inference.

## Public models

The three public model weights are not included. Their official sources,
pinned revisions, and downloading commands are in [models/README.md](../models/README.md).
The `download_models.py` script fetches those upstream dependencies;
`download_artifacts.py` installs the experiment-specific trained artifacts.
