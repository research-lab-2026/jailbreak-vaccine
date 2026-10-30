# Downloading the Trained Artifacts

## Publication status

The public repository is
[research-lab-2026/jailbreak-vaccine](https://github.com/research-lab-2026/jailbreak-vaccine).
The selected **best-safety SD1.4 LoRA is published in Release v1.0.0**:

[Download jailbreak_vaccine_sd14_best_safety.zip](https://github.com/research-lab-2026/jailbreak-vaccine/releases/download/v1.0.0/jailbreak_vaccine_sd14_best_safety.zip)

The **Vision Calibrator online upload is still pending**. Its trained archive
is prepared locally, but it is not currently available as a published Release
asset. Do not treat its local filename or an inferred GitHub URL as a working
online download.

## Best-safety LoRA only

The published asset is `jailbreak_vaccine_sd14_best_safety.zip` (about 12.8 MB).
It contains only the selected adapter plus its README, selection record, and
anonymized training configuration. No final/intermediate or other best-metric
LoRA weights are distributed.

Download and install the fixed, verified Release:

```bash
python scripts/download_artifacts.py --artifact lora --repo research-lab-2026/jailbreak-vaccine --release v1.0.0
```

Add `--dry_run` to inspect the URL and expected checksums without downloading.
The archive and every installed file are verified against `artifacts.json`.
Identical existing files are accepted; different files are protected.

Alternatively, download the ZIP using the link above, then extract it at the
repository root or install the local copy:

```bash
python scripts/download_artifacts.py --artifact lora --archive /path/to/jailbreak_vaccine_sd14_best_safety.zip
```

The adapter installs at:

```text
checkpoints/safety_lora/best_val_loss_safety_lora/pytorch_lora_weights.safetensors
```

Inference selects this supplied checkpoint by default when present. Use
`--lora_path checkpoints/safety_lora/best_val_loss_safety_lora` to select it
explicitly. The Vision Calibrator is not required for ordinary LoRA inference.

## Vision Calibrator: local installation only for now

The checker archive has not been published online. If you already have its
local `jailbreak_vaccine_vision_calibrator.zip`, install it with:

```bash
python scripts/download_artifacts.py --artifact vision_calibrator --archive /path/to/jailbreak_vaccine_vision_calibrator.zip
```

It installs under `models/clip_noise_finetuned_safety_checker/best/` and is
needed for LoRA training. An online command or download link will be added
only after that exact asset has been uploaded and verified.

## Public models

The three public model weights are not included. Their official sources,
pinned revisions, and downloading commands are in [models/README.md](../models/README.md).
The `download_models.py` script fetches those upstream dependencies;
`download_artifacts.py` installs the experiment-specific trained artifacts.
