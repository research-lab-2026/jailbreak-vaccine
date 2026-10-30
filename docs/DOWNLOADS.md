# Downloading the Trained Artifacts

## Published artifacts

Both trained artifacts are public in
[Release v1.0.0](https://github.com/research-lab-2026/jailbreak-vaccine/releases/tag/v1.0.0) of
[research-lab-2026/jailbreak-vaccine](https://github.com/research-lab-2026/jailbreak-vaccine).

| Artifact | Download | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| Selected best-safety SD1.4 LoRA | [LoRA ZIP](https://github.com/research-lab-2026/jailbreak-vaccine/releases/download/v1.0.0/jailbreak_vaccine_sd14_best_safety.zip) | 12,799,189 | `5cf96dc34c35fb3bc5a9baaa4627329319d94537e0710a6591d54501f71738e2` |
| Vision Calibrator | [Vision Calibrator ZIP](https://github.com/research-lab-2026/jailbreak-vaccine/releases/download/v1.0.0/jailbreak_vaccine_vision_calibrator.zip) | 1,215,983,626 | `d3a5b069ea09a4686a7f40332736d377afc00e5faffa70f94a5c291c52354432` |

The [refreshed code/data ZIP](https://github.com/research-lab-2026/jailbreak-vaccine/releases/download/v1.0.0/anonymous_code_data_published.zip) contains the code,
documentation, and six CSV inputs with the current publication instructions.
The original `anonymous_code_data.zip` remains as an earlier review snapshot.
GitHub's automatic `v1.0.0` source archives retain the initial tagged source;
use `main` or the refreshed bundle for the updated download documentation.

## Best-safety LoRA only

The LoRA asset contains only `best_val_loss_safety_lora` plus its README,
selection record, and anonymized training configuration. No final/intermediate
or other best-metric LoRA weights are distributed.

From the repository root, download and install the fixed Release:

```bash
python scripts/download_artifacts.py --artifact lora --repo research-lab-2026/jailbreak-vaccine --release v1.0.0
```

Add `--dry_run` to inspect the URL and expected checksums without downloading.
The installer verifies the archive and every installed file against
`artifacts.json`. Identical existing files are accepted; different files are
protected.

The adapter installs at:

```text
checkpoints/safety_lora/best_val_loss_safety_lora/pytorch_lora_weights.safetensors
```

Inference selects this supplied checkpoint by default when present. Use
`--lora_path checkpoints/safety_lora/best_val_loss_safety_lora` to select it
explicitly. The Vision Calibrator is not required for ordinary LoRA inference.

## Vision Calibrator

Download and install the trained checker needed for LoRA training:

```bash
python scripts/download_artifacts.py --artifact vision_calibrator --repo research-lab-2026/jailbreak-vaccine --release v1.0.0
```

Its four files install under `models/clip_noise_finetuned_safety_checker/best/`.
The training entry point uses these weights together with the public Image
Checker. The Vision Calibrator's own finetuning procedure is not included;
see [paper alignment](PAPER_ALIGNMENT.md) and
[checkpoint provenance](CHECKPOINT_PROVENANCE.md) for reproduction scope.

## Install an already-downloaded ZIP

Both ZIPs use paths relative to the repository root. Install a local copy with
the same integrity checks:

```bash
python scripts/download_artifacts.py --artifact lora --archive /path/to/jailbreak_vaccine_sd14_best_safety.zip
python scripts/download_artifacts.py --artifact vision_calibrator --archive /path/to/jailbreak_vaccine_vision_calibrator.zip
```

After both artifacts are installed, verify all supplied files:

```bash
python scripts/verify_release.py
```

## Public models

The three public model weights are not included. Their official sources,
pinned revisions, and download commands are in [models/README.md](../models/README.md).
`download_models.py` fetches those upstream dependencies;
`download_artifacts.py` installs the experiment-specific trained artifacts.
