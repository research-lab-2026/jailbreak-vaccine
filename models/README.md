# Model Dependencies and Supplied Checker

The experiment-specific finetuned safety checker implements the paper's
**Vision Calibrator**. Its independent asset is
`jailbreak_vaccine_vision_calibrator.zip`, installing under
`clip_noise_finetuned_safety_checker/best/`:

- `model.safetensors` (the model weights; 1,215,979,664 bytes).
- `config.json` (the `StableDiffusionSafetyChecker` architecture/configuration).
- `preprocessor_config.json` (CLIP image preprocessing configuration).
- `metrics.json` (metrics recorded alongside the supplied checkpoint).

This checker is used for the after-latent safety-delta training loss. Its own
finetuning procedure is not included. The stored configuration reports
Transformers 5.8.1 as its serialization version; the preparation environment
could not validate model loading with the candidate requirements.

The following public model weights are **not distributed** in either archive.
Download them directly from the respective upstream repositories:

| Local directory | Official source | Pinned revision |
| --- | --- | --- |
| `stable-diffusion-v1-4/` | [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4) | `133a221b8aa7292a167afc5127cb63fb5005638b` |
| `stable-diffusion-safety-checker/` | [CompVis/stable-diffusion-safety-checker](https://huggingface.co/CompVis/stable-diffusion-safety-checker) | `cb41f3a270d63d454d385fc2e4f571c487c253c5` |
| `clip-vit-large-patch14/` | [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) | `32bd64288804d66eefd0ccbe215aa642df71cc41` |

The revisions were recovered from the local Hugging Face download metadata
accompanying the supplied experiment models. File lists and revisions are
recorded in `public_models.json`; local cache directories are not distributed.

From the release root, install dependencies and run:

```bash
python -m pip install -r requirements.txt
python scripts/download_models.py --dry_run
python scripts/download_models.py
```

For inference alone, only the SD1.4 base model needs to be downloaded:

```bash
python scripts/download_models.py --models stable-diffusion-v1-4
```

The script uses the upstream
[`snapshot_download` API](https://huggingface.co/docs/huggingface_hub/guides/download),
pins the recorded revision, and selects the files needed by these entry points.
It avoids downloading additional frameworks' checkpoints and the SD model's
embedded safety checker. SD1.4 standard and fp16 safetensors are both selected
to support the original training and inference loading paths. The standalone
normal safety checker uses its upstream `pytorch_model.bin`; CLIP uses
`model.safetensors` plus tokenizer and preprocessing files.

If upstream access requires authentication or acceptance of terms, complete
that through Hugging Face before running the downloader. Do not include access
tokens or authentication caches in the repository. Upstream model cards and
license/notice files retain their original terms; the downloader requests
available license/notice files as well. No alternative license is assigned to
third-party models or their derivatives here.
