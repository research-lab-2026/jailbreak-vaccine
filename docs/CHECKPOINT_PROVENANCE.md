# Selected SD1.4 LoRA Provenance

The release distributes exactly one LoRA weight file:

`checkpoints/safety_lora/best_val_loss_safety_lora/pytorch_lora_weights.safetensors`

It is copied unchanged from the supplied experiment's `best_val_loss_safety_lora`
selection. Other LoRA snapshots, the final adapter, and other selection variants
are omitted from every newly prepared release archive. The source experiment
directory is preserved outside the release.

| Field | Recorded value |
| --- | --- |
| Backbone | Stable Diffusion v1.4 |
| LoRA rank / alpha | 16 / 16 |
| Training safety-start ratio | 0.40 |
| Global checkpoint step | 6000 |
| Continuation-stage local step | 3000 |
| Selection criterion | Lowest `val_loss_safety` |
| Recorded best validation safety loss | 0.00013566681924324277 |
| Adapter bytes | 12,792,952 |

The loss above is a training/validation objective, **not ASR** or a paper-table
metric. The selected record is preserved in `best_val_loss_safety_info.json`.
The anonymized historical configuration is included in `training_config.json`.

The configuration records `max_train_steps=3000`,
`resume_global_step_offset=3000`, and initialization from an earlier adapter.
Eight path fields were replaced with neutral project-relative values; all
non-path values were retained. `external/initial_lora` is a placeholder for the
historical initialization location, not a downloadable artifact supplied here.

The current fresh-run training entry point performs 6000 updates from the base
model and does not import the historical resume fields. It is not an exact
replay of the continuation stage. The historical initialization adapter and
optimizer/RNG state are not supplied, so exact retraining equivalence is not
claimed. The chosen weights can nevertheless be used directly for inference.

This is an adapter checkpoint, not a full diffusion model or optimizer-state
checkpoint. It requires the SD1.4 base model. Use `artifacts.json` for the
archive checksum and `SHA256SUMS` for supplied file checksums.
