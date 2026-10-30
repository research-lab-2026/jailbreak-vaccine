# Paper Evaluation Protocol

The Experiment Setup on manuscript page 5 uses matched row seeds
`s_i = 12345 + i`, with the final output safety checker disabled. It defines
DINO similarity as paired cosine similarity between normalized DINOv2
embeddings of original-backbone and defended images generated with the same
prompt and seed. ASR is reported on harmful tasks, FID/DINO measure image
fidelity, and SixCD additionally uses prompt-image CLIP similarity.

## Dataset/task correspondence

| Paper label | Task | Rows | Consumed column |
| --- | --- | --- | --- |
| SP-M | `sneakyprompt_meaningful_perturbed` | 200 | `perturbed_text` |
| SP-N | `sneakyprompt_nonsense_perturbed` | 200 | `perturbed_text` |
| P4D | `p4dn_16_prompt` | 151 | `prompt` |
| RAB | `nudity_eta_3_k_77` | 95 | `prompt` |
| SixCD | `clean` | 100 | `prompt` |

All CSVs preserve the supplied row order and prompt text. Numbers in filenames
are identifiers and do not override the actual CSV row count.

## Generate paired images

After downloading the public SD1.4 dependency and installing the selected LoRA:

```bash
python code_github_eval/infer.py --no_lora --model_alias sd14_base
python code_github_eval/infer.py --model_alias jailbreak_vaccine_sd14
```

Both commands use 50 denoising steps, CFG 7.5, batch size 4, 512 x 512 output,
and the same row seed rule. The local/pinned SD1.4 model's inference scheduler
is PNDM, matching the supplied evaluation script; the training trajectory uses
DDIM. The training safety-start ratio does not delay activation of the LoRA at
inference: the adapter is active for every sampling step.

Pair files by task and exact image filename under:

```text
output/generated_benchmarks/sd14_base/<task>/
output/generated_benchmarks/jailbreak_vaccine_sd14/<task>/
```

`prompts.csv` records row index, case ID, prompt, seed, and output filename.
`generation_config.json` records the paper dataset label, model/CSV identity,
generation settings, and CSV/LoRA checksums. Skipping existing images does not
renumber rows or change their assigned seeds. Changing batching or a model's
settings requires a new output alias/directory.

For a short generation smoke test, use another output root and at most two
prompts per task:

```bash
python code_github_eval/infer.py --max_prompts 2 --batch_size 1 --output_root output/smoke_test
```

## Paper-reported reference values

The following are transcribed from Table 1 (SD1.4, Ours). They are **reference
values reported by the manuscript**, not results recomputed in this package:

| Dataset | ASR (%) down | DINO up | FID down | CLIP up |
| --- | --- | --- | --- | --- |
| SP-M | 1.00 | 0.534 | 0.567 | - |
| SP-N | 2.50 | 0.564 | 0.544 | - |
| P4D | 3.31 | 0.603 | 0.537 | - |
| RAB | 13.68 | 0.625 | 0.547 | - |
| SixCD | - | 0.690 | 0.449 | 0.252 |

The numerical FID values above are copied in the manuscript's reported form;
no unstated scaling or normalization is introduced. Machine-readable reference
rows for Table 1 and the SD1.4 baseline in Table 2 are supplied in
`results/paper_reported_sd14.csv`.

This package currently implements image generation, not ASR/DINO/FID/CLIP
metric pipelines. The manuscript does not fully identify every detector
version, threshold, embedding checkpoint, or FID implementation detail needed
to reconstruct the quantitative tables. No choices are invented here. The
stored validation safety loss is not interchangeable with ASR. Actual GPU
generation and the quantitative table values have not been revalidated here.
