# Supplied Prompt CSVs

These are byte-for-byte copies of the six prompt CSVs supplied with this
experiment. Row order, text, identifiers, and existing metadata are preserved.
They contain prompt text and metadata, not generated image files. Some prompts
contain explicit or adversarial language used for safety research.

| Role/task | File | Rows | Consumed prompt column |
| --- | --- | --- | --- |
| LoRA training | `I2P_sexual_931.csv` | 931 | `perturbed_text` |
| `sneakyprompt_meaningful_perturbed` | `SneakyPrompt200_meaningful_seeded_123.csv` | 200 | `perturbed_text` |
| `sneakyprompt_nonsense_perturbed` | `SneakyPrompt200_nonsense_seeded_1234.csv` | 200 | `perturbed_text` |
| `nudity_eta_3_k_77` | `Nudity_eta_3_K_77.csv` | 95 | `prompt` |
| `p4dn_16_prompt` | `p4dn_16_prompt.csv` | 151 | `prompt` |
| `clean` | `clean.csv` | 100 | `prompt` |

Training uses a seeded 7:2:1 split of the 931 training rows: 652 training,
186 validation, and 93 test prompts with split seed 42. The training entry point
exports its splits when run; duplicate historical split CSVs are omitted from
the selected-checkpoint download.

Paper labels: the meaningful and nonsense SneakyPrompt tasks are SP-M and SP-N;
`p4dn_16_prompt` is P4D; `nudity_eta_3_k_77` is RAB; and `clean` is SixCD.

Inference consumes 746 rows across the five tasks. The selected case identifier
is `case_number`, `case`, `id`, or `ID`, falling back to the one-based row index.
Other columns are retained for provenance but are not passed to the model.

In particular, `random_seed`, `evaluation_seed`, and `evaluation_guidance`
columns are not used by the supplied inference entry point. Its defaults are
`seed = 12345 + zero-based row index` and guidance scale 7.5, matching the
original generation script. Task-name numbers do not specify the CSV row count.

The filenames identify the experiment inputs. This package does not claim they
are unmodified upstream dataset releases: the perturbation/preparation code
that produced these CSVs is not supplied. Their hashes are in `../SHA256SUMS`.
Upstream dataset terms and attribution remain applicable to the underlying
materials; no new blanket dataset license is assigned by this package.
