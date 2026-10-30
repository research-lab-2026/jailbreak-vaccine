"""Check task planning and generation control flow without loading diffusion models."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from inference_config import TASK_DEFINITIONS, build_tasks, parse_args, prepare_jobs, read_prompt_records
from inference import component_load_kwargs, generate_for_task


class InferenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def file(self, name, contents=""):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
        return path

    def args(self, *flags):
        return parse_args(["--project_root", str(self.root), *flags])

    def assets(self):
        base = "models/stable-diffusion-v1-4/"
        for folder, weight in (("text_encoder", "model.safetensors"),
                               ("unet", "diffusion_pytorch_model.safetensors"),
                               ("vae", "diffusion_pytorch_model.safetensors")):
            self.file(base + folder + "/config.json", "{}")
            self.file(base + folder + "/" + weight)
        for name in ("tokenizer_config.json", "vocab.json", "merges.txt"):
            self.file(base + "tokenizer/" + name)
        self.file(base + "scheduler/scheduler_config.json", '{"_class_name": "PNDMScheduler"}')
        self.file("output/safety_lora/best_val_loss_safety_lora/pytorch_lora_weights.safetensors", "adapter")
        for _, filename, column in TASK_DEFINITIONS:
            self.file("data/" + filename, f"case_number,{column}\n7,first prompt\n8,second prompt\n9,third prompt\n")

    def test_exact_task_set_and_subset_selection(self):
        args = self.args()
        self.assertEqual(args.tasks, ["sneakyprompt_meaningful_perturbed", "sneakyprompt_nonsense_perturbed",
                                     "nudity_eta_3_k_77", "p4dn_16_prompt", "clean"])
        subset = build_tasks(self.args("--tasks", "clean,p4dn_16_prompt,clean"))
        self.assertEqual([task.name for task in subset], ["clean", "p4dn_16_prompt"])

    def test_auto_checkpoint_selection_and_ambiguity(self):
        first = self.file("output/experiment_a/best_val_loss_safety_lora/pytorch_lora_weights.safetensors")
        self.assertEqual(self.args().lora_path, str(first))
        self.file("output/experiment_b/best_val_loss_safety_lora/pytorch_lora_weights.safetensors")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.args()
        preferred = self.file("output/safety_lora/best_val_loss_safety_lora/pytorch_lora_weights.safetensors")
        self.assertEqual(self.args().lora_path, str(preferred))
        self.assertEqual(self.args("--lora_path", str(first.parent)).lora_path, str(first))

    def test_root_override_and_baseline(self):
        args = self.args("--no_lora", "--output_root", "results")
        self.assertIsNone(args.lora_path)
        self.assertEqual(args.model_alias, "base")
        self.assertEqual(args.output_root, str(self.root / "results"))

    def test_release_checkpoint_is_used_without_occupying_training_output(self):
        packaged = self.file("checkpoints/safety_lora/best_val_loss_safety_lora/pytorch_lora_weights.safetensors")
        self.assertEqual(self.args().lora_path, str(packaged))
        trained = self.file("output/safety_lora/best_val_loss_safety_lora/pytorch_lora_weights.safetensors")
        self.assertEqual(self.args().lora_path, str(packaged))
        self.assertEqual(self.args("--lora_path", str(trained)).lora_path, str(trained))

    def test_input_checks_create_no_outputs_and_ignore_unselected_tasks(self):
        self.assets()
        jobs, errors = prepare_jobs(self.args())
        self.assertEqual(errors, [])
        self.assertEqual(sum(len(job.records) for job in jobs), 15)
        self.assertFalse((self.root / "output/generated_benchmarks").exists())
        (self.root / "data/SneakyPrompt200_meaningful_seeded_123.csv").unlink()
        self.assertEqual(prepare_jobs(self.args("--tasks", "clean"))[1], [])
        self.assertTrue(prepare_jobs(self.args())[1])

    def test_rows_keep_indices_seeds_and_prompt_whitespace(self):
        task = build_tasks(self.args("--tasks", "clean"))[0]
        self.file("data/clean.csv", 'id,prompt\n10.0," first prompt "\n20,second prompt\n')
        records = read_prompt_records(task, 12345, max_prompts=1)
        self.assertEqual(records[0].file_name, "0000_case10_seed12345.png")
        self.assertEqual(records[0].prompt, " first prompt ")
        self.assertEqual(len(records), 1)

    def test_empty_prompts_and_missing_columns_report_errors(self):
        self.assets()
        self.file("data/clean.csv", "wrong_column\nhello\n")
        self.assertTrue(any("missing column" in message for message in prepare_jobs(self.args("--tasks", "clean"))[1]))
        self.file("data/clean.csv", "prompt\n\"\"\n")
        self.assertTrue(any("empty prompt" in message for message in prepare_jobs(self.args("--tasks", "clean"))[1]))

    def test_configuration_changes_cannot_mix_existing_images(self):
        self.assets()
        args = self.args("--tasks", "clean")
        job = prepare_jobs(args)[0][0]
        job.output_dir.mkdir(parents=True)
        (job.output_dir / "generation_config.json").write_text(json.dumps(job.metadata), encoding="utf-8")
        self.assertEqual(prepare_jobs(args)[1], [])
        changed = self.args("--tasks", "clean", "--seed_base", "999")
        self.assertTrue(any("configuration differs" in message for message in prepare_jobs(changed)[1]))
        self.file("output/safety_lora/best_val_loss_safety_lora/pytorch_lora_weights.safetensors", "changed weights")
        self.assertTrue(any("configuration differs" in message for message in prepare_jobs(args)[1]))

    def test_untracked_existing_output_is_protected(self):
        self.assets()
        self.file("output/generated_benchmarks/safety_lora/clean/previous.png", "preserve")
        self.assertTrue(any("no generation_config" in message for message in prepare_jobs(self.args("--tasks", "clean"))[1]))

    def test_partial_fp16_variants_do_not_apply_to_entire_pipeline(self):
        base = self.root / "model"
        self.file("model/unet/diffusion_pytorch_model.fp16.safetensors")
        self.assertEqual(component_load_kwargs(base), {})
        self.file("model/vae/diffusion_pytorch_model.fp16.safetensors")
        self.file("model/text_encoder/model.fp16.safetensors")
        self.assertEqual(component_load_kwargs(base).get("variant"), "fp16")

    def test_invalid_task_alias_dimensions_and_precision(self):
        for flags in (("--tasks", "sneakyprompt_meaningful_original"), ("--model_alias", "../outside"),
                      ("--height", "511"), ("--batch_size", "0"), ("--guidance_scale", "nan"),
                      ("--device", "cpu", "--precision", "fp16"), ("--no_lora", "--lora_path", "somewhere")):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.args(*flags)

    def test_cli_from_different_working_directory(self):
        self.assets()
        result = subprocess.run([sys.executable, "-B", str(CODE_DIR / "infer.py"), "--project_root", str(self.root), "--check_only"],
                                cwd=self.root, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("5 tasks, 15 images planned", result.stdout)
        self.assertFalse((self.root / "output/generated_benchmarks").exists())

    def test_saved_metadata_paths_are_portable(self):
        self.assets()
        job = prepare_jobs(self.args("--tasks", "clean"))[0][0]
        serialized = json.dumps(job.metadata)
        self.assertNotIn(str(self.root), serialized)
        self.assertEqual(job.metadata["base_model_path"], "models/stable-diffusion-v1-4")

    def test_generation_batches_seeds_skip_and_regeneration(self):
        self.assets()
        args = self.args("--tasks", "clean", "--batch_size", "2")
        job = prepare_jobs(args)[0][0]
        calls = []

        class Generator:
            def __init__(self, device):
                self.device = device

            def manual_seed(self, seed):
                self.seed = seed
                return self

        class Image:
            def save(self, path, format):
                Path(path).write_bytes(b"test image payload")

        class Pipe:
            device = "cpu"

            def __call__(self, **kwargs):
                calls.append(kwargs)
                return types.SimpleNamespace(images=[Image() for _ in kwargs["prompt"]])

        fake_torch = types.SimpleNamespace(Generator=Generator, inference_mode=contextlib.nullcontext)
        fake_tqdm = types.ModuleType("tqdm")
        fake_tqdm_auto = types.ModuleType("tqdm.auto")
        fake_tqdm_auto.tqdm = MagicMock()
        with patch.dict(sys.modules, {"torch": fake_torch, "tqdm": fake_tqdm, "tqdm.auto": fake_tqdm_auto}), contextlib.redirect_stdout(io.StringIO()):
            generate_for_task(Pipe(), job, args)
            self.assertEqual([len(call["prompt"]) for call in calls], [2, 1])
            self.assertEqual([g.seed for call in calls for g in call["generator"]], [12345, 12346, 12347])
            self.assertEqual(len(list(job.output_dir.glob("*.png"))), 3)
            self.assertEqual(list(job.output_dir.glob("*.tmp")), [])
            calls.clear()
            generate_for_task(Pipe(), job, args)
            self.assertEqual(calls, [])
            args.skip_existing = False
            generate_for_task(Pipe(), job, args)
            self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
