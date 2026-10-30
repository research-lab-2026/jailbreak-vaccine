"""Test portable launch behavior without importing the training dependencies."""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from training_config import check_inputs, parse_args, split_sizes


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name).resolve()

    def args(self, *extra):
        return parse_args(["--project_root", str(self.root), *extra])

    def file(self, name, contents=""):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
        return path

    def assets(self, bin_only=False):
        self.file("data/I2P_sexual_931.csv", "perturbed_text\n" + "\n".join(f"prompt {i}" for i in range(20)))
        sd = "models/stable-diffusion-v1-4/"
        for component in ("text_encoder", "unet", "vae"):
            self.file(sd + component + "/config.json", "{}")
            prefix = "pytorch_model" if component == "text_encoder" else "diffusion_pytorch_model"
            name = prefix + ".bin" if bin_only else ("model" if component == "text_encoder" else prefix) + ".safetensors"
            self.file(sd + component + "/" + name)
        self.file(sd + "scheduler/scheduler_config.json", "{}")
        for filename in ("tokenizer_config.json", "vocab.json", "merges.txt"):
            self.file(sd + "tokenizer/" + filename)
            self.file("models/clip-vit-large-patch14/" + filename)
        for folder in ("stable-diffusion-safety-checker", "clip_noise_finetuned_safety_checker/best", "clip-vit-large-patch14"):
            self.file("models/" + folder + "/config.json", "{}")
            self.file("models/" + folder + ("/pytorch_model.bin" if bin_only else "/model.safetensors"))

    def test_defaults_do_not_follow_working_directory(self):
        before = parse_args([])
        old_cwd = Path.cwd()
        try:
            os.chdir(self.root)
            after = parse_args([])
        finally:
            os.chdir(old_cwd)
        self.assertEqual(before.csv_path, after.csv_path)
        self.assertEqual(before.project_root, str(CODE_DIR.parent))

    def test_project_root_anchors_custom_paths(self):
        args = self.args("--csv_path", "inputs/prompts.csv", "--output_dir", "results/new",
                         "--base_model_id", "./custom/sd")
        self.assertEqual(args.csv_path, str(self.root / "inputs/prompts.csv"))
        self.assertEqual(args.output_dir, str(self.root / "results/new"))
        self.assertEqual(args.base_model_id, str(self.root / "custom/sd"))

    def test_hub_id_stays_intact_and_local_only_can_be_disabled(self):
        args = self.args("--base_model_id", "hf://organization/model", "--no-local_files_only")
        self.assertEqual(args.base_model_id, "organization/model")
        self.assertFalse(args.local_files_only)
        _, errors = check_inputs(args)
        self.assertFalse(any("base_model_id" in error for error in errors))

    def test_both_weight_formats_and_no_output_creation(self):
        for bin_only in (False, True):
            with self.subTest(bin_only=bin_only):
                self.assets(bin_only=bin_only)
                _, errors = check_inputs(self.args())
                self.assertEqual(errors, [])
                self.assertFalse((self.root / "output").exists())
                # Make the second iteration check only the alternative weight format.
                for path in self.root.rglob("*.safetensors"):
                    path.unlink()

    def test_nonempty_output_is_protected(self):
        self.assets()
        checkpoint = self.file("output/safety_lora/previous_result.txt", "preserve this")
        _, errors = check_inputs(self.args())
        self.assertTrue(any("Output already exists" in error for error in errors))
        self.assertEqual(checkpoint.read_text(encoding="utf-8"), "preserve this")

    def test_empty_output_directory_is_allowed(self):
        self.assets()
        (self.root / "output/safety_lora").mkdir(parents=True)
        self.assertEqual(check_inputs(self.args())[1], [])

    def test_missing_assets_are_reported_together(self):
        _, errors = check_inputs(self.args())
        self.assertEqual(len(errors), 5)

    def test_oversized_batch_is_rejected_before_loading_models(self):
        self.assets()
        _, errors = check_inputs(self.args("--train_batch_size", "100"))
        self.assertTrue(any("exceeds the training split" in error for error in errors))

    def test_wrong_csv_column_and_tiny_dataset(self):
        self.assets()
        self.file("data/I2P_sexual_931.csv", "wrong_column\nprompt\n")
        self.assertTrue(any("column" in error for error in check_inputs(self.args())[1]))
        self.file("data/I2P_sexual_931.csv", "perturbed_text\nprompt\n")
        self.assertTrue(any("At least 10" in error for error in check_inputs(self.args())[1]))

    def test_invalid_options_fail_during_parsing(self):
        for flags in (("--resolution", "513"), ("--gradient_accumulation_steps", "0"),
                      ("--step_min_ratio", "0.9", "--step_max_ratio", "0.1"),
                      ("--learning_rate", "nan"), ("--split_val_ratio", "0")):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    self.args(*flags)
                self.assertEqual(raised.exception.code, 2)

    def test_split_preserves_default_counts_and_extreme_ratios(self):
        self.assertEqual(split_sizes(931, 0.7, 0.2, 0.1), (652, 186, 93))
        for ratios in ((0.999, 0.0005, 0.0005), (0.001, 0.998, 0.001), (7, 2, 1)):
            sizes = split_sizes(10, *ratios)
            self.assertEqual(sum(sizes), 10)
            self.assertTrue(all(size >= 1 for size in sizes))

    def test_cli_dry_run_outside_project_needs_no_assets(self):
        result = subprocess.run([sys.executable, "-B", str(CODE_DIR / "train.py"),
                                 "--project_root", str(self.root), "--dry_run"],
                                cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        config = json.loads(result.stdout)
        self.assertEqual(config["csv_path"], str(self.root / "data/I2P_sexual_931.csv"))
        self.assertEqual(config["train_batch_size"] * config["gradient_accumulation_steps"], 32)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_cli_check_only_with_fake_local_assets(self):
        self.assets()
        result = subprocess.run([sys.executable, "-B", str(CODE_DIR / "train.py"),
                                 "--project_root", str(self.root), "--check_only"],
                                cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Input checks passed", result.stdout)
        self.assertFalse((self.root / "output").exists())


if __name__ == "__main__":
    unittest.main()
