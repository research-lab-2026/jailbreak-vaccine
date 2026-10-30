"""Offline artifact installation tests; no model loading or network requests."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
import zipfile

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/download_artifacts.py"
SPEC = importlib.util.spec_from_file_location("download_artifacts", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ArtifactInstallationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.archive = self.root / "artifact.zip"

    def packed(self, filename="checkpoints/model/weights.safetensors", payload=b"model bytes"):
        with zipfile.ZipFile(self.archive, "w") as packed:
            packed.writestr(filename, payload)
        return dict(sha256=MODULE.sha256(self.archive), files={filename: hashlib.sha256(payload).hexdigest()})

    def test_install_and_idempotent_reinstall(self):
        metadata = self.packed()
        self.assertEqual(MODULE.install_archive(self.archive, metadata, self.root), 1)
        self.assertEqual(MODULE.install_archive(self.archive, metadata, self.root), 1)
        self.assertEqual((self.root / "checkpoints/model/weights.safetensors").read_bytes(), b"model bytes")

    def test_wrong_archive_hash_writes_nothing(self):
        metadata = self.packed()
        metadata["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Archive SHA-256 mismatch"):
            MODULE.install_archive(self.archive, metadata, self.root)
        self.assertFalse((self.root / "checkpoints").exists())

    def test_wrong_tensor_hash_writes_nothing(self):
        metadata = self.packed()
        metadata["files"]["checkpoints/model/weights.safetensors"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "file checksum mismatch"):
            MODULE.install_archive(self.archive, metadata, self.root)
        self.assertFalse((self.root / "checkpoints").exists())

    def test_existing_different_file_is_preserved(self):
        metadata = self.packed()
        target = self.root / "checkpoints/model/weights.safetensors"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"previous result")
        with self.assertRaisesRegex(ValueError, "Existing file differs"):
            MODULE.install_archive(self.archive, metadata, self.root)
        self.assertEqual(target.read_bytes(), b"previous result")

    def test_traversal_path_is_rejected(self):
        metadata = self.packed("../outside.bin")
        with self.assertRaisesRegex(ValueError, "unsafe path"):
            MODULE.install_archive(self.archive, metadata, self.root)

    def test_github_url_for_latest_and_pinned_release(self):
        self.assertEqual(MODULE.release_asset_url("review/repository", "latest", "best.zip"),
                         "https://github.com/review/repository/releases/latest/download/best.zip")
        self.assertEqual(MODULE.release_asset_url("review/repository", "v1", "best.zip"),
                         "https://github.com/review/repository/releases/download/v1/best.zip")
        with self.assertRaises(ValueError):
            MODULE.release_asset_url("../invalid/repository", "latest", "best.zip")


if __name__ == "__main__":
    unittest.main()
