"""Independent CLI commands must preserve training output provenance."""

import subprocess
import runpy
import tempfile
import unittest
from pathlib import Path

from experthtr.runtime.artifacts import prepare_output_directory
from experthtr.runtime.context import RuntimeState

validate_source_visibility = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "check_python_contract.py")
)["validate_source_visibility"]


class RuntimeArtifactTests(unittest.TestCase):
    def test_audit_then_train_then_evaluate_preserves_training_files(self):
        with tempfile.TemporaryDirectory() as root:
            config = {"output_dir": str(Path(root) / "run"),
                      "dense_init_allow_existing_output": False}
            output = prepare_output_directory(RuntimeState("audit", {"CFG": config}))
            self.assertEqual(list(output.iterdir()), [])
            prepare_output_directory(RuntimeState("train", {"CFG": config}))
            artifact = output / "dense_to_moe_initialization.json"
            artifact.write_text("original training provenance")
            for command in ("audit", "dataset-summary", "final-test", "routing", "position-diagnostics"):
                prepare_output_directory(RuntimeState(command, {"CFG": config}))
                self.assertEqual(artifact.read_text(), "original training provenance")
            with self.assertRaises(FileExistsError):
                prepare_output_directory(RuntimeState("train", {"CFG": config}))

    def test_source_ignore_rule_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            package = root / "src" / "experthtr"
            source = package / "data" / "dataset.py"
            source.parent.mkdir(parents=True)
            source.write_text("# source\n")
            (root / ".gitignore").write_text("data/\n")
            with self.assertRaisesRegex(ValueError, "hidden by Git"):
                validate_source_visibility(package, [source])
            (root / ".gitignore").write_text("/data/\n")
            validate_source_visibility(package, [source])
