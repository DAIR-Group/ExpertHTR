"""Exercise checkpoint selection, restore guards, and report writing without GPU."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from expert.evaluation.final_test import run_final_test
from expert.runtime.context import RuntimeState


class FinalTestTests(unittest.TestCase):
    def make_state(self, root):
        checkpoint = root / "selected"
        checkpoint.mkdir()
        markers = {"<del>": 101, "<gap>": 102}
        manifest = {"moe_signature": {"enabled": True},
                    "atomic_ocr_markers": list(markers), "atomic_ocr_marker_ids": markers}
        (checkpoint / "checkpoint_manifest.json").write_text(json.dumps(manifest))
        (checkpoint / "CHECKPOINT_COMPLETE").touch()
        (checkpoint / "full_model_state.pt").touch()
        (root / "run_summary.json").write_text(json.dumps({"best_checkpoint": {"path": str(checkpoint)}}))
        config = {"output_dir": str(root), "eval_max_new_tokens": 1024,
                  "atomic_ocr_markers": list(markers), "atomic_ocr_marker_ids": markers,
                  "dataset": "page_multitask", "excluded_sources": [],
                  "excluded_valid_tasks": [], "expected_sources": ["IAM"],
                  "expected_split_records": {"test": 1}, "num_workers": 0}
        torch = MagicMock()
        torch.cuda.is_available.return_value = False
        torch.load.return_value = {"model_state_dict": {"weights": "test"}}
        dataset = MagicMock()
        dataset.__len__.return_value = 1
        metrics = {"cer_generate": 0.25, "wer_generate": 1.0, "num_samples": 1,
                   "by_source": {"IAM": {"cer_generate": 0.25, "wer_generate": 1.0, "num_samples": 1}}}

        def evaluate(*args, **kwargs):
            kwargs["output_jsonl_path"].write_text('{"id": "test-page"}\n')
            return metrics

        return RuntimeState("final-test", {
            "CFG": config, "os": os, "Path": Path, "json": json, "torch": torch,
            "model": MagicMock(), "processor": MagicMock(), "LOGGER": MagicMock(),
            "CHECKPOINT_COMPLETE_FILE": "CHECKPOINT_COMPLETE",
            "CHECKPOINT_MANIFEST_FILE": "checkpoint_manifest.json",
            "FULL_MODEL_STATE_FILE": "full_model_state.pt",
            "_moe_signature": lambda: {"enabled": True},
            "resolve_split_path": lambda *args: root / "test.jsonl",
            "DATA_ROOT": root, "IMAGE_ROOT": root,
            "OCRJsonlDataset": lambda **kwargs: dataset,
            "_exclude_sources_inplace": lambda data, *args: data,
            "_exclude_tasks_inplace": lambda data, *args: data,
            "_dataset_samples_for_stats": lambda data: [],
            "_count_sources": lambda data: {"IAM": 1},
            "_count_tasks": lambda data: {"page_htr": 1},
            "DataLoader": MagicMock(), "collator": object(),
            "evaluate_loader": MagicMock(side_effect=evaluate),
        })

    def test_standalone_command_selects_checkpoint_and_writes_reports(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {"HWVLM_FINAL_TEST_CHECKPOINT": ""}):
            root = Path(root)
            state = self.make_state(root)
            self.assertNotIn("output_dir", state)
            run_final_test(state)
            state["model"].load_state_dict.assert_called_once_with({"weights": "test"}, strict=True)
            self.assertEqual(state["final_test_checkpoint"], root / "selected")
            report = json.loads((root / "test_metrics_validation_selected.json").read_text())
            self.assertEqual(report["cer_generate"], 0.25)
            self.assertTrue((root / "test_metrics_by_source_validation_selected.csv").is_file())
            self.assertTrue((root / "test_predictions_validation_selected.jsonl").is_file())

    def test_wrong_or_missing_marker_ids_fail_before_weight_load(self):
        for saved in ({}, {"<del>": 102, "<gap>": 101}):
            with self.subTest(saved=saved), tempfile.TemporaryDirectory() as root:
                root = Path(root)
                state = self.make_state(root)
                path = root / "selected" / "checkpoint_manifest.json"
                manifest = json.loads(path.read_text())
                manifest["atomic_ocr_marker_ids"] = saved
                path.write_text(json.dumps(manifest))
                with patch.dict(os.environ, {"HWVLM_FINAL_TEST_CHECKPOINT": str(path.parent)}):
                    with self.assertRaisesRegex(RuntimeError, "atomic-marker IDs"):
                        run_final_test(state)
                state["torch"].load.assert_not_called()
                state["evaluate_loader"].assert_not_called()
