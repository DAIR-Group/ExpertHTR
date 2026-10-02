"""Dependency-free regression tests for ExpertHTR's public orchestration API.

These tests intentionally avoid Torch and data access.  GPU training correctness
is guarded by the runtime token audit and checkpoint contracts, while this suite
protects the package/CLI boundary that reviewers and CI can exercise anywhere.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from expert.cli import main
from expert.config.reference import configure
from expert.runtime.context import RuntimeState
from expert.workflow import available_commands, phase_names_for, validate_phase_imports


class PublicContractTests(unittest.TestCase):
    """Keep the public command surface and frozen reference defaults stable."""

    def test_named_phases_are_registered_once(self) -> None:
        phases = validate_phase_imports()
        self.assertEqual(len(phases), 17)
        self.assertEqual(len(phases), len(set(phases)))
        self.assertIn("modeling.moe", phases)
        self.assertIn("evaluation.final_test", phases)

    def test_train_phase_order_guards_audit_before_updates(self) -> None:
        self.assertEqual(
            phase_names_for("train")[-5:],
            (
                "reporting.dataset_summary",
                "audit.token",
                "training.engine",
                "training.checkpointing",
                "training.run",
            ),
        )
        self.assertIn("routing", available_commands())

    def test_cli_validation_has_no_runtime_dependencies(self) -> None:
        self.assertEqual(main(["--validate-imports"]), 0)

    def test_reference_configuration_is_dense_initialized_and_native_sparsegen(self) -> None:
        with tempfile.TemporaryDirectory() as output_root:
            with patch.dict(
                os.environ,
                {
                    "HWVLM_DENSE_INIT_CHECKPOINT": "/tmp/selected-dense-checkpoint",
                    "HWVLM_OUTPUT_ROOT": output_root,
                },
                clear=False,
            ):
                state = configure(RuntimeState(command="unit-test"))

        self.assertEqual(state["INIT_MODE"], "dense_checkpoint")
        self.assertEqual(state["MOE_ABLATION_VARIANT"], "sparsegen_shared_lamsparse_balance")
        self.assertEqual(state["MOE_LAYER_INDICES"], (1, 5, 9, 13, 17, 21))
        self.assertFalse(state["ABLATION"]["enforce_top_k"])
        self.assertEqual(state["ABLATION"]["shared_experts"], 1)
        self.assertEqual(state["ABLATION"]["router_ce"], 0.0)

    def test_runtime_state_reports_missing_dependency(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "missing required state"):
            RuntimeState(command="unit-test").require("processor")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
