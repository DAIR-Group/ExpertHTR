"""Compose the audited ExpertHTR phases from a normal Python process.

Each phase now lives in a named domain module—configuration, data, modeling,
training, evaluation, or analysis—rather than in a numbered notebook cell.
The explicit order remains fixed because the original research pipeline shares
model, processor, loader, metric, and checkpoint objects across those phases.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .analysis.position_diagnostics import run_position_diagnostics
from .analysis.routing import run_routing_analysis
from .audit.token import run_token_audit
from .config.reference import configure
from .data.contracts import validate_data_contract
from .data.dataset import register_dataset_types
from .data.loaders import build_loaders
from .evaluation.final_test import run_final_test
from .evaluation.metrics import register_metrics
from .modeling.model import build_model
from .modeling.moe import register_moe_components
from .reporting.dataset_summary import write_dataset_summary
from .runtime.bootstrap import configure_runtime
from .runtime.context import RuntimeState
from .runtime.dependencies import register_dependencies
from .training.checkpointing import register_checkpointing
from .training.engine import build_training_engine
from .training.run import run_training


class WorkflowPhaseError(RuntimeError):
    """Raised with a named phase when a runtime phase fails."""


Phase = tuple[str, Callable[[RuntimeState], RuntimeState]]

_BASE_PHASES: tuple[Phase, ...] = (
    ("config.reference", configure),
    ("runtime.dependencies", register_dependencies),
    ("runtime.bootstrap", configure_runtime),
    ("data.contracts", validate_data_contract),
    ("evaluation.metrics", register_metrics),
    ("data.dataset", register_dataset_types),
    ("modeling.moe", register_moe_components),
    ("modeling.model", build_model),
    ("data.loaders", build_loaders),
)

_COMMAND_PHASES: Mapping[str, tuple[Phase, ...]] = {
    "audit": _BASE_PHASES
    + (("reporting.dataset_summary", write_dataset_summary), ("audit.token", run_token_audit)),
    "dataset-summary": _BASE_PHASES + (("reporting.dataset_summary", write_dataset_summary),),
    "train": _BASE_PHASES
    + (
        ("reporting.dataset_summary", write_dataset_summary),
        ("audit.token", run_token_audit),
        ("training.engine", build_training_engine),
        ("training.checkpointing", register_checkpointing),
        ("training.run", run_training),
    ),
    "final-test": _BASE_PHASES
    + (
        ("audit.token", run_token_audit),
        ("training.engine", build_training_engine),
        ("training.checkpointing", register_checkpointing),
        ("evaluation.final_test", run_final_test),
    ),
    "routing": _BASE_PHASES
    + (("training.checkpointing", register_checkpointing), ("analysis.routing", run_routing_analysis)),
    "position-diagnostics": _BASE_PHASES
    + (
        ("training.checkpointing", register_checkpointing),
        ("analysis.position_diagnostics", run_position_diagnostics),
    ),
}


def available_commands() -> tuple[str, ...]:
    """Return the stable public CLI commands."""

    return tuple(_COMMAND_PHASES)


def phases_for(command: str) -> tuple[Phase, ...]:
    """Return the explicit ordered phases for ``command``."""

    try:
        return _COMMAND_PHASES[command]
    except KeyError as exc:
        choices = ", ".join(available_commands())
        raise ValueError(f"Unknown ExpertHTR command {command!r}; choose one of: {choices}") from exc


def phase_names_for(command: str) -> tuple[str, ...]:
    """Return phase module names for presentation and release checks."""

    return tuple(name for name, _ in phases_for(command))


def run(command: str) -> RuntimeState:
    """Run a command and return its explicit runtime dependency container."""

    state = RuntimeState(command=command)
    for phase_name, phase in phases_for(command):
        try:
            phase(state)
        except BaseException as exc:
            message = f"ExpertHTR command {command!r} failed in phase {phase_name}: {exc}"
            raise WorkflowPhaseError(message) from exc
    return state


def validate_phase_imports() -> Sequence[str]:
    """Validate that every Python phase has been imported and registered."""

    names = tuple(sorted({name for phases in _COMMAND_PHASES.values() for name, _ in phases}))
    if len(names) != 17:
        raise RuntimeError(f"Expected 17 named workflow phases, found {len(names)}: {names}")
    return names
