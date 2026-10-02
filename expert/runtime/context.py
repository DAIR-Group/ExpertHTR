"""Explicit shared runtime state for the research workflow.

The HTR pipeline was originally written as one linear program: model, processor,
loaders, metric helpers, and checkpoint utilities are intentionally shared
between phases. ``RuntimeState`` makes that dependency explicit without
changing the audited computation or reinitialising state between commands.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuntimeState(MutableMapping[str, Any]):
    """A named, mutable dependency container for one CLI invocation."""

    command: str
    values: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.values[key] = value

    def __delitem__(self, key: str) -> None:
        del self.values[key]

    def __iter__(self):
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)

    def require(self, *keys: str) -> None:
        """Fail before a phase runs when its declared dependencies are absent."""

        missing = [key for key in keys if key not in self.values]
        if missing:
            raise RuntimeError(
                f"Runtime command {self.command!r} is missing required state: {missing}. "
                "Check the workflow phase order."
            )


def export_runtime_locals(state: RuntimeState, local_scope: Mapping[str, Any]) -> RuntimeState:
    """Publish phase-local values needed by a later phase.

    Private helpers are intentionally retained: metric, checkpoint, and MoE
    code share those helpers across phases.  Only wrapper implementation names
    are excluded.
    """

    state.values.update(
        {
            key: value
            for key, value in local_scope.items()
            if key not in {"state", "RuntimeState", "export_runtime_locals"}
        }
    )
    return state
