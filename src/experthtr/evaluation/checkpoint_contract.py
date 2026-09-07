"""Validate tokenizer semantics before restoring evaluation weights."""

from collections.abc import Mapping
from typing import Any


def validate_atomic_marker_contract(
    manifest: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    markers = list(config.get("atomic_ocr_markers") or [])
    if not markers or list(manifest.get("atomic_ocr_markers") or []) != markers:
        raise RuntimeError("Checkpoint atomic-marker policy differs from the active runtime.")
    expected_ids = dict(config.get("atomic_ocr_marker_ids") or {})
    saved_ids = dict(manifest.get("atomic_ocr_marker_ids") or {})
    if set(expected_ids) != set(markers) or saved_ids != expected_ids:
        raise RuntimeError(
            "Checkpoint atomic-marker IDs are missing or differ from the active tokenizer: "
            f"checkpoint={saved_ids}, runtime={expected_ids}"
        )
