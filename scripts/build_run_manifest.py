#!/usr/bin/env python3
"""Create a portable, hash-based manifest for a completed ExpertHTR run.

The script does not train or evaluate a model. It records the exact public
configuration, JSONL hashes, checkpoint metadata hashes, and runtime package
versions supplied by the caller. Keep the generated manifest next to the
paper's reported checkpoint; do not commit model weights or private paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_entry(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True, help="mthtr root or page_multitask directory")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Selected checkpoint directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs" / "moe_balance.yaml")
    parser.add_argument("--model-dir", type=Path, default=None, help="Optional local base model directory; only metadata files are hashed")
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    page_root = data_root if data_root.name == "page_multitask" else data_root / "page_multitask"
    checkpoint = args.checkpoint.expanduser().resolve()
    config = args.config.expanduser().resolve()
    require_file(config, "release configuration")

    dataset_files = {
        split: file_entry(require_file(page_root / f"{split}.jsonl", f"{split} JSONL"))
        for split in ("train", "valid", "test")
    }
    checkpoint_files = {}
    for name in ("CHECKPOINT_COMPLETE", "checkpoint_manifest.json", "metadata.json", "full_model_state.pt"):
        path = checkpoint / name
        if path.exists():
            checkpoint_files[name] = file_entry(path)
    if "checkpoint_manifest.json" not in checkpoint_files or "full_model_state.pt" not in checkpoint_files:
        raise FileNotFoundError("Checkpoint must include checkpoint_manifest.json and full_model_state.pt")

    model_files = {}
    if args.model_dir:
        model_dir = args.model_dir.expanduser().resolve()
        for name in ("config.json", "generation_config.json", "preprocessor_config.json", "tokenizer_config.json"):
            path = model_dir / name
            if path.exists():
                model_files[name] = file_entry(path)

    package_versions = {}
    for name in ("torch", "transformers", "numpy", "PIL"):
        try:
            module = __import__(name)
            package_versions[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            package_versions[name] = f"unavailable: {type(exc).__name__}"

    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "release_config": file_entry(config),
        "dataset_jsonl": dataset_files,
        "checkpoint": {"directory": str(checkpoint), "files": checkpoint_files},
        "base_model_metadata": model_files,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": package_versions,
        },
        "note": (
            "Paths are local provenance aids. Hashes, config, checkpoint manifest, and recorded package versions "
            "are the reproducibility evidence; raw data and model weights are intentionally excluded from Git."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
