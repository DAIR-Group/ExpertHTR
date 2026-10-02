#!/usr/bin/env python3
"""Static release checks for the canonical Python-first ExpertHTR package."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REQUIRED_CONFIG_SNIPPETS = (
    'MOE_ABLATION_VARIANT = "sparsegen_shared_lamsparse_balance"',
    'INIT_MODE = "dense_checkpoint"',
    'DENSE_INIT_ALLOW_SOURCE_SUPERSET = False',
    'REUSE_UNCHANGED_TOKEN_AUDIT = False',
    'MOE_LAYER_INDICES = (1, 5, 9, 13, 17, 21)',
    'LAM_SPARSE_COEF = 0.01',
    'LOAD_BALANCE_COEF = 0.01',
    'ROUTER_CE_COEF = 0.0',
)
FORBIDDEN_PATH_FRAGMENTS = ("/home/namhoai", "nguoimot", "namdh1706")


def validate_source_visibility(package_root: Path, sources: list[Path]) -> None:
    """Catch ignore rules that hide source files on the developer's machine."""
    repository = package_root.resolve().parent
    if not (repository / ".git").exists():
        return  # Source distributions need not carry Git metadata.
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--",
         *(str(path.resolve().relative_to(repository)) for path in sources)],
        cwd=repository, capture_output=True, text=True, check=False,
    )
    if result.returncode == 0:
        raise ValueError(f"Package source files are hidden by Git ignore rules:\n{result.stdout}")
    if result.returncode != 1:
        raise RuntimeError(f"Could not check source visibility: {result.stderr}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "expert",
    )
    args = parser.parse_args()
    package_root = args.package_root
    sources = sorted(
        path for path in package_root.rglob("*.py") if "__pycache__" not in path.parts
    )
    if not sources:
        raise FileNotFoundError(f"No Python package sources found under {package_root}")
    validate_source_visibility(package_root, sources)
    if (package_root / "stages").exists():
        raise ValueError("Deprecated stages/ directory is present; publish only named Python modules.")

    combined = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    for path in sources:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    missing = [snippet for snippet in REQUIRED_CONFIG_SNIPPETS if snippet not in combined]
    if missing:
        raise ValueError(f"Python package no longer matches the frozen balance arm: {missing}")
    leaked = [fragment for fragment in FORBIDDEN_PATH_FRAGMENTS if fragment in combined]
    if leaked:
        raise ValueError(f"Python package contains environment-specific path fragments: {leaked}")
    if "exec(" in combined or "runpy.run_path" in combined:
        raise ValueError("The public workflow must not use dynamic source execution.")

    sys.path.insert(0, str(package_root.parent))
    from expert.workflow import validate_phase_imports

    phases = validate_phase_imports()
    print(f"PASS: compiled {len(sources)} package files and registered {len(phases)} named phases")


if __name__ == "__main__":
    main()
