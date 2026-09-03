#!/usr/bin/env python3
"""Ensure the optional notebook is a clean CLI demonstration, not the runtime."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat


FORBIDDEN_PATH_FRAGMENTS = ("/home/namhoai", "nguoimot", "namdh1706")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--notebook",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "examples" / "demo_moe_balance.ipynb",
    )
    args = parser.parse_args()
    notebook = nbformat.read(args.notebook, as_version=4)
    nbformat.validate(notebook)

    outputs = sum(len(cell.get("outputs", [])) for cell in notebook.cells if cell.cell_type == "code")
    executions = sum(
        cell.get("execution_count") is not None
        for cell in notebook.cells
        if cell.cell_type == "code"
    )
    if outputs or executions:
        raise ValueError(f"Demo notebook must be cleared; outputs={outputs}, execution_counts={executions}")
    text = "\n".join(cell.source for cell in notebook.cells)
    if "experthtr" not in text:
        raise ValueError("Demo notebook must invoke the Python CLI, not carry an implementation copy.")
    leaked = [fragment for fragment in FORBIDDEN_PATH_FRAGMENTS if fragment in text]
    if leaked:
        raise ValueError(f"Demo notebook contains environment-specific path fragments: {leaked}")
    print(f"PASS: demo notebook is clean and delegates to the Python CLI ({args.notebook})")


if __name__ == "__main__":
    main()
