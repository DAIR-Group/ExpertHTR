#!/usr/bin/env python3
"""Compare core computation with a supplied reference notebook without executing it."""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path


CORE_MODULES = (
    "evaluation/metrics.py",
    "training/engine.py",
    "training/checkpointing.py",
    "modeling/moe.py",
    "data/dataset.py",
)
DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
CONFIG_NAMES = (
    "MOE_ABLATION_VARIANT", "INIT_MODE", "MOE_LAYER_INDICES",
    "LAM_SPARSE_COEF", "LOAD_BALANCE_COEF", "ROUTER_CE_COEF",
    "REUSE_UNCHANGED_TOKEN_AUDIT",
)


class NormalizeDocstrings(ast.NodeTransformer):
    """Ignore indentation introduced when a notebook cell becomes a function."""

    def generic_visit(self, node):
        if isinstance(node, DEFINITIONS) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                if isinstance(first.value.value, str):
                    first.value.value = inspect.cleandoc(first.value.value)
        return super().generic_visit(node)


def fingerprint(node: ast.AST) -> str:
    return ast.dump(NormalizeDocstrings().visit(node), include_attributes=False)


def literal_controls(nodes) -> dict:
    controls = {}
    for node in nodes:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in CONFIG_NAMES:
                try:
                    controls[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    return controls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args()
    notebook_bytes = args.notebook.read_bytes()
    notebook = json.loads(notebook_bytes)
    originals: dict[str, list[str]] = {}
    notebook_controls = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell.get("source", []))
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue  # Notebook installation/shell-magic cells are not executed.
        notebook_controls.update(literal_controls(tree.body))
        for node in tree.body:
            if isinstance(node, DEFINITIONS):
                originals.setdefault(node.name, []).append(fingerprint(node))

    package = Path(__file__).resolve().parents[1] / "src" / "experthtr"
    count = 0
    mismatches = []
    for relative in CORE_MODULES:
        tree = ast.parse((package / relative).read_text(encoding="utf-8"))
        module_count = 0
        for wrapper in tree.body:
            if not isinstance(wrapper, ast.FunctionDef):
                continue
            for node in wrapper.body:
                if not isinstance(node, DEFINITIONS):
                    continue
                module_count += 1
                if fingerprint(node) not in originals.get(node.name, []):
                    mismatches.append(f"{relative}:{node.name}")
        if not module_count:
            mismatches.append(f"{relative}: no core definitions found")
        count += module_count
    if mismatches:
        raise ValueError(f"Core definitions differ from the notebook: {mismatches}")
    print(f"PASS: {count} core function/class definitions match the reference notebook AST")
    print(f"Reference SHA-256: {hashlib.sha256(notebook_bytes).hexdigest()}")
    print("Scope: CER/WER, training/evaluation engine, MoE, checkpoint helpers, dataset/collator.")
    print("This does not validate runtime configuration, generated predictions, or GPU execution.")
    reference = ast.parse((package / "config/reference.py").read_text(encoding="utf-8"))
    configure = next(node for node in reference.body if isinstance(node, ast.FunctionDef) and node.name == "configure")
    repo_controls = literal_controls(configure.body)
    for name in CONFIG_NAMES:
        if name in notebook_controls and name in repo_controls:
            if notebook_controls[name] != repo_controls[name]:
                print(f"CONFIG DIFF: {name}: notebook={notebook_controls[name]!r}, repo={repo_controls[name]!r}")


if __name__ == "__main__":
    main()
