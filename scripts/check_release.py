#!/usr/bin/env python3
"""Validate the static ExpertHTR release contract without model dependencies.

This check is deliberately small: it catches accidental changes to the public
configuration before a repository is pushed, while the Python workflow keeps
the runtime/model-level assertions needed for a real run.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


EXPECTED_SOURCES = {
    "BRESSAY", "Bentham", "HWDB2.0", "IAM", "ICFHR2016",
    "Images_Courriers", "ScribbleLens",
}
EXPECTED_TASKS = {
    "page_htr": 4393,
    "line_count": 4393,
    "phrase_line_lookup": 4393,
    "line_range_htr": 7209,
}


def require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required configuration key: {key}")
    return mapping[key]


def validate(config: dict[str, Any]) -> None:
    data = require(config, "data")
    moe = require(config, "moe")
    model = require(config, "model")
    optimization = require(config, "optimization")
    experiment = require(config, "experiment")

    if experiment["id"] != "sparsegen_shared_lamsparse_balance":
        raise ValueError("Release configuration must identify the MoE-balance arm.")
    if experiment["initialization"] != "dense_checkpoint":
        raise ValueError("Paper release must use a dense checkpoint initialization.")
    if set(data["expected_sources"]) != EXPECTED_SOURCES:
        raise ValueError(f"Source scope mismatch: {data['expected_sources']}")
    if data["excluded_sources"] != ["Eparchos"]:
        raise ValueError("The paper-facing release excludes exactly Eparchos.")
    if data["expected_records"] != {"train": 20388, "valid": 962, "test": 1327}:
        raise ValueError(f"Unexpected split records: {data['expected_records']}")
    if data["tasks"]["train"] != EXPECTED_TASKS:
        raise ValueError(f"Unexpected train-task counts: {data['tasks']['train']}")
    if data["bbox_prediction"] != "disabled":
        raise ValueError("The release must not advertise bbox prediction.")
    if data["traversal"] != "full_pass":
        raise ValueError("The release uses a full-pass record traversal.")
    if data["source_balanced_sampler"] or data["group_balanced_sampler"]:
        raise ValueError("Balanced replacement samplers are outside this frozen configuration.")
    if moe["routing"] != "sparsegen" or moe["hard_top_k"]:
        raise ValueError("This configuration is native Sparsegen, not fixed Top-K.")
    if moe["routed_experts_per_layer"] != 4 or moe["shared_experts_per_layer"] != 1:
        raise ValueError("Expected four routed and one shared expert per selected layer.")
    if moe["layer_indices"] != [1, 5, 9, 13, 17, 21]:
        raise ValueError("Unexpected MoE placement.")
    if float(moe["router_cross_entropy_coefficient"]) != 0.0:
        raise ValueError("Router CE is intentionally disabled in the balance arm.")
    if float(moe["lam_sparse"]["coefficient"]) != 0.01:
        raise ValueError("LamSparse coefficient drifted from the frozen setting.")
    if float(moe["load_balance"]["coefficient"]) != 0.01:
        raise ValueError("Load-balance coefficient drifted from the frozen setting.")
    if model["max_text_length"] != 2560 or model["generation"]["max_new_tokens"] != 1024:
        raise ValueError("Text/generation caps differ from the frozen setting.")
    if model["image"] != {
        "resize_policy": "area_ratio", "min_pixels": 589824,
        "max_pixels": 1572864, "max_aspect_ratio": 200,
    }:
        raise ValueError("Image-budget contract drifted.")
    if optimization["per_device_batch_size"] * optimization["gradient_accumulation_steps"] != 32:
        raise ValueError("The reference effective batch must be 32.")
    if float(optimization["token_unlikelihood_coefficient"]) != 0.0:
        raise ValueError("Token unlikelihood is not part of the released MoE-balance arm.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "moe_balance.yaml",
    )
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate(config)
    print(f"PASS: static release contract is valid ({args.config})")


if __name__ == "__main__":
    main()
