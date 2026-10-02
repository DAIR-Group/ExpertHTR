"""audit.token phase for the ExpertHTR research workflow.

This module preserves the audited runtime computation. It is invoked by the
explicit CLI workflow and exports its declared runtime objects into a
:class:`~expert.runtime.context.RuntimeState`.
"""

from __future__ import annotations

from expert.runtime.context import RuntimeState, export_runtime_locals


def run_token_audit(state: RuntimeState) -> RuntimeState:
    """Execute this phase after its prerequisites have populated ``state``."""

    globals().update(state.values)
    # ─────────────────────────────────────────────────────────────────────────────
    # Full-dataset token budget audit (run before training)
    # ─────────────────────────────────────────────────────────────────────────────
    # This audit does not change the dataset, labels, or augmentation policy.
    # Set True only when you want to scan every retained record.
    RUN_FULL_TOKEN_AUDIT = os.getenv("HWVLM_RUN_FULL_TOKEN_AUDIT", "1").lower() in {"1", "true", "yes"}

    TOKEN_AUDIT_SPLITS = ("train", "valid", "test")
    TOKEN_AUDIT_BATCH_SIZE = 1          # 1 is safest for page images; increase only if RAM permits.
    TOKEN_AUDIT_MAX_SAMPLES = 0         # 0 = all records in every selected split.
    TOKEN_AUDIT_MAX_LENGTH = int(CFG["max_text_length"])
    TOKEN_AUDIT_OUTPUT_DIR = Path(KAGGLE_WORKING) / "page_multitask_token_audit"


    def _audit_nonpad_ids(encoded: Dict[str, Any], row: int) -> List[int]:
        ids = encoded["input_ids"][row]
        mask = encoded.get("attention_mask")
        if hasattr(ids, "detach"):
            ids = ids.detach().cpu().tolist()
        else:
            ids = list(ids)
        if mask is None:
            return [int(x) for x in ids]
        row_mask = mask[row]
        if hasattr(row_mask, "detach"):
            row_mask = row_mask.detach().cpu().tolist()
        return [int(token_id) for token_id, keep in zip(ids, row_mask) if int(keep) != 0]


    def _audit_common_prefix(a: List[int], b: List[int]) -> int:
        n = min(len(a), len(b))
        idx = 0
        while idx < n and a[idx] == b[idx]:
            idx += 1
        return idx


    def _audit_common_suffix(a: List[int], b: List[int], prefix: int = 0) -> int:
        max_suffix = min(len(a), len(b)) - prefix
        idx = 0
        while idx < max_suffix and a[len(a) - 1 - idx] == b[len(b) - 1 - idx]:
            idx += 1
        return idx


    def _audit_vision_token_ids() -> set:
        tokenizer = processor.tokenizer
        ids = set()
        for owner in (processor, tokenizer):
            for attr in ("image_token_id", "image_token_index"):
                value = getattr(owner, attr, None)
                if isinstance(value, int) and value >= 0:
                    ids.add(value)
        unk_id = getattr(tokenizer, "unk_token_id", None)
        for token in ("<|image_pad|>", "<|image_token|>", "<image>"):
            try:
                token_id = tokenizer.convert_tokens_to_ids(token)
            except Exception:
                continue
            if isinstance(token_id, int) and token_id >= 0 and token_id != unk_id:
                ids.add(token_id)
        return ids


    def _audit_raw_sample(dataset, idx: int) -> Dict[str, Any]:
        if isinstance(dataset, Subset):
            return _audit_raw_sample(dataset.dataset, int(dataset.indices[idx]))
        samples = getattr(dataset, "samples", None)
        return samples[idx] if samples is not None else {}


    def _audit_build_test_dataset():
        test_path = resolve_split_path(DATA_ROOT, CFG["dataset"], "test")
        if test_path is None:
            return None
        dataset = OCRJsonlDataset(
            jsonl_path=test_path,
            image_root=IMAGE_ROOT,
            prompt_text=None,
            augment=False,
        )
        dataset = _exclude_sources_inplace(dataset, CFG["excluded_sources"], "Token audit/test")
        # Use the validation task policy for held-out evaluation.
        dataset = _exclude_tasks_inplace(dataset, CFG["excluded_valid_tasks"], "Token audit/test")
        actual_sources = set(_count_sources(_dataset_samples_for_stats(dataset)))
        expected_sources = set(CFG["expected_sources"])
        if actual_sources != expected_sources:
            raise ValueError(
                "Test sources differ from canonical scope: "
                f"missing={sorted(expected_sources - actual_sources)}, "
                f"unexpected={sorted(actual_sources - expected_sources)}"
            )
        actual_tasks = set(_count_tasks(_dataset_samples_for_stats(dataset)))
        if actual_tasks != {"page_htr"}:
            raise ValueError(f"Test split must contain only page_htr records, got {sorted(actual_tasks)}")
        expected_records = int(CFG["expected_split_records"]["test"])
        if len(dataset) != expected_records:
            raise ValueError(
                f"Test record count differs from canonical artifact: got={len(dataset)} "
                f"expected={expected_records}"
            )
        return dataset


    def _audit_split_datasets() -> Dict[str, Any]:
        result = {"train": train_dataset, "valid": valid_dataset}
        if "test" in TOKEN_AUDIT_SPLITS:
            test_dataset_for_audit = _audit_build_test_dataset()
            if test_dataset_for_audit is not None:
                result["test"] = test_dataset_for_audit
        return {name: result[name] for name in TOKEN_AUDIT_SPLITS if name in result}


    def _audit_one_batch(
        split: str,
        samples: List[Dict[str, Any]],
        raw_samples: List[Dict[str, Any]],
        collator: QwenVLMCollator,
        vision_token_ids: set,
    ) -> List[Dict[str, Any]]:
        full_texts = [collator._full_text(sample) for sample in samples]
        prompt_texts = [
            collator._apply_template(collator._build_prompt_messages(sample), True)
            for sample in samples
        ]
        images = [[sample["image"]] for sample in samples]

        # Raw sequences reveal the required budget. Limited sequences reproduce training.
        raw_full = processor(
            text=full_texts, images=images, do_resize=False, padding=True,
            truncation=False, return_tensors="pt",
        )
        raw_prompt = processor(
            text=prompt_texts, images=images, do_resize=False, padding=True,
            truncation=False, return_tensors="pt",
        )
        limited_full = processor(
            text=full_texts, images=images, do_resize=False, padding=True, truncation=True,
            max_length=TOKEN_AUDIT_MAX_LENGTH, return_tensors="pt",
        )
        limited_prompt = processor(
            text=prompt_texts, images=images, do_resize=False, padding=True, truncation=True,
            max_length=TOKEN_AUDIT_MAX_LENGTH, return_tensors="pt",
        )

        rows = []
        for row_idx, (sample, raw_sample) in enumerate(zip(samples, raw_samples)):
            raw_full_ids = _audit_nonpad_ids(raw_full, row_idx)
            raw_prompt_ids = _audit_nonpad_ids(raw_prompt, row_idx)
            limited_full_ids = _audit_nonpad_ids(limited_full, row_idx)
            limited_prompt_ids = _audit_nonpad_ids(limited_prompt, row_idx)

            prompt_prefix = _audit_common_prefix(raw_full_ids, raw_prompt_ids)
            prompt_is_prefix = prompt_prefix == len(raw_prompt_ids)
            raw_answer_start = len(raw_prompt_ids) if prompt_is_prefix else prompt_prefix
            raw_answer_ids = raw_full_ids[raw_answer_start:]

            # This mirrors the current collator: mask prompt_lens tokens from the
            # first non-pad position of the truncated full sequence.
            limited_label_start = min(len(limited_prompt_ids), len(limited_full_ids))
            kept_answer_ids = limited_full_ids[limited_label_start:]

            answer_prefix_kept = _audit_common_prefix(raw_answer_ids, kept_answer_ids)
            answer_suffix_kept = _audit_common_suffix(
                raw_answer_ids, kept_answer_ids, answer_prefix_kept
            )
            missing = max(0, len(raw_answer_ids) - len(kept_answer_ids))
            if missing == 0 and raw_full_ids == limited_full_ids:
                truncation_side = "none"
                missing_from_answer_index = None
                missing_to_answer_index = None
            elif kept_answer_ids == raw_answer_ids[:len(kept_answer_ids)]:
                truncation_side = "tail"
                missing_from_answer_index = len(kept_answer_ids)
                missing_to_answer_index = len(raw_answer_ids) - 1
            elif kept_answer_ids == raw_answer_ids[-len(kept_answer_ids):] if kept_answer_ids else False:
                truncation_side = "head"
                missing_from_answer_index = 0
                missing_to_answer_index = missing - 1
            else:
                truncation_side = "mixed_or_boundary_mismatch"
                missing_from_answer_index = answer_prefix_kept
                missing_to_answer_index = max(
                    answer_prefix_kept,
                    len(raw_answer_ids) - answer_suffix_kept - 1,
                )

            meta = sample.get("metadata", {}) or {}
            image_rel = (raw_sample.get("images") or [f"{raw_sample.get('id', '')}.png"])[0]
            raw_vision = sum(token_id in vision_token_ids for token_id in raw_full_ids)
            limited_vision = sum(token_id in vision_token_ids for token_id in limited_full_ids)
            coverage = 1.0 if not raw_answer_ids else min(1.0, len(kept_answer_ids) / len(raw_answer_ids))
            raw_terminal_id = int(raw_answer_ids[-1]) if raw_answer_ids else None
            kept_terminal_id = int(kept_answer_ids[-1]) if kept_answer_ids else None
            reference_loop_ids = (
                raw_answer_ids[:-1]
                if raw_answer_ids and raw_terminal_id == int(collator.training_terminal_id)
                else raw_answer_ids
            )
            reference_loop_boundary = (
                _rolling_ngram_loop_boundary(reference_loop_ids)
                if str(meta.get("task") or "unknown") == "page_htr" else None
            )

            rows.append({
                "split": split,
                "id": str(sample.get("id", "")),
                "source": str(meta.get("source_dataset") or meta.get("dataset") or "unknown"),
                "task": str(meta.get("task") or "unknown"),
                "language": str(meta.get("language") or "unknown"),
                "image": str(image_rel),
                "max_length": TOKEN_AUDIT_MAX_LENGTH,
                "raw_full_tokens": len(raw_full_ids),
                "limited_full_tokens": len(limited_full_ids),
                "raw_prompt_tokens": len(raw_prompt_ids),
                "limited_prompt_tokens": len(limited_prompt_ids),
                "vision_tokens_raw": raw_vision if vision_token_ids else -1,
                "vision_tokens_limited": limited_vision if vision_token_ids else -1,
                "answer_tokens_required": len(raw_answer_ids),
                "label_tokens_kept": len(kept_answer_ids),
                "label_tokens_missing": missing,
                "label_token_coverage": round(coverage, 6),
                "raw_answer_start_position": raw_answer_start,
                "raw_answer_end_position": len(raw_full_ids) - 1 if raw_answer_ids else None,
                "limited_label_start_position": limited_label_start if kept_answer_ids else None,
                "limited_label_end_position": len(limited_full_ids) - 1 if kept_answer_ids else None,
                "missing_from_answer_index": missing_from_answer_index,
                "missing_to_answer_index": missing_to_answer_index,
                "first_missing_full_position": (
                    raw_answer_start + missing_from_answer_index
                    if missing_from_answer_index is not None else None
                ),
                "truncated": bool(raw_full_ids != limited_full_ids),
                "truncation_side": truncation_side,
                "zero_label_tokens": len(kept_answer_ids) == 0,
                "raw_terminal_label_id": raw_terminal_id,
                "kept_terminal_label_id": kept_terminal_id,
                "terminal_matches_training_eos": kept_terminal_id == int(collator.training_terminal_id),
                "prompt_is_exact_full_prefix": prompt_is_prefix,
                "reference_loop_guard_triggered": reference_loop_boundary is not None,
                "reference_loop_guard_boundary": reference_loop_boundary,
                "processor_exceeds_max_length": len(limited_full_ids) > TOKEN_AUDIT_MAX_LENGTH,
                "target_characters": len(str(sample.get("target", ""))),
                "original_width": int(sample["original_size"][0]),
                "original_height": int(sample["original_size"][1]),
                "resized_width": int(sample["resized_size"][0]),
                "resized_height": int(sample["resized_size"][1]),
                "resized_pixels": int(sample["resized_size"][0] * sample["resized_size"][1]),
                "image_over_max_pixels": int(sample["resized_size"][0] * sample["resized_size"][1]) > int(CFG["image_max_pixels"]),
                "image_below_min_pixels": int(sample["resized_size"][0] * sample["resized_size"][1]) < int(CFG["image_min_pixels"]),
                "error": "",
            })
        return rows


    def _audit_error_row(split: str, raw_sample: Dict[str, Any], exc: Exception) -> Dict[str, Any]:
        meta = raw_sample.get("metadata", {}) or {}
        image_rel = (raw_sample.get("images") or [f"{raw_sample.get('id', '')}.png"])[0]
        return {
            "split": split,
            "id": str(raw_sample.get("id", "")),
            "source": str(meta.get("source_dataset") or meta.get("dataset") or "unknown"),
            "task": str(meta.get("task") or "unknown"),
            "language": str(meta.get("language") or "unknown"),
            "image": str(image_rel),
            "error": f"{type(exc).__name__}: {exc}",
        }


    def _audit_percentile(values: List[float], q: float):
        if not values:
            return None
        ordered = sorted(values)
        pos = (len(ordered) - 1) * q
        lo, hi = int(math.floor(pos)), int(math.ceil(pos))
        if lo == hi:
            return ordered[lo]
        return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


    def _audit_group_summary(rows: List[Dict[str, Any]], keys: Tuple[str, ...]) -> List[Dict[str, Any]]:
        groups = defaultdict(list)
        for row in rows:
            groups[tuple(row.get(key, "") for key in keys)].append(row)

        result = []
        for group_key, group_rows in sorted(groups.items()):
            valid = [row for row in group_rows if not row.get("error")]
            required = sum(int(row.get("answer_tokens_required", 0)) for row in valid)
            kept = sum(int(row.get("label_tokens_kept", 0)) for row in valid)
            summary = {key: value for key, value in zip(keys, group_key)}
            summary.update({
                "records": len(group_rows),
                "errors": len(group_rows) - len(valid),
                "truncated_records": sum(bool(row.get("truncated")) for row in valid),
                "truncated_rate": round(
                    sum(bool(row.get("truncated")) for row in valid) / max(1, len(valid)), 6
                ),
                "zero_label_records": sum(bool(row.get("zero_label_tokens")) for row in valid),
                "terminal_mismatch_records": sum(
                    not bool(row.get("terminal_matches_training_eos")) for row in valid
                ),
                "prompt_prefix_mismatches": sum(
                    not bool(row.get("prompt_is_exact_full_prefix")) for row in valid
                ),
                "processor_over_max_records": sum(
                    bool(row.get("processor_exceeds_max_length")) for row in valid
                ),
                "image_over_max_pixel_records": sum(
                    bool(row.get("image_over_max_pixels")) for row in valid
                ),
                "image_below_min_pixel_records": sum(
                    bool(row.get("image_below_min_pixels")) for row in valid
                ),
                "reference_loop_guard_triggers": sum(
                    bool(row.get("reference_loop_guard_triggered")) for row in valid
                ),
                "tail_truncations": sum(row.get("truncation_side") == "tail" for row in valid),
                "head_truncations": sum(row.get("truncation_side") == "head" for row in valid),
                "mixed_truncations": sum(
                    row.get("truncation_side") == "mixed_or_boundary_mismatch" for row in valid
                ),
                "micro_label_token_coverage": round(kept / max(1, required), 6),
                "min_label_token_coverage": min(
                    (row.get("label_token_coverage", 1.0) for row in valid), default=None
                ),
                "raw_full_tokens_p50": _audit_percentile(
                    [row["raw_full_tokens"] for row in valid], 0.50
                ),
                "raw_full_tokens_p95": _audit_percentile(
                    [row["raw_full_tokens"] for row in valid], 0.95
                ),
                "raw_full_tokens_max": max(
                    (row["raw_full_tokens"] for row in valid), default=None
                ),
                "answer_tokens_p50": _audit_percentile(
                    [row["answer_tokens_required"] for row in valid], 0.50
                ),
                "answer_tokens_p95": _audit_percentile(
                    [row["answer_tokens_required"] for row in valid], 0.95
                ),
                "label_tokens_kept_min": min(
                    (row["label_tokens_kept"] for row in valid), default=None
                ),
                "vision_tokens_p50": _audit_percentile(
                    [row["vision_tokens_raw"] for row in valid if row["vision_tokens_raw"] >= 0],
                    0.50,
                ),
                "vision_tokens_max": max(
                    (row["vision_tokens_raw"] for row in valid if row["vision_tokens_raw"] >= 0),
                    default=None,
                ),
                "resized_pixels_p95": _audit_percentile(
                    [row["resized_pixels"] for row in valid], 0.95
                ),
                "resized_pixels_max": max(
                    (row["resized_pixels"] for row in valid), default=None
                ),
            })
            result.append(summary)
        return result


    def _audit_write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
        import csv
        if not rows:
            return
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


    def run_full_token_audit() -> Dict[str, Any]:
        import gc
        TOKEN_AUDIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        detail_path = TOKEN_AUDIT_OUTPUT_DIR / "token_audit_rows.jsonl"
        split_datasets = _audit_split_datasets()
        collator = QwenVLMCollator(processor, max_text_length=TOKEN_AUDIT_MAX_LENGTH)
        vision_token_ids = _audit_vision_token_ids()
        print("Vision token ids used for counting:", sorted(vision_token_ids) or "not discovered")
        print("Token audit output:", TOKEN_AUDIT_OUTPUT_DIR)

        all_rows = []
        with detail_path.open("w", encoding="utf-8") as detail_file:
            for split, dataset in split_datasets.items():
                limit = len(dataset)
                if TOKEN_AUDIT_MAX_SAMPLES > 0:
                    limit = min(limit, TOKEN_AUDIT_MAX_SAMPLES)
                progress = tqdm(range(0, limit, TOKEN_AUDIT_BATCH_SIZE), desc=f"Token audit/{split}")
                for start in progress:
                    indices = list(range(start, min(start + TOKEN_AUDIT_BATCH_SIZE, limit)))
                    raw_samples = [_audit_raw_sample(dataset, idx) for idx in indices]
                    loaded_samples = []
                    try:
                        loaded_samples = [dataset[idx] for idx in indices]
                        rows = _audit_one_batch(
                            split, loaded_samples, raw_samples, collator, vision_token_ids
                        )
                    except Exception as exc:
                        rows = [_audit_error_row(split, raw_sample, exc) for raw_sample in raw_samples]
                    for row in rows:
                        all_rows.append(row)
                        detail_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    detail_file.flush()
                    for sample in loaded_samples:
                        image = sample.get("image")
                        if image is not None:
                            try:
                                image.close()
                            except Exception:
                                pass
                gc.collect()

        by_source = _audit_group_summary(all_rows, ("split", "source"))
        by_source_task = _audit_group_summary(all_rows, ("split", "source", "task"))
        overall = _audit_group_summary(all_rows, ("split",))
        summary = {
            "config": {
                "max_length": TOKEN_AUDIT_MAX_LENGTH,
                "image_resize_policy": str(CFG["image_resize_policy"]),
                "image_resize_factor": int(CFG["image_resize_factor"]),
                "image_max_aspect_ratio": float(CFG["image_max_aspect_ratio"]),
                "image_min_pixels": int(CFG["image_min_pixels"]),
                "image_max_pixels": int(CFG["image_max_pixels"]),
                "processor_image_size": dict(CFG.get("processor_image_size", {})),
                "batch_size": TOKEN_AUDIT_BATCH_SIZE,
                "max_samples_per_split": TOKEN_AUDIT_MAX_SAMPLES,
                "excluded_sources": list(CFG.get("excluded_sources", [])),
                "excluded_train_tasks": list(CFG.get("excluded_train_tasks", [])),
                "excluded_valid_tasks": list(CFG.get("excluded_valid_tasks", [])),
                "vision_token_ids": sorted(vision_token_ids),
                "atomic_ocr_marker_ids": dict(CFG.get("atomic_ocr_marker_ids", {})),
                "loop_guard": {
                    "ngram": int(CFG["loop_ngram"]), "window": int(CFG["loop_window"]),
                    "threshold": float(CFG["loop_threshold"]), "min_tokens": int(CFG["loop_min_tokens"]),
                    "consecutive": int(CFG["loop_consecutive"]),
                },
            },
            "overall_by_split": overall,
            "by_source": by_source,
            "by_source_task": by_source_task,
        }
        with (TOKEN_AUDIT_OUTPUT_DIR / "token_audit_summary.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        _audit_write_csv(TOKEN_AUDIT_OUTPUT_DIR / "token_audit_by_source.csv", by_source)
        _audit_write_csv(
            TOKEN_AUDIT_OUTPUT_DIR / "token_audit_by_source_task.csv", by_source_task
        )

        worst = sorted(
            (row for row in all_rows if not row.get("error")),
            key=lambda row: (
                row.get("label_token_coverage", 1.0),
                -row.get("label_tokens_missing", 0),
            ),
        )[:200]
        with (TOKEN_AUDIT_OUTPUT_DIR / "token_audit_worst_200.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in worst:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        print("\nToken audit summary by split/source")
        print(
            f"{'split':<7} {'source':<24} {'n':>8} {'trunc%':>8} "
            f"{'zero':>6} {'coverage':>10} {'p95 full':>10} {'max full':>10} {'max px':>10}"
        )
        for row in by_source:
            print(
                f"{row['split']:<7} {row['source']:<24} {row['records']:>8} "
                f"{100.0 * row['truncated_rate']:>7.2f}% {row['zero_label_records']:>6} "
                f"{row['micro_label_token_coverage']:>10.4f} "
                f"{str(round(row['raw_full_tokens_p95'], 1) if row['raw_full_tokens_p95'] is not None else '-'):>10} "
                f"{str(row['raw_full_tokens_max']):>10} "
                f"{str(row['resized_pixels_max']):>10}"
            )
        print("\nFiles written:")
        for name in (
            "token_audit_rows.jsonl",
            "token_audit_summary.json",
            "token_audit_by_source.csv",
            "token_audit_by_source_task.csv",
            "token_audit_worst_200.jsonl",
        ):
            print(" -", TOKEN_AUDIT_OUTPUT_DIR / name)
        return summary


    if RUN_FULL_TOKEN_AUDIT:
        TOKEN_AUDIT_SUMMARY = run_full_token_audit()
    else:
        print(
            "Full token audit is disabled. Set RUN_FULL_TOKEN_AUDIT=True and run this "
            "cell before training. It scans all retained train/valid/test records."
        )

    return export_runtime_locals(state, locals())
