"""evaluation.final_test phase for the ExpertHTR research workflow.

This module preserves the audited runtime computation. It is invoked by the
explicit CLI workflow and exports its declared runtime objects into a
:class:`~experthtr.runtime.context.RuntimeState`.
"""

from __future__ import annotations

from experthtr.runtime.context import RuntimeState, export_runtime_locals


def run_final_test(state: RuntimeState) -> RuntimeState:
    """Execute this phase after its prerequisites have populated ``state``."""

    globals().update(state.values)
    # =============================================================================
    # FINAL TEST — validation-selected checkpoint only
    # =============================================================================
    # Leave empty to use output_dir/run_summary.json -> best_checkpoint.
    # Set this only to an exact checkpoint directory when reproducing a prior run.
    FINAL_TEST_CHECKPOINT_DIR = os.getenv("HWVLM_FINAL_TEST_CHECKPOINT", "").strip()
    FINAL_TEST_MAX_NEW_TOKENS = int(CFG["eval_max_new_tokens"])
    # Fixed one-page generation is required for comparable latency (s/page) by
    # source. Training batch size measures throughput, not per-page latency.
    FINAL_TEST_BATCH_SIZE = 1
    FINAL_TEST_PREDICTIONS_NAME = "test_predictions_validation_selected.jsonl"
    FINAL_TEST_METRICS_NAME = "test_metrics_validation_selected.json"
    FINAL_TEST_BY_SOURCE_REPORT_NAME = "test_metrics_by_source_validation_selected.csv"
    if FINAL_TEST_BATCH_SIZE != 1:
        raise ValueError("Final test uses batch_size=1 so reported s/page is comparable latency.")

    if "output_dir" not in globals() or "model" not in globals():
        raise RuntimeError(
            "Final test requires a constructed model, processor, and output directory. "
            "Invoke it through `experthtr final-test`."
        )

    output_dir = Path(output_dir)
    if FINAL_TEST_CHECKPOINT_DIR.strip():
        final_test_checkpoint = Path(FINAL_TEST_CHECKPOINT_DIR).expanduser()
    else:
        summary_path = output_dir / "run_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"Missing {summary_path}. Complete training first, or set FINAL_TEST_CHECKPOINT_DIR explicitly."
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        chosen = summary.get("best_checkpoint") or {}
        chosen_path = chosen.get("path") if isinstance(chosen, dict) else None
        if not chosen_path:
            raise RuntimeError("run_summary.json has no validation-selected best_checkpoint.")
        final_test_checkpoint = Path(chosen_path)

    for required in (CHECKPOINT_COMPLETE_FILE, CHECKPOINT_MANIFEST_FILE, FULL_MODEL_STATE_FILE):
        if not (final_test_checkpoint / required).exists():
            raise FileNotFoundError(final_test_checkpoint / required)

    final_manifest = json.loads(
        (final_test_checkpoint / CHECKPOINT_MANIFEST_FILE).read_text(encoding="utf-8")
    )
    if dict(final_manifest.get("moe_signature") or {}) != _moe_signature():
        raise RuntimeError(
            "Final-test checkpoint architecture differs from the currently constructed model. "
            f"checkpoint={final_manifest.get('moe_signature')} current={_moe_signature()}"
        )
    if list(final_manifest.get("atomic_ocr_markers") or []) != list(CFG["atomic_ocr_markers"]):
        raise RuntimeError("Final-test checkpoint atomic-marker policy differs from the active runtime.")

    try:
        checkpoint_payload = torch.load(
            final_test_checkpoint / FULL_MODEL_STATE_FILE,
            map_location="cpu", weights_only=True,
        )
    except TypeError:
        checkpoint_payload = torch.load(final_test_checkpoint / FULL_MODEL_STATE_FILE, map_location="cpu")
    checkpoint_state = checkpoint_payload.get("model_state_dict", checkpoint_payload)
    model.load_state_dict(checkpoint_state, strict=True)
    del checkpoint_payload, checkpoint_state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    final_test_path = resolve_split_path(DATA_ROOT, CFG["dataset"], "test")
    if final_test_path is None:
        raise FileNotFoundError(f"No test JSONL found for dataset={CFG['dataset']!r} under {DATA_ROOT}")
    final_test_dataset = OCRJsonlDataset(
        jsonl_path=final_test_path,
        image_root=IMAGE_ROOT,
        prompt_text=None,
        augment=False,
    )
    final_test_dataset = _exclude_sources_inplace(
        final_test_dataset, CFG["excluded_sources"], "Final test"
    )
    final_test_dataset = _exclude_tasks_inplace(
        final_test_dataset, CFG["excluded_valid_tasks"], "Final test"
    )
    final_test_samples = _dataset_samples_for_stats(final_test_dataset)
    final_test_sources = set(_count_sources(final_test_samples))
    final_test_tasks = set(_count_tasks(final_test_samples))
    if final_test_sources != set(CFG["expected_sources"]):
        raise ValueError(
            f"Final-test source scope differs: got={sorted(final_test_sources)} "
            f"expected={sorted(CFG['expected_sources'])}"
        )
    if final_test_tasks != {"page_htr"}:
        raise ValueError(
            f"Final test must contain only page_htr records, got={sorted(final_test_tasks)}"
        )
    expected_final_test_records = int(CFG["expected_split_records"]["test"])
    if len(final_test_dataset) != expected_final_test_records:
        raise ValueError(
            f"Final-test record count differs: got={len(final_test_dataset)} "
            f"expected={expected_final_test_records}"
        )
    final_test_loader = DataLoader(
        final_test_dataset,
        batch_size=FINAL_TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=int(CFG["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
        persistent_workers=bool(int(CFG["num_workers"]) > 0),
    )

    final_test_predictions = output_dir / FINAL_TEST_PREDICTIONS_NAME
    final_test_metrics = evaluate_loader(
        model,
        final_test_loader,
        processor.tokenizer,
        max_new_tokens=FINAL_TEST_MAX_NEW_TOKENS,
        output_jsonl_path=final_test_predictions,
        eval_context={
            "split": "test",
            "checkpoint": str(final_test_checkpoint),
            "selection_rule": "minimum_validation_micro_cer",
            "max_new_tokens": int(FINAL_TEST_MAX_NEW_TOKENS),
        },
    )
    final_test_metrics.update({
        "split": "test",
        "checkpoint": str(final_test_checkpoint),
        "selection_rule": "minimum_validation_micro_cer",
        "prediction_file": str(final_test_predictions),
    })

    # A compact, paper-ready source report. The full JSON retains distributions
    # (p50/p95/max) and per-page predictions; this CSV exposes all scalar metrics
    # needed to compare datasets without parsing nested dictionaries.
    import csv
    final_test_by_source_report = output_dir / FINAL_TEST_BY_SOURCE_REPORT_NAME
    def _diag_mean(diag, key):
        value = diag.get(key, {})
        return value.get("mean") if isinstance(value, dict) else None

    source_report_rows = []
    for source, source_metrics in sorted(final_test_metrics.get("by_source", {}).items()):
        source_diag = final_test_metrics.get("diagnostics_by_source", {}).get(source, {})
        safety_cer = final_test_metrics.get("safety_cer_by_source_diagnostic", {}).get(source)
        source_report_rows.append({
            "source": source,
            "samples": int(source_metrics.get("num_samples", 0)),
            "cer": float(source_metrics.get("cer_generate", 0.0)),
            "wer": float(source_metrics.get("wer_generate", 0.0)),
            "safety_cer": None if safety_cer is None else float(safety_cer),
            "seconds_per_page": _diag_mean(source_diag, "generation_seconds_per_page"),
            "seconds_per_page_p50": source_diag.get("generation_seconds_per_page", {}).get("p50"),
            "seconds_per_page_p95": source_diag.get("generation_seconds_per_page", {}).get("p95"),
            "generation_wall_seconds_total": source_diag.get("generation_wall_seconds_total"),
            "pages_per_second": source_diag.get("generation_pages_per_second"),
            "generated_tokens_per_second": source_diag.get("generated_tokens_per_second"),
            "output_tokens_mean": _diag_mean(source_diag, "output_tokens"),
            "reference_tokens_mean": _diag_mean(source_diag, "reference_tokens"),
            "output_reference_token_ratio_mean": _diag_mean(source_diag, "output_reference_token_ratio"),
            "eos_rate": source_diag.get("eos_emitted_rate"),
            "max_new_tokens_hit_rate": source_diag.get("max_new_tokens_hit_rate"),
            "empty_output_rate": source_diag.get("empty_output_rate"),
            "loop_guard_rate": source_diag.get("loop_guard_trigger_rate"),
            "repetition_3gram_fraction_mean": _diag_mean(source_diag, "repetition_3gram_fraction"),
            "region_count_exact_rate": source_diag.get("region_count_exact_rate"),
            "line_count_exact_rate": source_diag.get("line_count_exact_rate"),
            "del_reference_count": int(source_diag.get("marker_reference_counts", {}).get("<del>", 0)),
            "del_prediction_count": int(source_diag.get("marker_prediction_counts", {}).get("<del>", 0)),
            "gap_reference_count": int(source_diag.get("marker_reference_counts", {}).get("<gap>", 0)),
            "gap_prediction_count": int(source_diag.get("marker_prediction_counts", {}).get("<gap>", 0)),
        })
    if not source_report_rows:
        raise RuntimeError("Final test produced no source-level metrics.")
    with final_test_by_source_report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_report_rows[0]))
        writer.writeheader()
        writer.writerows(source_report_rows)
    final_test_metrics["by_source_report_csv"] = str(final_test_by_source_report)
    (output_dir / FINAL_TEST_METRICS_NAME).write_text(
        json.dumps(final_test_metrics, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    diagnostics = final_test_metrics.get("generation_diagnostics", {})
    LOGGER.info(
        "FINAL TEST | checkpoint=%s | CER=%.2f%% | WER=%.2f%% | n=%d | gen=%.3fs/page %.1f page/s %.1f tok/s | eos=%.2f%% | max_hit=%.2f%%",
        final_test_checkpoint,
        float(final_test_metrics["cer_generate"]) * 100.0,
        float(final_test_metrics.get("wer_generate", 0.0)) * 100.0,
        int(final_test_metrics["num_samples"]),
        float(diagnostics.get("generation_seconds_per_page", {}).get("mean", 0.0)),
        float(diagnostics.get("generation_pages_per_second", 0.0)),
        float(diagnostics.get("generated_tokens_per_second", 0.0)),
        float(diagnostics.get("eos_emitted_rate", 0.0)) * 100.0,
        float(diagnostics.get("max_new_tokens_hit_rate", 0.0)) * 100.0,
    )
    for source, source_metrics in sorted(final_test_metrics.get("by_source", {}).items()):
        source_diag = final_test_metrics.get("diagnostics_by_source", {}).get(source, {})
        LOGGER.info(
            "  TEST source=%s | n=%d | CER=%.2f%% WER=%.2f%% | gen=%.3fs/page %.1f page/s | eos=%.2f%% | max_hit=%.2f%% | loop=%.2f%%",
            source,
            int(source_metrics.get("num_samples", 0)),
            float(source_metrics.get("cer_generate", 0.0)) * 100.0,
            float(source_metrics.get("wer_generate", 0.0)) * 100.0,
            float(source_diag.get("generation_seconds_per_page", {}).get("mean", 0.0)),
            float(source_diag.get("generation_pages_per_second", 0.0)),
            float(source_diag.get("eos_emitted_rate", 0.0)) * 100.0,
            float(source_diag.get("max_new_tokens_hit_rate", 0.0)) * 100.0,
            float(source_diag.get("loop_guard_trigger_rate", 0.0)) * 100.0,
        )
    print("Test predictions:", final_test_predictions)
    print("Test metrics:", output_dir / FINAL_TEST_METRICS_NAME)
    print("Test source report:", final_test_by_source_report)
    print(json.dumps({
        "CER": final_test_metrics["cer_generate"],
        "WER": final_test_metrics.get("wer_generate"),
        "samples": final_test_metrics["num_samples"],
        "seconds_per_page": diagnostics.get("generation_seconds_per_page", {}).get("mean"),
        "pages_per_second": diagnostics.get("generation_pages_per_second"),
        "tokens_per_second": diagnostics.get("generated_tokens_per_second"),
        "eos_rate": diagnostics.get("eos_emitted_rate"),
        "max_token_hit_rate": diagnostics.get("max_new_tokens_hit_rate"),
        "loop_guard_rate": diagnostics.get("loop_guard_trigger_rate"),
        "checkpoint": str(final_test_checkpoint),
    }, ensure_ascii=False, indent=2))

    return export_runtime_locals(state, locals())
