"""training.checkpointing phase for the ExpertHTR research workflow.

This module preserves the audited runtime computation. It is invoked by the
explicit CLI workflow and exports its declared runtime objects into a
:class:`~experthtr.runtime.context.RuntimeState`.
"""

from __future__ import annotations

from experthtr.runtime.context import RuntimeState, export_runtime_locals


def register_checkpointing(state: RuntimeState) -> RuntimeState:
    """Execute this phase after its prerequisites have populated ``state``."""

    globals().update(state.values)
    import shutil
    from datetime import datetime, timezone


    def _capture_rng_state() -> Dict[str, Any]:
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
            "sampler_generator": SAMPLER_GENERATOR.get_state(),
            "loader_generator": LOADER_GENERATOR.get_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state


    def _moe_signature() -> Dict[str, Any]:
        """Complete signature; no two ablation arms may share it accidentally."""
        return {
            "enabled": bool(CFG["use_moe_mlp"]),
            "ablation_mode": str(CFG.get("moe_ablation_mode", "unknown")),
            "routing_grouping": str(CFG["routing_grouping"]),
            "language_groups": list(LANGUAGE_GROUPS),
            "language_experts": int(CFG["moe_mlp_language_experts"]),
            "shared_experts": int(CFG["moe_mlp_shared_experts"]),
            "shared_mix": float(CFG["moe_mlp_shared_mix"]),
            "layer_indices": list(CFG["moe_mlp_layer_indices"]),
            "routing_fn": str(CFG["moe_mlp_routing_fn"]),
            "router_ce_coef": float(CFG["moe_mlp_router_ce_coef"]),
            "router_ce_class_weighting": str(CFG["router_ce_class_weighting"]),
            "load_balance_coef": float(CFG["moe_mlp_load_balance_coef"]),
            "load_balance_warmup_fraction": float(CFG.get("moe_mlp_load_balance_warmup_fraction", 0.0)),
            "load_balance_warmup_updates": int(CFG.get("moe_mlp_load_balance_warmup_updates", 0)),
            "lam_sparse_coef": float(CFG.get("moe_mlp_lam_sparse_coef", 0.0)),
            "lam_sparse_warmup_fraction": float(CFG.get("moe_mlp_lam_sparse_warmup_fraction", 0.0)),
            "router_temperature": float(CFG["moe_mlp_router_temperature"]),
            "sparsegen_hidden": int(CFG["moe_mlp_sparsegen_hidden"]),
            "sparsegen_eps": float(CFG["moe_mlp_sparsegen_eps"]),
            "expert_init": str(CFG["moe_mlp_expert_init"]),
            "expert_train_dtype": str(CFG["moe_mlp_expert_train_dtype"]),
            "freeze_base": bool(CFG["moe_mlp_freeze_base"]),
        }


    def _atomic_replace_directory(temp_dir: Path, final_dir: Path) -> None:
        backup_dir = final_dir.parent / f".{final_dir.name}.backup"
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        try:
            if final_dir.exists():
                final_dir.replace(backup_dir)
            temp_dir.replace(final_dir)
        except Exception:
            if not final_dir.exists() and backup_dir.exists():
                backup_dir.replace(final_dir)
            raise
        if backup_dir.exists():
            shutil.rmtree(backup_dir)


    def save_checkpoint(
        model,
        processor,
        checkpoint_dir: Path,
        metadata: Dict[str, Any],
        optimizer=None,
        scheduler=None,
        include_training_state: bool = False,
    ):
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
        temp_dir = checkpoint_dir.parent / f".{checkpoint_dir.name}.tmp"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True)

        model_to_save = unwrap_model(model)
        torch.save(
            {"model_state_dict": model_to_save.state_dict()},
            temp_dir / FULL_MODEL_STATE_FILE,
        )
        processor.save_pretrained(temp_dir)
        if hasattr(model_to_save.base_model, "config"):
            model_to_save.base_model.config.save_pretrained(temp_dir)

        manifest = {
            "format_version": 2,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "base_model": str(CFG["llm_name"]),
            "moe_signature": _moe_signature(),
            "atomic_ocr_markers": list(CFG.get("atomic_ocr_markers", [])),
            "atomic_ocr_marker_ids": dict(CFG.get("atomic_ocr_marker_ids", {})),
            "full_model_state_file": FULL_MODEL_STATE_FILE,
            "training_state_included": bool(include_training_state),
            "dense_initialization": dict(CFG.get("dense_initialization", {})),
        }
        (temp_dir / CHECKPOINT_MANIFEST_FILE).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (temp_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        (temp_dir / "metrics.json").write_text(
            json.dumps(metadata.get("val_metrics", {}), ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

        if include_training_state:
            training_state = {
                "epoch": int(metadata.get("epoch", 0)),
                "epoch_completed": bool(metadata.get("epoch_completed", False)),
                "global_step": int(metadata.get("global_step", 0)),
                "best_metric": metadata.get("best_metric"),
                "evaluations_no_improve": int(metadata.get("evaluations_no_improve", 0)),
                "validation_has_started": bool(metadata.get("validation_has_started", False)),
                "top_checkpoints": list(metadata.get("top_checkpoints", [])),
                "rng_state": _capture_rng_state(),
                "cfg": dict(CFG),
                "optimizer": optimizer.state_dict() if optimizer is not None else None,
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
            }
            torch.save(training_state, temp_dir / "training_state.pt")

        (temp_dir / CHECKPOINT_COMPLETE_FILE).write_text("ok\n", encoding="utf-8")
        _atomic_replace_directory(temp_dir, checkpoint_dir)


    def _build_dataset_config_for_checkpoint() -> Dict[str, Any]:
        test_path = resolve_split_path(DATA_ROOT, CFG["dataset"], "test")
        return {
            "dataset_name": str(CFG["dataset"]),
            "data_root": str(DATA_ROOT),
            "train_json_path": str(train_path),
            "valid_json_path": str(valid_path),
            "test_json_path": str(test_path) if test_path is not None else None,
            "image_root": str(IMAGE_ROOT),
            "prompt_text": str(CFG["prompt_text"]),
            "expected_sources": list(CFG["expected_sources"]),
            "excluded_sources": list(CFG["excluded_sources"]),
            "excluded_train_tasks": list(CFG["excluded_train_tasks"]),
            "excluded_valid_tasks": list(CFG["excluded_valid_tasks"]),
            "structure_task_policy": "page_total_physical_line_count_v3",
            "image_resize_policy": str(CFG["image_resize_policy"]),
            "image_resize_factor": int(CFG["image_resize_factor"]),
            "image_max_aspect_ratio": float(CFG["image_max_aspect_ratio"]),
            "image_min_pixels": int(CFG["image_min_pixels"]),
            "image_max_pixels": int(CFG["image_max_pixels"]),
            "resolution_profile": str(CFG["resolution_profile"]),
            "max_text_length": int(CFG["max_text_length"]),
            "processor_image_size": dict(CFG.get("processor_image_size", {})),
            "sampling_mode": str(CFG["sampling_mode"]),
            "microbatch_invariant_token_loss": bool(CFG["microbatch_invariant_token_loss"]),
            "training_terminal_id": int(CFG["training_terminal_id"]),
            "generation_stop_ids": list(CFG["generation_stop_ids"]),
            "atomic_ocr_marker_ids": dict(CFG.get("atomic_ocr_marker_ids", {})),
            "page_htr_eos_weight": float(CFG["page_htr_eos_weight"]),
            "token_ul_coef": float(CFG["token_ul_coef"]),
            "token_ul_page_fraction": float(CFG["token_ul_page_fraction"]),
            "loop_guard": {
                "ngram": int(CFG["loop_ngram"]), "window": int(CFG["loop_window"]),
                "threshold": float(CFG["loop_threshold"]), "min_tokens": int(CFG["loop_min_tokens"]),
                "consecutive": int(CFG["loop_consecutive"]),
            },
        }


    def _build_eval_steps() -> List[int]:
        explicit = str(CFG.get("eval_epochs", "")).strip()
        if explicit:
            try:
                epoch_points = {round(float(value.strip()), 8) for value in explicit.split(",") if value.strip()}
            except ValueError as exc:
                raise ValueError(f"Invalid HWVLM_EVAL_EPOCHS={explicit!r}; use comma-separated positive epoch values.") from exc
            if not epoch_points or any(value <= 0 or value > float(CFG["num_epochs"]) for value in epoch_points):
                raise ValueError(
                    f"HWVLM_EVAL_EPOCHS must be non-empty and within (0, {CFG['num_epochs']}], got {sorted(epoch_points)}"
                )
        else:
            epoch_points = set()
            coarse = max(float(CFG["eval_coarse_fraction"]), 1e-6)
            dense = max(float(CFG["eval_dense_fraction"]), 1e-6)

            index = 1
            while index * coarse <= CFG["num_epochs"] + 1e-9:
                epoch_points.add(round(index * coarse, 8))
                index += 1

            dense_start = max(0.0, float(CFG["eval_dense_start_epoch"]))
            dense_end = min(float(CFG["num_epochs"]), float(CFG["eval_dense_end_epoch"]))
            index = 0
            while dense_start + index * dense <= dense_end + 1e-9:
                epoch_points.add(round(dense_start + index * dense, 8))
                index += 1

        points = {
            max(1, int(math.floor(epoch_value * updates_per_epoch + 0.5)))
            for epoch_value in epoch_points
        }
        points.add(total_update_steps)
        return sorted(points)


    def _log_validation_summary(global_step: int, epoch_progress: float, val_metrics: Dict[str, Any]) -> None:
        diagnostics = val_metrics.get("generation_diagnostics", {})
        LOGGER.info(
            "Validation step=%d epoch=%.3f | CER=%.2f%% WER=%.2f%% | samples=%d | "
            "out/ref tok=%.1f/%.1f | gen=%.3fs/page %.1f page/s %.1f tok/s | max_hit=%.2f%% | eos=%.2f%% | empty=%.2f%% | "
            "loop=%.2f%% safety_CER=%.2f%%",
            global_step,
            epoch_progress,
            float(val_metrics["cer_generate"]) * 100.0,
            float(val_metrics.get("wer_generate", 0.0)) * 100.0,
            int(val_metrics["num_samples"]),
            float(diagnostics.get("output_tokens", {}).get("mean", 0.0)),
            float(diagnostics.get("reference_tokens", {}).get("mean", 0.0)),
            float(diagnostics.get("generation_seconds_per_page", {}).get("mean", 0.0)),
            float(diagnostics.get("generation_pages_per_second", 0.0)),
            float(diagnostics.get("generated_tokens_per_second", 0.0)),
            float(diagnostics.get("max_new_tokens_hit_rate", 0.0)) * 100.0,
            float(diagnostics.get("eos_emitted_rate", 0.0)) * 100.0,
            float(diagnostics.get("empty_output_rate", 0.0)) * 100.0,
            float(diagnostics.get("loop_guard_trigger_rate", 0.0)) * 100.0,
            float(val_metrics.get("safety_cer_generate_diagnostic", 0.0)) * 100.0,
        )
        LOGGER.info(
            "  structure | region_exact=%.2f%% line_exact=%.2f%% | markers ref=%s pred=%s",
            float(diagnostics.get("region_count_exact_rate", 0.0)) * 100.0,
            float(diagnostics.get("line_count_exact_rate", 0.0)) * 100.0,
            diagnostics.get("marker_reference_counts", {}),
            diagnostics.get("marker_prediction_counts", {}),
        )
        for source, metrics in val_metrics.get("by_source", {}).items():
            source_diag = val_metrics.get("diagnostics_by_source", {}).get(source, {})
            LOGGER.info(
                "  source=%s | n=%d | CER=%.2f%% WER=%.2f%% | safety=%.2f%% | out/ref tok=%.1f/%.1f | gen=%.3fs/page %.1f page/s %.1f tok/s | max_hit=%.2f%% | loop=%.2f%%",
                source,
                int(metrics.get("num_samples", 0)),
                float(metrics.get("cer_generate", 0.0)) * 100.0,
                float(metrics.get("wer_generate", 0.0)) * 100.0,
                float(val_metrics.get("safety_cer_by_source_diagnostic", {}).get(source, 0.0)) * 100.0,
                float(source_diag.get("output_tokens", {}).get("mean", 0.0)),
                float(source_diag.get("reference_tokens", {}).get("mean", 0.0)),
                float(source_diag.get("generation_seconds_per_page", {}).get("mean", 0.0)),
                float(source_diag.get("generation_pages_per_second", 0.0)),
                float(source_diag.get("generated_tokens_per_second", 0.0)),
                float(source_diag.get("max_new_tokens_hit_rate", 0.0)) * 100.0,
                float(source_diag.get("loop_guard_trigger_rate", 0.0)) * 100.0,
            )
            LOGGER.info(
                "    structure region/line exact=%.2f%%/%.2f%% | markers ref=%s pred=%s",
                float(source_diag.get("region_count_exact_rate", 0.0)) * 100.0,
                float(source_diag.get("line_count_exact_rate", 0.0)) * 100.0,
                source_diag.get("marker_reference_counts", {}),
                source_diag.get("marker_prediction_counts", {}),
            )


    def _checkpoint_metadata(
        epoch_idx: int,
        epoch_progress: float,
        global_step: int,
        train_metrics: Dict[str, Any],
        val_metrics: Optional[Dict[str, Any]],
        checkpoint_dir: Path,
        epoch_completed: bool,
        run_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "stage": "canonical_v5_corrected_dense_moe_page_multitask",
            "epoch": int(epoch_idx),
            "epoch_progress": float(epoch_progress),
            "epoch_completed": bool(epoch_completed),
            "global_step": int(global_step),
            "llm_name": CFG["llm_name"],
            "resume_from_checkpoint": CFG["resume_from_checkpoint"],
            "dense_initialization": dict(CFG.get("dense_initialization", {})),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics or {},
            "checkpoint_dir": str(checkpoint_dir),
            "architecture": (
                "qwen3_5_vlm_script_grouped_supervised_mlp_moe"
                if CFG["use_moe_mlp"] else "qwen3_5_vlm_dense"
            ),
            "dataset_config": dataset_config_for_ckpt,
            "best_metric": (
                float(run_state["best_metric"])
                if math.isfinite(float(run_state["best_metric"]))
                else None
            ),
            "evaluations_no_improve": int(run_state["evaluations_no_improve"]),
            "validation_has_started": bool(run_state["validation_has_started"]),
            "top_checkpoints": list(run_state["top_checkpoints"]),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "loss_weighting": (
                "optimizer_window_eos_weighted_ce_plus_token_ul_plus_router_balance_and_lam_sparse"
                if CFG["use_moe_mlp"] else "optimizer_window_eos_weighted_ce_plus_token_ul"
            ),
            "moe_signature": _moe_signature(),
            "moe_mlp": {
                "routing_grouping": str(CFG["routing_grouping"]),
                "routing_group_description": str(CFG["routing_group_description"]),
                "routing_fn": str(CFG["moe_mlp_routing_fn"]),
                "router_temperature": float(CFG["moe_mlp_router_temperature"]),
                "load_balance_coef": float(CFG["moe_mlp_load_balance_coef"]),
                "load_balance_warmup_fraction": float(CFG.get("moe_mlp_load_balance_warmup_fraction", 0.0)),
                "load_balance_warmup_updates": int(CFG.get("moe_mlp_load_balance_warmup_updates", 0)),
                "router_ce_coef": float(CFG["moe_mlp_router_ce_coef"]),
                "lam_sparse_coef": float(CFG.get("moe_mlp_lam_sparse_coef", 0.0)),
                "lam_sparse_warmup_fraction": float(CFG.get("moe_mlp_lam_sparse_warmup_fraction", 0.0)),
                "lam_sparse_warmup_updates": int(CFG.get("moe_mlp_lam_sparse_warmup_updates", 0)),
                "shared_mix": float(CFG["moe_mlp_shared_mix"]),
                "hard_top_k": bool(CFG.get("moe_mlp_enforce_top_k", False)),
                "active_expert_budget": int(CFG["moe_mlp_active_expert_budget"]),
                "routed_top_k": int(CFG["moe_mlp_routed_top_k"]) if CFG["use_moe_mlp"] else 0,
                "freeze_base": bool(CFG["moe_mlp_freeze_base"]),
                "language_group_counts": dict(CFG.get("language_group_counts", {})),
                "language_group_class_weights": dict(CFG.get("language_group_class_weights", {})),
                "metadata_used_for_training_router_labels": bool(CFG["use_moe_mlp"] and CFG["moe_mlp_router_ce_coef"] > 0.0),
                "metadata_used_for_inference_routing": False,
            },
        }

    return export_runtime_locals(state, locals())
