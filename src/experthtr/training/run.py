"""training.run phase for the ExpertHTR research workflow.

This module preserves the audited runtime computation. It is invoked by the
explicit CLI workflow and exports its declared runtime objects into a
:class:`~experthtr.runtime.context.RuntimeState`.
"""

from __future__ import annotations

from experthtr.runtime.context import RuntimeState, export_runtime_locals


def run_training(state: RuntimeState) -> RuntimeState:
    """Execute this phase after its prerequisites have populated ``state``."""

    globals().update(state.values)
    dataset_config_for_ckpt = _build_dataset_config_for_checkpoint()
    output_dir = Path(CFG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "cfg": CFG,
                "dataset_config": dataset_config_for_ckpt,
                "updates_per_epoch": updates_per_epoch,
                "total_update_steps": total_update_steps,
                "scheduler_total_update_steps": scheduler_total_update_steps,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ) + "\n",
        encoding="utf-8",
    )

    eval_steps = set(_build_eval_steps())
    print("Full-validation optimizer steps:", sorted(eval_steps))
    best_metric_raw = resume_training_state.get("best_metric") if resume_training_state else None
    run_state = {
        "best_metric": float(best_metric_raw) if best_metric_raw is not None else float("inf"),
        "evaluations_no_improve": int(
            resume_training_state.get("evaluations_no_improve", 0)
            if resume_training_state else 0
        ),
        "validation_has_started": bool(
            resume_training_state.get("validation_has_started", False)
            if resume_training_state else False
        ),
        "top_checkpoints": list(
            resume_training_state.get("top_checkpoints", [])
            if resume_training_state else []
        ),
        "latest_val_metrics": {},
    }
    start_epoch = max(1, resume_completed_epochs + 1)
    global_step = int(resume_global_step)
    last_ckpt_dir = output_dir / "last_checkpoint"
    wandb_run = None
    if CFG["use_wandb"]:
        wandb_run = wandb.init(
            project=os.getenv("WANDB_PROJECT", "handwriting-vlm"),
            name=os.getenv("WANDB_RUN_NAME", CFG["run_name"]),
            config=CFG,
        )


    def _evaluate_and_maybe_checkpoint(
        global_step: int,
        epoch_progress: float,
        running_train_metrics: Dict[str, Any],
        grad_norm: float,
    ) -> bool:
        if global_step not in eval_steps:
            return False

        run_state["validation_has_started"] = True
        predictions_path = (
            output_dir / "validation_predictions" / f"step_{global_step:06d}.jsonl"
            if CFG["write_eval_predictions"] else None
        )
        val_metrics = evaluate_loader(
            model=model,
            data_loader=valid_loader,
            tokenizer=processor.tokenizer,
            max_new_tokens=CFG["eval_max_new_tokens"],
            max_eval_batches=CFG["eval_max_batches"],
            output_jsonl_path=predictions_path,
            eval_context={
                "split": "valid",
                "global_step": global_step,
                "epoch_progress": epoch_progress,
            },
        )
        run_state["latest_val_metrics"] = val_metrics
        _log_validation_summary(global_step, epoch_progress, val_metrics)

        selection_metric = float(val_metrics["cer_generate"])
        improved = selection_metric < float(run_state["best_metric"])
        if improved:
            run_state["best_metric"] = selection_metric
            run_state["evaluations_no_improve"] = 0
        else:
            run_state["evaluations_no_improve"] += 1

        history_row = {
            "event": "validation",
            "global_step": global_step,
            "epoch_progress": epoch_progress,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "grad_norm": float(grad_norm),
            "selection_metric": "micro_page_cer",
            "improved": bool(improved),
            "best_cer": float(run_state["best_metric"]),
            "evaluations_no_improve": int(run_state["evaluations_no_improve"]),
            "train_metrics_running": running_train_metrics,
            "val_metrics": val_metrics,
            "predictions_jsonl": str(predictions_path) if predictions_path is not None else None,
        }
        _append_jsonl(output_dir / "evaluation_history.jsonl", history_row)
        if wandb_run is not None:
            wandb_run.log(history_row, step=global_step)

        keep_top = max(1, int(CFG["keep_top_checkpoints"]))
        current_top = list(run_state["top_checkpoints"])
        worst_top = max(
            (float(item["cer"]) for item in current_top),
            default=float("inf"),
        )
        qualifies = len(current_top) < keep_top or selection_metric < worst_top
        if qualifies:
            checkpoint_dir = (
                output_dir
                / "checkpoints"
                / f"step_{global_step:06d}_cer_{selection_metric * 100.0:.4f}"
            )
            metadata = _checkpoint_metadata(
                epoch_idx=max(1, int(math.ceil(epoch_progress))),
                epoch_progress=epoch_progress,
                global_step=global_step,
                train_metrics=running_train_metrics,
                val_metrics=val_metrics,
                checkpoint_dir=checkpoint_dir,
                epoch_completed=False,
                run_state=run_state,
            )
            save_checkpoint(
                model,
                processor,
                checkpoint_dir,
                metadata,
                include_training_state=False,
            )
            current_top.append({
                "cer": selection_metric,
                "global_step": global_step,
                "epoch_progress": epoch_progress,
                "path": str(checkpoint_dir),
            })
            current_top.sort(key=lambda item: (float(item["cer"]), int(item["global_step"])))
            removed = current_top[keep_top:]
            current_top = current_top[:keep_top]
            run_state["top_checkpoints"] = current_top
            for item in removed:
                old_path = Path(item["path"])
                if old_path.exists() and old_path != checkpoint_dir:
                    shutil.rmtree(old_path)
            LOGGER.info("Top checkpoints: %s", run_state["top_checkpoints"])

        patience = max(1, int(CFG["early_stopping_patience_evals"]))
        return run_state["evaluations_no_improve"] >= patience


    stop_training = False
    for epoch_idx in range(start_epoch, CFG["num_epochs"] + 1):
        LOGGER.info("=" * 60)
        LOGGER.info("Epoch %d/%d | starting global_step=%d", epoch_idx, CFG["num_epochs"], global_step)
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            gradient_accumulation_steps=CFG["gradient_accumulation_steps"],
            gradient_clip_norm=CFG["gradient_clip_norm"],
            logging_steps=CFG["logging_steps"],
            max_train_batches=0,
            wandb_run=wandb_run,
            epoch=epoch_idx,
            global_step=global_step,
            optimizer_step_callback=_evaluate_and_maybe_checkpoint,
        )
        global_step = int(train_metrics["global_step"])
        stop_training = bool(train_metrics["stop_requested"])
        epoch_completed = bool(train_metrics["epoch_completed"])

        LOGGER.info(
            "Epoch %d train | total=%.4f lm=%.4f ul_scaled/lm=%.2f%% router=%.6f balance_raw=%.4f balance_scaled/lm=%.2f%% lam_raw=%.4f lam_scaled/lm=%.2f%% | "
            "global_step=%d completed=%s",
            epoch_idx,
            float(train_metrics["train_loss"]),
            float(train_metrics["train_lm_loss"]),
            float(train_metrics.get("train_token_ul_scaled_to_lm_ratio", 0.0)) * 100.0,
            float(train_metrics["train_moe_router_language_loss"]),
            float(train_metrics.get("train_moe_load_balance_raw_loss", 0.0)),
            float(train_metrics.get("train_moe_load_balance_to_lm_ratio", 0.0)) * 100.0,
            float(train_metrics.get("train_moe_lam_sparse_raw_loss", 0.0)),
            float(train_metrics.get("train_moe_lam_sparse_to_lm_ratio", 0.0)) * 100.0,
            global_step,
            epoch_completed,
        )
        if CFG.get("log_sample_exposure", False):
            exposure = train_metrics.get("sample_exposure_epoch", {})
            LOGGER.info("Epoch %d realized sample draws by source: %s", epoch_idx, exposure.get("by_source", {}))
            LOGGER.info("Epoch %d realized sample draws by task: %s", epoch_idx, exposure.get("by_task", {}))
            LOGGER.info("Epoch %d realized source x task: %s", epoch_idx, exposure.get("by_source_task", {}))
        last_metadata = _checkpoint_metadata(
            epoch_idx=epoch_idx,
            epoch_progress=global_step / max(updates_per_epoch, 1),
            global_step=global_step,
            train_metrics=train_metrics,
            val_metrics=run_state["latest_val_metrics"],
            checkpoint_dir=last_ckpt_dir,
            epoch_completed=epoch_completed,
            run_state=run_state,
        )
        save_checkpoint(
            model,
            processor,
            last_ckpt_dir,
            last_metadata,
            optimizer=optimizer,
            scheduler=scheduler,
            include_training_state=True,
        )
        LOGGER.info("Saved epoch-end last checkpoint -> %s", last_ckpt_dir)

        if stop_training:
            LOGGER.info(
                "Early stopping after %d validation events without CER improvement.",
                run_state["evaluations_no_improve"],
            )
            break

    if wandb_run is not None:
        wandb_run.finish()

    best_checkpoint = (
        min(run_state["top_checkpoints"], key=lambda item: float(item["cer"]))
        if run_state["top_checkpoints"] else None
    )
    summary = {
        "best_checkpoint": best_checkpoint,
        "best_cer": (
            float(run_state["best_metric"])
            if math.isfinite(float(run_state["best_metric"]))
            else None
        ),
        "last_checkpoint": str(last_ckpt_dir),
        "global_step": global_step,
        "validation_events": sum(1 for step in eval_steps if step <= global_step),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nTraining complete")
    print("Best checkpoint:", best_checkpoint)
    print("Last checkpoint:", last_ckpt_dir)
    print("Evaluation history:", output_dir / "evaluation_history.jsonl")

    return export_runtime_locals(state, locals())
