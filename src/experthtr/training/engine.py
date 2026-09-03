"""training.engine phase for the ExpertHTR research workflow.

This module preserves the audited runtime computation. It is invoked by the
explicit CLI workflow and exports its declared runtime objects into a
:class:`~experthtr.runtime.context.RuntimeState`.
"""

from __future__ import annotations

from experthtr.runtime.context import RuntimeState, export_runtime_locals


def build_training_engine(state: RuntimeState) -> RuntimeState:
    """Execute this phase after its prerequisites have populated ``state``."""

    globals().update(state.values)
    import time
    # -----------------------------------------------------------------------------
    #  Optimizer, scheduler, deterministic resume, train, and transparent eval
    # -----------------------------------------------------------------------------

    from collections import defaultdict
    from contextlib import nullcontext

    SKIP_TOKEN_AUDIT_GATE = os.getenv("HWVLM_SKIP_TOKEN_AUDIT_GATE", "0").lower() in {"1", "true", "yes"}
    if SKIP_TOKEN_AUDIT_GATE:
        LOGGER.warning(
            "Token-audit gate bypassed by explicit HWVLM_SKIP_TOKEN_AUDIT_GATE=1. "
            "Use this only when the dataset JSONL/images, model processor, prompts, image budgets, "
            "and max_text_length are exactly the configuration already audited."
        )
    else:
        if not globals().get("RUN_FULL_TOKEN_AUDIT", False) or "TOKEN_AUDIT_SUMMARY" not in globals():
            raise RuntimeError(
                "Full token audit is mandatory for this canonical run. "
                "Run the token-audit command with HWVLM_RUN_FULL_TOKEN_AUDIT=1, "
                "or explicitly set HWVLM_SKIP_TOKEN_AUDIT_GATE=1 only to reuse an unchanged audited setup."
            )
        unsafe_token_rows = [
            row for row in TOKEN_AUDIT_SUMMARY.get("by_source_task", [])
            if int(row.get("errors", 0)) > 0
            or int(row.get("zero_label_records", 0)) > 0
            or int(row.get("terminal_mismatch_records", 0)) > 0
            or int(row.get("prompt_prefix_mismatches", 0)) > 0
            or int(row.get("processor_over_max_records", 0)) > 0
            or int(row.get("image_over_max_pixel_records", 0)) > 0
            or (
                str(row.get("split", "")) in {"train", "valid"}
                and int(row.get("reference_loop_guard_triggers", 0)) > 0
            )
            or float(row.get("truncated_rate", 0.0)) > 0.0
            or float(row.get("micro_label_token_coverage", 1.0)) < 1.0
        ]
        if unsafe_token_rows:
            raise RuntimeError(
                "Token audit found errors, terminal/prompt contract mismatches, truncation, zero-label records, "
                "an image above the configured pixel budget, or a loop guard that fires on reference text; "
                "training is blocked. "
                f"First rows: {unsafe_token_rows[:5]}"
            )


    def _assert_microbatch_token_normalization() -> float:
        """Unit-test that uneven microbatches reproduce one full token-mean gradient."""
        generator = torch.Generator(device="cpu").manual_seed(20260824)
        inputs = torch.randn((5, 7, 4), generator=generator, dtype=torch.float64)
        targets = torch.randint(0, 6, (5, 7), generator=generator)
        valid = torch.rand((5, 7), generator=generator) > 0.35
        full_model = torch.nn.Linear(4, 6, bias=True).double()
        micro_model = copy.deepcopy(full_model)
        full_logits = full_model(inputs)
        full_loss = F.cross_entropy(full_logits[valid], targets[valid], reduction="sum") / valid.sum()
        full_loss.backward()
        total_tokens = 0
        for start, end in ((0, 1), (1, 3), (3, 5)):
            micro_valid = valid[start:end]
            micro_logits = micro_model(inputs[start:end])
            F.cross_entropy(
                micro_logits[micro_valid], targets[start:end][micro_valid], reduction="sum"
            ).backward()
            total_tokens += int(micro_valid.sum().item())
        for parameter in micro_model.parameters():
            parameter.grad.div_(total_tokens)
        max_difference = max(
            float((full_param.grad - micro_param.grad).abs().max().item())
            for full_param, micro_param in zip(full_model.parameters(), micro_model.parameters())
        )
        if max_difference > 1e-10:
            raise RuntimeError(f"Microbatch token-normalization test failed: max_grad_diff={max_difference}")
        return max_difference


    def _assert_collapse_objective_microbatch_normalization() -> float:
        """Verify EOS-weighted objective invariance across uneven microbatches."""
        generator = torch.Generator(device="cpu").manual_seed(20260827)
        # Use a compact remapped EOS for the unit test, then restore runtime ID.
        old_terminal = int(CFG["training_terminal_id"])
        old_ul_coef = float(CFG["token_ul_coef"])
        CFG["training_terminal_id"] = 7
        CFG["token_ul_coef"] = 0.0
        try:
            inputs = torch.randn((5, 7, 4), generator=generator, dtype=torch.float64)
            labels = torch.randint(0, 7, (5, 7), generator=generator)
            labels[:, 0] = IGNORE_INDEX
            labels[:, -1] = 7
            page_rows = torch.tensor([True, False, True, True, False])
            ul_rows = torch.zeros(5, dtype=torch.bool)
            full_model = torch.nn.Linear(4, 8, bias=True).double()
            micro_model = copy.deepcopy(full_model)
            full_obj = collapse_aware_lm_objective(full_model(inputs), labels, page_rows, ul_rows)
            (full_obj["objective_sum"] / full_obj["normalizer"]).backward()
            total_normalizer = 0.0
            for start, end in ((0, 1), (1, 3), (3, 5)):
                obj = collapse_aware_lm_objective(
                    micro_model(inputs[start:end]), labels[start:end],
                    page_rows[start:end], ul_rows[start:end],
                )
                obj["objective_sum"].backward()
                total_normalizer += float(obj["normalizer"].item())
            for parameter in micro_model.parameters():
                parameter.grad.div_(total_normalizer)
            max_difference = max(
                float((full_param.grad - micro_param.grad).abs().max().item())
                for full_param, micro_param in zip(full_model.parameters(), micro_model.parameters())
            )
            # collapse_aware_lm_objective intentionally evaluates CE in FP32,
            # even though this unit test constructs FP64 parameters. Different
            # reduction orders therefore have ordinary FP32 round-off. Keep the
            # gate tight enough to catch a missing/window-dependent normalizer,
            # but do not reject harmless differences around 1e-9.
            fp32_reduction_tolerance = 1e-7
            if max_difference > fp32_reduction_tolerance:
                raise RuntimeError(
                    "Collapse-objective microbatch test failed: "
                    f"max_grad_diff={max_difference} tolerance={fp32_reduction_tolerance}"
                )
            return max_difference
        finally:
            CFG["training_terminal_id"] = old_terminal
            CFG["token_ul_coef"] = old_ul_coef


    def _assert_token_unlikelihood_gradient() -> Tuple[float, int]:
        generator = torch.Generator(device="cpu").manual_seed(20260828)
        seq_len, vocab = max(140, int(CFG["token_ul_min_position"]) + 8), 16
        logits = torch.randn((1, seq_len, vocab), generator=generator, dtype=torch.float64, requires_grad=True)
        labels = torch.randint(0, vocab, (1, seq_len), generator=generator)
        labels[:, 0] = IGNORE_INDEX
        loss, count = token_unlikelihood_mean(logits, labels, torch.tensor([True]))
        if not torch.isfinite(loss) or int(count.item()) <= 0:
            raise RuntimeError(f"Token UL unit test failed: loss={loss.item()} candidates={count.item()}")
        loss.backward()
        if logits.grad is None or not torch.isfinite(logits.grad).all():
            raise RuntimeError("Token UL unit test produced missing/non-finite gradients.")
        return float(loss.item()), int(count.item())


    def _assert_lam_sparse_gradient() -> Tuple[float, float]:
        # LD-MoLE's objective is mean(1-lambda): descent must increase lambda.
        state = MoERoutingState(lam_sparse_coef=0.01)
        state.set_lam_sparse_warmup_scale(1.0)
        lam = torch.tensor([[0.20, 0.80]], dtype=torch.float32, requires_grad=True)
        state.add_lambda_values(lam)
        raw, scaled, effective_coef = state.lam_sparse_loss(lam.device)
        # `effective_coef` is a float32 tensor: accept its ordinary binary
        # representation error around 0.01, while still rejecting a wrong scale.
        if not torch.isfinite(raw) or not torch.isfinite(scaled) or abs(float(effective_coef.item()) - 0.01) > 1e-6:
            raise RuntimeError(f"LamSparseLoss unit test failed: raw={raw.item()} scaled={scaled.item()} coef={effective_coef.item()}")
        scaled.backward()
        if lam.grad is None or not torch.all(lam.grad < 0):
            raise RuntimeError(f"LamSparseLoss must increase lambda under descent, grad={lam.grad}")
        return float(raw.item()), float(scaled.item())


    if CFG.get("microbatch_invariant_token_loss", False):
        print("Microbatch token-normalization unit test max grad diff:", _assert_microbatch_token_normalization())
        print("Collapse-objective normalization max grad diff:", _assert_collapse_objective_microbatch_normalization())
        if float(CFG.get("token_ul_coef", 0.0)) > 0.0:
            print("Token UL unit test loss/candidates:", _assert_token_unlikelihood_gradient())
        else:
            print("Token UL unit test: skipped (HWVLM_TOKEN_UL_COEF=0).")
        if float(CFG.get("moe_mlp_lam_sparse_coef", 0.0)) > 0.0:
            print("LamSparseLoss unit test raw/scaled:", _assert_lam_sparse_gradient())

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=CFG["lr"],
        weight_decay=CFG["weight_decay"],
    )

    updates_per_epoch = max(
        1,
        math.ceil(len(train_loader) / max(1, CFG["gradient_accumulation_steps"])),
    )
    total_update_steps = max(1, updates_per_epoch * max(1, CFG["num_epochs"]))
    scheduler_total_update_steps = max(1, updates_per_epoch * max(1, CFG["scheduler_epochs"]))
    lam_sparse_coef = float(CFG.get("moe_mlp_lam_sparse_coef", 0.0))
    load_balance_coef = float(CFG.get("moe_mlp_load_balance_coef", 0.0))
    CFG["moe_mlp_lam_sparse_warmup_updates"] = (
        max(1, int(math.ceil(updates_per_epoch * float(CFG.get("moe_mlp_lam_sparse_warmup_fraction", 0.0)))))
        if lam_sparse_coef > 0.0 and float(CFG.get("moe_mlp_lam_sparse_warmup_fraction", 0.0)) > 0.0 else 0
    )
    CFG["moe_mlp_load_balance_warmup_updates"] = (
        max(1, int(math.ceil(updates_per_epoch * float(CFG.get("moe_mlp_load_balance_warmup_fraction", 0.0)))))
        if load_balance_coef > 0.0 and float(CFG.get("moe_mlp_load_balance_warmup_fraction", 0.0)) > 0.0 else 0
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=CFG["warmup_steps"],
        num_training_steps=scheduler_total_update_steps,
    )


    class FP16GradientCompatibleScaler(torch.amp.GradScaler):
        """Allow intentionally trainable FP16 experts when FP16 is selected."""

        def _unscale_grads_(self, optimizer, inv_scale, found_inf, allow_fp16):
            return super()._unscale_grads_(
                optimizer,
                inv_scale,
                found_inf,
                allow_fp16=True,
            )


    trainable_dtype_counts = Counter(
        str(param.dtype).replace("torch.", "")
        for param in model.parameters()
        if param.requires_grad
    )
    grad_scaler = FP16GradientCompatibleScaler(
        "cuda",
        enabled=(device.type == "cuda" and get_model_dtype(device) == torch.float16),
    )

    print(f"Optimizer updates per epoch: {updates_per_epoch}")
    print(f"Training optimizer updates: {total_update_steps}")
    print(f"Scheduler horizon optimizer updates: {scheduler_total_update_steps}")
    print(f"Warmup fraction of scheduler horizon: {CFG['warmup_steps'] / scheduler_total_update_steps:.2%}")
    if lam_sparse_coef > 0.0:
        print(
            "LamSparseLoss: coef=", lam_sparse_coef,
            "warmup_updates=", CFG["moe_mlp_lam_sparse_warmup_updates"],
            "of updates_per_epoch=", updates_per_epoch,
        )
    if load_balance_coef > 0.0:
        print(
            "Switch-style balance: coef=", load_balance_coef,
            "warmup_updates=", CFG["moe_mlp_load_balance_warmup_updates"],
            "of updates_per_epoch=", updates_per_epoch,
        )
    if CFG['warmup_steps'] >= scheduler_total_update_steps:
        raise ValueError("Warmup steps must be smaller than the one-epoch scheduler horizon.")
    print("Trainable parameter dtypes:", dict(trainable_dtype_counts))
    print("FP16-compatible GradScaler enabled:", grad_scaler.is_enabled())


    def _checkpoint_training_state_path(checkpoint_dir: Optional[Path]) -> Optional[Path]:
        if checkpoint_dir is None:
            return None
        path = Path(checkpoint_dir) / "training_state.pt"
        return path if path.exists() else None


    def _load_resume_training_state(checkpoint_dir: Optional[Path]) -> Dict[str, Any]:
        state_path = _checkpoint_training_state_path(checkpoint_dir)
        if state_path is None or not CFG["resume_training_state"]:
            return {}
        try:
            return torch.load(state_path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(state_path, map_location="cpu")


    def _move_optimizer_state_to_device(optimizer_obj, target_device):
        for state in optimizer_obj.state.values():
            for key, value in list(state.items()):
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(target_device)


    def _restore_rng_state(training_state: Dict[str, Any]) -> None:
        rng_state = training_state.get("rng_state", {}) if isinstance(training_state, dict) else {}
        if not isinstance(rng_state, dict):
            return
        if "python" in rng_state:
            random.setstate(rng_state["python"])
        if "numpy" in rng_state:
            np.random.set_state(rng_state["numpy"])
        if "torch" in rng_state:
            torch.random.set_rng_state(rng_state["torch"])
        if torch.cuda.is_available() and "cuda" in rng_state:
            torch.cuda.set_rng_state_all(rng_state["cuda"])
        if "sampler_generator" in rng_state:
            SAMPLER_GENERATOR.set_state(rng_state["sampler_generator"])
        if "loader_generator" in rng_state:
            LOADER_GENERATOR.set_state(rng_state["loader_generator"])


    # This workflow supports *dense initialization*, not generic checkpoint resume.
    # A dense checkpoint is intentionally loaded before MLPs are replaced.
    # Loading it again as a normal resume after MoE insertion would be architecturally
    # invalid. A true MoE resume also needs the identical MoE state to be restored
    # before optimizer/scheduler state, which is deliberately outside this one-epoch
    # ablation notebook. Fail explicitly instead of silently training the wrong model.
    _resume_path_raw = str(CFG.get("resume_from_checkpoint", "")).strip()
    if _resume_path_raw:
        raise ValueError(
            "HWVLM_RESUME_FROM_CHECKPOINT is unsupported in this one-epoch dense→MoE ablation. "
            "Use DENSE_INIT_CHECKPOINT_DIR for a dense checkpoint, or start a separate "
            "same-architecture MoE-resume workflow."
        )
    resume_checkpoint_dir = None
    resume_training_state = _load_resume_training_state(resume_checkpoint_dir)
    resume_completed_epochs = int(CFG["resume_completed_epochs"])
    resume_global_step = 0
    if resume_training_state:
        if not bool(resume_training_state.get("epoch_completed", False)):
            raise RuntimeError(
                "Exact mid-epoch resume is not supported because sampler position is not serialized. "
                "Resume from an epoch-complete last_checkpoint."
            )
        if "optimizer" in resume_training_state:
            optimizer.load_state_dict(resume_training_state["optimizer"])
            _move_optimizer_state_to_device(optimizer, device)
        if "scheduler" in resume_training_state:
            scheduler.load_state_dict(resume_training_state["scheduler"])
        _restore_rng_state(resume_training_state)
        resume_completed_epochs = int(resume_training_state.get("epoch", resume_completed_epochs))
        resume_global_step = int(resume_training_state.get("global_step", 0))
        LOGGER.info(
            "Loaded optimizer/scheduler/RNG state from completed epoch %d, global_step=%d",
            resume_completed_epochs,
            resume_global_step,
        )
    elif resume_checkpoint_dir is not None and CFG["resume_training_state"]:
        raise RuntimeError(
            f"Checkpoint {resume_checkpoint_dir} has no resumable training_state.pt."
        )


    def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


    def _running_train_metrics(
        total_loss: float,
        total_lm_loss: float,
        total_balance_raw_loss: float,
        total_balance_loss: float,
        total_router_loss: float,
        total_lam_sparse_raw_loss: float,
        total_lam_sparse_loss: float,
        total_batches: int,
        total_label_tokens: float,
    ) -> Dict[str, float]:
        batch_denominator = max(total_batches, 1)
        token_denominator = max(float(total_label_tokens), 1.0)
        return {
            "train_loss": total_loss / token_denominator,
            "train_lm_loss": total_lm_loss / token_denominator,
            "train_moe_load_balance_raw_loss": total_balance_raw_loss / batch_denominator,
            "train_moe_load_balance_loss": total_balance_loss / batch_denominator,
            "train_moe_router_language_loss": total_router_loss / batch_denominator,
            "train_moe_lam_sparse_raw_loss": total_lam_sparse_raw_loss / batch_denominator,
            "train_moe_lam_sparse_loss": total_lam_sparse_loss / batch_denominator,
            "train_batches": total_batches,
            "supervised_label_tokens": int(total_label_tokens),
        }


    def _new_sample_exposure_state() -> Dict[str, Any]:
        return {
            "draws": 0,
            "by_source": Counter(),
            "by_task": Counter(),
            "by_source_task": Counter(),
            "record_ids": set(),
            "page_keys": set(),
            "record_ids_by_source": defaultdict(set),
            "page_keys_by_source": defaultdict(set),
        }


    def _update_sample_exposure(state: Dict[str, Any], batch: Dict[str, Any]) -> None:
        ids = list(batch.get("ids", []))
        sources = list(batch.get("source_datasets", []))
        tasks = list(batch.get("tasks", []))
        metadata_rows = list(batch.get("metadata", []))
        page_keys = list(batch.get("page_keys", []))
        if not (len(ids) == len(sources) == len(tasks) == len(metadata_rows) == len(page_keys)):
            raise RuntimeError(
                "Cannot log sample exposure because batch metadata lengths differ: "
                f"ids={len(ids)} sources={len(sources)} tasks={len(tasks)} "
                f"metadata={len(metadata_rows)} page_keys={len(page_keys)}"
            )
        for record_id, source, task, metadata, page_key in zip(ids, sources, tasks, metadata_rows, page_keys):
            source = str(source)
            task = str(task)
            record_id = str(record_id)
            page_key = str(page_key)
            state["draws"] += 1
            state["by_source"][source] += 1
            state["by_task"][task] += 1
            state["by_source_task"][source, task] += 1
            state["record_ids"].add(record_id)
            state["page_keys"].add(page_key)
            state["record_ids_by_source"][source].add(record_id)
            state["page_keys_by_source"][source].add(page_key)


    def _sample_exposure_snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
        draws = int(state["draws"])
        unique_records = len(state["record_ids"])
        return {
            "draws": draws,
            "unique_records": unique_records,
            "duplicate_draws": draws - unique_records,
            "unique_pages": len(state["page_keys"]),
            "by_source": dict(sorted(state["by_source"].items())),
            "by_task": dict(sorted(state["by_task"].items())),
            "by_source_task": _nested_counter(state["by_source_task"]),
            "unique_records_by_source": {
                source: len(values) for source, values in sorted(state["record_ids_by_source"].items())
            },
            "unique_pages_by_source": {
                source: len(values) for source, values in sorted(state["page_keys_by_source"].items())
            },
        }


    TRAIN_SAMPLE_EXPOSURE_CUMULATIVE = _new_sample_exposure_state()


    def train_one_epoch(
        model,
        train_loader,
        optimizer,
        scheduler,
        gradient_accumulation_steps: int,
        gradient_clip_norm: float,
        logging_steps: int,
        max_train_batches: int,
        wandb_run,
        epoch: int,
        global_step: int,
        optimizer_step_callback=None,
    ) -> Dict[str, Any]:
        model.train()
        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        total_lm_loss = 0.0
        total_balance_raw_loss = 0.0
        total_balance_loss = 0.0
        total_router_loss = 0.0
        total_lam_sparse_raw_loss = 0.0
        total_lam_sparse_loss = 0.0
        total_token_ul_loss = 0.0
        total_token_ul_scaled_loss = 0.0
        total_token_ul_candidates = 0.0
        total_label_tokens = 0.0
        accumulated_label_tokens = 0.0
        source_lm_loss_sums: Dict[str, float] = defaultdict(float)
        source_lm_loss_counts: Dict[str, int] = defaultdict(int)
        epoch_exposure_state = _new_sample_exposure_state()
        total_batches = 0
        accumulated_batches = 0
        accumulation_target = max(1, gradient_accumulation_steps)
        stop_requested = False
        last_grad_norm: Optional[float] = None
        planned_batches = len(train_loader)
        if max_train_batches:
            planned_batches = min(planned_batches, max_train_batches)
        progress = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        model_for_objective = unwrap_model(model)

        skip_keys = {
            "texts", "prompts", "ids", "metadata", "source_datasets", "tasks", "languages", "page_keys",
            "language_groups", "resize_scales", "original_sizes", "resized_sizes",
        }

        for batch_idx, batch in enumerate(progress, start=1):
            if batch_idx > planned_batches:
                break
            if accumulated_batches == 0:
                remaining = planned_batches - batch_idx + 1
                accumulation_target = min(max(1, gradient_accumulation_steps), remaining)

            if CFG.get("log_sample_exposure", False):
                _update_sample_exposure(epoch_exposure_state, batch)
                _update_sample_exposure(TRAIN_SAMPLE_EXPOSURE_CUMULATIVE, batch)

            batch_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
                if not k.startswith("prompt_") and k not in skip_keys
            }
            autocast_ctx = (
                torch.amp.autocast("cuda", dtype=get_model_dtype(device))
                if device.type == "cuda" and get_model_dtype(device) != torch.float32
                else nullcontext()
            )
            lam_sparse_coef = float(CFG.get("moe_mlp_lam_sparse_coef", 0.0))
            lam_sparse_warmup_updates = int(CFG.get("moe_mlp_lam_sparse_warmup_updates", 0))
            lam_sparse_warmup_scale = (
                min(1.0, float(global_step) / float(lam_sparse_warmup_updates))
                if lam_sparse_coef > 0.0 and lam_sparse_warmup_updates > 0 else float(lam_sparse_coef > 0.0)
            )
            if hasattr(model_for_objective, "set_lam_sparse_warmup_scale"):
                model_for_objective.set_lam_sparse_warmup_scale(lam_sparse_warmup_scale)
            load_balance_coef = float(CFG.get("moe_mlp_load_balance_coef", 0.0))
            load_balance_warmup_updates = int(CFG.get("moe_mlp_load_balance_warmup_updates", 0))
            load_balance_warmup_scale = (
                min(1.0, float(global_step) / float(load_balance_warmup_updates))
                if load_balance_coef > 0.0 and load_balance_warmup_updates > 0 else float(load_balance_coef > 0.0)
            )
            if hasattr(model_for_objective, "set_load_balance_warmup_scale"):
                model_for_objective.set_load_balance_warmup_scale(load_balance_warmup_scale)
            with autocast_ctx:
                outputs = model(**batch_device)
                loss = outputs["loss"]
                if isinstance(loss, torch.Tensor) and loss.ndim > 0:
                    loss = loss.mean()
                lm_loss = outputs.get("lm_loss", loss.detach())
                balance_raw_loss = outputs.get("moe_load_balance_raw_loss", loss.detach() * 0.0)
                balance_loss = outputs.get("moe_load_balance_loss", loss.detach() * 0.0)
                router_loss = outputs.get("moe_router_language_loss", loss.detach() * 0.0)
                lam_sparse_raw_loss = outputs.get("moe_lam_sparse_raw_loss", loss.detach() * 0.0)
                lam_sparse_loss = outputs.get("moe_lam_sparse_loss", loss.detach() * 0.0)
                for name, value in (
                    ("lm_loss", lm_loss),
                    ("balance_raw_loss", balance_raw_loss),
                    ("balance_loss", balance_loss),
                    ("router_loss", router_loss),
                ):
                    if isinstance(value, torch.Tensor) and value.ndim > 0:
                        if name == "lm_loss":
                            lm_loss = value.mean()
                        elif name == "balance_raw_loss":
                            balance_raw_loss = value.mean()
                        elif name == "balance_loss":
                            balance_loss = value.mean()
                        else:
                            router_loss = value.mean()
                if isinstance(lam_sparse_raw_loss, torch.Tensor) and lam_sparse_raw_loss.ndim > 0:
                    lam_sparse_raw_loss = lam_sparse_raw_loss.mean()
                if isinstance(lam_sparse_loss, torch.Tensor) and lam_sparse_loss.ndim > 0:
                    lam_sparse_loss = lam_sparse_loss.mean()
                router_acc = outputs.get("moe_router_language_accuracy")
                router_entropy = outputs.get("moe_router_entropy")
                loss_sum = outputs.get("loss_sum")
                lm_loss_sum = outputs.get("lm_loss_sum")
                lm_token_count = outputs.get("lm_token_count")
                lm_loss_normalizer = outputs.get("lm_loss_normalizer")
                token_ul_loss = outputs.get("token_ul_loss", loss.detach() * 0.0)
                token_ul_scaled_loss = outputs.get("token_ul_scaled_loss", loss.detach() * 0.0)
                token_ul_candidate_count = outputs.get("token_ul_candidate_count", loss.detach() * 0.0)
                if CFG.get("microbatch_invariant_token_loss", False):
                    if not all(isinstance(value, torch.Tensor) for value in (loss_sum, lm_loss_sum, lm_token_count, lm_loss_normalizer)):
                        raise RuntimeError("Model must return loss_sum, lm_loss_sum, lm_token_count and lm_loss_normalizer.")
                    loss_sum = loss_sum.sum()
                    lm_loss_sum = lm_loss_sum.sum()
                    lm_token_count = lm_token_count.sum()
                    lm_loss_normalizer = lm_loss_normalizer.sum()
                    batch_label_tokens = float(lm_loss_normalizer.detach().item())
                    if batch_label_tokens <= 0:
                        raise RuntimeError(f"Microbatch has no supervised label tokens: ids={batch.get('ids', [])}")
                else:
                    batch_label_tokens = 1.0

            if not torch.isfinite(loss):
                failure_row = {
                    "event": "non_finite_loss",
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "global_step": global_step,
                    "ids": list(batch.get("ids", [])[:8]),
                    "loss": loss.detach().cpu().tolist(),
                }
                _append_jsonl(Path(CFG["output_dir"]) / "training_history.jsonl", failure_row)
                raise FloatingPointError(
                    f"Non-finite loss at epoch={epoch}, batch={batch_idx}, ids={batch.get('ids', [])[:8]}, "
                    f"loss={loss.detach().cpu().tolist()}"
                )

            scaled_loss = (
                loss_sum
                if CFG.get("microbatch_invariant_token_loss", False)
                else loss / max(1, accumulation_target)
            )
            if grad_scaler.is_enabled():
                grad_scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if CFG.get("microbatch_invariant_token_loss", False):
                total_loss += float(loss_sum.detach().item())
                total_lm_loss += float(lm_loss_sum.detach().item())
            else:
                total_loss += float(loss.detach().item())
                total_lm_loss += float(lm_loss.detach().item())
            total_balance_raw_loss += float(balance_raw_loss.detach().item())
            total_balance_loss += float(balance_loss.detach().item())
            total_router_loss += float(router_loss.detach().item())
            total_lam_sparse_raw_loss += float(lam_sparse_raw_loss.detach().item())
            total_lam_sparse_loss += float(lam_sparse_loss.detach().item())
            total_token_ul_loss += float(token_ul_loss.detach().mean().item())
            total_token_ul_scaled_loss += float(token_ul_scaled_loss.detach().mean().item())
            total_token_ul_candidates += float(token_ul_candidate_count.detach().sum().item())
            total_label_tokens += batch_label_tokens
            accumulated_label_tokens += batch_label_tokens
            total_batches += 1
            accumulated_batches += 1

            if CFG.get("log_train_loss_by_source", False):
                sample_lm_losses = outputs.get("sample_lm_losses")
                if isinstance(sample_lm_losses, torch.Tensor) and sample_lm_losses.ndim == 1:
                    for source, sample_loss in zip(
                        batch.get("source_datasets", []),
                        sample_lm_losses.detach().float().cpu().tolist(),
                    ):
                        if math.isfinite(float(sample_loss)):
                            source_lm_loss_sums[str(source)] += float(sample_loss)
                            source_lm_loss_counts[str(source)] += 1

            if accumulated_batches >= accumulation_target:
                if grad_scaler.is_enabled():
                    grad_scaler.unscale_(optimizer)
                if CFG.get("microbatch_invariant_token_loss", False):
                    if accumulated_label_tokens <= 0:
                        raise RuntimeError("Optimizer window has no supervised label tokens.")
                    inverse_tokens = 1.0 / float(accumulated_label_tokens)
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(inverse_tokens)
                max_norm = gradient_clip_norm if gradient_clip_norm > 0 else float("inf")
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=max_norm,
                    error_if_nonfinite=True,
                )
                grad_norm = float(grad_norm_tensor.detach().item())
                last_grad_norm = grad_norm
                if grad_scaler.is_enabled():
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated_batches = 0
                accumulated_label_tokens = 0.0
                global_step += 1

                running = _running_train_metrics(
                    total_loss,
                    total_lm_loss,
                    total_balance_raw_loss,
                    total_balance_loss,
                    total_router_loss,
                    total_lam_sparse_raw_loss,
                    total_lam_sparse_loss,
                    total_batches,
                    total_label_tokens,
                )
                epoch_progress = (epoch - 1) + batch_idx / max(planned_batches, 1)
                if CFG.get("log_sample_exposure", False):
                    running["sample_exposure_epoch"] = _sample_exposure_snapshot(epoch_exposure_state)
                    running["sample_exposure_cumulative"] = _sample_exposure_snapshot(TRAIN_SAMPLE_EXPOSURE_CUMULATIVE)
                if optimizer_step_callback is not None:
                    stop_requested = bool(
                        optimizer_step_callback(
                            global_step=global_step,
                            epoch_progress=epoch_progress,
                            running_train_metrics=running,
                            grad_norm=grad_norm,
                        )
                    )
                    model.train()
                if stop_requested:
                    break

            should_log = logging_steps and (
                batch_idx % logging_steps == 0 or batch_idx == planned_batches or stop_requested
            )
            if should_log:
                running = _running_train_metrics(
                    total_loss,
                    total_lm_loss,
                    total_balance_raw_loss,
                    total_balance_loss,
                    total_router_loss,
                    total_lam_sparse_raw_loss,
                    total_lam_sparse_loss,
                    total_batches,
                    total_label_tokens,
                )
                model_for_logging = unwrap_model(model)
                expert_usage = (
                    model_for_logging.get_and_reset_expert_usage()
                    if hasattr(model_for_logging, "get_and_reset_expert_usage")
                    else {}
                )
                log_row = {
                    "event": "train_progress",
                    "epoch": epoch,
                    "epoch_progress": (epoch - 1) + batch_idx / max(planned_batches, 1),
                    "batch_idx": batch_idx,
                    "global_step": global_step,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "grad_norm": last_grad_norm,
                    **running,
                    "router_accuracy_last_batch": (
                        float(router_acc.detach().item()) if isinstance(router_acc, torch.Tensor) else None
                    ),
                    "router_entropy_last_batch": (
                        float(router_entropy.detach().item()) if isinstance(router_entropy, torch.Tensor) else None
                    ),
                    "expert_usage_since_last_log": expert_usage,
                    "token_ul_loss_last_batch": float(token_ul_loss.detach().mean().item()),
                    "token_ul_scaled_loss_last_batch": float(token_ul_scaled_loss.detach().mean().item()),
                    "token_ul_candidates_last_batch": float(token_ul_candidate_count.detach().sum().item()),
                    "load_balance_raw_loss_last_batch": float(balance_raw_loss.detach().item()),
                    "load_balance_loss_last_batch": float(balance_loss.detach().item()),
                    "load_balance_warmup_scale": float(load_balance_warmup_scale),
                    "lam_sparse_raw_loss_last_batch": float(lam_sparse_raw_loss.detach().item()),
                    "lam_sparse_loss_last_batch": float(lam_sparse_loss.detach().item()),
                    "lam_sparse_warmup_scale": float(lam_sparse_warmup_scale),
                }
                _append_jsonl(Path(CFG["output_dir"]) / "training_history.jsonl", log_row)
                progress.set_postfix(
                    loss=f"{running['train_loss']:.4f}",
                    lm=f"{running['train_lm_loss']:.4f}",
                    step=global_step,
                )
                if wandb_run is not None:
                    wandb_run.log(log_row, step=global_step)

        train_lm_loss_by_source = {
            source: source_lm_loss_sums[source] / max(1, source_lm_loss_counts[source])
            for source in sorted(source_lm_loss_sums)
        }
        result = _running_train_metrics(
            total_loss,
            total_lm_loss,
            total_balance_raw_loss,
            total_balance_loss,
            total_router_loss,
            total_lam_sparse_raw_loss,
            total_lam_sparse_loss,
            total_batches,
            total_label_tokens,
        )
        result.update({
            "train_lm_loss_by_source": train_lm_loss_by_source,
            "global_step": global_step,
            "stop_requested": stop_requested,
            "epoch_completed": (not stop_requested and total_batches == planned_batches),
            "train_token_ul_loss_batch_mean": total_token_ul_loss / max(total_batches, 1),
            "train_token_ul_scaled_loss_batch_mean": total_token_ul_scaled_loss / max(total_batches, 1),
            "train_token_ul_candidate_count": total_token_ul_candidates,
        })
        result["train_token_ul_scaled_to_lm_ratio"] = (
            result["train_token_ul_scaled_loss_batch_mean"] / max(float(result["train_lm_loss"]), 1e-12)
        )
        result["train_moe_lam_sparse_to_lm_ratio"] = (
            result["train_moe_lam_sparse_loss"] / max(float(result["train_lm_loss"]), 1e-12)
        )
        result["train_moe_load_balance_to_lm_ratio"] = (
            result["train_moe_load_balance_loss"] / max(float(result["train_lm_loss"]), 1e-12)
        )
        if CFG.get("token_ul_coef", 0.0) > 0.0:
            ratio = float(result["train_token_ul_scaled_to_lm_ratio"])
            if ratio > 0.15:
                LOGGER.warning("Token UL contributes %.2f%% of LM loss (>15%%); halve HWVLM_TOKEN_UL_COEF next run.", ratio * 100.0)
            elif ratio < 0.02:
                LOGGER.warning("Token UL contributes %.2f%% of LM loss (<2%%); inspect candidates before increasing its coefficient.", ratio * 100.0)
        if CFG.get("moe_mlp_lam_sparse_coef", 0.0) > 0.0:
            lam_ratio = float(result["train_moe_lam_sparse_to_lm_ratio"])
            if lam_ratio > 0.05:
                LOGGER.warning("LamSparseLoss contributes %.2f%% of LM loss (>5%%); do not increase its coefficient.", lam_ratio * 100.0)
        if CFG.get("moe_mlp_load_balance_coef", 0.0) > 0.0:
            balance_ratio = float(result["train_moe_load_balance_to_lm_ratio"])
            if balance_ratio > 0.05:
                LOGGER.warning("Balance loss contributes %.2f%% of LM loss (>5%%); do not increase its coefficient.", balance_ratio * 100.0)
        if CFG.get("log_sample_exposure", False):
            result["sample_exposure_epoch"] = _sample_exposure_snapshot(epoch_exposure_state)
            result["sample_exposure_cumulative"] = _sample_exposure_snapshot(TRAIN_SAMPLE_EXPOSURE_CUMULATIVE)
            if result["epoch_completed"] and CFG.get("sampling_mode") == "full_pass":
                exposure = result["sample_exposure_epoch"]
                expected_records = len(train_loader.dataset)
                expected_pages = int(FULL_PASS_PAGE_CONTRACT["pages"])
                if (
                    int(exposure["draws"]) != expected_records
                    or int(exposure["unique_records"]) != expected_records
                    or int(exposure["duplicate_draws"]) != 0
                    or int(exposure["unique_pages"]) != expected_pages
                ):
                    raise RuntimeError(
                        "Full-pass exposure invariant failed: "
                        f"observed={exposure} expected_records={expected_records} "
                        f"expected_pages={expected_pages}"
                    )
            exposure_row = {
                "event": "train_sample_exposure",
                "epoch": int(epoch),
                "global_step": int(global_step),
                "epoch_completed": bool(result["epoch_completed"]),
                "epoch_exposure": result["sample_exposure_epoch"],
                "cumulative_exposure": result["sample_exposure_cumulative"],
            }
            _append_jsonl(Path(CFG["output_dir"]) / "training_sample_exposure.jsonl", exposure_row)
        return result


    def _metric_tracker_to_dict(tracker: MetricTracker) -> Dict[str, float]:
        m = tracker.compute()
        return {
            "cer_generate": float(m["cer"]),
            "wer_generate": float(m["wer"]),
            "num_samples": int(m["num_samples"]),
        }


    def _describe_numeric(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": int(array.size),
            "mean": float(array.mean()),
            "p50": float(np.percentile(array, 50)),
            "p95": float(np.percentile(array, 95)),
            "p99": float(np.percentile(array, 99)),
            "max": float(array.max()),
        }


    def _ngram_repetition_fraction(token_ids: List[int], n: int = 3) -> float:
        if len(token_ids) < n:
            return 0.0
        ngrams = [tuple(token_ids[i : i + n]) for i in range(len(token_ids) - n + 1)]
        return 1.0 - len(set(ngrams)) / max(1, len(ngrams))


    def _reference_token_ids(tokenizer, text: str) -> List[int]:
        encoded = tokenizer(text, add_special_tokens=False)
        ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(token_id) for token_id in ids]


    _EVAL_REGION_HEADER_RE = re.compile(r"(?m)^\[R\d+\]:[ \t]*$")


    def _eval_structure_counts(text: str) -> Tuple[int, int]:
        value = str(text).replace("\r\n", "\n").replace("\r", "\n")
        region_count = len(_EVAL_REGION_HEADER_RE.findall(value))
        line_count = sum(
            1 for line in value.split("\n")
            if line.strip() and not _EVAL_REGION_HEADER_RE.fullmatch(line.strip())
        )
        return region_count, line_count


    def _eval_marker_counts(text: str) -> Dict[str, int]:
        return {marker: str(text).count(marker) for marker in CFG.get("atomic_ocr_markers", [])}


    def _generation_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not rows:
            return {
                "num_samples": 0,
                "output_tokens": _describe_numeric([]),
                "reference_tokens": _describe_numeric([]),
                "output_reference_token_ratio": _describe_numeric([]),
                "prediction_chars": _describe_numeric([]),
                "reference_chars": _describe_numeric([]),
                "max_new_tokens_hit_rate": 0.0,
                "eos_emitted_rate": 0.0,
                "empty_output_rate": 0.0,
                "loop_guard_trigger_rate": 0.0,
                "repetition_3gram_fraction": _describe_numeric([]),
                "region_count_exact_rate": 0.0,
                "line_count_exact_rate": 0.0,
                "marker_reference_counts": {},
                "marker_prediction_counts": {},
                "generation_seconds_per_page": _describe_numeric([]),
                "generation_seconds_per_sample": _describe_numeric([]),  # compatibility alias
                "generation_wall_seconds_total": 0.0,
                "generation_pages_per_second": 0.0,
                "generated_tokens_per_second": 0.0,
            }
        return {
            "num_samples": len(rows),
            "output_tokens": _describe_numeric([row["output_tokens"] for row in rows]),
            "reference_tokens": _describe_numeric([row["reference_tokens"] for row in rows]),
            "output_reference_token_ratio": _describe_numeric(
                [row["output_reference_token_ratio"] for row in rows]
            ),
            "prediction_chars": _describe_numeric([row["prediction_chars"] for row in rows]),
            "reference_chars": _describe_numeric([row["reference_chars"] for row in rows]),
            "max_new_tokens_hit_rate": float(np.mean([row["hit_max_new_tokens"] for row in rows])),
            "eos_emitted_rate": float(np.mean([row["eos_emitted"] for row in rows])),
            "empty_output_rate": float(np.mean([row["empty_output"] for row in rows])),
            "loop_guard_trigger_rate": float(np.mean([row["loop_guard_triggered"] for row in rows])),
            "repetition_3gram_fraction": _describe_numeric(
                [row["repetition_3gram_fraction"] for row in rows]
            ),
            "region_count_exact_rate": float(np.mean([row["region_count_exact"] for row in rows])),
            "line_count_exact_rate": float(np.mean([row["line_count_exact"] for row in rows])),
            "marker_reference_counts": {
                marker: int(sum(row["reference_marker_counts"].get(marker, 0) for row in rows))
                for marker in CFG.get("atomic_ocr_markers", [])
            },
            "marker_prediction_counts": {
                marker: int(sum(row["prediction_marker_counts"].get(marker, 0) for row in rows))
                for marker in CFG.get("atomic_ocr_markers", [])
            },
            "generation_seconds_per_page": _describe_numeric([row.get("generation_seconds_per_page", 0.0) for row in rows]),
            "generation_seconds_per_sample": _describe_numeric([row.get("generation_seconds_per_page", 0.0) for row in rows]),  # compatibility alias
            "generation_wall_seconds_total": float(sum(row.get("generation_seconds_per_page", 0.0) for row in rows)),
            "generation_pages_per_second": float(len(rows) / max(sum(row.get("generation_seconds_per_page", 0.0) for row in rows), 1e-12)),
            "generated_tokens_per_second": float(sum(row["output_tokens"] for row in rows) / max(sum(row.get("generation_seconds_per_page", 0.0) for row in rows), 1e-12)),
        }


    def evaluate_loader(
        model,
        data_loader,
        tokenizer,
        max_new_tokens: int,
        max_eval_batches: int = 0,
        output_jsonl_path: Optional[Path] = None,
        eval_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        overall_tracker = MetricTracker()
        safety_tracker = MetricTracker()
        source_trackers: Dict[str, MetricTracker] = defaultdict(MetricTracker)
        safety_source_trackers: Dict[str, MetricTracker] = defaultdict(MetricTracker)
        task_trackers: Dict[str, MetricTracker] = defaultdict(MetricTracker)
        group_trackers: Dict[str, MetricTracker] = defaultdict(MetricTracker)
        diagnostic_rows: List[Dict[str, Any]] = []
        diagnostics_by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        output_rows: List[Dict[str, Any]] = []
        model.eval()
        model_for_generate = unwrap_model(model)
        progress = tqdm(data_loader, desc="eval", leave=False)

        generation_stop_ids = resolve_generation_stop_ids(tokenizer, model_for_generate)
        eos_ids = {int(value) for value in generation_stop_ids}
        generation_eos_value = (
            generation_stop_ids[0] if len(generation_stop_ids) == 1 else generation_stop_ids
        )
        pad_id = getattr(tokenizer, "pad_token_id", None)
        context = dict(eval_context or {})
        evaluation_started = time.perf_counter()

        with torch.no_grad():
            for batch_idx, batch in enumerate(progress, start=1):
                if max_eval_batches and batch_idx > max_eval_batches:
                    break

                batch_device = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                prompt_batch = {
                    k.replace("prompt_", ""): v
                    for k, v in batch_device.items()
                    if k.startswith("prompt_") and k != "prompt_lens"
                }
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                generation_started = time.perf_counter()
                generated_ids = model_for_generate.generate(
                    **prompt_batch,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    eos_token_id=generation_eos_value,
                    pad_token_id=getattr(tokenizer, "pad_token_id", None),
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                generation_wall_seconds = time.perf_counter() - generation_started
                prompt_len = int(batch_device["prompt_input_ids"].shape[1])
                references = [text.strip() for text in batch["texts"]]
                sources = batch.get("source_datasets", ["unknown"] * len(references))
                tasks = batch.get("tasks", ["unknown"] * len(references))
                groups = batch.get("language_groups", ["unknown"] * len(references))
                ids = batch.get("ids", ["unknown"] * len(references))

                predictions: List[str] = []
                safety_predictions: List[str] = []
                answer_token_lists: List[List[int]] = []
                loop_boundaries: List[Optional[int]] = []
                eos_flags: List[bool] = []
                max_flags: List[bool] = []
                for sequence in generated_ids:
                    raw_new_ids = [int(token_id) for token_id in sequence[prompt_len:].detach().cpu().tolist()]
                    answer_ids: List[int] = []
                    eos_emitted = False
                    stopped_by_pad = False
                    for token_id in raw_new_ids:
                        if token_id in eos_ids:
                            eos_emitted = True
                            break
                        if pad_id is not None and token_id == int(pad_id):
                            stopped_by_pad = True
                            break
                        answer_ids.append(token_id)
                    hit_max = (
                        not eos_emitted
                        and not stopped_by_pad
                        and len(raw_new_ids) >= max_new_tokens
                    )
                    prediction = tokenizer.decode(
                        answer_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
                    ).strip()
                    loop_boundary = _rolling_ngram_loop_boundary(answer_ids)
                    safety_ids = answer_ids[:loop_boundary] if loop_boundary is not None else answer_ids
                    safety_prediction = tokenizer.decode(
                        safety_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
                    ).strip()
                    predictions.append(prediction)
                    safety_predictions.append(safety_prediction)
                    answer_token_lists.append(answer_ids)
                    loop_boundaries.append(loop_boundary)
                    eos_flags.append(eos_emitted)
                    max_flags.append(hit_max)

                metric_predictions = [
                    page_htr_text_for_cer(pred) if str(task) == "page_htr" else pred
                    for pred, task in zip(predictions, tasks)
                ]
                metric_references = [
                    page_htr_text_for_cer(ref) if str(task) == "page_htr" else ref
                    for ref, task in zip(references, tasks)
                ]
                safety_metric_predictions = [
                    page_htr_text_for_cer(pred) if str(task) == "page_htr" else pred
                    for pred, task in zip(safety_predictions, tasks)
                ]

                overall_tracker.update(metric_predictions, metric_references, batch_size=len(references))
                safety_tracker.update(safety_metric_predictions, metric_references, batch_size=len(references))
                for index, (pred, ref, source, task, group, sample_id) in enumerate(zip(
                    metric_predictions,
                    metric_references,
                    sources,
                    tasks,
                    groups,
                    ids,
                )):
                    source = str(source)
                    task = str(task)
                    group = str(group)
                    source_trackers[source].update([pred], [ref], batch_size=1)
                    safety_source_trackers[source].update([safety_metric_predictions[index]], [ref], batch_size=1)
                    task_trackers[task].update([pred], [ref], batch_size=1)
                    group_trackers[group].update([pred], [ref], batch_size=1)

                    ref_token_ids = _reference_token_ids(tokenizer, references[index])
                    output_tokens = len(answer_token_lists[index])
                    reference_tokens = len(ref_token_ids)
                    pred_regions, pred_lines = _eval_structure_counts(predictions[index])
                    ref_regions, ref_lines = _eval_structure_counts(references[index])
                    row = {
                        **context,
                        "id": str(sample_id),
                        "source": source,
                        "task": task,
                        "language_group": group,
                        "output_tokens": output_tokens,
                        "reference_tokens": reference_tokens,
                        "generation_batch_seconds": float(generation_wall_seconds),
                        "generation_batch_size": int(len(references)),
                        # Throughput-normalized page time, not single-page latency.
                        "generation_seconds_per_page": float(generation_wall_seconds) / max(len(references), 1),
                        "output_reference_token_ratio": output_tokens / max(reference_tokens, 1),
                        "prediction_chars": len(predictions[index]),
                        "reference_chars": len(references[index]),
                        "hit_max_new_tokens": bool(max_flags[index]),
                        "eos_emitted": bool(eos_flags[index]),
                        "empty_output": not bool(predictions[index]),
                        "repetition_3gram_fraction": _ngram_repetition_fraction(answer_token_lists[index]),
                        "loop_guard_triggered": loop_boundaries[index] is not None,
                        "loop_guard_boundary": loop_boundaries[index],
                        "safety_prediction_tokens": (
                            int(loop_boundaries[index]) if loop_boundaries[index] is not None else output_tokens
                        ),
                        "sample_cer": _levenshtein(tuple(ref), tuple(pred)) / max(len(ref), 1),
                        "sample_safety_cer": _levenshtein(
                            tuple(ref), tuple(safety_metric_predictions[index])
                        ) / max(len(ref), 1),
                        "prediction_region_count": pred_regions,
                        "reference_region_count": ref_regions,
                        "region_count_exact": pred_regions == ref_regions,
                        "prediction_line_count": pred_lines,
                        "reference_line_count": ref_lines,
                        "line_count_exact": pred_lines == ref_lines,
                        "prediction_marker_counts": _eval_marker_counts(predictions[index]),
                        "reference_marker_counts": _eval_marker_counts(references[index]),
                    }
                    diagnostic_rows.append(row)
                    diagnostics_by_source[source].append(row)
                    output_rows.append({
                        **row,
                        "prediction": predictions[index],
                        "safety_prediction": safety_predictions[index],
                        "reference": references[index],
                        "metric_prediction": pred,
                        "metric_reference": ref,
                    })

        metrics = _metric_tracker_to_dict(overall_tracker)
        safety_metrics = _metric_tracker_to_dict(safety_tracker)
        metrics["safety_cer_generate_diagnostic"] = float(safety_metrics["cer_generate"])
        metrics["loop_guard_trigger_rate"] = float(np.mean([
            row["loop_guard_triggered"] for row in diagnostic_rows
        ])) if diagnostic_rows else 0.0
        metrics["by_source"] = {
            source: _metric_tracker_to_dict(tracker)
            for source, tracker in sorted(source_trackers.items())
        }
        metrics["safety_cer_by_source_diagnostic"] = {
            source: float(_metric_tracker_to_dict(tracker)["cer_generate"])
            for source, tracker in sorted(safety_source_trackers.items())
        }
        metrics["macro_source_cer_diagnostic"] = float(np.mean([
            source_metrics["cer_generate"]
            for source_metrics in metrics["by_source"].values()
        ])) if metrics["by_source"] else 0.0
        metrics["macro_page_cer_diagnostic"] = float(np.mean([
            row["sample_cer"] for row in diagnostic_rows
        ])) if diagnostic_rows else 0.0
        metrics["by_task"] = {
            task: _metric_tracker_to_dict(tracker)
            for task, tracker in sorted(task_trackers.items())
        }
        metrics["by_language_group"] = {
            group: _metric_tracker_to_dict(tracker)
            for group, tracker in sorted(group_trackers.items())
        }
        metrics["generation_diagnostics"] = _generation_summary(diagnostic_rows)
        metrics["evaluation_wall_seconds"] = float(time.perf_counter() - evaluation_started)
        metrics["diagnostics_by_source"] = {
            source: _generation_summary(rows)
            for source, rows in sorted(diagnostics_by_source.items())
        }

        if output_jsonl_path is not None:
            output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with output_jsonl_path.open("w", encoding="utf-8") as f:
                for row in output_rows:
                    f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        return metrics

    return export_runtime_locals(state, locals())
