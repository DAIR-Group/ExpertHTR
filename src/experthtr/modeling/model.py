"""modeling.model phase for the ExpertHTR research workflow.

This module preserves the audited runtime computation. It is invoked by the
explicit CLI workflow and exports its declared runtime objects into a
:class:`~experthtr.runtime.context.RuntimeState`.
"""

from __future__ import annotations

from experthtr.runtime.context import RuntimeState, export_runtime_locals


def build_model(state: RuntimeState) -> RuntimeState:
    """Execute this phase after its prerequisites have populated ``state``."""

    globals().update(state.values)
    # ─────────────────────────────────────────────────────────────────────────────
    #  8a. Prepare HF cache / strict offline mode
    # ─────────────────────────────────────────────────────────────────────────────

    hf_cache = KAGGLE_WORKING / "hf_cache"
    hf_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_cache / "transformers"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_cache / "datasets"))
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    FULL_MODEL_STATE_FILE = "full_model_state.pt"
    CHECKPOINT_MANIFEST_FILE = "checkpoint_manifest.json"
    CHECKPOINT_COMPLETE_FILE = "CHECKPOINT_COMPLETE"

    set_seed(CFG["seed"])
    device = get_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        print(f"CUDA GPUs: {torch.cuda.device_count()} | {gpu_names} | dtype={get_model_dtype(device)} | data_parallel={CFG['use_data_parallel']}")

    # ─────────────────────────────────────────────────────────────────────────────
    #  8b. Load model + processor
    # ─────────────────────────────────────────────────────────────────────────────

    # -------------------------------------------------------------------------
    # Dense checkpoint -> MoE transfer. This is deliberately NOT a normal
    # `resume_from_checkpoint` path: dense state is loaded before MLP insertion.
    # -------------------------------------------------------------------------
    import gc
    import hashlib

    def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
        return digest.hexdigest()


    def _load_json_file(path: Path, label: str) -> Dict[str, Any]:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Cannot read {label}: {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"{label} must be a JSON object: {path}")
        return payload


    def _find_dense_run_config(checkpoint_dir: Path) -> Optional[Path]:
        # Standard layout is <output>/checkpoints/<step>/, but accept a copied
        # checkpoint that carries run_config.json directly or one level above.
        for candidate in (
            checkpoint_dir / "run_config.json",
            checkpoint_dir.parent / "run_config.json",
            checkpoint_dir.parent.parent / "run_config.json",
        ):
            if candidate.exists():
                return candidate
        return None


    def _assert_dense_init_contract(
        checkpoint_dir: Path,
        manifest: Dict[str, Any],
        metadata: Dict[str, Any],
        run_config: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        signature = manifest.get("moe_signature")
        if not isinstance(signature, dict) or signature.get("enabled") is not False:
            raise RuntimeError(
                "DENSE_INIT_CHECKPOINT_DIR must contain a verified dense checkpoint with "
                "checkpoint_manifest.json -> moe_signature.enabled=false."
            )
        for required in (CHECKPOINT_COMPLETE_FILE, CHECKPOINT_MANIFEST_FILE, FULL_MODEL_STATE_FILE):
            if not (checkpoint_dir / required).exists():
                raise RuntimeError(f"Dense init checkpoint is missing {required}: {checkpoint_dir}")
        expected_markers = list(CFG["atomic_ocr_markers"])
        checkpoint_markers = list(manifest.get("atomic_ocr_markers") or [])
        if checkpoint_markers != expected_markers:
            raise RuntimeError(
                f"Dense checkpoint marker policy differs: checkpoint={checkpoint_markers} "
                f"current={expected_markers}"
            )
        dataset_cfg = metadata.get("dataset_config") or {}
        source_scope_transfer = {"policy": "exact", "checkpoint_sources": [], "active_sources": []}
        if dataset_cfg:
            expected_dataset = str(CFG["dataset"])
            if str(dataset_cfg.get("dataset_name", "")) != expected_dataset:
                raise RuntimeError(
                    f"Dense checkpoint dataset differs: {dataset_cfg.get('dataset_name')!r} != {expected_dataset!r}"
                )
            expected_sources = sorted(CFG["expected_sources"])
            actual_sources = sorted(dataset_cfg.get("expected_sources") or [])
            source_scope_transfer = {
                "policy": "exact", "checkpoint_sources": actual_sources, "active_sources": expected_sources,
            }
            if actual_sources and actual_sources != expected_sources:
                active_set, checkpoint_set = set(expected_sources), set(actual_sources)
                missing_active = sorted(active_set - checkpoint_set)
                extra_checkpoint = sorted(checkpoint_set - active_set)
                allowed_extra = set(CFG.get("dense_init_allowed_extra_sources", []))
                if not (
                    bool(CFG.get("dense_init_allow_source_superset", False))
                    and not missing_active
                    and set(extra_checkpoint).issubset(allowed_extra)
                ):
                    raise RuntimeError(
                        "Dense checkpoint source scope differs outside the declared warm-start policy: "
                        f"checkpoint={actual_sources} current={expected_sources} "
                        f"missing_active={missing_active} extra_checkpoint={extra_checkpoint} "
                        f"allowed_extra={sorted(allowed_extra)}"
                    )
                source_scope_transfer = {
                    "policy": "checkpoint_source_superset_allowed",
                    "checkpoint_sources": actual_sources, "active_sources": expected_sources,
                    "excluded_from_new_run": extra_checkpoint,
                }
                LOGGER.warning(
                    "Warm-starting from a source-superset dense checkpoint; excluded from this new run: %s",
                    extra_checkpoint,
                )
        elif CFG["dense_init_require_run_config"]:
            raise RuntimeError("Dense checkpoint metadata has no dataset_config for strict contract validation.")

        compared = {}
        waived_contract_keys = []
        if run_config is None:
            if CFG["dense_init_require_run_config"]:
                raise RuntimeError(
                    "Could not find run_config.json next to the dense checkpoint. "
                    "Copy it with the checkpoint or set DENSE_INIT_REQUIRE_RUN_CONFIG=False deliberately."
                )
        else:
            old_cfg = run_config.get("cfg") or {}
            compare_keys = (
                "dataset", "expected_sources", "expected_split_records",
                "image_resize_policy", "image_min_pixels", "image_max_pixels",
                "image_max_aspect_ratio", "max_text_length", "eval_max_new_tokens",
                "atomic_ocr_markers", "page_htr_eos_weight", "token_ul_coef",
                "token_ul_page_fraction", "sampling_mode",
            )
            # Scope/counts necessarily differ for the explicitly allowed EPARCHOS
            # superset transfer; every input/loss contract remains strict.
            scope_keys = {"expected_sources", "expected_split_records"}
            mismatches = {}
            for key in compare_keys:
                if key not in old_cfg:
                    mismatches[key] = {"checkpoint": "<missing>", "current": CFG.get(key)}
                elif key in scope_keys and source_scope_transfer["policy"] == "checkpoint_source_superset_allowed":
                    waived_contract_keys.append({"key": key, "reason": "declared_source_superset_transfer"})
                    continue
                elif (
                    key == "expected_split_records"
                    and bool(CFG.get("dense_init_allow_missing_split_records", False))
                    and isinstance(old_cfg.get(key), dict)
                    and set(old_cfg[key]) == {"train", "valid", "test"}
                    and all(value is None for value in old_cfg[key].values())
                ):
                    waived_contract_keys.append({"key": key, "reason": "checkpoint_serialized_all_split_counts_as_null"})
                    LOGGER.warning(
                        "Dense initialization checkpoint does not serialize split counts; "
                        "accepting only this declared missing-metadata exception."
                    )
                    continue
                elif old_cfg.get(key) != CFG.get(key):
                    mismatches[key] = {"checkpoint": old_cfg.get(key), "current": CFG.get(key)}
            if mismatches:
                raise RuntimeError(
                    "Dense checkpoint input/loss contract differs from this ablation. "
                    f"Mismatches: {json.dumps(mismatches, ensure_ascii=False, default=str)}"
                )
            compared = {
                key: CFG.get(key) for key in compare_keys
                if not (key in scope_keys and source_scope_transfer["policy"] == "checkpoint_source_superset_allowed")
                and key not in {entry["key"] for entry in waived_contract_keys}
            }

        state_path = checkpoint_dir / FULL_MODEL_STATE_FILE
        return {
            "mode": "dense_checkpoint_expand_moe_v1",
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_manifest": manifest,
            "checkpoint_metadata_path": str(checkpoint_dir / "metadata.json"),
            "run_config_checked": run_config is not None,
            "compared_contract": compared,
            "waived_contract_keys": waived_contract_keys,
            "source_scope_transfer": source_scope_transfer,
            "full_model_state_file": str(state_path),
            "full_model_state_bytes": int(state_path.stat().st_size),
            "full_model_state_sha256": _sha256_file(state_path) if CFG["dense_init_hash_full_state"] else None,
        }


    def _temporary_torch_seed(seed: int):
        class _SeedContext:
            def __enter__(self):
                self.cpu_state = torch.random.get_rng_state()
                self.cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                torch.manual_seed(int(seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(seed))

            def __exit__(self, exc_type, exc, traceback):
                torch.random.set_rng_state(self.cpu_state)
                if self.cuda_state is not None:
                    torch.cuda.set_rng_state_all(self.cuda_state)

        return _SeedContext()


    def _assert_copy_initialized_moe(base_model_obj: nn.Module, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Verify copied experts and mixture function before the first optimizer step."""
        if not cfg["use_moe_mlp"]:
            return {"checked": False, "reason": "dense_continuation"}
        layers, layer_path = _find_decoder_layers(base_model_obj)
        selected = list(cfg["moe_mlp_resolved_layer_indices"])
        results = []
        for layer_idx in selected:
            moe = layers[layer_idx].mlp
            if not isinstance(moe, LanguageGroupedMoEMLP):
                raise RuntimeError(f"Selected layer {layer_idx} is not a LanguageGroupedMoEMLP after transfer.")
            if len(moe.language_experts) != int(cfg["moe_mlp_language_experts"]):
                raise RuntimeError(f"Layer {layer_idx} routed-expert count changed during transfer.")
            if int(moe.routed_top_k) != int(cfg["moe_mlp_routed_top_k"]):
                raise RuntimeError(f"Layer {layer_idx} has an incorrect routed top-k after transfer.")
            if int(cfg["moe_mlp_shared_experts"]) == 0:
                if len(moe.shared_experts) != 0 or float(moe.shared_mix) != 0.0:
                    raise RuntimeError(f"No-shared ablation has shared state at layer {layer_idx}.")
            else:
                if len(moe.shared_experts) != 1 or not 0.0 < float(moe.shared_mix) < 1.0:
                    raise RuntimeError(f"Shared ablation has invalid shared state at layer {layer_idx}.")

            reference_state = moe.language_experts[0].state_dict()
            for name, expert in enumerate(list(moe.language_experts)[1:] + list(moe.shared_experts), start=1):
                candidate = expert.state_dict()
                if reference_state.keys() != candidate.keys() or any(
                    not torch.equal(reference_state[key], candidate[key]) for key in reference_state
                ):
                    raise RuntimeError(
                        f"Expert copy mismatch at layer {layer_idx}, expert index {name}; "
                        "dense-to-MoE initialization is not function preserving."
                    )

            first_parameter = next(moe.language_experts[0].parameters())
            hidden_size = moe._infer_hidden_size(moe.language_experts[0])
            probe = torch.randn(
                (1, 3, hidden_size), device=first_parameter.device, dtype=first_parameter.dtype
            )
            prior_training = moe.training
            moe.eval()
            with torch.no_grad():
                dense_equivalent = moe._run_expert(moe.language_experts[0], probe)
                expanded_output = moe(probe)
            if prior_training:
                moe.train()
            diff = (expanded_output.float() - dense_equivalent.float()).abs()
            ref_scale = dense_equivalent.float().abs().mean().clamp_min(1e-6)
            max_abs = float(diff.max().item())
            mean_abs = float(diff.mean().item())
            rel_mean = float((diff.mean() / ref_scale).item())
            if not torch.isfinite(expanded_output).all() or max_abs > 0.05 or rel_mean > 0.02:
                raise RuntimeError(
                    f"Dense-to-MoE function-preservation test failed at layer {layer_idx}: "
                    f"max_abs={max_abs:.6g}, mean_rel={rel_mean:.6g}."
                )
            results.append({
                "layer": int(layer_idx), "max_abs_diff": max_abs,
                "mean_abs_diff": mean_abs, "mean_relative_diff": rel_mean,
                "routed_experts": len(moe.language_experts),
                "shared_experts": len(moe.shared_experts),
            })
        return {
            "checked": True,
            "layer_path": layer_path,
            "layers": results,
            "tolerance": {"max_abs": 0.05, "mean_relative": 0.02},
        }


    init_mode = str(CFG["initialization_mode"])
    dense_init_dir = Path(str(CFG["dense_init_checkpoint_dir"])).expanduser() if init_mode == "dense_checkpoint" else None
    dense_manifest = None
    dense_metadata = None
    dense_run_config = None
    DENSE_INITIALIZATION = {"mode": "fresh_base_v1"}

    if dense_init_dir is not None:
        if not dense_init_dir.exists():
            raise FileNotFoundError(f"DENSE_INIT_CHECKPOINT_DIR does not exist: {dense_init_dir}")
        for required_file in (CHECKPOINT_COMPLETE_FILE, CHECKPOINT_MANIFEST_FILE, FULL_MODEL_STATE_FILE, "metadata.json"):
            if not (dense_init_dir / required_file).exists():
                raise RuntimeError(f"Dense init checkpoint missing {required_file}: {dense_init_dir}")
        dense_manifest = _load_json_file(dense_init_dir / CHECKPOINT_MANIFEST_FILE, "dense checkpoint manifest")
        dense_metadata = _load_json_file(dense_init_dir / "metadata.json", "dense checkpoint metadata")
        dense_run_config_path = _find_dense_run_config(dense_init_dir)
        dense_run_config = _load_json_file(dense_run_config_path, "dense run config") if dense_run_config_path else None
        DENSE_INITIALIZATION = _assert_dense_init_contract(
            dense_init_dir, dense_manifest, dense_metadata, dense_run_config
        )

    processor_source = str(dense_init_dir) if dense_init_dir is not None else CFG["llm_name"]
    LOGGER.info("Loading processor from: %s", processor_source)
    processor = load_processor(
        processor_source,
        min_pixels=CFG["image_min_pixels"],
        max_pixels=CFG["image_max_pixels"],
    )
    CFG["processor_image_size"] = dict(processor._htr_image_budget)
    image_processor = getattr(processor, "image_processor", None)
    patch_size = getattr(image_processor, "patch_size", None)
    merge_size = getattr(image_processor, "merge_size", 1)
    if isinstance(patch_size, (list, tuple)):
        if len(set(int(value) for value in patch_size)) != 1:
            raise RuntimeError(f"Anisotropic image patch size is unsupported: {patch_size}")
        patch_size = patch_size[0]
    try:
        image_resize_factor = int(patch_size) * int(merge_size)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Cannot infer Qwen image resize factor from patch_size={patch_size!r}, merge_size={merge_size!r}."
        ) from exc
    if image_resize_factor <= 0:
        raise RuntimeError(f"Invalid Qwen image resize factor: {image_resize_factor}")
    CFG["image_patch_size"] = int(patch_size)
    CFG["image_merge_size"] = int(merge_size)
    CFG["image_resize_factor"] = image_resize_factor
    LOGGER.info(
        "Area/ratio resize contract: factor=%d min_pixels=%d max_pixels=%d",
        image_resize_factor, CFG["image_min_pixels"], CFG["image_max_pixels"],
    )

    LOGGER.info("Loading canonical base architecture from: %s", CFG["llm_name"])
    base_model = load_qwen3_vlm(CFG["llm_name"], device, use_flash=True)
    ATOMIC_OCR_MARKER_IDS = None

    if dense_init_dir is not None:
        # The saved processor must reproduce marker IDs before strict dense state loading.
        if not _atomic_ocr_markers_ready(processor.tokenizer, CFG["atomic_ocr_markers"]):
            raise RuntimeError(
                "Dense checkpoint processor does not expose the canonical atomic markers. "
                "Refusing a non-identical tokenizer transfer."
            )
        ATOMIC_OCR_MARKER_IDS = ensure_atomic_ocr_markers(
            processor.tokenizer, base_model, CFG["atomic_ocr_markers"], initialize_added_rows=False
        )
        checkpoint_marker_ids = dict(dense_manifest.get("atomic_ocr_marker_ids") or {})
        if checkpoint_marker_ids and checkpoint_marker_ids != dict(ATOMIC_OCR_MARKER_IDS):
            raise RuntimeError(
                f"Atomic marker IDs differ: checkpoint={checkpoint_marker_ids} "
                f"current={dict(ATOMIC_OCR_MARKER_IDS)}"
            )

        # Strictly restore the trained dense function before any expert exists.
        routing_state = MoERoutingState()
        model = Qwen3VLMForLM(base_model=base_model, routing_state=routing_state)
        try:
            payload = torch.load(
                dense_init_dir / FULL_MODEL_STATE_FILE, map_location="cpu", weights_only=False
            )
        except TypeError:
            payload = torch.load(dense_init_dir / FULL_MODEL_STATE_FILE, map_location="cpu")
        dense_state = payload.get("model_state_dict", payload)
        model.load_state_dict(dense_state, strict=True)
        del payload, dense_state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        LOGGER.info("Strictly loaded dense initialization checkpoint from %s", dense_init_dir)

        # Only now replace trained MLPs. Copy initialization preserves each selected
        # dense MLP at step zero; router/lambda are new and deterministic by seed.
        if CFG["use_moe_mlp"]:
            with _temporary_torch_seed(CFG["moe_init_seed"]):
                routing_state, moe_mlp_replaced = apply_moe_mlp_adapters(base_model, CFG)
            model.routing_state = routing_state
            DENSE_INITIALIZATION["moe_transfer"] = _assert_copy_initialized_moe(base_model, CFG)
        else:
            moe_mlp_replaced = 0
            DENSE_INITIALIZATION["moe_transfer"] = {"checked": False, "reason": "dense_continuation"}
    else:
        # Explicit fresh-base fallback. It is intentionally tagged separately.
        if _atomic_ocr_markers_ready(processor.tokenizer, CFG["atomic_ocr_markers"]):
            ATOMIC_OCR_MARKER_IDS = ensure_atomic_ocr_markers(
                processor.tokenizer, base_model, CFG["atomic_ocr_markers"], initialize_added_rows=False
            )
        with _temporary_torch_seed(CFG["moe_init_seed"]):
            routing_state, moe_mlp_replaced = apply_moe_mlp_adapters(base_model, CFG)
        model = Qwen3VLMForLM(base_model=base_model, routing_state=routing_state)
        if ATOMIC_OCR_MARKER_IDS is None:
            ATOMIC_OCR_MARKER_IDS = ensure_atomic_ocr_markers(
                processor.tokenizer, base_model, CFG["atomic_ocr_markers"], initialize_added_rows=True
            )
        if CFG["use_moe_mlp"]:
            DENSE_INITIALIZATION["moe_transfer"] = _assert_copy_initialized_moe(base_model, CFG)

    if ATOMIC_OCR_MARKER_IDS is None:
        raise RuntimeError("Atomic OCR marker setup did not produce token IDs.")
    CFG["atomic_ocr_marker_ids"] = dict(ATOMIC_OCR_MARKER_IDS)
    CFG["dense_initialization"] = DENSE_INITIALIZATION
    CFG["moe_mlp_replaced"] = int(moe_mlp_replaced)

    ablation_output_dir = Path(CFG["output_dir"])
    if state.command == "train":
        (ablation_output_dir / "dense_to_moe_initialization.json").write_text(
            json.dumps(DENSE_INITIALIZATION, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    GENERATION_STOP_IDS = resolve_generation_stop_ids(processor.tokenizer, model)
    _training_terminal_ids = _token_id_list(getattr(processor.tokenizer, "eos_token_id", None))
    if len(_training_terminal_ids) != 1:
        raise RuntimeError(f"Expected one tokenizer EOS ID, got {_training_terminal_ids}.")
    TRAINING_TERMINAL_ID = int(_training_terminal_ids[0])
    if TRAINING_TERMINAL_ID not in set(GENERATION_STOP_IDS):
        raise RuntimeError(
            f"Training terminal {TRAINING_TERMINAL_ID} is absent from generation stops {GENERATION_STOP_IDS}."
        )
    generation_eos_value = GENERATION_STOP_IDS[0] if len(GENERATION_STOP_IDS) == 1 else list(GENERATION_STOP_IDS)
    if getattr(base_model, "generation_config", None) is not None:
        base_model.generation_config.eos_token_id = generation_eos_value
    if getattr(base_model, "config", None) is not None:
        base_model.config.eos_token_id = generation_eos_value
    CFG["training_terminal_id"] = TRAINING_TERMINAL_ID
    CFG["generation_stop_ids"] = list(GENERATION_STOP_IDS)
    LOGGER.info("Terminal contract: training_eos=%d generation_stop_ids=%s", TRAINING_TERMINAL_ID, GENERATION_STOP_IDS)

    if CFG["gradient_checkpointing"]:
        if not hasattr(base_model, "gradient_checkpointing_enable"):
            raise RuntimeError("Base model does not expose gradient_checkpointing_enable().")
        try:
            base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            CFG["gradient_checkpointing_use_reentrant"] = False
        except TypeError as exc:
            if CFG.get("use_moe_mlp", False):
                raise RuntimeError(
                    "This MoE collects differentiable router losses during decoder forward, so it "
                    "requires non-reentrant gradient checkpointing support. Upgrade the matching "
                    "Transformers/PyTorch offline bundle or set HWVLM_GRADIENT_CHECKPOINTING=0."
                ) from exc
            base_model.gradient_checkpointing_enable()
            CFG["gradient_checkpointing_use_reentrant"] = None
        LOGGER.info(
            "Gradient checkpointing enabled | use_reentrant=%s",
            CFG.get("gradient_checkpointing_use_reentrant"),
        )
    if hasattr(base_model, "config") and hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False

    model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    LOGGER.info("Parameters: trainable=%.2fM total=%.2fM", trainable / 1e6, total / 1e6)
    print(f"Trainable params: {trainable / 1e6:.2f}M / {total / 1e6:.2f}M")
    model = maybe_wrap_data_parallel(model, CFG)

    return export_runtime_locals(state, locals())
