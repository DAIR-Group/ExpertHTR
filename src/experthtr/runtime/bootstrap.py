"""runtime.bootstrap phase for the ExpertHTR research workflow.

This module preserves the audited runtime computation. It is invoked by the
explicit CLI workflow and exports its declared runtime objects into a
:class:`~experthtr.runtime.context.RuntimeState`.
"""

from __future__ import annotations

from experthtr.runtime.context import RuntimeState, export_runtime_locals


def configure_runtime(state: RuntimeState) -> RuntimeState:
    """Execute this phase after its prerequisites have populated ``state``."""

    globals().update(state.values)
    # --- Constants -----------------------------------------------------------
    IGNORE_INDEX = -100
    DEFAULT_QWEN_VLM_NAME = os.getenv("HWVLM_MODEL_NAME", str(OFFLINE_BUNDLE_ROOT / "models" / "qwen35-08b-base" / "Qwen3.5-0.8B-Base"))
    DEFAULT_OCR_PROMPT = os.getenv(
        "HWVLM_PROMPT_TEXT",
        "Transcribe the handwritten text exactly. Output only the answer.",
    )

    DATASET_SUBDIR_ALIASES = {
        "word": "iam_word",
        "iam_word": "iam_word",
        "line": "line",
        "line_no_markup": "line_no_markup",
        "iam_lines": "iam_lines",
        "paragraph": "paragraph",
        "paragraph_multitask": "paragraph_multitask",
        "page_multitask": "page_multitask",
        "iam_paragraph": "iam_paragraph",
        "combined": "combined",
    }

    # Router CE is disabled in this workflow. Routing remains hidden-state based
    # at training and inference; metadata is retained only for source diagnostics.
    ROUTING_GROUPING = os.getenv("HWVLM_ROUTING_GROUPING", "unsupervised4").strip().lower()
    ROUTING_GROUP_SPECS = {
        "unsupervised4": {
            "description": "four source-agnostic routed experts; no router supervision",
            "groups": ["all"],
            "source_to_group": {
                "BRESSAY": "all", "Bentham": "all", "HWDB2.0": "all",
                "IAM": "all", "ICFHR2016": "all",
                "Images_Courriers": "all", "ScribbleLens": "all",
                "Eparchos": "all",  # tolerated only before the runtime exclusion filter
            },
        },
        "script3": {
            "description": "three routed experts supervised by writing script",
            "groups": ["latin", "greek", "han"],
            "source_to_group": {
                "BRESSAY": "latin",
                "Bentham": "latin",
                "IAM": "latin",
                "ICFHR2016": "latin",
                "Images_Courriers": "latin",
                "ScribbleLens": "latin",
                "Eparchos": "greek",
                "HWDB2.0": "han",
            },
        },
        "hybrid5": {
            "description": "legacy five language/source routing groups",
            "groups": [
                "fr_images_courriers", "en_latin", "zh_han",
                "pt_latin", "de_grc_low_resource",
            ],
            "source_to_group": {
                "Images_Courriers": "fr_images_courriers",
                "Bentham": "en_latin",
                "IAM": "en_latin",
                "ScribbleLens": "en_latin",
                "HWDB2.0": "zh_han",
                "BRESSAY": "pt_latin",
                "ICFHR2016": "de_grc_low_resource",
                "Eparchos": "de_grc_low_resource",
            },
        },
    }
    if ROUTING_GROUPING not in ROUTING_GROUP_SPECS:
        raise ValueError(
            f"HWVLM_ROUTING_GROUPING must be one of {sorted(ROUTING_GROUP_SPECS)}, "
            f"got {ROUTING_GROUPING!r}"
        )
    ROUTING_GROUP_SPEC = ROUTING_GROUP_SPECS[ROUTING_GROUPING]
    LANGUAGE_GROUPS = list(ROUTING_GROUP_SPEC["groups"])
    LANGUAGE_GROUP_TO_ID = {name: idx for idx, name in enumerate(LANGUAGE_GROUPS)}


    def language_group_from_metadata(meta: Dict[str, Any]) -> Tuple[int, str]:
        source = str(meta.get("source_dataset", "")).strip()
        mapping = ROUTING_GROUP_SPEC["source_to_group"]
        if source not in mapping:
            raise ValueError(f"Unexpected source in canonical page_multitask bundle: {source!r}")
        group = mapping[source]
        return LANGUAGE_GROUP_TO_ID[group], group


    LOGGER = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # --- Kaggle / local paths ------------------------------------------------
    KAGGLE_INPUT = Path("/kaggle/input")
    KAGGLE_WORKING = Path("/kaggle/working")
    if not KAGGLE_WORKING.exists():
        KAGGLE_WORKING = Path.cwd()


    def discover_data_root() -> Path:
        explicit = os.getenv("HWVLM_DATA_ROOT", "").strip()
        if explicit:
            candidate = Path(explicit).expanduser()
            if (candidate / "page_multitask" / "train.jsonl").exists():
                return candidate
            if candidate.name == "page_multitask" and (candidate / "train.jsonl").exists():
                return candidate
            raise FileNotFoundError(
                "HWVLM_DATA_ROOT must be the canonical mthtr root or its page_multitask directory: "
                f"{candidate}"
            )


        roots = set()
        if KAGGLE_INPUT.exists():
            for train_file in KAGGLE_INPUT.glob("**/page_multitask/train.jsonl"):
                roots.add(train_file.parent.parent.resolve())
        if len(roots) == 1:
            return next(iter(roots))
        if len(roots) > 1:
            raise RuntimeError(
                "Multiple page_multitask artifacts found. Set HWVLM_DATA_ROOT explicitly: "
                + ", ".join(str(path) for path in sorted(roots))
            )
        raise FileNotFoundError(
            "Canonical page_multitask artifact not found. Set HWVLM_DATA_ROOT explicitly."
        )


    DATA_ROOT = discover_data_root()
    print(f"Dataset root: {DATA_ROOT}")
    print("Dataset policy: InkData_paragraph removed; line_htr removed; line_range_htr uses deterministic pseudo-random R-local spans of 3-7 lines: <15 page lines -> 1 record, 15-29 -> 2, >=30 -> 3.")

    # --- Hyperparameters -----------------------------------------------------
    CFG = {
        "llm_name": os.getenv("HWVLM_MODEL_NAME", DEFAULT_QWEN_VLM_NAME),
        "dataset": os.getenv("HWVLM_DATASET", "page_multitask"),
        "run_name": os.getenv("HWVLM_RUN_NAME", "page_multitask_v5_dense_offline").strip(),
        "output_dir": os.getenv(
            "HWVLM_OUTPUT_DIR",
            str(KAGGLE_WORKING / "outputs" / "page_multitask_v5_dense_offline"),
        ).strip(),
        "num_epochs": int(os.getenv("HWVLM_NUM_EPOCHS", "3")),
        "scheduler_epochs": int(os.getenv("HWVLM_SCHEDULER_EPOCHS", os.getenv("HWVLM_NUM_EPOCHS", "3"))),
        "per_device_batch_size": int(os.getenv("HWVLM_BATCH_SIZE", "4")),
        "gradient_accumulation_steps": int(os.getenv("HWVLM_GRAD_ACC", "8")),
        "lr": float(os.getenv("HWVLM_LR", "4e-6")),
        "weight_decay": float(os.getenv("HWVLM_WEIGHT_DECAY", "0.01")),
        "warmup_steps": int(os.getenv("HWVLM_WARMUP_STEPS", "100")),
        "gradient_clip_norm": float(os.getenv("HWVLM_GRAD_CLIP", "1.0")),
        "image_resize_policy": os.getenv("HWVLM_IMAGE_RESIZE_POLICY", "area_ratio").strip().lower(),
        "image_min_pixels": int(os.getenv("HWVLM_IMAGE_MIN_PIXELS", "589824")),
        "image_max_pixels": int(os.getenv("HWVLM_IMAGE_MAX_PIXELS", "1572864")),
        "image_max_aspect_ratio": float(os.getenv("HWVLM_IMAGE_MAX_ASPECT_RATIO", "200")),
        "resolution_profile": os.getenv("HWVLM_RESOLUTION_PROFILE", "area_ratio_min589824_max1572864_ctx2560").strip(),
        "max_text_length": int(os.getenv("HWVLM_MAX_TEXT_LENGTH", "2560")),
        "eval_max_new_tokens": int(os.getenv("HWVLM_EVAL_MAX_NEW_TOKENS", "2048")),
        "eval_max_batches": int(os.getenv("HWVLM_EVAL_MAX_BATCHES", "0")),
        "eval_epochs": os.getenv("HWVLM_EVAL_EPOCHS", "").strip(),
        "eval_coarse_fraction": float(os.getenv("HWVLM_EVAL_COARSE_FRACTION", "0.5")),
        "eval_dense_start_epoch": float(os.getenv("HWVLM_EVAL_DENSE_START_EPOCH", "2.5")),
        "eval_dense_end_epoch": float(os.getenv("HWVLM_EVAL_DENSE_END_EPOCH", "3.6")),
        "eval_dense_fraction": float(os.getenv("HWVLM_EVAL_DENSE_FRACTION", "0.1")),
        "early_stopping_patience_evals": int(os.getenv("HWVLM_EARLY_STOPPING_EVALS", "10")),
        "keep_top_checkpoints": int(os.getenv("HWVLM_KEEP_TOP_CHECKPOINTS", "3")),
        "write_eval_predictions": os.getenv("HWVLM_WRITE_EVAL_PREDICTIONS", "1").lower() in {"1", "true", "yes"},
        "run_final_test": os.getenv("HWVLM_RUN_FINAL_TEST", "0").lower() in {"1", "true", "yes"},
        "num_workers": int(os.getenv("HWVLM_NUM_WORKERS", "16")),
        "use_data_parallel": os.getenv("HWVLM_USE_DATA_PARALLEL", "0").lower() in {"1", "true", "yes"},
        "torch_dtype": os.getenv("HWVLM_TORCH_DTYPE", "float16").strip().lower(),
        "logging_steps": int(os.getenv("HWVLM_LOGGING_STEPS", "100")),
        "valid_split_ratio": float(os.getenv("HWVLM_VALID_SPLIT_RATIO", "1.0")),
        "seed": int(os.getenv("HWVLM_SEED", "42")),
        "gradient_checkpointing": os.getenv("HWVLM_GRADIENT_CHECKPOINTING", "1").lower() in {"1", "true", "yes"},
        "use_augmentation": os.getenv("HWVLM_USE_AUGMENTATION", "0").lower() in {"1", "true", "yes"},
        "use_wandb": os.getenv("HWVLM_USE_WANDB", "0").lower() in {"1", "true", "yes"},
        "prompt_text": os.getenv("HWVLM_PROMPT_TEXT", "").strip(),
        "excluded_sources": [
            item.strip()
            for item in os.getenv("HWVLM_EXCLUDED_SOURCES", "").split(",")
            if item.strip()
        ],
        # Leave-one-task-out ablation: applied to train split by default.
        # Use HWVLM_EXCLUDED_VALID_TASKS only if you intentionally want to filter validation too.
        "excluded_train_tasks": [
            item.strip()
            for item in (os.getenv("HWVLM_EXCLUDED_TRAIN_TASKS", "") or os.getenv("HWVLM_EXCLUDED_TASKS", "")).split(",")
            if item.strip()
        ],
        "excluded_valid_tasks": [
            item.strip()
            for item in os.getenv("HWVLM_EXCLUDED_VALID_TASKS", "").split(",")
            if item.strip()
        ],
        # Source-agnostic sparse MLP-MoE. Router CE is disabled; inference uses
        # hidden states only and source metadata never becomes a router target.
        "use_moe_mlp": os.getenv("HWVLM_USE_MOE_MLP", "0").lower() in {"1", "true", "yes"},
        "moe_mlp_freeze_base": os.getenv("HWVLM_MOE_MLP_FREEZE_BASE", "0").lower() in {"1", "true", "yes"},
        # Four routed experts are an architectural budget, not metadata labels.
        "moe_mlp_language_experts": int(os.getenv("HWVLM_MOE_LANGUAGE_EXPERTS", "4")),
        "moe_mlp_shared_experts": int(os.getenv("HWVLM_MOE_SHARED_EXPERTS", "1")),
        "moe_mlp_routing_fn": os.getenv("HWVLM_MOE_ROUTING_FN", "sparsegen").strip().lower(),
        "moe_mlp_router_temperature": float(os.getenv("HWVLM_MOE_ROUTER_TEMPERATURE", "1.0")),
        "moe_mlp_sparsegen_hidden": int(os.getenv("HWVLM_MOE_SPARSEGEN_HIDDEN", "128")),
        "moe_mlp_sparsegen_eps": float(os.getenv("HWVLM_MOE_SPARSEGEN_EPS", "0.01")),
        "moe_mlp_active_expert_budget": int(os.getenv("HWVLM_MOE_ACTIVE_EXPERT_BUDGET", "0")),
        "moe_mlp_enforce_top_k": os.getenv("HWVLM_MOE_ENFORCE_TOP_K", "0").lower() in {"1", "true", "yes"},
        "moe_mlp_load_balance_coef": float(os.getenv("HWVLM_MOE_LOAD_BALANCE_COEF", "0.0")),
        "moe_mlp_load_balance_warmup_fraction": float(os.getenv("HWVLM_MOE_LOAD_BALANCE_WARMUP_FRACTION", "0.0")),
        "moe_mlp_router_ce_coef": float(os.getenv("HWVLM_MOE_ROUTER_CE_COEF", "0.0")),
        "moe_mlp_lam_sparse_coef": float(os.getenv("HWVLM_MOE_LAM_SPARSE_COEF", "0.0")),
        "moe_mlp_lam_sparse_warmup_fraction": float(os.getenv("HWVLM_MOE_LAM_SPARSE_WARMUP_FRACTION", "0.0")),
        "moe_mlp_shared_mix": float(os.getenv("HWVLM_MOE_SHARED_MIX", "0.25")),
        "moe_mlp_layer_fraction": float(os.getenv("HWVLM_MOE_LAYER_FRACTION", "0.3333333333")),
        "moe_mlp_layer_spacing": int(os.getenv("HWVLM_MOE_LAYER_SPACING", "3")),
        "moe_mlp_layer_indices": [
            int(item.strip())
            for item in os.getenv("HWVLM_MOE_LAYER_INDICES", "").split(",")
            if item.strip()
        ],
        "log_train_loss_by_source": os.getenv("HWVLM_LOG_TRAIN_LOSS_BY_SOURCE", "1").lower() in {"1", "true", "yes"},
        "log_sample_exposure": os.getenv("HWVLM_LOG_SAMPLE_EXPOSURE", "1").lower() in {"1", "true", "yes"},
        "moe_mlp_expert_init": os.getenv("HWVLM_MOE_EXPERT_INIT", "copy").strip().lower(),
        "moe_mlp_expert_train_dtype": os.getenv("HWVLM_MOE_EXPERT_TRAIN_DTYPE", "match_base").strip().lower(),
        "use_group_balanced_sampler": os.getenv("HWVLM_USE_GROUP_BALANCED_SAMPLER", "0").lower() in {"1", "true", "yes"},
        "group_sampling_alpha": float(os.getenv("HWVLM_GROUP_SAMPLING_ALPHA", "0.5")),
        "use_source_balanced_sampler": os.getenv("HWVLM_USE_SOURCE_BALANCED_SAMPLER", "1").lower() in {"1", "true", "yes"},
        "source_sampling_alpha": float(os.getenv("HWVLM_SOURCE_SAMPLING_ALPHA", "0.5")),
        "sampling_mode": os.getenv("HWVLM_SAMPLING_MODE", "full_pass").strip().lower(),
        "microbatch_invariant_token_loss": os.getenv("HWVLM_MICROBATCH_INVARIANT_TOKEN_LOSS", "1").lower() in {"1", "true", "yes"},
        "lm_source_loss_weighting": os.getenv("HWVLM_LM_SOURCE_LOSS_WEIGHTING", "none").strip().lower(),
        "atomic_ocr_markers": [
            item.strip() for item in os.getenv("HWVLM_ATOMIC_OCR_MARKERS", "<del>,<gap>").split(",") if item.strip()
        ],
        "page_htr_eos_weight": float(os.getenv("HWVLM_PAGE_HTR_EOS_WEIGHT", "4.0")),
        "token_ul_coef": float(os.getenv("HWVLM_TOKEN_UL_COEF", "0.05")),
        "token_ul_page_fraction": float(os.getenv("HWVLM_TOKEN_UL_PAGE_FRACTION", "0.10")),
        "token_ul_context_window": int(os.getenv("HWVLM_TOKEN_UL_CONTEXT_WINDOW", "64")),
        "token_ul_min_position": int(os.getenv("HWVLM_TOKEN_UL_MIN_POSITION", "128")),
        "loop_ngram": int(os.getenv("HWVLM_LOOP_NGRAM", "3")),
        "loop_window": int(os.getenv("HWVLM_LOOP_WINDOW", "96")),
        "loop_threshold": float(os.getenv("HWVLM_LOOP_THRESHOLD", "0.60")),
        "loop_min_tokens": int(os.getenv("HWVLM_LOOP_MIN_TOKENS", "128")),
        "loop_consecutive": int(os.getenv("HWVLM_LOOP_CONSECUTIVE", "2")),
        "router_ce_class_weighting": os.getenv("HWVLM_ROUTER_CE_CLASS_WEIGHTING", "none").strip().lower(),
        "routing_grouping": ROUTING_GROUPING,

        "moe_ablation_mode": globals().get("MOE_ABLATION_VARIANT", "legacy"),
        "initialization_mode": globals().get("INIT_MODE", "fresh_base"),
        "dense_init_checkpoint_dir": globals().get("DENSE_INIT_CHECKPOINT_DIR", ""),
        "dense_init_require_run_config": bool(globals().get("DENSE_INIT_REQUIRE_RUN_CONFIG", False)),
        "dense_init_hash_full_state": bool(globals().get("DENSE_INIT_HASH_FULL_STATE", False)),
        "dense_init_allow_source_superset": bool(globals().get("DENSE_INIT_ALLOW_SOURCE_SUPERSET", False)),
        "dense_init_allowed_extra_sources": list(globals().get("DENSE_INIT_ALLOWED_EXTRA_SOURCES", ())),
        "dense_init_allow_missing_split_records": bool(globals().get("DENSE_INIT_ALLOW_MISSING_SPLIT_RECORDS", False)),
        "dense_init_allow_existing_output": bool(globals().get("ALLOW_EXISTING_OUTPUT_DIR", False)),
        "moe_init_seed": int(globals().get("MOE_INIT_SEED", 0)),
        "routing_group_description": str(ROUTING_GROUP_SPEC["description"]),
        "language_groups": list(LANGUAGE_GROUPS),
        # Kept here, rather than only in the validation cell below, so dataset
        # construction cannot fail when a fresh session skips that small cell.
        "expected_sources": [
            "BRESSAY", "Bentham", "HWDB2.0", "IAM",
            "ICFHR2016", "Images_Courriers", "ScribbleLens",
        ],
        "expected_split_records": {"train": 20388, "valid": 962, "test": 1327},
        # Resume controls. If the checkpoint only has model weights, set
        # HWVLM_RESUME_COMPLETED_EPOCHS to the number of finished epochs.
        "resume_from_checkpoint": os.getenv("HWVLM_RESUME_FROM_CHECKPOINT", "").strip(),
        "resume_completed_epochs": int(os.getenv("HWVLM_RESUME_COMPLETED_EPOCHS", "0")),
        "resume_training_state": os.getenv("HWVLM_RESUME_TRAINING_STATE", "1").lower() in {"1", "true", "yes"},
    }

    if CFG["num_epochs"] <= 0 or CFG["scheduler_epochs"] < CFG["num_epochs"]:
        raise ValueError(
            f"Require positive num_epochs and scheduler_epochs >= num_epochs, got "
            f"{CFG['num_epochs']} and {CFG['scheduler_epochs']}"
        )
    if CFG["image_resize_policy"] != "area_ratio":
        raise ValueError(f"Unsupported image_resize_policy={CFG['image_resize_policy']!r}")
    for key in ("image_min_pixels", "image_max_pixels", "image_max_aspect_ratio"):
        if int(CFG[key]) <= 0:
            raise ValueError(f"{key} must be positive, got {CFG[key]}")
    if CFG["image_min_pixels"] > CFG["image_max_pixels"]:
        raise ValueError(
            "image_min_pixels must be <= image_max_pixels: "
            f"{CFG['image_min_pixels']} > {CFG['image_max_pixels']}"
        )
    if CFG["sampling_mode"] not in {"full_pass", "source_balanced", "group_balanced"}:
        raise ValueError(f"Unknown sampling_mode={CFG['sampling_mode']!r}")
    if CFG["sampling_mode"] == "full_pass" and (
        CFG["use_source_balanced_sampler"] or CFG["use_group_balanced_sampler"]
    ):
        raise ValueError("full_pass requires both replacement-based balanced samplers to be disabled.")
    if CFG["microbatch_invariant_token_loss"] and CFG["lm_source_loss_weighting"] not in {"none", "off", "0"}:
        raise ValueError("Token-normalized accumulation currently requires lm_source_loss_weighting=none.")
    if CFG["atomic_ocr_markers"] != ["<del>", "<gap>"]:
        raise ValueError(f"Canonical atomic marker order must be ['<del>', '<gap>'], got {CFG['atomic_ocr_markers']}")
    if CFG["page_htr_eos_weight"] < 1.0:
        raise ValueError("page_htr_eos_weight must be >= 1.0.")
    if CFG["token_ul_coef"] < 0.0 or not 0.0 <= CFG["token_ul_page_fraction"] <= 1.0:
        raise ValueError("Require token_ul_coef >= 0 and token_ul_page_fraction in [0,1].")
    if CFG["token_ul_context_window"] <= 0 or CFG["token_ul_min_position"] < 1:
        raise ValueError("Invalid token-level unlikelihood window/minimum position.")
    if CFG["loop_ngram"] <= 0 or CFG["loop_window"] < CFG["loop_ngram"]:
        raise ValueError("Invalid loop n-gram/window configuration.")
    if not 0.0 <= CFG["loop_threshold"] <= 1.0 or CFG["loop_min_tokens"] < CFG["loop_window"] or CFG["loop_consecutive"] <= 0:
        raise ValueError("Invalid loop threshold/min-token/consecutive configuration.")
    if CFG["use_moe_mlp"]:
        if CFG["use_data_parallel"]:
            raise ValueError(
                "Language-grouped MoE uses one mutable routing state per forward and is not "
                "safe with torch.nn.DataParallel. Keep HWVLM_USE_DATA_PARALLEL=0."
            )
        if CFG["moe_mlp_shared_experts"] not in {0, 1}:

            raise ValueError("This ablation supports zero or one shared expert only.")

        if CFG["moe_mlp_shared_experts"] == 0 and CFG["moe_mlp_shared_mix"] != 0.0:

            raise ValueError("moe_mlp_shared_mix must be exactly 0 when shared experts are disabled.")

        if CFG["moe_mlp_shared_experts"] == 1 and not 0.0 < CFG["moe_mlp_shared_mix"] < 1.0:

            raise ValueError("moe_mlp_shared_mix must lie in (0, 1) when one shared expert is enabled.")
        if CFG["moe_mlp_enforce_top_k"]:
            # Softmax Top-K is a conventional routed branch. The optional shared
            # path is allowed so `topk_shared` matches Sparsegen+shared capacity.
            total_active_budget = int(CFG["moe_mlp_active_expert_budget"])
            if not 1 <= total_active_budget <= int(CFG["moe_mlp_language_experts"]):
                raise ValueError(
                    "HWVLM_MOE_ACTIVE_EXPERT_BUDGET must be in [1, language_experts] "
                    "for the dedicated Top-K router."
                )
            CFG["moe_mlp_routed_top_k"] = int(total_active_budget)
        else:
            if int(CFG["moe_mlp_active_expert_budget"]) != 0:
                raise ValueError(
                    "Native Sparsegen must use HWVLM_MOE_ACTIVE_EXPERT_BUDGET=0; "
                    "it has no hard Top-K cap."
                )
            # Passed to the module only as a valid placeholder; enforcement is off.
            CFG["moe_mlp_routed_top_k"] = int(CFG["moe_mlp_language_experts"])
        if CFG["moe_mlp_routing_fn"] not in {"sparsegen", "softmax"}:
            raise ValueError(f"Unsupported MoE routing function: {CFG['moe_mlp_routing_fn']!r}")
        if CFG["moe_mlp_router_temperature"] <= 0:
            raise ValueError("moe_mlp_router_temperature must be positive.")
        if not 0.0 <= CFG["moe_mlp_shared_mix"] < 1.0:
            raise ValueError("moe_mlp_shared_mix must lie in [0, 1).")
        if (
            CFG["moe_mlp_router_ce_coef"] < 0
            or CFG["moe_mlp_load_balance_coef"] < 0
            or CFG["moe_mlp_lam_sparse_coef"] < 0
            or not 0.0 <= CFG["moe_mlp_lam_sparse_warmup_fraction"] <= 1.0
            or not 0.0 <= CFG["moe_mlp_load_balance_warmup_fraction"] <= 1.0
        ):
            raise ValueError("MoE auxiliary-loss coefficients must be non-negative.")
        if CFG["moe_mlp_router_ce_coef"] != 0.0:
            raise ValueError("Router CE is intentionally disabled in this source-agnostic MoE ablation.")
        if CFG["moe_mlp_lam_sparse_coef"] > 0.0 and CFG["moe_mlp_routing_fn"] != "sparsegen":
            raise ValueError("LamSparseLoss is defined only for native Sparsegen routing.")
        if CFG["moe_mlp_expert_init"] not in {"copy", "noise"}:
            raise ValueError(f"Unsupported expert initialization: {CFG['moe_mlp_expert_init']!r}")
        if CFG["moe_mlp_expert_train_dtype"] not in {"match_base", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"}:
            raise ValueError(f"Unsupported expert dtype: {CFG['moe_mlp_expert_train_dtype']!r}")
        if not CFG["moe_mlp_layer_indices"]:
            raise ValueError("Canonical MoE run requires explicit layer indices.")
        if len(CFG["moe_mlp_layer_indices"]) != len(set(CFG["moe_mlp_layer_indices"])):
            raise ValueError(f"Duplicate MoE layer indices: {CFG['moe_mlp_layer_indices']}")
        if any(index < 0 for index in CFG["moe_mlp_layer_indices"]):
            raise ValueError(f"Negative MoE layer index: {CFG['moe_mlp_layer_indices']}")

    if CFG["use_wandb"] and wandb is None:
        LOGGER.warning("wandb is not installed. Disabling HWVLM_USE_WANDB for this run.")
        CFG["use_wandb"] = False

    os.makedirs(CFG["output_dir"], exist_ok=True)
    print("Configuration:")
    for k, v in CFG.items():
        print(f"  {k}: {v}")

    return export_runtime_locals(state, locals())
