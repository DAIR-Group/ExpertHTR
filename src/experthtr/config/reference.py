"""config.reference phase for the ExpertHTR research workflow.

This module preserves the audited runtime computation. It is invoked by the
explicit CLI workflow and exports its declared runtime objects into a
:class:`~experthtr.runtime.context.RuntimeState`.
"""

from __future__ import annotations

from experthtr.runtime.context import RuntimeState, export_runtime_locals


def configure(state: RuntimeState) -> RuntimeState:
    """Execute this phase after its prerequisites have populated ``state``."""

    globals().update(state.values)
    # Paper configuration: dense-initialized Sparsegen + shared + LamSparse + Switch balance.
    #
    # This workflow intentionally has a separate dense-init path. Do NOT put a
    # dense checkpoint in HWVLM_RESUME_FROM_CHECKPOINT: the original resume path
    # inserts MoE before strict loading and therefore cannot load dense MLP keys.

    import os
    from pathlib import Path

    OFFLINE_BUNDLE_ROOT = Path(os.getenv(
        "HWVLM_OFFLINE_BUNDLE_ROOT",
        "/kaggle/input/offline_bundle",
    ))
    WHEELHOUSE_PATH = OFFLINE_BUNDLE_ROOT / "wheelhouse"
    DATASET_NAME = "page_multitask"
    PROMPT_TEXT = ""  # Keep the canonical per-record prompts.
    EXCLUDED_SOURCES = "Eparchos"

    # -------------------------------------------------------------------------
    # User-editable experiment controls
    # -------------------------------------------------------------------------
    # The default is the paper-facing balance configuration. Other arms are retained only for controlled ablations. `dense_continuation` is the control
    # from the same dense checkpoint: "does MoE help beyond another dense epoch?"
    # `topk` is a separate conventional router: softmax -> fixed Top-K -> renormalize.
    # It is NOT Sparsegen plus Top-K. Router CE is disabled in every arm.
    MOE_ABLATION_VARIANT = "sparsegen_shared_lamsparse_balance"
    # Supported: dense_continuation, sparsegen, sparsegen_shared, topk, topk_shared,
    #            sparsegen_shared_lamsparse, sparsegen_shared_lamsparse_balance

    # HARD PATH: checkpoint DIRECTORY, not its output/checkpoints parent.
    # Example: /kaggle/input/<your-dense-checkpoint>/step_002301_cer_19.5838
    DENSE_INIT_CHECKPOINT_DIR = os.getenv("HWVLM_DENSE_INIT_CHECKPOINT", "").strip()

    # A dense checkpoint is required for the paper-facing continuation. `fresh_base`
    # remains available only as a separately named control and must not be compared
    # as a dense-warm-start result.
    INIT_MODE = "dense_checkpoint"  # dense_checkpoint | fresh_base
    DENSE_INIT_REQUIRE_RUN_CONFIG = True
    DENSE_INIT_HASH_FULL_STATE = True
    # Keep this False for the paper-facing direct comparison. Set it True only
    # for a separately documented transfer from a dense source-superset checkpoint;
    # all image, prompt, marker, context, and loss settings still must match.
    DENSE_INIT_ALLOW_SOURCE_SUPERSET = False
    DENSE_INIT_ALLOWED_EXTRA_SOURCES = ("Eparchos",)
    # Permit only an older checkpoint whose split counts were not serialized
    # (null for every split). A present-but-different count remains a hard error.
    DENSE_INIT_ALLOW_MISSING_SPLIT_RECORDS = True

    # Keep this identical across all ablation arms. False means full-model
    # one-epoch continuation; True freezes inherited base modules and trains
    # only newly inserted experts/router/lambda modules. Do not mix it across
    # arms. Default False follows the previous full-SFT pipeline.
    MOE_TRANSFER_FREEZE_BASE = False
    MOE_INIT_SEED = 20260830

    # Four routed experts are deliberately source/script agnostic. With EPARCHOS
    # excluded, no supervised group label remains in the objective.
    ROUTING_GROUPING = "unsupervised4"
    ROUTING_EXPERTS = {"unsupervised4": 4}
    MOE_LAYER_INDICES = (1, 5, 9, 13, 17, 21)
    ROUTER_CE_COEF = 0.0  # Invariant: router CE is disabled for every ablation arm.
    SHARED_EXPERT_MIX = 0.25
    # LamSparseLoss is evaluated only by the dedicated final ablation below.
    # It starts at zero and ramps to this coefficient over 10% of optimizer
    # updates so copied experts can first receive task gradients.
    LAM_SPARSE_COEF = 0.01
    LAM_SPARSE_WARMUP_FRACTION = 0.10
    # The Switch-style balance loss is only evaluated by the final balance arm.
    # Its raw scale is O(1), so 0.01 is deliberately a small initial contribution.
    # It shares the same short warm-up to avoid perturbing copied experts at step 0.
    LOAD_BALANCE_COEF = 0.01
    LOAD_BALANCE_WARMUP_FRACTION = 0.10

    # Fixed routed-expert count only for the conventional Softmax Top-K arms.
    # TOP_K=2 evaluates exactly two of the four routed MLPs per token.
    # Native Sparsegen arms are deliberately *not* hard-capped: their
    # active support is learned and therefore variable; report realized support.
    TOP_K = 2  # valid: 1..4; used only by MOE_ABLATION_VARIANT="topk"/"topk_shared"

    ABLATION_SPECS = {
        "dense_continuation": {
            "use_moe": False, "routing_fn": "none", "enforce_top_k": False,
            "router_ce": 0.0, "shared_experts": 0, "shared_mix": 0.0, "lam_sparse_coef": 0.0, "load_balance_coef": 0.0,
            "label": "dense-continuation-control",
        },
        "sparsegen": {
            "use_moe": True, "routing_fn": "sparsegen", "enforce_top_k": False,
            "router_ce": 0.0, "shared_experts": 0, "shared_mix": 0.0, "lam_sparse_coef": 0.0, "load_balance_coef": 0.0,
            "label": "native-sparsegen",
        },
        "sparsegen_shared": {
            "use_moe": True, "routing_fn": "sparsegen", "enforce_top_k": False,
            "router_ce": 0.0, "shared_experts": 1,
            "shared_mix": SHARED_EXPERT_MIX, "lam_sparse_coef": 0.0, "load_balance_coef": 0.0,
            "label": "native-sparsegen-shared",
        },
        "topk": {
            "use_moe": True, "routing_fn": "softmax", "enforce_top_k": True,
            "router_ce": 0.0, "shared_experts": 0, "shared_mix": 0.0, "lam_sparse_coef": 0.0, "load_balance_coef": 0.0,
            "label": "softmax-top-k",
        },
        "topk_shared": {
            "use_moe": True, "routing_fn": "softmax", "enforce_top_k": True,
            "router_ce": 0.0, "shared_experts": 1,
            "shared_mix": SHARED_EXPERT_MIX, "lam_sparse_coef": 0.0, "load_balance_coef": 0.0,
            "label": "softmax-top-k-shared",
        },
        # Final arm: isolate lambda sparsity regularization against
        # `sparsegen_shared` while retaining the intended 4-routed + 1-shared
        # deployment architecture.  The shared expert is always active; the
        # lambda loss regularizes only routed-expert support.
        "sparsegen_shared_lamsparse": {
            "use_moe": True, "routing_fn": "sparsegen", "enforce_top_k": False,
            "router_ce": 0.0, "shared_experts": 1, "shared_mix": SHARED_EXPERT_MIX,
            "lam_sparse_coef": LAM_SPARSE_COEF, "load_balance_coef": 0.0,
            "label": "native-sparsegen-shared-lambda-sparsity",
        },
        # Final architecture ablation: retain the shared expert and LamSparseLoss,
        # then test whether a small Switch-style routed-expert balance penalty
        # prevents collapse without overwhelming the HTR objective.
        "sparsegen_shared_lamsparse_balance": {
            "use_moe": True, "routing_fn": "sparsegen", "enforce_top_k": False,
            "router_ce": 0.0, "shared_experts": 1, "shared_mix": SHARED_EXPERT_MIX,
            "lam_sparse_coef": LAM_SPARSE_COEF, "load_balance_coef": LOAD_BALANCE_COEF,
            "label": "native-sparsegen-shared-lambda-sparsity-switch-balance",
        },
    }
    if MOE_ABLATION_VARIANT not in ABLATION_SPECS:
        raise ValueError(
            f"Unknown MOE_ABLATION_VARIANT={MOE_ABLATION_VARIANT!r}; "
            f"choose one of {sorted(ABLATION_SPECS)}"
        )
    if INIT_MODE not in {"dense_checkpoint", "fresh_base"}:
        raise ValueError("INIT_MODE must be 'dense_checkpoint' or 'fresh_base'.")
    if INIT_MODE == "dense_checkpoint" and not DENSE_INIT_CHECKPOINT_DIR.strip():
        raise ValueError(
            "Set DENSE_INIT_CHECKPOINT_DIR to the selected dense checkpoint directory, "
            "or set INIT_MODE='fresh_base' explicitly."
        )

    ABLATION = dict(ABLATION_SPECS[MOE_ABLATION_VARIANT])
    if ROUTING_GROUPING not in ROUTING_EXPERTS:
        raise ValueError(f"Unsupported ROUTING_GROUPING={ROUTING_GROUPING!r}")

    # -------------------------------------------------------------------------
    # Frozen common conditions. Change only in a new, clearly named experiment.
    # -------------------------------------------------------------------------
    TRAIN_EPOCHS = 1
    SCHEDULER_EPOCHS = 1
    BATCH_SIZE = 2
    GRAD_ACC = 16                       # effective batch = 32
    ADAPT_LR = "1e-6"                  # fresh optimizer after dense transfer
    # Fixed across ablation arms. The realised fraction is printed after the loader is built.
    WARMUP_STEPS = "40"

    RESOLUTION_PROFILE = "area_ratio_min589824_max1572864_ctx2560"
    IMAGE_MIN_PIXELS = "589824"
    IMAGE_MAX_PIXELS = "1572864"
    IMAGE_MAX_ASPECT_RATIO = "200"
    MAX_TEXT_LENGTH = "2560"
    EVAL_MAX_NEW_TOKENS = "1024"

    # The data/processor contract was audited previously. This bypass is valid
    # only when the strict dense-init contract check below succeeds. Set False
    # after any data/prompt/processor/resize/context change.
    # The active records keep the already-audited prompt/image/context contract;
    # EPARCHOS is only excluded. Set False after any record or processor change.
    REUSE_UNCHANGED_TOKEN_AUDIT = False
    ALLOW_EXISTING_OUTPUT_DIR = False

    init_path = Path(DENSE_INIT_CHECKPOINT_DIR).expanduser() if DENSE_INIT_CHECKPOINT_DIR.strip() else None
    init_tag = "freshbase" if INIT_MODE == "fresh_base" else init_path.name
    init_tag = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in init_tag)
    freeze_tag = "frozenbase" if MOE_TRANSFER_FREEZE_BASE else "fullbase"
    # Change this string for every retry that created any output artefact.
    # Never set ALLOW_EXISTING_OUTPUT_DIR=True merely to bypass this safeguard.
    RUN_SUFFIX = "paper_balance_r1"
    if not RUN_SUFFIX.strip() or any(ch in RUN_SUFFIX for ch in "/\\"):
        raise ValueError("RUN_SUFFIX must be a non-empty path-safe label.")
    RUN_TAG = (
        f"page_multitask_moe_ablation_v2_{MOE_ABLATION_VARIANT}_"
        f"{('topk' + str(TOP_K)) if ABLATION['enforce_top_k'] else 'native_sparsegen'}_routed{ROUTING_EXPERTS[ROUTING_GROUPING]}_shared{ABLATION['shared_experts']}_{INIT_MODE}_{init_tag}_"
        f"{freeze_tag}_l{'_'.join(map(str, MOE_LAYER_INDICES))}_bs{BATCH_SIZE}ga{GRAD_ACC}_"
        f"lr{ADAPT_LR}_epoch1_{RUN_SUFFIX.strip()}"
    )

    # Clear stale branch variables.  The dense checkpoint contributes weights
    # only; explicitly start a new optimizer/scheduler/epoch counter below.
    for _name in (
        "HWVLM_RESUME_FROM_CHECKPOINT", "HWVLM_CHECKPOINT_DIAGNOSTIC_DIR",
    ):
        os.environ.pop(_name, None)

    os.environ["HWVLM_EXCLUDED_SOURCES"] = EXCLUDED_SOURCES
    os.environ["HWVLM_ROUTING_GROUPING"] = ROUTING_GROUPING
    os.environ["HWVLM_NUM_EPOCHS"] = str(TRAIN_EPOCHS)
    os.environ["HWVLM_SCHEDULER_EPOCHS"] = str(SCHEDULER_EPOCHS)
    os.environ["HWVLM_RESUME_TRAINING_STATE"] = "0"
    os.environ["HWVLM_RESUME_COMPLETED_EPOCHS"] = "0"
    os.environ["HWVLM_BATCH_SIZE"] = str(BATCH_SIZE)
    os.environ["HWVLM_GRAD_ACC"] = str(GRAD_ACC)
    os.environ["HWVLM_LR"] = ADAPT_LR
    os.environ["HWVLM_WARMUP_STEPS"] = WARMUP_STEPS
    os.environ["HWVLM_RUN_NAME"] = RUN_TAG
    OUTPUT_ROOT = Path(os.getenv("HWVLM_OUTPUT_ROOT", "/kaggle/working/outputs")).expanduser()
    os.environ["HWVLM_OUTPUT_DIR"] = str(OUTPUT_ROOT / RUN_TAG)
    os.environ["HWVLM_EVAL_EPOCHS"] = "1.0"
    os.environ["HWVLM_EARLY_STOPPING_EVALS"] = "1"
    os.environ["HWVLM_KEEP_TOP_CHECKPOINTS"] = "1"
    os.environ["HWVLM_WRITE_EVAL_PREDICTIONS"] = "1"
    os.environ["HWVLM_RUN_FINAL_TEST"] = "0"
    os.environ["HWVLM_RUN_FULL_TOKEN_AUDIT"] = "0" if REUSE_UNCHANGED_TOKEN_AUDIT else "1"
    os.environ["HWVLM_SKIP_TOKEN_AUDIT_GATE"] = "1" if REUSE_UNCHANGED_TOKEN_AUDIT else "0"

    os.environ["HWVLM_IMAGE_RESIZE_POLICY"] = "area_ratio"
    os.environ["HWVLM_IMAGE_MIN_PIXELS"] = IMAGE_MIN_PIXELS
    os.environ["HWVLM_IMAGE_MAX_PIXELS"] = IMAGE_MAX_PIXELS
    os.environ["HWVLM_IMAGE_MAX_ASPECT_RATIO"] = IMAGE_MAX_ASPECT_RATIO
    os.environ["HWVLM_MAX_TEXT_LENGTH"] = MAX_TEXT_LENGTH
    os.environ["HWVLM_EVAL_MAX_NEW_TOKENS"] = EVAL_MAX_NEW_TOKENS
    os.environ["HWVLM_RESOLUTION_PROFILE"] = RESOLUTION_PROFILE
    os.environ["HWVLM_SAMPLING_MODE"] = "full_pass"
    os.environ["HWVLM_USE_SOURCE_BALANCED_SAMPLER"] = "0"
    os.environ["HWVLM_USE_GROUP_BALANCED_SAMPLER"] = "0"
    os.environ["HWVLM_MICROBATCH_INVARIANT_TOKEN_LOSS"] = "1"
    os.environ["HWVLM_ATOMIC_OCR_MARKERS"] = "<del>,<gap>"
    os.environ["HWVLM_PAGE_HTR_EOS_WEIGHT"] = "4.0"
    os.environ["HWVLM_TOKEN_UL_COEF"] = "0.0"
    os.environ["HWVLM_TOKEN_UL_PAGE_FRACTION"] = "0.0"
    os.environ["HWVLM_TOKEN_UL_CONTEXT_WINDOW"] = "64"
    os.environ["HWVLM_TOKEN_UL_MIN_POSITION"] = "128"
    os.environ["HWVLM_LOOP_NGRAM"] = "3"
    os.environ["HWVLM_LOOP_WINDOW"] = "96"
    os.environ["HWVLM_LOOP_THRESHOLD"] = "0.60"
    os.environ["HWVLM_LOOP_MIN_TOKENS"] = "128"
    os.environ["HWVLM_LOOP_CONSECUTIVE"] = "2"

    os.environ["HWVLM_USE_MOE_MLP"] = "1" if ABLATION["use_moe"] else "0"
    os.environ["HWVLM_MOE_LANGUAGE_EXPERTS"] = str(ROUTING_EXPERTS[ROUTING_GROUPING])
    os.environ["HWVLM_MOE_SHARED_EXPERTS"] = str(ABLATION["shared_experts"])
    os.environ["HWVLM_MOE_SHARED_MIX"] = str(ABLATION["shared_mix"])
    os.environ["HWVLM_MOE_ACTIVE_EXPERT_BUDGET"] = str(TOP_K if ABLATION["enforce_top_k"] else 0)
    os.environ["HWVLM_MOE_ENFORCE_TOP_K"] = "1" if ABLATION["enforce_top_k"] else "0"
    os.environ["HWVLM_MOE_LAYER_INDICES"] = ",".join(map(str, MOE_LAYER_INDICES))
    os.environ["HWVLM_MOE_ROUTING_FN"] = str(ABLATION["routing_fn"])
    os.environ["HWVLM_MOE_ROUTER_CE_COEF"] = str(ABLATION["router_ce"])
    os.environ["HWVLM_MOE_LOAD_BALANCE_COEF"] = str(ABLATION.get("load_balance_coef", 0.0))
    os.environ["HWVLM_MOE_LOAD_BALANCE_WARMUP_FRACTION"] = str(LOAD_BALANCE_WARMUP_FRACTION)
    os.environ["HWVLM_MOE_LAM_SPARSE_COEF"] = str(ABLATION.get("lam_sparse_coef", 0.0))
    os.environ["HWVLM_MOE_LAM_SPARSE_WARMUP_FRACTION"] = str(LAM_SPARSE_WARMUP_FRACTION)
    os.environ["HWVLM_MOE_ROUTER_TEMPERATURE"] = "1.0"
    os.environ["HWVLM_MOE_SPARSEGEN_HIDDEN"] = "128"
    os.environ["HWVLM_MOE_SPARSEGEN_EPS"] = "0.01"
    os.environ["HWVLM_ROUTER_CE_CLASS_WEIGHTING"] = "none"
    os.environ["HWVLM_MOE_EXPERT_INIT"] = "copy"
    os.environ["HWVLM_MOE_EXPERT_TRAIN_DTYPE"] = "match_base"
    os.environ["HWVLM_MOE_MLP_FREEZE_BASE"] = "1" if MOE_TRANSFER_FREEZE_BASE else "0"
    os.environ["HWVLM_GRADIENT_CHECKPOINTING"] = "1"
    os.environ["HWVLM_USE_AUGMENTATION"] = "0"
    os.environ["HWVLM_USE_WANDB"] = "0"
    os.environ["HWVLM_NUM_WORKERS"] = "16"
    os.environ["HWVLM_TORCH_DTYPE"] = "bf16"
    os.environ["HWVLM_USE_DATA_PARALLEL"] = "0"
    os.environ["HWVLM_SEED"] = "42"
    os.environ["HWVLM_WEIGHT_DECAY"] = "0.01"
    os.environ["HWVLM_GRAD_CLIP"] = "1.0"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["WANDB_MODE"] = "offline"

    print("Ablation variant:", MOE_ABLATION_VARIANT, "|", ABLATION["label"])
    print("Initialization:", INIT_MODE, "| dense checkpoint:", DENSE_INIT_CHECKPOINT_DIR or "(not used)")
    if INIT_MODE == "fresh_base":
        print("WARNING: fresh_base is a separate from-scratch comparison, not a dense-warm-start ablation.")
    print("MoE / routed / shared / router / CE / balance / LamSparse:", ABLATION["use_moe"], ROUTING_EXPERTS[ROUTING_GROUPING], ABLATION["shared_experts"], ABLATION["routing_fn"], ABLATION["router_ce"], ABLATION.get("load_balance_coef", 0.0), ABLATION.get("lam_sparse_coef", 0.0))
    if ABLATION["enforce_top_k"]:
        print("Fixed Top-K routed experts per token:", TOP_K)
    else:
        print("Native Sparsegen support: no hard Top-K; realized active support is logged.")
    print("Freeze inherited base:", MOE_TRANSFER_FREEZE_BASE, "| MoE init seed:", MOE_INIT_SEED)
    print("Epochs / scheduler / effective batch / LR / warmup:", TRAIN_EPOCHS, SCHEDULER_EPOCHS, BATCH_SIZE * GRAD_ACC, ADAPT_LR, WARMUP_STEPS)
    print("Resolution / context:", RESOLUTION_PROFILE, MAX_TEXT_LENGTH)
    print("Output:", os.environ["HWVLM_OUTPUT_DIR"])

    return export_runtime_locals(state, locals())
