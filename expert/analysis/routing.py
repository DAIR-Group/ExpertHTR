"""analysis.routing phase for the ExpertHTR research workflow.

This module preserves the audited runtime computation. It is invoked by the
explicit CLI workflow and exports its declared runtime objects into a
:class:`~expert.runtime.context.RuntimeState`.
"""

from __future__ import annotations

from expert.runtime.context import RuntimeState, export_runtime_locals
from expert.evaluation.checkpoint_contract import validate_atomic_marker_contract


def run_routing_analysis(state: RuntimeState) -> RuntimeState:
    """Execute this phase after its prerequisites have populated ``state``."""

    globals().update(state.values)
    # =============================================================================
    # Sparsegen routing visualizations on a deterministic stratified TEST sample
    # =============================================================================
    # This is a teacher-forced routing probe, not generation. It measures routing at
    # every attended input position: vision tokens, prompt/context tokens, and
    # transcription tokens. Padding is excluded by attention_mask=0.
    #
    # IMPORTANT: this tuple must match the checkpoint exactly. For an existing
    # checkpoint using (0, 4, 8, 12, 16, 20), change this tuple exactly; never
    # relabel that architecture as (1, 5, 9, 13, 17, 21) in a figure.
    ROUTING_ANALYSIS_LAYER_INDICES = (1, 5, 9, 13, 17, 21)
    ROUTING_ANALYSIS_ROUTED_EXPERTS = 4
    ROUTING_ANALYSIS_PAGES_PER_SOURCE = 50
    ROUTING_ANALYSIS_SEED = 20260901
    ROUTING_ANALYSIS_BATCH_SIZE = 1
    ROUTING_ANALYSIS_BOOTSTRAP_REPLICATES = 2000  # page-resampled 95% CI for E[K].
    # Empty -> output_dir/run_summary.json -> validation-selected checkpoint.
    ROUTING_ANALYSIS_CHECKPOINT_DIR = os.getenv("HWVLM_ROUTING_ANALYSIS_CHECKPOINT", "").strip()
    # Empty -> derive <run_dir>/routing_visualizations_test from the checkpoint.
    # Set an absolute path only when you deliberately want another destination.
    ROUTING_ANALYSIS_OUTPUT_DIR = os.getenv("HWVLM_ROUTING_ANALYSIS_OUTPUT", "").strip()

    import csv
    import logging
    import re
    from contextlib import nullcontext
    from collections import defaultdict
    from IPython.display import display
    import matplotlib.pyplot as plt

    if not CFG.get("use_moe_mlp", False):
        raise RuntimeError("Routing visualizations require an MoE model; the active runtime is dense.")
    if str(CFG.get("moe_mlp_routing_fn", "")).lower() != "sparsegen":
        raise RuntimeError(
            "Support-size K=1..K=4 is the Sparsegen experiment. Reload a Sparsegen checkpoint; "
            f"current routing={CFG.get('moe_mlp_routing_fn')!r}."
        )
    if bool(CFG.get("moe_mlp_enforce_top_k", False)):
        raise RuntimeError(
            "This probe rejects the hard-Top-K ablation: fixed support cannot test "
            "Sparsegen's dynamic support size."
        )
    if int(ROUTING_ANALYSIS_BATCH_SIZE) != 1:
        raise ValueError("Use batch_size=1 so every captured routing tensor belongs to one source/page.")
    if int(ROUTING_ANALYSIS_PAGES_PER_SOURCE) < 1:
        raise ValueError("ROUTING_ANALYSIS_PAGES_PER_SOURCE must be positive.")
    if int(ROUTING_ANALYSIS_BOOTSTRAP_REPLICATES) < 100:
        raise ValueError("ROUTING_ANALYSIS_BOOTSTRAP_REPLICATES must be >=100.")

    routing_model = unwrap_model(model)
    if isinstance(model, nn.DataParallel):
        raise RuntimeError("Routing hooks are intentionally disabled under DataParallel; run this probe on one GPU.")

    # Strictly restore the checkpoint under study. Thus the visual evidence is tied
    # to validation-selected weights rather than whichever in-memory step happened last.
    if str(ROUTING_ANALYSIS_CHECKPOINT_DIR).strip():
        routing_checkpoint = Path(ROUTING_ANALYSIS_CHECKPOINT_DIR).expanduser()
    else:
        if "output_dir" not in globals():
            raise RuntimeError(
                "ROUTING_ANALYSIS_CHECKPOINT_DIR is empty and output_dir is unavailable. "
                "Set an explicit analysis checkpoint path or run `experthtr train` first."
            )
        routing_summary_path = Path(output_dir) / "run_summary.json"
        if not routing_summary_path.exists():
            raise FileNotFoundError(
                f"Missing {routing_summary_path}; set ROUTING_ANALYSIS_CHECKPOINT_DIR explicitly."
            )
        routing_summary = json.loads(routing_summary_path.read_text(encoding="utf-8"))
        routing_best = routing_summary.get("best_checkpoint") or {}
        routing_best_path = routing_best.get("path") if isinstance(routing_best, dict) else None
        if not routing_best_path:
            raise RuntimeError("run_summary.json has no validation-selected best_checkpoint.")
        routing_checkpoint = Path(routing_best_path)

    if str(ROUTING_ANALYSIS_OUTPUT_DIR).strip():
        routing_output_dir = Path(ROUTING_ANALYSIS_OUTPUT_DIR).expanduser()
    elif "output_dir" in globals() and not str(Path(output_dir)).startswith("/kaggle/input/"):
        routing_output_dir = Path(output_dir) / "routing_visualizations_test"
    else:
        # /kaggle/input is read-only. Keep results in the writable runtime and
        # namespace them by run directory so probes from two checkpoints do not mix.
        routing_writable_root = Path(globals().get("KAGGLE_WORKING", Path.cwd()))
        routing_output_dir = (
            routing_writable_root / "routing_visualizations_test" / routing_checkpoint.parent.parent.name
        )
    try:
        routing_output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(
            f"Routing output directory is not writable: {routing_output_dir}. "
            "Set ROUTING_ANALYSIS_OUTPUT_DIR to a writable path such as /kaggle/working/routing_visualizations_test."
        ) from exc

    for _required_name in (CHECKPOINT_COMPLETE_FILE, CHECKPOINT_MANIFEST_FILE, FULL_MODEL_STATE_FILE):
        if not (routing_checkpoint / _required_name).exists():
            raise FileNotFoundError(routing_checkpoint / _required_name)
    routing_manifest = json.loads(
        (routing_checkpoint / CHECKPOINT_MANIFEST_FILE).read_text(encoding="utf-8")
    )
    routing_checkpoint_signature = dict(routing_manifest.get("moe_signature") or {})
    if not routing_checkpoint_signature:
        raise RuntimeError("Checkpoint manifest has no moe_signature; cannot verify the routing architecture.")
    validate_atomic_marker_contract(routing_manifest, CFG)
    try:
        _routing_payload = torch.load(
            routing_checkpoint / FULL_MODEL_STATE_FILE, map_location="cpu", weights_only=True
        )
    except TypeError:
        _routing_payload = torch.load(routing_checkpoint / FULL_MODEL_STATE_FILE, map_location="cpu")
    routing_model.load_state_dict(_routing_payload.get("model_state_dict", _routing_payload), strict=True)
    del _routing_payload
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Discover actual MoE modules and fail rather than silently mislabelling layers.
    _layer_pattern = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp$")
    routing_modules = {}
    for _module_name, _module in routing_model.named_modules():
        if not isinstance(_module, LanguageGroupedMoEMLP):
            continue
        _match = _layer_pattern.search(_module_name)
        if _match is None:
            raise RuntimeError(f"Cannot recover decoder layer index from MoE module name: {_module_name!r}")
        _layer_index = int(_match.group(1))
        if _layer_index in routing_modules:
            raise RuntimeError(f"Duplicate MoE module at decoder layer {_layer_index}.")
        routing_modules[_layer_index] = _module
    actual_routing_layers = tuple(sorted(routing_modules))
    if actual_routing_layers != tuple(ROUTING_ANALYSIS_LAYER_INDICES):
        raise RuntimeError(
            "Requested visualization layers differ from the loaded checkpoint: "
            f"requested={tuple(ROUTING_ANALYSIS_LAYER_INDICES)} actual={actual_routing_layers}. "
            "Set ROUTING_ANALYSIS_LAYER_INDICES to the actual checkpoint placement."
        )
    for _layer_index, _module in routing_modules.items():
        if len(_module.language_experts) != int(ROUTING_ANALYSIS_ROUTED_EXPERTS):
            raise RuntimeError(
                f"Layer {_layer_index} has {len(_module.language_experts)} routed experts; "
                f"expected {ROUTING_ANALYSIS_ROUTED_EXPERTS}."
            )
        if str(_module.routing_fn).lower() != "sparsegen" or bool(_module.enforce_top_k):
            raise RuntimeError(f"Layer {_layer_index} is not unconstrained Sparsegen.")

    # Training/checkpoint registration normally defines _moe_signature(), but this probe must also run
    # independently. Verify the route-changing structural
    # fields locally; strict state_dict loading above verifies parameter topology.
    _first_routing_module = routing_modules[actual_routing_layers[0]]
    routing_runtime_signature = {
        "enabled": True,
        "routing_grouping": str(CFG["routing_grouping"]),
        "language_groups": list(LANGUAGE_GROUPS),
        "language_experts": len(_first_routing_module.language_experts),
        "shared_experts": len(_first_routing_module.shared_experts),
        "shared_mix": float(_first_routing_module.shared_mix),
        "layer_indices": list(actual_routing_layers),
        "routing_fn": str(_first_routing_module.routing_fn),
        "router_temperature": float(_first_routing_module.router_temperature),
        "sparsegen_eps": float(_first_routing_module.sparsegen_eps),
        "expert_init": str(_first_routing_module.expert_init),
        "expert_train_dtype": str(_first_routing_module.expert_train_dtype),
    }
    _routing_signature_mismatches = {}
    for _key, _runtime_value in routing_runtime_signature.items():
        _checkpoint_value = routing_checkpoint_signature.get(_key)
        if isinstance(_runtime_value, float):
            _matches = isinstance(_checkpoint_value, (int, float)) and math.isclose(
                float(_checkpoint_value), _runtime_value, rel_tol=0.0, abs_tol=1e-12
            )
        else:
            _matches = _checkpoint_value == _runtime_value
        if not _matches:
            _routing_signature_mismatches[_key] = {
                "checkpoint": _checkpoint_value, "runtime": _runtime_value
            }
    if _routing_signature_mismatches:
        raise RuntimeError(
            "Routing checkpoint differs in route-changing structural fields: "
            + json.dumps(_routing_signature_mismatches, ensure_ascii=False, default=str)
        )

    # Build a page_htr-only test probe, then sample within each source without replacement.
    routing_test_path = resolve_split_path(DATA_ROOT, CFG["dataset"], "test")
    if routing_test_path is None:
        raise FileNotFoundError(f"No test JSONL found for dataset={CFG['dataset']!r} under {DATA_ROOT}")
    routing_test_dataset = OCRJsonlDataset(
        jsonl_path=routing_test_path, image_root=IMAGE_ROOT, prompt_text=None, augment=False
    )
    routing_test_dataset = _exclude_sources_inplace(
        routing_test_dataset, CFG["excluded_sources"], "Routing visualization test"
    )
    routing_test_dataset = _exclude_tasks_inplace(
        routing_test_dataset, CFG["excluded_valid_tasks"], "Routing visualization test"
    )
    routing_raw_samples = _dataset_samples_for_stats(routing_test_dataset)
    if set(_count_sources(routing_raw_samples)) != set(CFG["expected_sources"]):
        raise RuntimeError("Routing-visualization source scope differs from the active configuration.")
    if set(_count_tasks(routing_raw_samples)) != {"page_htr"}:
        raise RuntimeError("Routing visualizations require page_htr-only test records.")

    routing_indices_by_source = defaultdict(list)
    for _index, _sample in enumerate(routing_raw_samples):
        _source = str((_sample.get("metadata") or {}).get("source_dataset", "unknown"))
        routing_indices_by_source[_source].append(_index)

    routing_selected_indices = []
    routing_sample_manifest = {}
    for _source in sorted(CFG["expected_sources"]):
        _available = sorted(routing_indices_by_source.get(_source, []))
        if not _available:
            raise RuntimeError(f"No test page_htr records for source={_source!r}.")
        _take = min(int(ROUTING_ANALYSIS_PAGES_PER_SOURCE), len(_available))
        _rng = random.Random(f"{ROUTING_ANALYSIS_SEED}::{_source}")
        _chosen = sorted(_rng.sample(_available, _take))
        routing_selected_indices.extend(_chosen)
        routing_sample_manifest[_source] = {
            "available_test_pages": len(_available),
            "selected_pages": _take,
            "sampling": "all_pages" if _take == len(_available) else "uniform_without_replacement",
            "record_ids": [str(routing_raw_samples[i].get("id", "")) for i in _chosen],
        }
    routing_selected_indices = sorted(routing_selected_indices)
    routing_probe_dataset = torch.utils.data.Subset(routing_test_dataset, routing_selected_indices)
    routing_probe_loader = DataLoader(
        routing_probe_dataset,
        batch_size=ROUTING_ANALYSIS_BATCH_SIZE,
        shuffle=False,
        num_workers=int(CFG["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
        persistent_workers=bool(int(CFG["num_workers"]) > 0),
    )

    class _RoutingAccumulator:
        def __init__(self, sources, layers, experts):
            self.layers = tuple(int(value) for value in layers)
            self.experts = int(experts)
            self.data = {
                str(source): {
                    int(layer): {
                        "tokens": 0,
                        "support_counts": [0] * (self.experts + 1),
                        "expert_active_counts": [0] * self.experts,
                        "expert_mass_sums": [0.0] * self.experts,
                        # One entry/page supports a page-resampled CI without
                        # incorrectly treating correlated tokens as i.i.d.
                        "page_token_counts": [],
                        "page_support_sums": [],
                    }
                    for layer in self.layers
                }
                for source in sources
            }

        def add(self, source, layer, weights):
            if source not in self.data or layer not in self.data[source]:
                raise RuntimeError(f"Unexpected routing capture source/layer: {source!r}/{layer}.")
            if weights.ndim != 2 or weights.shape[1] != self.experts:
                raise RuntimeError(f"Unexpected routing tensor shape at layer {layer}: {tuple(weights.shape)}")
            weights = weights.detach().float()
            if not torch.isfinite(weights).all():
                raise RuntimeError(f"Non-finite routing weights captured at layer {layer}.")
            active = weights > 0
            support = active.sum(dim=-1)
            if support.numel() == 0 or int(support.min()) < 1 or int(support.max()) > self.experts:
                raise RuntimeError(
                    f"Invalid Sparsegen support at layer {layer}: "
                    f"min={int(support.min()) if support.numel() else None}, "
                    f"max={int(support.max()) if support.numel() else None}"
                )
            bucket = self.data[source][layer]
            bucket["tokens"] += int(support.numel())
            support_counts = torch.bincount(
                support.to(torch.long).cpu(), minlength=self.experts + 1
            ).tolist()
            bucket["support_counts"] = [
                old + int(new) for old, new in zip(bucket["support_counts"], support_counts)
            ]
            activation_counts = active.sum(dim=0).to(torch.long).cpu().tolist()
            bucket["expert_active_counts"] = [
                old + int(new) for old, new in zip(bucket["expert_active_counts"], activation_counts)
            ]
            mass_sums = weights.sum(dim=0).cpu().tolist()
            bucket["expert_mass_sums"] = [
                float(old) + float(new) for old, new in zip(bucket["expert_mass_sums"], mass_sums)
            ]
            bucket["page_token_counts"].append(int(support.numel()))
            bucket["page_support_sums"].append(float(support.sum().item()))

        def pooled(self):
            result = {
                int(layer): {
                    "tokens": 0,
                    "support_counts": [0] * (self.experts + 1),
                    "expert_active_counts": [0] * self.experts,
                    "expert_mass_sums": [0.0] * self.experts,
                    "page_token_counts": [],
                    "page_support_sums": [],
                }
                for layer in self.layers
            }
            for source_data in self.data.values():
                for layer, bucket in source_data.items():
                    result[layer]["tokens"] += int(bucket["tokens"])
                    result[layer]["support_counts"] = [
                        old + new
                        for old, new in zip(result[layer]["support_counts"], bucket["support_counts"])
                    ]
                    result[layer]["expert_active_counts"] = [
                        old + new
                        for old, new in zip(
                            result[layer]["expert_active_counts"], bucket["expert_active_counts"]
                        )
                    ]
                    result[layer]["expert_mass_sums"] = [
                        float(old) + float(new)
                        for old, new in zip(result[layer]["expert_mass_sums"], bucket["expert_mass_sums"])
                    ]
                    result[layer]["page_token_counts"].extend(bucket["page_token_counts"])
                    result[layer]["page_support_sums"].extend(bucket["page_support_sums"])
            return result

    routing_accumulator = _RoutingAccumulator(
        sources=sorted(CFG["expected_sources"]),
        layers=ROUTING_ANALYSIS_LAYER_INDICES,
        experts=ROUTING_ANALYSIS_ROUTED_EXPERTS,
    )
    routing_runtime = {"source": None, "record_id": None, "all_attended_mask": None}
    routing_original_methods = {}

    # Capture native p before MLP dispatch. Hard Top-K is forbidden above, so p>0
    # is exactly the Sparsegen support used for routed MLP compute at all attended
    # positions (vision + prompt/context + transcription; no padding).
    def _make_routing_capture(original_method, layer_index):
        def _captured_routing_weight(logits, x):
            weights, lam = original_method(logits, x)
            all_attended_mask = routing_runtime["all_attended_mask"]
            source = routing_runtime["source"]
            if all_attended_mask is None or source is None:
                raise RuntimeError("Routing capture was called outside the prepared page context.")
            if weights.ndim != 3 or tuple(weights.shape[:2]) != tuple(all_attended_mask.shape):
                raise RuntimeError(
                    f"Cannot align routing positions with attention_mask at layer {layer_index}: "
                    f"weights={tuple(weights.shape)} attention_mask={tuple(all_attended_mask.shape)}"
                )
            selected_weights = weights[all_attended_mask.to(device=weights.device)]
            routing_accumulator.add(source, layer_index, selected_weights)
            return weights, lam
        return _captured_routing_weight

    for _layer_index, _module in routing_modules.items():
        routing_original_methods[_layer_index] = _module._routing_weight
        _module._routing_weight = _make_routing_capture(
            routing_original_methods[_layer_index], _layer_index
        )

    routing_was_training = bool(routing_model.training)
    routing_model.eval()
    routing_skip_keys = {
        "texts", "prompts", "ids", "metadata", "source_datasets", "tasks", "languages", "page_keys",
        "language_groups", "resize_scales", "original_sizes", "resized_sizes",
    }
    try:
        with torch.no_grad():
            for _batch in tqdm(routing_probe_loader, desc="routing probe", leave=False):
                _source_values = list(_batch.get("source_datasets", []))
                _ids = list(_batch.get("ids", []))
                if len(_source_values) != 1 or len(_ids) != 1:
                    raise RuntimeError("Routing probe requires exactly one source/page per batch.")
                _batch_device = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in _batch.items()
                    if not key.startswith("prompt_") and key not in routing_skip_keys
                }
                _attention_mask = _batch_device.get("attention_mask")
                if not isinstance(_attention_mask, torch.Tensor) or _attention_mask.ndim != 2:
                    raise RuntimeError("Routing probe received no rank-2 attention_mask tensor.")
                routing_runtime.update({
                    "source": str(_source_values[0]),
                    "record_id": str(_ids[0]),
                    "all_attended_mask": _attention_mask.bool(),
                })
                _autocast = (
                    torch.amp.autocast("cuda", dtype=get_model_dtype(device))
                    if device.type == "cuda" and get_model_dtype(device) != torch.float32
                    else nullcontext()
                )
                with _autocast:
                    _ = routing_model(**_batch_device)
    finally:
        for _layer_index, _module in routing_modules.items():
            _module._routing_weight = routing_original_methods[_layer_index]
        routing_runtime.update({"source": None, "record_id": None, "all_attended_mask": None})
        routing_model.train(routing_was_training)

    def _routing_layer_report(bucket):
        tokens = int(bucket["tokens"])
        if tokens <= 0:
            raise RuntimeError("No attended routing positions were captured.")
        support_counts = [int(value) for value in bucket["support_counts"]]
        expert_counts = [int(value) for value in bucket["expert_active_counts"]]
        expert_mass_sums = [float(value) for value in bucket["expert_mass_sums"]]
        if sum(support_counts[1:]) != tokens:
            raise RuntimeError(f"Support-count invariant failed: {sum(support_counts[1:])} != {tokens}")
        mean_support = sum(k * support_counts[k] for k in range(1, ROUTING_ANALYSIS_ROUTED_EXPERTS + 1)) / tokens
        activation_total = sum(expert_counts) / tokens
        if not math.isclose(mean_support, activation_total, rel_tol=0.0, abs_tol=1e-6):
            raise RuntimeError(
                f"Activation/support invariant failed: E[K]={mean_support} vs sum_e P(active)={activation_total}"
            )
        return {
            "all_attended_positions": tokens,
            "mean_active_routed_experts": mean_support,
            "support_counts": {
                f"K={k}": support_counts[k] for k in range(1, ROUTING_ANALYSIS_ROUTED_EXPERTS + 1)
            },
            "support_percent": {
                f"K={k}": 100.0 * support_counts[k] / tokens
                for k in range(1, ROUTING_ANALYSIS_ROUTED_EXPERTS + 1)
            },
            "expert_active_counts": {
                f"E{expert}": expert_counts[expert - 1]
                for expert in range(1, ROUTING_ANALYSIS_ROUTED_EXPERTS + 1)
            },
            "expert_activation_percent": {
                f"E{expert}": 100.0 * expert_counts[expert - 1] / tokens
                for expert in range(1, ROUTING_ANALYSIS_ROUTED_EXPERTS + 1)
            },
            "expert_routing_mass_percent": {
                f"E{expert}": 100.0 * expert_mass_sums[expert - 1] / tokens
                for expert in range(1, ROUTING_ANALYSIS_ROUTED_EXPERTS + 1)
            },
        }

    routing_report_by_source = {
        source: {str(layer): _routing_layer_report(bucket) for layer, bucket in sorted(layer_data.items())}
        for source, layer_data in sorted(routing_accumulator.data.items())
    }
    routing_report_pooled = {
        str(layer): _routing_layer_report(bucket)
        for layer, bucket in sorted(routing_accumulator.pooled().items())
    }

    routing_csv_rows = []
    for _scope, _report in list(routing_report_by_source.items()) + [
        ("All_sources_pooled", routing_report_pooled)
    ]:
        for _layer_text, _row in sorted(_report.items(), key=lambda item: int(item[0])):
            _csv_row = {
                "scope": _scope,
                "layer": int(_layer_text),
                "all_attended_positions": _row["all_attended_positions"],
                "mean_active_routed_experts": _row["mean_active_routed_experts"],
            }
            for _key, _value in _row["support_counts"].items():
                _csv_row[f"support_count_{_key.replace('=', '')}"] = _value
            for _key, _value in _row["support_percent"].items():
                _csv_row[f"support_pct_{_key.replace('=', '')}"] = _value
            for _key, _value in _row["expert_active_counts"].items():
                _csv_row[f"activation_count_{_key}"] = _value
            for _key, _value in _row["expert_activation_percent"].items():
                _csv_row[f"activation_pct_{_key}"] = _value
            for _key, _value in _row["expert_routing_mass_percent"].items():
                _csv_row[f"routing_mass_pct_{_key}"] = _value
            routing_csv_rows.append(_csv_row)

    routing_csv_path = routing_output_dir / "routing_support_and_utilization.csv"
    with routing_csv_path.open("w", newline="", encoding="utf-8") as _handle:
        _writer = csv.DictWriter(_handle, fieldnames=list(routing_csv_rows[0]))
        _writer.writeheader()
        _writer.writerows(routing_csv_rows)

    def _safe_figure_name(text):
        return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")

    def _plot_legacy_routing_summary(scope, report, page_count, output_path, pooled=False):
        layers = [int(layer) for layer in ROUTING_ANALYSIS_LAYER_INDICES]
        support = np.array([
            [report[str(layer)]["support_percent"][f"K={k}"]
             for k in range(1, ROUTING_ANALYSIS_ROUTED_EXPERTS + 1)]
            for layer in layers
        ], dtype=float)
        activation = np.array([
            [report[str(layer)]["expert_activation_percent"][f"E{expert}"]
             for expert in range(1, ROUTING_ANALYSIS_ROUTED_EXPERTS + 1)]
            for layer in layers
        ], dtype=float)
        token_positions = sum(
            report[str(layer)]["target_token_positions"] for layer in layers
        )

        plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold"})
        figure, (ax_support, ax_heatmap) = plt.subplots(
            1, 2, figsize=(15.5, 6.5), gridspec_kw={"width_ratios": [1.35, 1]}
        )
        figure.patch.set_facecolor("#FBFCFE")

        support_colors = ["#2A6FBB", "#37A987", "#F3B33D", "#D95F76"]
        bottoms = np.zeros(len(layers), dtype=float)
        x_positions = np.arange(len(layers))
        for column, k in enumerate(range(1, ROUTING_ANALYSIS_ROUTED_EXPERTS + 1)):
            bars = ax_support.bar(
                x_positions, support[:, column], bottom=bottoms, width=0.72,
                color=support_colors[column], edgecolor="white", linewidth=1.1, label=f"K={k}",
            )
            for bar, value, bottom in zip(bars, support[:, column], bottoms):
                if value >= 6.0:
                    ax_support.text(
                        bar.get_x() + bar.get_width() / 2.0, bottom + value / 2.0, f"{value:.0f}%",
                        ha="center", va="center", fontsize=9, color="white", fontweight="bold",
                    )
            bottoms += support[:, column]
        ax_support.set_title("Sparsegen support size")
        ax_support.set_xlabel("MoE decoder layer")
        ax_support.set_ylabel("Target-token positions (%)")
        ax_support.set_xticks(x_positions, [str(layer) for layer in layers])
        ax_support.set_ylim(0, 100)
        ax_support.grid(axis="y", color="#DDE3EA", linewidth=0.8, alpha=0.8)
        ax_support.set_axisbelow(True)
        ax_support.spines[["top", "right"]].set_visible(False)
        ax_support.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False)

        image = ax_heatmap.imshow(activation, cmap="YlGnBu", vmin=0.0, vmax=100.0, aspect="auto")
        ax_heatmap.set_title("Routed-expert activation")
        ax_heatmap.set_xlabel("Routed expert")
        ax_heatmap.set_ylabel("MoE decoder layer")
        ax_heatmap.set_xticks(
            range(ROUTING_ANALYSIS_ROUTED_EXPERTS),
            [f"E{expert}" for expert in range(1, ROUTING_ANALYSIS_ROUTED_EXPERTS + 1)],
        )
        ax_heatmap.set_yticks(range(len(layers)), [str(layer) for layer in layers])
        for row in range(activation.shape[0]):
            for column in range(activation.shape[1]):
                value = activation[row, column]
                ax_heatmap.text(
                    column, row, f"{value:.1f}", ha="center", va="center", fontsize=10,
                    color="white" if value >= 55.0 else "#17324D", fontweight="bold",
                )
        colorbar = figure.colorbar(image, ax=ax_heatmap, fraction=0.05, pad=0.04)
        colorbar.set_label(r"$P(p_{\ell,e}>0)$ (%)")
        ax_heatmap.tick_params(length=0)
        for spine in ax_heatmap.spines.values():
            spine.set_visible(False)

        source_note = "pooled across sources by target-token positions" if pooled else "sampled within this source"
        figure.suptitle(
            f"{scope} | {page_count} test pages | {token_positions:,} target-token/layer positions",
            fontsize=16, fontweight="bold", color="#162B45", y=0.99,
        )
        figure.text(
            0.5, 0.015,
            "Teacher-forced page_htr targets; prompt/vision/padding and the always-active shared expert "
            f"are excluded; {source_note}.",
            ha="center", fontsize=9.5, color="#52657A",
        )
        figure.tight_layout(rect=(0, 0.06, 1, 0.93))
        figure.savefig(output_path, dpi=240, bbox_inches="tight", facecolor=figure.get_facecolor())
        display(figure)
        plt.close(figure)

    def _bootstrap_mean_support_ci(layer_bucket, seed):
        page_tokens = np.asarray(layer_bucket["page_token_counts"], dtype=np.float64)
        page_support = np.asarray(layer_bucket["page_support_sums"], dtype=np.float64)
        if page_tokens.size != page_support.size or page_tokens.size == 0:
            raise RuntimeError("Invalid per-page support data for bootstrap CI.")
        point = float(page_support.sum() / page_tokens.sum())
        if page_tokens.size == 1:
            return point, point, point
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, page_tokens.size, size=(int(ROUTING_ANALYSIS_BOOTSTRAP_REPLICATES), page_tokens.size))
        means = page_support[indices].sum(axis=1) / page_tokens[indices].sum(axis=1)
        low, high = np.quantile(means, [0.025, 0.975]).tolist()
        return point, float(low), float(high)

    def _plot_routing_main(scope, layer_data, report, output_path):
        # Experts are layer-local modules. Curves are therefore ranked by activation
        # within each layer, rather than pretending that E1 at two layers is one expert.
        layers = [int(layer) for layer in ROUTING_ANALYSIS_LAYER_INDICES]
        mean_support, ci_low, ci_high = [], [], []
        activation = []
        for layer in layers:
            point, low, high = _bootstrap_mean_support_ci(
                layer_data[layer], seed=int(ROUTING_ANALYSIS_SEED) + 1009 * int(layer)
            )
            reported = float(report[str(layer)]["mean_active_routed_experts"])
            if not math.isclose(point, reported, rel_tol=0.0, abs_tol=1e-8):
                raise RuntimeError(f"Bootstrap/report E[K] mismatch at layer={layer}: {point} vs {reported}")
            mean_support.append(point); ci_low.append(low); ci_high.append(high)
            activation.append([
                float(report[str(layer)]["expert_activation_percent"][f"E{expert}"])
                for expert in range(1, ROUTING_ANALYSIS_ROUTED_EXPERTS + 1)
            ])
        mean_support = np.asarray(mean_support, dtype=float)
        ci_low = np.asarray(ci_low, dtype=float); ci_high = np.asarray(ci_high, dtype=float)
        activation = np.asarray(activation, dtype=float)
        ranked_activation = np.sort(activation, axis=1)[:, ::-1]
        balanced_reference = 100.0 * mean_support / float(ROUTING_ANALYSIS_ROUTED_EXPERTS)
        if not np.allclose(activation.sum(axis=1), 100.0 * mean_support, atol=1e-6):
            raise RuntimeError("Figure invariant failed: sum expert activation must equal 100 * E[K].")

        # Keep renderer/font substitution messages out of exported notebook figures.
        for _logger_name in ("matplotlib", "matplotlib.mathtext", "matplotlib.font_manager"):
            logging.getLogger(_logger_name).setLevel(logging.WARNING)
        with plt.rc_context({
            "font.family": "DejaVu Sans", "font.size": 10.0,
            "mathtext.fontset": "dejavusans",
            "axes.labelcolor": "#243746", "xtick.color": "#425466",
            "ytick.color": "#425466", "axes.titleweight": "semibold",
        }):
            figure, (ax_support, ax_util) = plt.subplots(
                1, 2, figsize=(10.4, 3.95), gridspec_kw={"width_ratios": [0.95, 1.18]}
            )
            figure.patch.set_facecolor("white")
            x = np.arange(len(layers), dtype=float)

            ax_support.fill_between(x, ci_low, ci_high, color="#74A9CF", alpha=0.64, linewidth=0, label="95% page-bootstrap CI")
            ax_support.plot(x, mean_support, color="#0B4F71", linewidth=2.15, marker="o", markersize=5.8, zorder=3)
            ax_support.set_title("(a) Adaptive support size", loc="left", pad=8)
            ax_support.set_xlabel("MoE layer")
            ax_support.set_ylabel(r"Mean active routed experts $\mathbb{E}[K_\ell]$")
            ax_support.set_xticks(x, [str(layer) for layer in layers])
            y_low = max(0.95, float(np.floor((ci_low.min() - 0.12) * 5.0) / 5.0))
            y_high = min(float(ROUTING_ANALYSIS_ROUTED_EXPERTS), float(np.ceil((ci_high.max() + 0.12) * 5.0) / 5.0))
            ax_support.set_ylim(y_low, y_high)
            ax_support.grid(axis="y", color="#DCE5EB", linewidth=0.65)
            ax_support.set_axisbelow(True)
            ax_support.legend(loc="lower left", frameon=False, fontsize=8.4, handlelength=1.4)

            rank_colors = ["#083B66", "#2C75A7", "#68A9CF", "#A9D2E5"]
            rank_markers = ["o", "s", "^", "D"]
            rank_styles = ["-", "--", ":", "-."]
            rank_labels = ["1st most active", "2nd most active", "3rd most active", "Least active"]
            ax_util.plot(x, balanced_reference, color="#5D6670", linestyle=(0, (4, 3)), linewidth=1.5, label=r"Equal utilization: $\mathbb{E}[K_\ell]/4$", zorder=1)
            for rank in range(ROUTING_ANALYSIS_ROUTED_EXPERTS):
                ax_util.plot(x, ranked_activation[:, rank], color=rank_colors[rank], linestyle=rank_styles[rank], linewidth=2.0, marker=rank_markers[rank], markersize=5.0, label=rank_labels[rank], zorder=2)
            ax_util.set_title("(b) Ranked routed-expert activation", loc="left", pad=23)
            ax_util.text(0.0, 1.035, "Experts ranked by activation frequency within each layer", transform=ax_util.transAxes, fontsize=8.3, color="#5A6D7E", va="bottom")
            ax_util.set_xlabel("MoE layer")
            ax_util.set_ylabel(r"Activation probability $P(p_{\ell,e}>0)$ (%)")
            ax_util.set_xticks(x, [str(layer) for layer in layers])
            ax_util.set_ylim(0.0, 100.0)
            ax_util.set_yticks([0, 25, 50, 75, 100])
            ax_util.grid(axis="y", color="#DCE5EB", linewidth=0.65)
            ax_util.set_axisbelow(True)
            ax_util.legend(loc="lower center", bbox_to_anchor=(0.5, -0.49), ncol=2, frameon=False, fontsize=7.9, columnspacing=1.0, handlelength=1.7)

            for axis in (ax_support, ax_util):
                axis.spines["top"].set_visible(False); axis.spines["right"].set_visible(False)
                axis.spines["left"].set_color("#AEBBC5"); axis.spines["bottom"].set_color("#AEBBC5")
                axis.spines["left"].set_linewidth(0.65); axis.spines["bottom"].set_linewidth(0.65)
                axis.tick_params(width=0.65, length=3.0)
            figure.subplots_adjust(left=0.09, right=0.985, top=0.88, bottom=0.32, wspace=0.34)
            figure.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
            figure.savefig(Path(output_path).with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
            display(figure)
            plt.close(figure)
        return {
            "mean_active_routed_experts": {str(layer): float(value) for layer, value in zip(layers, mean_support)},
            "mean_support_ci95_page_bootstrap": {str(layer): [float(low), float(high)] for layer, low, high in zip(layers, ci_low, ci_high)},
            "equal_utilization_reference_percent": {str(layer): float(value) for layer, value in zip(layers, balanced_reference)},
            "ranked_activation_percent": {str(layer): [float(value) for value in ranked_activation[idx]] for idx, layer in enumerate(layers)},
        }

    routing_figure_paths = {}
    routing_figure_statistics = {}
    for _source, _report in sorted(routing_report_by_source.items()):
        _figure_path = routing_output_dir / f"routing_main_{_safe_figure_name(_source)}.png"
        routing_figure_statistics[_source] = _plot_routing_main(
            scope=_source, layer_data=routing_accumulator.data[_source], report=_report, output_path=_figure_path
        )
        routing_figure_paths[_source] = str(_figure_path)

    routing_all_figure_path = routing_output_dir / "routing_main_all_sources_pooled.png"
    routing_pooled_data = routing_accumulator.pooled()
    routing_figure_statistics["All_sources_pooled"] = _plot_routing_main(
        scope="All test sources", layer_data=routing_pooled_data, report=routing_report_pooled, output_path=routing_all_figure_path
    )
    routing_figure_paths["All_sources_pooled"] = str(routing_all_figure_path)

    routing_caption_path = routing_output_dir / "routing_main_figure_caption.txt"
    routing_caption_path.write_text(
        "Sparsegen routing on a deterministic stratified held-out test sample (up to 50 pages per source; the exact source-level selection is in routing_analysis_manifest.json). "
        "(a) Mean active routed experts E[K] at every attention_mask=1 position; ribbons are 95% percentile bootstrap intervals obtained by resampling pages. "
        "(b) Routed-expert activation probabilities, ranked independently within each layer because experts are layer-local modules. "
        "The dashed reference is E[K]/4, the activation probability under equal utilization of four routed experts. "
        "Vision, prompt/context, and teacher-forced transcription positions are included; padding and the always-active shared expert are excluded.\n",
        encoding="utf-8",
    )

    routing_report_path = routing_output_dir / "routing_support_and_utilization.json"
    routing_analysis_manifest_path = routing_output_dir / "routing_analysis_manifest.json"
    routing_report_path.write_text(
        json.dumps(
            {
                "definition": {
                    "support_size": "K = number of routed E1..E4 with native Sparsegen probability p>0 at one attention_mask=1 position.",
                    "expert_utilization": "A_(layer, expert) = P(p_(layer, expert)>0) over all attention_mask=1 positions.",
                    "routing_mass": "M_(layer, expert) = E[p_(layer, expert)] over all attention_mask=1 positions; stored for a collapse diagnostic but not plotted in the two-panel main figure.",
                    "included_positions": ["vision", "prompt/context", "teacher-forced transcription"],
                    "excluded_positions": ["padding"],
                    "excluded_experts": ["always-active shared expert"],
                    "aggregation": "all-source figure pools all attended positions from the deterministic stratified test sample, not an unweighted macro-average over sources",
                },
                "by_source": routing_report_by_source,
                "all_sources_pooled": routing_report_pooled,
            },
            ensure_ascii=False, indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    routing_analysis_manifest_path.write_text(
        json.dumps(
            {
                "checkpoint": str(routing_checkpoint),
                "checkpoint_moe_signature": routing_manifest.get("moe_signature"),
                "test_jsonl": str(routing_test_path),
                "seed": int(ROUTING_ANALYSIS_SEED),
                "pages_requested_per_source": int(ROUTING_ANALYSIS_PAGES_PER_SOURCE),
                "sample_by_source": routing_sample_manifest,
                "layer_indices": list(ROUTING_ANALYSIS_LAYER_INDICES),
                "routed_experts": int(ROUTING_ANALYSIS_ROUTED_EXPERTS),
                "figures": routing_figure_paths,
                "main_figure_statistics": routing_figure_statistics,
                "csv": str(routing_csv_path),
                "main_figure_caption": str(routing_caption_path),
                "statistics_json": str(routing_report_path),
            },
            ensure_ascii=False, indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print("Routing figures:", json.dumps(routing_figure_paths, ensure_ascii=False, indent=2))
    print("Routing statistics:", routing_report_path)
    print("Routing CSV:", routing_csv_path)
    print("Main-figure caption:", routing_caption_path)
    print("Routing sample/checkpoint manifest:", routing_analysis_manifest_path)
    # End of test routing probe.

    return export_runtime_locals(state, locals())
