"""analysis.position_diagnostics phase for the ExpertHTR research workflow.

This module preserves the audited runtime computation. It is invoked by the
explicit CLI workflow and exports its declared runtime objects into a
:class:`~experthtr.runtime.context.RuntimeState`.
"""

from __future__ import annotations

from experthtr.runtime.context import RuntimeState, export_runtime_locals
from experthtr.evaluation.checkpoint_contract import validate_atomic_marker_contract


def run_position_diagnostics(state: RuntimeState) -> RuntimeState:
    """Execute this phase after its prerequisites have populated ``state``."""

    globals().update(state.values)
    # =============================================================================
    # Validation routing diagnostic: all-attended main figure + position partitions
    # =============================================================================
    # Read-only checkpoint probe. It does not train, generate, alter labels, or
    # alter the dataset. The publication-style figure uses all attention_mask=1
    # positions; the Vision/Prompt/HTR partitions remain in the CSV/JSON only.
    #
    # Position definitions (causal LM alignment):
    #   all_attended: every non-padding position in attention_mask (the current
    #                 load-balance objective's mask).
    #   vision:       image-pad positions only (not vision boundary text tokens).
    #   htr_target:   hidden-state position t whose next token labels[t+1] is a
    #                 supervised page_htr target token.
    #   prompt_other_context: the exact residual of all_attended after vision and
    #                 htr_target; it includes prompt/chat boundary positions and
    #                 any attended terminal position with no supervised next token.
    #
    # The groups form an exact partition of all_attended. Reported routed experts
    # are E1..E4 only; the always-active shared expert is excluded from K and A.

    POSITION_DIAG_CHECKPOINT_DIR = os.getenv("HWVLM_POSITION_DIAG_CHECKPOINT", "").strip()  # REQUIRED: absolute folder of the MoE checkpoint to analyse.
    POSITION_DIAG_OUTPUT_DIR = os.getenv("HWVLM_POSITION_DIAG_OUTPUT", "").strip()      # Empty -> writable /kaggle/working (or cwd) directory.
    POSITION_DIAG_LAYER_INDICES = (1, 5, 9, 13, 17, 21)  # Must match checkpoint exactly.
    POSITION_DIAG_ROUTED_EXPERTS = 4
    POSITION_DIAG_MAX_VALID_PAGES_PER_SOURCE = 0  # 0 = all eligible validation pages.
    POSITION_DIAG_SEED = 20260901
    POSITION_DIAG_BOOTSTRAP_REPLICATES = 2000  # page-resampled 95% CI for E[K].

    import csv
    import logging
    import re
    from contextlib import nullcontext
    from collections import defaultdict
    from IPython.display import display
    import matplotlib.pyplot as plt

    if not str(POSITION_DIAG_CHECKPOINT_DIR).strip():
        raise ValueError(
            "Set POSITION_DIAG_CHECKPOINT_DIR to the exact MoE checkpoint folder "
            "(CHECKPOINT_COMPLETE, checkpoint_manifest.json, full_model_state.pt)."
        )
    if int(POSITION_DIAG_MAX_VALID_PAGES_PER_SOURCE) < 0:
        raise ValueError("POSITION_DIAG_MAX_VALID_PAGES_PER_SOURCE must be >= 0.")
    if int(POSITION_DIAG_ROUTED_EXPERTS) < 1:
        raise ValueError("POSITION_DIAG_ROUTED_EXPERTS must be positive.")
    if int(POSITION_DIAG_BOOTSTRAP_REPLICATES) < 100:
        raise ValueError("POSITION_DIAG_BOOTSTRAP_REPLICATES must be >=100.")
    if isinstance(model, nn.DataParallel):
        raise RuntimeError("Run this hook-based diagnostic on one GPU, not DataParallel.")

    posdiag_checkpoint = Path(POSITION_DIAG_CHECKPOINT_DIR).expanduser()
    for _required in (CHECKPOINT_COMPLETE_FILE, CHECKPOINT_MANIFEST_FILE, FULL_MODEL_STATE_FILE):
        if not (posdiag_checkpoint / _required).is_file():
            raise FileNotFoundError(posdiag_checkpoint / _required)
    posdiag_manifest = json.loads(
        (posdiag_checkpoint / CHECKPOINT_MANIFEST_FILE).read_text(encoding="utf-8")
    )
    validate_atomic_marker_contract(posdiag_manifest, CFG)

    if str(POSITION_DIAG_OUTPUT_DIR).strip():
        posdiag_output_dir = Path(POSITION_DIAG_OUTPUT_DIR).expanduser()
    else:
        posdiag_writable_root = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
        posdiag_output_dir = (
            posdiag_writable_root / "routing_position_diagnostics_valid" / posdiag_checkpoint.parent.parent.name
        )
    posdiag_output_dir.mkdir(parents=True, exist_ok=True)

    # The model must have been constructed with the same MoE topology before this
    # strict restore. The workflow build path does that; strict loading prevents
    # mislabelled plots.
    posdiag_model = unwrap_model(model)
    try:
        posdiag_payload = torch.load(
            posdiag_checkpoint / FULL_MODEL_STATE_FILE, map_location="cpu", weights_only=True
        )
    except TypeError:
        posdiag_payload = torch.load(posdiag_checkpoint / FULL_MODEL_STATE_FILE, map_location="cpu")
    posdiag_model.load_state_dict(posdiag_payload.get("model_state_dict", posdiag_payload), strict=True)
    del posdiag_payload
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Discover actual MoE layers from the restored model.
    posdiag_layer_pattern = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp$")
    posdiag_modules = {}
    for _module_name, _module in posdiag_model.named_modules():
        if not isinstance(_module, LanguageGroupedMoEMLP):
            continue
        _match = posdiag_layer_pattern.search(_module_name)
        if _match is None:
            raise RuntimeError(f"Cannot recover decoder layer index from MoE name: {_module_name!r}")
        _layer = int(_match.group(1))
        if _layer in posdiag_modules:
            raise RuntimeError(f"Duplicate routed MLP at decoder layer {_layer}.")
        posdiag_modules[_layer] = _module

    posdiag_actual_layers = tuple(sorted(posdiag_modules))
    if posdiag_actual_layers != tuple(POSITION_DIAG_LAYER_INDICES):
        raise RuntimeError(
            "Requested POSITION_DIAG_LAYER_INDICES differs from checkpoint model: "
            f"requested={tuple(POSITION_DIAG_LAYER_INDICES)} actual={posdiag_actual_layers}."
        )
    for _layer, _module in posdiag_modules.items():
        if len(_module.language_experts) != int(POSITION_DIAG_ROUTED_EXPERTS):
            raise RuntimeError(
                f"Layer {_layer} has {len(_module.language_experts)} routed experts, "
                f"not {POSITION_DIAG_ROUTED_EXPERTS}."
            )
        if str(_module.routing_fn).lower() != "sparsegen" or bool(_module.enforce_top_k):
            raise RuntimeError(
                f"Layer {_layer} is routing={_module.routing_fn!r}, hard_top_k={_module.enforce_top_k}; "
                "this diagnostic is for unconstrained Sparsegen only."
            )

    posdiag_checkpoint_signature = dict(posdiag_manifest.get("moe_signature") or {})
    if not posdiag_checkpoint_signature:
        raise RuntimeError("Checkpoint manifest has no moe_signature.")
    posdiag_first_module = posdiag_modules[posdiag_actual_layers[0]]
    posdiag_signature_checks = {
        "enabled": True,
        "language_experts": len(posdiag_first_module.language_experts),
        "shared_experts": len(posdiag_first_module.shared_experts),
        "layer_indices": list(posdiag_actual_layers),
        "routing_fn": str(posdiag_first_module.routing_fn),
    }
    posdiag_signature_mismatches = {
        key: {"checkpoint": posdiag_checkpoint_signature.get(key), "runtime": value}
        for key, value in posdiag_signature_checks.items()
        if posdiag_checkpoint_signature.get(key) != value
    }
    if posdiag_signature_mismatches:
        raise RuntimeError(
            "Checkpoint MoE signature and restored model disagree: "
            + json.dumps(posdiag_signature_mismatches, ensure_ascii=False)
        )

    # Use valid_dataset created by the main loader component: same prompt/image
    # processor/exclusions as runtime. Validation must be page_htr-only.
    posdiag_valid_samples = list(getattr(valid_dataset, "samples", []))
    if not posdiag_valid_samples:
        raise RuntimeError(
            "valid_dataset.samples is unavailable or empty; run the workflow through "
            "the data-loader component first."
        )
    posdiag_indices_by_source = defaultdict(list)
    for _index, _sample in enumerate(posdiag_valid_samples):
        _task = str((_sample.get("metadata") or {}).get("task", ""))
        if _task != "page_htr":
            raise RuntimeError(
                "This diagnostic requires page_htr-only validation; "
                f"found task={_task!r} at index={_index}."
            )
        _source = str((_sample.get("metadata") or {}).get("source_dataset", "unknown"))
        posdiag_indices_by_source[_source].append(_index)

    posdiag_expected_sources = set(CFG.get("expected_sources") or [])
    if posdiag_expected_sources and set(posdiag_indices_by_source) != posdiag_expected_sources:
        raise RuntimeError(
            "Validation source scope differs from CFG: "
            f"validation={sorted(posdiag_indices_by_source)} expected={sorted(posdiag_expected_sources)}"
        )

    posdiag_selected_indices = []
    posdiag_selection = {}
    for _source in sorted(posdiag_indices_by_source):
        _available = sorted(posdiag_indices_by_source[_source])
        _cap = int(POSITION_DIAG_MAX_VALID_PAGES_PER_SOURCE)
        if _cap == 0 or _cap >= len(_available):
            _chosen, _selection = _available, "all_validation_pages"
        else:
            _rng = random.Random(f"{POSITION_DIAG_SEED}::{_source}")
            _chosen, _selection = sorted(_rng.sample(_available, _cap)), "uniform_without_replacement"
        posdiag_selected_indices.extend(_chosen)
        posdiag_selection[_source] = {
            "available_valid_pages": len(_available),
            "selected_pages": len(_chosen),
            "sampling": _selection,
            "record_ids": [str(posdiag_valid_samples[i].get("id", "")) for i in _chosen],
        }
    posdiag_selected_indices = sorted(posdiag_selected_indices)
    if not posdiag_selected_indices:
        raise RuntimeError("No validation page_htr records were selected.")

    posdiag_dataset = torch.utils.data.Subset(valid_dataset, posdiag_selected_indices)
    posdiag_loader = DataLoader(
        posdiag_dataset, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=torch.cuda.is_available(), collate_fn=collator,
    )

    # Vision = Qwen image-pad feature tokens, not visual delimiter/chat tokens.
    posdiag_tokenizer = getattr(processor, "tokenizer", None)
    posdiag_vision_ids = set()
    for _holder in (
        getattr(processor, "image_processor", None),
        processor,
        getattr(posdiag_model, "config", None),
        getattr(getattr(posdiag_model, "base_model", None), "config", None),
    ):
        for _attribute in ("image_token_id", "image_token_index"):
            _value = getattr(_holder, _attribute, None)
            if isinstance(_value, (int, np.integer)) and int(_value) >= 0:
                posdiag_vision_ids.add(int(_value))
    if posdiag_tokenizer is not None:
        _unknown_id = getattr(posdiag_tokenizer, "unk_token_id", None)
        for _token in ("<|image_pad|>", "<|image_token|>", "<image>"):
            _token_id = posdiag_tokenizer.convert_tokens_to_ids(_token)
            if isinstance(_token_id, (int, np.integer)) and _token_id >= 0 and _token_id != _unknown_id:
                posdiag_vision_ids.add(int(_token_id))
    if not posdiag_vision_ids:
        raise RuntimeError(
            "Could not resolve an image-pad token ID. Refusing to produce a fake 'Vision' panel."
        )

    class _PositionRoutingAccumulator:
        def __init__(self, groups, layers, experts):
            self.groups = tuple(groups)
            self.layers = tuple(int(layer) for layer in layers)
            self.experts = int(experts)
            self.data = {
                group: {
                    layer: {
                        "positions": 0,
                        "support_counts": [0] * (self.experts + 1),
                        "activation_counts": [0] * self.experts,
                        "expert_mass_sums": [0.0] * self.experts,
                        "page_token_counts": [],
                        "page_support_sums": [],
                    }
                    for layer in self.layers
                }
                for group in self.groups
            }

        def add(self, group, layer, weights):
            if weights.ndim != 2 or int(weights.shape[1]) != self.experts:
                raise RuntimeError(
                    f"Unexpected routing tensor for {group}/layer={layer}: {tuple(weights.shape)}."
                )
            weights = weights.detach().float()
            if not torch.isfinite(weights).all():
                raise RuntimeError(f"Non-finite routing weights at {group}/layer={layer}.")
            active = weights > 0
            support = active.sum(dim=-1)
            if support.numel() == 0:
                return
            if int(support.min()) < 1 or int(support.max()) > self.experts:
                raise RuntimeError(
                    f"Invalid Sparsegen support at {group}/layer={layer}: "
                    f"min={int(support.min())} max={int(support.max())}."
                )
            bucket = self.data[group][layer]
            bucket["positions"] += int(support.numel())
            counts = torch.bincount(support.to(torch.long).cpu(), minlength=self.experts + 1).tolist()
            bucket["support_counts"] = [int(old) + int(new) for old, new in zip(bucket["support_counts"], counts)]
            active_counts = active.sum(dim=0).to(torch.long).cpu().tolist()
            bucket["activation_counts"] = [
                int(old) + int(new) for old, new in zip(bucket["activation_counts"], active_counts)
            ]
            mass_sums = weights.sum(dim=0).cpu().tolist()
            bucket["expert_mass_sums"] = [
                float(old) + float(new) for old, new in zip(bucket["expert_mass_sums"], mass_sums)
            ]
            bucket["page_token_counts"].append(int(support.numel()))
            bucket["page_support_sums"].append(float(support.sum().item()))

        def report(self):
            result = {}
            for group, by_layer in self.data.items():
                result[group] = {}
                for layer, bucket in sorted(by_layer.items()):
                    positions = int(bucket["positions"])
                    if positions <= 0:
                        raise RuntimeError(f"No captured positions for group={group!r}, layer={layer}.")
                    support_counts = [int(value) for value in bucket["support_counts"]]
                    activation_counts = [int(value) for value in bucket["activation_counts"]]
                    expert_mass_sums = [float(value) for value in bucket["expert_mass_sums"]]
                    if sum(support_counts[1:]) != positions:
                        raise RuntimeError(
                            f"Support invariant failed for {group}/layer={layer}: "
                            f"{sum(support_counts[1:])} != {positions}."
                        )
                    result[group][str(layer)] = {
                        "positions": positions,
                        "mean_active_routed_experts": sum(k * support_counts[k] for k in range(1, self.experts + 1)) / positions,
                        "support_counts": {f"K={k}": support_counts[k] for k in range(1, self.experts + 1)},
                        "support_percent": {
                            f"K={k}": 100.0 * support_counts[k] / positions
                            for k in range(1, self.experts + 1)
                        },
                        "expert_activation_counts": {
                            f"E{expert}": activation_counts[expert - 1]
                            for expert in range(1, self.experts + 1)
                        },
                        "expert_activation_percent": {
                            f"E{expert}": 100.0 * activation_counts[expert - 1] / positions
                            for expert in range(1, self.experts + 1)
                        },
                        "expert_routing_mass_percent": {
                            f"E{expert}": 100.0 * expert_mass_sums[expert - 1] / positions
                            for expert in range(1, self.experts + 1)
                        },
                    }
            return result

    posdiag_groups = ("all_attended", "vision", "prompt_other_context", "htr_target")
    posdiag_accumulator = _PositionRoutingAccumulator(
        groups=posdiag_groups, layers=posdiag_actual_layers, experts=POSITION_DIAG_ROUTED_EXPERTS,
    )
    # Source-wise accumulators reuse the same forward pass; they do not add inference work.
    posdiag_accumulators_by_source = {
        source: _PositionRoutingAccumulator(
            groups=posdiag_groups, layers=posdiag_actual_layers, experts=POSITION_DIAG_ROUTED_EXPERTS,
        )
        for source in sorted(posdiag_selection)
    }
    posdiag_runtime = {"masks": None, "source": None}
    posdiag_original_routing_methods = {}

    # Capture native Sparsegen p before dispatch: p>0 is the actual dynamic
    # routed-compute support, not a post-hoc proxy.
    def _make_posdiag_capture(original_method, layer):
        def _capture(logits, x):
            weights, lam = original_method(logits, x)
            masks = posdiag_runtime["masks"]
            source = posdiag_runtime["source"]
            if masks is None or source is None:
                raise RuntimeError("Routing capture ran outside a prepared validation batch.")
            if source not in posdiag_accumulators_by_source:
                raise RuntimeError(f"Unexpected validation source in routing capture: {source!r}.")
            if weights.ndim != 3:
                raise RuntimeError(f"Expected rank-3 routing weights, got {tuple(weights.shape)}.")
            for group, mask in masks.items():
                if tuple(mask.shape) != tuple(weights.shape[:2]):
                    raise RuntimeError(
                        f"Cannot align {group} mask {tuple(mask.shape)} with routing weights "
                        f"{tuple(weights.shape)} at layer {layer}."
                    )
                selected_weights = weights[mask.to(device=weights.device)]
                posdiag_accumulator.add(group, layer, selected_weights)
                posdiag_accumulators_by_source[source].add(group, layer, selected_weights)
            return weights, lam
        return _capture

    for _layer, _module in posdiag_modules.items():
        posdiag_original_routing_methods[_layer] = _module._routing_weight
        _module._routing_weight = _make_posdiag_capture(posdiag_original_routing_methods[_layer], _layer)

    posdiag_skip_keys = {
        "texts", "prompts", "ids", "metadata", "source_datasets", "tasks", "languages", "page_keys",
        "language_groups", "resize_scales", "original_sizes", "resized_sizes",
    }
    posdiag_was_training = bool(posdiag_model.training)
    posdiag_model.eval()
    posdiag_partition_totals = {group: 0 for group in posdiag_groups}
    try:
        with torch.no_grad():
            for _batch in tqdm(posdiag_loader, desc="validation routing-position probe", leave=False):
                _source_values = list(_batch.get("source_datasets", []))
                if len(_source_values) != 1:
                    raise RuntimeError("This diagnostic requires exactly one source/page per batch.")
                posdiag_runtime["source"] = str(_source_values[0])
                _batch_device = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in _batch.items()
                    if not key.startswith("prompt_") and key not in posdiag_skip_keys
                }
                _input_ids = _batch_device.get("input_ids")
                _attention_mask = _batch_device.get("attention_mask")
                _labels = _batch_device.get("labels")
                if not all(
                    isinstance(value, torch.Tensor) and value.ndim == 2
                    for value in (_input_ids, _attention_mask, _labels)
                ):
                    raise RuntimeError("Probe requires rank-2 input_ids, attention_mask, and labels.")
                _all_attended = _attention_mask.bool()
                _vision = torch.zeros_like(_all_attended)
                for _vision_id in posdiag_vision_ids:
                    _vision |= _input_ids.eq(int(_vision_id))
                _vision &= _all_attended
                _htr_target = torch.zeros_like(_all_attended)
                _htr_target[:, :-1] = _labels[:, 1:].ne(IGNORE_INDEX)
                _htr_target &= _all_attended
                if bool((_vision & _htr_target).any().item()):
                    raise RuntimeError("Vision positions overlap HTR targets; prompt/label contract is inconsistent.")
                _prompt_other = _all_attended & ~(_vision | _htr_target)
                _partition = _vision.to(torch.int16) + _htr_target.to(torch.int16) + _prompt_other.to(torch.int16)
                if not bool(torch.equal(_partition, _all_attended.to(torch.int16))):
                    raise RuntimeError("All/Vision/Prompt-other/HTR-target masks do not partition attended positions.")
                posdiag_runtime["masks"] = {
                    "all_attended": _all_attended,
                    "vision": _vision,
                    "prompt_other_context": _prompt_other,
                    "htr_target": _htr_target,
                }
                for _group, _mask in posdiag_runtime["masks"].items():
                    posdiag_partition_totals[_group] += int(_mask.sum().item())
                _autocast = (
                    torch.amp.autocast("cuda", dtype=get_model_dtype(device))
                    if device.type == "cuda" and get_model_dtype(device) != torch.float32
                    else nullcontext()
                )
                with _autocast:
                    _ = posdiag_model(**_batch_device)
    finally:
        for _layer, _module in posdiag_modules.items():
            _module._routing_weight = posdiag_original_routing_methods[_layer]
        posdiag_runtime["masks"] = None
        posdiag_runtime["source"] = None
        posdiag_model.train(posdiag_was_training)

    if posdiag_partition_totals["vision"] <= 0:
        raise RuntimeError(
            "No image-pad positions were found. Resolved image token IDs="
            f"{sorted(posdiag_vision_ids)}; refuse to label a panel 'Vision'."
        )
    if posdiag_partition_totals["htr_target"] <= 0:
        raise RuntimeError("No supervised HTR-target positions were found.")

    posdiag_report = posdiag_accumulator.report()
    posdiag_report_by_source = {
        source: accumulator.report()
        for source, accumulator in sorted(posdiag_accumulators_by_source.items())
    }
    posdiag_csv_rows = []
    for _scope, _scope_report in [
        ("All_sources_pooled", posdiag_report),
        *sorted(posdiag_report_by_source.items()),
    ]:
        for _group in posdiag_groups:
            for _layer_text, _row in sorted(_scope_report[_group].items(), key=lambda item: int(item[0])):
                _csv_row = {
                    "scope": _scope,
                    "position_group": _group,
                    "layer": int(_layer_text),
                    "positions": int(_row["positions"]),
                    "mean_active_routed_experts": float(_row["mean_active_routed_experts"]),
                }
                for _key, _value in _row["support_counts"].items():
                    _csv_row[f"support_count_{_key.replace('=', '')}"] = int(_value)
                for _key, _value in _row["support_percent"].items():
                    _csv_row[f"support_pct_{_key.replace('=', '')}"] = float(_value)
                for _key, _value in _row["expert_activation_counts"].items():
                    _csv_row[f"activation_count_{_key}"] = int(_value)
                for _key, _value in _row["expert_activation_percent"].items():
                    _csv_row[f"activation_pct_{_key}"] = float(_value)
                for _key, _value in _row["expert_routing_mass_percent"].items():
                    _csv_row[f"routing_mass_pct_{_key}"] = float(_value)
                posdiag_csv_rows.append(_csv_row)

    posdiag_csv_path = posdiag_output_dir / "validation_routing_position_statistics.csv"
    with posdiag_csv_path.open("w", newline="", encoding="utf-8") as _handle:
        _writer = csv.DictWriter(_handle, fieldnames=list(posdiag_csv_rows[0]))
        _writer.writeheader()
        _writer.writerows(posdiag_csv_rows)

    def _legacy_position_dashboard(report, output_path, scope_label, page_count):
        """Render a faceted manuscript figure and return its external caption."""
        layers = list(posdiag_actual_layers)
        support_colors = ["#E1EFF8", "#AACEE7", "#5F9DC8", "#174E7B"]  # ordinal K=1 -> K=4
        row_labels = {
            "all_attended": "All attended",
            "vision": "Vision",
            "prompt_other_context": "Prompt / other",
            "htr_target": "HTR target",
        }
        denominator_text = {
            row_labels[group]: int(report[group][str(layers[0])]["positions"])
            for group in posdiag_groups
        }

        plt.rcParams.update({
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.labelcolor": "#23384A",
            "xtick.color": "#445C70",
            "ytick.color": "#445C70",
        })
        # Header row + four token-position facets. Column headers and legend occur
        # once; all other panels share the same visual grammar.
        figure = plt.figure(figsize=(10.8, 11.5), facecolor="white", constrained_layout=True)
        grid = figure.add_gridspec(
            nrows=len(posdiag_groups) + 1,
            ncols=2,
            height_ratios=[0.20] + [1.0] * len(posdiag_groups),
            width_ratios=[1.32, 1.0],
            hspace=0.04,
            wspace=0.12,
        )
        header_support = figure.add_subplot(grid[0, 0])
        header_heatmap = figure.add_subplot(grid[0, 1])
        for header_axis in (header_support, header_heatmap):
            header_axis.set_axis_off()
        header_support.text(
            0.0, 0.92, "(a) Adaptive support size",
            fontsize=12.5, fontweight="bold", color="#152C43", va="top",
        )
        header_support.text(
            0.0, 0.36,
            r"$K$ = number of routed experts with $p_{\ell,e}>0$",
            fontsize=9.2, color="#536B7D", va="top",
        )
        header_heatmap.text(
            0.0, 0.92, "(b) Expert utilization",
            fontsize=12.5, fontweight="bold", color="#152C43", va="top",
        )
        header_heatmap.text(
            0.0, 0.36,
            r"Activation probability $P(p_{\ell,e}>0)$",
            fontsize=9.2, color="#536B7D", va="top",
        )

        support_axes, heatmap_axes = [], []
        legend_handles = None
        last_image = None
        for row_index, group in enumerate(posdiag_groups):
            ax_support = figure.add_subplot(grid[row_index + 1, 0])
            ax_heatmap = figure.add_subplot(grid[row_index + 1, 1])
            support_axes.append(ax_support)
            heatmap_axes.append(ax_heatmap)

            support = np.array([
                [report[group][str(layer)]["support_percent"][f"K={k}"]
                 for k in range(1, POSITION_DIAG_ROUTED_EXPERTS + 1)]
                for layer in layers
            ], dtype=float)
            activation = np.array([
                [report[group][str(layer)]["expert_activation_percent"][f"E{expert}"]
                 for expert in range(1, POSITION_DIAG_ROUTED_EXPERTS + 1)]
                for layer in layers
            ], dtype=float)

            x = np.arange(len(layers))
            bottom = np.zeros(len(layers), dtype=float)
            current_handles = []
            for column, k in enumerate(range(1, POSITION_DIAG_ROUTED_EXPERTS + 1)):
                bars = ax_support.bar(
                    x, support[:, column], bottom=bottom, width=0.72,
                    color=support_colors[column], edgecolor="white", linewidth=0.55,
                )
                current_handles.append(bars[0])
                for bar, value, base in zip(bars, support[:, column], bottom):
                    # Exact values live in CSV; label only segments with enough area.
                    if value > 10.0:
                        ax_support.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            base + value / 2.0,
                            f"{value:.0f}%",
                            ha="center", va="center", fontsize=8.0, fontweight="semibold",
                            color="white" if column >= 2 else "#17354C",
                        )
                bottom += support[:, column]
            if legend_handles is None:
                legend_handles = current_handles

            ax_support.set_ylim(0, 100)
            ax_support.set_yticks([0, 25, 50, 75, 100])
            ax_support.set_ylabel(f"{row_labels[group]}\nshare (%)", labelpad=10, fontweight="semibold")
            ax_support.grid(axis="y", color="#E6ECF2", linewidth=0.55)
            ax_support.set_axisbelow(True)
            ax_support.set_xticks(x)
            if row_index == len(posdiag_groups) - 1:
                ax_support.set_xticklabels([str(layer) for layer in layers])
                ax_support.set_xlabel("MoE decoder layer", labelpad=5)
            else:
                ax_support.set_xticklabels([])
                ax_support.tick_params(axis="x", length=0)
            for spine_name in ("top", "right"):
                ax_support.spines[spine_name].set_visible(False)
            for spine_name in ("bottom", "left"):
                ax_support.spines[spine_name].set_color("#B7C4D0")
                ax_support.spines[spine_name].set_linewidth(0.55)
            ax_support.tick_params(width=0.55, length=2.5)

            last_image = ax_heatmap.imshow(
                activation, cmap=plt.get_cmap("Blues"),
                vmin=0.0, vmax=100.0, aspect="auto", interpolation="nearest",
            )
            ax_heatmap.set_xticks(range(POSITION_DIAG_ROUTED_EXPERTS))
            if row_index == len(posdiag_groups) - 1:
                ax_heatmap.set_xticklabels(
                    [f"E{expert}" for expert in range(1, POSITION_DIAG_ROUTED_EXPERTS + 1)]
                )
                ax_heatmap.set_xlabel("Routed expert (layer-local)", labelpad=5)
            else:
                ax_heatmap.set_xticklabels([])
            ax_heatmap.set_yticks(range(len(layers)), [str(layer) for layer in layers])
            ax_heatmap.set_ylabel("MoE layer", labelpad=5)
            ax_heatmap.tick_params(length=0)
            # Fine white separators clarify cells without making a heavy table.
            ax_heatmap.set_xticks(np.arange(-0.5, POSITION_DIAG_ROUTED_EXPERTS, 1), minor=True)
            ax_heatmap.set_yticks(np.arange(-0.5, len(layers), 1), minor=True)
            ax_heatmap.grid(which="minor", color="white", linewidth=0.8)
            ax_heatmap.tick_params(which="minor", bottom=False, left=False)
            for spine in ax_heatmap.spines.values():
                spine.set_visible(False)
            for y in range(activation.shape[0]):
                for x_index in range(activation.shape[1]):
                    value = activation[y, x_index]
                    ax_heatmap.text(
                        x_index, y, f"{value:.1f}",
                        ha="center", va="center", fontsize=8.1, fontweight="semibold",
                        color="white" if value >= 58.0 else "#16334A",
                    )

        header_support.legend(
            legend_handles, [f"K={k}" for k in range(1, POSITION_DIAG_ROUTED_EXPERTS + 1)],
            ncol=4, loc="upper right", bbox_to_anchor=(1.0, 1.10),
            frameon=False, handlelength=1.15, columnspacing=1.0, fontsize=8.7,
        )
        colorbar = figure.colorbar(last_image, ax=heatmap_axes, fraction=0.025, pad=0.025)
        colorbar.set_label(r"$P(p_{\ell,e}>0)$ (%)", color="#23384A", labelpad=7)
        colorbar.set_ticks([0, 25, 50, 75, 100])
        colorbar.outline.set_linewidth(0.45)
        colorbar.outline.set_edgecolor("#B7C4D0")
        colorbar.ax.tick_params(width=0.5, length=2.5)

        figure.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
        figure.savefig(Path(output_path).with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
        display(figure)
        plt.close(figure)

        denominators = "; ".join(f"{label}={count:,}" for label, count in denominator_text.items())
        return (
            f"Sparsegen routing on validation {scope_label} ({page_count} input page(s)). "
            f"(a) shows adaptive support size K, the number of routed experts with p>0; "
            f"(b) shows routed-expert activation P(p_(l,e)>0), using a fixed 0–100% colour scale. "
            f"Routing is recorded teacher-forced over all attended positions and partitioned into Vision, "
            f"Prompt/other, and HTR-target positions; the always-active shared expert is excluded. "
            f"Per-layer denominators: {denominators}."
        )

    def _posdiag_safe_filename(text):
        return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")

    def _posdiag_bootstrap_ci(layer_bucket, seed):
        page_tokens = np.asarray(layer_bucket["page_token_counts"], dtype=np.float64)
        page_support = np.asarray(layer_bucket["page_support_sums"], dtype=np.float64)
        if page_tokens.size != page_support.size or page_tokens.size == 0:
            raise RuntimeError("Invalid per-page all-attended support statistics.")
        point = float(page_support.sum() / page_tokens.sum())
        if page_tokens.size == 1:
            return point, point, point
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, page_tokens.size, size=(int(POSITION_DIAG_BOOTSTRAP_REPLICATES), page_tokens.size))
        values = page_support[indices].sum(axis=1) / page_tokens[indices].sum(axis=1)
        low, high = np.quantile(values, [0.025, 0.975]).tolist()
        return point, float(low), float(high)

    def _position_main_figure(raw_all_attended, report_all_attended, output_path):
        # Expert indices are layer-local. Rank activation independently at each
        # layer instead of drawing an invalid cross-layer E1/E2/E3/E4 identity.
        layers = list(posdiag_actual_layers)
        mean_support, ci_low, ci_high, activation = [], [], [], []
        for layer in layers:
            point, low, high = _posdiag_bootstrap_ci(
                raw_all_attended[layer], seed=int(POSITION_DIAG_SEED) + 1009 * int(layer)
            )
            row = report_all_attended[str(layer)]
            reported = float(row["mean_active_routed_experts"])
            if not math.isclose(point, reported, rel_tol=0.0, abs_tol=1e-8):
                raise RuntimeError(f"E[K] mismatch at layer={layer}: bootstrap input={point} report={reported}")
            expert_activation = np.asarray([
                float(row["expert_activation_percent"][f"E{expert}"])
                for expert in range(1, POSITION_DIAG_ROUTED_EXPERTS + 1)
            ], dtype=float)
            if not math.isclose(float(expert_activation.sum() / 100.0), point, rel_tol=0.0, abs_tol=1e-6):
                raise RuntimeError(f"Support/activation invariant failed at layer={layer}.")
            mean_support.append(point); ci_low.append(low); ci_high.append(high); activation.append(expert_activation)
        mean_support = np.asarray(mean_support, dtype=float)
        ci_low = np.asarray(ci_low, dtype=float); ci_high = np.asarray(ci_high, dtype=float)
        activation = np.asarray(activation, dtype=float)
        ranked_activation = np.sort(activation, axis=1)[:, ::-1]
        balanced_reference = 100.0 * mean_support / float(POSITION_DIAG_ROUTED_EXPERTS)

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
            ax_support.set_ylim(max(0.95, float(np.floor((ci_low.min() - 0.12) * 5) / 5)), min(float(POSITION_DIAG_ROUTED_EXPERTS), float(np.ceil((ci_high.max() + 0.12) * 5) / 5)))
            ax_support.grid(axis="y", color="#DCE5EB", linewidth=0.65); ax_support.set_axisbelow(True)
            ax_support.legend(loc="lower left", frameon=False, fontsize=8.4, handlelength=1.4)

            rank_colors = ["#083B66", "#2C75A7", "#68A9CF", "#A9D2E5"]
            rank_markers = ["o", "s", "^", "D"]
            rank_styles = ["-", "--", ":", "-."]
            rank_labels = ["1st most active", "2nd most active", "3rd most active", "Least active"]
            ax_util.plot(x, balanced_reference, color="#5D6670", linestyle=(0, (4, 3)), linewidth=1.5, label=r"Equal utilization: $\mathbb{E}[K_\ell]/4$", zorder=1)
            for rank in range(POSITION_DIAG_ROUTED_EXPERTS):
                ax_util.plot(x, ranked_activation[:, rank], color=rank_colors[rank], linestyle=rank_styles[rank], linewidth=2.0, marker=rank_markers[rank], markersize=5.0, label=rank_labels[rank], zorder=2)
            ax_util.set_title("(b) Ranked routed-expert activation", loc="left", pad=23)
            ax_util.text(0.0, 1.035, "Experts ranked by activation frequency within each layer", transform=ax_util.transAxes, fontsize=8.3, color="#5A6D7E", va="bottom")
            ax_util.set_xlabel("MoE layer")
            ax_util.set_ylabel(r"Activation probability $P(p_{\ell,e}>0)$ (%)")
            ax_util.set_xticks(x, [str(layer) for layer in layers]); ax_util.set_ylim(0.0, 100.0)
            ax_util.set_yticks([0, 25, 50, 75, 100]); ax_util.grid(axis="y", color="#DCE5EB", linewidth=0.65); ax_util.set_axisbelow(True)
            ax_util.legend(loc="lower center", bbox_to_anchor=(0.5, -0.49), ncol=2, frameon=False, fontsize=7.9, columnspacing=1.0, handlelength=1.7)
            for axis in (ax_support, ax_util):
                axis.spines["top"].set_visible(False); axis.spines["right"].set_visible(False)
                axis.spines["left"].set_color("#AEBBC5"); axis.spines["bottom"].set_color("#AEBBC5")
                axis.spines["left"].set_linewidth(0.65); axis.spines["bottom"].set_linewidth(0.65); axis.tick_params(width=0.65, length=3.0)
            figure.subplots_adjust(left=0.09, right=0.985, top=0.88, bottom=0.32, wspace=0.34)
            figure.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
            figure.savefig(Path(output_path).with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
            display(figure); plt.close(figure)
        return {
            "mean_active_routed_experts": {str(layer): float(value) for layer, value in zip(layers, mean_support)},
            "mean_support_ci95_page_bootstrap": {str(layer): [float(low), float(high)] for layer, low, high in zip(layers, ci_low, ci_high)},
            "equal_utilization_reference_percent": {str(layer): float(value) for layer, value in zip(layers, balanced_reference)},
            "ranked_activation_percent": {str(layer): [float(value) for value in ranked_activation[idx]] for idx, layer in enumerate(layers)},
        }

    posdiag_figure_paths = {}
    posdiag_caption_texts = {}
    posdiag_figure_statistics = {}
    for _source, _source_report in sorted(posdiag_report_by_source.items()):
        _source_figure_path = posdiag_output_dir / f"routing_main_all_attended_{_posdiag_safe_filename(_source)}.png"
        posdiag_figure_statistics[_source] = _position_main_figure(
            posdiag_accumulators_by_source[_source].data["all_attended"],
            _source_report["all_attended"], _source_figure_path,
        )
        posdiag_figure_paths[_source] = str(_source_figure_path)

    posdiag_figure_path = posdiag_output_dir / "routing_main_all_attended_all_sources_pooled.png"
    posdiag_figure_statistics["All_sources_pooled"] = _position_main_figure(
        posdiag_accumulator.data["all_attended"], posdiag_report["all_attended"], posdiag_figure_path,
    )
    posdiag_figure_paths["All_sources_pooled"] = str(posdiag_figure_path)
    posdiag_caption_texts["All_sources_pooled"] = (
        "Sparsegen routing on all attention_mask=1 positions. Panel (a) reports E[K] with 95% page-bootstrap intervals; "
        "panel (b) ranks routed-expert activation independently within each layer and compares it with E[K]/4. "
        "Vision, prompt/context, and teacher-forced transcription positions are included; padding and the shared expert are excluded."
    )

    posdiag_caption_path = posdiag_output_dir / "validation_routing_position_captions.txt"
    posdiag_caption_path.write_text(
        "\n\n".join(
            f"[{scope}]\n{caption}" for scope, caption in posdiag_caption_texts.items()
        ) + "\n",
        encoding="utf-8",
    )

    posdiag_vector_figure_paths = {
        scope: str(Path(path).with_suffix(".pdf"))
        for scope, path in posdiag_figure_paths.items()
    }
    posdiag_json_path = posdiag_output_dir / "validation_routing_position_report.json"
    posdiag_json_path.write_text(
        json.dumps(
            {
                "checkpoint": str(posdiag_checkpoint),
                "checkpoint_moe_signature": posdiag_checkpoint_signature,
                "selection": {
                    "split": "valid",
                    "task": "page_htr",
                    "max_valid_pages_per_source": int(POSITION_DIAG_MAX_VALID_PAGES_PER_SOURCE),
                    "seed_when_capped": int(POSITION_DIAG_SEED),
                    "by_source": posdiag_selection,
                },
                "position_definition": {
                    "all_attended": "Every attention_mask=1 position. This is the mask used by the current global load-balance loss.",
                    "vision": "Non-padding image-pad token positions; vision boundary tokens are excluded.",
                    "htr_target": "Causal hidden state t where labels[t+1] is supervised (labels[t+1] != IGNORE_INDEX).",
                    "prompt_other_context": "Exact residual all_attended minus vision minus htr_target; includes prompt/chat boundary and attended terminal context.",
                    "partition_invariant": "all_attended = vision disjoint-union htr_target disjoint-union prompt_other_context.",
                    "support_size": "K = number of routed E1..E4 with native Sparsegen probability p>0.",
                    "expert_activation": "P(p_(layer,expert)>0) over positions in that category.",
                    "routing_mass": "E[p_(layer,expert)] over positions in that category; retained in CSV/JSON as a complementary collapse diagnostic.",
                    "shared_expert": "Excluded because it is always active and is not Sparsegen-routed.",
                    "aggregation": "Micro-average over positions; no source reweighting.",
                },
                "verified_image_pad_token_ids": sorted(posdiag_vision_ids),
                "partition_totals_all_layers": posdiag_partition_totals,
                "statistics_all_sources_pooled": posdiag_report,
                "statistics_by_source": posdiag_report_by_source,
                "figure_paths_png": posdiag_figure_paths,
                "figure_paths_pdf": posdiag_vector_figure_paths,
                "main_figure_statistics": posdiag_figure_statistics,
                "paper_captions": posdiag_caption_texts,
            },
            ensure_ascii=False, indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print("Validation routing-position diagnostic complete.")
    print("Selected validation pages by source:", {
        source: info["selected_pages"] for source, info in posdiag_selection.items()
    })
    print("Verified image-pad token IDs:", sorted(posdiag_vision_ids))
    print("PNG dashboards by source:", posdiag_figure_paths)
    print("PDF dashboards by source:", posdiag_vector_figure_paths)
    print("CSV:", posdiag_csv_path)
    print("Captions:", posdiag_caption_path)
    print("JSON:", posdiag_json_path)

    return export_runtime_locals(state, locals())
