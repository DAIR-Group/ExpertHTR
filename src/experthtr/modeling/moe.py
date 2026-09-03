"""modeling.moe phase for the ExpertHTR research workflow.

This module preserves the audited runtime computation. It is invoked by the
explicit CLI workflow and exports its declared runtime objects into a
:class:`~experthtr.runtime.context.RuntimeState`.
"""

from __future__ import annotations

from experthtr.runtime.context import RuntimeState, export_runtime_locals


def register_moe_components(state: RuntimeState) -> RuntimeState:
    """Execute this phase after its prerequisites have populated ``state``."""

    globals().update(state.values)
    # -----------------------------------------------------------------------------
    #  Model loading + source-agnostic routed MLP-MoE wrapper
    # -----------------------------------------------------------------------------


    def get_device() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def get_model_dtype(device: torch.device) -> torch.dtype:
        requested = os.getenv("HWVLM_TORCH_DTYPE", globals().get("CFG", {}).get("torch_dtype", "float16")).strip().lower()
        if device.type != "cuda":
            return torch.float32
        if requested in {"bf16", "bfloat16"}:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            LOGGER.warning("BF16 requested but not supported by this GPU; falling back to float16.")
            return torch.float16
        if requested in {"fp32", "float32"}:
            return torch.float32
        return torch.float16


    def unwrap_model(model_obj: nn.Module) -> nn.Module:
        return model_obj.module if isinstance(model_obj, nn.DataParallel) else model_obj


    def maybe_wrap_data_parallel(model_obj: nn.Module, cfg: Dict[str, Any]) -> nn.Module:
        if device.type == "cuda" and torch.cuda.device_count() > 1 and cfg.get("use_data_parallel", True):
            LOGGER.info("Using torch.nn.DataParallel across %d CUDA GPUs", torch.cuda.device_count())
            print(f"DataParallel: using {torch.cuda.device_count()} CUDA GPUs")
            return nn.DataParallel(model_obj)
        if device.type == "cuda" and torch.cuda.device_count() > 1:
            LOGGER.info("Multiple GPUs detected, but DataParallel is disabled for Qwen VLM stability. Set HWVLM_USE_DATA_PARALLEL=1 only for experiments.")
            print(f"Multiple CUDA GPUs detected ({torch.cuda.device_count()}), using one GPU because DataParallel can break Qwen VLM visual modules.")
        return model_obj


    def set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


    def _processor_size_value(size: Any, key: str) -> Optional[int]:
        if size is None:
            return None
        try:
            value = size[key]
        except (KeyError, TypeError, AttributeError):
            value = getattr(size, key, None)
        return None if value is None else int(value)


    def _configure_processor_image_budget(processor, min_pixels: int, max_pixels: int) -> Dict[str, int]:
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is None:
            raise RuntimeError("Loaded processor has no image_processor; cannot freeze image pixel budget.")

        min_pixels = int(min_pixels)
        max_pixels = int(max_pixels)
        configured = False
        size = getattr(image_processor, "size", None)
        size_update_error = None
        if size is not None:
            try:
                # Transformers 5.x uses image_utils.SizeDict here. It is not a
                # built-in dict, but intentionally supports item assignment.
                size["shortest_edge"] = min_pixels
                size["longest_edge"] = max_pixels
                configured = True
            except (KeyError, TypeError, AttributeError) as exc:
                size_update_error = exc
                try:
                    setattr(size, "shortest_edge", min_pixels)
                    setattr(size, "longest_edge", max_pixels)
                    configured = True
                except (AttributeError, TypeError) as attr_exc:
                    size_update_error = attr_exc
        for attr, value in (("min_pixels", min_pixels), ("max_pixels", max_pixels)):
            if hasattr(image_processor, attr):
                setattr(image_processor, attr, value)
                configured = True
        if not configured:
            raise RuntimeError(
                "Image processor pixel budget is not configurable. "
                f"processor={type(image_processor).__name__} "
                f"size_type={type(size).__name__} size={size!r} "
                f"last_error={size_update_error!r}"
            )

        effective_size = getattr(image_processor, "size", None)
        actual_min = _processor_size_value(effective_size, "shortest_edge")
        actual_max = _processor_size_value(effective_size, "longest_edge")
        if actual_min is None or actual_max is None:
            actual_min = getattr(image_processor, "min_pixels", None)
            actual_max = getattr(image_processor, "max_pixels", None)
            actual_min = None if actual_min is None else int(actual_min)
            actual_max = None if actual_max is None else int(actual_max)
        if (actual_min, actual_max) != (min_pixels, max_pixels):
            raise RuntimeError(
                "Failed to freeze processor image pixel budget: "
                f"requested={(min_pixels, max_pixels)} actual={(actual_min, actual_max)} "
                f"processor={type(image_processor).__name__} size={effective_size!r}"
            )
        LOGGER.info(
            "Processor image pixel budget frozen: min_pixels=%d max_pixels=%d "
            "processor=%s size_type=%s",
            actual_min,
            actual_max,
            type(image_processor).__name__,
            type(effective_size).__name__,
        )
        return {"shortest_edge": actual_min, "longest_edge": actual_max}


    def load_processor(model_name_or_path: str, min_pixels: int, max_pixels: int):
        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            local_files_only=True,
            size={"shortest_edge": int(min_pixels), "longest_edge": int(max_pixels)},
        )
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None:
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
        processor._htr_image_budget = _configure_processor_image_budget(
            processor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        return processor


    def _single_token_id(tokenizer, text: str) -> Optional[int]:
        encoded = tokenizer(text, add_special_tokens=False, return_attention_mask=False).get("input_ids", [])
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        return int(encoded[0]) if len(encoded) == 1 else None


    def _atomic_ocr_markers_ready(tokenizer, markers: List[str]) -> bool:
        special_ids = {int(value) for value in getattr(tokenizer, "all_special_ids", [])}
        return all(
            (token_id := _single_token_id(tokenizer, marker)) is not None and token_id not in special_ids
            for marker in markers
        )


    def ensure_atomic_ocr_markers(
        tokenizer,
        model_obj: nn.Module,
        markers: List[str],
        initialize_added_rows: bool,
    ) -> Dict[str, int]:
        """Add CER markers as ordinary atomic tokens and resize model embeddings safely."""
        markers = list(markers)
        old_vocab_size = int(model_obj.get_input_embeddings().weight.shape[0])
        old_piece_ids = {}
        for marker in markers:
            encoded = tokenizer(marker, add_special_tokens=False, return_attention_mask=False).get("input_ids", [])
            if encoded and isinstance(encoded[0], list):
                encoded = encoded[0]
            old_piece_ids[marker] = [int(value) for value in encoded if 0 <= int(value) < old_vocab_size]

        if not _atomic_ocr_markers_ready(tokenizer, markers):
            added_tokens = [
                transformers.AddedToken(
                    marker, single_word=False, lstrip=False, rstrip=False,
                    normalized=False, special=False,
                )
                for marker in markers
            ]
            tokenizer.add_tokens(added_tokens, special_tokens=False)

        target_vocab_size = len(tokenizer)
        input_module_before = model_obj.get_input_embeddings()
        output_module_before = model_obj.get_output_embeddings()
        input_seeds = {}
        output_seeds = {}
        if initialize_added_rows:
            with torch.no_grad():
                for marker, pieces in old_piece_ids.items():
                    if not pieces:
                        continue
                    piece_tensor = torch.tensor(pieces, device=input_module_before.weight.device, dtype=torch.long)
                    input_seeds[marker] = input_module_before.weight.index_select(0, piece_tensor).mean(dim=0).detach().clone()
                    if output_module_before is not None:
                        output_piece_tensor = piece_tensor.to(output_module_before.weight.device)
                        output_seeds[marker] = output_module_before.weight.index_select(0, output_piece_tensor).mean(dim=0).detach().clone()
        if int(model_obj.get_input_embeddings().weight.shape[0]) != target_vocab_size:
            try:
                model_obj.resize_token_embeddings(target_vocab_size, mean_resizing=False)
            except TypeError:
                model_obj.resize_token_embeddings(target_vocab_size)

        marker_ids = {marker: _single_token_id(tokenizer, marker) for marker in markers}
        if any(value is None for value in marker_ids.values()):
            raise RuntimeError(f"OCR markers are not atomic after tokenizer update: {marker_ids}")
        special_ids = {int(value) for value in getattr(tokenizer, "all_special_ids", [])}
        if any(int(value) in special_ids for value in marker_ids.values()):
            raise RuntimeError("<del>/<gap> must be ordinary added tokens; special tokens are stripped at decode.")

        if initialize_added_rows:
            input_after = model_obj.get_input_embeddings().weight
            output_after_module = model_obj.get_output_embeddings()
            with torch.no_grad():
                for marker, marker_id in marker_ids.items():
                    marker_id = int(marker_id)
                    if marker_id < old_vocab_size:
                        continue
                    if marker not in input_seeds:
                        raise RuntimeError(f"Cannot initialize new marker {marker!r}: no pre-addition subtokens.")
                    input_after[marker_id].copy_(input_seeds[marker].to(device=input_after.device, dtype=input_after.dtype))
                    if output_after_module is not None and marker in output_seeds:
                        output_after_module.weight[marker_id].copy_(
                            output_seeds[marker].to(device=output_after_module.weight.device, dtype=output_after_module.weight.dtype)
                        )

        for marker, marker_id in marker_ids.items():
            decoded = tokenizer.decode([int(marker_id)], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            if decoded != marker:
                raise RuntimeError(f"Atomic marker round-trip failed: {marker!r} -> {marker_id} -> {decoded!r}")
        LOGGER.info("Atomic OCR marker contract: %s | vocab=%d", marker_ids, target_vocab_size)
        return {marker: int(marker_id) for marker, marker_id in marker_ids.items()}


    def _force_config_attention_implementation(config_obj, implementation: str = "eager") -> None:
        if config_obj is None:
            return
        for attr in ["_attn_implementation", "attn_implementation", "_attn_implementation_internal"]:
            try:
                setattr(config_obj, attr, implementation)
            except Exception:
                pass
        for child_name in ["text_config", "vision_config", "llm_config", "language_config"]:
            child = getattr(config_obj, child_name, None)
            if child is not None and child is not config_obj:
                _force_config_attention_implementation(child, implementation)


    def _model_attention_backend_name(model_obj) -> str:
        cfg = getattr(model_obj, "config", None)
        return str(
            getattr(cfg, "_attn_implementation", None)
            or getattr(cfg, "attn_implementation", None)
            or getattr(cfg, "_attn_implementation_internal", None)
            or "unknown"
        )


    def load_qwen3_vlm(model_name_or_path: str, device: torch.device, use_flash: bool = True):
        try:
            from transformers import Qwen3_5ForConditionalGeneration
            model_cls = Qwen3_5ForConditionalGeneration
            LOGGER.info("Using Qwen3_5ForConditionalGeneration")
        except ImportError:
            try:
                from transformers import AutoModelForVision2Seq
                model_cls = AutoModelForVision2Seq
                LOGGER.info("Using AutoModelForVision2Seq")
            except ImportError:
                from transformers import AutoModelForCausalLM
                model_cls = AutoModelForCausalLM
                LOGGER.warning("Falling back to AutoModelForCausalLM")

        load_kwargs = {
            "pretrained_model_name_or_path": model_name_or_path,
            "torch_dtype": get_model_dtype(device),
            "trust_remote_code": True,
            "local_files_only": True,
        }
        if device.type == "cuda" and use_flash:
            load_kwargs["attn_implementation"] = "flash_attention_2"
        elif not use_flash:
            try:
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=True,
                    local_files_only=True,
                )
                _force_config_attention_implementation(config, "eager")
                load_kwargs["config"] = config
            except Exception as exc:
                LOGGER.warning("Could not preload config for eager attention: %s", exc)
            load_kwargs["attn_implementation"] = "eager"

        try:
            model = model_cls.from_pretrained(**load_kwargs)
        except Exception as exc:
            failed_attn_impl = load_kwargs.pop("attn_implementation", None)
            if failed_attn_impl is not None:
                LOGGER.warning("Falling back after attn_implementation=%s failed: %s", failed_attn_impl, exc)
            model = model_cls.from_pretrained(**load_kwargs)

        if not use_flash:
            _force_config_attention_implementation(getattr(model, "config", None), "eager")
            LOGGER.info("Loaded model attention backend: %s", _model_attention_backend_name(model))

        model.to(device)
        return model




    # -----------------------------------------------------------------------------
    #  Source-agnostic sparse / Top-K MLP-MoE
    # -----------------------------------------------------------------------------

    from collections import Counter

    MOE_MLP_STATE_FILE = "moe_mlp_state.pt"
    MOE_ROUTED_PREFIX = "lang"
    MOE_SHARED_PREFIX = "shared"


    def _expert_names(prefix: str, count: int) -> List[str]:
        return [f"{prefix}_{idx}" for idx in range(max(0, int(count)))]


    def per_sample_masked_lm_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if logits is None or labels is None:
            device = logits.device if isinstance(logits, torch.Tensor) else labels.device
            batch_size = int(labels.shape[0]) if isinstance(labels, torch.Tensor) and labels.ndim > 0 else 0
            zeros = torch.zeros((batch_size,), device=device)
            return zeros, torch.zeros((batch_size,), dtype=torch.bool, device=device)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        valid_mask = shift_labels.ne(IGNORE_INDEX)
        if not valid_mask.any():
            zeros = torch.zeros((shift_labels.shape[0],), device=logits.device)
            return zeros, torch.zeros((shift_labels.shape[0],), dtype=torch.bool, device=logits.device)
        token_losses = F.cross_entropy(
            shift_logits.float().view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).view_as(shift_labels)
        valid_float = valid_mask.float()
        token_counts = valid_float.sum(dim=1)
        sample_losses = (token_losses * valid_float).sum(dim=1) / token_counts.clamp_min(1.0)
        valid_samples = token_counts.gt(0)
        sample_losses = torch.where(valid_samples, sample_losses, torch.zeros_like(sample_losses))
        return sample_losses, valid_samples


    def masked_lm_loss_sum_and_count(
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return CE numerator and supervised-token denominator for exact accumulation."""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        valid = shift_labels.ne(IGNORE_INDEX)
        token_count = valid.sum().to(device=logits.device, dtype=torch.float32)
        if not bool(valid.any().item()):
            return logits.sum() * 0.0, token_count
        loss_sum = F.cross_entropy(
            shift_logits.float().view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
        return loss_sum, token_count


    def token_unlikelihood_mean(
        logits: torch.Tensor,
        labels: torch.Tensor,
        active_rows: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Previous-context token UL (Welleck et al.) on selected page_htr rows."""
        zero = logits.sum() * 0.0
        if CFG.get("token_ul_coef", 0.0) <= 0.0 or active_rows is None:
            return zero, torch.zeros((), device=logits.device, dtype=torch.float32)
        active_rows = active_rows.to(device=logits.device, dtype=torch.bool).view(-1)
        if not bool(active_rows.any().item()):
            return zero, torch.zeros((), device=logits.device, dtype=torch.float32)

        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        context_window = int(CFG["token_ul_context_window"])
        min_position = int(CFG["token_ul_min_position"])
        ul_sum = zero
        candidate_count = 0

        for row_idx in active_rows.nonzero(as_tuple=False).flatten().tolist():
            valid_positions = shift_labels[row_idx].ne(IGNORE_INDEX).nonzero(as_tuple=False).flatten()
            if valid_positions.numel() <= min_position:
                continue
            answer_ids = shift_labels[row_idx].index_select(0, valid_positions).detach().long()
            position_specs = []
            for answer_pos in range(min_position, int(answer_ids.numel())):
                current_id = int(answer_ids[answer_pos].item())
                start = max(0, answer_pos - context_window)
                candidates = sorted({
                    int(value) for value in answer_ids[start:answer_pos].tolist()
                    if int(value) != current_id and int(value) != IGNORE_INDEX
                })
                if candidates:
                    position_specs.append((int(valid_positions[answer_pos].item()), candidates))

            # Chunk selected positions so UL never materializes B x T x V float32 probabilities.
            for chunk_start in range(0, len(position_specs), 16):
                chunk = position_specs[chunk_start:chunk_start + 16]
                positions = torch.tensor([item[0] for item in chunk], device=logits.device, dtype=torch.long)
                selected_logits = shift_logits[row_idx].index_select(0, positions).float()
                log_normalizers = torch.logsumexp(selected_logits, dim=-1)
                for local_idx, (_position, candidates) in enumerate(chunk):
                    candidate_ids = torch.tensor(candidates, device=logits.device, dtype=torch.long)
                    log_probs = selected_logits[local_idx].index_select(0, candidate_ids) - log_normalizers[local_idx]
                    probs = log_probs.exp().clamp(min=0.0, max=1.0 - 1e-6)
                    ul_sum = ul_sum + (-torch.log1p(-probs)).sum()
                    candidate_count += int(candidate_ids.numel())

        if candidate_count == 0:
            return zero, torch.zeros((), device=logits.device, dtype=torch.float32)
        return ul_sum / float(candidate_count), torch.tensor(float(candidate_count), device=logits.device)


    def collapse_aware_lm_objective(
        logits: torch.Tensor,
        labels: torch.Tensor,
        page_htr_rows: Optional[torch.Tensor],
        ul_active_rows: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        valid = shift_labels.ne(IGNORE_INDEX)
        actual_token_count = valid.sum().to(device=logits.device, dtype=torch.float32)
        if not bool(valid.any().item()):
            zero = logits.sum() * 0.0
            return {
                "objective_sum": zero, "normalizer": actual_token_count,
                "ce_sum": zero, "ul_mean": zero,
                "ul_scaled_mean": zero, "ul_candidate_count": actual_token_count,
                "actual_token_count": actual_token_count,
            }

        token_losses = F.cross_entropy(
            shift_logits.float().view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1), ignore_index=IGNORE_INDEX, reduction="none",
        ).view_as(shift_labels)
        token_weights = valid.float()
        if page_htr_rows is not None and float(CFG.get("page_htr_eos_weight", 1.0)) > 1.0:
            page_htr_rows = page_htr_rows.to(device=logits.device, dtype=torch.bool).view(-1, 1)
            eos_mask = valid & page_htr_rows & shift_labels.eq(int(CFG["training_terminal_id"]))
            eos_counts = eos_mask.sum(dim=1)
            active_page_rows = page_htr_rows.view(-1)
            if bool(active_page_rows.any().item()) and not bool(torch.all(eos_counts[active_page_rows] == 1).item()):
                raise RuntimeError(f"Each page_htr answer must supervise exactly one EOS; counts={eos_counts.tolist()}")
            token_weights = token_weights + eos_mask.float() * (float(CFG["page_htr_eos_weight"]) - 1.0)

        normalizer = token_weights.sum().clamp_min(1.0)
        ce_sum = (token_losses * token_weights).sum()
        ul_mean, ul_candidate_count = token_unlikelihood_mean(logits, labels, ul_active_rows)
        ul_scaled_mean = ul_mean * float(CFG.get("token_ul_coef", 0.0))
        objective_sum = ce_sum + ul_scaled_mean * normalizer
        return {
            "objective_sum": objective_sum, "normalizer": normalizer,
            "ce_sum": ce_sum, "ul_mean": ul_mean,
            "ul_scaled_mean": ul_scaled_mean, "ul_candidate_count": ul_candidate_count,
            "actual_token_count": actual_token_count,
        }


    def weighted_sample_lm_loss(
        sample_losses: torch.Tensor,
        valid_samples: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if sample_losses.numel() == 0 or not valid_samples.any():
            return sample_losses.sum() * 0.0
        if sample_weights is None:
            return sample_losses[valid_samples].mean()
        weights = sample_weights.to(device=sample_losses.device, dtype=torch.float32).view(-1)
        if weights.numel() != sample_losses.numel():
            return sample_losses[valid_samples].mean()
        weights = torch.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(0.0)
        valid_weights = weights * valid_samples.float()
        denom = valid_weights.sum().clamp_min(1e-8)
        return (sample_losses * valid_weights).sum() / denom


    def masked_lm_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if logits is None or labels is None:
            device = logits.device if isinstance(logits, torch.Tensor) else labels.device
            return torch.zeros((), device=device)
        sample_losses, valid_samples = per_sample_masked_lm_loss(logits, labels)
        if not valid_samples.any():
            return logits.sum() * 0.0
        return weighted_sample_lm_loss(sample_losses, valid_samples, sample_weights)


    class MoERoutingState:
        """Collects optional router regularizers and routed-expert usage statistics."""

        def __init__(
            self, load_balance_coef: float = 0.0, router_ce_coef: float = 0.0,
            lam_sparse_coef: float = 0.0,
        ):
            self.load_balance_coef = float(load_balance_coef)
            self.router_ce_coef = float(router_ce_coef)
            self.lam_sparse_coef = float(lam_sparse_coef)
            self.lam_sparse_warmup_scale = 1.0
            self.load_balance_warmup_scale = 1.0
            self._balance_losses: List[torch.Tensor] = []
            self._router_logits: List[torch.Tensor] = []
            self._lambda_values: List[torch.Tensor] = []
            self._token_mask: Optional[torch.Tensor] = None
            self.expected_router_layers = 0
            self.usage: Counter = Counter()
            # Counts token-layer positions rather than just token rows, because every
            # selected MoE layer makes an independent routing decision.
            self.routed_token_layer_positions = 0
            self.routed_active_assignments = 0

        def reset_forward(self) -> None:
            self._balance_losses = []
            self._router_logits = []
            self._lambda_values = []
            self._token_mask = None

        def set_token_mask(self, token_mask: Optional[torch.Tensor]) -> None:
            if isinstance(token_mask, torch.Tensor) and token_mask.ndim == 2:
                self._token_mask = token_mask.detach().ne(0)
            else:
                self._token_mask = None

        def token_mask_for(self, routing_weight: torch.Tensor) -> Optional[torch.Tensor]:
            if not isinstance(self._token_mask, torch.Tensor) or routing_weight.ndim < 3:
                return None
            if tuple(self._token_mask.shape) != tuple(routing_weight.shape[:2]):
                return None
            return self._token_mask.to(device=routing_weight.device, dtype=torch.bool)

        def add_balance_loss(self, loss: torch.Tensor) -> None:
            self._balance_losses.append(loss)

        def add_router_logits(self, logits: torch.Tensor) -> None:
            self._router_logits.append(logits)

        def set_lam_sparse_warmup_scale(self, scale: float) -> None:
            self.lam_sparse_warmup_scale = min(max(float(scale), 0.0), 1.0)

        def set_load_balance_warmup_scale(self, scale: float) -> None:
            self.load_balance_warmup_scale = min(max(float(scale), 0.0), 1.0)

        def add_lambda_values(self, lam: Optional[torch.Tensor]) -> None:
            if self.lam_sparse_coef <= 0 or not isinstance(lam, torch.Tensor) or lam.numel() == 0:
                return
            values = lam.float()
            token_mask = self.token_mask_for(values.unsqueeze(-1))
            if token_mask is not None:
                values = values[token_mask]
            if values.numel() > 0:
                self._lambda_values.append(values)

        def add_usage(self, routing_weight: torch.Tensor, expert_names: List[str]) -> None:
            if routing_weight.numel() == 0:
                return
            flat_weight = routing_weight.detach().float().reshape(-1, len(expert_names))
            token_mask = self.token_mask_for(routing_weight)
            if token_mask is not None:
                flat_weight = flat_weight[token_mask.reshape(-1)]
                if flat_weight.numel() == 0:
                    return
            active = flat_weight.gt(0)
            self.routed_token_layer_positions += int(flat_weight.shape[0])
            self.routed_active_assignments += int(active.sum().item())
            masses = flat_weight.sum(dim=0).cpu()
            for idx, mass in enumerate(masses.tolist()):
                if mass:
                    self.usage[expert_names[idx]] += float(mass)

        def balance_loss(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            zero = torch.zeros((), device=device)
            effective_coef = self.load_balance_coef * self.load_balance_warmup_scale
            if not self._balance_losses or effective_coef <= 0:
                return zero, zero, zero
            # Switch-style importance × realized dispatch-frequency loss, averaged
            # across selected MoE layers. Shared experts are intentionally excluded.
            raw = torch.stack([loss.to(device) for loss in self._balance_losses]).mean()
            scaled = raw * effective_coef
            return raw, scaled, torch.as_tensor(effective_coef, device=device)

        def lam_sparse_loss(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            zero = torch.zeros((), device=device)
            effective_coef = self.lam_sparse_coef * self.lam_sparse_warmup_scale
            if effective_coef <= 0 or not self._lambda_values:
                return zero, zero, zero
            # With LogSigmoid parameterization lambda is strictly below 1, so this
            # is exactly LD-MoLE's LamSparseLoss: mean(1 - lambda).
            raw = torch.stack([1.0 - values.mean() for values in self._lambda_values]).mean()
            scaled = raw * effective_coef
            return raw, scaled, torch.as_tensor(effective_coef, device=device)

        def router_language_loss(
            self,
            labels: Optional[torch.Tensor],
            language_group_ids: Optional[torch.Tensor],
            class_weights: Optional[torch.Tensor],
            device: torch.device,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if self.router_ce_coef <= 0 or labels is None or language_group_ids is None or not self._router_logits:
                zero = torch.zeros((), device=device)
                return zero, zero, zero

            # Causal logits at position t predict labels at t+1. Supervise the
            # router at the same hidden-state positions used by the shifted LM loss.
            shifted_labels = labels[..., 1:]
            label_mask = shifted_labels.ne(IGNORE_INDEX)
            if label_mask.ndim != 2 or language_group_ids.ndim != 1 or language_group_ids.shape[0] != label_mask.shape[0]:
                zero = torch.zeros((), device=device)
                return zero, zero, zero
            if not label_mask.any():
                zero = torch.zeros((), device=device)
                return zero, zero, zero

            target_matrix = language_group_ids.to(device=device).view(-1, 1).expand_as(shifted_labels)
            losses = []
            accuracies = []
            entropies = []
            weight = class_weights.to(device=device, dtype=torch.float32) if isinstance(class_weights, torch.Tensor) else None

            for logits in self._router_logits:
                logits = logits.to(device=device)
                if logits.ndim != 3 or logits.shape[:2] != labels.shape:
                    continue
                shifted_logits = logits[..., :-1, :]
                valid_logits = torch.nan_to_num(shifted_logits[label_mask].float(), nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)
                valid_targets = target_matrix[label_mask].long()
                if valid_logits.numel() == 0:
                    continue
                losses.append(F.cross_entropy(valid_logits, valid_targets, weight=weight, reduction="mean"))
                pred = valid_logits.argmax(dim=-1)
                accuracies.append((pred == valid_targets).float().mean())
                probs = torch.softmax(valid_logits, dim=-1)
                entropies.append((-(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)).mean())

            if not losses:
                zero = torch.zeros((), device=device)
                return zero, zero, zero

            raw_loss = torch.stack(losses).mean()
            acc = torch.stack(accuracies).mean() if accuracies else torch.zeros((), device=device)
            entropy = torch.stack(entropies).mean() if entropies else torch.zeros((), device=device)
            return raw_loss * self.router_ce_coef, acc, entropy

        def get_and_reset_usage(self) -> Dict[str, Any]:
            token_layer_positions = int(self.routed_token_layer_positions)
            active_assignments = int(self.routed_active_assignments)
            usage = {
                "expert_mass": dict(self.usage),
                "routed_token_layer_positions": token_layer_positions,
                "routed_active_assignments": active_assignments,
                "mean_active_routed_experts_per_token_layer": (
                    active_assignments / max(token_layer_positions, 1)
                ),
            }
            self.usage = Counter()
            self.routed_token_layer_positions = 0
            self.routed_active_assignments = 0
            return usage


    class LanguageGroupedMoEMLP(nn.Module):
        def __init__(
            self,
            base_mlp: nn.Module,
            routing_state: MoERoutingState,
            language_experts: int,
            shared_experts: int,
            routing_fn: str,
            router_temperature: float,
            sparsegen_hidden: int,
            sparsegen_eps: float,
            shared_mix: float,
            routed_top_k: int,
            enforce_top_k: bool,
            expert_init: str = "copy",
            expert_train_dtype: str = "float32",
        ):
            super().__init__()
            self.routing_state = routing_state
            self.language_names = _expert_names(MOE_ROUTED_PREFIX, language_experts)
            self.shared_names = _expert_names(MOE_SHARED_PREFIX, shared_experts)
            self.routing_fn = str(routing_fn or "sparsegen").lower()
            self.router_temperature = max(float(router_temperature), 1e-4)
            self.sparsegen_eps = max(float(sparsegen_eps), 1e-6)
            self.shared_mix = min(max(float(shared_mix), 0.0), 0.95)
            self.routed_top_k = int(routed_top_k)
            self.enforce_top_k = bool(enforce_top_k)
            if not 1 <= self.routed_top_k <= len(self.language_names):
                raise ValueError(f"routed_top_k={self.routed_top_k} is invalid for {len(self.language_names)} routed experts.")
            self.expert_init = str(expert_init or "copy").lower()
            self.expert_train_dtype = str(expert_train_dtype or "float32").lower()

            hidden_size = self._infer_hidden_size(base_mlp)
            device, dtype = self._infer_device_dtype(base_mlp)
            router_dtype = torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype

            self.language_experts = nn.ModuleList([
                self._make_expert(base_mlp, idx) for idx in range(len(self.language_names))
            ])
            self.shared_experts = nn.ModuleList([
                self._make_expert(base_mlp, idx) for idx in range(len(self.shared_names))
            ])
            self.router = nn.Linear(hidden_size, len(self.language_names), bias=False, device=device, dtype=router_dtype)
            nn.init.normal_(self.router.weight, mean=0.0, std=1e-3)
            self.lambda_mlp = (
                nn.Sequential(
                    nn.Linear(hidden_size, int(sparsegen_hidden), device=device, dtype=router_dtype),
                    nn.ReLU(),
                    nn.Linear(int(sparsegen_hidden), 1, device=device, dtype=router_dtype),
                    nn.LogSigmoid(),
                )
                if self.routing_fn == "sparsegen" else None
            )

            for param in self.parameters():
                param.requires_grad = True

        def _make_expert(self, base_mlp: nn.Module, idx: int) -> nn.Module:
            expert = copy.deepcopy(base_mlp)
            _, base_dtype = self._infer_device_dtype(base_mlp)
            if self.expert_train_dtype in {"fp32", "float32"} and base_dtype in {torch.float16, torch.bfloat16}:
                expert = expert.float()
            if self.expert_init == "noise" and idx > 0:
                with torch.no_grad():
                    for param in expert.parameters():
                        if param.is_floating_point():
                            param.add_(torch.randn_like(param) * 1e-5)
            return expert

        def _run_expert(self, expert: nn.Module, x: torch.Tensor) -> torch.Tensor:
            expert_device, expert_dtype = self._infer_device_dtype(expert)
            expert_input = x.to(device=expert_device, dtype=expert_dtype)
            if expert_input.device.type == "cuda" and expert_dtype == torch.float32:
                with torch.amp.autocast("cuda", enabled=False):
                    return expert(expert_input).to(dtype=x.dtype)
            return expert(expert_input).to(dtype=x.dtype)

        @staticmethod
        def _infer_device_dtype(module: nn.Module) -> Tuple[torch.device, torch.dtype]:
            for param in module.parameters():
                return param.device, param.dtype
            return torch.device("cpu"), torch.float32

        @staticmethod
        def _infer_hidden_size(module: nn.Module) -> int:
            for attr in ["gate_proj", "up_proj", "fc1", "dense_h_to_4h"]:
                layer = getattr(module, attr, None)
                if isinstance(layer, nn.Linear):
                    return int(layer.in_features)
            for child in module.modules():
                if isinstance(child, nn.Linear):
                    return int(child.in_features)
            raise ValueError(f"Could not infer MLP hidden size for {module.__class__.__name__}")

        def _sparsegen(self, logits: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            if self.lambda_mlp is None:
                raise RuntimeError("Sparsegen routing requested but lambda_mlp is unavailable.")
            num_experts = logits.shape[-1]
            flat_logits = torch.nan_to_num(logits.reshape(-1, num_experts).float(), nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)
            flat_x = torch.nan_to_num(x.reshape(-1, x.shape[-1]).float(), nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
            z_sorted = torch.sort(flat_logits, descending=True, dim=-1)[0]
            z_cumsum = torch.cumsum(z_sorted, dim=-1)
            k = torch.arange(1, num_experts + 1, device=logits.device, dtype=flat_logits.dtype).view(1, -1)
            # lambda_mlp ends in LogSigmoid, matching LD-MoLE: lambda is
            # strictly below 1 after adding (1-eps), so LamSparseLoss is bounded.
            lam = self.lambda_mlp(flat_x).float() + (1.0 - self.sparsegen_eps)
            support = (1.0 - lam + k * z_sorted) > z_cumsum
            support_f = support.float()
            k_z = support_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
            tau = ((support_f * z_sorted).sum(dim=-1, keepdim=True) - 1.0 + lam) / k_z
            probs = (flat_logits - tau).clamp(min=0.0)
            denom = torch.clamp(1.0 - lam, min=self.sparsegen_eps)
            probs = probs / denom
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            return probs.reshape_as(logits).to(dtype=logits.dtype), lam.reshape(*logits.shape[:-1]).to(dtype=logits.dtype)

        def _routing_weight(self, logits: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            if self.routing_fn == "softmax":
                return torch.softmax(logits.float(), dim=-1).to(dtype=x.dtype), None
            if self.routing_fn != "sparsegen":
                raise ValueError(f"Unsupported MLP-MoE routing_fn={self.routing_fn!r}; use 'sparsegen' or 'softmax'.")
            return self._sparsegen(logits, x)

        def _enforce_routed_top_k(
            self,
            routing_weight: torch.Tensor,
            router_logits: torch.Tensor,
        ) -> torch.Tensor:
            """For the dedicated Top-K arm, keep exactly k routed experts/token."""
            expert_count = len(self.language_names)
            if not self.enforce_top_k or self.routed_top_k >= expert_count:
                return routing_weight
            indices = torch.topk(router_logits, k=self.routed_top_k, dim=-1).indices
            selected = routing_weight.gather(dim=-1, index=indices).float() + 1e-8
            selected = selected / selected.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            constrained = torch.zeros_like(routing_weight, dtype=selected.dtype)
            constrained.scatter_(dim=-1, index=indices, src=selected)
            return constrained.to(dtype=routing_weight.dtype)


        def _load_balance_loss(self, routing_weight: torch.Tensor) -> torch.Tensor:
            num_experts = len(self.language_names)
            flat_weight = routing_weight.reshape(-1, num_experts).float()
            if flat_weight.numel() == 0:
                return routing_weight.sum() * 0.0
            token_mask = self.routing_state.token_mask_for(routing_weight)
            if token_mask is not None:
                flat_weight = flat_weight[token_mask.reshape(-1)]
                if flat_weight.numel() == 0:
                    return routing_weight.sum() * 0.0
            active = (flat_weight > 0).float()
            activate_count = active.sum().clamp_min(1.0)
            freq = active.sum(dim=0) / activate_count
            prop = flat_weight.sum(dim=0) / flat_weight.shape[0]
            return num_experts * torch.sum(prop * freq)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
            router_input = x.to(dtype=self.router.weight.dtype).clamp(-1e4, 1e4)
            router_logits = self.router(router_input) / self.router_temperature
            router_logits = torch.nan_to_num(router_logits.float(), nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)
            routing_weight, _lam = self._routing_weight(router_logits, router_input)
            routing_weight = torch.nan_to_num(routing_weight, nan=0.0, posinf=0.0, neginf=0.0)
            routing_weight = self._enforce_routed_top_k(routing_weight, router_logits)

            if self.training:
                self.routing_state.add_router_logits(router_logits)
                self.routing_state.add_lambda_values(_lam)
                self.routing_state.add_balance_loss(self._load_balance_loss(routing_weight))
                self.routing_state.add_usage(routing_weight, self.language_names)

            # Dispatch only selected tokens, so top-k changes actual routed MLP
            # compute rather than merely multiplying already-computed outputs by zero.
            flat_x = x.reshape(-1, x.shape[-1])
            flat_weight = routing_weight.reshape(-1, len(self.language_experts))
            flat_routed = torch.zeros_like(flat_x)
            for idx, expert in enumerate(self.language_experts):
                selected_tokens = flat_weight[:, idx] > 0
                if not bool(selected_tokens.any().item()):
                    continue
                selected_indices = torch.nonzero(selected_tokens, as_tuple=False).squeeze(-1)
                expert_out = self._run_expert(expert, flat_x.index_select(0, selected_indices))
                contribution = expert_out * flat_weight.index_select(0, selected_indices)[:, idx:idx + 1].to(dtype=expert_out.dtype)
                flat_routed = flat_routed.index_add(0, selected_indices, contribution)
            routed = flat_routed.reshape_as(x)

            if not self.shared_experts or self.shared_mix <= 0:
                return routed

            shared = None
            for expert in self.shared_experts:
                expert_out = self._run_expert(expert, x)
                shared = expert_out if shared is None else shared + expert_out
            shared = shared / max(1, len(self.shared_experts))
            return routed * (1.0 - self.shared_mix) + shared * self.shared_mix


    def _get_nested_attr(root: nn.Module, path: str):
        obj = root
        for part in path.split("."):
            if not part:
                continue
            obj = getattr(obj, part)
        return obj


    def _is_decoder_layer_list(module: nn.Module) -> bool:
        if not isinstance(module, nn.ModuleList) or len(module) == 0:
            return False
        first = module[0]
        mlp = getattr(first, "mlp", None)
        # The same decoder list must remain discoverable after layer 0 itself
        # has been replaced by MoE.  A dense MLP exposes gate/up projections
        # directly; LanguageGroupedMoEMLP stores those inside copied experts.
        if isinstance(mlp, LanguageGroupedMoEMLP):
            return True
        return mlp is not None and any(isinstance(getattr(mlp, attr, None), nn.Linear) for attr in ["gate_proj", "up_proj", "fc1", "dense_h_to_4h"])


    def _find_decoder_layers(base_model_obj: nn.Module) -> Tuple[nn.ModuleList, str]:
        candidate_paths = [
            "model.layers",
            "language_model.model.layers",
            "language_model.layers",
            "llm.model.layers",
            "text_model.layers",
            "base_model.model.layers",
        ]
        for path in candidate_paths:
            try:
                candidate = _get_nested_attr(base_model_obj, path)
            except AttributeError:
                continue
            if _is_decoder_layer_list(candidate):
                return candidate, path

        for name, module in base_model_obj.named_modules():
            if _is_decoder_layer_list(module):
                return module, name
        raise ValueError("Could not find decoder layers with .mlp modules for MLP-MoE insertion.")


    def _select_moe_layer_indices(num_layers: int, cfg: Dict[str, Any]) -> List[int]:
        requested = [int(idx) for idx in cfg.get("moe_mlp_layer_indices", [])]
        if requested:
            invalid = [idx for idx in requested if idx < 0 or idx >= num_layers]
            if invalid:
                raise ValueError(
                    f"MoE layer indices {invalid} are invalid for a decoder with {num_layers} layers. "
                    "Refusing to silently run a different architecture."
                )
            if len(requested) != len(set(requested)):
                raise ValueError(f"Duplicate MoE layer indices: {requested}")
            return sorted(requested)
        fraction = min(max(float(cfg.get("moe_mlp_layer_fraction", 1.0 / 3.0)), 0.0), 1.0)
        spacing = max(1, int(cfg.get("moe_mlp_layer_spacing", 3)))
        start = int(math.ceil(num_layers * (1.0 - fraction)))
        start = min(max(0, start), max(0, num_layers - 1))
        selected = list(range(start, num_layers, spacing))
        return selected or [num_layers - 1]


    def apply_moe_mlp_adapters(base_model_obj: nn.Module, cfg: Dict[str, Any]) -> Tuple[MoERoutingState, int]:
        routing_state = MoERoutingState(
            load_balance_coef=cfg.get("moe_mlp_load_balance_coef", 0.0),
            router_ce_coef=cfg.get("moe_mlp_router_ce_coef", 0.0),
            lam_sparse_coef=cfg.get("moe_mlp_lam_sparse_coef", 0.0),
        )
        if not cfg.get("use_moe_mlp", True):
            LOGGER.info("MLP-MoE disabled; training dense base model parameters according to requires_grad flags.")
            return routing_state, 0

        if cfg.get("moe_mlp_freeze_base", True):
            for param in base_model_obj.parameters():
                param.requires_grad = False

        layers, path = _find_decoder_layers(base_model_obj)
        selected = _select_moe_layer_indices(len(layers), cfg)
        routing_state.expected_router_layers = len(selected)
        language_experts = int(cfg.get("moe_mlp_language_experts", 4))
        if language_experts < 1:
            raise ValueError("At least one routed expert is required when MoE is enabled.")
        replaced = 0
        for layer_idx in selected:
            layer = layers[layer_idx]
            if isinstance(getattr(layer, "mlp", None), LanguageGroupedMoEMLP):
                continue
            layer.mlp = LanguageGroupedMoEMLP(
                base_mlp=layer.mlp,
                routing_state=routing_state,
                language_experts=language_experts,
                shared_experts=int(cfg.get("moe_mlp_shared_experts", 2)),
                routing_fn=str(cfg.get("moe_mlp_routing_fn", "sparsegen")).lower(),
                router_temperature=float(cfg.get("moe_mlp_router_temperature", 1.0)),
                sparsegen_hidden=int(cfg.get("moe_mlp_sparsegen_hidden", 128)),
                sparsegen_eps=float(cfg.get("moe_mlp_sparsegen_eps", 0.01)),
                shared_mix=float(cfg.get("moe_mlp_shared_mix", 0.25)),
                routed_top_k=int(cfg.get("moe_mlp_routed_top_k", cfg.get("moe_mlp_language_experts", len(LANGUAGE_GROUPS)))),
                enforce_top_k=bool(cfg.get("moe_mlp_enforce_top_k", False)),
                expert_init=str(cfg.get("moe_mlp_expert_init", "copy")),
                expert_train_dtype=str(cfg.get("moe_mlp_expert_train_dtype", "float32")),
            )
            replaced += 1
        if replaced != len(selected):
            raise RuntimeError(
                f"MoE construction replaced {replaced}/{len(selected)} requested MLP layers: {selected}."
            )
        cfg["moe_mlp_resolved_layer_indices"] = list(selected)

        LOGGER.info(
            "Applied source-agnostic MLP-MoE to %d decoder layers at %s indices=%s | routed_experts=%d shared=%d routing=%s hard_top_k=%s router_ce=%.4g balance=%.4g",
            replaced, path, selected, language_experts, int(cfg.get("moe_mlp_shared_experts", 2)),
            str(cfg.get("moe_mlp_routing_fn", "sparsegen")), bool(cfg.get("moe_mlp_enforce_top_k", False)), routing_state.router_ce_coef, routing_state.load_balance_coef,
        )
        print(
            "MLP-MoE:",
            f"layers={selected}",
            f"routed_experts={language_experts}",
            f"shared={int(cfg.get('moe_mlp_shared_experts', 2))}",
            f"routing={str(cfg.get('moe_mlp_routing_fn', 'sparsegen'))}",
            f"hard_top_k={bool(cfg.get('moe_mlp_enforce_top_k', False))}",
            f"routed_top_k={int(cfg.get('moe_mlp_routed_top_k', 0))}",
            f"shared_mix={float(cfg.get('moe_mlp_shared_mix', 0.25))}",
            f"expert_train_dtype={cfg.get('moe_mlp_expert_train_dtype', 'float32')}",
        )
        return routing_state, replaced


    def moe_mlp_state_dict(model_obj: nn.Module) -> Dict[str, torch.Tensor]:
        keep_markers = (
            ".language_experts.", ".shared_experts.", ".router.", ".lambda_mlp.",
        )
        return {
            name: tensor.detach().cpu()
            for name, tensor in model_obj.state_dict().items()
            if any(marker in name for marker in keep_markers)
        }


    def load_moe_mlp_state(model_obj: nn.Module, checkpoint_dir: Optional[Path]) -> None:
        if checkpoint_dir is None:
            return
        state_path = Path(checkpoint_dir) / MOE_MLP_STATE_FILE
        if not state_path.exists():
            return
        try:
            payload = torch.load(state_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(state_path, map_location="cpu")
        state = payload.get("moe_mlp_state_dict", payload) if isinstance(payload, dict) else payload
        missing, unexpected = model_obj.load_state_dict(state, strict=False)
        LOGGER.info("Loaded MLP-MoE state from %s | missing=%d unexpected=%d", state_path, len(missing), len(unexpected))


    class Qwen3VLMForLM(torch.nn.Module):
        """Thin wrapper that adds answer-token LM loss and MLP-MoE router losses."""

        def __init__(self, base_model, routing_state: Optional[MoERoutingState] = None):
            super().__init__()
            self.base_model = base_model
            self.routing_state = routing_state or MoERoutingState()
            self.register_buffer("language_group_class_weights", torch.ones(len(LANGUAGE_GROUPS)), persistent=False)

        def set_language_group_class_weights(self, weights: List[float]) -> None:
            tensor = torch.tensor(weights, dtype=torch.float32, device=self.language_group_class_weights.device)
            if tensor.numel() != self.language_group_class_weights.numel():
                raise ValueError(f"Expected {self.language_group_class_weights.numel()} class weights, got {tensor.numel()}")
            self.language_group_class_weights.copy_(tensor)

        def get_and_reset_expert_usage(self) -> Dict[str, Any]:
            return self.routing_state.get_and_reset_usage()

        def set_lam_sparse_warmup_scale(self, scale: float) -> None:
            self.routing_state.set_lam_sparse_warmup_scale(scale)

        def set_load_balance_warmup_scale(self, scale: float) -> None:
            self.routing_state.set_load_balance_warmup_scale(scale)

        def forward(self, **kwargs) -> Dict[str, Any]:
            labels = kwargs.get("labels", None)
            attention_mask = kwargs.get("attention_mask", None)
            language_group_ids = kwargs.pop("language_group_ids", None)
            source_loss_weights = kwargs.pop("source_loss_weights", None)
            page_htr_rows = kwargs.pop("page_htr_mask", None)
            ul_active_rows = kwargs.pop("token_ul_active_mask", None)
            self.routing_state.reset_forward()
            self.routing_state.set_token_mask(attention_mask)
            outputs = self.base_model(**kwargs)
            logits = getattr(outputs, "logits", None)
            if logits is None:
                lm_loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=next(self.parameters()).device)
                if isinstance(labels, torch.Tensor) and labels.ndim > 0:
                    lm_token_count = labels[..., 1:].ne(IGNORE_INDEX).sum().to(device=lm_loss.device, dtype=torch.float32)
                    sample_lm_losses = torch.zeros((labels.shape[0],), device=lm_loss.device)
                else:
                    lm_token_count = torch.ones((), device=lm_loss.device, dtype=torch.float32)
                    sample_lm_losses = torch.zeros((0,), device=lm_loss.device)
                lm_normalizer = lm_token_count.clamp_min(1.0)
                lm_loss_sum = lm_loss * lm_token_count.clamp_min(1.0)
                ce_loss_sum = lm_loss_sum
                ul_loss = lm_loss * 0.0
                ul_scaled_loss = lm_loss * 0.0
                ul_candidate_count = torch.zeros((), device=lm_loss.device)
            elif labels is None:
                lm_loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=logits.device)
                lm_token_count = torch.ones((), device=logits.device, dtype=torch.float32)
                lm_loss_sum = lm_loss * lm_token_count
                lm_normalizer = lm_token_count
                ce_loss_sum = lm_loss_sum
                ul_loss = lm_loss * 0.0
                ul_scaled_loss = lm_loss * 0.0
                ul_candidate_count = torch.zeros((), device=lm_loss.device)
                sample_lm_losses = torch.zeros((0,), device=logits.device)
            else:
                collapse_objective = collapse_aware_lm_objective(
                    logits, labels, page_htr_rows=page_htr_rows, ul_active_rows=ul_active_rows
                )
                lm_loss_sum = collapse_objective["objective_sum"]
                lm_normalizer = collapse_objective["normalizer"]
                lm_token_count = collapse_objective["actual_token_count"]
                ce_loss_sum = collapse_objective["ce_sum"]
                ul_loss = collapse_objective["ul_mean"]
                ul_scaled_loss = collapse_objective["ul_scaled_mean"]
                ul_candidate_count = collapse_objective["ul_candidate_count"]
                source_weighting_scheme = str(CFG.get("lm_source_loss_weighting", "none")).lower()
                needs_source_weighting = source_weighting_scheme not in {"none", "off", "0"}
                needs_source_logging = bool(CFG.get("log_train_loss_by_source", False))
                if needs_source_weighting or needs_source_logging:
                    sample_lm_losses, valid_lm_samples = per_sample_masked_lm_loss(logits, labels)
                else:
                    sample_lm_losses = torch.zeros((labels.shape[0],), device=logits.device)
                    valid_lm_samples = torch.zeros((labels.shape[0],), dtype=torch.bool, device=logits.device)

                if needs_source_weighting:
                    lm_loss = weighted_sample_lm_loss(
                        sample_lm_losses,
                        valid_lm_samples,
                        source_loss_weights,
                    )
                    lm_loss_sum = lm_loss * lm_normalizer.clamp_min(1.0)
                else:
                    # Exact objective-weighted mean; source logging does not alter it.
                    lm_loss = lm_loss_sum / lm_normalizer.clamp_min(1.0)
            if self.training and CFG.get("use_moe_mlp", False):
                observed_router_layers = len(self.routing_state._router_logits)
                expected_router_layers = int(self.routing_state.expected_router_layers)
                if observed_router_layers != expected_router_layers:
                    raise RuntimeError(
                        f"MoE forward collected router outputs from {observed_router_layers}/"
                        f"{expected_router_layers} layers. Check decoder insertion and checkpointing."
                    )
            balance_raw, balance_loss, balance_effective_coef = self.routing_state.balance_loss(lm_loss.device)
            router_loss, router_acc, router_entropy = self.routing_state.router_language_loss(
                labels=labels,
                language_group_ids=language_group_ids,
                class_weights=self.language_group_class_weights,
                device=lm_loss.device,
            )
            lam_sparse_raw, lam_sparse_loss, lam_sparse_effective_coef = self.routing_state.lam_sparse_loss(
                lm_loss.device
            )
            if self.training and CFG.get("use_moe_mlp", False):
                if self.routing_state.load_balance_coef > 0 and balance_effective_coef.item() > 0 and not balance_loss.requires_grad:
                    raise RuntimeError(
                        "MoE balance loss is detached. Use non-reentrant gradient checkpointing."
                    )
                if self.routing_state.router_ce_coef > 0 and not router_loss.requires_grad:
                    raise RuntimeError(
                        "MoE router CE loss is detached. Use non-reentrant gradient checkpointing."
                    )
                if self.routing_state.lam_sparse_coef > 0 and lam_sparse_effective_coef.item() > 0 and not lam_sparse_loss.requires_grad:
                    raise RuntimeError("LamSparseLoss is detached. Check Sparsegen lambda collection/checkpointing.")
            loss = lm_loss + balance_loss + router_loss + lam_sparse_loss
            objective_loss_sum = lm_loss_sum + (balance_loss + router_loss + lam_sparse_loss) * lm_normalizer.clamp_min(1.0)
            return {
                "loss": loss,
                "loss_sum": objective_loss_sum,
                "lm_loss": lm_loss.detach(),
                "lm_loss_sum": lm_loss_sum.detach(),
                "lm_token_count": lm_token_count.detach(),
                "lm_loss_normalizer": lm_normalizer.detach(),
                "ce_loss_sum": ce_loss_sum.detach(),
                "token_ul_loss": ul_loss.detach(),
                "token_ul_scaled_loss": ul_scaled_loss.detach(),
                "token_ul_candidate_count": ul_candidate_count.detach(),
                "moe_load_balance_raw_loss": balance_raw.detach(),
                "moe_load_balance_loss": balance_loss.detach(),
                "moe_load_balance_effective_coef": balance_effective_coef.detach(),
                "moe_router_language_loss": router_loss.detach(),
                "moe_lam_sparse_raw_loss": lam_sparse_raw.detach(),
                "moe_lam_sparse_loss": lam_sparse_loss.detach(),
                "moe_lam_sparse_effective_coef": lam_sparse_effective_coef.detach(),
                "moe_router_language_accuracy": router_acc.detach(),
                "moe_router_entropy": router_entropy.detach(),
                "sample_lm_losses": sample_lm_losses.detach(),
                "logits": logits,
            }

        def generate(self, **kwargs):
            return self.base_model.generate(**kwargs)

    return export_runtime_locals(state, locals())
