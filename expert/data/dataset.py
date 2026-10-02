"""data.dataset phase for the ExpertHTR research workflow.

This module preserves the audited runtime computation. It is invoked by the
explicit CLI workflow and exports its declared runtime objects into a
:class:`~expert.runtime.context.RuntimeState`.
"""

from __future__ import annotations

from expert.runtime.context import RuntimeState, export_runtime_locals


def register_dataset_types(state: RuntimeState) -> RuntimeState:
    """Execute this phase after its prerequisites have populated ``state``."""

    globals().update(state.values)
    # ─────────────────────────────────────────────────────────────────────────────
    #  OCRJsonlDataset + QwenVLMCollator
    # ─────────────────────────────────────────────────────────────────────────────

    def _normalize_user_prompt(content: Any) -> str:
        if isinstance(content, str):
            lines = [line.strip() for line in content.replace("<image>", "").splitlines()]
            return " ".join(line for line in lines if line).strip()
        if isinstance(content, list):
            chunks = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = str(item.get("text", "")).strip()
                    if text:
                        chunks.append(text)
            return " ".join(chunks).strip()
        return ""


    def _extract_prompt_and_target(
        sample: Dict[str, Any],
        prompt_override: Optional[str],
    ) -> Tuple[str, str]:
        prompt = prompt_override.strip() if prompt_override else ""
        target = ""
        for message in sample.get("messages", []):
            role = message.get("role")
            if role == "user" and not prompt:
                prompt = _normalize_user_prompt(message.get("content"))
            elif role == "assistant" and not target:
                content = message.get("content", "")
                if isinstance(content, str):
                    target = content.strip()
                elif isinstance(content, list):
                    parts = [
                        str(item.get("text", "")).strip()
                        for item in content
                        if isinstance(item, dict) and item.get("type") == "text"
                    ]
                    target = " ".join(part for part in parts if part).strip()
        if not prompt:
            prompt = DEFAULT_OCR_PROMPT
        if not target:
            target = str(sample.get("text_raw", sample.get("text", ""))).strip()
        return prompt, target


    def _token_id_list(value: Any) -> List[int]:
        if value is None:
            return []
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().view(-1).tolist()
        if isinstance(value, (list, tuple, set)):
            return [int(item) for item in value]
        return [int(value)]


    def resolve_generation_stop_ids(tokenizer, model_obj=None) -> List[int]:
        """Return the audited terminal IDs shared by SFT labels and generation."""
        stop_ids = set(_token_id_list(getattr(tokenizer, "eos_token_id", None)))
        runtime = getattr(model_obj, "module", model_obj) if model_obj is not None else None
        runtime = getattr(runtime, "base_model", runtime) if runtime is not None else None
        for owner in (
            getattr(runtime, "config", None),
            getattr(runtime, "generation_config", None),
        ):
            stop_ids.update(_token_id_list(getattr(owner, "eos_token_id", None)))
        all_special_ids = {int(value) for value in getattr(tokenizer, "all_special_ids", [])}
        unk_id = getattr(tokenizer, "unk_token_id", None)
        for token_text in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>"):
            try:
                token_id = int(tokenizer.convert_tokens_to_ids(token_text))
            except Exception:
                continue
            if (unk_id is None or token_id != int(unk_id)) and token_id in all_special_ids:
                stop_ids.add(token_id)
        if not stop_ids:
            raise RuntimeError("Tokenizer/model expose no valid generation stop token IDs.")
        return sorted(stop_ids)


    def _stable_unit_interval(text: str, seed: int) -> float:
        digest = hashlib.sha256(f"{int(seed)}::{text}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") / float(1 << 64)


    def _rolling_ngram_loop_boundary(
        token_ids: List[int],
        n: Optional[int] = None,
        window: Optional[int] = None,
        threshold: Optional[float] = None,
        min_tokens: Optional[int] = None,
        consecutive: Optional[int] = None,
    ) -> Optional[int]:
        """Return a conservative start boundary for a highly repetitive suffix."""
        n = int(n if n is not None else CFG["loop_ngram"])
        window = int(window if window is not None else CFG["loop_window"])
        threshold = float(threshold if threshold is not None else CFG["loop_threshold"])
        min_tokens = int(min_tokens if min_tokens is not None else CFG["loop_min_tokens"])
        consecutive = int(consecutive if consecutive is not None else CFG["loop_consecutive"])
        ids = [int(value) for value in token_ids]
        if len(ids) < max(min_tokens, window, n):
            return None
        streak = 0
        for end in range(max(min_tokens, window), len(ids) + 1):
            values = ids[end - window:end]
            ngrams = [tuple(values[index:index + n]) for index in range(len(values) - n + 1)]
            repeat_fraction = 1.0 - len(set(ngrams)) / max(1, len(ngrams))
            streak = streak + 1 if repeat_fraction >= threshold else 0
            if streak >= consecutive:
                return max(0, end - window - consecutive + 1)
        return None


    class OCRJsonlDataset(Dataset):
        """
        JSONL OCR dataset. Each line: {"id", "images": ["..."], "messages": [...], "text_raw": "..."}
        Images are relative to `image_root`.
        """
        def __init__(
            self, jsonl_path: Path, image_root: Path,
            prompt_text: Optional[str] = None,
            augment: bool = False,
        ):
            self.jsonl_path = Path(jsonl_path)
            self.image_root = Path(image_root)
            self.prompt_text = prompt_text
            self.augment = augment
            self.samples = self._load_jsonl()
            self._aug = None
            if self.augment:
                raise NotImplementedError(
                    "HWVLM_USE_AUGMENTATION=1 but no augmentation transform is implemented. "
                    "Keep it disabled or add an audited transform explicitly."
                )

        def _load_jsonl(self) -> List[Dict[str, Any]]:
            samples = []
            with open(self.jsonl_path, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    if not line.strip():
                        continue
                    try:
                        sample = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Malformed JSONL row: {self.jsonl_path}:{line_no}: {exc}"
                        ) from exc
                    task_name = str((sample.get("metadata") or {}).get("task", ""))
                    if task_name not in {"page_htr", "line_count", "phrase_line_lookup", "line_range_htr"}:
                        raise RuntimeError(
                            "Local dataset has an unsupported task; first id="
                            + repr(sample.get("id", "unknown")) + ", task=" + repr(task_name)
                        )
                    samples.append(sample)
            if not samples:
                raise ValueError(f"No records loaded from {self.jsonl_path}")
            return samples

        def _resize_keep_ratio(self, image: Image.Image) -> Tuple[Image.Image, float, float]:
            """Qwen-style smart resize by area, preserving image aspect ratio."""
            orig_w, orig_h = image.size
            if orig_w <= 0 or orig_h <= 0:
                raise ValueError(f"Invalid image dimensions: {(orig_w, orig_h)}")
            factor = int(CFG.get("image_resize_factor", 0))
            if factor <= 0:
                raise RuntimeError(
                    "image_resize_factor is unset; load the processor before building datasets."
                )
            aspect_ratio = max(orig_w, orig_h) / min(orig_w, orig_h)
            if aspect_ratio > float(CFG["image_max_aspect_ratio"]):
                raise ValueError(
                    f"Image aspect ratio {aspect_ratio:.3f} exceeds the configured maximum "
                    f"{CFG['image_max_aspect_ratio']}: size={(orig_w, orig_h)}"
                )

            def round_factor(value: float) -> int:
                return max(factor, int(round(value / factor)) * factor)

            def floor_factor(value: float) -> int:
                return max(factor, int(math.floor(value / factor)) * factor)

            def ceil_factor(value: float) -> int:
                return max(factor, int(math.ceil(value / factor)) * factor)

            new_w = round_factor(orig_w)
            new_h = round_factor(orig_h)
            min_pixels = int(CFG["image_min_pixels"])
            max_pixels = int(CFG["image_max_pixels"])
            if new_w * new_h > max_pixels:
                beta = math.sqrt((orig_w * orig_h) / max_pixels)
                new_w = floor_factor(orig_w / beta)
                new_h = floor_factor(orig_h / beta)
            elif new_w * new_h < min_pixels:
                beta = math.sqrt(min_pixels / (orig_w * orig_h))
                new_w = ceil_factor(orig_w * beta)
                new_h = ceil_factor(orig_h * beta)

            resized_pixels = new_w * new_h
            if not (min_pixels <= resized_pixels <= max_pixels):
                raise RuntimeError(
                    f"Area/ratio resize escaped its pixel budget: original={(orig_w, orig_h)} "
                    f"resized={(new_w, new_h)} pixels={resized_pixels} "
                    f"budget={(min_pixels, max_pixels)} factor={factor}"
                )
            if (new_w, new_h) == (orig_w, orig_h):
                return image, 1.0, 1.0
            resampling = getattr(Image, "Resampling", Image).BICUBIC
            resized = image.resize((new_w, new_h), resampling)
            return resized, new_w / orig_w, new_h / orig_h

        def _load_image(self, sample: Dict[str, Any]) -> Tuple[Image.Image, float, float, Tuple[int, int], Tuple[int, int]]:
            image_paths = sample.get("images", [])
            if not image_paths:
                image_paths = [f"{sample.get('id', '')}.png"]
            image_path = self.image_root / image_paths[0]
            try:
                with Image.open(image_path) as opened:
                    image = opened.convert("RGB")
            except Exception as exc:
                raise RuntimeError(
                    f"Image decode failed for id={sample.get('id')!r}, path={image_path}: {exc}"
                ) from exc
            original_size = image.size
            image, scale_x, scale_y = self._resize_keep_ratio(image)
            resized_size = image.size
            return image, scale_x, scale_y, original_size, resized_size

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            sample = self.samples[idx]
            prompt, target = _extract_prompt_and_target(sample, self.prompt_text)
            image, scale_x, scale_y, original_size, resized_size = self._load_image(sample)
            meta = sample.get("metadata", {}) or {}
            language_group_id, language_group = language_group_from_metadata(meta)
            image_relpath = str((sample.get("images") or [f"{sample.get('id', '')}.png"])[0])
            source_dataset = str(meta.get("source_dataset", "unknown"))
            task = str(meta.get("task", "unknown"))
            record_id = str(sample.get("id", str(idx)))
            token_ul_active = (
                self.jsonl_path.stem.lower() == "train"
                and
                task == "page_htr"
                and CFG.get("token_ul_coef", 0.0) > 0.0
                and _stable_unit_interval(record_id, CFG["seed"]) < float(CFG["token_ul_page_fraction"])
            )
            return {
                "id": record_id,
                "image": image,
                "prompt": prompt,
                "target": target,
                "metadata": meta,
                "source_dataset": source_dataset,
                "page_key": f"{source_dataset}::{image_relpath}",
                "task": task,
                "token_ul_active": bool(token_ul_active),
                "language": str(meta.get("language", "unknown")),
                "language_group_id": int(language_group_id),
                "language_group": language_group,
                "source_loss_weight": float(sample.get("_lm_source_loss_weight", 1.0)),
                "resize_scale": (scale_x, scale_y),
                "original_size": original_size,
                "resized_size": resized_size,
            }


    class QwenVLMCollator:
        """
        Multimodal collator: builds messages, applies chat template,
        tokenizes with processor, masks prompt tokens -> -100.
        """
        def __init__(self, processor, max_text_length: int = 512):
            self.processor = processor
            self.max_text_length = max_text_length
            tokenizer = getattr(processor, "tokenizer", None)
            if tokenizer is None:
                raise RuntimeError("QwenVLMCollator requires processor.tokenizer.")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            terminal_ids = _token_id_list(getattr(tokenizer, "eos_token_id", None))
            if len(terminal_ids) != 1 or not getattr(tokenizer, "eos_token", None):
                raise RuntimeError(
                    f"Training requires one tokenizer EOS token, got ids={terminal_ids} "
                    f"token={getattr(tokenizer, 'eos_token', None)!r}."
                )
            self.training_terminal_id = int(terminal_ids[0])
            self.training_terminal_token = str(tokenizer.eos_token)

        def _build_messages(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
            return [
                {"role": "user", "content": [
                    {"type": "image", "image": sample["image"]},
                    {"type": "text",  "text":  sample["prompt"]},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": sample["target"]}]},
            ]

        def _build_prompt_messages(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
            return [
                {"role": "user", "content": [
                    {"type": "image", "image": sample["image"]},
                    {"type": "text",  "text":  sample["prompt"]},
                ]},
            ]

        def _apply_template(self, messages: List[Dict[str, Any]], add_generation_prompt: bool) -> str:
            return self.processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

        def _ensure_training_terminal(self, text: str) -> str:
            encoded = self.processor.tokenizer(
                text, add_special_tokens=False, return_attention_mask=False
            ).get("input_ids", [])
            if encoded and isinstance(encoded[0], list):
                encoded = encoded[0]
            if encoded and int(encoded[-1]) == self.training_terminal_id:
                return text
            return text + self.training_terminal_token

        def _full_text(self, sample: Dict[str, Any]) -> str:
            return self._ensure_training_terminal(
                self._apply_template(self._build_messages(sample), False)
            )

        def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
            full_messages = [self._build_messages(s) for s in batch]
            prompt_messages = [self._build_prompt_messages(s) for s in batch]

            full_texts = [self._ensure_training_terminal(self._apply_template(m, False)) for m in full_messages]
            prompt_texts = [self._apply_template(m, True) for m in prompt_messages]
            batch_images = [[s["image"]] for s in batch]

            full_inputs = self.processor(
                text=full_texts, images=batch_images, do_resize=False,
                padding=True, truncation=True,
                max_length=self.max_text_length, return_tensors="pt",
            )
            prompt_inputs = self.processor(
                text=prompt_texts, images=batch_images, do_resize=False,
                padding=True, truncation=True,
                max_length=self.max_text_length, return_tensors="pt",
            )
            full_grid = full_inputs.get("image_grid_thw")
            prompt_grid = prompt_inputs.get("image_grid_thw")
            if not isinstance(full_grid, torch.Tensor) or full_grid.ndim != 2 or full_grid.shape[0] != len(batch):
                raise RuntimeError(
                    f"Cannot verify model image geometry: image_grid_thw={getattr(full_grid, 'shape', None)} "
                    f"batch={len(batch)}"
                )
            if not isinstance(prompt_grid, torch.Tensor) or not torch.equal(full_grid, prompt_grid):
                raise RuntimeError("Full/prompt processor passes produced different image grids.")
            patch_size = int(CFG["image_patch_size"])
            for row_idx, sample in enumerate(batch):
                effective_h = int(full_grid[row_idx, 1].item()) * patch_size
                effective_w = int(full_grid[row_idx, 2].item()) * patch_size
                expected_w, expected_h = (int(value) for value in sample["resized_size"])
                if (effective_w, effective_h) != (expected_w, expected_h):
                    raise RuntimeError(
                        f"Processor/model image geometry differs from the resized image for "
                        f"id={sample['id']!r}: model={(effective_w, effective_h)} "
                        f"resized={(expected_w, expected_h)}."
                    )

            labels = full_inputs["input_ids"].clone()
            attention_mask = full_inputs.get("attention_mask", torch.ones_like(labels))
            prompt_attn = prompt_inputs.get("attention_mask", torch.ones_like(prompt_inputs["input_ids"]))
            prompt_lens = prompt_attn.sum(dim=1)
            for row_idx in range(labels.shape[0]):
                full_ids = full_inputs["input_ids"][row_idx][attention_mask[row_idx].bool()]
                prompt_ids = prompt_inputs["input_ids"][row_idx][prompt_attn[row_idx].bool()]
                if full_ids.numel() < prompt_ids.numel() or not torch.equal(
                    full_ids[: prompt_ids.numel()], prompt_ids
                ):
                    raise ValueError(
                        f"Prompt is not an exact prefix of the full chat sequence for id={batch[row_idx]['id']!r}. "
                        "Label masking would be invalid."
                    )
                nonpad_positions = attention_mask[row_idx].nonzero(as_tuple=False).flatten()
                if nonpad_positions.numel() == 0:
                    raise ValueError(f"Empty tokenized sequence for id={batch[row_idx]['id']!r}")
                prompt_start = int(nonpad_positions[0])
                prompt_end = prompt_start + int(prompt_lens[row_idx])
                labels[row_idx, prompt_start:prompt_end] = IGNORE_INDEX
            labels[attention_mask == 0] = IGNORE_INDEX
            for row_idx in range(labels.shape[0]):
                if not torch.any(labels[row_idx] != IGNORE_INDEX):
                    raise ValueError(
                        f"Zero retained answer tokens after truncation for id={batch[row_idx]['id']!r}; "
                        f"increase max_text_length or repair the prompt/target."
                    )
                supervised = labels[row_idx][labels[row_idx] != IGNORE_INDEX]
                if int(supervised[-1].item()) != self.training_terminal_id:
                    raise ValueError(
                        f"Terminal EOS was truncated or not serialized for id={batch[row_idx]['id']!r}: "
                        f"last_label={int(supervised[-1].item())} expected={self.training_terminal_id}."
                    )

            result = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in full_inputs.items()}
            result["labels"] = labels
            result["prompt_lens"] = prompt_lens
            result["texts"] = [s["target"] for s in batch]
            result["prompts"] = [s["prompt"] for s in batch]
            result["ids"] = [s["id"] for s in batch]
            result["metadata"] = [s["metadata"] for s in batch]
            result["source_datasets"] = [s["source_dataset"] for s in batch]
            result["page_keys"] = [s["page_key"] for s in batch]
            result["tasks"] = [s["task"] for s in batch]
            result["page_htr_mask"] = torch.tensor([s["task"] == "page_htr" for s in batch], dtype=torch.bool)
            result["token_ul_active_mask"] = torch.tensor([s.get("token_ul_active", False) for s in batch], dtype=torch.bool)
            result["languages"] = [s["language"] for s in batch]
            result["language_groups"] = [s["language_group"] for s in batch]
            result["language_group_ids"] = torch.tensor([s["language_group_id"] for s in batch], dtype=torch.long)
            result["source_loss_weights"] = torch.tensor([s.get("source_loss_weight", 1.0) for s in batch], dtype=torch.float32)
            result["resize_scales"] = [s["resize_scale"] for s in batch]
            result["original_sizes"] = [s["original_size"] for s in batch]
            result["resized_sizes"] = [s["resized_size"] for s in batch]
            for key, val in prompt_inputs.items():
                result[f"prompt_{key}"] = val.clone() if isinstance(val, torch.Tensor) else val
            return result

    return export_runtime_locals(state, locals())
