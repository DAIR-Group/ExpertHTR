"""data.loaders phase for the ExpertHTR research workflow.

This module preserves the audited runtime computation. It is invoked by the
explicit CLI workflow and exports its declared runtime objects into a
:class:`~expert.runtime.context.RuntimeState`.
"""

from __future__ import annotations

from expert.runtime.context import RuntimeState, export_runtime_locals


def build_loaders(state: RuntimeState) -> RuntimeState:
    """Execute this phase after its prerequisites have populated ``state``."""

    globals().update(state.values)
    # ─────────────────────────────────────────────────────────────────────────────
    #  8c. Build datasets
    # ─────────────────────────────────────────────────────────────────────────────

    def resolve_split_path(data_root: Path, dataset_name: str, split: str) -> Optional[Path]:
        alias = DATASET_SUBDIR_ALIASES.get(dataset_name, dataset_name)
        dataset_dir = data_root if data_root.name == alias else data_root / alias
        candidate = dataset_dir / f"{split}.jsonl"
        return candidate if candidate.exists() else None


    def _read_first_sample(jsonl_path: Path) -> Dict[str, Any]:
        with open(jsonl_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    return json.loads(line)
        raise ValueError(f"JSONL file is empty: {jsonl_path}")


    def discover_image_root(train_jsonl_path: Path) -> Path:
        sample = _read_first_sample(train_jsonl_path)
        image_relpath = Path(sample.get("images", [f"{sample.get('id', '')}.png"])[0])

        explicit = os.getenv("HWVLM_IMAGE_ROOT", "").strip()
        candidates: List[Path] = []
        if explicit:
            candidates.append(Path(explicit))

        candidates.extend([DATA_ROOT, DATA_ROOT.parent])

        if KAGGLE_INPUT.exists():
            for sub in sorted(KAGGLE_INPUT.iterdir()):
                if not sub.is_dir():
                    continue
                candidates.extend([
                    sub,
                    sub / "data",
                    sub / "images",
                    sub / "data" / "images",
                ])

        seen = set()
        ordered_candidates = []
        for candidate in candidates:
            key = str(candidate)
            if key not in seen:
                seen.add(key)
                ordered_candidates.append(candidate)

        for candidate in ordered_candidates:
            if (candidate / image_relpath).exists():
                return candidate
            if candidate.name == "images" and (candidate.parent / image_relpath).exists():
                return candidate.parent

        raise FileNotFoundError(
            f"Could not resolve image root for relative path '{image_relpath}'. "
            f"Set HWVLM_IMAGE_ROOT explicitly if needed."
        )


    def _dataset_samples_for_stats(dataset) -> List[Dict[str, Any]]:
        if hasattr(dataset, "samples"):
            return list(dataset.samples)
        if isinstance(dataset, Subset) and hasattr(dataset.dataset, "samples"):
            return [dataset.dataset.samples[idx] for idx in dataset.indices]
        return []


    def _count_language_groups_from_samples(samples: List[Dict[str, Any]]) -> Dict[str, int]:
        counts = {name: 0 for name in LANGUAGE_GROUPS}
        for sample in samples:
            group_id, group_name = language_group_from_metadata(sample.get("metadata", {}) or {})
            counts[group_name] = counts.get(group_name, 0) + 1
        return counts


    SAMPLER_GENERATOR = torch.Generator().manual_seed(int(CFG["seed"]))
    LOADER_GENERATOR = torch.Generator().manual_seed(int(CFG["seed"]) + 1)


    def _build_group_balanced_sampler(dataset, alpha: float) -> Optional[WeightedRandomSampler]:
        samples = _dataset_samples_for_stats(dataset)
        if not samples:
            return None
        alpha = min(max(float(alpha), 0.0), 1.0)
        group_ids = [language_group_from_metadata(sample.get("metadata", {}) or {})[0] for sample in samples]
        counts = Counter(group_ids)
        weights = []
        for group_id in group_ids:
            count = max(1, counts[group_id])
            # p_group proportional to count ** alpha, so each sample has weight count ** (alpha - 1).
            weights.append(float(count ** (alpha - 1.0)))
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
            generator=SAMPLER_GENERATOR,
        )


    def _source_name_from_sample(sample: Dict[str, Any]) -> str:
        meta = sample.get("metadata", {}) or {}
        return str(meta.get("source_dataset") or meta.get("dataset") or "unknown").strip() or "unknown"


    def _count_sources_from_samples(samples: List[Dict[str, Any]]) -> Dict[str, int]:
        counts = Counter(_source_name_from_sample(sample) for sample in samples)
        return dict(sorted(counts.items(), key=lambda kv: kv[0]))


    def _build_source_balanced_sampler(dataset, alpha: float) -> Optional[WeightedRandomSampler]:
        samples = _dataset_samples_for_stats(dataset)
        if not samples:
            return None
        alpha = min(max(float(alpha), 0.0), 1.0)
        source_names = [_source_name_from_sample(sample) for sample in samples]
        counts = Counter(source_names)
        weights = []
        for source in source_names:
            count = max(1, counts[source])
            # p_source proportional to count ** alpha, so each sample has weight count ** (alpha - 1).
            weights.append(float(count ** (alpha - 1.0)))
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
            generator=SAMPLER_GENERATOR,
        )


    def configure_source_loss_weights(dataset, cfg: Dict[str, Any]) -> None:
        samples = _dataset_samples_for_stats(dataset)
        if not samples:
            cfg["source_counts"] = {}
            cfg["lm_source_loss_weights"] = {}
            return
        counts = Counter(_source_name_from_sample(sample) for sample in samples)
        scheme = str(cfg.get("lm_source_loss_weighting", "inverse_sqrt")).lower()
        if scheme in {"none", "off", "0"}:
            raw_weights = {source: 1.0 for source in counts}
        elif scheme in {"inverse", "inv"}:
            raw_weights = {source: 1.0 / max(1, count) for source, count in counts.items()}
        else:
            raw_weights = {source: 1.0 / math.sqrt(max(1, count)) for source, count in counts.items()}
        sample_mean = sum(raw_weights[_source_name_from_sample(sample)] for sample in samples) / max(1, len(samples))
        weights = {source: weight / max(sample_mean, 1e-12) for source, weight in raw_weights.items()}
        for sample in samples:
            sample["_lm_source_loss_weight"] = float(weights[_source_name_from_sample(sample)])
        cfg["source_counts"] = dict(sorted(counts.items(), key=lambda kv: kv[0]))
        cfg["lm_source_loss_weights"] = dict(sorted(weights.items(), key=lambda kv: kv[0]))
        print("Source counts:", cfg["source_counts"])
        print(f"LM source loss weights ({scheme}):", cfg["lm_source_loss_weights"])


    def configure_language_group_class_weights(model_obj, dataset, cfg: Dict[str, Any]) -> None:
        # Final ablations deliberately use source-agnostic routing and no router CE.
        # Keep the model buffer well-defined, but do not manufacture/report a
        # meaningless one-class weighting scheme.
        if float(cfg.get("moe_mlp_router_ce_coef", 0.0)) <= 0.0:
            cfg["language_group_counts"] = {}
            cfg["language_group_class_weights"] = {}
            print("Router CE disabled: source metadata is evaluation-only, not a routing target.")
            return

        samples = _dataset_samples_for_stats(dataset)
        counts_by_name = _count_language_groups_from_samples(samples)
        counts = [max(1, int(counts_by_name.get(name, 0))) for name in LANGUAGE_GROUPS]
        scheme = str(cfg.get("router_ce_class_weighting", "inverse_sqrt")).lower()
        if scheme in {"none", "off", "0"}:
            weights = [1.0 for _ in counts]
        elif scheme in {"inverse", "inv"}:
            weights = [1.0 / c for c in counts]
        else:
            weights = [1.0 / math.sqrt(c) for c in counts]
        mean_weight = sum(weights) / max(1, len(weights))
        weights = [w / mean_weight for w in weights]
        model_for_cfg = unwrap_model(model_obj)
        if hasattr(model_for_cfg, "set_language_group_class_weights"):
            model_for_cfg.set_language_group_class_weights(weights)
        cfg["language_group_counts"] = dict(counts_by_name)
        cfg["language_group_class_weights"] = dict(zip(LANGUAGE_GROUPS, weights))
        print("Language group counts:", cfg["language_group_counts"])
        print("Router CE class weights:", cfg["language_group_class_weights"])


    def _make_loader(
        dataset,
        processor,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
    ):
        collator = QwenVLMCollator(processor=processor, max_text_length=CFG["max_text_length"])
        sampler = None
        effective_shuffle = shuffle
        sampling_mode = str(CFG.get("sampling_mode", "full_pass")).lower()
        if shuffle and sampling_mode == "full_pass":
            # RandomSampler without replacement: every record appears exactly once.
            effective_shuffle = True
            print("Using deterministic full-pass shuffle without replacement")
        elif shuffle and sampling_mode == "source_balanced":
            sampler = _build_source_balanced_sampler(dataset, CFG.get("source_sampling_alpha", 1.0))
            if sampler is not None:
                effective_shuffle = False
                print(f"Using source-balanced sampler with alpha={CFG.get('source_sampling_alpha', 1.0)}")
        elif shuffle and sampling_mode == "group_balanced":
            sampler = _build_group_balanced_sampler(dataset, CFG.get("group_sampling_alpha", 1.0))
            if sampler is not None:
                effective_shuffle = False
                print(f"Using group-balanced sampler with alpha={CFG.get('group_sampling_alpha', 1.0)}")
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=effective_shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            collate_fn=collator,
            generator=LOADER_GENERATOR,
        )
        return loader, collator

    def _resolve_epoch_logging_steps(loader) -> int:
        explicit = int(os.getenv("HWVLM_LOGGING_STEPS", "0") or 0)
        if explicit > 0:
            return explicit
        # Default: log roughly every 5% of an epoch, i.e. about 20 logs/epoch.
        return max(1, int(math.ceil(len(loader) * 0.05)))


    def rebuild_student_dataloaders(current_processor):
        global collator, train_loader, valid_loader
        train_loader, collator = _make_loader(
            train_dataset,
            current_processor,
            batch_size=CFG["per_device_batch_size"],
            shuffle=True,
            num_workers=CFG["num_workers"],
        )
        valid_loader, _ = _make_loader(
            valid_dataset,
            current_processor,
            batch_size=CFG["per_device_batch_size"],
            shuffle=False,
            num_workers=CFG["num_workers"],
        )
        CFG["logging_steps"] = _resolve_epoch_logging_steps(train_loader)
        print(f"Train batches: {len(train_loader)}")
        print(f"Valid batches: {len(valid_loader)}")
        print(f"W&B/train progress logging every {CFG['logging_steps']} batches (~5% epoch by default)")


    def _sample_source_name(sample: Dict[str, Any]) -> str:
        return str(sample.get("metadata", {}).get("source_dataset", "unknown")).strip() or "unknown"


    def _count_sources(samples: List[Dict[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for sample in samples:
            source = _sample_source_name(sample)
            counts[source] = counts.get(source, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: kv[0]))


    def _exclude_sources_inplace(dataset, excluded_sources: List[str], split_name: str):
        if not excluded_sources or not hasattr(dataset, "samples"):
            return dataset

        excluded = {item.strip() for item in excluded_sources if item.strip()}
        before = len(dataset.samples)
        before_counts = _count_sources(dataset.samples)
        dataset.samples = [
            sample for sample in dataset.samples
            if _sample_source_name(sample) not in excluded
        ]
        after = len(dataset.samples)
        after_counts = _count_sources(dataset.samples)

        print(f"{split_name} excluded sources: {sorted(excluded)}")
        print(f"{split_name} samples: {before} -> {after}")
        print(f"{split_name} source counts before: {before_counts}")
        print(f"{split_name} source counts after : {after_counts}")
        return dataset


    def _sample_task_name(sample: Dict[str, Any]) -> str:
        return str(sample.get("metadata", {}).get("task", "unknown")).strip() or "unknown"


    def _count_tasks(samples: List[Dict[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for sample in samples:
            task = _sample_task_name(sample)
            counts[task] = counts.get(task, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: kv[0]))


    def validate_full_pass_page_contract(dataset) -> Dict[str, Any]:
        samples = _dataset_samples_for_stats(dataset)
        record_ids = [str(sample.get("id", "")) for sample in samples]
        duplicate_ids = [record_id for record_id, count in Counter(record_ids).items() if count != 1]
        if duplicate_ids:
            raise ValueError(f"Full-pass sampling requires unique record IDs: {duplicate_ids[:10]}")
        pages = defaultdict(list)
        for sample in samples:
            source = _source_name_from_sample(sample)
            image_relpath = str((sample.get("images") or [sample.get("id", "")])[0])
            pages[f"{source}::{image_relpath}"].append(str((sample.get("metadata") or {}).get("task", "unknown")))
        bad_pages = {
            page_key: tasks
            for page_key, tasks in pages.items()
            if tasks.count("page_htr") != 1
        }
        if bad_pages:
            raise ValueError(
                "Every page must have exactly one page_htr record before full-pass training. "
                f"First violations: {list(bad_pages.items())[:10]}"
            )
        return {
            "records": len(samples),
            "pages": len(pages),
            "page_htr_records": sum(tasks.count("page_htr") for tasks in pages.values()),
            "tasks": _count_tasks(samples),
            "sources": _count_sources_from_samples(samples),
        }


    def _exclude_tasks_inplace(dataset, excluded_tasks: List[str], split_name: str):
        if not excluded_tasks or not hasattr(dataset, "samples"):
            return dataset

        excluded = {item.strip() for item in excluded_tasks if item.strip()}
        before = len(dataset.samples)
        before_counts = _count_tasks(dataset.samples)
        dataset.samples = [
            sample for sample in dataset.samples
            if _sample_task_name(sample) not in excluded
        ]
        after = len(dataset.samples)
        after_counts = _count_tasks(dataset.samples)

        if after == 0:
            raise ValueError(f"{split_name} split is empty after excluding tasks: {sorted(excluded)}")

        print(f"{split_name} excluded tasks: {sorted(excluded)}")
        print(f"{split_name} samples: {before} -> {after}")
        print(f"{split_name} task counts before: {before_counts}")
        print(f"{split_name} task counts after : {after_counts}")
        return dataset


    train_path = resolve_split_path(DATA_ROOT, CFG["dataset"], "train")
    valid_path = resolve_split_path(DATA_ROOT, CFG["dataset"], "valid")

    if train_path is None:
        raise FileNotFoundError(
            f"Could not find train split for dataset '{CFG['dataset']}' under {DATA_ROOT}"
        )

    IMAGE_ROOT = discover_image_root(train_path)

    print(f"Train JSONL: {train_path}")
    print(f"Valid JSONL: {valid_path}")
    print(f"Image root : {IMAGE_ROOT}")

    train_dataset = OCRJsonlDataset(
        jsonl_path=train_path,
        image_root=IMAGE_ROOT,
        prompt_text=None,
        augment=CFG["use_augmentation"],
    )
    train_dataset = _exclude_sources_inplace(train_dataset, CFG["excluded_sources"], "Train")
    train_dataset = _exclude_tasks_inplace(train_dataset, CFG["excluded_train_tasks"], "Train")
    FULL_PASS_PAGE_CONTRACT = validate_full_pass_page_contract(train_dataset)
    print("Full-pass page contract:", FULL_PASS_PAGE_CONTRACT)
    _train_samples = _dataset_samples_for_stats(train_dataset)
    TOKEN_UL_ELIGIBLE_RECORDS = sum(
        str((sample.get("metadata") or {}).get("task", "unknown")) == "page_htr"
        and _stable_unit_interval(str(sample.get("id", "")), CFG["seed"]) < float(CFG["token_ul_page_fraction"])
        for sample in _train_samples
    )
    CFG["token_ul_eligible_train_records"] = int(TOKEN_UL_ELIGIBLE_RECORDS)
    if CFG["token_ul_coef"] > 0.0 and TOKEN_UL_ELIGIBLE_RECORDS == 0:
        raise RuntimeError("Token UL is enabled but the deterministic train subset is empty.")
    print("Token-UL train page_htr records:", TOKEN_UL_ELIGIBLE_RECORDS, "/", FULL_PASS_PAGE_CONTRACT["page_htr_records"])
    configure_source_loss_weights(train_dataset, CFG)
    configure_language_group_class_weights(model, train_dataset, CFG)

    if valid_path is None:
        raise FileNotFoundError(
            "Canonical page_multitask requires its declared valid.jsonl; "
            "refusing to derive a random validation split."
        )
    valid_dataset = OCRJsonlDataset(
        jsonl_path=valid_path,
        image_root=IMAGE_ROOT,
        prompt_text=None,
        augment=False,
    )
    valid_dataset = _exclude_sources_inplace(valid_dataset, CFG["excluded_sources"], "Valid")
    valid_dataset = _exclude_tasks_inplace(valid_dataset, CFG["excluded_valid_tasks"], "Valid")

    for split_name, dataset_obj in (("train", train_dataset), ("valid", valid_dataset)):
        actual_sources = set(_count_sources(_dataset_samples_for_stats(dataset_obj)))
        expected_sources = set(CFG["expected_sources"])
        if actual_sources != expected_sources:
            raise ValueError(
                f"{split_name} sources differ from canonical scope: "
                f"missing={sorted(expected_sources - actual_sources)}, "
                f"unexpected={sorted(actual_sources - expected_sources)}"
            )
        expected_records = int(CFG["expected_split_records"][split_name])
        if len(dataset_obj) != expected_records:
            raise ValueError(
                f"{split_name} record count differs from canonical artifact: "
                f"got={len(dataset_obj)} expected={expected_records}"
            )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(valid_dataset)}")
    print("Train source counts:", _count_sources(_dataset_samples_for_stats(train_dataset)))
    print("Valid source counts:", _count_sources(_dataset_samples_for_stats(valid_dataset)))


    def _sampling_page_key(sample: Dict[str, Any]) -> str:
        source = _source_name_from_sample(sample)
        image_relpath = str((sample.get("images") or [sample.get("id", "")])[0])
        return f"{source}::{image_relpath}"


    def _nested_counter(counter: Counter) -> Dict[str, Dict[str, float]]:
        result: Dict[str, Dict[str, float]] = {}
        for (source, task), value in sorted(counter.items()):
            result.setdefault(str(source), {})[str(task)] = value
        return result


    def build_sampling_plan(dataset, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Compute expected record draws from the exact active sampler formula."""
        samples = _dataset_samples_for_stats(dataset)
        if not samples:
            raise RuntimeError("Cannot audit sampling: active train dataset exposes no samples")
        total_draws = len(samples)
        source_counts = Counter(_source_name_from_sample(sample) for sample in samples)
        group_counts = Counter(language_group_from_metadata(sample.get("metadata", {}) or {})[1] for sample in samples)
        raw_source_task = Counter(
            (_source_name_from_sample(sample), _sample_task_name(sample)) for sample in samples
        )
        raw_task = Counter(_sample_task_name(sample) for sample in samples)

        sampling_mode = str(cfg.get("sampling_mode", "full_pass")).lower()
        if sampling_mode == "source_balanced":
            alpha = min(max(float(cfg.get("source_sampling_alpha", 1.0)), 0.0), 1.0)
            sampler_name = "source_balanced_with_replacement"
            sample_weights = [source_counts[_source_name_from_sample(sample)] ** (alpha - 1.0) for sample in samples]
        elif sampling_mode == "group_balanced":
            alpha = min(max(float(cfg.get("group_sampling_alpha", 1.0)), 0.0), 1.0)
            sampler_name = "script_group_balanced_with_replacement"
            sample_weights = [
                group_counts[language_group_from_metadata(sample.get("metadata", {}) or {})[1]] ** (alpha - 1.0)
                for sample in samples
            ]
        elif sampling_mode == "full_pass":
            alpha = 1.0
            sampler_name = "full_pass_shuffle_without_replacement"
            sample_weights = [1.0 for _ in samples]
        else:
            raise ValueError(f"Unsupported sampling_mode={sampling_mode!r}")

        weight_sum = max(sum(sample_weights), 1e-12)
        expected_source = Counter()
        expected_task = Counter()
        expected_source_task = Counter()
        expected_page_htr_by_source = Counter()
        pages_by_source: Dict[str, set] = defaultdict(set)
        for sample, weight in zip(samples, sample_weights):
            source = _source_name_from_sample(sample)
            task = _sample_task_name(sample)
            expected = total_draws * float(weight) / weight_sum
            expected_source[source] += expected
            expected_task[task] += expected
            expected_source_task[source, task] += expected
            pages_by_source[source].add(_sampling_page_key(sample))
            if task == "page_htr":
                expected_page_htr_by_source[source] += expected

        num_epochs = int(cfg.get("num_epochs", 1))
        plan = {
            "sampler": sampler_name,
            "alpha": alpha,
            "replacement": sampler_name.endswith("with_replacement"),
            "records_drawn_per_epoch": total_draws,
            "planned_epochs": num_epochs,
            "raw_records_by_source": dict(sorted(source_counts.items())),
            "raw_records_by_task": dict(sorted(raw_task.items())),
            "raw_records_by_source_task": _nested_counter(raw_source_task),
            "expected_draws_per_epoch_by_source": {key: float(value) for key, value in sorted(expected_source.items())},
            "expected_draws_per_epoch_by_task": {key: float(value) for key, value in sorted(expected_task.items())},
            "expected_draws_per_epoch_by_source_task": _nested_counter(expected_source_task),
            "expected_draws_total_by_source": {key: float(value * num_epochs) for key, value in sorted(expected_source.items())},
            "expected_draws_total_by_task": {key: float(value * num_epochs) for key, value in sorted(expected_task.items())},
            "expected_draws_total_by_source_task": _nested_counter(Counter({key: value * num_epochs for key, value in expected_source_task.items()})),
            "unique_pages_by_source": {key: len(value) for key, value in sorted(pages_by_source.items())},
            "expected_page_htr_draws_per_page_over_run": {
                source: float(expected_page_htr_by_source[source] * num_epochs / max(1, len(pages)))
                for source, pages in sorted(pages_by_source.items())
            },
        }
        return plan


    rebuild_student_dataloaders(processor)
    SAMPLING_PLAN = build_sampling_plan(train_dataset, CFG)
    sampling_plan_path = Path(CFG["output_dir"]) / "sampling_plan.json"
    if state.command == "train":
        sampling_plan_path.parent.mkdir(parents=True, exist_ok=True)
        sampling_plan_path.write_text(json.dumps(SAMPLING_PLAN, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Sampling plan:", SAMPLING_PLAN["sampler"], "alpha=", SAMPLING_PLAN["alpha"])
    print("Expected draws/epoch by source:", {key: round(value, 1) for key, value in SAMPLING_PLAN["expected_draws_per_epoch_by_source"].items()})
    print("Expected draws/epoch by task:", {key: round(value, 1) for key, value in SAMPLING_PLAN["expected_draws_per_epoch_by_task"].items()})
    if state.command == "train":
        print("Sampling plan written:", sampling_plan_path)

    return export_runtime_locals(state, locals())
