"""reporting.dataset_summary phase for the ExpertHTR research workflow.

This module preserves the audited runtime computation. It is invoked by the
explicit CLI workflow and exports its declared runtime objects into a
:class:`~expert.runtime.context.RuntimeState`.
"""

from __future__ import annotations

from expert.runtime.context import RuntimeState, export_runtime_locals


def write_dataset_summary(state: RuntimeState) -> RuntimeState:
    """Execute this phase after its prerequisites have populated ``state``."""

    globals().update(state.values)
    # ─────────────────────────────────────────────────────────────────────────────
    #  Dataset task/span statistics for current page_multitask build
    # ─────────────────────────────────────────────────────────────────────────────

    import json
    from collections import Counter, defaultdict
    from pathlib import Path
    from typing import Any, Dict, List


    def _summarize_task_span_by_dataset(records: List[Dict[str, Any]]) -> Dict[str, Any]:
        task_by_dataset: Dict[str, Counter] = defaultdict(Counter)
        span_by_dataset: Dict[str, Counter] = defaultdict(Counter)
        total = 0
        for record in records:
            meta = record.get("metadata", {}) or {}
            dataset = str(meta.get("source_dataset") or meta.get("dataset") or "UNKNOWN")
            task_name = str(meta.get("task") or "UNKNOWN")
            task_by_dataset[dataset][task_name] += 1
            total += 1
            if task_name == "line_range_htr":
                span_by_dataset[dataset][int(meta.get("range_span", 0) or 0)] += 1
        return {
            "total": total,
            "task_by_dataset": task_by_dataset,
            "span_by_dataset": span_by_dataset,
        }


    def _print_counter_table(title: str, counters: Dict[str, Counter], columns: List[Any]) -> None:
        print("\n" + title)
        header = ["dataset"] + [str(col) for col in columns] + ["total"]
        widths = [max(7, len(str(item))) for item in header]
        rows = []
        for dataset in sorted(counters):
            counter = counters[dataset]
            values = [counter.get(col, 0) for col in columns]
            row = [dataset] + values + [sum(counter.values())]
            rows.append(row)
            widths = [max(width, len(str(value))) for width, value in zip(widths, row)]
        fmt = "  ".join("{" + f":<{width}" + "}" for width in widths)
        print(fmt.format(*header))
        print(fmt.format(*["-" * width for width in widths]))
        for row in rows:
            print(fmt.format(*row))


    # Report the records actually consumed by training, after the runtime-only
    # structure adapter.  The JSONL on disk remains unchanged.
    train_stats = _summarize_task_span_by_dataset(train_dataset.samples)
    print(f"Train records after runtime task adapter: {train_stats['total']}")
    _print_counter_table(
        "Task counts by source dataset",
        train_stats["task_by_dataset"],
        ["page_htr", "phrase_line_lookup", "line_count", "line_range_htr"],
    )
    _print_counter_table(
        "line_range_htr random R-local span distribution by source dataset",
        train_stats["span_by_dataset"],
        list(range(3, 8)),
    )

    all_train_tasks = Counter()
    for counter in train_stats["task_by_dataset"].values():
        all_train_tasks.update(counter)
    print("\nTotal train task counts:", dict(sorted(all_train_tasks.items())))
    print("InkData_paragraph records:", sum(counter.total() for dataset, counter in train_stats["task_by_dataset"].items() if dataset == "InkData_paragraph"))
    print("Active structure task is page-level line_count only:", all_train_tasks.get("line_count", 0))
    print("Legacy line_htr records:", all_train_tasks.get("line_htr", 0))

    return export_runtime_locals(state, locals())
