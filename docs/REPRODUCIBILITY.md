# Reproducibility protocol

## Two levels of reproducibility

This repository can reproduce the **code path** immediately. Reproducing a reported number additionally requires the exact data artifact, model files, dense checkpoint, and runtime environment. The former is necessary but not sufficient for the latter.

| Artifact | Required for a paper-level rerun | Publicly committed here |
| --- | --- | --- |
| Clean Python workflow sources | yes | yes |
| Frozen balance configuration | yes | yes |
| Raw source data / images | yes | no |
| Canonical JSONL bundle | yes | no |
| Qwen model and processor files | yes | no |
| Dense initialization checkpoint | yes | no |
| Selected MoE checkpoint | needed for evaluation only | no |
| Token-audit report | yes | no |
| Run/checkpoint/prediction manifests | yes | generated locally |
| Dependency lock and wheel hashes | yes | generated locally |

## Input contract

Before running, make the following decisions explicit in the configuration stage or shell environment, never by an inherited interactive-session environment variable:

- dataset root and JSONL/image hashes;
- model and processor path/revision;
- dense checkpoint directory;
- output directory and new run suffix;
- GPU dtype, workers, and seed;
- image resize budget and multimodal context cap;
- task prompts, marker vocabulary, and terminal/stop IDs.

The Python workflow enforces the seven-source runtime scope and split-record counts. It also refuses a dense initialization checkpoint whose saved input/loss contract differs, except for a deliberately configured and documented source-superset transfer. Keep that exception disabled for a paper-final direct comparison unless the manuscript states the difference.

## Token audit gate

The release workflow defaults to `REUSE_UNCHANGED_TOKEN_AUDIT = False`. Therefore, it requires a full token audit before training. The audit must run through the actual processor/chat template and report, by split/source/task:

- tokenized prompt, vision, and target lengths;
- target-token coverage and truncation;
- zero-label rows, prompt-prefix mismatches, and terminal-token mismatches;
- processor/image-budget violations;
- reference loop-detector triggers.

Set the reuse flag to `True` only after preserving a matching audit artifact and confirming that JSONL, images, model/processor, prompts, atomic markers, resize budget, and context cap are byte-for-byte or contract-identical. It is never a substitute for an audit after a modification.

## Checkpoint provenance

Each checkpoint must carry:

- `CHECKPOINT_COMPLETE`;
- `checkpoint_manifest.json` with the MoE signature and marker IDs;
- `metadata.json`;
- `full_model_state.pt`;
- the parent `run_config.json`, training history, and validation-selection summary.

After final selection, create one portable run manifest:

```bash
python scripts/build_run_manifest.py \
  --data-root /path/to/mthtr \
  --checkpoint /path/to/checkpoint \
  --model-dir /path/to/Qwen3.5-0.8B-Base \
  --output /path/to/output/run_manifest.json
```

The manifest records SHA-256 hashes of JSONLs, checkpoint files, selected base-model metadata, public configuration, and package versions. Preserve `pip freeze` and wheel/model hashes in addition; package versions alone do not identify a patched offline wheel.

## Determinism

The workflow seeds Python, NumPy, and PyTorch, but the release does not claim bitwise deterministic GPU execution. Data-loader ordering, CUDA kernels, compiler choices, and distributed topology may still introduce small variation. Report results as seeded runs, retain per-run manifests, and use multiple seeds for a final comparative claim where budget permits.

## What to archive with a submission

1. Exact commit SHA of this repository.
2. `moe_balance.yaml` and a resolved copy of all runtime variables.
3. Full token-audit summary and worst-case rows.
4. Dense and MoE checkpoint manifests plus run manifests.
5. Validation prediction JSONL used for selection.
6. One final test prediction JSONL, metrics JSON, and source CSV.
7. Routing-analysis CSVs and final figures.
8. Environment lock (`pip freeze`), wheel hashes, model metadata hashes, GPU model, CUDA/PyTorch versions.
