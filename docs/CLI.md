# Command-line workflow

`experthtr` is the canonical execution interface. It composes named Python
phases from `config`, `runtime`, `data`, `modeling`, `training`, `evaluation`,
`analysis`, and `audit`; it does not depend on notebook state.

## Commands

| Command | Purpose | Does it modify outputs? |
| --- | --- | --- |
| `experthtr --validate-imports` | Validate registration of all named Python phases without loading model/data. | No |
| `experthtr dataset-summary` | Check retained source/task/split scope and print summary statistics. | No |
| `experthtr audit` | Perform the full token/image/prompt/terminal audit. | Writes only audit artifacts configured by the workflow. |
| `experthtr train` | Run the one-epoch dense-initialized MoE continuation and validation selection. | Yes |
| `experthtr final-test` | Evaluate exactly one validation-selected MoE checkpoint on held-out test pages. | Yes |
| `experthtr routing` | Produce teacher-forced Sparsegen routing summaries and figures. | Yes |
| `experthtr position-diagnostics` | Partition routing by vision, prompt, HTR target, and all attended positions. | Yes |

Use `experthtr COMMAND --print-phases` before a new release or code review to
show exactly which named components are executed for that command.

Only `train` writes `dense_to_moe_initialization.json` and `sampling_plan.json`.
Auditing a new run does not populate its training directory, so `audit` followed
by `train` works with the same configuration. A new `train` still refuses a
nonempty training directory. Evaluation and analysis can use an existing run
directory without overwriting those training provenance files.

Evaluation checkpoints must contain both `atomic_ocr_markers` and matching
`atomic_ocr_marker_ids` in their manifest. Missing IDs require the original
checkpoint/tokenizer provenance to be recovered; do not guess them.

## Required environment variables

Read paths from `configs/paths.env.example`; never hard-code a user path in a
source file. The minimum configuration for the reference arm is:

```bash
export HWVLM_DATA_ROOT=/path/to/mthtr
export HWVLM_MODEL_NAME=/path/to/Qwen3.5-0.8B-Base
export HWVLM_DENSE_INIT_CHECKPOINT=/path/to/dense/checkpoint
export HWVLM_OUTPUT_ROOT=/path/to/writable/outputs
```

`final-test`, `routing`, and `position-diagnostics` also take their explicit
checkpoint environment variables as documented in the README. Use a writable
analysis output path for routing/position figures when the checkpoint is
mounted read-only.

## Why phases share runtime state

The audited runtime intentionally shares a few objects across the training
path: `CFG`, model, processor, datasets/loaders, and checkpoint helpers.
`RuntimeState` makes that dependency explicit in one Python process. This
compatibility boundary is safer than reimplementing thousands of lines as
unrelated classes and accidentally altering answer masking, loss normalisation,
checkpoint compatibility, or the evaluation metric. A phase boundary is an
execution boundary, not a claim that every component is independently runnable.

New work should add a focused module plus a corresponding static contract check;
it should not copy code into the demo notebook.
