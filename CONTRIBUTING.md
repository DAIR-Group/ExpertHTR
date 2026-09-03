# Contributing to ExpertHTR

ExpertHTR is a paper-reproducibility artifact. A contribution is acceptable
only when it keeps the reported experiment auditable and does not silently
change its data, evaluation, or checkpoint contract.

## Before opening a pull request

1. Do not commit data, model weights, checkpoints, outputs, prediction JSONL,
   notebook output, tokens, or absolute personal paths.
2. Keep the runtime in `src/experthtr/`; notebooks in `examples/` may only call
   the public CLI.
3. Put a new behavior behind an explicit configuration setting and document
   whether it is the frozen reference arm or a separately named ablation.
4. Do not change the primary page-CER definition, the atomic-marker treatment,
   source/split scope, or checkpoint-selection protocol without a new
   reproducibility contract and a paper-visible rationale.
5. Run the local checks below.

```bash
python scripts/check_release.py
python scripts/check_python_contract.py
python scripts/check_demo_notebook.py
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Changes that require a new token audit

Run `experthtr audit` again before training or reporting results if a change
affects JSONL records, image roots, preprocessing, prompts, processor/model,
atomic marker IDs, resize budget, context length, or answer serialization.
Changing a loss, sampling policy, MoE topology, optimizer schedule, or
generation policy requires a separately named run and must not overwrite the
reference configuration.

## Pull-request evidence

Describe the motivation, configuration delta, checks executed, and whether the
change is expected to alter reported metrics. For behavior changes, attach the
new run manifest and keep test evaluation held out until validation selection is
complete.
