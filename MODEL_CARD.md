# ExpertHTR model card

## Status

This repository is a reproducibility package. It currently publishes code and
documentation only; it does not distribute base-model weights, dense HTR
initialization checkpoints, Sparse MoE checkpoints, training data, predictions,
or evaluation outputs.

## Model and method

ExpertHTR fine-tunes a dense handwritten-text-recognition initialization based
on Qwen3.5-0.8B-Base, then replaces selected decoder MLPs with four routed
experts and one always-active shared expert. The frozen reference arm uses
Sparsegen routing, LamSparse regularization, and a routed-expert load-balance
loss. The exact method is specified in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Intended use

The code is intended for research on page-level handwritten text recognition
and the auxiliary tasks documented in [docs/DATASET_AND_TASKS.md](docs/DATASET_AND_TASKS.md).
It is not a claim of suitability for high-stakes transcription, legal record
interpretation, identity verification, or autonomous archival decisions.

## Licensing boundary

- Original ExpertHTR code and documentation: [Apache-2.0](LICENSE).
- Qwen base model: obtain separately from its upstream provider and follow its
  model card and license.
- Fine-tuned checkpoints: not released yet. Any future release must state its
  exact base-model revision, checkpoint hash, training-data provenance, and a
  license compatible with all upstream terms.
- Datasets and derived records: see [DATA_LICENSES.md](DATA_LICENSES.md).

## Required evidence for a reported checkpoint

Each released or paper-reported checkpoint must be accompanied by its hash,
base-model revision, dense initialization hash, frozen configuration, dataset
manifest hashes, matching token-audit report, validation-selection record,
held-out test predictions, metric report, hardware/runtime details, and known
limitations.
