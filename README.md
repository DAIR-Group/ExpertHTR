# ExpertHTR

> Sparsegen-routed mixture-of-experts for full-input handwritten text recognition.

ExpertHTR is a reproducibility artifact for a [Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base)-based handwritten-text-recognition pipeline. It fine-tunes a dense HTR checkpoint with answer-only, full-input multi-task supervision, then expands six decoder MLPs into a **four-routed-expert plus one shared-expert** mixture. The released reference arm combines native Sparsegen routing, a LamSparse regularizer, and a small Switch-style routed-expert load-balance penalty.

The repository is deliberately designed for paper review:

- the canonical runtime is a Python CLI; the notebook is an optional walkthrough only;
- all dataset/model/checkpoint paths are explicit inputs rather than hidden local assumptions;
- checkpoint compatibility, atomic-marker IDs, source scope, record counts, and test-selection rules are checked at runtime;
- raw data, model weights, prediction files, and experiment outputs are excluded from Git.

## What this repository contains

```text
ExpertHTR/
├── configs/
│   ├── moe_balance.yaml          # frozen paper-facing experiment contract
│   └── paths.env.example         # non-secret path template
├── docs/
│   ├── ARCHITECTURE.md           # exact MoE and loss definitions
│   ├── RESULTS.md                # manuscript-aligned results and evidence gate
│   ├── CLI.md                    # Python commands and phase contracts
│   ├── DATASET_AND_TASKS.md      # data scope and task grammars
│   ├── EVALUATION.md             # selection, test, and metric protocol
│   ├── IMPLEMENTATION_SURVEY.md  # MoE implementation patterns consulted
│   ├── RELEASE_CHECKLIST.md      # pre-GitHub / pre-submission gate
│   └── REPRODUCIBILITY.md        # end-to-end run contract
├── examples/
│   └── demo_moe_balance.ipynb    # optional CLI walkthrough; no implementation
├── scripts/
│   ├── check_release.py          # static contract validation
│   ├── check_python_contract.py  # validates/compiles named Python modules
│   ├── check_demo_notebook.py    # keeps the notebook presentation-only
│   └── build_run_manifest.py     # records hashes and runtime provenance
├── src/experthtr/
│   ├── cli.py                    # `experthtr <command>` entry point
│   ├── workflow.py               # explicit phase composition
│   ├── data/                     # dataset contract, JSONL records, loaders
│   ├── modeling/                 # Qwen loading, dense transfer, Sparsegen MoE
│   ├── training/                 # objective, checkpointing, train loop
│   ├── evaluation/               # CER/WER and final held-out test
│   ├── analysis/                 # routing and position diagnostics
│   └── audit/                    # token/image/prompt audit
├── requirements-core.txt
├── CITATION.cff                  # temporary manuscript citation metadata
├── DATA_LICENSES.md              # dataset provenance and license boundary
├── LICENSE                       # Apache-2.0 for original code/documentation
├── MODEL_CARD.md                 # model/checkpoint release boundary
└── NOTICE                        # copyright and third-party attribution notice
```

Contribution and review rules are in [CONTRIBUTING.md](CONTRIBUTING.md). They
make explicit which changes require a new token audit or a separately named
ablation.

## Reference experiment

| Component | Frozen reference setting |
| --- | --- |
| Initialization | Verified dense HTR checkpoint, then one MoE continuation epoch |
| Sources | BRESSAY, Bentham, HWDB2.0, IAM, ICFHR2016, Images_Courriers, ScribbleLens |
| Explicit exclusion | EPARCHOS |
| Train / valid / test records | 20,388 / 962 / 1,327 |
| Training tasks | Full-input HTR, line count, phrase-to-line lookup, line-span HTR |
| MoE placement | Decoder MLP layers 1, 5, 9, 13, 17, 21 |
| Experts per selected layer | 4 routed + 1 always-active shared expert |
| Router | Native Sparsegen; no fixed Top-*k*; router CE = 0 |
| Regularization | LamSparse = 0.01 and routed-expert load balance = 0.01, each warmed up over 10% of one epoch |
| Resolution / context | Area-ratio resize, 589,824–1,572,864 pixels; 2,560 multimodal tokens |
| Training | AdamW, LR 1e-6, weight decay 0.01, effective batch 32, cosine schedule |
| Selection | Minimum validation micro page-CER |
| Test | Greedy decoding, batch size 1, max 1,024 new tokens, only after validation selection |

The reference configuration is machine-checked by:

```bash
python scripts/check_release.py
python scripts/check_python_contract.py
```

## Reported results

The current manuscript-aligned table is in [docs/RESULTS.md](docs/RESULTS.md).
Sparse MoE records **2.92% CER on IAM page-level evaluation** in that table,
compared with 3.03% for the strongest previously reported IAM value shown.
This is a provisional comparison until the cited baselines, exact evaluation
protocol, checkpoint manifests, token audit, and predictions are archived.

## Quick start

### 1. Prepare an offline runtime

This artifact expects a local/offline installation of the Qwen3.5 model and its compatible `transformers` stack. The small `requirements-core.txt` is only a minimal dependency declaration; it is **not** a lock file. For a paper result, preserve the exact wheelhouse, model revision, and `pip freeze` from the run environment.

```bash
python -m pip install -e .
python -m pip install -r requirements-core.txt
```

### 2. Supply data and a dense initialization checkpoint

The repository does not redistribute the eight original corpora, the derived `page_multitask` JSONL bundle, Qwen weights, or checkpoints. Follow their original access terms, construct the canonical artifact, and set the paths as environment variables. The Python configuration module is [`src/experthtr/config/reference.py`](src/experthtr/config/reference.py); do not commit a local path edit.

```bash
export HWVLM_DATA_ROOT=/path/to/mthtr
export HWVLM_MODEL_NAME=/path/to/Qwen3.5-0.8B-Base
export HWVLM_DENSE_INIT_CHECKPOINT=/path/to/selected/dense/checkpoint
export HWVLM_OUTPUT_ROOT=/path/to/writable/outputs
```

The required inputs are:

```text
HWVLM_DATA_ROOT                  # mthtr root or page_multitask directory
HWVLM_OFFLINE_BUNDLE_ROOT        # wheelhouse/model bundle, if used
HWVLM_MODEL_NAME                 # Qwen3.5-0.8B-Base directory
HWVLM_DENSE_INIT_CHECKPOINT      # exact selected dense checkpoint directory
HWVLM_OUTPUT_ROOT                # writable parent directory for this run
```

Use [`configs/paths.env.example`](configs/paths.env.example) as a non-secret template. The dense checkpoint is required by default. A fresh-base run is a separately labeled control, not a replacement for the reference continuation.

### 3. Run the Python workflow

The normal research workflow is command-line driven. Each command composes exactly the named phases it needs, so there is no hidden notebook state or cell-order dependency.

```bash
# Validate named Python phases without loading a model or dataset.
experthtr --validate-imports

# Mandatory after a data/image/processor/prompt/resize/context change.
experthtr audit

# One dense-initialized MoE continuation epoch; validation selects the checkpoint.
experthtr train

# Run only after validation selection. This is never part of train.
HWVLM_FINAL_TEST_CHECKPOINT=/path/to/validation-selected/checkpoint \\
  experthtr final-test

# Routing analysis and position diagnostics require a matching Sparsegen checkpoint.
HWVLM_ROUTING_ANALYSIS_CHECKPOINT=/path/to/validation-selected/checkpoint \\
HWVLM_ROUTING_ANALYSIS_OUTPUT=/path/to/writable/routing_figures \\
  experthtr routing
HWVLM_POSITION_DIAG_CHECKPOINT=/path/to/validation-selected/checkpoint \\
HWVLM_POSITION_DIAG_OUTPUT=/path/to/writable/position_diagnostics \\
  experthtr position-diagnostics
```

Use `experthtr COMMAND --print-phases` to inspect the exact implementation order for any command. [`examples/demo_moe_balance.ipynb`](examples/demo_moe_balance.ipynb) merely mirrors these calls for interactive teaching; it contains no training or evaluation logic. The rationale for the package design is in [`docs/IMPLEMENTATION_SURVEY.md`](docs/IMPLEMENTATION_SURVEY.md).

The final-test command intentionally uses `batch_size=1`: its seconds/page figure is latency, not batch-throughput.

### 4. Freeze provenance beside the chosen checkpoint

```bash
python scripts/build_run_manifest.py \
  --data-root /path/to/mthtr \
  --checkpoint /path/to/selected/checkpoint \
  --model-dir /path/to/Qwen3.5-0.8B-Base \
  --output /path/to/output/run_manifest.json
```

Keep the generated manifest, the checkpoint's own manifest, token-audit report, validation predictions, final-test predictions, and the exact runtime package list with the submitted result. Do not commit weights or restricted data to the public repository.

## Design at a glance

![ExpertHTR architecture: multi-task supervision over a Qwen3.5 decoder with a shared-expert Sparse MoE layer.](docs/assets/experthtr-architecture.png)

For a selected decoder layer \(l\) and token state \(h_{l,t}\), the router creates a Sparsegen-normalized distribution \(p_{l,t}\) over four routed MLP copies. A fifth, shared MLP is always evaluated:

\[
\tilde h_{l,t} = (1-\alpha)\sum_{e=1}^{4} p_{l,t,e} E_{l,e}(h_{l,t}) + \alpha S_l(h_{l,t}), \qquad \alpha=0.25.
\]

Every expert begins as a copy of the corresponding dense MLP. Thus the inserted module preserves the inherited MLP function at initialization up to numerical precision; training, not a random architectural jump, creates specialization.

The optimization objective is the answer-token HTR loss plus two small routed-expert regularizers:

\[
\mathcal L = \mathcal L_{\mathrm{HTR}} + 0.01\,\mathcal L_{\mathrm{LamSparse}} + 0.01\,\mathcal L_{\mathrm{balance}}.
\]

`router CE`, token unlikelihood, group-balanced sampling, source-balanced sampling, augmentation, and bbox prediction are disabled in the frozen arm. Full definitions and limitations are in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Responsible reporting

- Report micro page-CER by source and overall. The canonical metric removes only synthetic `[Rk]:` headers and evaluates `<del>` and `<gap>` as atomic units; it retains physical newlines, spaces, and punctuation.
- Treat WER on Han text as not directly comparable to whitespace-delimited Latin-script WER unless a Chinese segmentation policy is separately defined.
- Do not select a checkpoint by test CER. The test evaluator records the selected checkpoint and emits predictions, source-level metrics, generation caps, EOS rates, loop diagnostics, and latency.
- Do not claim source/script-disentangled expert specialization: four routed experts are source-agnostic, and source distributions remain confounded with language, handwriting style, layout, and text length.
- Do not claim compute savings without measured FLOPs/latency under a fixed hardware and decoding protocol. Sparse support does not itself establish an end-to-end efficiency claim.

## Citation and licensing

The code and documentation originally authored for ExpertHTR are licensed under
[Apache-2.0](LICENSE). The temporary manuscript metadata is in
[CITATION.cff](CITATION.cff); update it with the final repository URL, venue,
publication year, and DOI when available.

Raw datasets, derived dataset artifacts, base-model weights, and fine-tuned
checkpoints are not covered by Apache-2.0 and are not redistributed here. Their
release conditions are documented in [DATA_LICENSES.md](DATA_LICENSES.md) and
[MODEL_CARD.md](MODEL_CARD.md).

## Status

This release is a **reproducibility package**, not a pretrained-model release. It intentionally covers the supervised MoE-balance pipeline only; PPO/RL fine-tuning and the alternate IAM-line transfer protocol are separate experiments and are not folded into this code path. Final scientific claims still require the selected checkpoint manifest, matching token audit, held-out test artifacts, and the exact environment captured for the reported run.
