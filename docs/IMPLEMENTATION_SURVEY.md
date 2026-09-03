# Implementation survey and repository design decisions

**Scope.** This is an engineering survey, not a claim that ExpertHTR implements
every method below. Paper claims are supported by the original publications;
repository star counts are only a practical screening threshold requested for
choosing mature public implementation patterns. Counts were checked on
**2026-09-02** and are necessarily time-varying.

## Included references

| Paper / system | Public implementation consulted | Stars at check | What ExpertHTR adopts | What it deliberately does not copy |
| --- | --- | ---: | --- | --- |
| [Switch Transformers](https://arxiv.org/abs/2101.03961) | [Hugging Face Transformers](https://github.com/huggingface/transformers) | 164.7k | Separate configuration, model components, and task/evaluation code. | Top-1 routing and its capacity/drop policy; ExpertHTR uses native Sparsegen. |
| [GShard](https://arxiv.org/abs/2006.16668) | [T5X](https://github.com/google-research/t5x) | 3.0k | Configured train/eval/inference entry points; no hidden interactive state. | TPU auto-sharding and expert parallelism. |
| [V-MoE](https://arxiv.org/abs/2106.05974) | [google-research/vmoe](https://github.com/google-research/vmoe) | 726 | A dedicated MoE component boundary and explicit router diagnostics. | Vision-classification targets and capacity-based token priority. |
| [MegaBlocks](https://arxiv.org/abs/2211.15841) | [databricks/megablocks](https://github.com/databricks/megablocks) | 1.6k | Model/runtime separation; measured sparse behavior rather than inferred efficiency. | Distributed grouped-GEMM kernels and dropless routing claims. |
| [Mixtral 8x7B](https://arxiv.org/abs/2401.04088) | [mistral-inference](https://github.com/mistralai/mistral-inference) | 11k | Lightweight CLI-oriented run interface and packaged inference path. | Its fixed Top-*k* gate; this repository compares that only as an ablation. |
| [DeepSeekMoE](https://arxiv.org/abs/2401.06066) | [deepseek-ai/DeepSeek-MoE](https://github.com/deepseek-ai/DeepSeek-MoE) | 1.6k | Keep shared-expert behavior explicit in architecture/configuration. | Fine-grained expert segmentation and claims of language-specific experts. |
| [OpenMoE](https://arxiv.org/abs/2402.01739) | [XueFuzhao/OpenMoE](https://github.com/XueFuzhao/OpenMoE) | 1.7k | Configuration-driven experiments, ablation visibility, and routing analysis. | Fixed-capacity token dropping; ExpertHTR has no token queue or drop path. |
| [OLMoE](https://arxiv.org/abs/2409.02060) | [allenai/OLMoE](https://github.com/allenai/OLMoE) | 1.0k | Separate configurations, scripts, reproducibility artifacts, and analysis outputs. | Its pretraining recipe, data mixture, and parameter-scale conclusions. |
| [MoE-LLaVA](https://arxiv.org/abs/2401.15947) | [PKU-YuanGroup/MoE-LLaVA](https://github.com/PKU-YuanGroup/MoE-LLaVA) | 2.3k | Treat multimodal data/model/evaluation as separate public surfaces. | Its Top-*k* deployment gate and object-hallucination task claims. |
| [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | [deepseek-ai/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) | 104.4k | Explicit architecture and checkpoint compatibility metadata. | Expert parallelism, multi-token prediction, MLA, and large-scale system claims. |

The repositories above all exceeded the requested 500-star screen when
checked. Stars are **not** evidence of correctness or scientific quality; they
were used only to avoid copying a transient or unmaintained project.

## Architecture adopted in ExpertHTR

```text
src/experthtr/
├── config/       # frozen reference controls and environment boundary
├── runtime/      # dependency bootstrap and explicit RuntimeState
├── data/         # corpus contract, JSONL dataset, collator, loaders
├── modeling/     # Qwen loading, dense-transfer validation, routed MLP-MoE
├── training/     # objective, optimizer/scheduler, checkpointing, train loop
├── evaluation/   # canonical CER/WER, generation, held-out final test
├── analysis/     # teacher-forced routing and position diagnostics
├── audit/        # processor-accurate token/image/prompt audit
├── reporting/    # dataset summary only
└── cli.py        # stable command-line surface
```

This follows the recurring pattern in T5X, Transformers, Megatron-style
libraries, OLMoE, and MoE-LLaVA: public commands compose named components;
the repository does not depend on notebook execution order. Train, held-out
test, and routing analysis are distinct commands so test metrics cannot enter
checkpoint selection accidentally.

## Compatibility boundary

The previous implementation was a single linear runtime. The first
Python-first refactor retains the audited execution order through an explicit
`RuntimeState`: a phase exports model/processor/loader/checkpoint helpers only
after constructing them, and downstream phases consume that state. This is a
compatibility boundary, not an architectural claim. It preserves answer masks,
atomic marker IDs, loss normalisation, dense-to-MoE transfer checks, and metric
definitions while removing the notebook and dynamic `exec` runner.

Do not introduce capacity factors, token dropping, Top-*k*, source labels, or
expert-parallel performance claims merely because one surveyed repository uses
them. Those mechanisms would change the frozen Sparsegen experiment and require
a new ablation plus token-audit and checkpoint-contract validation.
