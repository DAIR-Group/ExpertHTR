# Evaluation protocol

## Selection before test

1. Train the one-epoch MoE continuation with the frozen input/loss contract.
2. Evaluate the `page_htr` validation split using greedy decoding.
3. Rank checkpoints by **minimum micro page-CER** over all validation reference characters.
4. Select one checkpoint using validation only.
5. Load that checkpoint strictly and run held-out test once.

The training command stores the selected checkpoint path in `run_summary.json`, then the final-test command validates its MoE signature and atomic-marker IDs before generation. Test CER must never enter checkpoint selection or hyperparameter tuning.

## Primary metric

For reference \(r_i\) and prediction \(p_i\), page CER is aggregated micro-style:

\[
\mathrm{CER}=\frac{\sum_i \operatorname{Lev}(p_i,r_i)}{\sum_i |r_i|}.
\]

Before Levenshtein distance, both strings use the same minimal evaluation transform:

1. normalize CRLF/CR to LF;
2. remove only synthetic `[Rk]:` or `[Para k]:` formatting headers;
3. encode `<del>` and `<gap>` as two distinct single comparison units;
4. preserve all remaining transcription, including spaces, punctuation, and physical newlines.

This is a canonical page-CER, not a tokenizer metric and not a formatting-insensitive OCR score. If a raw serialization CER, text-only diagnostic CER, or safety CER is additionally reported, name it explicitly and never substitute it for the primary value.

## Required reports

The final-test command writes a full JSON metric file, prediction JSONL, and a flat source CSV. The report contains:

- overall and per-source page-CER and WER;
- number of pages and generated/reference token statistics;
- seconds/page, pages/s, and generated tokens/s from batch-size-one greedy generation;
- EOS rate, max-new-token cap-hit rate, empty-output rate, loop diagnostic, and 3-gram repetition diagnostic;
- region-count and line-count exact diagnostics;
- reference/prediction counts for `<del>` and `<gap>`.

WER is a supplementary diagnostic. It is not directly comparable for HWDB2.0 without a declared Chinese word-segmentation procedure; report its value as `N/A` in a cross-source WER table unless such a policy is implemented before testing.

## Generation contract

| Setting | Value |
| --- | ---: |
| Decoding | greedy (`do_sample=False`) |
| Final-test batch size | 1 |
| Maximum generated tokens | 1,024 |
| Terminal behavior | resolved from the tokenizer/model and logged |
| Model selection split | validation `page_htr` only |

A cap hit, loop warning, or missing EOS is a generation diagnostic. The primary CER remains scored on the raw prediction; a safety-trimmed CER may be logged separately to quantify the contribution of runaway suffixes.

## Routing analysis

The routing probe is teacher-forced and uses a deterministic, stratified test sample of up to 50 pages per source. It measures all attended positions (vision, prompt/context, and transcription; padding excluded), not only output text positions.

For each selected MoE layer, the probe exports:

- support distribution \(P(K=1),\ldots,P(K=4)\);
- mean active routed experts \(\mathbb{E}[K]\) with page-bootstrap confidence intervals;
- routed-expert activation \(P(p_e>0)\);
- equal-utilization reference \(\mathbb{E}[K]/4\).

The four activation curves are sorted **within each layer** for the paper-facing ranked view. They are not identities of one fixed expert carried through depth. State this in the caption to avoid an invalid cross-layer specialization claim.
