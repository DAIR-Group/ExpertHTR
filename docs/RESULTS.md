# Reported results

## Status and evaluation boundary

The table below is the current manuscript-aligned result table supplied by the
authors. It is retained here for traceability, but it is **not** by itself a
reproducible benchmark claim. Before a public release or paper submission, link
each number to its cited source or released checkpoint and archive the ExpertHTR
checkpoint manifest, dataset-manifest hashes, matching token audit, predictions,
and CER report.

All entries are CER (%); lower is better. `†` denotes a value reported in the
original publication and `‡` a value evaluated from a released checkpoint.
`Param.` is the total parameter count in millions. A dash means the result was
not reported in the manuscript comparison.

| Method | Param. | BRES. | Bent. | HWDB | IAM | READ | RIMES | Scrib. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DAN† | 23.0 | 4.25 | — | 0.84 | 4.94 | — | 3.72 | 4.93 |
| Faster DAN† | 23.0 | 2.35 | — | 1.40 | 3.03 | 3.92 | 3.45 | 4.84 |
| Meta-DAN† | 23.0 | 4.32 | — | 1.59 | 3.24 | — | 3.25 | 4.37 |
| LightOnOCR++† | 1005 | 22.60 | 7.24 | 150.00 | 3.71 | 202.00 | 42.61 | 124.40 |
| PaddleOCR-VL‡ | 900 | 58.58 | 9.80 | 6.52 | 6.25 | 385.00 | 17.48 | 206.00 |
| DeepSeek-OCR‡ | 3336 | 157.00 | 35.95 | 22.03 | 9.54 | 1102.00 | 23.44 | 206.00 |
| CHTR adap† | 3755 | 40.60 | 27.17 | 15.40 | 7.80 | 177.00 | 27.77 | 49.28 |
| Qwen3.5 Base | 852.74 | 51.94 | 27.13 | 25.43 | 11.55 | 124.49 | 31.66 | 91.45 |
| **Sparse MoE** | **1117.79** | **19.96** | **17.57** | **4.63** | **2.92** | **52.23** | **5.59** | **42.52** |

On IAM page-level evaluation, Sparse MoE obtains 2.92% CER in this comparison;
the strongest prior value shown in the table is Faster DAN at 3.03% CER. This
statement must remain qualified by the exact split, preprocessing, CER
normalization, training-data scope, and source citations before being promoted
to a state-of-the-art claim.

## Citation completion checklist

For each non-ExpertHTR row, add the paper or released-checkpoint URL, version,
evaluation date, and the precise protocol used by the comparison. For Sparse
MoE, add the repository commit, base-model revision, dense and MoE checkpoint
hashes, dataset-manifest hashes, token-audit report, and prediction/metric
artifact.
