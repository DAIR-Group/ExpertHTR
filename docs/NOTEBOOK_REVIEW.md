# Notebook-to-package review

Reference: `moe_dense_warmstart_ablation.ipynb`, inspected 2026-09-05 without
executing its cells. SHA-256:
`7a965303bc8e4b72e1bdc773f654ddd6adc404ec8dcb4cefa3dc68bf63e28804`.

## Metric correction

[LOCAL] The package preserves the notebook's page-CER computation. An earlier
review identified a documentation mismatch; it did not establish a regression
in the metric relative to this notebook. The evaluation documentation has now
been corrected, and the metric implementation has been retained.

Both implementations remove synthetic line-leading `[Rk]:` region headers
and legacy `[Parak]:` headers, where `k` is a numeric index. They retain literal
`R` characters and the transcription inside each region. Thus `[R1]: Red` and
`[R9]: Red` have zero CER relative to one another: region indexing is outside
this text metric and is handled separately by structural diagnostics.

The exact transform normalizes CRLF/CR, removes headers plus their adjacent
horizontal whitespace, maps `<del>` and `<gap>` to distinct single comparison
units, and strips the outer whitespace of the whole page. Internal newlines,
spaces, and punctuation remain. A header on its own line can therefore leave
an internal blank line; this review does not change that notebook behavior.

[LOCAL] `evaluate_loader` applies this transform symmetrically to references
and predictions for `page_htr`, then updates overall and per-source trackers.
The primary metric divides summed edit distances by summed reference lengths
(micro CER). Safety-trimmed predictions feed separate diagnostic trackers.
The primary score is not an average of per-page percentages and is not the
safety-trimmed score. CER values above 100% are possible through insertions.

## Verified computation and differing defaults

[LOCAL] AST comparison, normalizing only docstring indentation, found 74
matching core function/class definitions across the metric, training/evaluation
engine, MoE, checkpoint-helper, and dataset/collator modules. In particular,
`OCRJsonlDataset` and `QwenVLMCollator` differed only in docstring indentation.
The reusable check is `scripts/check_notebook_parity.py`.

The saved defaults are nevertheless different:

| Control | Reference notebook | Frozen repository |
| --- | --- | --- |
| `MOE_ABLATION_VARIANT` | `sparsegen_shared` | `sparsegen_shared_lamsparse_balance` |
| `INIT_MODE` | `fresh_base` | `dense_checkpoint` |
| `REUSE_UNCHANGED_TOKEN_AUDIT` | `True` | `False` |

These defaults identify different experiments despite shared computation.
The repository's documented balance-arm defaults are retained; the notebook
filename alone is not evidence that the saved configuration uses dense
initialization. Layer placement `(1, 5, 9, 13, 17, 21)` and the declared
LamSparse/load-balance coefficients of `0.01` match, but whether a loss is
enabled depends on the selected ablation arm.

Routing reports also differ: the notebook includes lambda statistics and a
raw Switch-style balance diagnostic omitted from the package's routing report.
Those reporting differences do not change primary CER or the core MoE code.

## CLI and release fixes

- Restrict the raw-data ignore rule to `/data/`, making `src/experthtr/data/`
  visible to Git and distributable. Add a release check for ignored Python sources.
- Resolve final-test output paths from the current runtime configuration. The
  notebook's top-level `output_dir = Path(output_dir)` was valid there but became
  an unbound local when copied into a function; standalone CLI execution also
  lacked the training cell's output variable. This distinction follows the
  [Python execution model](https://docs.python.org/3/reference/executionmodel.html).
- Export the configured output path during runtime bootstrap. Apply the
  nonempty-directory guard to new training runs, before model loading.
- Write initialization provenance and sampling plans only for `train`.
  An audit can precede training, and evaluation/analysis can access an existing
  run without replacing those training artifacts.
- Check both marker strings and marker IDs before loading evaluation or
  analysis weights. Missing or mismatched IDs now fail explicitly.

## Validation and limits

[LOCAL] All 14 regression tests passed, including numerical CER examples,
standalone final-test selection/report writing with a fake model, marker-ID
rejection before weight loading, output protection, and Git ignore regression.
Release configuration, Python phase registration, and notebook hygiene passed.

A temporary Git snapshot containing the revised working files was archived
into a clean directory. All checks and tests passed there. A wheel built from
that archive included the data package and installed successfully in a separate
virtual environment; the installed `experthtr --validate-imports` passed.
The user's repository index and commit history were not changed.

The supplied notebook contains 25 cells and no saved outputs. It establishes
source lineage, but cannot by itself establish which configuration produced
the reported IAM 2.92% CER. No GPU training, real checkpoint restore, full
dataset audit, or recomputation from original predictions was performed in this
review. Those require the exact run configuration, model, data, and prediction
artifacts. The original notebook and dataset files were not modified.
