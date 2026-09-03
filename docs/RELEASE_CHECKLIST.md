# Release checklist

Run this checklist immediately before pushing a public repository or submitting paper artifacts.

## Repository hygiene

- [ ] `python scripts/check_release.py` passes.
- [ ] `git status --ignored` confirms that raw data, checkpoints, predictions, `outputs/`, caches, and local `.env` files are ignored.
- [ ] The optional demo notebook has no executed outputs, private paths, access tokens, account names, or unpublished result claims.
- [ ] The canonical Python workflow passes `python scripts/check_python_contract.py`.
- [ ] No source archive, derived JSONL, image, model weight, or checkpoint is committed.
- [x] A project owner selected Apache-2.0 and added `LICENSE`.
- [x] A temporary `CITATION.cff` records the working title and author list.
- [ ] Update `CITATION.cff` with final repository URL, venue, publication year, and DOI.

## Scientific contract

- [ ] The release config matches the exact reported architecture: four routed + one shared, Sparsegen, six layer indices, LamSparse and balance coefficients, no router CE.
- [ ] The data scope states that EPARCHOS is excluded; source and task counts match the run manifest.
- [ ] Bbox prediction is not claimed in the method, data table, or figures.
- [ ] Validation selection uses only micro page-CER.
- [ ] Test evaluation is run once after selection and is not used to tune settings.
- [ ] All metric tables identify primary page-CER versus diagnostic variants.
- [ ] HWDB WER is omitted or labelled not comparable without a declared segmentation policy.

## Runtime evidence

- [ ] Full token audit matches the exact JSONL/image/model/processor/prompt/budget/context contract.
- [ ] Dense initialization and selected MoE checkpoint manifests are archived.
- [ ] `build_run_manifest.py` was executed for the reported checkpoint.
- [ ] `pip freeze`, wheel hashes, base-model metadata hashes, GPU, CUDA, and PyTorch versions are archived.
- [ ] The test prediction JSONL, metrics JSON, source CSV, and routing-analysis files are retained privately or deposited appropriately.

## Figure claims

- [ ] Routing figures name whether positions are all-attended, vision, prompt, or target positions.
- [ ] Ranked activation curves explicitly state that experts are re-ranked within each layer.
- [ ] The dashed equal-utilization reference is defined as \(\mathbb{E}[K]/4\).
- [ ] Do not describe dynamic Sparsegen support as fixed Top-*k*.
- [ ] Do not claim pure script specialization or compute savings without their required ablations/measurements.
