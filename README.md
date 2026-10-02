# ExpertHTR

Page-level handwritten text recognition using Qwen3.5-0.8B-Base and Sparse MoE. The model handles page transcription, line counting, phrase-to-line lookup, and line-range transcription.

<p align="center"><a href="https://huggingface.co/DAIR-Group/ExpertHTR">Model</a> · <a href="https://huggingface.co/datasets/DAIR-Group/ExpertHTR-Dataset">Dataset</a> · <a href="hf_space_demo/">Demo</a></p>

<p align="center"><img src="docs/assets/experthtr-architecture.png" alt="ExpertHTR architecture" width="800"></p>

## Method and results

Selected decoder MLPs are replaced by four routed experts and one shared expert. The reference arm uses Sparsegen routing with LamSparse and load-balance losses.

The selected checkpoint (`step 1914`) reaches **16.439% micro page CER** on 962 validation pages. CER removes region headers (`[Rk]:`) and keeps `<del>` and `<gap>` as symbols.

| Method | M params | BRES. | Bent. | HWDB | IAM | READ | RIMES | Scrib. |
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

All table values are CER (%), lower is better. `†` marks a published result; `‡` marks an evaluation from a released checkpoint. The table is manuscript-supplied, not a reproducible benchmark claim. The IAM comparison (2.92% vs. 3.03%) needs aligned splits, preprocessing, data scope, and citations before any state-of-the-art claim.

## Install and run

```bash
git clone https://github.com/DAIR-Group/ExpertHTR.git
cd ExpertHTR
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e . -r requirements-core.txt
experthtr --validate-imports
```

Set the dataset, base-model, dense-checkpoint, and output paths:

```bash
export HWVLM_DATA_ROOT=/path/to/page_multitask
export HWVLM_MODEL_NAME=/path/to/Qwen3.5-0.8B-Base
export HWVLM_DENSE_INIT_CHECKPOINT=/path/to/dense_checkpoint
export HWVLM_OUTPUT_ROOT=/path/to/outputs
experthtr audit
experthtr train
HWVLM_FINAL_TEST_CHECKPOINT=/path/to/selected_checkpoint experthtr final-test
```

Other commands: `experthtr dataset-summary`, `experthtr routing`, and `experthtr position-diagnostics`.

## Data, release, and demo

The local package includes no model weights, checkpoints, datasets, or predictions. The reference data uses BRESSAY, Bentham, HWDB2.0, IAM, ICFHR2016, RIMES, and ScribbleLens; EPARCHOS is excluded. Obtain data and the Qwen base model from their providers and follow their terms. The mixed-source dataset is private or gated. Document source, license, permission, and citation before sharing derived artifacts.

The code is Apache-2.0; upstream model and dataset licenses remain in force. This project is for research, not high-stakes transcription. A reported checkpoint should include model and checkpoint hashes, data manifests, token-audit report, validation selection, test predictions and metrics, runtime details, and known limitations.

The [demo](hf_space_demo/) accepts an image upload. Set `MODEL_ID` to the published model repository; run `python app.py` in the demo environment.
