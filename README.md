# ExpertHTR

<p align="center"><b>Page-level handwritten text recognition with multi-task learning and Sparse MoE</b></p>

<p align="center"><a href="https://huggingface.co/DAIR-Group/ExpertHTR">🤗 Model</a> · <a href="https://huggingface.co/datasets/DAIR-Group/ExpertHTR-Dataset">🤗 Dataset</a> · <a href="hf_space_demo/">🌐 Demo</a></p>

ExpertHTR fine-tunes Qwen3.5-0.8B-Base for handwritten pages. It jointly learns page transcription, line counting, phrase-to-line lookup and line-range transcription, with four routed experts and one shared expert.

<p align="center">
  <img src="docs/assets/experthtr-architecture.png" alt="ExpertHTR architecture" width="950">
</p>

## Results

The validation-selected checkpoint (`step 1914`) reaches **16.439% micro page CER** on 962 validation pages. CER removes only the leading region marker (`[Rk]:`); `<del>` and `<gap>` remain distinct OCR symbols.

## Quick start

```bash
git clone https://github.com/DAIR-Group/ExpertHTR.git
cd ExpertHTR
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e .
python -m pip install -r requirements-core.txt
experthtr --validate-imports
```

```bash
export HWVLM_DATA_ROOT=/path/to/mthtr_page_multitask_7source
export HWVLM_MODEL_NAME=/path/to/Qwen3.5-0.8B-Base
export HWVLM_DENSE_INIT_CHECKPOINT=/path/to/dense_checkpoint
export HWVLM_OUTPUT_ROOT=/path/to/outputs
experthtr audit
experthtr train
HWVLM_FINAL_TEST_CHECKPOINT=/path/to/step_001914_cer_16.4390 experthtr final-test
```

Check one real image:

```bash
python scripts/smoke_test_image.py /path/to/mthtr_page_multitask_7source --split test
```

## Dataset and model

The dataset uses `page_multitask/{train,valid,test}.jsonl` plus a shared `img/` tree. Read [DATA_LICENSES.md](DATA_LICENSES.md) and the dataset card before redistribution. The mixed-source bundle is private or gated pending permission review.

The [HF Space demo](hf_space_demo/) provides a clean upload-and-transcribe UI. Set `MODEL_ID` to the published model repository and run `python app.py`.

## Repository layout

`src/experthtr/` workflow · `configs/` reference settings · `scripts/` audits and release tools · `docs/` method and reproducibility notes · `hf_space_demo/` Gradio demo.

## License

Code and documentation are Apache-2.0. The base model, checkpoint and handwriting datasets retain their upstream licenses; see [MODEL_CARD.md](MODEL_CARD.md) and [DATA_LICENSES.md](DATA_LICENSES.md).
