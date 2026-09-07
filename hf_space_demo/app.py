import os
from functools import lru_cache
import gradio as gr

MODEL_ID = os.getenv("MODEL_ID", "DAIR-Group/ExpertHTR")

@lru_cache(maxsize=1)
def load_pipeline():
    try:
        import torch
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype="auto", device_map="auto", trust_remote_code=True
        )
        model.eval()
        return processor, model
    except Exception as exc:
        raise RuntimeError(f"Could not load {MODEL_ID}: {exc}") from exc

def transcribe(image, prompt):
    if image is None:
        return "", "Upload a page image to begin."
    try:
        processor, model = load_pipeline()
        import torch
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=1024)
        answer = processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        return answer, f"Model: `{MODEL_ID}`"
    except Exception as exc:
        return "", f"Model loading/inference failed: {exc}"

CSS = """
.gradio-container {max-width: 1120px !important; margin: auto !important;}
.hero {text-align:center; padding: 28px 16px 16px;}
.hero h1 {font-size: clamp(2.2rem, 6vw, 4.2rem); letter-spacing:-.05em; margin:0 0 8px;}
.hero p {color:#64748b; font-size:1.08rem; margin:0 auto 18px; max-width:680px;}
.authors {color:#334155; font-size:1rem; margin:8px auto 4px;}
.affiliations {color:#64748b; font-size:.88rem; max-width:850px; margin:0 auto 12px;}
.pill-row {display:flex; justify-content:center; flex-wrap:wrap; gap:10px; margin:18px 0 28px;}
.pill {background:#20242b; color:white !important; border-radius:999px; padding:10px 18px; text-decoration:none !important; font-weight:600; font-size:.92rem; transition:transform .15s, background .15s;}
.pill:hover {background:#4f46e5; transform:translateY(-2px);}
.card {border:1px solid #e2e8f0; border-radius:18px; padding:18px; box-shadow:0 8px 28px rgba(15,23,42,.06);}
footer {display:none !important;}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate"), css=CSS, title="ExpertHTR · Handwritten Text Recognition") as demo:
    gr.HTML("""
    <section class="hero">
      <h1>ExpertHTR</h1>
      <div class="authors">Dang Hoai Nam · Nguyen Duy Hieu · Quang Huu Hieu · Vo Nguyen Le Duy</div>
      <div class="affiliations">University of Information Technology · Vietnam National University · AJ Technologies · RIKEN Center for Advanced Intelligence Project</div>
      <p>Page-level handwritten text recognition with multi-task learning and Sparse Mixture-of-Experts.</p>
      <div class="pill-row">
        <a class="pill" href="https://github.com/DAIR-Group/ExpertHTR" target="_blank">💻 Code</a>
        <a class="pill" href="https://huggingface.co/DAIR-Group/ExpertHTR" target="_blank">🤗 Model</a>
        <a class="pill" href="https://huggingface.co/datasets/DAIR-Group/ExpertHTR-Dataset" target="_blank">🗂️ Dataset</a>
      </div>
    </section>
    """)
    gr.Markdown("""### Abstract
Handwritten text recognition resources differ in language, script, document style and annotation structure. ExpertHTR unifies these collections with a common Page–Region–Line representation, four complementary tasks and a shared-expert Sparse Mixture-of-Experts architecture. The model is designed for page-level transcription while also learning physical line counting, text-to-line localization and line-span transcription.
""")
    gr.Markdown("### Try the model\nUpload a scanned handwritten page and receive a clean transcription.")
    with gr.Row():
        with gr.Column(scale=1, elem_classes="card"):
            image = gr.Image(type="pil", label="Handwritten page", sources=["upload", "clipboard"])
            prompt = gr.Textbox(value="Transcribe the handwritten text exactly. Output only the answer.", label="Instruction")
            run = gr.Button("Transcribe", variant="primary")
        with gr.Column(scale=1, elem_classes="card"):
            output = gr.Textbox(label="Transcription", lines=18, show_copy_button=True)
            status = gr.Markdown("Ready")
    gr.Markdown("Research demo · Results may contain errors. Do not use for high-stakes decisions.")
    run.click(transcribe, [image, prompt], [output, status])

if __name__ == "__main__":
    demo.launch()
