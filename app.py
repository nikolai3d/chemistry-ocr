"""
OCR/LaTeX Multi-Model Web Service
Gradio UI + REST API on port 7860
"""
from __future__ import annotations
import io
import os
import sys
import traceback
from pathlib import Path

import gradio as gr
from PIL import Image
from fastapi import UploadFile, File, Form
from fastapi.responses import JSONResponse

from model_manager import manager, VRAM_BUDGET

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SERVICE_DIR = Path(__file__).parent
STATIC_DIR = SERVICE_DIR / "static"
MATHJAX_LOCAL = STATIC_DIR / "mathjax.min.js"

MODEL_CHOICES = [
    ("RapidLaTeXOCR (CPU, fast)", "rapidlatex"),
    ("pix2tex (~300 MiB VRAM)", "pix2tex"),
    ("TexTeller 3.0 (~2000 MiB VRAM)", "texteller"),
    ("GOT-OCR2 (~2500 MiB VRAM)", "got_ocr2"),
    ("Surya (~3500 MiB VRAM)", "surya"),
    ("OlmOCR-2 7B (~9500 MiB VRAM — slow first load)", "olmocr"),
]

MATHJAX_SRC = (
    "/static/mathjax.min.js"
    if MATHJAX_LOCAL.exists()
    else "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"
)

MATHJAX_HEAD = f"""
<script>
  window.MathJax = {{
    tex: {{inlineMath: [['$','$'],['\\\\(','\\\\)']]}},
    options: {{skipHtmlTags: ['script','noscript','style','textarea','pre']}}
  }};
</script>
<script src="{MATHJAX_SRC}" async></script>
"""

# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def run_ocr(image: Image.Image | None, model_key: str) -> tuple[str, str]:
    """Returns (latex_text, preview_html)."""
    if image is None:
        return "", "<p style='color:grey'>Upload an image first.</p>"

    if not model_key:
        return "", "<p style='color:red'>Select a model.</p>"

    # OlmOCR loading message
    loading_note = ""
    if model_key == "olmocr" and "olmocr" not in manager.status()["loaded_models"]:
        loading_note = "⏳ OlmOCR first-load: downloading model (~14 GB) and compiling CUDA kernels (10-20 min)…\n\n"

    try:
        latex = manager.run(model_key, image)
    except Exception as e:
        tb = traceback.format_exc()
        return f"ERROR: {e}\n\n{tb}", f"<pre style='color:red'>{e}</pre>"

    latex = latex.strip()
    preview_html = _make_preview(latex)
    return loading_note + latex, preview_html


def _make_preview(latex: str) -> str:
    """Wrap LaTeX in MathJax-renderable HTML."""
    if not latex:
        return "<p style='color:grey'>No output.</p>"

    # Escape for HTML display
    escaped = latex.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    return f"""
{MATHJAX_HEAD}
<div style="padding: 1em; background: #fafafa; border-radius: 6px; font-size: 1.2em;">
  $${latex}$$
</div>
<details style="margin-top:0.5em">
  <summary style="cursor:pointer; color:#555">Raw LaTeX</summary>
  <pre style="background:#f0f0f0; padding:0.5em; overflow-x:auto">{escaped}</pre>
</details>
<script>
  if (window.MathJax && window.MathJax.typesetPromise) {{
    MathJax.typesetPromise();
  }}
</script>
"""


# ---------------------------------------------------------------------------
# Status panel
# ---------------------------------------------------------------------------

def build_status_html() -> str:
    st = manager.status()
    loaded = st["loaded_models"] or ["(none)"]
    vram_used = st["vram_used_mib"]
    vram_free = st["vram_free_mib"]
    total = st["gpu_total_mib"]

    bar_pct = min(100, int(vram_used / total * 100))
    bar_color = "#e74c3c" if bar_pct > 85 else "#f39c12" if bar_pct > 60 else "#2ecc71"

    rows = ""
    for label, key in MODEL_CHOICES:
        status = "✅ loaded" if key in st["loaded_models"] else "—"
        vram = f"{VRAM_BUDGET[key]} MiB" if VRAM_BUDGET[key] > 0 else "CPU"
        rows += f"<tr><td>{label}</td><td>{vram}</td><td>{status}</td></tr>"

    return f"""
<style>
  .status-table {{ border-collapse: collapse; width: 100%; font-size: 0.85em; }}
  .status-table th, .status-table td {{ border: 1px solid #ddd; padding: 4px 8px; text-align: left; }}
  .status-table th {{ background: #f0f0f0; }}
  .vram-bar {{ height: 14px; background: #eee; border-radius: 4px; overflow: hidden; margin: 4px 0; }}
  .vram-fill {{ height: 100%; background: {bar_color}; width: {bar_pct}%; transition: width 0.3s; }}
</style>
<b>GPU VRAM:</b> {vram_used} MiB used / {vram_free} MiB free / {total} MiB total
<div class="vram-bar"><div class="vram-fill"></div></div>
<table class="status-table">
  <tr><th>Model</th><th>VRAM</th><th>Status</th></tr>
  {rows}
</table>
"""


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui():
    with gr.Blocks(title="OCR/LaTeX Service") as demo:
        gr.Markdown("## OCR / LaTeX Recognition Service\nUpload an image containing math or text, select a model, and click **Run**.")

        with gr.Row():
            # Left column — inputs
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Input Image", height=320)
                model_dropdown = gr.Dropdown(
                    choices=[(label, key) for label, key in MODEL_CHOICES],
                    value="rapidlatex",
                    label="Model",
                )
                run_btn = gr.Button("Run OCR", variant="primary")

                gr.Markdown("### GPU Status")
                status_box = gr.HTML(value=build_status_html())

            # Right column — outputs
            with gr.Column(scale=1):
                latex_output = gr.Textbox(
                    label="LaTeX Output",
                    lines=8,
                )
                preview_box = gr.HTML(label="Rendered Preview")

        # Wire up
        run_btn.click(
            fn=run_ocr,
            inputs=[image_input, model_dropdown],
            outputs=[latex_output, preview_box],
        )

        # Auto-refresh status every 5s (Gradio 6+ uses gr.Timer)
        timer = gr.Timer(value=5)
        timer.tick(fn=build_status_html, outputs=[status_box])

    return demo


# ---------------------------------------------------------------------------
# REST API routes
# ---------------------------------------------------------------------------

def add_api_routes(fastapi_app):
    """Mount REST endpoints on the underlying FastAPI app before launch."""

    @fastapi_app.post("/api/ocr")
    async def api_ocr(model: str = Form(...), image: UploadFile = File(...)):
        try:
            data = await image.read()
            pil_img = Image.open(io.BytesIO(data)).convert("RGB")
            latex = manager.run(model, pil_img)
            return {"latex": latex, "model": model}
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": str(e), "model": model, "traceback": traceback.format_exc()},
            )

    @fastapi_app.get("/api/models")
    async def api_models():
        return {"models": [{"key": k, "vram_mib": v} for k, v in VRAM_BUDGET.items()]}

    @fastapi_app.get("/api/status")
    async def api_status():
        return manager.status()


if __name__ == "__main__":
    # Build Gradio UI
    demo = build_ui()

    # Create a standalone FastAPI app so our /api/* routes survive Gradio's
    # internal App.create_app() call (which would replace demo.app if we used
    # the old demo.app approach).
    from fastapi import FastAPI
    import uvicorn

    fastapi_app = FastAPI(title="OCR/LaTeX Service")
    add_api_routes(fastapi_app)

    allowed = [str(STATIC_DIR)] if STATIC_DIR.exists() else None
    gr.mount_gradio_app(
        fastapi_app,
        demo,
        path="/",
        allowed_paths=allowed,
    )

    uvicorn.run(fastapi_app, host="0.0.0.0", port=7860, log_level="info")
