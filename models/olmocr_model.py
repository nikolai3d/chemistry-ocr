"""OlmOCR-2 7B — ~9500 MiB VRAM. Runs vLLM server in venv-olmocr subprocess."""
from __future__ import annotations
import subprocess
import os
import base64
import io
import time
import threading
import requests
from PIL import Image

_proc = None
_proc_lock = threading.Lock()
_PORT = 8765
_VENV = os.path.expanduser("~/ocr-service/venv-olmocr")
_MODEL = "allenai/olmOCR-7B-0225-preview"


def _python():
    return os.path.join(_VENV, "bin", "python")


def load():
    global _proc
    with _proc_lock:
        if _proc is not None and _proc.poll() is None:
            return  # already running

        env = os.environ.copy()
        env["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
        env["CUDA_VISIBLE_DEVICES"] = "0"
        # Triton needs CUDA_HOME to find libcuda stubs for JIT compilation
        env.setdefault("CUDA_HOME", "/usr")

        _proc = subprocess.Popen(
            [
                _python(), "-m", "vllm.entrypoints.openai.api_server",
                "--model", _MODEL,
                "--quantization", "bitsandbytes",
                "--max-model-len", "4096",
                # RTX 2080 Ti has 11264 MiB; the main OCR process holds ~500 MiB
                # CUDA context. Lower gpu_memory_utilization to leave room for
                # the sampler warmup allocation (~446 MiB for 256 dummy seqs).
                # Also lower max_num_seqs to reduce warmup batch size.
                "--gpu-memory-utilization", "0.87",
                "--max-num-seqs", "32",
                "--port", str(_PORT),
                "--host", "127.0.0.1",
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait up to 900s — FlashInfer JIT-compiles CUDA kernels on first start
        deadline = time.time() + 900
        while time.time() < deadline:
            try:
                resp = requests.get(f"http://127.0.0.1:{_PORT}/health", timeout=2)
                if resp.status_code == 200:
                    return
            except Exception:
                pass
            if _proc.poll() is not None:
                stdout, stderr = _proc.communicate()
                raise RuntimeError(
                    f"vLLM server exited early.\nstdout: {stdout.decode()}\nstderr: {stderr.decode()}"
                )
            time.sleep(2)

        _proc.terminate()
        _proc = None
        raise TimeoutError("OlmOCR vLLM server did not start within 900s")


def unload():
    global _proc
    with _proc_lock:
        if _proc is not None:
            _proc.terminate()
            try:
                _proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                _proc.kill()
            _proc = None


def _resize_for_model(image: Image.Image) -> Image.Image:
    """Resize image so total pixels ≤ MAX_PIXELS to stay within max-model-len."""
    MAX_PIXELS = 1_003_520  # 1280 × 28² — produces ≤ ~1280 image tokens
    w, h = image.size
    if w * h <= MAX_PIXELS:
        return image
    import math
    scale = math.sqrt(MAX_PIXELS / (w * h))
    new_w = int(w * scale)
    new_h = int(h * scale)
    # Round to nearest multiple of 28 (Qwen2-VL patch size)
    new_w = max(28, (new_w // 28) * 28)
    new_h = max(28, (new_h // 28) * 28)
    return image.resize((new_w, new_h), Image.LANCZOS)


def _image_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def run(image: Image.Image) -> str:
    load()

    image = _resize_for_model(image)
    b64 = _image_to_b64(image)

    w, h = image.size
    prompt = (
        f"Attached is the image of one page of a PDF document."
        f"Just return the plain text representation of this document as if you were reading it naturally.\n"
        f"Turn equations and math symbols into a LaTeX representation, make sure to use \\( and \\) as a delimiter for inline math, and \\[ and \\] for block math.\n"
        f"Read any natural handwriting.\n"
        f"Do not hallucinate.\n"
        f"Page width: {w}, Page height: {h}"
    )

    payload = {
        "model": _MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 2048,
        "temperature": 0,
    }

    resp = requests.post(
        f"http://127.0.0.1:{_PORT}/v1/chat/completions",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]

    # olmOCR returns JSON with natural_text field
    try:
        import json as _json
        parsed = _json.loads(content)
        if isinstance(parsed, dict) and "natural_text" in parsed:
            return parsed["natural_text"] or ""
    except Exception:
        pass

    return content
