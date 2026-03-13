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


def _image_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def run(image: Image.Image) -> str:
    load()

    b64 = _image_to_b64(image)

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
                    {
                        "type": "text",
                        "text": (
                            "Extract all text and mathematical formulas from this image. "
                            "Return LaTeX for all math expressions."
                        ),
                    },
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
    return data["choices"][0]["message"]["content"]
