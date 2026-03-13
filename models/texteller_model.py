"""TexTeller 3.0 — ~2000 MiB VRAM. Runs in venv-texteller subprocess."""
from __future__ import annotations
import subprocess
import sys
import os
import tempfile
import base64
import json
import time
import threading
import requests
from PIL import Image

_proc = None
_proc_lock = threading.Lock()
_PORT = 8766
_VENV = os.path.expanduser("~/ocr-service/venv-texteller")


def _python():
    return os.path.join(_VENV, "bin", "python")


def load():
    global _proc
    with _proc_lock:
        if _proc is not None and _proc.poll() is None:
            return  # already running

        env = os.environ.copy()
        env["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

        # Try serve mode first
        _proc = subprocess.Popen(
            [_python(), "-m", "texteller.serve", "--port", str(_PORT), "--host", "127.0.0.1"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait up to 30s for server to come up
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                requests.get(f"http://127.0.0.1:{_PORT}/health", timeout=1)
                return  # server up
            except Exception:
                if _proc.poll() is not None:
                    # Server died — fall through to CLI mode
                    _proc = None
                    return
                time.sleep(0.5)

        # Timeout — kill and fall back to CLI
        _proc.terminate()
        _proc = None


def unload():
    global _proc
    with _proc_lock:
        if _proc is not None:
            _proc.terminate()
            try:
                _proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _proc.kill()
            _proc = None


def _run_via_server(image_path: str) -> str:
    resp = requests.post(
        f"http://127.0.0.1:{_PORT}/predict",
        json={"image_path": image_path},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json().get("latex", "")


def _run_via_cli(image_path: str) -> str:
    env = os.environ.copy()
    env["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
    result = subprocess.run(
        [_python(), "-m", "texteller", "infer", "--image", image_path],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"TexTeller CLI error: {result.stderr}")
    return result.stdout.strip()


def run(image: Image.Image) -> str:
    load()  # ensure server started (or will use CLI)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
        image.save(tmp_path)

    try:
        if _proc is not None and _proc.poll() is None:
            return _run_via_server(tmp_path)
        else:
            return _run_via_cli(tmp_path)
    finally:
        os.unlink(tmp_path)
