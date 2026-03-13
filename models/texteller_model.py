"""TexTeller 3.0 — ~2000 MiB VRAM. Runs via venv-texteller subprocess."""
from __future__ import annotations
import subprocess
import os
import json
import tempfile
from pathlib import Path
from PIL import Image

_VENV = Path(__file__).parent.parent / "venv-texteller"
_WORKER = Path(__file__).parent / "texteller_worker.py"
_PYTHON = str(_VENV / "bin" / "python")


def load():
    pass  # stateless subprocess — nothing to pre-load


def unload():
    pass  # no persistent process to kill


def run(image: Image.Image) -> str:
    env = os.environ.copy()
    env["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
        image.save(tmp_path)

    try:
        result = subprocess.run(
            [_PYTHON, str(_WORKER), tmp_path],
            capture_output=True,
            text=True,
            env=env,
            timeout=180,
        )
    finally:
        os.unlink(tmp_path)

    # Find JSON line in stdout (worker may print progress to stdout before it)
    stdout = result.stdout.strip()
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                data = json.loads(line)
                if "error" in data:
                    raise RuntimeError(data["error"])
                return data.get("latex", "")
            except json.JSONDecodeError:
                continue

    if result.returncode != 0:
        raise RuntimeError(f"texteller_worker failed (rc={result.returncode}):\n{result.stderr[-2000:]}")
    raise RuntimeError(f"No JSON output from texteller_worker. stdout={stdout[:500]}")
