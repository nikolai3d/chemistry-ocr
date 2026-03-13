# OCR/LaTeX Multi-Model Service — Replication Guide

Standalone guide. No prior context needed. Tested on Ubuntu 24.04, RTX 2080 Ti 11 GB, CUDA driver 590 (CUDA 13.1).

---

## What This Is

A single Gradio web app (port 7860) that exposes six open-source OCR/LaTeX
recognition models behind a unified upload-and-click UI. Designed for
digitizing chemistry equations, math formulas, and handwritten notes.

| Model | VRAM | Notes |
|---|---|---|
| RapidLaTeXOCR | 0 (CPU) | Fast ONNX baseline |
| pix2tex | ~300 MiB | Good for clean printed math |
| TexTeller 3.0 | ~2000 MiB | Runs in isolated subprocess |
| GOT-OCR2 | ~2500 MiB | Multi-page, general OCR |
| Surya | ~3500 MiB | Excellent layout detection |
| OlmOCR-2 7B | ~9500 MiB | Best accuracy; slow first load |

Models are loaded lazily on first use. LRU eviction frees VRAM when switching
models. OlmOCR-2 triggers full GPU eviction before loading.

---

## Prerequisites

- Ubuntu 22.04/24.04
- NVIDIA GPU ≥ 11 GB VRAM (tested RTX 2080 Ti)
- NVIDIA driver ≥ 530 (CUDA 12.x); driver 590 = CUDA 13.1 → use cu126 wheels
- `sudo` access
- Internet access (models download from HuggingFace on first use, ~14 GB for OlmOCR)

---

## Directory Layout

```
~/ocr-service/
├── app.py                    # Gradio UI + dispatch
├── model_manager.py          # lazy loader + LRU eviction
├── models/
│   ├── __init__.py
│   ├── rapidlatex_model.py
│   ├── pix2tex_model.py
│   ├── surya_model.py
│   ├── got_ocr2_model.py
│   ├── texteller_model.py    # subprocess → venv-texteller
│   └── olmocr_model.py       # subprocess → vLLM on :8765
├── requirements.txt
├── requirements-texteller.txt
├── static/
│   └── mathjax.min.js        # downloaded during install
├── install.sh                # run this
├── ocr-service.service       # systemd unit
└── venv/                     # main venv (created by install.sh)
    venv-texteller/           # isolated for transformers==4.47
    venv-olmocr/              # isolated for vLLM (brings its own torch)
```

---

## Why Three Venvs?

Dependency conflicts in the `transformers` package:

| Venv | `transformers` version | Reason |
|---|---|---|
| `venv/` (main) | `>=4.37,<4.47` | GOT-OCR2 + Surya |
| `venv-texteller/` | `==4.47` | TexTeller pins this exactly |
| `venv-olmocr/` | `==4.57.3` | vLLM ships its own; do NOT pre-install torch |

All three share `~/.cache/huggingface` so model weights are downloaded once.

---

## Install

```bash
cd ~/ocr-service
./install.sh
```

The script is idempotent — safe to re-run. It:
1. `apt install python3.12-venv libgl1 libglib2.0-0 wget`
2. Adds UFW rule: `allow from 192.168.1.0/24 to any port 7860 tcp`
3. Creates `venv/`, `venv-texteller/`, `venv-olmocr/`
4. Installs PyTorch with `--index-url https://download.pytorch.org/whl/cu126`
5. Installs packages in each venv
6. Downloads MathJax to `static/mathjax.min.js`
7. Copies + enables + starts systemd unit `ocr-service`

Expected runtime: 10-30 min (vLLM compiles CUDA kernels the first time).

---

## Critical Gotchas

### 1. PyTorch wheel URL: use `cu126`, not `cu121`

NVIDIA driver 590 exposes CUDA 13.1. The `cu121` wheels fail at runtime
with a CUDA version mismatch. Always use:

```bash
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu126
```

### 2. `venv-olmocr`: do NOT pre-install PyTorch

`olmocr[gpu]` depends on `vllm`, which ships its own bundled PyTorch.
Pre-installing a different torch version causes ABI conflicts and cryptic
`undefined symbol` errors at import time.

### 3. `python3.12-venv` must be installed before creating venvs

On Ubuntu 24.04 the `venv` module ships in a separate package:

```bash
sudo apt install python3.12-venv
```

Without this, `python3 -m venv` exits with "ensurepip is not available".

### 4. OlmOCR first load: ~14 GB download + 10-20 min CUDA compilation

On first use the UI shows a loading message. The vLLM subprocess starts
a server on `localhost:8765` and waits up to 5 minutes for it to become
healthy. Subsequent loads are fast (weights cached).

### 5. Ollama VRAM contention

If Ollama has a model loaded when OlmOCR tries to start, the combined VRAM
may exceed 11 GB. The ModelManager checks free VRAM before loading OlmOCR
and raises a clear error. Unload the Ollama model first:

```bash
# List loaded models
curl http://localhost:11434/api/ps

# Pull it into memory to use it, or restart ollama to flush all models
sudo systemctl restart ollama
```

### 6. `timm` version pin

pix2tex requires `timm==0.5.4`. Newer timm versions rename APIs that pix2tex
calls directly. Pin it in `requirements.txt`:

```
timm==0.5.4
```

### 7. UFW + Docker bridge (inherited from Ollama setup)

If Docker is running, `ufw deny <port>` also blocks Docker bridge traffic.
See `~/INSTALL-OLLAMA.md` for the fix pattern (insert allow-from-172.17.0.0/16
*before* the deny rule).

### 8. GOT-OCR2 `trust_remote_code=True`

The model ships a custom architecture not in the transformers core. The flag is
required; without it you get an `UntrustedCode` exception. The model is from
the `ucaslcl` org on HuggingFace.

### 9. TexTeller serve vs CLI fallback

TexTeller 1.x may or may not expose a `serve` subcommand depending on the
installed version. `texteller_model.py` tries the HTTP server first, falls
back to `python -m texteller infer --image <path>` if the server exits early.

---

## Manual Install Steps (if install.sh fails)

```bash
# System deps
sudo apt install python3.12-venv libgl1 libglib2.0-0 wget

# UFW
sudo ufw allow from 192.168.1.0/24 to any port 7860 proto tcp

# --- Main venv ---
python3.12 -m venv ~/ocr-service/venv
~/ocr-service/venv/bin/pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu126
~/ocr-service/venv/bin/pip install -r ~/ocr-service/requirements.txt

# --- venv-texteller ---
python3.12 -m venv ~/ocr-service/venv-texteller
~/ocr-service/venv-texteller/bin/pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu126
~/ocr-service/venv-texteller/bin/pip install -r ~/ocr-service/requirements-texteller.txt

# --- venv-olmocr (NO torch pre-install!) ---
python3.12 -m venv ~/ocr-service/venv-olmocr
~/ocr-service/venv-olmocr/bin/pip install "olmocr[gpu]"

# MathJax
mkdir -p ~/ocr-service/static
wget -O ~/ocr-service/static/mathjax.min.js \
  https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js

# Systemd
sudo cp ~/ocr-service/ocr-service.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ocr-service
sudo systemctl start ocr-service
```

---

## Verification

```bash
# Service status
systemctl status ocr-service

# Tail logs
journalctl -u ocr-service -f

# Open in browser
http://192.168.1.191:7860

# Test: upload a photo of "E = mc²" and try each model
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `No module named 'venv'` | `python3.12-venv` not installed | `sudo apt install python3.12-venv` |
| `CUDA error: no kernel image` | Wrong PyTorch wheel (cu121 vs cu126) | Reinstall torch with `--index-url …/cu126` |
| `undefined symbol: …` in venv-olmocr | torch pre-installed before vLLM | Delete venv-olmocr, recreate without torch |
| OlmOCR timeout after 300s | vLLM failed to start (OOM or driver issue) | Check `journalctl -u ocr-service`, check `nvidia-smi` |
| TexTeller always uses CLI | `texteller serve` not available in this version | Expected; CLI fallback works fine |
| MathJax not rendering | CDN blocked or local file missing | Re-run `wget` command in install notes above |
| Port 7860 unreachable from LAN | UFW not configured | `sudo ufw allow from 192.168.1.0/24 to any port 7860 proto tcp` |

---

## Service Management

```bash
sudo systemctl start ocr-service
sudo systemctl stop ocr-service
sudo systemctl restart ocr-service
journalctl -u ocr-service -f --since "10 min ago"
```

---

## File: `requirements.txt` (main venv)

```
gradio>=4.0.0
numpy<2.0.0
Pillow<11.0.0
timm==0.5.4
requests
pix2tex
surya-ocr
rapid-latex-ocr
transformers>=4.37.0,<4.47
accelerate
sentencepiece
tiktoken
verovio
```

## File: `requirements-texteller.txt`

```
transformers==4.47
texteller
torch
numpy<2.0.0
Pillow<11.0.0
```
