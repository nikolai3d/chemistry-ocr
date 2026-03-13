# OCR/LaTeX Multi-Model Service — Replication Guide

Standalone guide. No prior context needed. Tested on Ubuntu 24.04, RTX 2080 Ti 11 GB, CUDA driver 590 (CUDA 13.1).

---

## What This Is

A Gradio web app + REST API (port 7860) that exposes six open-source OCR/LaTeX
recognition models behind a unified upload-and-click UI and a JSON HTTP API.
Designed for digitizing chemistry equations, math formulas, and handwritten notes.

REST endpoints:
- `POST /api/ocr` — `multipart/form-data` with `model=<key>` and `image=<file>`; returns `{"latex": "...","model": "..."}`
- `GET /api/models` — list available model keys and VRAM budgets
- `GET /api/status` — current load/eviction state from ModelManager

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
├── app.py                    # Gradio UI + FastAPI REST API
├── model_manager.py          # lazy loader + LRU eviction
├── models/
│   ├── __init__.py
│   ├── rapidlatex_model.py
│   ├── pix2tex_model.py
│   ├── surya_model.py
│   ├── got_ocr2_model.py
│   ├── texteller_model.py    # spawns texteller_worker.py in venv-texteller
│   ├── texteller_worker.py   # standalone worker: reads image, prints JSON
│   └── olmocr_model.py       # subprocess → vLLM on :8765
├── test_local.py             # direct model_manager smoke test (no HTTP)
├── test_api.py               # HTTP smoke test against /api/ocr
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

### 9. TexTeller: stateless subprocess worker

The old HTTP-server + CLI-fallback approach for TexTeller was replaced with a
simple stateless subprocess. `texteller_model.py` now calls
`models/texteller_worker.py` directly inside `venv-texteller` for each
inference request. The worker prints a single JSON line (`{"latex": "..."}` or
`{"error": "..."}`) and exits. This is more reliable — no port management,
no server-up polling, no fallback logic.

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

## Recent Changes and Fixes

### REST API added (`app.py`)

The service now runs under **uvicorn** with a FastAPI app. Gradio is mounted
at `/` via `gr.mount_gradio_app()`. This approach keeps the `/api/*` routes
alive through Gradio's internal `App.create_app()` call, which would have
replaced them if we had patched `demo.app` directly.

Three new endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/api/ocr` | POST | Run OCR. `model` (form field) + `image` (file upload). Returns `{"latex":"...","model":"..."}` |
| `/api/models` | GET | List model keys and VRAM budgets |
| `/api/status` | GET | ModelManager load/eviction state |

### GOT-OCR2 model class fix (`models/got_ocr2_model.py`)

`AutoModelForCausalLM` → `AutoModel`. GOT-OCR2 uses a custom encoder-decoder
architecture; the `CausalLM` class rejected it at load time.

### OlmOCR three fixes (`models/olmocr_model.py`)

1. **VRAM headroom**: added `--gpu-memory-utilization 0.87` and
   `--max-num-seqs 32` to the vLLM subprocess. Without these, the sampler
   warmup on an 11 GB card allocates ~446 MiB for 256 dummy sequences and
   hits OOM.

2. **Image resize**: new `_resize_for_model()` caps images at ~1 003 520 px
   (1280 × 28²) and rounds dimensions to the nearest 28-pixel Qwen2-VL patch
   boundary before sending to vLLM. Prevents token-budget overflow for
   high-resolution scans.

3. **Prompt + response parsing**: the prompt now instructs the model to use
   `\(…\)` / `\[…\]` LaTeX delimiters and include page dimensions. Response
   parsing now extracts `natural_text` from the JSON that olmOCR returns
   (previously the raw JSON string was returned as the result).

### RapidLaTeXOCR API fix (`models/rapidlatex_model.py`)

Class renamed `LatexOCR` → `LaTeXOCR` to match the current `rapid-latex-ocr`
package. The model now receives a **numpy array** instead of a PIL image,
as required by the updated API.

### Surya predictor-based API (`models/surya_model.py`)

Updated for **surya ≥ 0.7**. The old `load_model` / `load_processor` +
`run_ocr()` function no longer exists. Replaced with `DetectionPredictor` +
`RecognitionPredictor` objects, called as `_rec_predictor([image], [["en"]], det_predictor=_det_predictor)`.

### TexTeller stateless worker (`models/texteller_model.py` + `models/texteller_worker.py`)

Replaced the HTTP-server-with-CLI-fallback approach with a simple stateless
subprocess. For each inference request `texteller_model.py` spawns
`venv-texteller/bin/python models/texteller_worker.py <tmp.png>`. The worker
loads the model, runs inference, and prints a JSON result line. The caller
scans stdout bottom-up for the first `{`-prefixed line and parses it. No ports,
no polling, no fallback branches.

### Smoke tests (`test_local.py`, `test_api.py`)

- `test_local.py` — calls `model_manager.run()` directly (no HTTP needed)
- `test_api.py` — POSTs to `/api/ocr` for each model; supports `--host`,
  `--port`, `--models`, `--all` flags. Both download a test image from Dropbox
  and print a pass/fail table.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `No module named 'venv'` | `python3.12-venv` not installed | `sudo apt install python3.12-venv` |
| `CUDA error: no kernel image` | Wrong PyTorch wheel (cu121 vs cu126) | Reinstall torch with `--index-url …/cu126` |
| `undefined symbol: …` in venv-olmocr | torch pre-installed before vLLM | Delete venv-olmocr, recreate without torch |
| OlmOCR timeout after 300s | vLLM failed to start (OOM or driver issue) | Check `journalctl -u ocr-service`, check `nvidia-smi` |
| TexTeller returns no JSON output | Worker crashed before printing result | Check stderr in log; run `venv-texteller/bin/python models/texteller_worker.py <img>` manually |
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
