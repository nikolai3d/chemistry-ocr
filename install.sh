#!/usr/bin/env bash
# install.sh — OCR/LaTeX Multi-Model Service installer
# Idempotent: safe to re-run. See ~/INSTALL-OCR-SERVICE.md for full details.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="ocr-service"
HF_HOME="${HOME}/.cache/huggingface"
PYTORCH_INDEX="https://download.pytorch.org/whl/cu126"

log() { echo "[install.sh] $*"; }
err() { echo "[install.sh] ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
log "Installing system packages…"
sudo apt-get update -qq
sudo apt-get install -y python3.12-venv python3.12-dev libgl1 libglib2.0-0 wget ninja-build

# ---------------------------------------------------------------------------
# 2. UFW rule — allow LAN access to Gradio
# ---------------------------------------------------------------------------
if command -v ufw &>/dev/null; then
  log "Adding UFW rule for port 7860…"
  sudo ufw allow from 192.168.1.0/24 to any port 7860 proto tcp || true
fi

# ---------------------------------------------------------------------------
# 3. Main venv
# ---------------------------------------------------------------------------
VENV="${SCRIPT_DIR}/venv"
if [ ! -d "${VENV}" ]; then
  log "Creating main venv…"
  python3.12 -m venv "${VENV}"
fi

log "Installing PyTorch (cu126) in main venv…"
"${VENV}/bin/pip" install --quiet --upgrade pip
"${VENV}/bin/pip" install --quiet \
  torch torchvision torchaudio \
  --index-url "${PYTORCH_INDEX}"

log "Installing main requirements…"
"${VENV}/bin/pip" install --quiet -r "${SCRIPT_DIR}/requirements.txt"

# Smoke test main venv
log "Smoke-testing main venv…"
"${VENV}/bin/python" -c "import gradio, torch, PIL; print('  gradio, torch, PIL OK')"
"${VENV}/bin/python" -c "import pix2tex; print('  pix2tex OK')" || log "  WARNING: pix2tex import failed — check requirements"
"${VENV}/bin/python" -c "from rapid_latex_ocr import LatexOCR; print('  rapid_latex_ocr OK')" || log "  WARNING: rapid_latex_ocr import failed"
"${VENV}/bin/python" -c "import surya; print('  surya OK')" || log "  WARNING: surya import failed"

# ---------------------------------------------------------------------------
# 4. venv-texteller (pins transformers==4.47)
# ---------------------------------------------------------------------------
VENV_TX="${SCRIPT_DIR}/venv-texteller"
if [ ! -d "${VENV_TX}" ]; then
  log "Creating venv-texteller…"
  python3.12 -m venv "${VENV_TX}"
fi

log "Installing PyTorch (cu126) in venv-texteller…"
"${VENV_TX}/bin/pip" install --quiet --upgrade pip
"${VENV_TX}/bin/pip" install --quiet \
  torch torchvision torchaudio \
  --index-url "${PYTORCH_INDEX}"

log "Installing texteller requirements…"
"${VENV_TX}/bin/pip" install --quiet -r "${SCRIPT_DIR}/requirements-texteller.txt"

log "Smoke-testing venv-texteller…"
"${VENV_TX}/bin/python" -c "import texteller; print('  texteller OK')" || log "  WARNING: texteller import failed"

# ---------------------------------------------------------------------------
# 5. venv-olmocr (do NOT pre-install torch — vLLM ships its own)
# ---------------------------------------------------------------------------
VENV_OLM="${SCRIPT_DIR}/venv-olmocr"
if [ ! -d "${VENV_OLM}" ]; then
  log "Creating venv-olmocr…"
  python3.12 -m venv "${VENV_OLM}"
fi

log "Installing olmocr[gpu] + vLLM (this compiles CUDA kernels — may take 10-20 min)…"
"${VENV_OLM}/bin/pip" install --quiet --upgrade pip
"${VENV_OLM}/bin/pip" install --quiet "olmocr[gpu]"

log "Smoke-testing venv-olmocr…"
"${VENV_OLM}/bin/python" -c "import vllm; print('  vllm OK')" || log "  WARNING: vllm import failed"

# ---------------------------------------------------------------------------
# 6. Download MathJax locally
# ---------------------------------------------------------------------------
MATHJAX_PATH="${SCRIPT_DIR}/static/mathjax.min.js"
if [ ! -f "${MATHJAX_PATH}" ]; then
  log "Downloading MathJax locally…"
  mkdir -p "${SCRIPT_DIR}/static"
  wget -q -O "${MATHJAX_PATH}" \
    "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" \
    || log "  WARNING: MathJax download failed — will use CDN at runtime"
fi

# ---------------------------------------------------------------------------
# 7. Systemd service
# ---------------------------------------------------------------------------
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
log "Installing systemd service…"
sudo cp "${SCRIPT_DIR}/${SERVICE_NAME}.service" "${SERVICE_FILE}"
sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}"
sudo systemctl restart "${SERVICE_NAME}"

log ""
log "==========================================="
log " Install complete!"
log " Service: systemctl status ${SERVICE_NAME}"
log " UI:      http://192.168.1.191:7860"
log " Logs:    journalctl -u ${SERVICE_NAME} -f"
log "==========================================="
