"""
Local smoke test — calls model_manager.run() directly (no HTTP).

Usage:
    venv/bin/python test_local.py [--models rapidlatex,pix2tex,...] [--all]

By default, olmocr is skipped (requires vLLM subprocess + ~14 GB download).
Pass --all to include it.
"""
from __future__ import annotations
import argparse
import sys
import tempfile
import traceback
import urllib.request
from pathlib import Path

from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DROPBOX_URL = (
    "https://www.dropbox.com/scl/fi/w8oggxalwipsf46bk62yi/IMG_7268.jpeg"
    "?rlkey=c8809nnfzpeu5tmobx5a2rff3&dl=1"
)

DEFAULT_MODELS = ["rapidlatex", "pix2tex", "got_ocr2", "surya", "texteller"]
ALL_MODELS = DEFAULT_MODELS + ["olmocr"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_image(url: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".jpeg", delete=False)
    print(f"Downloading test image to {tmp.name} …")
    urllib.request.urlretrieve(url, tmp.name)
    print("Download complete.")
    return Path(tmp.name)


def run_model(model_key: str, pil_img: Image.Image) -> tuple[bool, str]:
    """Returns (passed, detail)."""
    # Import here so failures in individual models don't crash the runner
    try:
        from model_manager import manager
        result = manager.run(model_key, pil_img)
        if result and result.strip():
            return True, result.strip()[:120]
        return False, f"Empty output: {result!r}"
    except Exception:
        return False, traceback.format_exc()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Local OCR smoke test")
    parser.add_argument("--models", help="Comma-separated model keys to test")
    parser.add_argument("--all", action="store_true", help="Include olmocr")
    args = parser.parse_args()

    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    elif args.all:
        models = ALL_MODELS
    else:
        models = DEFAULT_MODELS

    img_path = download_image(DROPBOX_URL)
    pil_img = Image.open(img_path).convert("RGB")

    results: list[tuple[str, bool, str]] = []
    for key in models:
        print(f"\n[{key}] Running …", flush=True)
        passed, detail = run_model(key, pil_img)
        status = "PASS" if passed else "FAIL"
        print(f"[{key}] {status}: {detail[:80]}")
        results.append((key, passed, detail))

    # Summary table
    print("\n" + "=" * 60)
    print(f"{'Model':<15} {'Result':<8} Detail")
    print("-" * 60)
    any_fail = False
    for key, passed, detail in results:
        tag = "PASS" if passed else "FAIL"
        if not passed:
            any_fail = True
        print(f"{key:<15} {tag:<8} {detail[:36]}")
    print("=" * 60)

    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
