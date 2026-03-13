"""
API smoke test — POSTs to /api/ocr for each model via HTTP.

Usage:
    venv/bin/python test_api.py [--host 192.168.1.191] [--port 7860] [--models ...] [--all]

By default, olmocr is skipped. Pass --all to include it.
Requires the service to be running before executing this script.
"""
from __future__ import annotations
import argparse
import sys
import tempfile
import traceback
import urllib.request
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: 'requests' not installed. Run: pip install requests")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DROPBOX_URL = (
    "https://www.dropbox.com/scl/fi/w8oggxalwipsf46bk62yi/IMG_7268.jpeg"
    "?rlkey=c8809nnfzpeu5tmobx5a2rff3&dl=1"
)

DEFAULT_MODELS = ["rapidlatex", "pix2tex", "got_ocr2", "surya", "texteller"]
ALL_MODELS = DEFAULT_MODELS + ["olmocr"]

TIMEOUT_SECONDS = 300  # some models take a while on first load


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_image(url: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".jpeg", delete=False)
    print(f"Downloading test image to {tmp.name} …")
    urllib.request.urlretrieve(url, tmp.name)
    print("Download complete.")
    return Path(tmp.name)


def post_ocr(base_url: str, model_key: str, img_path: Path) -> tuple[bool, str]:
    """POST to /api/ocr; returns (passed, detail)."""
    endpoint = f"{base_url}/api/ocr"
    try:
        with open(img_path, "rb") as f:
            resp = requests.post(
                endpoint,
                data={"model": model_key},
                files={"image": ("image.jpeg", f, "image/jpeg")},
                timeout=TIMEOUT_SECONDS,
            )
        if resp.status_code == 200:
            body = resp.json()
            latex = body.get("latex", "")
            if latex and latex.strip():
                return True, latex.strip()[:120]
            return False, f"Empty latex in response: {body}"
        else:
            body = resp.text[:200]
            return False, f"HTTP {resp.status_code}: {body}"
    except requests.exceptions.Timeout:
        return False, f"Timeout after {TIMEOUT_SECONDS}s"
    except Exception:
        return False, traceback.format_exc()


def check_endpoints(base_url: str):
    """Verify /api/models and /api/status are reachable."""
    for path in ["/api/models", "/api/status"]:
        try:
            r = requests.get(f"{base_url}{path}", timeout=10)
            print(f"  {path} → HTTP {r.status_code}")
        except Exception as e:
            print(f"  {path} → ERROR: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="API smoke test for OCR service")
    parser.add_argument("--host", default="192.168.1.191")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--models", help="Comma-separated model keys")
    parser.add_argument("--all", action="store_true", help="Include olmocr")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    elif args.all:
        models = ALL_MODELS
    else:
        models = DEFAULT_MODELS

    print(f"Target: {base_url}")
    print("Checking utility endpoints …")
    check_endpoints(base_url)

    img_path = download_image(DROPBOX_URL)

    results: list[tuple[str, bool, str]] = []
    for key in models:
        print(f"\n[{key}] POSTing to /api/ocr …", flush=True)
        passed, detail = post_ocr(base_url, key, img_path)
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
