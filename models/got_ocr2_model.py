"""GOT-OCR2 model — ~2500 MiB VRAM. Requires trust_remote_code=True."""
from __future__ import annotations
import tempfile
import os
from PIL import Image

_model = None
_tokenizer = None


def load():
    global _model, _tokenizer
    import torch
    from transformers import AutoTokenizer, AutoModel

    model_id = "ucaslcl/GOT-OCR2_0"
    _tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True
    )
    _model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    _model.eval()


def unload():
    global _model, _tokenizer
    _model = None
    _tokenizer = None
    import torch
    torch.cuda.empty_cache()


def run(image: Image.Image) -> str:
    if _model is None:
        load()

    # GOT-OCR2 expects a file path
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
        image.save(tmp_path)

    try:
        result = _model.chat(
            _tokenizer,
            tmp_path,
            ocr_type="format",
        )
    finally:
        os.unlink(tmp_path)

    return result
