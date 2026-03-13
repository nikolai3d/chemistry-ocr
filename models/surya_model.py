"""Surya OCR model — ~3500 MiB VRAM."""
from __future__ import annotations
from PIL import Image

_det_model = None
_det_processor = None
_rec_model = None
_rec_processor = None


def load():
    global _det_model, _det_processor, _rec_model, _rec_processor
    from surya.model.detection.model import load_model as load_det_model
    from surya.model.detection.processor import load_processor as load_det_processor
    from surya.model.recognition.model import load_model as load_rec_model
    from surya.model.recognition.processor import load_processor as load_rec_processor

    _det_model = load_det_model()
    _det_processor = load_det_processor()
    _rec_model = load_rec_model()
    _rec_processor = load_rec_processor()


def unload():
    global _det_model, _det_processor, _rec_model, _rec_processor
    _det_model = None
    _det_processor = None
    _rec_model = None
    _rec_processor = None
    import torch
    torch.cuda.empty_cache()


def run(image: Image.Image) -> str:
    if _det_model is None:
        load()
    from surya.ocr import run_ocr
    langs = ["en"]
    predictions = run_ocr(
        [image], [langs], _det_model, _det_processor, _rec_model, _rec_processor
    )
    lines = []
    for page in predictions:
        for line in page.text_lines:
            lines.append(line.text)
    return "\n".join(lines)
