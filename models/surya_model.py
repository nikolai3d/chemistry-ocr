"""Surya OCR model — ~3500 MiB VRAM. Uses predictor-based API (surya >= 0.7)."""
from __future__ import annotations
from PIL import Image

_det_predictor = None
_rec_predictor = None


def load():
    global _det_predictor, _rec_predictor
    from surya.detection import DetectionPredictor
    from surya.recognition import RecognitionPredictor

    _det_predictor = DetectionPredictor()
    _rec_predictor = RecognitionPredictor()


def unload():
    global _det_predictor, _rec_predictor
    _det_predictor = None
    _rec_predictor = None
    import torch
    torch.cuda.empty_cache()


def run(image: Image.Image) -> str:
    if _det_predictor is None:
        load()
    results = _rec_predictor([image], [["en"]], det_predictor=_det_predictor)
    lines = []
    for page in results:
        for line in page.text_lines:
            lines.append(line.text)
    return "\n".join(lines)
