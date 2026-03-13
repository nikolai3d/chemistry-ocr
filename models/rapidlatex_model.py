"""RapidLaTeXOCR model — CPU-only ONNX, no VRAM needed."""
from __future__ import annotations
from PIL import Image

_model = None


def load():
    global _model
    from rapid_latex_ocr import LatexOCR
    _model = LatexOCR()
    return _model


def unload():
    global _model
    _model = None


def run(image: Image.Image) -> str:
    if _model is None:
        load()
    result, _ = _model(image)
    return result
