"""RapidLaTeXOCR model — CPU-only ONNX, no VRAM needed."""
from __future__ import annotations
from PIL import Image

_model = None


def load():
    global _model
    from rapid_latex_ocr import LaTeXOCR
    _model = LaTeXOCR()
    return _model


def unload():
    global _model
    _model = None


def run(image: Image.Image) -> str:
    import numpy as np
    if _model is None:
        load()
    img_array = np.array(image)
    result, _ = _model(img_array)
    return result
