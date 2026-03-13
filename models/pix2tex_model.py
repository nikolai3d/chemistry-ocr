"""pix2tex (LaTeX-OCR) model — ~300 MiB VRAM."""
from __future__ import annotations
from PIL import Image

_model = None


def load():
    global _model
    from pix2tex.cli import LatexOCR
    _model = LatexOCR()
    return _model


def unload():
    global _model
    _model = None
    import torch
    torch.cuda.empty_cache()


def run(image: Image.Image) -> str:
    if _model is None:
        load()
    return _model(image)
