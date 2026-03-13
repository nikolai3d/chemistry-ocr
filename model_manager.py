"""
ModelManager — lazy loader with LRU eviction and VRAM budgeting.

OlmOCR (7B) requires the full GPU budget; it triggers eviction of all
other GPU models before loading. GPU-free models (rapidlatex) are never
evicted.
"""
from __future__ import annotations
import time
import threading
import subprocess
from typing import Any

VRAM_BUDGET: dict[str, int] = {
    "rapidlatex": 0,
    "pix2tex": 300,
    "texteller": 2000,
    "got_ocr2": 2500,
    "surya": 3500,
    "olmocr": 9500,
}

GPU_TOTAL_MIB = 11264

# Models that run as subprocesses (not in-process torch)
SUBPROCESS_MODELS = {"texteller", "olmocr"}

# Import map — each module exposes load(), unload(), run(image) -> str
_MODULE_MAP = {
    "rapidlatex": "models.rapidlatex_model",
    "pix2tex": "models.pix2tex_model",
    "surya": "models.surya_model",
    "got_ocr2": "models.got_ocr2_model",
    "texteller": "models.texteller_model",
    "olmocr": "models.olmocr_model",
}


class ModelManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._loaded: set[str] = set()
        self._last_used: dict[str, float] = {}
        self._modules: dict[str, Any] = {}

    def _import(self, name: str):
        if name not in self._modules:
            import importlib
            self._modules[name] = importlib.import_module(_MODULE_MAP[name])
        return self._modules[name]

    def _vram_used(self) -> int:
        return sum(VRAM_BUDGET[n] for n in self._loaded)

    def _evict_lru(self, needed_mib: int):
        """Evict GPU models LRU until `needed_mib` MiB is free."""
        gpu_loaded = [n for n in self._loaded if VRAM_BUDGET[n] > 0]
        gpu_loaded.sort(key=lambda n: self._last_used.get(n, 0))  # oldest first

        for name in gpu_loaded:
            free = GPU_TOTAL_MIB - self._vram_used()
            if free >= needed_mib:
                break
            self._unload(name)

    def _unload(self, name: str):
        mod = self._import(name)
        try:
            mod.unload()
        except Exception as e:
            print(f"[ModelManager] Warning: error unloading {name}: {e}")
        self._loaded.discard(name)
        self._last_used.pop(name, None)

    def ensure_loaded(self, name: str):
        with self._lock:
            if name in self._loaded:
                self._last_used[name] = time.time()
                return

            needed = VRAM_BUDGET[name]

            if needed == 0:
                # CPU model — just load
                mod = self._import(name)
                mod.load()
                self._loaded.add(name)
                self._last_used[name] = time.time()
                return

            # OlmOCR needs the full GPU — evict everything
            if name == "olmocr":
                for loaded_name in list(self._loaded):
                    if VRAM_BUDGET[loaded_name] > 0:
                        self._unload(loaded_name)
                # Also check if Ollama is holding VRAM
                free = _get_free_vram_mib()
                if free < needed:
                    raise RuntimeError(
                        f"Not enough free VRAM for OlmOCR (need ~{needed} MiB, "
                        f"only {free} MiB free). "
                        "If Ollama has a model loaded, unload it first."
                    )
            else:
                # Standard LRU eviction
                free = GPU_TOTAL_MIB - self._vram_used()
                if free < needed:
                    self._evict_lru(needed)

            mod = self._import(name)
            mod.load()
            self._loaded.add(name)
            self._last_used[name] = time.time()

    def run(self, name: str, image) -> str:
        self.ensure_loaded(name)
        mod = self._import(name)
        result = mod.run(image)
        with self._lock:
            self._last_used[name] = time.time()
        return result

    def status(self) -> dict:
        """Return current status for the UI status panel."""
        with self._lock:
            loaded = list(self._loaded)
            vram_used = self._vram_used()

        free_vram = _get_free_vram_mib()
        return {
            "loaded_models": loaded,
            "vram_used_mib": vram_used,
            "vram_free_mib": free_vram,
            "gpu_total_mib": GPU_TOTAL_MIB,
        }


def _get_free_vram_mib() -> int:
    """Query actual free VRAM via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
        return int(out.strip())
    except Exception:
        return GPU_TOTAL_MIB  # assume free if query fails


# Singleton
manager = ModelManager()
