"""Vendored compatibility copy of simple-lama-inpainting 0.1.2.

The upstream package is Apache-2.0 licensed and its runtime code is tiny, but
its PyPI metadata still pins Pillow<10. This repo runs with the modern
Torch/Transformers/MangaOCR stack, so we vendor the runtime wrapper here and
avoid the stale packaging constraint while preserving the same inference path.
"""

from app.third_party.simple_lama_inpainting.model import SimpleLama

__all__ = ["SimpleLama"]
