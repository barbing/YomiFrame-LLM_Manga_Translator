# -*- coding: utf-8 -*-
"""Deprecated renderer-package compatibility wrapper for cleanup inpainting."""

from __future__ import annotations

from app.pipeline.cleanup_inpainting import ai_inpaint_cleanup, clear_model_cache


def ai_inpaint(image, mask, use_gpu: bool = True, model_id: str = "dreMaz/AnimeMangaInpainting"):
    """Compatibility wrapper; active cleanup ownership lives in app.pipeline."""

    return ai_inpaint_cleanup(image, mask, use_gpu=use_gpu, model_id=model_id)
