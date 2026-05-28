# -*- coding: utf-8 -*-
"""Cleanup-owned AI inpainting backend.

Renderer code may keep compatibility wrappers, but Phase 5 cleanup execution
must call this module so inpainting ownership stays inside the cleanup family.
"""

from __future__ import annotations

import json
import os
import time
from functools import lru_cache

from app.config.defaults import IOPAINT_ANIME_MANGA_BIG_LAMA

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    Image = None


FIXED_CLEANUP_INPAINT_MODEL_ID = IOPAINT_ANIME_MANGA_BIG_LAMA
FIXED_CLEANUP_INPAINT_MODEL_NAME = "SimpleLama(iopaint/anime-manga-big-lama)"
FIXED_CLEANUP_INPAINT_MODEL_RELATIVE_PATH = (
    "models",
    "inpaint",
    "iopaint",
    "anime-manga-big-lama.pt",
)
FIXED_CLEANUP_INPAINT_SELECTION_POLICY = "fixed_cleanup_iopaint_model"


def _page014_timeout_diag_enabled() -> bool:
    return str(os.environ.get("MT_PAGE014_TIMEOUT_DIAGNOSTIC") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _cleanup_perf_contract_diag_enabled() -> bool:
    return str(os.environ.get("MT_CLEANUP_PERF_CONTRACT_DIAGNOSTIC") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _cleanup_perf_contract_json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _cleanup_perf_contract_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_cleanup_perf_contract_json_safe(item) for item in list(value)[:80]]
    shape = getattr(value, "shape", None)
    if shape is not None:
        return {"shape": [int(item) for item in tuple(shape)]}
    return str(value)


def _cleanup_perf_contract_checkpoint(stage: str, event: str, **fields) -> None:
    if not _cleanup_perf_contract_diag_enabled():
        return
    try:
        debug_dir = str(os.environ.get("MT_DEBUG_DIR") or "")
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            path = os.path.join(debug_dir, "cleanup_perf_contract_checkpoints.jsonl")
        else:
            path = os.path.abspath("cleanup_perf_contract_checkpoints.jsonl")
        payload = {
            "ts": time.time(),
            "monotonic": time.monotonic(),
            "module": "app.pipeline.cleanup_inpainting",
            "stage": stage,
            "event": event,
        }
        payload.update(_cleanup_perf_contract_json_safe(fields))
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception:
        return


def _page014_timeout_checkpoint(stage: str, event: str, **fields) -> None:
    _cleanup_perf_contract_checkpoint(stage, event, **fields)
    if not _page014_timeout_diag_enabled():
        return
    try:
        debug_dir = str(os.environ.get("MT_DEBUG_DIR") or "")
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            path = os.path.join(debug_dir, "page014_timeout_checkpoints.jsonl")
        else:
            path = os.path.abspath("page014_timeout_checkpoints.jsonl")
        payload = {
            "ts": time.time(),
            "module": "app.pipeline.cleanup_inpainting",
            "stage": stage,
            "event": event,
        }
        payload.update(fields)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception:
        return


def clear_model_cache() -> None:
    """Clear the cleanup inpainting model cache."""

    _load_lama_model.cache_clear()


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resolve_cleanup_inpaint_model(model_id: str = FIXED_CLEANUP_INPAINT_MODEL_ID) -> dict[str, str]:
    """Resolve the one authorized local cleanup model.

    ``model_id`` is retained only as requested-model provenance. Cleanup
    inpainting is intentionally fixed to the vetted iopaint TorchScript model
    so UI/config strings, absolute paths, or legacy candidate IDs cannot switch
    the production cleanup backend.
    """

    root = _repo_root()
    requested = str(model_id or "").strip()
    fixed_path = os.path.join(root, *FIXED_CLEANUP_INPAINT_MODEL_RELATIVE_PATH)
    return {
        "requested_model_id": requested,
        "configured_model_id": FIXED_CLEANUP_INPAINT_MODEL_ID,
        "selection_policy": FIXED_CLEANUP_INPAINT_SELECTION_POLICY,
        "actual_model_name": FIXED_CLEANUP_INPAINT_MODEL_NAME,
        "actual_model_path": fixed_path,
        "model_available": os.path.exists(fixed_path),
        "ignored_requested_model": requested not in {"", FIXED_CLEANUP_INPAINT_MODEL_ID},
    }


@lru_cache(maxsize=4)
def _load_lama_model(device: str, model_path: str = ""):
    """Load the LaMa model for cleanup-owned inpainting."""

    import torch

    try:
        from app.third_party.simple_lama_inpainting import SimpleLama
    except ImportError:
        try:
            from simple_lama_inpainting import SimpleLama
        except ImportError as exc:
            raise RuntimeError(
                "AI inpainting runtime is unavailable. "
                "The vendored SimpleLama wrapper could not be imported."
            ) from exc

    if not model_path:
        raise RuntimeError("fixed cleanup inpaint model path required")
    if not os.path.exists(model_path):
        raise RuntimeError(f"fixed cleanup inpaint model missing: {model_path}")

    print(f"[Cleanup Inpaint] Loading SimpleLama model on {device}: {model_path}")
    _page014_timeout_checkpoint("cleanup_inpaint_model", "load_start", device=device, model_path=model_path)
    old_model_path = os.environ.get("LAMA_MODEL")
    os.environ["LAMA_MODEL"] = model_path
    try:
        lama = SimpleLama(device=torch.device(device))
    finally:
        if old_model_path is None:
            os.environ.pop("LAMA_MODEL", None)
        else:
            os.environ["LAMA_MODEL"] = old_model_path
    print("[Cleanup Inpaint] SimpleLama model loaded successfully")
    _page014_timeout_checkpoint("cleanup_inpaint_model", "load_end", device=device, model_path=model_path)
    return lama


def ai_inpaint_cleanup(
    image,
    mask,
    use_gpu: bool = True,
    model_id: str = FIXED_CLEANUP_INPAINT_MODEL_ID,
):
    """Perform cleanup-owned AI inpainting using LaMa."""

    started = time.time()
    _page014_timeout_checkpoint(
        "cleanup_ai_inpaint",
        "start",
        use_gpu=use_gpu,
        model_id=model_id,
        image_size=getattr(image, "size", None),
        mask_shape=getattr(mask, "shape", None),
    )
    if Image is None:
        raise RuntimeError("Pillow is not installed.")

    try:
        import cv2
        import numpy as np
    except ImportError:
        cv2 = None
        np = None

    if cv2 is None or np is None:
        raise RuntimeError("cv2 and numpy are required for AI inpainting")

    kernel_size = max(5, int(max(mask.shape) * 0.005))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=2)

    device = "cuda" if use_gpu else "cpu"
    model_info = resolve_cleanup_inpaint_model(model_id)
    actual_model_path = model_info.get("actual_model_path", "")

    try:
        lama = _load_lama_model(device, actual_model_path)
    except Exception as exc:
        print(f"[Cleanup Inpaint] Failed to load LaMa model: {exc}")
        raise

    mask_image = Image.fromarray(dilated_mask).convert("L")
    bbox = mask_image.getbbox()
    if not bbox:
        _page014_timeout_checkpoint(
            "cleanup_ai_inpaint",
            "end",
            backend="none",
            reason="empty_mask_bbox",
            elapsed_ms=round((time.time() - started) * 1000.0, 3),
        )
        return image

    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    pad = max(32, int(max(w, h) * 0.2))
    cx0 = max(0, x0 - pad)
    cy0 = max(0, y0 - pad)
    cx1 = min(image.width, x1 + pad)
    cy1 = min(image.height, y1 + pad)

    crop_w = cx1 - cx0
    crop_h = cy1 - cy0
    crop_img = image.crop((cx0, cy0, cx1, cy1))
    crop_mask = mask_image.crop((cx0, cy0, cx1, cy1))

    print(f"[Cleanup Inpaint] Processing region: {crop_w}x{crop_h}")
    _page014_timeout_checkpoint(
        "cleanup_ai_inpaint",
        "crop",
        device=device,
        crop_bbox=[cx0, cy0, cx1, cy1],
        crop_width=crop_w,
        crop_height=crop_h,
    )
    result = lama(crop_img, crop_mask)

    if result.size != (crop_w, crop_h):
        print(f"[Cleanup Inpaint] Resizing result from {result.size} to {(crop_w, crop_h)}")
        result = result.resize((crop_w, crop_h), Image.LANCZOS)

    if crop_mask.size != result.size:
        crop_mask = crop_mask.resize(result.size, Image.NEAREST)

    out_crop = Image.composite(result, crop_img, crop_mask)
    out = image.copy()
    out.paste(out_crop, (cx0, cy0))

    print("[Cleanup Inpaint] Success")
    _page014_timeout_checkpoint(
        "cleanup_ai_inpaint",
        "end",
        backend="simple_lama",
        device=device,
        crop_bbox=[cx0, cy0, cx1, cy1],
        crop_width=crop_w,
        crop_height=crop_h,
        elapsed_ms=round((time.time() - started) * 1000.0, 3),
    )
    return out


def ai_inpaint(
    image,
    mask,
    use_gpu: bool = True,
    model_id: str = FIXED_CLEANUP_INPAINT_MODEL_ID,
):
    """Compatibility alias for cleanup-owned callers."""

    return ai_inpaint_cleanup(image, mask, use_gpu=use_gpu, model_id=model_id)
