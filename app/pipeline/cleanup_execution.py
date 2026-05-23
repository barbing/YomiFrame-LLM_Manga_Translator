"""Cleanup-owned execution primitives.

This module executes selected cleanup plans without choosing routes or text
admission. AI inpainting ownership lives under app.pipeline, not app.render.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover - optional dependency
    Image = None
    ImageDraw = None

try:
    import cv2
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    cv2 = None
    np = None

from app.pipeline.debug_artifacts import mask_stats


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


def _cleanup_perf_contract_json_safe(value: Any) -> Any:
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


def _cleanup_perf_contract_checkpoint(stage: str, event: str, **fields: Any) -> None:
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
            "module": "app.pipeline.cleanup_execution",
            "stage": stage,
            "event": event,
        }
        payload.update(_cleanup_perf_contract_json_safe(fields))
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception:
        return


def _page014_timeout_checkpoint(stage: str, event: str, **fields: Any) -> None:
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
            "module": "app.pipeline.cleanup_execution",
            "stage": stage,
            "event": event,
        }
        payload.update(fields)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception:
        return


@dataclass(frozen=True)
class CleanupExecutionResult:
    cleaned_image: Any
    backend: str | None = None
    backend_detail: Any = None
    effective_inpaint_mode: str | None = None
    crop_bbox: list[int] | None = None
    crop_area: int | None = None
    mask_ratio: float | None = None
    errors: list[str] = field(default_factory=list)
    fallback_status: str | None = None


def apply_text_removal(
    image: Any,
    text_mask: Any,
    mode: str,
    use_gpu: bool,
    model_id: str = "dreMaz/AnimeMangaInpainting",
    debug_info: dict | None = None,
) -> CleanupExecutionResult:
    started = time.time()
    if text_mask is None:
        _set_cleanup_backend(debug_info, "none", backend_detail="empty_mask")
        _page014_timeout_checkpoint("cleanup_apply_text_removal", "end", backend="none", reason="empty_mask")
        return _result(image, debug_info)
    mode = (mode or "fast").lower()
    if debug_info is not None:
        debug_info["effective_inpaint_mode"] = mode

    if cv2 is None or np is None:
        print("[TextRemoval] No CV2/numpy, using simple white mask")
        _set_cleanup_backend(debug_info, "white_mask", backend_detail="cv2_or_numpy_unavailable")
        _page014_timeout_checkpoint(
            "cleanup_apply_text_removal",
            "end",
            backend="white_mask",
            reason="cv2_or_numpy_unavailable",
            elapsed_ms=round((time.time() - started) * 1000.0, 3),
        )
        return _result(_apply_white_mask(image, text_mask), debug_info)

    mask_pixels = int((text_mask > 0).sum())
    image_area = int(text_mask.shape[0] * text_mask.shape[1])
    mask_ratio = mask_pixels / max(1, image_area)
    stats = mask_stats(text_mask)
    box = _mask_stats_box(stats)
    if debug_info is not None:
        debug_info["mask_pixels"] = mask_pixels
        debug_info["crop_area"] = image_area
        debug_info["mask_ratio"] = round(mask_ratio, 4)
        debug_info["crop_bbox"] = list(box) if box else None
    _page014_timeout_checkpoint(
        "cleanup_apply_text_removal",
        "start",
        mode=mode,
        use_gpu=use_gpu,
        model_id=model_id,
        mask_pixels=mask_pixels,
        image_area=image_area,
        mask_ratio=round(mask_ratio, 6),
        crop_bbox=list(box) if box else None,
    )

    # Always try white bubble fill first (cleanest option)
    bubble_fill = _white_bubble_fill(image, text_mask)
    if bubble_fill is not None:
        print("[TextRemoval] Used WHITE BUBBLE FILL")
        _set_cleanup_backend(debug_info, "white_bubble_fill", backend_detail=None)
        _page014_timeout_checkpoint(
            "cleanup_apply_text_removal",
            "end",
            backend="white_bubble_fill",
            elapsed_ms=round((time.time() - started) * 1000.0, 3),
        )
        return _result(bubble_fill, debug_info)

    # Try uniform fill for simple backgrounds (handles most cases)
    uniform_fill = _apply_uniform_fill(image, text_mask)
    if uniform_fill is not None:
        print("[TextRemoval] Used UNIFORM FILL")
        _set_cleanup_backend(debug_info, "uniform_fill", backend_detail=None)
        _page014_timeout_checkpoint(
            "cleanup_apply_text_removal",
            "end",
            backend="uniform_fill",
            elapsed_ms=round((time.time() - started) * 1000.0, 3),
        )
        return _result(uniform_fill, debug_info)

    # For remaining complex areas: use AI inpainting if enabled, otherwise CV2
    ai_failed = False
    if mode == "ai" and use_gpu:
        try:
            print(f"[TextRemoval] Using AI INPAINT with model: {model_id}")
            _page014_timeout_checkpoint(
                "cleanup_apply_text_removal_ai",
                "start",
                model_id=model_id,
                mask_pixels=mask_pixels,
                image_area=image_area,
            )
            from app.pipeline.cleanup_inpainting import ai_inpaint_cleanup
            result = ai_inpaint_cleanup(image, text_mask, use_gpu=use_gpu, model_id=model_id)
            print("[TextRemoval] AI INPAINT Success")
            _set_cleanup_backend(debug_info, "cleanup_ai_inpaint", backend_detail=model_id)
            _page014_timeout_checkpoint(
                "cleanup_apply_text_removal",
                "end",
                backend="cleanup_ai_inpaint",
                elapsed_ms=round((time.time() - started) * 1000.0, 3),
            )
            return _result(result, debug_info)
        except Exception as e:
            print(f"[TextRemoval] AI INPAINT failed: {e}, falling back to CV2")
            ai_failed = True

    print("[TextRemoval] Used CV2 INPAINT (fallback)")
    cv2_detail = "after_ai_failure" if ai_failed else "fallback"
    # CV2 inpainting with moderate dilation for complex areas
    img_np = np.array(image)
    kernel_size = max(5, int(max(text_mask.shape) * 0.004))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(text_mask, kernel, iterations=2)
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=1)

    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    inpainted = cv2.inpaint(img_bgr, dilated, 5, cv2.INPAINT_NS)
    rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
    _set_cleanup_backend(debug_info, "cv2_inpaint", backend_detail=cv2_detail)
    _page014_timeout_checkpoint(
        "cleanup_apply_text_removal",
        "end",
        backend="cv2_inpaint",
        backend_detail=cv2_detail,
        elapsed_ms=round((time.time() - started) * 1000.0, 3),
    )
    return _result(Image.fromarray(rgb), debug_info)


def apply_local_text_removal(
    image: Any,
    local_mask: Any,
    mode: str,
    use_gpu: bool,
    model_id: str = "dreMaz/AnimeMangaInpainting",
    cleanup_tag: str | None = None,
    debug_info: dict | None = None,
) -> CleanupExecutionResult:
    started = time.time()
    cleanup_tag_normalized = str(cleanup_tag or "").strip().lower()
    _page014_timeout_checkpoint(
        "cleanup_apply_local_text_removal",
        "start",
        cleanup_tag=cleanup_tag_normalized,
        mode=(mode or "fast").lower(),
        use_gpu=use_gpu,
        model_id=model_id,
    )
    if cv2 is None or np is None or local_mask is None:
        _set_cleanup_backend(debug_info, "none", backend_detail="cv2_numpy_or_mask_unavailable")
        _page014_timeout_checkpoint(
            "cleanup_apply_local_text_removal",
            "end",
            backend="none",
            reason="cv2_numpy_or_mask_unavailable",
            elapsed_ms=round((time.time() - started) * 1000.0, 3),
        )
        return _result(image, debug_info)
    ys, xs = np.where(local_mask > 0)
    if ys.size == 0 or xs.size == 0:
        _set_cleanup_backend(debug_info, "none", backend_detail="empty_local_mask")
        _page014_timeout_checkpoint(
            "cleanup_apply_local_text_removal",
            "end",
            backend="none",
            reason="empty_local_mask",
            elapsed_ms=round((time.time() - started) * 1000.0, 3),
        )
        return _result(image, debug_info)
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    mask_h, mask_w = local_mask.shape[:2]
    recovered_anchor_local = cleanup_tag_normalized == "speech_recovered_anchor_glyph_local"
    side_caption_local = cleanup_tag_normalized == "side_caption_glyph_local"
    speech_glyph_local = cleanup_tag_normalized == "speech_glyph_local"
    glyph_stroke_local = cleanup_tag_normalized in {"glyph_stroke_local", "background_narration_glyph_local"}
    force_strong_local = cleanup_tag_normalized in {"caption_strong", "speech_strong"} or recovered_anchor_local
    if force_strong_local:
        pad = max(12, int(max(x1 - x0, y1 - y0) * 0.18))
    else:
        pad = max(6, int(max(x1 - x0, y1 - y0) * 0.10))
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(mask_w, x1 + pad)
    y1 = min(mask_h, y1 + pad)
    if x1 <= x0 or y1 <= y0:
        _page014_timeout_checkpoint(
            "cleanup_apply_local_text_removal",
            "end",
            backend="none",
            reason="invalid_crop_bounds",
            elapsed_ms=round((time.time() - started) * 1000.0, 3),
        )
        return _result(image, debug_info)
    crop_mask = local_mask[y0:y1, x0:x1]
    if crop_mask.size == 0 or not np.any(crop_mask):
        _set_cleanup_backend(debug_info, "none", backend_detail="empty_crop_mask")
        _page014_timeout_checkpoint(
            "cleanup_apply_local_text_removal",
            "end",
            backend="none",
            reason="empty_crop_mask",
            elapsed_ms=round((time.time() - started) * 1000.0, 3),
        )
        return _result(image, debug_info)
    crop_img = image.crop((x0, y0, x1, y1))
    crop_h, crop_w = crop_mask.shape[:2]
    crop_area = crop_mask.shape[0] * crop_mask.shape[1]
    mask_pixels = int((crop_mask > 0).sum())
    local_mode = (mode or "fast").lower()
    requested_mode = local_mode
    if not force_strong_local and (crop_area > 220000 or mask_pixels > 18000):
        local_mode = "fast"
    elif not force_strong_local and local_mode == "ai" and (crop_area > 140000 or mask_pixels > 9000):
        local_mode = "fast"
    mask_ratio = mask_pixels / max(1, crop_area)
    if debug_info is not None:
        debug_info["requested_inpaint_mode"] = requested_mode
        debug_info["effective_inpaint_mode"] = local_mode
        debug_info["cleanup_tag"] = cleanup_tag_normalized
        debug_info["crop_bbox"] = [x0, y0, x1, y1]
        debug_info["crop_area"] = int(crop_area)
        debug_info["mask_pixels"] = int(mask_pixels)
        debug_info["mask_ratio"] = round(mask_ratio, 4)
    _page014_timeout_checkpoint(
        "cleanup_apply_local_text_removal",
        "crop",
        cleanup_tag=cleanup_tag_normalized,
        requested_mode=requested_mode,
        effective_mode=local_mode,
        crop_bbox=[x0, y0, x1, y1],
        crop_area=int(crop_area),
        mask_pixels=int(mask_pixels),
        mask_ratio=round(mask_ratio, 6),
    )
    directional_local = (
        mask_ratio <= 0.30
        and (
            crop_h >= crop_w * 1.8
            or crop_w >= crop_h * 2.2
        )
    )
    conservative_local = (
        mask_ratio <= 0.22
        and (
            crop_w <= 140
            or crop_h >= crop_w * 1.8
        )
    )
    if cleanup_tag_normalized == "speech_recovered_anchor_glyph_local":
        cleaned = _apply_speech_strong_local_fill(crop_img, crop_mask, allow_partial=False)
        backend = "speech_strong_local_fill"
        repair_attempted = cleaned is None
        repair_backend = None
        repair_status = "not_required_bright_speech_bubble_fill"
        if cleaned is None and directional_local:
            cleaned = _apply_directional_local_fill(crop_img, crop_mask)
            backend = "directional_local_fill"
            repair_backend = backend
        if cleaned is None:
            cleaned = _apply_conservative_local_inpaint(crop_img, crop_mask)
            backend = "conservative_local_inpaint"
            repair_backend = backend
        if cleaned is None:
            nested_debug: dict[str, object] = {}
            cleaned = apply_text_removal(
                crop_img,
                crop_mask,
                local_mode,
                use_gpu,
                model_id=model_id,
                debug_info=nested_debug,
            ).cleaned_image
            backend = f"nested:{nested_debug.get('backend') or 'unknown'}"
            repair_backend = backend
            if debug_info is not None:
                debug_info["backend_detail"] = nested_debug.get("backend_detail")
        if repair_attempted:
            repair_status = "applied" if cleaned is not None else "failed"
        if debug_info is not None:
            debug_info["root_cleanup_repair_v2_attempted"] = bool(repair_attempted)
            debug_info["root_cleanup_repair_v2_status"] = repair_status
            debug_info["root_cleanup_repair_v2_backend"] = repair_backend or backend
            debug_info["root_cleanup_repair_v2_attempt_count"] = 1 if repair_attempted else 0
            debug_info["root_cleanup_repair_v2_reason"] = (
                "speech_strong_local_fill_rejected_complex_or_partial_components"
                if repair_attempted
                else "primary_bright_speech_bubble_fill_accepted"
            )
    elif speech_glyph_local:
        cleaned = _apply_speech_strong_local_fill(crop_img, crop_mask, allow_partial=True)
        backend = "speech_glyph_local_fill"
        if cleaned is None and directional_local:
            cleaned = _apply_directional_local_fill(crop_img, crop_mask)
            backend = "directional_local_fill"
        if cleaned is None:
            cleaned = _apply_conservative_local_inpaint(crop_img, crop_mask)
            backend = "conservative_local_inpaint"
    elif glyph_stroke_local:
        cleaned = _apply_source_residual_local_fill(crop_img, crop_mask)
        backend = "glyph_stroke_local_fill"
        if cleaned is None and directional_local:
            cleaned = _apply_directional_local_fill(crop_img, crop_mask)
            backend = "directional_local_fill"
        if cleaned is None:
            cleaned = _apply_conservative_local_inpaint(crop_img, crop_mask)
            backend = "conservative_local_inpaint"
    elif cleanup_tag_normalized == "speech_strong":
        cleaned = _apply_speech_strong_local_fill(crop_img, crop_mask)
        backend = "speech_strong_local_fill"
        if cleaned is None:
            nested_debug = {}
            cleaned = apply_text_removal(
                crop_img,
                crop_mask,
                local_mode,
                use_gpu,
                model_id=model_id,
                debug_info=nested_debug,
            ).cleaned_image
            backend = f"nested:{nested_debug.get('backend') or 'unknown'}"
            if debug_info is not None:
                debug_info["backend_detail"] = nested_debug.get("backend_detail")
    elif force_strong_local:
        nested_debug = {}
        cleaned = apply_text_removal(
            crop_img,
            crop_mask,
            local_mode,
            use_gpu,
            model_id=model_id,
            debug_info=nested_debug,
        ).cleaned_image
        backend = f"nested:{nested_debug.get('backend') or 'unknown'}"
        if debug_info is not None:
            debug_info["backend_detail"] = nested_debug.get("backend_detail")
    elif side_caption_local:
        cleaned = _apply_speech_strong_local_fill(crop_img, crop_mask, allow_partial=True)
        backend = "speech_strong_local_fill"
        if cleaned is None:
            cleaned = _apply_conservative_local_inpaint(crop_img, crop_mask)
            backend = "conservative_local_inpaint"
        if cleaned is None:
            nested_debug = {}
            cleaned = apply_text_removal(
                crop_img,
                crop_mask,
                "fast",
                use_gpu,
                model_id=model_id,
                debug_info=nested_debug,
            ).cleaned_image
            backend = f"nested:{nested_debug.get('backend') or 'unknown'}"
            if debug_info is not None:
                debug_info["backend_detail"] = nested_debug.get("backend_detail")
                debug_info["effective_inpaint_mode"] = "fast"
    elif cleanup_tag_normalized == "cleanup_effectiveness_retry":
        cleaned = _apply_source_residual_local_fill(crop_img, crop_mask)
        backend = "source_residual_local_fill"
        if cleaned is None:
            cleaned = _apply_conservative_local_inpaint(crop_img, crop_mask)
            backend = "conservative_local_inpaint"
        if cleaned is None:
            nested_debug = {}
            cleaned = apply_text_removal(
                crop_img,
                crop_mask,
                "fast",
                use_gpu,
                model_id=model_id,
                debug_info=nested_debug,
            ).cleaned_image
            backend = f"nested:{nested_debug.get('backend') or 'unknown'}"
            if debug_info is not None:
                debug_info["backend_detail"] = nested_debug.get("backend_detail")
                debug_info["effective_inpaint_mode"] = "fast"
    elif cleanup_tag_normalized == "deterministic_local_fill":
        cleaned = _apply_source_residual_local_fill(crop_img, crop_mask)
        backend = "source_residual_local_fill"
        if cleaned is None and directional_local:
            cleaned = _apply_directional_local_fill(crop_img, crop_mask)
            backend = "directional_local_fill"
        if cleaned is None:
            cleaned = _apply_conservative_local_inpaint(crop_img, crop_mask)
            backend = "conservative_local_inpaint"
    elif cleanup_tag_normalized == "minimal_erasure":
        minimal_mask = crop_mask
        if cv2 is not None and np is not None:
            minimal_mask = cv2.erode(
                (crop_mask > 0).astype(np.uint8),
                np.ones((2, 2), np.uint8),
                iterations=1,
            )
            if not np.any(minimal_mask > 0):
                minimal_mask = crop_mask
        cleaned = _apply_source_residual_local_fill(crop_img, minimal_mask)
        backend = "minimal_source_residual_local_fill"
        if cleaned is None:
            cleaned = _apply_conservative_local_inpaint(crop_img, minimal_mask)
            backend = "minimal_conservative_local_inpaint"
    elif directional_local:
        cleaned = _apply_directional_local_fill(crop_img, crop_mask)
        backend = "directional_local_fill"
    elif conservative_local:
        cleaned = _apply_conservative_local_inpaint(crop_img, crop_mask)
        backend = "conservative_local_inpaint"
    else:
        nested_debug = {}
        cleaned = apply_text_removal(
            crop_img,
            crop_mask,
            local_mode,
            use_gpu,
            model_id=model_id,
            debug_info=nested_debug,
        ).cleaned_image
        backend = f"nested:{nested_debug.get('backend') or 'unknown'}"
        if debug_info is not None:
            debug_info["backend_detail"] = nested_debug.get("backend_detail")
    if debug_info is not None:
        debug_info["backend"] = debug_info.get("backend") or backend
    if cleaned is None:
        _set_cleanup_backend(debug_info, "none", backend_detail=f"{backend}_produced_no_output")
        _page014_timeout_checkpoint(
            "cleanup_apply_local_text_removal",
            "end",
            backend="none",
            reason=f"{backend}_produced_no_output",
            cleanup_tag=cleanup_tag_normalized,
            crop_bbox=[x0, y0, x1, y1],
            crop_area=int(crop_area),
            mask_pixels=int(mask_pixels),
            elapsed_ms=round((time.time() - started) * 1000.0, 3),
        )
        return _result(image, debug_info)
    patched = image.copy()
    patched.paste(cleaned, (x0, y0))
    _page014_timeout_checkpoint(
        "cleanup_apply_local_text_removal",
        "end",
        backend=backend,
        cleanup_tag=cleanup_tag_normalized,
        crop_bbox=[x0, y0, x1, y1],
        crop_area=int(crop_area),
        mask_pixels=int(mask_pixels),
        elapsed_ms=round((time.time() - started) * 1000.0, 3),
    )
    return _result(patched, debug_info)


def apply_bubble_fill(image: Any, bubble_mask: Any, text_mask: Any, reference_np: Any) -> CleanupExecutionResult:
    if bubble_mask is None or text_mask is None:
        return CleanupExecutionResult(cleaned_image=image, backend="none", backend_detail="missing_mask")
    if cv2 is None or np is None:
        return CleanupExecutionResult(
            cleaned_image=_apply_white_mask(image, text_mask),
            backend="white_mask",
            backend_detail="cv2_or_numpy_unavailable",
        )
    result = np.array(image)
    ref = reference_np if reference_np is not None else result
    num_labels, labels = cv2.connectedComponents(bubble_mask)
    if num_labels <= 1:
        return CleanupExecutionResult(
            cleaned_image=_apply_white_mask(image, text_mask),
            backend="white_mask",
            backend_detail="no_bubble_components",
        )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    text_binary = (text_mask > 0).astype(np.uint8)
    for label in range(1, num_labels):
        component = (labels == label).astype(np.uint8)
        if component.sum() == 0:
            continue
        text_region = cv2.bitwise_and(component, text_binary)
        if text_region.sum() == 0:
            continue
        ys, xs = np.where(text_region > 0)
        if ys.size == 0 or xs.size == 0:
            continue
        tx0, ty0, tx1, ty1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        pad = max(4, int(max(tx1 - tx0, ty1 - ty0) * 0.35))
        bx0 = max(0, tx0 - pad)
        by0 = max(0, ty0 - pad)
        bx1 = min(component.shape[1] - 1, tx1 + pad)
        by1 = min(component.shape[0] - 1, ty1 + pad)
        local = np.zeros_like(component)
        local[by0 : by1 + 1, bx0 : bx1 + 1] = component[by0 : by1 + 1, bx0 : bx1 + 1]
        halo = cv2.dilate(text_region, kernel, iterations=1)
        sample_mask = cv2.bitwise_and(local, cv2.bitwise_not(halo))
        samples = ref[sample_mask > 0]
        if samples.size < 12:
            inner = cv2.erode(component, kernel, iterations=2)
            if inner.sum() > 0:
                samples = ref[inner > 0]
        if samples.size < 3:
            color = (255, 255, 255)
        else:
            median = np.median(samples.reshape(-1, 3), axis=0)
            color = tuple(int(v) for v in median.tolist())
        fill_region = cv2.dilate(text_region, kernel, iterations=2)
        fill_region = cv2.bitwise_and(fill_region, component)
        result[fill_region > 0] = color
    return CleanupExecutionResult(cleaned_image=Image.fromarray(result), backend="bubble_fill")


def _result(image: Any, debug_info: dict | None) -> CleanupExecutionResult:
    debug_info = debug_info or {}
    return CleanupExecutionResult(
        cleaned_image=image,
        backend=debug_info.get("backend"),
        backend_detail=debug_info.get("backend_detail"),
        effective_inpaint_mode=debug_info.get("effective_inpaint_mode"),
        crop_bbox=debug_info.get("crop_bbox"),
        crop_area=debug_info.get("crop_area"),
        mask_ratio=debug_info.get("mask_ratio"),
        fallback_status=debug_info.get("fallback_status"),
    )


def _set_cleanup_backend(debug_info: dict | None, backend: str, **fields: object) -> None:
    if debug_info is None:
        return
    debug_info["backend"] = backend
    for key, value in fields.items():
        debug_info[key] = value


def _mask_stats_box(stats) -> tuple[int, int, int, int] | None:
    if not isinstance(stats, dict):
        return None
    box = stats.get("bbox")
    if not box:
        return None
    try:
        x0, y0, x1, y1 = [int(v) for v in box[:4]]
    except Exception:
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _apply_conservative_local_inpaint(image, mask):
    if cv2 is None or np is None or mask is None:
        return image
    img_np = np.array(image)
    if img_np.size == 0:
        return image
    if len(mask.shape) != 2:
        return image
    mask_u8 = mask.astype(np.uint8)
    if not np.any(mask_u8):
        return image
    max_dim = max(mask_u8.shape[:2])
    kernel_size = 3 if max_dim <= 220 else 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(mask_u8, kernel, iterations=1)
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=1)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    radius = 2 if max_dim <= 220 else 3
    inpainted = cv2.inpaint(img_bgr, dilated, radius, cv2.INPAINT_TELEA)
    rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _apply_directional_local_fill(image, mask):
    if cv2 is None or np is None or mask is None:
        return image
    img_np = np.array(image)
    if img_np.size == 0:
        return image
    mask_u8 = mask.astype(np.uint8)
    if not np.any(mask_u8):
        return image
    filled = img_np.copy()
    h, w = mask_u8.shape[:2]
    vertical = h >= w * 1.5
    mask_bool = mask_u8 > 0

    if vertical:
        row_fallback = img_np[~mask_bool]
        fallback_color = (
            np.median(row_fallback.reshape(-1, 3), axis=0)
            if row_fallback.size
            else np.array([255, 255, 255], dtype=np.float32)
        )
        for y in range(h):
            xs = np.where(mask_bool[y])[0]
            if xs.size == 0:
                continue
            left = xs[0]
            right = xs[-1]
            left_candidates = np.where(~mask_bool[y, :left])[0]
            right_candidates = np.where(~mask_bool[y, right + 1 :])[0]
            if left_candidates.size and right_candidates.size:
                left_idx = int(left_candidates[-1])
                right_idx = int(right + 1 + right_candidates[0])
                left_color = filled[y, left_idx].astype(np.float32)
                right_color = filled[y, right_idx].astype(np.float32)
                span = max(1, right - left)
                for x in range(left, right + 1):
                    t = (x - left) / span if span > 0 else 0.0
                    filled[y, x] = np.clip((1.0 - t) * left_color + t * right_color, 0, 255)
            elif left_candidates.size:
                filled[y, left : right + 1] = filled[y, int(left_candidates[-1])]
            elif right_candidates.size:
                filled[y, left : right + 1] = filled[y, int(right + 1 + right_candidates[0])]
            else:
                filled[y, left : right + 1] = fallback_color
    else:
        col_fallback = img_np[~mask_bool]
        fallback_color = (
            np.median(col_fallback.reshape(-1, 3), axis=0)
            if col_fallback.size
            else np.array([255, 255, 255], dtype=np.float32)
        )
        for x in range(w):
            ys = np.where(mask_bool[:, x])[0]
            if ys.size == 0:
                continue
            top = ys[0]
            bottom = ys[-1]
            top_candidates = np.where(~mask_bool[:top, x])[0]
            bottom_candidates = np.where(~mask_bool[bottom + 1 :, x])[0]
            if top_candidates.size and bottom_candidates.size:
                top_idx = int(top_candidates[-1])
                bottom_idx = int(bottom + 1 + bottom_candidates[0])
                top_color = filled[top_idx, x].astype(np.float32)
                bottom_color = filled[bottom_idx, x].astype(np.float32)
                span = max(1, bottom - top)
                for y in range(top, bottom + 1):
                    t = (y - top) / span if span > 0 else 0.0
                    filled[y, x] = np.clip((1.0 - t) * top_color + t * bottom_color, 0, 255)
            elif top_candidates.size:
                filled[top : bottom + 1, x] = filled[int(top_candidates[-1]), x]
            elif bottom_candidates.size:
                filled[top : bottom + 1, x] = filled[int(bottom + 1 + bottom_candidates[0])]
            else:
                filled[top : bottom + 1, x] = fallback_color

    blur = cv2.GaussianBlur(filled, (3, 3), 0)
    feather = cv2.dilate(mask_u8, np.ones((3, 3), np.uint8), iterations=1)
    feather = feather.astype(np.float32) / 255.0
    feather = feather[..., None]
    blended = img_np.astype(np.float32) * (1.0 - feather) + blur.astype(np.float32) * feather
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def _apply_source_residual_local_fill(image, mask):
    if cv2 is None or np is None or mask is None:
        return None
    img_np = np.array(image)
    if img_np.size == 0:
        return None
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    if not np.any(mask_u8):
        return None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fill_mask = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    fill_mask = cv2.bitwise_and(fill_mask, mask_u8)
    if not np.any(fill_mask):
        fill_mask = mask_u8
    sample_seed = cv2.dilate(mask_u8, kernel, iterations=1)
    sample_ring = cv2.dilate(sample_seed, kernel, iterations=3)
    sample_ring = cv2.subtract(sample_ring, sample_seed)
    samples = img_np[sample_ring > 0]
    if samples.size < 18:
        return None
    sample_arr = samples.reshape(-1, 3)
    median = np.median(sample_arr, axis=0)
    std = float(sample_arr.std(axis=0).mean())
    gray_samples = cv2.cvtColor(sample_arr.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2GRAY).reshape(-1)
    median_luma = float(np.median(gray_samples))
    if std > 72 and median_luma < 188:
        return None
    result = img_np.copy()
    result[fill_mask > 0] = median.astype(np.uint8)
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def _apply_uniform_fill(image, text_mask):
    if cv2 is None or np is None:
        return None
    if text_mask is None:
        return None
    if not np.any(text_mask):
        return None
    img_np = np.array(image)
    num_labels, labels = cv2.connectedComponents(text_mask)
    if num_labels <= 1:
        return None
    result = img_np.copy()
    applied = False
    skipped_complex = False  # Track if we skipped any complex backgrounds
    for label in range(1, num_labels):
        component = (labels == label).astype(np.uint8)
        if component.sum() < 8:
            continue

        # First, sample border to determine background complexity
        sample_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        sample_dilated = cv2.dilate(component, sample_kernel, iterations=2)
        sample_border = cv2.subtract(sample_dilated, component)
        samples = img_np[sample_border > 0]

        if samples.size < 9:
            # Not enough samples - skip, let AI handle it
            skipped_complex = True
            continue

        sample_arr = samples.reshape(-1, 3)
        std = float(sample_arr.std(axis=0).mean())
        median = np.median(sample_arr, axis=0)

        # ONLY handle uniform backgrounds - skip complex ones for AI
        if std <= 15:
            # Very uniform (white bubble): moderate dilation to preserve borders
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilated = cv2.dilate(component, kernel, iterations=2)
            result[dilated > 0] = median.astype(np.uint8)
            applied = True
        elif std <= 30:
            # Moderately uniform: minimal dilation
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            dilated = cv2.dilate(component, kernel, iterations=1)
            result[dilated > 0] = median.astype(np.uint8)
            applied = True
        else:
            # Complex background (artwork): SKIP - let AI handle it
            skipped_complex = True
            continue

    # Only return if we processed some regions AND didn't skip any complex ones
    # If we skipped complex regions, return None so AI inpainting handles everything
    if applied and not skipped_complex:
        return Image.fromarray(result)
    return None


def _apply_speech_strong_local_fill(image, text_mask, allow_partial: bool = False):
    if cv2 is None or np is None:
        return None
    if text_mask is None or not np.any(text_mask):
        return None
    img_np = np.array(image)
    if img_np.size == 0:
        return None
    num_labels, labels = cv2.connectedComponents((text_mask > 0).astype(np.uint8))
    if num_labels <= 1:
        return None
    result = img_np.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    applied_pixels = 0
    skipped_complex = False
    for label in range(1, num_labels):
        component = (labels == label).astype(np.uint8)
        component_pixels = int(component.sum())
        if component_pixels < 6:
            continue
        halo = cv2.dilate(component, kernel, iterations=2)
        border = cv2.subtract(halo, component)
        samples = img_np[border > 0]
        if samples.size < 9:
            skipped_complex = True
            continue
        sample_arr = samples.reshape(-1, 3)
        median = np.median(sample_arr, axis=0)
        gray = cv2.cvtColor(sample_arr.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2GRAY).reshape(-1)
        std = float(sample_arr.std(axis=0).mean())
        median_luma = float(np.median(gray))
        # Speech-bubble cleanup should look like local paper, not CV2 smears.
        # Only use this shortcut when the immediate background is plausibly
        # bubble paper or a gentle tone; complex art still falls back.
        if median_luma < 135 or std > 62:
            skipped_complex = True
            continue
        fill = cv2.dilate(component, kernel, iterations=1)
        result[fill > 0] = median.astype(np.uint8)
        applied_pixels += component_pixels
    if applied_pixels <= 0:
        return None
    if skipped_complex and not allow_partial:
        return None
    return Image.fromarray(result)


def _apply_white_mask(image, text_mask):
    if cv2 is None or np is None:
        ImageDraw.Draw(image)
        return image
    img_np = np.array(image)
    # Aggressively dilate mask to ensure complete text coverage
    kernel_size = max(7, int(max(text_mask.shape) * 0.006))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(text_mask, kernel, iterations=4)
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)
    img_np[dilated > 0] = (255, 255, 255)
    return Image.fromarray(img_np)


def _white_bubble_fill(image, text_mask):
    if cv2 is None or np is None:
        return None
    ys, xs = np.where(text_mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    pad = 12
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(text_mask.shape[1] - 1, x1 + pad)
    y1 = min(text_mask.shape[0] - 1, y1 + pad)
    img_np = np.array(image)
    roi = img_np[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    mean = float(gray.mean())
    std = float(gray.std())
    if mean > 220 and std < 18:
        filled = _apply_white_mask(image, text_mask)
        return filled
    return None
