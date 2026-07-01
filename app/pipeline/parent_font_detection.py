# -*- coding: utf-8 -*-
"""Parent-owned font and style detection for execution bundles."""
from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.models.resolution import (
    resolve_noto_cjk_sc_font_file,
    resolve_yuzumarker_font_labels_file,
    resolve_yuzumarker_font_onnx_file,
)
from app.pipeline.parent_execution_bundle import PARENT_RENDER_STYLE_VERSION


FONT_COUNT = 6150
YUZUMARKER_PROVIDER = "YuzuMarker.FontDetection"
YUZUMARKER_PROVIDER_MODEL = "ogkalu/yuzumarker-font-detection-onnx:font-detector.onnx"
YUZUMARKER_STYLE_SOURCE = "parent_execution_bundle_yuzumarker_font_detection"
FALLBACK_STYLE_SOURCE = "parent_execution_bundle_font_fallback"
MIN_STYLE_CONFIDENCE = 0.02


@dataclass
class ParentFontDetectionRunResult:
    page_id: str
    mode: str
    enabled: bool = False
    applied_count: int = 0
    fallback_count: int = 0
    skipped_count: int = 0
    model_path: str = ""
    labels_path: str = ""
    errors: list[str] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "parent_font_detection_version": "parent_font_detection_v1",
            "page_id": self.page_id,
            "mode": self.mode,
            "enabled": self.enabled,
            "applied_count": self.applied_count,
            "fallback_count": self.fallback_count,
            "skipped_count": self.skipped_count,
            "model_path": self.model_path,
            "labels_path": self.labels_path,
            "errors": list(self.errors),
            "records": [dict(record) for record in self.records],
        }


class YuzuMarkerOnnxFontDetector:
    """Small ONNX adapter for YuzuMarker.FontDetection."""

    def __init__(
        self,
        *,
        model_path: str | None = None,
        labels_path: str | None = None,
        use_gpu: bool = False,
    ) -> None:
        self.model_path = model_path or resolve_yuzumarker_font_onnx_file() or ""
        self.labels_path = labels_path or resolve_yuzumarker_font_labels_file() or ""
        if not self.model_path or not os.path.isfile(self.model_path):
            raise FileNotFoundError("YuzuMarker ONNX model is missing")
        if not self.labels_path or not os.path.isfile(self.labels_path):
            raise FileNotFoundError("YuzuMarker font labels are missing")
        self._labels = _load_font_labels(self.labels_path)
        self._session = _load_onnx_session(self.model_path, use_gpu=use_gpu)
        inputs = self._session.get_inputs()
        if not inputs:
            raise RuntimeError("YuzuMarker ONNX model has no inputs")
        self._input_name = inputs[0].name

    def detect(self, image: Any) -> dict[str, Any]:
        import numpy as np
        from PIL import ImageOps

        image = ImageOps.exif_transpose(image).convert("RGB").resize((512, 512))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)[None, ...]
        output = self._session.run(None, {self._input_name: arr})[0]
        vector = np.asarray(output, dtype=np.float32).reshape(-1)
        if vector.shape[0] < FONT_COUNT + 12:
            raise RuntimeError(f"Unexpected YuzuMarker output length: {vector.shape[0]}")

        font_logits = vector[:FONT_COUNT]
        font_prob = _softmax(font_logits)
        top_indices = np.argsort(-font_prob)[:5]
        top_candidates = []
        for index in top_indices:
            label = _label_at(self._labels, int(index))
            top_candidates.append(
                {
                    "index": int(index),
                    "confidence": float(font_prob[int(index)]),
                    "path": str(label.get("path") or ""),
                    "language": str(label.get("language") or ""),
                    "serif": bool(label.get("serif")),
                }
            )

        direction_logits = vector[FONT_COUNT : FONT_COUNT + 2]
        direction_prob = _softmax(direction_logits)
        direction_index = int(direction_prob.argmax())
        regression = vector[FONT_COUNT + 2 : FONT_COUNT + 12]
        top_label = top_candidates[0] if top_candidates else {}
        return {
            "font_index": int(top_indices[0]) if len(top_indices) else -1,
            "confidence": float(top_candidates[0]["confidence"]) if top_candidates else 0.0,
            "font_path": str(top_label.get("path") or ""),
            "font_language": str(top_label.get("language") or ""),
            "font_serif": bool(top_label.get("serif")),
            "top_candidates": top_candidates,
            "direction": "ltr" if direction_index == 0 else "ttb",
            "direction_confidence": float(direction_prob[direction_index]),
            "text_color": _rgb_from_unit_values(regression[0:3]),
            "text_size_ratio": _float(regression[3]),
            "stroke_width_ratio": _float(regression[4]),
            "stroke_color": _rgb_from_unit_values(regression[5:8]),
            "line_spacing_ratio": _float(regression[8]),
            "angle_degrees": round((_float(regression[9]) - 0.5) * 180.0, 3),
        }


_SESSION_CACHE: dict[tuple[str, bool], Any] = {}


def apply_parent_font_detection(
    *,
    page_id: str,
    image_path: str,
    parent_execution_bundles: Sequence[Any],
    mode: str,
    default_font_name: str = "",
    use_gpu: bool = False,
    models_dir: str | None = None,
    detector: Any | None = None,
) -> ParentFontDetectionRunResult:
    """Attach parent-owned font/style evidence to execution bundles."""

    normalized_mode = str(mode or "off").strip().lower()
    result = ParentFontDetectionRunResult(page_id=page_id, mode=normalized_mode)
    if normalized_mode == "off":
        result.skipped_count = len(list(parent_execution_bundles or []))
        return result
    if normalized_mode not in {"yuzumarker", "heuristic"}:
        result.errors.append(f"unsupported_font_detection_mode:{normalized_mode}")
        normalized_mode = "heuristic"
        result.mode = normalized_mode

    bundles = list(parent_execution_bundles or [])
    if not bundles:
        return result
    result.enabled = True

    image = None
    try:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        result.errors.append(f"image_open_failed:{type(exc).__name__}:{exc}")

    active_detector = detector
    if normalized_mode == "yuzumarker" and active_detector is None:
        try:
            active_detector = YuzuMarkerOnnxFontDetector(
                model_path=resolve_yuzumarker_font_onnx_file(models_dir),
                labels_path=resolve_yuzumarker_font_labels_file(models_dir),
                use_gpu=use_gpu,
            )
            result.model_path = getattr(active_detector, "model_path", "") or ""
            result.labels_path = getattr(active_detector, "labels_path", "") or ""
        except Exception as exc:
            result.errors.append(f"yuzumarker_unavailable:{type(exc).__name__}:{exc}")
            active_detector = None

    for bundle in bundles:
        record = _apply_style_to_bundle(
            bundle,
            image=image,
            mode=normalized_mode,
            detector=active_detector,
            default_font_name=default_font_name,
            models_dir=models_dir,
        )
        result.records.append(record)
        status = str(record.get("status") or "")
        if status == "applied":
            result.applied_count += 1
        elif status == "skipped":
            result.skipped_count += 1
        else:
            result.fallback_count += 1

    try:
        if image is not None:
            image.close()
    except Exception:
        pass
    return result


def _apply_style_to_bundle(
    bundle: Any,
    *,
    image: Any | None,
    mode: str,
    detector: Any | None,
    default_font_name: str,
    models_dir: str | None,
) -> dict[str, Any]:
    bundle_id = str(getattr(bundle, "bundle_id", "") or "")
    parent_id = str(getattr(bundle, "parent_id", "") or "")
    if not bool(getattr(bundle, "render_required", False)):
        return {
            "bundle_id": bundle_id,
            "parent_id": parent_id,
            "status": "skipped",
            "reason": "render_not_required",
        }

    bbox = _best_style_bbox(bundle)
    crop = _crop_image(image, bbox)
    detection = None
    status = "fallback"
    reason = "detector_unavailable"
    if mode == "yuzumarker" and detector is not None and crop is not None:
        try:
            detection = detector.detect(crop)
            confidence = _float(detection.get("confidence")) if isinstance(detection, Mapping) else 0.0
            if confidence >= MIN_STYLE_CONFIDENCE:
                status = "applied"
                reason = ""
            else:
                status = "fallback"
                reason = "low_font_confidence"
        except Exception as exc:
            status = "fallback"
            reason = f"detector_failed:{type(exc).__name__}"
    elif mode == "heuristic" and crop is not None:
        detection = _heuristic_detection(crop)
        status = "applied"
        reason = ""
    elif crop is None:
        reason = "invalid_parent_style_crop"

    style = _style_for_bundle(
        bundle,
        detection=detection if isinstance(detection, Mapping) else None,
        status=status,
        fallback_reason=reason,
        default_font_name=default_font_name,
        models_dir=models_dir,
    )
    _merge_render_style(bundle, style)
    if hasattr(bundle, "execution_region"):
        try:
            bundle.execution_region = bundle.to_region_record()
        except Exception:
            pass

    record = {
        "bundle_id": bundle_id,
        "parent_id": parent_id,
        "root_id": str(getattr(bundle, "root_id", "") or ""),
        "status": status,
        "fallback_reason": reason,
        "crop_bbox": list(bbox) if bbox else [],
        "render_style_source": style.get("render_style_source"),
        "render_style_provider": style.get("render_style_provider"),
        "render_style_provider_model": style.get("render_style_provider_model"),
        "render_style_confidence": style.get("render_style_confidence"),
        "font_family": style.get("font_family"),
        "font_weight": style.get("font_weight"),
        "style_class": style.get("style_class"),
        "source_font_label": style.get("source_font_label"),
        "source_font_language": style.get("source_font_language"),
        "source_font_serif": style.get("source_font_serif"),
    }
    return {key: value for key, value in record.items() if value not in (None, "", [])}


def _style_for_bundle(
    bundle: Any,
    *,
    detection: Mapping[str, Any] | None,
    status: str,
    fallback_reason: str,
    default_font_name: str,
    models_dir: str | None,
) -> dict[str, Any]:
    existing = dict(getattr(bundle, "render_style", {}) or {})
    role = str(getattr(bundle, "role", "") or "")
    lowered_role = role.lower()
    caption_like = any(token in lowered_role for token in ("caption", "background", "narration", "sign"))
    confidence = _float(detection.get("confidence")) if detection else 0.0
    trusted_detection = detection if detection and confidence >= MIN_STYLE_CONFIDENCE else {}
    source_label = str(trusted_detection.get("font_path") or "")
    source_serif = bool(trusted_detection.get("font_serif")) if trusted_detection else False
    source_language = str(trusted_detection.get("font_language") or "")
    weight = _font_weight_from_label(source_label, caption_like=caption_like)
    serif = bool(source_serif)
    if not trusted_detection and caption_like:
        weight = "bold"
    font_path = resolve_noto_cjk_sc_font_file(base_dir=models_dir, serif=serif, weight=weight)
    if not font_path:
        font_path = default_font_name or existing.get("font_family") or "Microsoft YaHei"

    direction = str((trusted_detection or detection or {}).get("direction") or "")
    source_orientation = "vertical" if direction == "ttb" else "horizontal" if direction == "ltr" else ""
    style: dict[str, Any] = {
        "render_style_version": PARENT_RENDER_STYLE_VERSION,
        "render_style_owner": "parent_execution_bundle",
        "render_style_source": YUZUMARKER_STYLE_SOURCE if trusted_detection else FALLBACK_STYLE_SOURCE,
        "render_style_provider": YUZUMARKER_PROVIDER if detection else "parent_font_fallback",
        "render_style_provider_model": YUZUMARKER_PROVIDER_MODEL if detection else "",
        "style_class": existing.get("style_class") or ("caption" if caption_like else "dialogue"),
        "font_family": font_path,
        "font_weight": weight,
        "fill_color": existing.get("fill_color") or "#000000",
        "stroke_color": existing.get("stroke_color") or "#FFFFFF",
        "stroke_width": existing.get("stroke_width") if existing.get("stroke_width") is not None else (2 if caption_like else 1),
        "source_orientation": existing.get("source_orientation") or source_orientation or "vertical",
        "wrap_mode": existing.get("wrap_mode") or ("vertical" if (source_orientation != "horizontal") else "horizontal"),
        "line_height": existing.get("line_height") or (1.1 if caption_like else 1.0),
        "align": existing.get("align") or "center",
        "fallback_reason": fallback_reason,
        "font_detection_status": status,
    }
    style.update(
        _layout_hints_for_bundle(
            bundle,
            detection=trusted_detection if trusted_detection else None,
            caption_like=caption_like,
            source_orientation=str(style.get("source_orientation") or ""),
        )
    )
    if detection:
        style.update(
            {
                "render_style_confidence": confidence,
                "source_font_label": str(detection.get("font_path") or ""),
                "source_font_language": str(detection.get("font_language") or ""),
                "source_font_serif": bool(detection.get("font_serif")),
                "source_font_top_candidates": _compact_candidates(detection.get("top_candidates")),
                "source_text_color": detection.get("text_color"),
                "source_stroke_color": detection.get("stroke_color"),
                "source_stroke_width_ratio": _float(detection.get("stroke_width_ratio")),
                "source_text_size_ratio": _float(detection.get("text_size_ratio")),
                "source_line_spacing_ratio": _float(detection.get("line_spacing_ratio")),
                "source_angle_degrees": _float(detection.get("angle_degrees")),
                "source_direction": str(detection.get("direction") or ""),
                "source_direction_confidence": _float(detection.get("direction_confidence")),
            }
        )
    if trusted_detection:
        style["fallback_reason"] = ""
    return {key: value for key, value in style.items() if value not in (None, "", [])}


def _merge_render_style(bundle: Any, style: Mapping[str, Any]) -> None:
    existing = dict(getattr(bundle, "render_style", {}) or {})
    merged = dict(existing)
    merged.update(dict(style))
    setattr(bundle, "render_style", merged)


def _layout_hints_for_bundle(
    bundle: Any,
    *,
    detection: Mapping[str, Any] | None,
    caption_like: bool,
    source_orientation: str,
) -> dict[str, Any]:
    bbox = _best_style_bbox(bundle)
    if not bbox:
        return {}
    _x, _y, width, height = bbox
    vertical = str(source_orientation or "").strip().lower() != "horizontal"
    content_count = _content_char_count(
        str(getattr(bundle, "source_text", "") or getattr(bundle, "translated_text", "") or "")
    )
    size_ratio = _clamp_float(
        detection.get("text_size_ratio") if isinstance(detection, Mapping) else 0.0,
        0.0,
        0.80,
    )
    line_spacing_ratio = _clamp_float(
        detection.get("line_spacing_ratio") if isinstance(detection, Mapping) else 0.0,
        0.0,
        0.65,
    )

    if vertical:
        if content_count <= 3:
            width_factor, height_factor = 0.50, 0.26
        elif content_count <= 6:
            width_factor, height_factor = 0.46, 0.23
        elif content_count <= 12:
            width_factor, height_factor = 0.40, 0.18
        else:
            width_factor, height_factor = 0.36, 0.145
        geometry_size = min(width * width_factor, height * height_factor)
        if size_ratio > 0:
            ratio_size = min(
                width * (0.30 + min(size_ratio, 0.50) * 0.12),
                height * (0.11 + min(size_ratio, 0.50) * 0.16),
            )
            hint = int(round(geometry_size * 0.78 + ratio_size * 0.22))
        else:
            hint = int(round(geometry_size))
        readable_min = 20 if content_count <= 3 else 18 if content_count <= 8 else 16
        if caption_like:
            readable_min = max(18, readable_min)
        line_height = max(
            1.10 if caption_like else 1.06,
            1.06 + min(line_spacing_ratio, 0.50) * 0.14,
        )
        line_height = _clamp_float(line_height, 1.06, 1.18)
    else:
        geometry_size = min(height * 0.72, width * 0.16)
        if size_ratio > 0:
            ratio_size = min(
                height * (0.45 + min(size_ratio, 0.50) * 0.35),
                width * (0.08 + min(size_ratio, 0.45) * 0.16),
            )
            hint = int(round(geometry_size * 0.60 + ratio_size * 0.40))
        else:
            hint = int(round(geometry_size))
        readable_min = 16 if caption_like else 15
        line_height = max(
            1.16 if not caption_like else 1.18,
            1.14 + min(line_spacing_ratio, 0.50) * 0.24,
        )
        line_height = _clamp_float(line_height, 1.14, 1.32)

    if vertical:
        if caption_like:
            hint_cap = 58
        elif content_count >= 20:
            hint_cap = 50
        elif content_count >= 13:
            hint_cap = 52
        elif content_count <= 3:
            hint_cap = 46
        else:
            hint_cap = 56
    else:
        hint_cap = 64 if caption_like else 58
    hint = max(readable_min, min(hint_cap, int(hint or 0)))
    min_size = max(12, min(hint, int(round(hint * 0.86))))
    max_size = max(hint, min(hint_cap, int(round(hint * 1.08))))
    return {
        "font_size_hint": hint,
        "font_size_min": min_size,
        "font_size_max": max_size,
        "line_height": round(float(line_height), 3),
        "spacing_profile": {
            "source": "yuzumarker" if detection else "parent_geometry_fallback",
            "orientation": "vertical" if vertical else "horizontal",
            "content_count": content_count,
            "font_size_hint": hint,
            "font_size_min": min_size,
            "font_size_max": max_size,
            "line_height": round(float(line_height), 3),
            "minimum_readable_font_size": readable_min,
            "source_text_size_ratio": round(float(size_ratio), 4),
            "source_line_spacing_ratio": round(float(line_spacing_ratio), 4),
        },
    }


def _content_char_count(text: str) -> int:
    count = 0
    for char in str(text or ""):
        if char.isspace() or char in "…。、，,.!?！？ー─-—~〜・：:；;「」『』（）()[]【】":
            continue
        count += 1
    return count or len(str(text or ""))


def _clamp_float(value: Any, low: float, high: float) -> float:
    try:
        number = float(value)
    except Exception:
        number = 0.0
    return max(float(low), min(float(high), number))


def _load_onnx_session(model_path: str, *, use_gpu: bool) -> Any:
    key = (os.path.abspath(model_path), bool(use_gpu))
    if key in _SESSION_CACHE:
        return _SESSION_CACHE[key]
    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    if use_gpu:
        available = set(ort.get_available_providers())
        if "CUDAExecutionProvider" in available:
            providers.insert(0, "CUDAExecutionProvider")
    session = ort.InferenceSession(model_path, providers=providers)
    _SESSION_CACHE[key] = session
    return session


def _load_font_labels(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        labels = json.load(handle)
    if not isinstance(labels, list):
        raise RuntimeError("YuzuMarker font labels must be a list")
    return [dict(item) if isinstance(item, Mapping) else {"path": str(item)} for item in labels]


def _label_at(labels: Sequence[Mapping[str, Any]], index: int) -> Mapping[str, Any]:
    if 0 <= index < len(labels):
        return labels[index]
    return {}


def _softmax(values: Any) -> Any:
    import numpy as np

    arr = np.asarray(values, dtype=np.float32)
    arr = arr - float(arr.max())
    exp = np.exp(arr)
    denom = float(exp.sum())
    if denom <= 0:
        return np.zeros_like(arr)
    return exp / denom


def _rgb_from_unit_values(values: Any) -> str:
    try:
        raw_values = list(values)
    except Exception:
        raw_values = []
    vals = [_float(value) for value in raw_values[:3]]
    while len(vals) < 3:
        vals.append(0.0)
    channels = [max(0, min(255, int(round(value * 255.0)))) for value in vals]
    return "#{:02X}{:02X}{:02X}".format(*channels)


def _heuristic_detection(image: Any) -> dict[str, Any]:
    import numpy as np

    arr = np.asarray(image.convert("L"), dtype=np.float32)
    mean = float(arr.mean()) if arr.size else 255.0
    dark_ratio = float((arr < 96).mean()) if arr.size else 0.0
    light_on_dark = mean < 120.0
    return {
        "confidence": 1.0,
        "font_path": "heuristic/serif" if dark_ratio < 0.04 else "heuristic/sans",
        "font_language": "CJK",
        "font_serif": bool(dark_ratio < 0.04),
        "top_candidates": [],
        "direction": "ttb" if image.height >= image.width else "ltr",
        "direction_confidence": 1.0,
        "text_color": "#FFFFFF" if light_on_dark else "#000000",
        "stroke_color": "#000000" if light_on_dark else "#FFFFFF",
        "stroke_width_ratio": 0.004 if light_on_dark else 0.002,
        "text_size_ratio": 0.0,
        "line_spacing_ratio": 0.0,
        "angle_degrees": 0.0,
    }


def _best_style_bbox(bundle: Any) -> list[int]:
    for attr in (
        "source_contract_bbox",
        "parent_bbox",
        "render_allowed_area",
        "cleanup_target_bbox",
        "root_bbox",
    ):
        bbox = _bbox(getattr(bundle, attr, None))
        if bbox:
            return bbox
    execution_region = getattr(bundle, "execution_region", {}) or {}
    if isinstance(execution_region, Mapping):
        for key in ("source_contract_bbox", "bbox", "render_allowed_area"):
            bbox = _bbox(execution_region.get(key))
            if bbox:
                return bbox
    return []


def _crop_image(image: Any | None, bbox: Sequence[int]) -> Any | None:
    if image is None:
        return None
    box = _bbox(bbox)
    if not box:
        return None
    x, y, w, h = box
    pad = max(2, int(round(min(w, h) * 0.04)))
    left = max(0, x - pad)
    top = max(0, y - pad)
    right = min(int(image.width), x + w + pad)
    bottom = min(int(image.height), y + h + pad)
    if right <= left or bottom <= top:
        return None
    return image.crop((left, top, right, bottom))


def _bbox(value: Any) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 4:
        return []
    try:
        x, y, w, h = [int(round(float(value[index]))) for index in range(4)]
    except Exception:
        return []
    if w <= 0 or h <= 0:
        return []
    return [x, y, w, h]


def _font_weight_from_label(label: str, *, caption_like: bool) -> str:
    lowered = str(label or "").lower()
    if any(token in lowered for token in ("black", "heavy", "ultra", "w9", "w10", "w12", "w14")):
        return "black"
    if any(token in lowered for token in ("bold", "semibold", "demibold", "-b.", "_b.", "hei", "gothic")):
        return "bold"
    return "bold" if caption_like else "regular"


def _compact_candidates(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    candidates: list[dict[str, Any]] = []
    for item in value[:5]:
        if not isinstance(item, Mapping):
            continue
        candidates.append(
            {
                "index": item.get("index"),
                "confidence": _float(item.get("confidence")),
                "path": str(item.get("path") or ""),
                "language": str(item.get("language") or ""),
                "serif": bool(item.get("serif")),
            }
        )
    return candidates


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0
