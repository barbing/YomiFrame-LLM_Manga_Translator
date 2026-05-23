"""First-class Phase 4 bubble/text-area detection service.

The service owns the selected visual model stack for high-accuracy bubble
evidence:

- kitsumed/yolov8m_seg-speech-bubble for speech-bubble masks
- ogkalu/comic-text-and-bubble-detector for bubble/text_bubble/text_free boxes

It is metadata-only in this phase. Callers receive evidence, fused containers,
memberships, conflicts, runtime metadata, and fallback/error status. No routing,
cleanup, OCR, translation, rendering, or project output is mutated here.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


BUBBLE_DETECTION_VERSION = "phase4b16_bubble_detection_service_v1"
BUBBLE_DETECTION_CACHE_VERSION = "phase4b16_bubble_detection_evidence_cache_v1"

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
KITSUMED_MODEL = ROOT / "models" / "yolov8m_seg-speech-bubble" / "model_dynamic.onnx"
OGKALU_MODEL = ROOT / "models" / "comic-text-and-bubble-detector" / "detector.onnx"
OGKALU_CONFIG = ROOT / "models" / "comic-text-and-bubble-detector" / "config.json"

KITSUMED_MODEL_NAME = "kitsumed/yolov8m_seg-speech-bubble"
OGKALU_MODEL_NAME = "ogkalu/comic-text-and-bubble-detector"
KITSUMED_MODEL_ROLE = "speech_bubble_mask"
OGKALU_MODEL_ROLE = "text_area_detector"
PROVIDER_PREFERENCE = ["CUDAExecutionProvider", "CPUExecutionProvider"]
KITSUMED_CONFIDENCE_THRESHOLD = 0.30
KITSUMED_NMS_IOU_THRESHOLD = 0.50
KITSUMED_MASK_THRESHOLD = 0.50
OGKALU_CONFIDENCE_THRESHOLD = 0.50


@dataclass
class BubbleDetectionInput:
    page_id: str
    image_path: Path | str
    image_size: Optional[Tuple[int, int]] = None
    regions: Sequence[Mapping[str, Any]] = field(default_factory=list)
    deterministic_diagnostics: Mapping[str, Any] = field(default_factory=dict)
    debug_page_dir: Optional[Path | str] = None
    mode: str = "diagnostic"
    requested_models: Sequence[str] = field(default_factory=lambda: ["kitsumed_speech_bubble", "ogkalu_text_bubble"])


@dataclass
class BubbleModelEvidence:
    model_evidence_id: str
    model_name: str
    model_role: str
    model_path: str
    model_hash: str
    provider: str
    class_name: str
    confidence: float
    bbox_xyxy: List[float]
    mask_ref: Optional[str]
    mask_bbox_xyxy: List[float]
    mask_area_px: float
    mask_polygon: List[List[int]]
    latency_sec: float
    fallback_used: bool

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class TextAreaModelEvidence:
    model_evidence_id: str
    model_name: str
    model_role: str
    model_path: str
    model_hash: str
    provider: str
    class_name: str
    confidence: float
    bbox_xyxy: List[float]
    linked_bubble_mask_ids: List[str]
    conflict_flags: List[str]
    latency_sec: float
    fallback_used: bool

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class FusedBubbleContainer:
    container_id: str
    container_type: str
    confidence_tier: str
    bbox_xyxy: List[float]
    mask_ref: Optional[str]
    source_model_ids: List[str]
    linked_region_ids: List[str]
    role_constraints: List[str]
    conflict_flags: List[str]
    fallback_reason: Optional[str]
    human_review_required: bool

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class TextContainerMembership:
    region_id: str
    container_id: str
    membership_type: str
    inside_ratio: float
    center_inside: bool
    ownership_confidence: float
    ownership_reason_codes: List[str]
    must_not_mutate: bool

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class BubbleDetectionDecision:
    region_id: str
    decision_type: str
    decision_status: str
    supported_actions: List[str]
    blocked_actions: List[str]
    reason_codes: List[str]
    would_change_behavior: bool
    requires_flag: str
    requires_visual_review: bool

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class BubbleDetectionResult:
    page_id: str
    version: str
    generated: bool
    image_path: str
    image_size: Optional[Tuple[int, int]]
    model_names: Dict[str, str]
    model_paths: Dict[str, str]
    model_hashes: Dict[str, str]
    providers_requested: List[str]
    providers_used: Dict[str, List[str]]
    provider_fallback_used: bool
    runtime: Dict[str, Any]
    model_evidence: List[Dict[str, Any]]
    bubble_model_evidence: List[Dict[str, Any]]
    text_area_model_evidence: List[Dict[str, Any]]
    fused_containers: List[Dict[str, Any]]
    memberships: List[Dict[str, Any]]
    decisions: List[Dict[str, Any]]
    region_model_links: List[Dict[str, Any]]
    conflicts: List[Dict[str, Any]]
    fallback_used: bool
    error: Optional[str]
    runtime_sec: float
    cache_enabled: bool = False
    cache_key: Optional[str] = None
    cache_hit: bool = False
    cache_read_path: Optional[str] = None
    cache_write_path: Optional[str] = None
    cache_error: Optional[str] = None
    cache_invalidation_reason: Optional[str] = None
    raw_kitsumed_detections: List[Dict[str, Any]] = field(default_factory=list, repr=False)
    raw_ogkalu_detections: List[Dict[str, Any]] = field(default_factory=list, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_id": self.page_id,
            "version": self.version,
            "generated": self.generated,
            "image_path": self.image_path,
            "image_size": list(self.image_size) if self.image_size else None,
            "model_names": self.model_names,
            "model_paths": self.model_paths,
            "model_hashes": self.model_hashes,
            "providers_requested": self.providers_requested,
            "providers_used": self.providers_used,
            "provider_fallback_used": self.provider_fallback_used,
            "runtime": self.runtime,
            "model_evidence": self.model_evidence,
            "bubble_model_evidence": self.bubble_model_evidence,
            "text_area_model_evidence": self.text_area_model_evidence,
            "fused_containers": self.fused_containers,
            "memberships": self.memberships,
            "decisions": self.decisions,
            "region_model_links": self.region_model_links,
            "conflicts": self.conflicts,
            "fallback_used": self.fallback_used,
            "error": self.error,
            "runtime_sec": self.runtime_sec,
            "cache_enabled": self.cache_enabled,
            "cache_key": self.cache_key,
            "cache_hit": self.cache_hit,
            "cache_read_path": self.cache_read_path,
            "cache_write_path": self.cache_write_path,
            "cache_error": self.cache_error,
            "cache_invalidation_reason": self.cache_invalidation_reason,
        }

    def legacy_model_fusion_evidence(self) -> Dict[str, Any]:
        return {
            "backend": {
                "bubble_detection_version": self.version,
                "providers_requested": self.providers_requested,
                "kitsumed_session_providers": self.providers_used.get("kitsumed", []),
                "ogkalu_session_providers": self.providers_used.get("ogkalu", []),
                "kitsumed_provider_used": _primary_provider(self.providers_used.get("kitsumed", [])),
                "ogkalu_provider_used": _primary_provider(self.providers_used.get("ogkalu", [])),
                "kitsumed_latency_sec": self.runtime.get("latency_sec", {}).get("kitsumed", {}),
                "ogkalu_latency_sec": self.runtime.get("latency_sec", {}).get("ogkalu", {}),
                "model_paths": self.model_paths,
                "model_hashes": self.model_hashes,
                "provider_fallback_used": self.provider_fallback_used,
                "service_runtime_sec": self.runtime_sec,
                "cache_enabled": self.cache_enabled,
                "cache_key": self.cache_key,
                "cache_hit": self.cache_hit,
                "cache_read_path": self.cache_read_path,
                "cache_write_path": self.cache_write_path,
                "cache_error": self.cache_error,
                "cache_invalidation_reason": self.cache_invalidation_reason,
            },
            "kitsumed_speech_bubbles": self.bubble_model_evidence,
            "ogkalu_text_bubble_detections": self.text_area_model_evidence,
            "region_model_links": self.region_model_links,
            "fused_containers": self.fused_containers,
            "bubble_detection_service": self.to_dict(),
        }


@dataclass
class _BubbleDetectionRuntime:
    fusion_module: Any
    kitsumed_session: Any
    ogkalu_session: Any
    kitsumed_input_size: int
    kitsumed_class_names: Dict[int, str]
    ogkalu_labels: Dict[int, str]
    providers_requested: List[str]
    providers_available: List[str]
    kitsumed_providers: List[str]
    ogkalu_providers: List[str]
    kitsumed_model_hash: str
    ogkalu_model_hash: str


_RUNTIME_CACHE: Optional[_BubbleDetectionRuntime] = None


def run_bubble_detection(request: BubbleDetectionInput | Mapping[str, Any]) -> BubbleDetectionResult:
    """Run the model stack and return metadata-only bubble detection evidence.

    This function fails closed. On model/runtime/image errors it returns a
    result with ``generated=False`` and fallback status set, leaving callers free
    to continue deterministic pipeline behavior.
    """

    started = time.perf_counter()
    req = _coerce_input(request)
    image_path = Path(req.image_path)
    page_id = str(req.page_id or image_path.stem)
    fallback_result = _empty_result(page_id, image_path, req.image_size, started)
    cache_enabled = _cache_enabled()
    fallback_result.cache_enabled = cache_enabled

    try:
        runtime = _get_runtime()
        image = runtime.fusion_module.cv2.imread(str(image_path), runtime.fusion_module.cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read source image: {image_path}")

        image_h, image_w = image.shape[:2]
        image_size = req.image_size or (int(image_w), int(image_h))
        cache_state = _build_cache_state(req, image_path, image_size, runtime, cache_enabled)
        cached_result = _read_cached_result(cache_state, started)
        if cached_result is not None:
            return cached_result

        kitsumed_raw, kitsumed_latency = _run_kitsumed_once(runtime, image)
        ogkalu_raw, ogkalu_latency = _run_ogkalu_once(runtime, image)

        kitsumed_provider = _primary_provider(runtime.kitsumed_providers)
        ogkalu_provider = _primary_provider(runtime.ogkalu_providers)
        provider_fallback = kitsumed_provider != "CUDAExecutionProvider" or ogkalu_provider != "CUDAExecutionProvider"

        bubble_evidence = _serialize_kitsumed(page_id, kitsumed_raw, kitsumed_provider, kitsumed_latency, runtime)
        text_area_evidence = _serialize_ogkalu(page_id, ogkalu_raw, ogkalu_provider, ogkalu_latency, runtime, kitsumed_raw)
        region_links = runtime.fusion_module.link_regions(list(req.regions or []), kitsumed_raw, ogkalu_raw)
        fused_containers = runtime.fusion_module.build_fusion(page_id, kitsumed_raw, ogkalu_raw, region_links)
        memberships = _build_memberships(fused_containers, region_links)
        decisions = _build_decisions(memberships, fused_containers)
        conflicts = _build_service_conflicts(fused_containers)

        runtime_payload = {
            "provider_preference": list(PROVIDER_PREFERENCE),
            "providers_available": runtime.providers_available,
            "providers_requested": runtime.providers_requested,
            "providers_used": {
                "kitsumed": runtime.kitsumed_providers,
                "ogkalu": runtime.ogkalu_providers,
            },
            "provider_fallback_used": provider_fallback,
            "latency_sec": {
                "kitsumed": kitsumed_latency,
                "ogkalu": ogkalu_latency,
            },
            "counts": {
                "kitsumed": kitsumed_latency.get("counts", {}),
                "ogkalu_detections": len(ogkalu_raw),
                "fused_containers": len(fused_containers),
                "memberships": len(memberships),
                "conflicts": len(conflicts),
            },
            "mode": req.mode,
            "requested_models": list(req.requested_models),
            "thresholds": {
                "kitsumed_confidence": KITSUMED_CONFIDENCE_THRESHOLD,
                "kitsumed_nms_iou": KITSUMED_NMS_IOU_THRESHOLD,
                "kitsumed_mask": KITSUMED_MASK_THRESHOLD,
                "ogkalu_confidence": OGKALU_CONFIDENCE_THRESHOLD,
            },
        }

        result = BubbleDetectionResult(
            page_id=page_id,
            version=BUBBLE_DETECTION_VERSION,
            generated=True,
            image_path=str(image_path),
            image_size=image_size,
            model_names={"kitsumed": KITSUMED_MODEL_NAME, "ogkalu": OGKALU_MODEL_NAME},
            model_paths={"kitsumed": str(KITSUMED_MODEL), "ogkalu": str(OGKALU_MODEL)},
            model_hashes={"kitsumed": runtime.kitsumed_model_hash, "ogkalu": runtime.ogkalu_model_hash},
            providers_requested=runtime.providers_requested,
            providers_used={"kitsumed": runtime.kitsumed_providers, "ogkalu": runtime.ogkalu_providers},
            provider_fallback_used=provider_fallback,
            runtime=runtime_payload,
            model_evidence=bubble_evidence + text_area_evidence,
            bubble_model_evidence=bubble_evidence,
            text_area_model_evidence=text_area_evidence,
            fused_containers=fused_containers,
            memberships=memberships,
            decisions=decisions,
            region_model_links=region_links,
            conflicts=conflicts,
            fallback_used=provider_fallback,
            error=None,
            runtime_sec=round(time.perf_counter() - started, 6),
            cache_enabled=bool(cache_state.get("enabled")),
            cache_key=cache_state.get("key"),
            cache_hit=False,
            cache_read_path=str(cache_state.get("path")) if cache_state.get("path") and cache_state.get("enabled") else None,
            cache_write_path=str(cache_state.get("path")) if cache_state.get("path") and cache_state.get("enabled") else None,
            cache_error=cache_state.get("error"),
            cache_invalidation_reason=cache_state.get("invalidation_reason"),
            raw_kitsumed_detections=kitsumed_raw,
            raw_ogkalu_detections=ogkalu_raw,
        )
        _write_cached_result(cache_state, result)
        return result
    except Exception as exc:  # pragma: no cover - exercised through pipeline fallback validation
        fallback_result.error = f"{type(exc).__name__}: {exc}"
        fallback_result.runtime_sec = round(time.perf_counter() - started, 6)
        fallback_result.fallback_used = True
        fallback_result.provider_fallback_used = True
        fallback_result.runtime["error"] = fallback_result.error
        return fallback_result


def draw_bubble_detection_overlay(result: BubbleDetectionResult, output_path: Path | str, title: str | None = None) -> None:
    """Write a diagnostic overlay for a generated result."""

    if not result.generated:
        raise RuntimeError(result.error or "bubble detection result was not generated")
    if not result.raw_kitsumed_detections or not result.raw_ogkalu_detections:
        _draw_serialized_bubble_detection_overlay(result, output_path, title)
        return
    runtime = _get_runtime()
    image = runtime.fusion_module.cv2.imread(str(result.image_path), runtime.fusion_module.cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read source image: {result.image_path}")
    runtime.fusion_module.draw_fusion_overlay(
        image,
        result.raw_kitsumed_detections,
        result.raw_ogkalu_detections,
        result.region_model_links,
        result.fused_containers,
        Path(output_path),
        title or f"BubbleDetection service {result.page_id}",
    )


def _draw_serialized_bubble_detection_overlay(
    result: BubbleDetectionResult,
    output_path: Path | str,
    title: str | None,
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(f"Pillow unavailable for cached bubble overlay: {exc}") from exc

    image_path = Path(result.image_path)
    if not image_path.exists():
        raise RuntimeError(f"failed to read source image: {image_path}")
    base = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    for evidence in result.bubble_model_evidence:
        polygon = evidence.get("mask_polygon") or []
        if polygon:
            points = []
            for point in polygon:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    continue
                x, y = point[:2]
                if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                    points.append((int(x), int(y)))
            if len(points) >= 3:
                draw_overlay.polygon(points, fill=(40, 120, 255, 72))
    composed = Image.alpha_composite(base, overlay)
    draw = ImageDraw.Draw(composed)

    for evidence in result.bubble_model_evidence:
        box = _int_box(evidence.get("bbox_xyxy") or evidence.get("bbox"))
        if not box:
            continue
        draw.rectangle(box, outline=(35, 110, 255, 255), width=4)
        _draw_cached_overlay_label(
            draw,
            (box[0] + 2, max(0, box[1] - 18)),
            f"{evidence.get('model_evidence_id')} mask {float(evidence.get('confidence') or 0.0):.2f}",
            (0, 70, 210),
        )

    for evidence in result.text_area_model_evidence:
        box = _int_box(evidence.get("bbox_xyxy") or evidence.get("bbox"))
        if not box:
            continue
        class_name = str(evidence.get("class_name") or "")
        if class_name == "bubble":
            color = (0, 210, 230)
        elif class_name == "text_bubble":
            color = (0, 190, 70)
        else:
            color = (245, 140, 0)
        draw.rectangle(box, outline=color + (255,), width=3)
        _draw_cached_overlay_label(
            draw,
            (box[0] + 2, max(0, box[1] - 18)),
            f"{evidence.get('model_evidence_id')} {class_name} {float(evidence.get('confidence') or 0.0):.2f}",
            color,
        )

    conflict_regions = {
        str(region_id)
        for container in result.fused_containers
        if container.get("conflict_flags")
        for region_id in container.get("affected_current_region_ids", []) or []
    }
    for link in result.region_model_links:
        box = _int_box(link.get("bbox"))
        if not box:
            continue
        region_id = str(link.get("region_id") or "")
        if link.get("is_decorative_or_sfx"):
            draw.rectangle(box, outline=(230, 0, 180, 255), width=3)
            _draw_cached_overlay_label(draw, (box[0] + 2, max(0, box[1] - 18)), f"{region_id} preserve", (190, 0, 150))
        if region_id in conflict_regions:
            draw.rectangle(box, outline=(255, 0, 0, 255), width=5)
            _draw_cached_overlay_label(draw, (box[0] + 2, box[3] + 2), f"{region_id} conflict", (220, 0, 0))

    _draw_cached_overlay_label(
        draw,
        (10, 10),
        title or f"BubbleDetection service {result.page_id}",
        (0, 0, 0),
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    composed.convert("RGB").save(output_path, quality=92)


def _draw_cached_overlay_label(draw: Any, xy: Tuple[int, int], text: str, fill: Tuple[int, int, int]) -> None:
    try:
        from PIL import ImageFont

        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = None
    x, y = xy
    try:
        bbox = draw.textbbox((x, y), text, font=font)
    except Exception:
        bbox = (x, y, x + max(1, len(str(text))) * 8, y + 16)
    draw.rectangle(bbox, fill=(255, 255, 255, 255))
    draw.text((x, y), text, fill=fill + (255,), font=font)


def _int_box(value: Any) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = [int(round(float(item))) for item in value[:4]]
    except Exception:
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _coerce_input(request: BubbleDetectionInput | Mapping[str, Any]) -> BubbleDetectionInput:
    if isinstance(request, BubbleDetectionInput):
        return request
    image_path = request.get("image_path")
    if image_path is None:
        raise ValueError("BubbleDetectionInput requires image_path")
    page_id = str(request.get("page_id") or Path(str(image_path)).stem)
    image_size_value = request.get("image_size")
    image_size = None
    if isinstance(image_size_value, (list, tuple)) and len(image_size_value) >= 2:
        image_size = (int(image_size_value[0]), int(image_size_value[1]))
    return BubbleDetectionInput(
        page_id=page_id,
        image_path=image_path,
        image_size=image_size,
        regions=request.get("regions", []) or [],
        deterministic_diagnostics=request.get("deterministic_diagnostics", {}) or {},
        debug_page_dir=request.get("debug_page_dir"),
        mode=str(request.get("mode") or "diagnostic"),
        requested_models=request.get("requested_models", ["kitsumed_speech_bubble", "ogkalu_text_bubble"]) or [],
    )


def _empty_result(
    page_id: str,
    image_path: Path,
    image_size: Optional[Tuple[int, int]],
    started: float,
) -> BubbleDetectionResult:
    runtime_payload = {
        "provider_preference": list(PROVIDER_PREFERENCE),
        "providers_available": [],
        "providers_requested": [],
        "providers_used": {"kitsumed": [], "ogkalu": []},
        "provider_fallback_used": True,
        "latency_sec": {"kitsumed": {}, "ogkalu": {}},
        "counts": {},
    }
    return BubbleDetectionResult(
        page_id=page_id,
        version=BUBBLE_DETECTION_VERSION,
        generated=False,
        image_path=str(image_path),
        image_size=image_size,
        model_names={"kitsumed": KITSUMED_MODEL_NAME, "ogkalu": OGKALU_MODEL_NAME},
        model_paths={"kitsumed": str(KITSUMED_MODEL), "ogkalu": str(OGKALU_MODEL)},
        model_hashes={},
        providers_requested=[],
        providers_used={"kitsumed": [], "ogkalu": []},
        provider_fallback_used=True,
        runtime=runtime_payload,
        model_evidence=[],
        bubble_model_evidence=[],
        text_area_model_evidence=[],
        fused_containers=[],
        memberships=[],
        decisions=[],
        region_model_links=[],
        conflicts=[],
        fallback_used=True,
        error=None,
        runtime_sec=round(time.perf_counter() - started, 6),
    )


def _cache_enabled() -> bool:
    value = os.environ.get("MT_BUBBLE_DETECTION_CACHE", "").strip().lower()
    if value in {"0", "false", "no", "off"}:
        return False
    return True


def _cache_root() -> Path:
    configured = os.environ.get("MT_BUBBLE_DETECTION_CACHE_DIR", "").strip()
    if configured:
        return Path(configured)
    return ROOT / "output" / "bubble_detection_cache"


def _build_cache_state(
    req: BubbleDetectionInput,
    image_path: Path,
    image_size: Tuple[int, int],
    runtime: _BubbleDetectionRuntime,
    enabled: bool,
) -> Dict[str, Any]:
    identity: Dict[str, Any] = {
        "cache_schema_version": BUBBLE_DETECTION_CACHE_VERSION,
        "service_version": BUBBLE_DETECTION_VERSION,
        "page_id": str(req.page_id or image_path.stem),
        "source_image_path": str(image_path.resolve()),
        "source_image_sha256": _sha256_file(image_path),
        "source_image_stat": _file_stat_identity(image_path),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "model_paths": {"kitsumed": str(KITSUMED_MODEL.resolve()), "ogkalu": str(OGKALU_MODEL.resolve())},
        "model_hashes": {"kitsumed": runtime.kitsumed_model_hash, "ogkalu": runtime.ogkalu_model_hash},
        "provider_preference": list(PROVIDER_PREFERENCE),
        "providers_requested": list(runtime.providers_requested),
        "providers_used": {"kitsumed": list(runtime.kitsumed_providers), "ogkalu": list(runtime.ogkalu_providers)},
        "model_config": {
            "kitsumed_input_size": int(runtime.kitsumed_input_size),
            "kitsumed_class_names": {str(key): value for key, value in runtime.kitsumed_class_names.items()},
            "ogkalu_labels": {str(key): value for key, value in runtime.ogkalu_labels.items()},
            "ogkalu_config_path": str(OGKALU_CONFIG.resolve()),
            "ogkalu_config_sha256": _sha256_file(OGKALU_CONFIG) if OGKALU_CONFIG.exists() else None,
        },
        "requested_models": list(req.requested_models),
        "regions_signature": _regions_signature(req.regions),
    }
    key = hashlib.sha256(_stable_json_bytes(identity)).hexdigest()
    path = _cache_root() / key[:2] / f"{key}.json"
    state = {
        "enabled": bool(enabled),
        "identity": identity,
        "key": key,
        "path": path,
        "error": None,
        "invalidation_reason": None if enabled else "cache_disabled",
    }
    if enabled and not path.exists():
        state["invalidation_reason"] = "cache_file_missing"
    return state


def _read_cached_result(cache_state: Mapping[str, Any], started: float) -> BubbleDetectionResult | None:
    if not cache_state.get("enabled"):
        return None
    path = cache_state.get("path")
    if not isinstance(path, Path) or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        _validate_cache_payload(payload, cache_state)
        result_payload = payload["result"]
        result = _result_from_payload(result_payload)
        source_runtime_sec = result.runtime_sec
        result.runtime_sec = round(time.perf_counter() - started, 6)
        result.cache_enabled = True
        result.cache_key = str(cache_state.get("key") or "")
        result.cache_hit = True
        result.cache_read_path = str(path)
        result.cache_write_path = str(path)
        result.cache_error = None
        result.cache_invalidation_reason = None
        result.runtime.setdefault("cache", {})
        result.runtime["cache"].update(
            {
                "cache_schema_version": BUBBLE_DETECTION_CACHE_VERSION,
                "cache_hit": True,
                "cache_read_path": str(path),
                "source_runtime_sec": source_runtime_sec,
                "cache_read_runtime_sec": result.runtime_sec,
            }
        )
        return result
    except Exception as exc:
        if isinstance(cache_state, dict):
            cache_state["error"] = f"{type(exc).__name__}: {exc}"
            cache_state["invalidation_reason"] = "cache_read_failed"
        return None


def _write_cached_result(cache_state: Mapping[str, Any], result: BubbleDetectionResult) -> None:
    if not cache_state.get("enabled") or result.cache_hit:
        return
    path = cache_state.get("path")
    if not isinstance(path, Path):
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cache_schema_version": BUBBLE_DETECTION_CACHE_VERSION,
            "cache_key": cache_state.get("key"),
            "identity": cache_state.get("identity"),
            "result": result.to_dict(),
        }
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)
        result.cache_write_path = str(path)
        result.runtime.setdefault("cache", {})
        result.runtime["cache"].update(
            {
                "cache_schema_version": BUBBLE_DETECTION_CACHE_VERSION,
                "cache_hit": False,
                "cache_write_path": str(path),
            }
        )
    except Exception as exc:
        result.cache_error = f"{type(exc).__name__}: {exc}"
        result.runtime.setdefault("cache", {})
        result.runtime["cache"]["cache_write_error"] = result.cache_error


def _validate_cache_payload(payload: Mapping[str, Any], cache_state: Mapping[str, Any]) -> None:
    if payload.get("cache_schema_version") != BUBBLE_DETECTION_CACHE_VERSION:
        raise ValueError("cache_schema_version_mismatch")
    if payload.get("cache_key") != cache_state.get("key"):
        raise ValueError("cache_key_mismatch")
    identity = payload.get("identity")
    if identity != cache_state.get("identity"):
        raise ValueError("cache_identity_mismatch")
    result = payload.get("result")
    if not isinstance(result, Mapping):
        raise ValueError("cache_result_missing")
    if result.get("version") != BUBBLE_DETECTION_VERSION:
        raise ValueError("service_version_mismatch")
    expected_identity = cache_state.get("identity") or {}
    if result.get("model_hashes") != expected_identity.get("model_hashes"):
        raise ValueError("model_hash_mismatch")
    if result.get("providers_used") != expected_identity.get("providers_used"):
        raise ValueError("providers_used_mismatch")


def _result_from_payload(payload: Mapping[str, Any]) -> BubbleDetectionResult:
    image_size_value = payload.get("image_size")
    image_size = None
    if isinstance(image_size_value, (list, tuple)) and len(image_size_value) >= 2:
        image_size = (int(image_size_value[0]), int(image_size_value[1]))
    return BubbleDetectionResult(
        page_id=str(payload.get("page_id") or ""),
        version=str(payload.get("version") or BUBBLE_DETECTION_VERSION),
        generated=bool(payload.get("generated")),
        image_path=str(payload.get("image_path") or ""),
        image_size=image_size,
        model_names=dict(payload.get("model_names") or {}),
        model_paths=dict(payload.get("model_paths") or {}),
        model_hashes=dict(payload.get("model_hashes") or {}),
        providers_requested=list(payload.get("providers_requested") or []),
        providers_used=dict(payload.get("providers_used") or {}),
        provider_fallback_used=bool(payload.get("provider_fallback_used")),
        runtime=dict(payload.get("runtime") or {}),
        model_evidence=list(payload.get("model_evidence") or []),
        bubble_model_evidence=list(payload.get("bubble_model_evidence") or []),
        text_area_model_evidence=list(payload.get("text_area_model_evidence") or []),
        fused_containers=list(payload.get("fused_containers") or []),
        memberships=list(payload.get("memberships") or []),
        decisions=list(payload.get("decisions") or []),
        region_model_links=list(payload.get("region_model_links") or []),
        conflicts=list(payload.get("conflicts") or []),
        fallback_used=bool(payload.get("fallback_used")),
        error=payload.get("error"),
        runtime_sec=float(payload.get("runtime_sec") or 0.0),
        cache_enabled=bool(payload.get("cache_enabled")),
        cache_key=payload.get("cache_key"),
        cache_hit=bool(payload.get("cache_hit")),
        cache_read_path=payload.get("cache_read_path"),
        cache_write_path=payload.get("cache_write_path"),
        cache_error=payload.get("cache_error"),
        cache_invalidation_reason=payload.get("cache_invalidation_reason"),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_stat_identity(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _regions_signature(regions: Sequence[Mapping[str, Any]]) -> str:
    normalized = []
    for region in regions or []:
        if not isinstance(region, Mapping):
            continue
        render = region.get("render") if isinstance(region.get("render"), Mapping) else {}
        flags = region.get("flags") if isinstance(region.get("flags"), Mapping) else {}
        normalized.append(
            {
                "region_id": region.get("region_id"),
                "bbox": region.get("bbox"),
                "polygon": region.get("polygon"),
                "ocr_text": region.get("ocr_text"),
                "semantic_class": region.get("semantic_class") or region.get("type"),
                "cleanup_mode": region.get("cleanup_mode") or render.get("cleanup_mode"),
                "classification_reason": region.get("classification_reason") or render.get("classification_reason"),
                "flags": {str(key): flags.get(key) for key in sorted(flags)},
                "diagnostic_text_container_id": region.get("diagnostic_text_container_id"),
                "diagnostic_container_type": region.get("diagnostic_container_type"),
            }
        )
    normalized.sort(key=lambda item: str(item.get("region_id") or ""))
    return hashlib.sha256(_stable_json_bytes(normalized)).hexdigest()


def _stable_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _get_runtime() -> _BubbleDetectionRuntime:
    global _RUNTIME_CACHE
    if _RUNTIME_CACHE is not None:
        return _RUNTIME_CACHE

    fusion_module = _load_script_module("phase4b3_model_evidence_fusion", SCRIPTS_DIR / "phase4b3_model_evidence_fusion.py")
    available = list(fusion_module.ort.get_available_providers())
    requested = [provider for provider in PROVIDER_PREFERENCE if provider in available]
    if not requested:
        requested = available or ["CPUExecutionProvider"]

    kitsumed_session = fusion_module.ort.InferenceSession(str(KITSUMED_MODEL), providers=requested)
    ogkalu_session = fusion_module.ort.InferenceSession(str(OGKALU_MODEL), providers=requested)

    metadata = kitsumed_session.get_modelmeta().custom_metadata_map
    input_size = _parse_input_size(metadata.get("imgsz"))
    class_names = fusion_module.kit.parse_names(metadata.get("names", "{0: 'speech bubble'}"))
    labels = fusion_module.og.load_labels(OGKALU_CONFIG)
    kitsumed_hash = fusion_module.kit.sha256(KITSUMED_MODEL)
    ogkalu_hash = fusion_module.kit.sha256(OGKALU_MODEL)

    _RUNTIME_CACHE = _BubbleDetectionRuntime(
        fusion_module=fusion_module,
        kitsumed_session=kitsumed_session,
        ogkalu_session=ogkalu_session,
        kitsumed_input_size=input_size,
        kitsumed_class_names=class_names,
        ogkalu_labels=labels,
        providers_requested=requested,
        providers_available=available,
        kitsumed_providers=list(kitsumed_session.get_providers()),
        ogkalu_providers=list(ogkalu_session.get_providers()),
        kitsumed_model_hash=kitsumed_hash,
        ogkalu_model_hash=ogkalu_hash,
    )
    return _RUNTIME_CACHE


def _load_script_module(name: str, path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(f"_mt_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parse_input_size(value: Optional[str]) -> int:
    if not value:
        return 640
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (list, tuple)) and parsed:
            return int(parsed[0])
        return int(parsed)
    except Exception:
        return 640


def _primary_provider(providers: Sequence[str]) -> str:
    return providers[0] if providers else "unknown"


def _run_kitsumed_once(runtime: _BubbleDetectionRuntime, image: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    fusion = runtime.fusion_module
    tensor, prep = fusion.kit.letterbox(image, runtime.kitsumed_input_size)
    started = time.perf_counter()
    raw0, raw1 = runtime.kitsumed_session.run(None, {"images": tensor})
    elapsed = time.perf_counter() - started
    detections, counts = fusion.kit.decode_outputs(
        raw0,
        raw1,
        prep,
        image.shape[:2],
        runtime.kitsumed_class_names,
        KITSUMED_CONFIDENCE_THRESHOLD,
        KITSUMED_NMS_IOU_THRESHOLD,
        KITSUMED_MASK_THRESHOLD,
    )
    for idx, detection in enumerate(detections):
        detection["model_evidence_id"] = f"k{idx:03d}"
    return detections, {
        "latency_seconds": {
            "runs": [round(elapsed, 6)],
            "mean": round(elapsed, 6),
            "median": round(elapsed, 6),
            "min": round(elapsed, 6),
            "max": round(elapsed, 6),
        },
        "counts": counts,
        "measurement_mode": "single_debug_inference",
    }


def _run_ogkalu_once(runtime: _BubbleDetectionRuntime, image: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    fusion = runtime.fusion_module
    tensor, orig_target_sizes = fusion.og.preprocess(image)
    inputs = {"images": tensor, "orig_target_sizes": orig_target_sizes}
    started = time.perf_counter()
    outputs = runtime.ogkalu_session.run(None, inputs)
    elapsed = time.perf_counter() - started

    label_arr = fusion.np.asarray(outputs[0]).reshape(-1)
    boxes_arr = fusion.np.asarray(outputs[1]).reshape(-1, 4)
    scores_arr = fusion.np.asarray(outputs[2]).reshape(-1)
    height, width = image.shape[:2]
    detections: List[Dict[str, Any]] = []
    for idx, score in enumerate(scores_arr):
        if float(score) < OGKALU_CONFIDENCE_THRESHOLD:
            continue
        class_id = int(label_arr[idx])
        class_name = runtime.ogkalu_labels.get(class_id, f"class_{class_id}")
        box = fusion.og.normalize_box(boxes_arr[idx], width=width, height=height)
        detections.append(
            {
                "model_evidence_id": f"o{len(detections):03d}",
                "anchor_index": int(idx),
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(float(score), 6),
                "bbox_xyxy": [round(v, 2) for v in box],
                "area_px": round(fusion.bbox_area(box), 2),
            }
        )
    return detections, {
        "latency_seconds": {
            "runs": [round(elapsed, 6)],
            "mean": round(elapsed, 6),
            "median": round(elapsed, 6),
            "min": round(elapsed, 6),
            "max": round(elapsed, 6),
        },
        "measurement_mode": "single_debug_inference",
    }


def _serialize_kitsumed(
    page_id: str,
    detections: Sequence[Mapping[str, Any]],
    provider: str,
    latency: Mapping[str, Any],
    runtime: _BubbleDetectionRuntime,
) -> List[Dict[str, Any]]:
    fallback_used = provider != "CUDAExecutionProvider"
    records: List[Dict[str, Any]] = []
    for detection in detections:
        record = BubbleModelEvidence(
            model_evidence_id=str(detection["model_evidence_id"]),
            model_name=KITSUMED_MODEL_NAME,
            model_role=KITSUMED_MODEL_ROLE,
            model_path=str(KITSUMED_MODEL),
            model_hash=runtime.kitsumed_model_hash,
            provider=provider,
            class_name=str(detection.get("class_name") or "speech bubble"),
            confidence=float(detection.get("confidence") or 0.0),
            bbox_xyxy=_float_list(detection.get("bbox_xyxy")),
            mask_ref=None,
            mask_bbox_xyxy=_float_list(detection.get("mask_bbox_xyxy")),
            mask_area_px=float(detection.get("mask_area_px") or 0.0),
            mask_polygon=runtime.fusion_module.contour_polygon(detection["mask"]),
            latency_sec=float((latency.get("latency_seconds") or {}).get("mean") or 0.0),
            fallback_used=fallback_used,
        ).to_dict()
        record.update(
            {
                "evidence_id": record["model_evidence_id"],
                "page": page_id,
                "source_model": record["model_name"],
                "evidence_type": "BubbleSegmentationEvidence",
                "bbox": record["bbox_xyxy"],
                "mask_bbox": record["mask_bbox_xyxy"],
                "latency_mean_sec": record["latency_sec"],
            }
        )
        records.append(record)
    return records


def _serialize_ogkalu(
    page_id: str,
    detections: Sequence[Mapping[str, Any]],
    provider: str,
    latency: Mapping[str, Any],
    runtime: _BubbleDetectionRuntime,
    kitsumed_raw: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    fallback_used = provider != "CUDAExecutionProvider"
    records: List[Dict[str, Any]] = []
    for detection in detections:
        linked_masks = _linked_kitsumed_mask_ids(detection, kitsumed_raw, runtime)
        conflict_flags: List[str] = []
        record = TextAreaModelEvidence(
            model_evidence_id=str(detection["model_evidence_id"]),
            model_name=OGKALU_MODEL_NAME,
            model_role=OGKALU_MODEL_ROLE,
            model_path=str(OGKALU_MODEL),
            model_hash=runtime.ogkalu_model_hash,
            provider=provider,
            class_name=str(detection["class_name"]),
            confidence=float(detection.get("confidence") or 0.0),
            bbox_xyxy=_float_list(detection.get("bbox_xyxy")),
            linked_bubble_mask_ids=linked_masks,
            conflict_flags=conflict_flags,
            latency_sec=float((latency.get("latency_seconds") or {}).get("mean") or 0.0),
            fallback_used=fallback_used,
        ).to_dict()
        record.update(
            {
                "evidence_id": record["model_evidence_id"],
                "page": page_id,
                "source_model": record["model_name"],
                "evidence_type": "TextBubbleDetectionEvidence",
                "bbox": record["bbox_xyxy"],
                "latency_mean_sec": record["latency_sec"],
            }
        )
        records.append(record)
    return records


def _linked_kitsumed_mask_ids(
    ogkalu_detection: Mapping[str, Any],
    kitsumed_raw: Sequence[Mapping[str, Any]],
    runtime: _BubbleDetectionRuntime,
) -> List[str]:
    box = _float_list(ogkalu_detection.get("bbox_xyxy"))
    linked: List[str] = []
    for detection in kitsumed_raw:
        mask = detection.get("mask")
        if mask is None:
            continue
        ratio = runtime.fusion_module.mask_overlap_ratio(box, mask)
        center_hit = runtime.fusion_module.center_inside_mask(box, mask)
        if ratio >= 0.08 or center_hit:
            linked.append(str(detection.get("model_evidence_id")))
    return linked


def _build_memberships(
    fused_containers: Sequence[Mapping[str, Any]],
    region_links: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    links_by_region = {str(link.get("region_id")): link for link in region_links if link.get("region_id") is not None}
    memberships: List[Dict[str, Any]] = []
    for container in fused_containers:
        container_id = str(container.get("fused_container_id") or "")
        if not container_id:
            continue
        confidence = _ownership_confidence(container)
        for region_id in [str(rid) for rid in container.get("affected_current_region_ids", []) if rid is not None]:
            link = links_by_region.get(region_id, {})
            k_links = link.get("kitsumed_links", []) or []
            o_links = link.get("ogkalu_links", []) or []
            inside_ratio = _best_inside_ratio(k_links, o_links)
            center_inside = any(bool(item.get("center_inside_mask") or item.get("center_inside_bbox")) for item in k_links + o_links)
            if container.get("conflict_flags"):
                membership_type = "conflict"
            elif center_inside or inside_ratio >= 0.65:
                membership_type = "inside"
            elif inside_ratio > 0.0:
                membership_type = "overlap"
            else:
                membership_type = "near"
            reason_codes = sorted(set(_as_list(container.get("reason_codes")) + [f"membership_type:{membership_type}"]))
            memberships.append(
                TextContainerMembership(
                    region_id=region_id,
                    container_id=container_id,
                    membership_type=membership_type,
                    inside_ratio=round(inside_ratio, 4),
                    center_inside=center_inside,
                    ownership_confidence=confidence,
                    ownership_reason_codes=reason_codes,
                    must_not_mutate=True,
                ).to_dict()
            )
    return memberships


def _build_decisions(
    memberships: Sequence[Mapping[str, Any]],
    fused_containers: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    containers = {str(container.get("fused_container_id")): container for container in fused_containers}
    decisions: List[Dict[str, Any]] = []
    for membership in memberships:
        container = containers.get(str(membership.get("container_id")), {})
        conflict_flags = _as_list(container.get("conflict_flags"))
        confidence = str(container.get("confidence") or "low")
        if conflict_flags:
            status = "blocked"
            supported_actions = ["review_conflict"]
            blocked_actions = ["automatic_route_mutation", "automatic_cleanup_mutation", "automatic_render_mutation"]
        elif confidence == "high":
            status = "supported"
            supported_actions = ["text_container_only", "ownership_hint", "render_constraint_hint"]
            blocked_actions = ["automatic_mutation_without_explicit_proof_flag"]
        else:
            status = "review_only"
            supported_actions = ["review_only"]
            blocked_actions = ["automatic_mutation_without_explicit_proof_flag"]
        decisions.append(
            BubbleDetectionDecision(
                region_id=str(membership.get("region_id")),
                decision_type="review_only",
                decision_status=status,
                supported_actions=supported_actions,
                blocked_actions=blocked_actions,
                reason_codes=sorted(set(_as_list(container.get("reason_codes")) + _as_list(membership.get("ownership_reason_codes")))),
                would_change_behavior=False,
                requires_flag="future_explicit_proof_flag",
                requires_visual_review=True,
            ).to_dict()
        )
    return decisions


def _build_service_conflicts(fused_containers: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    conflicts: List[Dict[str, Any]] = []
    for container in fused_containers:
        flags = _as_list(container.get("conflict_flags"))
        if not flags:
            continue
        conflicts.append(
            {
                "container_id": container.get("fused_container_id"),
                "container_type": container.get("fused_container_type"),
                "linked_region_ids": container.get("affected_current_region_ids", []),
                "conflict_flags": flags,
                "reason_codes": container.get("reason_codes", []),
                "decision_status": "review_only_conflict",
                "would_change_behavior": False,
            }
        )
    return conflicts


def _ownership_confidence(container: Mapping[str, Any]) -> float:
    confidence = str(container.get("confidence") or "").lower()
    if confidence == "high":
        return 0.9
    if confidence == "medium":
        return 0.6
    if confidence == "low":
        return 0.3
    return 0.0


def _best_inside_ratio(k_links: Sequence[Mapping[str, Any]], o_links: Sequence[Mapping[str, Any]]) -> float:
    values: List[float] = []
    for item in k_links:
        values.append(_safe_float(item.get("mask_overlap_ratio")) or 0.0)
    for item in o_links:
        values.append(_safe_float(item.get("bbox_overlap_ratio")) or 0.0)
    return max(values) if values else 0.0


def _float_list(value: Any) -> List[float]:
    if not isinstance(value, (list, tuple)):
        return []
    result: List[float] = []
    for item in value:
        number = _safe_float(item)
        if number is not None:
            result.append(round(number, 4))
    return result


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]
