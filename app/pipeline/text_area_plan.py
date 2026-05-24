# -*- coding: utf-8 -*-
"""BubbleDetection-first text-area planning contracts and helpers.

This module is intentionally model-runtime free. BubbleDetection owns visual
model inference; the controller and debug artifacts use this module to turn the
service result into a page-level plan for scoped detection/OCR decisions.
"""
from __future__ import annotations

import copy
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    cv2 = None

try:
    import numpy as np
except Exception:  # pragma: no cover - optional runtime dependency
    np = None


TEXT_AREA_PLAN_VERSION = "bubble_detection_default_text_area_plan_v1"

CONTAINER_SPEECH = "speech_bubble"
CONTAINER_CAPTION = "caption_background"
CONTAINER_SFX = "sfx_decorative_art"
CONTAINER_UNKNOWN = "unknown_fallback"

ROUTE_TRANSLATE_SPEECH = "translate_speech"
ROUTE_TRANSLATE_CAPTION = "translate_caption_background"
ROUTE_PRESERVE_SFX = "preserve_sfx_decorative"
ROUTE_REVIEW_FALLBACK = "review_or_fallback"

AUTH_CLEANUP_TRANSLATE_SPEECH = "cleanup_translate_speech"
AUTH_CLEANUP_TRANSLATE_BACKGROUND = "cleanup_translate_background"
AUTH_CLEANUP_TRANSLATE_CAPTION = "cleanup_translate_caption"
AUTH_PROTECT_SFX_DECORATIVE = "protect_sfx_decorative"
AUTH_PROTECT_ART_OR_NON_TEXT = "protect_art_or_non_text"
AUTH_REVIEW_UNKNOWN_NOT_CLEANUP = "review_unknown_not_cleanup"
AUTH_OUTSIDE_CLEANUP_SCOPE = "outside_cleanup_scope"
AUTH_AMBIGUOUS_COMPONENT_OWNER = "ambiguous_component_owner"

TEXT_AREA_COMPONENT_AUTHORIZATION_MAP_VERSION = "text_area_component_authorization_map_v1"

COMPONENT_AUTHORIZATION_STATES = {
    AUTH_CLEANUP_TRANSLATE_SPEECH,
    AUTH_CLEANUP_TRANSLATE_BACKGROUND,
    AUTH_CLEANUP_TRANSLATE_CAPTION,
    AUTH_PROTECT_SFX_DECORATIVE,
    AUTH_PROTECT_ART_OR_NON_TEXT,
    AUTH_REVIEW_UNKNOWN_NOT_CLEANUP,
    AUTH_OUTSIDE_CLEANUP_SCOPE,
    AUTH_AMBIGUOUS_COMPONENT_OWNER,
}

COMPONENT_AUTHORIZATION_COLORS = {
    AUTH_CLEANUP_TRANSLATE_SPEECH: "green",
    AUTH_CLEANUP_TRANSLATE_BACKGROUND: "green",
    AUTH_CLEANUP_TRANSLATE_CAPTION: "green",
    AUTH_PROTECT_SFX_DECORATIVE: "red",
    AUTH_PROTECT_ART_OR_NON_TEXT: "purple",
    AUTH_REVIEW_UNKNOWN_NOT_CLEANUP: "gray",
    AUTH_OUTSIDE_CLEANUP_SCOPE: "blue",
    AUTH_AMBIGUOUS_COMPONENT_OWNER: "orange",
}

DETECTION_SCOPED = "scoped"
DETECTION_COMPATIBILITY_FALLBACK = "compatibility_fallback"
DETECTION_BLOCKED = "blocked_by_text_area_plan"
DETECTION_CAPTION_RECOVERY = "caption_container_text_instance_recovery"

TOP_BAND_CAPTION_SEARCH_REASON = "text_area_plan:deterministic_top_band_caption_search"
SIDE_CAPTION_SEARCH_REASON = "text_area_plan:deterministic_vertical_side_caption_search"

DECORATIVE_REGION_REASONS = {
    "nonbubble_short_kana_art_text_candidate",
    "nonbubble_short_reaction_art_text_candidate",
    "nonbubble_short_reaction_art_sfx_candidate",
    "nonbubble_breath_sfx_art_text_candidate",
    "short_reaction_without_visual_speech_ownership",
    "medium_large_katakana_sfx_candidate",
    "low_conf_dark_short_art_sfx_candidate",
    "large_low_confidence_nonbubble_sfx_candidate",
    "large_short_decorative_sfx_candidate",
    "text_area_review_only_unknown_not_auto_translated",
}

CAPTION_REGION_REASONS = {
    "top_row_background_caption_candidate",
    "top_row_caption_fragment_candidate",
}

SPEECH_REGION_REASONS = {
    "bubble_contained_short_laugh_speech",
    "speech_bubble_missed_text_recovery",
    "bubble_local_nested_speech_fragment_ownership",
    "adjacent_vertical_speech_text_conservation_recovery",
}


@dataclass
class TextAreaPlanEvidence:
    source: str
    source_ids: List[str] = field(default_factory=list)
    confidence: float | str | None = None
    reason_codes: List[str] = field(default_factory=list)
    conflict_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "source_ids": list(self.source_ids),
            "confidence": self.confidence,
            "reason_codes": list(self.reason_codes),
            "conflict_flags": list(self.conflict_flags),
        }


@dataclass
class TextAreaRouteIntent:
    route_intent: str
    translation_eligible: bool
    preserve_source: bool
    human_review_required: bool
    reason_codes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class TextAreaFallbackReason:
    reason: str
    detail: str = ""
    safe_to_ocr: bool = False
    safe_to_translate: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class TextAreaContainer:
    container_id: str
    page_id: str
    container_type: str
    bbox: List[int]
    mask_ref: Optional[str] = None
    mask_summary: Dict[str, Any] = field(default_factory=dict)
    source_model_ids: List[str] = field(default_factory=list)
    confidence: float | str | None = None
    confidence_tier: str = "low"
    route_intent: str = ROUTE_REVIEW_FALLBACK
    ocr_eligible: bool = True
    comic_text_detector_scope_eligible: bool = True
    fallback_reason: Optional[str] = None
    evidence_reason_codes: List[str] = field(default_factory=list)
    conflict_flags: List[str] = field(default_factory=list)
    debug_behavior_marker: str = "text_area_plan_default_owner"
    would_change_behavior: bool = False
    human_review_required: bool = False
    ocr_eligibility_reason: str = ""
    text_area_pre_ocr_authority: bool = True
    text_area_enriched_from_region: bool = False
    cleanup_authorization: str = ""
    must_not_mutate: bool = False
    protection_reason: str = ""
    pre_ocr_authority: bool = True
    source_stage: str = "text_area_plan"
    parent_source_evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class TextAreaScope:
    scope_id: str
    page_id: str
    container_id: str
    bbox: List[int]
    container_type: str
    route_intent: str
    ocr_eligible: bool
    comic_text_detector_scope_eligible: bool
    fallback_reason: Optional[str] = None
    source: str = "text_area_plan"
    ocr_eligibility_reason: str = ""
    text_area_pre_ocr_authority: bool = True
    text_area_enriched_from_region: bool = False
    cleanup_authorization: str = ""
    must_not_mutate: bool = False
    protection_reason: str = ""
    pre_ocr_authority: bool = True
    source_stage: str = "text_area_plan"

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class ScopedDetectionCandidate:
    detection_id: str
    page_id: str
    bbox: List[int]
    polygon: List[List[float]]
    confidence: float
    text_area_container_id: Optional[str]
    container_type: str
    route_intent: str
    ocr_eligible: bool
    detection_source: str
    fallback_reason: Optional[str]
    reason_codes: List[str] = field(default_factory=list)
    conflict_flags: List[str] = field(default_factory=list)
    text_area_pre_ocr_authority: bool = True
    text_area_enriched_from_region: bool = False
    would_change_behavior: bool = False
    cleanup_authorization: str = ""
    must_not_mutate: bool = False
    protection_reason: str = ""
    pre_ocr_authority: bool = True
    source_stage: str = "text_area_plan"

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class ScopedOcrCandidate:
    ocr_candidate_id: str
    page_id: str
    region_id: str
    bbox: List[int]
    text_area_container_id: Optional[str]
    container_type: str
    route_intent: str
    crop_source: str
    fallback_reason: Optional[str]
    ocr_text: str
    ocr_confidence: float
    accepted: bool
    reason_codes: List[str] = field(default_factory=list)
    text_area_pre_ocr_authority: bool = True
    text_area_enriched_from_region: bool = False
    would_change_behavior: bool = False
    cleanup_authorization: str = ""
    must_not_mutate: bool = False
    protection_reason: str = ""
    pre_ocr_authority: bool = True
    source_stage: str = "text_area_plan"

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class TextAreaPlanRuntime:
    version: str = TEXT_AREA_PLAN_VERSION
    generated: bool = False
    runtime_sec: float = 0.0
    bubble_detection_generated: bool = False
    bubble_detection_cache_hit: Optional[bool] = None
    provider_fallback_used: Optional[bool] = None
    true_scoped_detector_available: bool = False
    compatibility_mode: str = "scoped_detector_by_text_area_plan"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class TextAreaPlan:
    page_id: str
    image_path: str
    image_size: Tuple[int, int]
    version: str = TEXT_AREA_PLAN_VERSION
    generated: bool = False
    containers: List[TextAreaContainer] = field(default_factory=list)
    scopes: List[TextAreaScope] = field(default_factory=list)
    fallback_reasons: List[TextAreaFallbackReason] = field(default_factory=list)
    evidence: List[TextAreaPlanEvidence] = field(default_factory=list)
    runtime: TextAreaPlanRuntime = field(default_factory=TextAreaPlanRuntime)
    summary: Dict[str, Any] = field(default_factory=dict)
    stage: str = "pre_ocr"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_id": self.page_id,
            "image_path": self.image_path,
            "image_size": list(self.image_size),
            "version": self.version,
            "generated": self.generated,
            "containers": [item.to_dict() for item in self.containers],
            "scopes": [item.to_dict() for item in self.scopes],
            "fallback_reasons": [item.to_dict() for item in self.fallback_reasons],
            "evidence": [item.to_dict() for item in self.evidence],
            "runtime": self.runtime.to_dict(),
            "summary": dict(self.summary),
            "stage": self.stage,
        }


@dataclass
class TextAreaComponentAuthorizationRecord:
    page_id: str
    component_id: str
    component_bbox: List[int]
    component_pixel_count: int
    authorization_state: str
    cleanup_authorization: str
    route_intent: str
    must_not_mutate: bool
    owning_container_ids: List[str] = field(default_factory=list)
    protection_container_ids: List[str] = field(default_factory=list)
    source_stage: str = "text_area_plan_component_authorization"
    confidence_tier: str = "low"
    reason_codes: List[str] = field(default_factory=list)
    conflict_flags: List[str] = field(default_factory=list)
    review_required: bool = False
    visual_debug_color: str = "gray"
    group_id: str = ""
    group_authorization_state: str = ""
    label: int = 0
    owner_cleanup_job_id: str = ""
    scope_cleanup_job_ids: List[str] = field(default_factory=list)
    candidate_cleanup_job_ids: List[str] = field(default_factory=list)
    owning_region_ids: List[str] = field(default_factory=list)
    protection_region_ids: List[str] = field(default_factory=list)
    overlap_pixels: int = 0
    overlap_ratio: float = 0.0
    protected_overlap_pixels: int = 0
    protected_overlap_ratio: float = 0.0
    centroid: List[float] = field(default_factory=list)
    candidate_container_ids: List[str] = field(default_factory=list)
    candidate_region_ids: List[str] = field(default_factory=list)
    sourceglyph_overlap_pixels: int = 0
    sourceglyph_missing: bool = True
    ambiguity_reasons: List[str] = field(default_factory=list)
    semantic_authorization_state: str = ""
    semantic_visual_color: str = ""
    job_binding_state: str = ""
    job_binding_failure_reason: str = ""
    final_mask_authorization_state: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class TextAreaComponentAuthorizationMap:
    page_id: str
    image_size: Tuple[int, int] | List[int] | None
    version: str = TEXT_AREA_COMPONENT_AUTHORIZATION_MAP_VERSION
    components: List[TextAreaComponentAuthorizationRecord] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_audit_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "page_id": self.page_id,
            "image_size": list(self.image_size) if self.image_size else None,
            "components": [record.to_dict() for record in self.components],
            "errors": list(self.errors),
            "summary": dict(self.summary),
        }

    def to_dict(self) -> Dict[str, Any]:
        return self.to_audit_dict()


def build_text_area_component_authorization_map(
    *,
    page_id: str,
    text_foreground_segmentation: Any,
    text_area_plan: Any = None,
    page_region_records: Sequence[Mapping[str, Any]] | None = None,
    cleanup_jobs: Sequence[Any] | None = None,
) -> TextAreaComponentAuthorizationMap:
    """Project TextAreaPlan semantic authorization onto CTD components.

    This is the only production handoff that decides whether a refined CTD
    component is cleanup-owned, protected, review-only, outside-scope, or
    ambiguous. CleanupMask consumes this map without reinterpreting semantics.
    """

    mask, image_size, mask_error = _component_auth_segmentation_mask(text_foreground_segmentation)
    errors: List[str] = []
    if mask_error:
        errors.append(mask_error)
    if mask is None or np is None:
        return TextAreaComponentAuthorizationMap(
            page_id=str(page_id),
            image_size=image_size,
            components=[],
            errors=errors or ["text_foreground_segmentation_missing"],
            summary={
                "component_count": 0,
                "state_counts": {},
                "semantic_authority": "text_area_plan",
                "segmentation_source": "text_foreground_segmentation",
                "component_classification_complete": False,
            },
        )

    binary = (mask > 0).astype(np.uint8)
    labels, stats, centroids = _component_auth_connected_components(binary)
    plan_dict = _component_auth_plan_dict(text_area_plan)
    scopes = _component_auth_scopes(
        plan=plan_dict,
        page_region_records=page_region_records or [],
        cleanup_jobs=cleanup_jobs or [],
        mask_shape=binary.shape,
    )

    components: List[TextAreaComponentAuthorizationRecord] = []
    for label in range(1, int(len(stats))):
        x, y, w, h, pixel_count = [int(item) for item in stats[label][:5]]
        if pixel_count <= 0:
            continue
        component_mask = labels == label
        candidates = _component_auth_candidates(component_mask, pixel_count, centroids[label], scopes)
        record = _component_auth_record(
            page_id=str(page_id),
            component_index=len(components),
            label=label,
            bbox=[x, y, x + w, y + h],
            pixel_count=pixel_count,
            centroid=centroids[label],
            candidates=candidates,
        )
        components.append(record)

    _component_auth_apply_vertical_review_text_groups(components)
    _component_auth_apply_orphan_text_near_cleanup_groups(components)
    _component_auth_apply_protected_sfx_grouping(components)
    _component_auth_apply_large_decorative_review_rule(components)
    _component_auth_apply_large_decorative_review_groups(components)
    _component_auth_apply_unowned_display_neighbor_conflicts(components)
    _component_auth_apply_protected_sfx_grouping(components)
    _component_auth_apply_review_only_caption_guard(components)
    _component_auth_assign_groups(str(page_id), components)
    state_counts: Dict[str, int] = {}
    for record in components:
        state_counts[record.authorization_state] = state_counts.get(record.authorization_state, 0) + 1
    return TextAreaComponentAuthorizationMap(
        page_id=str(page_id),
        image_size=image_size or [int(binary.shape[1]), int(binary.shape[0])],
        components=components,
        errors=errors,
        summary={
            "component_count": len(components),
            "state_counts": state_counts,
            "semantic_authority": "text_area_plan",
            "segmentation_source": "text_foreground_segmentation",
            "component_classification_complete": True,
            "component_record_count_matches_ctd": True,
            "text_area_scope_count": len(scopes),
            "cleanup_job_count": len(cleanup_jobs or []),
        },
    )


def _component_auth_segmentation_mask(segmentation: Any) -> tuple[Any | None, List[int] | None, str]:
    if np is None:
        return None, None, "numpy_unavailable_for_component_authorization"
    image_size = _component_auth_image_size(segmentation)
    mask = _component_auth_get_value(segmentation, "refined_mask")
    if mask is None:
        mask = _component_auth_get_value(segmentation, "mask")
    if mask is None:
        mask_ref = str(
            _component_auth_get_value(segmentation, "refined_mask_ref")
            or _component_auth_get_value(segmentation, "mask_ref")
            or ""
        )
        if mask_ref:
            try:
                from PIL import Image

                path = Path(mask_ref)
                if path.exists():
                    mask = np.asarray(Image.open(path).convert("L"))
            except Exception:
                mask = None
    if mask is None:
        return None, image_size, "text_foreground_segmentation_missing_refined_mask"
    try:
        arr = np.asarray(mask)
    except Exception:
        return None, image_size, "text_foreground_segmentation_mask_unreadable"
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.size <= 0:
        return None, image_size, "text_foreground_segmentation_mask_empty"
    if image_size is None:
        image_size = [int(arr.shape[1]), int(arr.shape[0])]
    return (arr > 0).astype(np.uint8), image_size, ""


def _component_auth_image_size(segmentation: Any) -> List[int] | None:
    value = _component_auth_get_value(segmentation, "image_size")
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return [int(value[0]), int(value[1])]
        except Exception:
            return None
    return None


def _component_auth_get_value(source: Any, key: str) -> Any:
    if source is None:
        return None
    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)


def _component_auth_connected_components(binary: Any) -> tuple[Any, Any, Any]:
    if cv2 is not None:
        _label_count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        return labels.astype(np.int32), stats, centroids
    height, width = binary.shape[:2]
    labels = np.zeros((height, width), dtype=np.int32)
    stats = [[0, 0, 0, 0, 0]]
    centroids = [[0.0, 0.0]]
    label = 0
    for y in range(height):
        for x in range(width):
            if binary[y, x] <= 0 or labels[y, x] != 0:
                continue
            label += 1
            stack = [(x, y)]
            labels[y, x] = label
            xs: List[int] = []
            ys: List[int] = []
            while stack:
                sx, sy = stack.pop()
                xs.append(sx)
                ys.append(sy)
                for ny in range(max(0, sy - 1), min(height, sy + 2)):
                    for nx in range(max(0, sx - 1), min(width, sx + 2)):
                        if binary[ny, nx] > 0 and labels[ny, nx] == 0:
                            labels[ny, nx] = label
                            stack.append((nx, ny))
            stats.append([min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1, len(xs)])
            centroids.append([sum(xs) / float(max(1, len(xs))), sum(ys) / float(max(1, len(ys)))])
    return labels, np.asarray(stats, dtype=np.int32), np.asarray(centroids, dtype=float)


def _component_auth_plan_dict(text_area_plan: Any) -> Dict[str, Any]:
    if text_area_plan is None:
        return {}
    if isinstance(text_area_plan, Mapping):
        return dict(text_area_plan)
    if hasattr(text_area_plan, "to_dict"):
        try:
            value = text_area_plan.to_dict()
            if isinstance(value, Mapping):
                return dict(value)
        except Exception:
            return {}
    return {}


def _component_auth_scopes(
    *,
    plan: Mapping[str, Any],
    page_region_records: Sequence[Mapping[str, Any]],
    cleanup_jobs: Sequence[Any],
    mask_shape: tuple[int, int],
) -> List[Dict[str, Any]]:
    region_jobs: Dict[str, List[str]] = {}
    container_jobs: Dict[str, List[str]] = {}
    for job in cleanup_jobs or []:
        job_id = str(getattr(job, "cleanup_job_id", "") or "")
        if not job_id:
            continue
        for region_id in getattr(job, "target_region_ids", []) or []:
            region_jobs.setdefault(str(region_id), []).append(job_id)
        container_id = str(getattr(job, "text_area_container_id", "") or "")
        if container_id:
            container_jobs.setdefault(container_id, []).append(job_id)

    scopes: List[Dict[str, Any]] = []
    for index, container in enumerate(plan.get("containers") or []):
        if isinstance(container, Mapping):
            scope = _component_auth_scope(
                source=container,
                source_kind="text_area_container",
                fallback_id=f"container_{index:04d}",
                bbox_mode="xywh",
                mask_shape=mask_shape,
                region_jobs=region_jobs,
                container_jobs=container_jobs,
            )
            if scope:
                scopes.append(scope)
    for index, scope_record in enumerate(plan.get("scopes") or []):
        if isinstance(scope_record, Mapping):
            scope = _component_auth_scope(
                source=scope_record,
                source_kind="text_area_scope",
                fallback_id=f"scope_{index:04d}",
                bbox_mode="xywh",
                mask_shape=mask_shape,
                region_jobs=region_jobs,
                container_jobs=container_jobs,
            )
            if scope:
                scopes.append(scope)
    for index, region in enumerate(page_region_records or []):
        if isinstance(region, Mapping):
            scope = _component_auth_scope(
                source=region,
                source_kind="region_record",
                fallback_id=f"region_{index:04d}",
                bbox_mode="region",
                mask_shape=mask_shape,
                region_jobs=region_jobs,
                container_jobs=container_jobs,
            )
            if scope:
                scopes.append(scope)
    return scopes


def _component_auth_scope(
    *,
    source: Mapping[str, Any],
    source_kind: str,
    fallback_id: str,
    bbox_mode: str,
    mask_shape: tuple[int, int],
    region_jobs: Mapping[str, Sequence[str]],
    container_jobs: Mapping[str, Sequence[str]],
) -> Dict[str, Any] | None:
    bbox_value = source.get("bbox") or source.get("xyxy") or source.get("bounds")
    bbox = (
        _component_auth_xywh_to_xyxy(bbox_value, mask_shape)
        if bbox_mode == "xywh"
        else _component_auth_valid_or_xywh_bbox(bbox_value, mask_shape)
    )
    if not bbox:
        return None
    auth = str(source.get("cleanup_authorization") or source.get("text_area_cleanup_authorization") or "")
    protection_reason = str(source.get("protection_reason") or source.get("text_area_protection_reason") or "")
    must_not_mutate = bool(source.get("must_not_mutate") or source.get("text_area_must_not_mutate"))
    if not auth:
        try:
            auth, protection_reason_from_container, must_not_mutate_from_container = _cleanup_authorization_for_container(source)
            protection_reason = protection_reason or protection_reason_from_container
            must_not_mutate = must_not_mutate or must_not_mutate_from_container
        except Exception:
            auth = _component_auth_infer_authorization(source)
    if auth not in COMPONENT_AUTHORIZATION_STATES:
        auth = _component_auth_infer_authorization(source)
    family = _component_auth_family(auth)
    container_id = str(source.get("container_id") or source.get("text_area_container_id") or source.get("scope_id") or fallback_id)
    region_id = str(source.get("region_id") or source.get("id") or "")
    job_ids = []
    for job_id in region_jobs.get(region_id, []) if region_id else []:
        if job_id not in job_ids:
            job_ids.append(str(job_id))
    for job_id in container_jobs.get(container_id, []) if container_id else []:
        if job_id not in job_ids:
            job_ids.append(str(job_id))
    reason_codes = _component_auth_list(
        source.get("reason_codes")
        or source.get("evidence_reason_codes")
        or source.get("text_area_reason_codes")
        or []
    )
    if protection_reason and protection_reason not in reason_codes:
        reason_codes.append(protection_reason)
    conflict_flags = _component_auth_list(source.get("conflict_flags") or source.get("text_area_conflict_flags") or [])
    return {
        "source_kind": source_kind,
        "bbox": bbox,
        "authorization_state": auth,
        "family": family,
        "container_id": container_id,
        "region_id": region_id,
        "route_intent": str(source.get("route_intent") or source.get("text_area_route_intent") or source.get("intent") or ""),
        "must_not_mutate": must_not_mutate or family in {"protected", "review", "outside", "ambiguous"},
        "source_stage": str(source.get("source_stage") or source.get("text_area_authorization_source_stage") or source_kind),
        "confidence_tier": str(source.get("confidence_tier") or source.get("text_area_confidence_tier") or "low"),
        "reason_codes": reason_codes,
        "conflict_flags": conflict_flags,
        "cleanup_job_ids": job_ids,
    }


def _component_auth_infer_authorization(source: Mapping[str, Any]) -> str:
    marker = " ".join(
        str(source.get(key) or "")
        for key in (
            "cleanup_authorization",
            "text_area_cleanup_authorization",
            "route_intent",
            "text_area_route_intent",
            "container_type",
            "semantic_class",
            "cleanup_mode",
            "classification_reason",
            "protection_reason",
            "text_area_protection_reason",
            "fallback_reason",
        )
    ).lower()
    if any(token in marker for token in ("non_text", "non-text", "art_only", "non_translation_art")):
        return AUTH_PROTECT_ART_OR_NON_TEXT
    if any(token in marker for token in ("sfx", "decorative", "preserve_sfx_decorative")):
        return AUTH_PROTECT_SFX_DECORATIVE
    if "translate_speech" in marker or "speech_bubble" in marker:
        return AUTH_CLEANUP_TRANSLATE_SPEECH
    if "caption" in marker and "background" not in marker:
        return AUTH_CLEANUP_TRANSLATE_CAPTION
    if "translate_caption" in marker or "background" in marker:
        return AUTH_CLEANUP_TRANSLATE_BACKGROUND
    if "outside" in marker:
        return AUTH_OUTSIDE_CLEANUP_SCOPE
    return AUTH_REVIEW_UNKNOWN_NOT_CLEANUP


def _component_auth_family(auth: str) -> str:
    if auth in {AUTH_CLEANUP_TRANSLATE_SPEECH, AUTH_CLEANUP_TRANSLATE_BACKGROUND, AUTH_CLEANUP_TRANSLATE_CAPTION}:
        return "cleanup"
    if auth in {AUTH_PROTECT_SFX_DECORATIVE, AUTH_PROTECT_ART_OR_NON_TEXT}:
        return "protected"
    if auth == AUTH_OUTSIDE_CLEANUP_SCOPE:
        return "outside"
    if auth == AUTH_AMBIGUOUS_COMPONENT_OWNER:
        return "ambiguous"
    return "review"


def _component_auth_candidates(
    component_mask: Any,
    pixel_count: int,
    centroid: Sequence[float],
    scopes: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    cx = float(centroid[0])
    cy = float(centroid[1])
    for scope in scopes or []:
        bbox = scope.get("bbox")
        if not isinstance(bbox, Sequence) or len(bbox) < 4:
            continue
        x0, y0, x1, y1 = [int(item) for item in bbox[:4]]
        if x1 <= x0 or y1 <= y0:
            continue
        overlap = int(np.count_nonzero(component_mask[y0:y1, x0:x1]))
        centroid_inside = x0 <= cx < x1 and y0 <= cy < y1
        if overlap <= 0 and not centroid_inside:
            continue
        ratio = overlap / float(max(1, pixel_count))
        eligible = ratio >= 0.05 or overlap >= 3 or centroid_inside
        candidates.append(
            {
                **dict(scope),
                "overlap_pixels": overlap,
                "overlap_ratio": ratio,
                "centroid_inside": centroid_inside,
                "eligible": eligible,
            }
        )
    candidates.sort(key=lambda item: (int(item.get("overlap_pixels") or 0), bool(item.get("centroid_inside"))), reverse=True)
    return candidates


def _component_auth_record(
    *,
    page_id: str,
    component_index: int,
    label: int,
    bbox: List[int],
    pixel_count: int,
    centroid: Sequence[float],
    candidates: Sequence[Mapping[str, Any]],
) -> TextAreaComponentAuthorizationRecord:
    eligible = [item for item in candidates if item.get("eligible")]
    cleanup_candidates = [item for item in eligible if item.get("family") == "cleanup"]
    protected_candidates = [item for item in eligible if item.get("family") == "protected"]
    review_candidates = [item for item in eligible if item.get("family") == "review"]
    outside_candidates = [item for item in eligible if item.get("family") == "outside"]
    best_cleanup = cleanup_candidates[0] if cleanup_candidates else None
    best_protected = protected_candidates[0] if protected_candidates else None
    ambiguity_reasons: List[str] = []

    selected: Mapping[str, Any] | None = None
    semantic_authorization_state = AUTH_REVIEW_UNKNOWN_NOT_CLEANUP
    if best_cleanup and best_protected:
        cleanup_ratio = float(best_cleanup.get("overlap_ratio") or 0.0)
        protected_ratio = float(best_protected.get("overlap_ratio") or 0.0)
        if protected_ratio >= 0.75 and cleanup_ratio < 0.25:
            selected = best_protected
            semantic_authorization_state = str(best_protected.get("authorization_state") or AUTH_PROTECT_SFX_DECORATIVE)
        elif cleanup_ratio >= 0.75 and protected_ratio < 0.25:
            selected = best_cleanup
            semantic_authorization_state = str(best_cleanup.get("authorization_state") or AUTH_CLEANUP_TRANSLATE_SPEECH)
        else:
            selected = best_cleanup
            semantic_authorization_state = AUTH_AMBIGUOUS_COMPONENT_OWNER
            ambiguity_reasons = ["conflicting_authorization_evidence"]
    elif best_cleanup:
        selected = best_cleanup
        semantic_authorization_state = str(best_cleanup.get("authorization_state") or AUTH_CLEANUP_TRANSLATE_SPEECH)
    elif best_protected:
        selected = best_protected
        semantic_authorization_state = str(best_protected.get("authorization_state") or AUTH_PROTECT_SFX_DECORATIVE)
    elif review_candidates:
        selected = review_candidates[0]
        semantic_authorization_state = AUTH_REVIEW_UNKNOWN_NOT_CLEANUP
    elif outside_candidates:
        selected = outside_candidates[0]
        semantic_authorization_state = AUTH_OUTSIDE_CLEANUP_SCOPE
    else:
        selected = None
        semantic_authorization_state = AUTH_REVIEW_UNKNOWN_NOT_CLEANUP

    family = _component_auth_family(semantic_authorization_state)
    reason_codes = _component_auth_merged_values(eligible, "reason_codes")
    conflict_flags = _component_auth_merged_values(eligible, "conflict_flags")
    if not eligible and "no_upstream_text_area_authorization" not in reason_codes:
        reason_codes.append("no_upstream_text_area_authorization")
    if semantic_authorization_state == AUTH_AMBIGUOUS_COMPONENT_OWNER and "conflicting_authorization_evidence" not in reason_codes:
        if not ambiguity_reasons:
            reason_codes.append("conflicting_authorization_evidence")
    for reason in ambiguity_reasons:
        if reason and reason not in reason_codes:
            reason_codes.append(reason)
    component_id = f"tauthcomp_{_component_auth_safe_id(page_id)}_{component_index:04d}"
    owning_candidates = cleanup_candidates if family == "cleanup" else []
    protection_candidates = protected_candidates if family == "protected" else []
    if semantic_authorization_state == AUTH_AMBIGUOUS_COMPONENT_OWNER:
        owning_candidates = cleanup_candidates
        protection_candidates = protected_candidates
    binding_state, binding_failure_reason, owner_cleanup_job_id, scope_cleanup_job_ids = _component_auth_job_binding(
        semantic_authorization_state,
        cleanup_candidates,
    )
    if binding_failure_reason and binding_failure_reason not in reason_codes:
        reason_codes.append(binding_failure_reason)
    if binding_state in {"missing_cleanup_job", "non_unique_cleanup_job"} and binding_state not in reason_codes:
        reason_codes.append(binding_state)
    final_mask_authorization_state = semantic_authorization_state
    return TextAreaComponentAuthorizationRecord(
        page_id=page_id,
        component_id=component_id,
        component_bbox=[int(item) for item in bbox],
        component_pixel_count=int(pixel_count),
        authorization_state=final_mask_authorization_state,
        cleanup_authorization=final_mask_authorization_state if family in {"cleanup", "protected", "ambiguous"} else "",
        route_intent=str(selected.get("route_intent") or "") if selected else "",
        must_not_mutate=family != "cleanup",
        owning_container_ids=_component_auth_unique_ids(owning_candidates, "container_id"),
        protection_container_ids=_component_auth_unique_ids(protection_candidates, "container_id"),
        source_stage=str(selected.get("source_stage") or "text_area_plan_component_authorization") if selected else "text_area_plan_component_authorization",
        confidence_tier=str(selected.get("confidence_tier") or "low") if selected else "low",
        reason_codes=reason_codes,
        conflict_flags=conflict_flags,
        review_required=family in {"review", "outside", "ambiguous"},
        visual_debug_color=COMPONENT_AUTHORIZATION_COLORS.get(final_mask_authorization_state, "gray"),
        label=int(label),
        owner_cleanup_job_id=owner_cleanup_job_id,
        scope_cleanup_job_ids=scope_cleanup_job_ids,
        candidate_cleanup_job_ids=_component_auth_unique_jobs(eligible),
        owning_region_ids=_component_auth_unique_ids(owning_candidates, "region_id"),
        protection_region_ids=_component_auth_unique_ids(protection_candidates, "region_id"),
        overlap_pixels=int(selected.get("overlap_pixels") or 0) if selected else 0,
        overlap_ratio=round(float(selected.get("overlap_ratio") or 0.0), 4) if selected else 0.0,
        protected_overlap_pixels=int(best_protected.get("overlap_pixels") or 0) if best_protected else 0,
        protected_overlap_ratio=round(float(best_protected.get("overlap_ratio") or 0.0), 4) if best_protected else 0.0,
        centroid=[round(float(centroid[0]), 3), round(float(centroid[1]), 3)],
        candidate_container_ids=_component_auth_unique_ids(eligible, "container_id"),
        candidate_region_ids=_component_auth_unique_ids(eligible, "region_id"),
        ambiguity_reasons=ambiguity_reasons,
        semantic_authorization_state=semantic_authorization_state,
        semantic_visual_color=COMPONENT_AUTHORIZATION_COLORS.get(semantic_authorization_state, "gray"),
        job_binding_state=binding_state,
        job_binding_failure_reason=binding_failure_reason,
        final_mask_authorization_state=final_mask_authorization_state,
    )


def _component_auth_assign_groups(page_id: str, records: Sequence[TextAreaComponentAuthorizationRecord]) -> None:
    groups: Dict[tuple[str, str], List[TextAreaComponentAuthorizationRecord]] = {}
    for record in records:
        if record.owning_container_ids:
            key = ("cleanup", record.owning_container_ids[0])
        elif record.protection_container_ids:
            key = ("protected", record.protection_container_ids[0])
        elif record.candidate_container_ids:
            key = ("candidate", record.candidate_container_ids[0])
        else:
            key = ("component", record.component_id)
        groups.setdefault(key, []).append(record)
    for index, (_key, group) in enumerate(groups.items()):
        states = {record.authorization_state for record in group}
        cleanup_states = sorted(state for state in states if _component_auth_family(state) == "cleanup")
        protected_states = sorted(state for state in states if _component_auth_family(state) == "protected")
        if AUTH_AMBIGUOUS_COMPONENT_OWNER in states or (cleanup_states and protected_states):
            group_state = AUTH_AMBIGUOUS_COMPONENT_OWNER
        elif cleanup_states:
            group_state = cleanup_states[0]
        elif protected_states:
            group_state = protected_states[0]
        elif AUTH_OUTSIDE_CLEANUP_SCOPE in states:
            group_state = AUTH_OUTSIDE_CLEANUP_SCOPE
        else:
            group_state = AUTH_REVIEW_UNKNOWN_NOT_CLEANUP
        group_id = f"tauthgrp_{_component_auth_safe_id(page_id)}_{index:04d}"
        for record in group:
            record.group_id = group_id
            record.group_authorization_state = group_state
            if record.authorization_state in {AUTH_REVIEW_UNKNOWN_NOT_CLEANUP, AUTH_OUTSIDE_CLEANUP_SCOPE} and group_state not in {
                AUTH_REVIEW_UNKNOWN_NOT_CLEANUP,
                AUTH_OUTSIDE_CLEANUP_SCOPE,
                AUTH_AMBIGUOUS_COMPONENT_OWNER,
            }:
                record.authorization_state = group_state
                record.cleanup_authorization = group_state
                record.must_not_mutate = _component_auth_family(group_state) != "cleanup"
                record.review_required = False
                record.visual_debug_color = COMPONENT_AUTHORIZATION_COLORS.get(group_state, record.visual_debug_color)
                if "group_authorization_promoted_review_component" not in record.reason_codes:
                    record.reason_codes.append("group_authorization_promoted_review_component")


def _component_auth_apply_protected_sfx_grouping(records: Sequence[TextAreaComponentAuthorizationRecord]) -> None:
    seeds = [
        record
        for record in records
        if record.authorization_state == AUTH_PROTECT_SFX_DECORATIVE and record.component_pixel_count >= 20
    ]
    if not seeds:
        return
    for record in records:
        if record.authorization_state == AUTH_PROTECT_SFX_DECORATIVE:
            continue
        matching_seeds = [seed for seed in seeds if _component_auth_same_sfx_band(record, seed)]
        if not matching_seeds:
            continue
        matching_seed = next(
            (seed for seed in matching_seeds if _component_auth_is_local_sfx_continuation(record, seed)),
            matching_seeds[0],
        )
        is_local_sfx_continuation = _component_auth_is_local_sfx_continuation(record, matching_seed)
        if _component_auth_is_speech_record(record) and not is_local_sfx_continuation:
            continue
        if record.authorization_state in {
            AUTH_CLEANUP_TRANSLATE_SPEECH,
            AUTH_CLEANUP_TRANSLATE_BACKGROUND,
            AUTH_CLEANUP_TRANSLATE_CAPTION,
        }:
            if not (
                _component_auth_has_decorative_evidence(record)
                or is_local_sfx_continuation
            ):
                continue
            _component_auth_set_state(
                record,
                AUTH_AMBIGUOUS_COMPONENT_OWNER,
                reason="sfx_group_conflicts_with_cleanup_authorization",
                ambiguity=True,
            )
        else:
            _component_auth_set_state(
                record,
                AUTH_PROTECT_SFX_DECORATIVE,
                reason="sfx_group_propagated_from_protected_neighbor",
            )
        for container_id in matching_seed.protection_container_ids:
            if container_id and container_id not in record.protection_container_ids:
                record.protection_container_ids.append(container_id)


def _component_auth_apply_vertical_review_text_groups(records: Sequence[TextAreaComponentAuthorizationRecord]) -> None:
    candidates = [
        record
        for record in records
        if _component_auth_is_vertical_review_text_candidate(record)
    ]
    visited: set[str] = set()
    for record in candidates:
        if record.component_id in visited:
            continue
        group: List[TextAreaComponentAuthorizationRecord] = []
        stack = [record]
        visited.add(record.component_id)
        while stack:
            current = stack.pop()
            group.append(current)
            for other in candidates:
                if other.component_id in visited:
                    continue
                if _component_auth_bboxes_adjacent(current.component_bbox, other.component_bbox):
                    visited.add(other.component_id)
                    stack.append(other)
        if not _component_auth_is_vertical_review_text_group(group):
            continue
        for item in group:
            _component_auth_set_state(
                item,
                AUTH_CLEANUP_TRANSLATE_BACKGROUND,
                reason="vertical_review_text_group_promoted_to_cleanup_background",
            )
            item.route_intent = "translate_caption_background"
            item.protection_container_ids = []
            item.reason_codes = [
                reason
                for reason in item.reason_codes
                if not any(token in str(reason).lower() for token in ("sfx", "decorative", "preserve"))
            ]
            item.conflict_flags = [
                flag
                for flag in item.conflict_flags
                if not any(token in str(flag).lower() for token in ("sfx", "decorative", "preserve"))
            ]
            if "vertical_review_text_group_promoted_to_cleanup_background" not in item.reason_codes:
                item.reason_codes.append("vertical_review_text_group_promoted_to_cleanup_background")
            item.job_binding_state = "missing_cleanup_job"
            item.job_binding_failure_reason = "cleanup_job_binding_contract_error"
            item.owner_cleanup_job_id = ""
            item.scope_cleanup_job_ids = []
            if item.job_binding_failure_reason not in item.reason_codes:
                item.reason_codes.append(item.job_binding_failure_reason)
            if item.job_binding_state not in item.reason_codes:
                item.reason_codes.append(item.job_binding_state)


def _component_auth_is_vertical_review_text_candidate(record: TextAreaComponentAuthorizationRecord) -> bool:
    if record.authorization_state in {
        AUTH_CLEANUP_TRANSLATE_SPEECH,
        AUTH_CLEANUP_TRANSLATE_BACKGROUND,
        AUTH_CLEANUP_TRANSLATE_CAPTION,
        AUTH_AMBIGUOUS_COMPONENT_OWNER,
        AUTH_OUTSIDE_CLEANUP_SCOPE,
    }:
        return False
    if _component_auth_is_speech_record(record):
        return False
    if len(record.component_bbox) < 4:
        return False
    width = max(1, int(record.component_bbox[2]) - int(record.component_bbox[0]))
    height = max(1, int(record.component_bbox[3]) - int(record.component_bbox[1]))
    if width > 70 or height > 90 or record.component_pixel_count > 1200:
        return False
    return True


def _component_auth_is_vertical_review_text_group(group: Sequence[TextAreaComponentAuthorizationRecord]) -> bool:
    if len(group) < 6:
        return False
    x0 = min(int(item.component_bbox[0]) for item in group)
    y0 = min(int(item.component_bbox[1]) for item in group)
    x1 = max(int(item.component_bbox[2]) for item in group)
    y1 = max(int(item.component_bbox[3]) for item in group)
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    if height < 140 or width < 35 or width > 220:
        return False
    if height < width * 1.25:
        return False
    seed_count = sum(1 for item in group if _component_auth_has_review_text_seed(item))
    if seed_count <= 0:
        return False
    large_items = [
        item
        for item in group
        if (
            max(1, int(item.component_bbox[2]) - int(item.component_bbox[0])) > 70
            or max(1, int(item.component_bbox[3]) - int(item.component_bbox[1])) > 90
            or item.component_pixel_count > 1200
        )
    ]
    return not large_items


def _component_auth_has_review_text_seed(record: TextAreaComponentAuthorizationRecord) -> bool:
    marker = " ".join(
        [
            str(record.route_intent or ""),
            str(record.confidence_tier or ""),
            " ".join(record.reason_codes or []),
            " ".join(record.candidate_container_ids or []),
        ]
    ).lower()
    return bool(
        record.candidate_container_ids
        and any(token in marker for token in ("review_or_fallback", "unknown_fallback", "bubble_without_kitsumed", "dark_or_art_context"))
    )


def _component_auth_apply_orphan_text_near_cleanup_groups(records: Sequence[TextAreaComponentAuthorizationRecord]) -> None:
    candidates = [
        record
        for record in records
        if _component_auth_is_orphan_text_candidate(record)
    ]
    visited: set[str] = set()
    for record in candidates:
        if record.component_id in visited:
            continue
        group: List[TextAreaComponentAuthorizationRecord] = []
        stack = [record]
        visited.add(record.component_id)
        while stack:
            current = stack.pop()
            group.append(current)
            for other in candidates:
                if other.component_id in visited:
                    continue
                if _component_auth_bboxes_adjacent(current.component_bbox, other.component_bbox):
                    visited.add(other.component_id)
                    stack.append(other)
        anchor = _component_auth_cleanup_anchor_for_orphan_group(group, records)
        if anchor is None:
            continue
        state = AUTH_CLEANUP_TRANSLATE_SPEECH if str(anchor.route_intent or "") == "translate_speech" else AUTH_CLEANUP_TRANSLATE_BACKGROUND
        route = "translate_speech" if state == AUTH_CLEANUP_TRANSLATE_SPEECH else "translate_caption_background"
        for item in group:
            _component_auth_set_state(
                item,
                state,
                reason="small_orphan_text_group_promoted_near_cleanup_text",
            )
            item.route_intent = route
            item.job_binding_state = "missing_cleanup_job"
            item.job_binding_failure_reason = "cleanup_job_binding_contract_error"
            item.owner_cleanup_job_id = ""
            item.scope_cleanup_job_ids = []
            if item.job_binding_failure_reason not in item.reason_codes:
                item.reason_codes.append(item.job_binding_failure_reason)
            if item.job_binding_state not in item.reason_codes:
                item.reason_codes.append(item.job_binding_state)


def _component_auth_is_orphan_text_candidate(record: TextAreaComponentAuthorizationRecord) -> bool:
    if record.authorization_state != AUTH_REVIEW_UNKNOWN_NOT_CLEANUP:
        return False
    if record.candidate_container_ids or record.candidate_cleanup_job_ids or record.owning_container_ids or record.protection_container_ids:
        return False
    if _component_auth_has_decorative_evidence(record) or _component_auth_is_speech_record(record):
        return False
    if len(record.component_bbox) < 4:
        return False
    width = max(1, int(record.component_bbox[2]) - int(record.component_bbox[0]))
    height = max(1, int(record.component_bbox[3]) - int(record.component_bbox[1]))
    return bool(15 <= record.component_pixel_count <= 600 and width <= 50 and height <= 55)


def _component_auth_cleanup_anchor_for_orphan_group(
    group: Sequence[TextAreaComponentAuthorizationRecord],
    records: Sequence[TextAreaComponentAuthorizationRecord],
) -> TextAreaComponentAuthorizationRecord | None:
    if len(group) < 2:
        return None
    x0 = min(int(item.component_bbox[0]) for item in group)
    y0 = min(int(item.component_bbox[1]) for item in group)
    x1 = max(int(item.component_bbox[2]) for item in group)
    y1 = max(int(item.component_bbox[3]) for item in group)
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    pixels = sum(int(item.component_pixel_count) for item in group)
    if pixels < 120 or width > 100 or height > 100:
        return None
    group_box = [x0, y0, x1, y1]
    for protected in records:
        if protected.authorization_state not in {AUTH_PROTECT_SFX_DECORATIVE, AUTH_PROTECT_ART_OR_NON_TEXT, AUTH_AMBIGUOUS_COMPONENT_OWNER}:
            continue
        if protected.component_pixel_count >= 200 and _component_auth_bbox_gap(group_box, protected.component_bbox) <= 120:
            return None
    anchors = [
        record
        for record in records
        if _component_auth_family(record.authorization_state) == "cleanup"
        and not _component_auth_has_decorative_evidence(record)
        and _component_auth_bbox_gap(group_box, record.component_bbox) <= 180
    ]
    if not anchors:
        return None
    anchors.sort(key=lambda item: _component_auth_bbox_gap(group_box, item.component_bbox))
    return anchors[0]


def _component_auth_bbox_gap(a: Sequence[int], b: Sequence[int]) -> int:
    if len(a) < 4 or len(b) < 4:
        return 10**9
    ax0, ay0, ax1, ay1 = [int(item) for item in a[:4]]
    bx0, by0, bx1, by1 = [int(item) for item in b[:4]]
    x_gap = max(0, max(ax0, bx0) - min(ax1, bx1))
    y_gap = max(0, max(ay0, by0) - min(ay1, by1))
    return int(max(x_gap, y_gap))


def _component_auth_same_sfx_band(
    record: TextAreaComponentAuthorizationRecord,
    seed: TextAreaComponentAuthorizationRecord,
) -> bool:
    box = record.component_bbox
    seed_box = seed.component_bbox
    if len(box) < 4 or len(seed_box) < 4:
        return False
    width = max(1, int(box[2]) - int(box[0]))
    height = max(1, int(box[3]) - int(box[1]))
    seed_width = max(1, int(seed_box[2]) - int(seed_box[0]))
    seed_height = max(1, int(seed_box[3]) - int(seed_box[1]))
    if record.component_pixel_count < 20 and width < 8 and height < 8:
        return False
    y_overlap = max(0, min(int(box[3]), int(seed_box[3])) - max(int(box[1]), int(seed_box[1])))
    y_ratio = y_overlap / float(max(1, min(height, seed_height)))
    x_overlap = max(0, min(int(box[2]), int(seed_box[2])) - max(int(box[0]), int(seed_box[0])))
    x_ratio = x_overlap / float(max(1, min(width, seed_width)))
    if int(box[2]) < int(seed_box[0]):
        gap = int(seed_box[0]) - int(box[2])
    elif int(seed_box[2]) < int(box[0]):
        gap = int(box[0]) - int(seed_box[2])
    else:
        gap = 0
    if int(box[3]) < int(seed_box[1]):
        vertical_gap = int(seed_box[1]) - int(box[3])
    elif int(seed_box[3]) < int(box[1]):
        vertical_gap = int(box[1]) - int(seed_box[3])
    else:
        vertical_gap = 0
    if y_ratio < 0.28 and not (x_ratio >= 0.35 and vertical_gap <= max(120, int(max(height, seed_height) * 1.2))):
        return False
    if gap > max(120, int(max(width, seed_width) * 2.0)):
        return False
    center_y = (int(box[1]) + int(box[3])) / 2.0
    seed_center_y = (int(seed_box[1]) + int(seed_box[3])) / 2.0
    if abs(center_y - seed_center_y) > max(160.0, float(max(height, seed_height)) * 1.4):
        return False
    return bool(record.component_pixel_count >= 20 or seed_width >= 40)


def _component_auth_is_speech_record(record: TextAreaComponentAuthorizationRecord) -> bool:
    marker = " ".join(
        [
            str(record.route_intent or ""),
            " ".join(record.reason_codes or []),
            " ".join(record.owning_container_ids or []),
            " ".join(record.candidate_container_ids or []),
        ]
    ).lower()
    return any(token in marker for token in ("translate_speech", "speech_bubble", "speech_mask_container"))


def _component_auth_has_decorative_evidence(record: TextAreaComponentAuthorizationRecord) -> bool:
    marker = " ".join(
        [
            str(record.route_intent or ""),
            " ".join(record.reason_codes or []),
            " ".join(record.conflict_flags or []),
            " ".join(record.protection_container_ids or []),
            " ".join(record.candidate_container_ids or []),
        ]
    ).lower()
    return any(token in marker for token in ("sfx", "decorative", "preserve", "art_text", "art_sfx", "non_text", "non-text"))


def _component_auth_is_local_sfx_continuation(
    record: TextAreaComponentAuthorizationRecord,
    seed: TextAreaComponentAuthorizationRecord,
) -> bool:
    box = record.component_bbox
    seed_box = seed.component_bbox
    if len(box) < 4 or len(seed_box) < 4:
        return False
    width = max(1, int(box[2]) - int(box[0]))
    height = max(1, int(box[3]) - int(box[1]))
    seed_width = max(1, int(seed_box[2]) - int(seed_box[0]))
    seed_height = max(1, int(seed_box[3]) - int(seed_box[1]))
    if int(box[2]) < int(seed_box[0]):
        gap = int(seed_box[0]) - int(box[2])
    elif int(seed_box[2]) < int(box[0]):
        gap = int(box[0]) - int(seed_box[2])
    else:
        gap = 0
    y_overlap = max(0, min(int(box[3]), int(seed_box[3])) - max(int(box[1]), int(seed_box[1])))
    y_ratio = y_overlap / float(max(1, min(height, seed_height)))
    x_overlap = max(0, min(int(box[2]), int(seed_box[2])) - max(int(box[0]), int(seed_box[0])))
    x_ratio = x_overlap / float(max(1, min(width, seed_width)))
    if int(box[3]) < int(seed_box[1]):
        vertical_gap = int(seed_box[1]) - int(box[3])
    elif int(seed_box[3]) < int(box[1]):
        vertical_gap = int(box[1]) - int(seed_box[3])
    else:
        vertical_gap = 0
    if y_ratio < 0.35 and not (x_ratio >= 0.35 and vertical_gap <= 120):
        return False
    horizontal_stroke = gap <= 80 and width >= 45 and width >= height * 2.0
    compact_neighbor = gap <= 80 and width >= 18 and seed_width >= 60
    large_sfx_cluster_neighbor = (
        gap <= 220
        and width >= 80
        and height >= 80
        and seed_width >= 80
        and seed_height >= 80
    )
    vertical_sfx_stack = (
        vertical_gap <= 120
        and x_ratio >= 0.35
        and width >= 45
        and height >= 45
        and seed_width >= 45
        and seed_height >= 45
    )
    return bool(horizontal_stroke or compact_neighbor or large_sfx_cluster_neighbor or vertical_sfx_stack)


def _component_auth_apply_review_only_caption_guard(records: Sequence[TextAreaComponentAuthorizationRecord]) -> None:
    guarded: List[TextAreaComponentAuthorizationRecord] = []
    for record in records:
        if not _component_auth_is_large_review_only_caption_seed(record):
            continue
        if record.authorization_state != AUTH_AMBIGUOUS_COMPONENT_OWNER:
            _component_auth_set_state(
                record,
                AUTH_AMBIGUOUS_COMPONENT_OWNER,
                reason="large_review_only_caption_component_conflicts_with_cleanup_authority",
                ambiguity=True,
            )
        guarded.append(record)
    for group in _component_auth_review_only_caption_groups(records):
        if not _component_auth_is_large_review_only_caption_group(group):
            continue
        for record in group:
            if record.authorization_state != AUTH_AMBIGUOUS_COMPONENT_OWNER:
                _component_auth_set_state(
                    record,
                    AUTH_AMBIGUOUS_COMPONENT_OWNER,
                    reason="large_review_only_caption_group_conflicts_with_cleanup_authority",
                    ambiguity=True,
                )
            if record not in guarded:
                guarded.append(record)
    if not guarded:
        return

    for record in records:
        if record in guarded:
            continue
        if record.authorization_state not in {
            AUTH_CLEANUP_TRANSLATE_SPEECH,
            AUTH_CLEANUP_TRANSLATE_BACKGROUND,
            AUTH_CLEANUP_TRANSLATE_CAPTION,
        }:
            continue
        if _component_auth_is_speech_record(record):
            continue
        if str(record.route_intent or "") != "translate_caption_background":
            continue
        if not any(_component_auth_shares_local_caption_guard(record, seed) for seed in guarded):
            continue
        _component_auth_set_state(
            record,
            AUTH_AMBIGUOUS_COMPONENT_OWNER,
            reason="large_review_only_caption_group_conflicts_with_cleanup_authority",
            ambiguity=True,
        )


def _component_auth_is_large_review_only_caption_component(record: TextAreaComponentAuthorizationRecord) -> bool:
    if record.authorization_state not in {
        AUTH_CLEANUP_TRANSLATE_BACKGROUND,
        AUTH_CLEANUP_TRANSLATE_CAPTION,
    }:
        return False
    return _component_auth_is_large_review_only_caption_seed(record)


def _component_auth_is_large_review_only_caption_seed(record: TextAreaComponentAuthorizationRecord) -> bool:
    if record.authorization_state not in {
        AUTH_CLEANUP_TRANSLATE_BACKGROUND,
        AUTH_CLEANUP_TRANSLATE_CAPTION,
        AUTH_AMBIGUOUS_COMPONENT_OWNER,
    }:
        return False
    if _component_auth_is_speech_record(record):
        return False
    if str(record.route_intent or "") != "translate_caption_background":
        return False
    marker = " ".join(
        [
            str(record.confidence_tier or ""),
            " ".join(record.reason_codes or []),
            " ".join(record.conflict_flags or []),
        ]
    ).lower()
    if not any(
        token in marker
        for token in (
            "text_free_review_only",
            "ogkalu_text_free_without_kitsumed_mask",
            "deterministic_side_caption_requires_review",
            "deterministic_top_band_far_right_caption_search",
            "sfx_group_conflicts_with_cleanup_authorization",
        )
    ):
        return False
    if len(record.component_bbox) < 4:
        return False
    width = max(1, int(record.component_bbox[2]) - int(record.component_bbox[0]))
    height = max(1, int(record.component_bbox[3]) - int(record.component_bbox[1]))
    if record.component_pixel_count < 1000:
        return False
    return bool(width >= 120 or (width >= 90 and height >= 90))


def _component_auth_review_only_caption_groups(
    records: Sequence[TextAreaComponentAuthorizationRecord],
) -> List[List[TextAreaComponentAuthorizationRecord]]:
    groups: Dict[str, List[TextAreaComponentAuthorizationRecord]] = {}
    for record in records:
        if not _component_auth_is_review_only_caption_group_candidate(record):
            continue
        ids = list(record.owning_container_ids or []) + list(record.candidate_container_ids or [])
        key = ids[0] if ids else record.component_id
        groups.setdefault(key, []).append(record)
    return [group for group in groups.values() if len(group) >= 2]


def _component_auth_is_review_only_caption_group_candidate(record: TextAreaComponentAuthorizationRecord) -> bool:
    if record.authorization_state not in {
        AUTH_CLEANUP_TRANSLATE_BACKGROUND,
        AUTH_CLEANUP_TRANSLATE_CAPTION,
        AUTH_AMBIGUOUS_COMPONENT_OWNER,
    }:
        return False
    if _component_auth_is_speech_record(record):
        return False
    if str(record.route_intent or "") != "translate_caption_background":
        return False
    marker = " ".join(
        [
            str(record.confidence_tier or ""),
            " ".join(record.reason_codes or []),
            " ".join(record.owning_container_ids or []),
            " ".join(record.candidate_container_ids or []),
        ]
    ).lower()
    has_text_free_marker = any(token in marker for token in ("text_free_review_only", "ogkalu_text_free_without_kitsumed_mask"))
    if (
        "deterministic_top_band_far_right_caption_search" in marker
        and not has_text_free_marker
        and len(record.component_bbox) >= 4
        and int(record.component_bbox[1]) > 280
    ):
        return False
    return any(
        token in marker
        for token in (
            "text_free_review_only",
            "ogkalu_text_free_without_kitsumed_mask",
            "deterministic_top_band_far_right_caption_search",
            "deterministic_top_band_day_caption_search",
            "sfx_group_conflicts_with_cleanup_authorization",
        )
    )


def _component_auth_is_large_review_only_caption_group(group: Sequence[TextAreaComponentAuthorizationRecord]) -> bool:
    if len(group) < 2:
        return False
    x0 = min(int(item.component_bbox[0]) for item in group)
    y0 = min(int(item.component_bbox[1]) for item in group)
    x1 = max(int(item.component_bbox[2]) for item in group)
    y1 = max(int(item.component_bbox[3]) for item in group)
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    pixels = sum(int(item.component_pixel_count) for item in group)
    if pixels < 2500 or height < 80 or width < 45:
        return False
    if any(item.authorization_state == AUTH_AMBIGUOUS_COMPONENT_OWNER for item in group):
        return True
    display_like_components = 0
    for item in group:
        item_width = max(1, int(item.component_bbox[2]) - int(item.component_bbox[0]))
        item_height = max(1, int(item.component_bbox[3]) - int(item.component_bbox[1]))
        if item.component_pixel_count >= 1200 or (item_width >= 45 and item_height >= 80):
            display_like_components += 1
    if display_like_components < 2:
        return False
    return bool((len(group) >= 3 and (width >= 75 or height >= 160)) or (len(group) >= 2 and pixels >= 5000 and height >= 160))


def _component_auth_shares_local_caption_guard(
    record: TextAreaComponentAuthorizationRecord,
    seed: TextAreaComponentAuthorizationRecord,
) -> bool:
    shared_containers = set(record.owning_container_ids or []) & set(seed.owning_container_ids or [])
    shared_candidates = set(record.candidate_container_ids or []) & set(seed.candidate_container_ids or [])
    if not shared_containers and not shared_candidates:
        return False
    if len(record.component_bbox) < 4 or len(seed.component_bbox) < 4:
        return False
    box = [int(item) for item in record.component_bbox[:4]]
    seed_box = [int(item) for item in seed.component_bbox[:4]]
    margin = 35
    return not (
        box[2] < seed_box[0] - margin
        or box[0] > seed_box[2] + margin
        or box[3] < seed_box[1] - margin
        or box[1] > seed_box[3] + margin
    )


def _component_auth_set_state(
    record: TextAreaComponentAuthorizationRecord,
    state: str,
    *,
    reason: str = "",
    ambiguity: bool = False,
) -> None:
    family = _component_auth_family(state)
    record.authorization_state = state
    record.semantic_authorization_state = state
    record.final_mask_authorization_state = state
    record.cleanup_authorization = state if family in {"cleanup", "protected", "ambiguous"} else ""
    record.must_not_mutate = family != "cleanup"
    record.review_required = family in {"review", "outside", "ambiguous"}
    color = COMPONENT_AUTHORIZATION_COLORS.get(state, "gray")
    record.visual_debug_color = color
    record.semantic_visual_color = color
    if family != "cleanup":
        record.job_binding_state = "not_applicable_semantic_conflict" if state == AUTH_AMBIGUOUS_COMPONENT_OWNER else "not_applicable_non_cleanup"
        record.job_binding_failure_reason = ""
        record.reason_codes = [
            item
            for item in record.reason_codes
            if item not in {"cleanup_job_binding_contract_error", "missing_cleanup_job", "non_unique_cleanup_job"}
        ]
    if reason and reason not in record.reason_codes:
        record.reason_codes.append(reason)
    if ambiguity and reason and reason not in record.ambiguity_reasons:
        record.ambiguity_reasons.append(reason)


def _component_auth_apply_large_decorative_review_rule(records: Sequence[TextAreaComponentAuthorizationRecord]) -> None:
    for record in records:
        if record.authorization_state in {AUTH_PROTECT_SFX_DECORATIVE, AUTH_PROTECT_ART_OR_NON_TEXT, AUTH_AMBIGUOUS_COMPONENT_OWNER}:
            continue
        if len(record.component_bbox) < 4:
            continue
        width = max(1, int(record.component_bbox[2]) - int(record.component_bbox[0]))
        height = max(1, int(record.component_bbox[3]) - int(record.component_bbox[1]))
        if record.component_pixel_count < 6000 or width < 80 or height < 80:
            continue
        if record.authorization_state in {
            AUTH_CLEANUP_TRANSLATE_SPEECH,
            AUTH_CLEANUP_TRANSLATE_BACKGROUND,
            AUTH_CLEANUP_TRANSLATE_CAPTION,
        }:
            if _component_auth_has_decorative_evidence(record):
                _component_auth_set_state(
                    record,
                    AUTH_AMBIGUOUS_COMPONENT_OWNER,
                    reason="large_decorative_component_conflicts_with_cleanup_authority",
                    ambiguity=True,
                )
            continue
        if record.authorization_state == AUTH_REVIEW_UNKNOWN_NOT_CLEANUP and _component_auth_has_decorative_evidence(record):
            _component_auth_set_state(
                record,
                AUTH_PROTECT_SFX_DECORATIVE,
                reason="large_decorative_component_without_translation_authority",
            )


def _component_auth_apply_large_decorative_review_groups(records: Sequence[TextAreaComponentAuthorizationRecord]) -> None:
    candidates = [
        record
        for record in records
        if record.authorization_state == AUTH_REVIEW_UNKNOWN_NOT_CLEANUP
        and not record.owning_container_ids
        and not record.candidate_cleanup_job_ids
        and record.component_pixel_count >= 400
        and not _component_auth_is_speech_record(record)
    ]
    visited: set[str] = set()
    for record in candidates:
        if record.component_id in visited:
            continue
        group: List[TextAreaComponentAuthorizationRecord] = []
        stack = [record]
        visited.add(record.component_id)
        while stack:
            current = stack.pop()
            group.append(current)
            for other in candidates:
                if other.component_id in visited:
                    continue
                if _component_auth_bboxes_adjacent(current.component_bbox, other.component_bbox):
                    visited.add(other.component_id)
                    stack.append(other)
        if len(group) < 2:
            continue
        x0 = min(int(item.component_bbox[0]) for item in group)
        y0 = min(int(item.component_bbox[1]) for item in group)
        x1 = max(int(item.component_bbox[2]) for item in group)
        y1 = max(int(item.component_bbox[3]) for item in group)
        width = max(1, x1 - x0)
        height = max(1, y1 - y0)
        pixels = sum(int(item.component_pixel_count) for item in group)
        if pixels < 2500 or height < 80:
            continue
        if width < 55 and height < 140:
            continue
        for item in group:
            _component_auth_set_state(
                item,
                AUTH_PROTECT_SFX_DECORATIVE,
                reason="large_decorative_component_group_without_translation_authority",
            )


def _component_auth_apply_unowned_display_neighbor_conflicts(records: Sequence[TextAreaComponentAuthorizationRecord]) -> None:
    unowned = [
        record
        for record in records
        if record.authorization_state == AUTH_REVIEW_UNKNOWN_NOT_CLEANUP
        and not record.candidate_container_ids
        and not record.candidate_cleanup_job_ids
        and not record.owning_container_ids
        and _component_auth_is_large_display_fragment(record)
    ]
    if not unowned:
        return
    for record in records:
        if record.authorization_state not in {
            AUTH_CLEANUP_TRANSLATE_SPEECH,
            AUTH_CLEANUP_TRANSLATE_BACKGROUND,
            AUTH_CLEANUP_TRANSLATE_CAPTION,
        }:
            continue
        if not _component_auth_is_large_display_fragment(record):
            continue
        neighbor = next((item for item in unowned if _component_auth_display_fragments_adjacent(record, item)), None)
        if neighbor is None:
            continue
        _component_auth_set_state(
            neighbor,
            AUTH_PROTECT_SFX_DECORATIVE,
            reason="unowned_display_fragment_protected_near_cleanup_claim",
        )
        _component_auth_set_state(
            record,
            AUTH_AMBIGUOUS_COMPONENT_OWNER,
            reason="unowned_display_neighbor_conflicts_with_cleanup_authority",
            ambiguity=True,
        )


def _component_auth_is_large_display_fragment(record: TextAreaComponentAuthorizationRecord) -> bool:
    if len(record.component_bbox) < 4:
        return False
    width = max(1, int(record.component_bbox[2]) - int(record.component_bbox[0]))
    height = max(1, int(record.component_bbox[3]) - int(record.component_bbox[1]))
    return bool(record.component_pixel_count >= 1000 and (height >= 55 or width >= 55))


def _component_auth_display_fragments_adjacent(
    record: TextAreaComponentAuthorizationRecord,
    other: TextAreaComponentAuthorizationRecord,
) -> bool:
    if len(record.component_bbox) < 4 or len(other.component_bbox) < 4:
        return False
    box = [int(item) for item in record.component_bbox[:4]]
    other_box = [int(item) for item in other.component_bbox[:4]]
    if _component_auth_bbox_gap(box, other_box) > 45:
        return False
    x_overlap = max(0, min(box[2], other_box[2]) - max(box[0], other_box[0]))
    y_overlap = max(0, min(box[3], other_box[3]) - max(box[1], other_box[1]))
    min_width = max(1, min(box[2] - box[0], other_box[2] - other_box[0]))
    min_height = max(1, min(box[3] - box[1], other_box[3] - other_box[1]))
    return bool(x_overlap / float(min_width) >= 0.25 or y_overlap / float(min_height) >= 0.25)


def _component_auth_bboxes_adjacent(a: Sequence[int], b: Sequence[int]) -> bool:
    if len(a) < 4 or len(b) < 4:
        return False
    ax0, ay0, ax1, ay1 = [int(item) for item in a[:4]]
    bx0, by0, bx1, by1 = [int(item) for item in b[:4]]
    x_gap = max(0, max(ax0, bx0) - min(ax1, bx1))
    y_gap = max(0, max(ay0, by0) - min(ay1, by1))
    x_overlap = max(0, min(ax1, bx1) - max(ax0, bx0))
    y_overlap = max(0, min(ay1, by1) - max(ay0, by0))
    if x_overlap > 0 and y_gap <= 90:
        return True
    if y_overlap > 0 and x_gap <= 90:
        return True
    return x_gap <= 80 and y_gap <= 80


def _component_auth_xywh_to_xyxy(value: Any, mask_shape: tuple[int, int]) -> List[int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 4:
        return None
    try:
        x, y, w, h = [int(round(float(item))) for item in value[:4]]
    except Exception:
        return None
    if w <= 0 or h <= 0:
        return None
    return _component_auth_clip_bbox([x, y, x + w, y + h], mask_shape)


def _component_auth_valid_or_xywh_bbox(value: Any, mask_shape: tuple[int, int]) -> List[int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 4:
        return None
    try:
        a, b, c, d = [int(round(float(item))) for item in value[:4]]
    except Exception:
        return None
    if c > a and d > b:
        return _component_auth_clip_bbox([a, b, c, d], mask_shape)
    if c > 0 and d > 0:
        return _component_auth_clip_bbox([a, b, a + c, b + d], mask_shape)
    return None


def _component_auth_clip_bbox(bbox: Sequence[int], mask_shape: tuple[int, int]) -> List[int] | None:
    height, width = int(mask_shape[0]), int(mask_shape[1])
    x0 = max(0, min(width, int(bbox[0])))
    y0 = max(0, min(height, int(bbox[1])))
    x1 = max(0, min(width, int(bbox[2])))
    y1 = max(0, min(height, int(bbox[3])))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def _component_auth_list(value: Any) -> List[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    if value:
        return [str(value)]
    return []


def _component_auth_merged_values(candidates: Sequence[Mapping[str, Any]], key: str) -> List[str]:
    output: List[str] = []
    for candidate in candidates or []:
        for value in _component_auth_list(candidate.get(key) or []):
            if value not in output:
                output.append(value)
    return output


def _component_auth_unique_ids(candidates: Sequence[Mapping[str, Any]], key: str) -> List[str]:
    output: List[str] = []
    for candidate in candidates or []:
        value = str(candidate.get(key) or "")
        if value and value not in output:
            output.append(value)
    return output


def _component_auth_unique_jobs(candidates: Sequence[Mapping[str, Any]]) -> List[str]:
    output: List[str] = []
    for candidate in candidates or []:
        for value in candidate.get("cleanup_job_ids") or []:
            text = str(value or "")
            if text and text not in output:
                output.append(text)
    return output


def _component_auth_job_binding(
    semantic_authorization_state: str,
    cleanup_candidates: Sequence[Mapping[str, Any]],
) -> tuple[str, str, str, List[str]]:
    if _component_auth_family(semantic_authorization_state) != "cleanup":
        if semantic_authorization_state == AUTH_AMBIGUOUS_COMPONENT_OWNER:
            return "not_applicable_semantic_conflict", "", "", []
        return "not_applicable_non_cleanup", "", "", []
    jobs = _component_auth_unique_jobs(cleanup_candidates)
    if not jobs:
        return "missing_cleanup_job", "cleanup_job_binding_contract_error", "", []
    if len(jobs) > 1:
        return "non_unique_cleanup_job", "cleanup_job_binding_contract_error", jobs[0], jobs
    return "bound_unique", "", jobs[0], jobs


def _component_auth_first_job_id(candidates: Sequence[Mapping[str, Any]]) -> str:
    jobs = _component_auth_unique_jobs(candidates)
    return jobs[0] if jobs else ""


def _component_auth_safe_id(value: Any) -> str:
    text = "".join(ch if ch.isalnum() else "_" for ch in str(value or "page"))
    text = "_".join(part for part in text.split("_") if part)
    return text or "page"


def build_text_area_plan(
    page_id: str,
    image_path: str | Path,
    image_size: Tuple[int, int],
    bubble_detection_result: Any,
    current_region_records: Optional[Sequence[Mapping[str, Any]]] = None,
) -> TextAreaPlan:
    """Build a front-of-pipeline TextAreaPlan from BubbleDetection evidence."""

    started = time.perf_counter()
    result = _result_to_mapping(bubble_detection_result)
    plan = TextAreaPlan(page_id=str(page_id), image_path=str(image_path), image_size=(int(image_size[0]), int(image_size[1])))
    plan.runtime.bubble_detection_generated = bool(result.get("generated"))
    plan.runtime.bubble_detection_cache_hit = result.get("cache_hit")
    plan.runtime.provider_fallback_used = result.get("provider_fallback_used")

    try:
        if not result.get("generated"):
            reason = TextAreaFallbackReason(
                reason="bubble_detection_missing_or_failed",
                detail=str(result.get("error") or ""),
                safe_to_ocr=True,
                safe_to_translate=True,
            )
            plan.fallback_reasons.append(reason)
            plan.containers.append(_full_page_fallback_container(page_id, image_size, reason.reason))
            plan.generated = False
            return _finish_plan(plan, started)

        luma_image = _load_luma_image(image_path)
        seen: set[str] = set()
        for fused in result.get("fused_containers", []) or []:
            container = _container_from_fused(page_id, fused, image_size, luma_image=luma_image)
            if not container or container.container_id in seen:
                continue
            seen.add(container.container_id)
            plan.containers.append(container)

        for evidence in result.get("text_area_model_evidence", []) or []:
            container = _container_from_unlinked_text_area_evidence(page_id, evidence, result, image_size, luma_image=luma_image)
            if not container or container.container_id in seen:
                continue
            seen.add(container.container_id)
            plan.containers.append(container)

        _append_deterministic_top_band_caption_containers(
            plan,
            page_id,
            result,
            image_size,
            luma_image=luma_image,
            seen=seen,
        )
        _append_deterministic_vertical_side_caption_containers(
            plan,
            page_id,
            result,
            image_size,
            luma_image=luma_image,
            seen=seen,
        )

        if not plan.containers:
            reason = TextAreaFallbackReason(
                reason="bubble_detection_no_usable_text_area_containers_blocked",
                safe_to_ocr=False,
                safe_to_translate=False,
            )
            plan.fallback_reasons.append(reason)
            plan.containers.append(_blocked_full_page_review_container(page_id, image_size, reason.reason))

        for container in plan.containers:
            if container.comic_text_detector_scope_eligible:
                plan.scopes.append(
                    TextAreaScope(
                        scope_id=f"scope_{container.container_id}",
                        page_id=page_id,
                        container_id=container.container_id,
                        bbox=list(container.bbox),
                        container_type=container.container_type,
                        route_intent=container.route_intent,
                        ocr_eligible=container.ocr_eligible,
                        comic_text_detector_scope_eligible=container.comic_text_detector_scope_eligible,
                        fallback_reason=container.fallback_reason,
                        ocr_eligibility_reason=container.ocr_eligibility_reason,
                        text_area_pre_ocr_authority=container.text_area_pre_ocr_authority,
                        text_area_enriched_from_region=container.text_area_enriched_from_region,
                    )
                )
            plan.evidence.append(
                TextAreaPlanEvidence(
                    source="bubble_detection",
                    source_ids=list(container.source_model_ids),
                    confidence=container.confidence,
                    reason_codes=list(container.evidence_reason_codes),
                    conflict_flags=list(container.conflict_flags),
                )
            )

        plan.generated = True
        return _finish_plan(plan, started)
    except Exception as exc:
        plan.runtime.error = f"{type(exc).__name__}: {exc}"
        plan.fallback_reasons.append(
            TextAreaFallbackReason(
                reason="text_area_plan_exception_compatibility_fallback",
                detail=plan.runtime.error,
                safe_to_ocr=True,
                safe_to_translate=True,
            )
        )
        plan.containers = [_full_page_fallback_container(page_id, image_size, "text_area_plan_exception_compatibility_fallback")]
        return _finish_plan(plan, started)


def enrich_text_area_plan_with_region_records(
    plan: TextAreaPlan | Mapping[str, Any] | None,
    current_region_records: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Return a post-region audit view of a TextAreaPlan.

    The returned plan is diagnostic/enrichment data only. It documents where
    deterministic OCR-time routing agreed with or overrode pre-OCR model
    ownership, but it must not be used to claim stronger pre-OCR gating.
    """

    plan_dict = copy.deepcopy(plan.to_dict() if isinstance(plan, TextAreaPlan) else dict(plan or {}))
    if not plan_dict:
        return {}
    plan_dict["stage"] = "post_region_enriched"
    containers = {
        str(container.get("container_id") or ""): container
        for container in plan_dict.get("containers", []) or []
        if str(container.get("container_id") or "")
    }
    enriched_count = 0
    for region in current_region_records or []:
        if not isinstance(region, Mapping):
            continue
        assignment = assign_bbox_to_text_area_plan(plan_dict, region.get("bbox") or [0, 0, 0, 0], detection_source=region.get("text_area_detection_source") or DETECTION_SCOPED)
        container = containers.get(str(assignment.get("text_area_container_id") or ""))
        if not container:
            continue
        update = _region_enrichment_update(region, container)
        if not update:
            continue
        _apply_container_enrichment(container, update)
        enriched_count += 1
    plan_dict["summary"] = _summary_for_container_dicts(plan_dict.get("containers") or [], len(plan_dict.get("fallback_reasons") or []), plan_dict.get("runtime", {}).get("compatibility_mode"))
    plan_dict["summary"]["enriched_from_region_count"] = enriched_count
    return plan_dict


def assign_bbox_to_text_area_plan(
    plan: TextAreaPlan | Mapping[str, Any] | None,
    bbox: Sequence[Any],
    *,
    min_overlap: float = 0.12,
    detection_source: str = DETECTION_SCOPED,
) -> Dict[str, Any]:
    containers = _plan_containers(plan)
    box = _xywh_to_xyxy(bbox)
    if not box:
        return _fallback_assignment("invalid_detection_bbox")

    best: tuple[float, float, Mapping[str, Any]] | None = None
    for container in containers:
        cbox = _xywh_to_xyxy(container.get("bbox") or [])
        if not cbox:
            continue
        overlap = _intersection_area(box, cbox) / max(1.0, _area(box))
        center = _center_inside(box, cbox)
        score = overlap + (0.35 if center else 0.0) + _container_priority(container) * 0.01
        if best is None or score > best[0]:
            best = (score, overlap, container)

    if not best or (best[1] < min_overlap and not _center_inside(box, _xywh_to_xyxy(best[2].get("bbox") or []))):
        return _fallback_assignment("no_text_area_container_overlap")

    container = best[2]
    source = detection_source or DETECTION_SCOPED
    if not bool(container.get("ocr_eligible", True)):
        source = DETECTION_BLOCKED
    return {
        "text_area_container_id": container.get("container_id"),
        "text_area_container_type": container.get("container_type") or CONTAINER_UNKNOWN,
        "text_area_route_intent": container.get("route_intent") or ROUTE_REVIEW_FALLBACK,
        "text_area_cleanup_authorization": container.get("cleanup_authorization") or _cleanup_authorization_for_container(container)[0],
        "text_area_must_not_mutate": bool(container.get("must_not_mutate", False)),
        "text_area_protection_reason": container.get("protection_reason") or _cleanup_authorization_for_container(container)[1],
        "text_area_authorization_source_stage": container.get("source_stage") or "text_area_plan",
        "text_area_ocr_eligible": bool(container.get("ocr_eligible", True)),
        "text_area_detection_source": source,
        "text_area_fallback_reason": container.get("fallback_reason"),
        "text_area_confidence_tier": container.get("confidence_tier") or "low",
        "text_area_container_bbox": list(container.get("bbox") or []),
        "text_area_reason_codes": list(container.get("evidence_reason_codes") or []),
        "text_area_conflict_flags": list(container.get("conflict_flags") or []),
        "text_area_pre_ocr_authority": bool(container.get("text_area_pre_ocr_authority", True)),
        "text_area_enriched_from_region": bool(container.get("text_area_enriched_from_region", False)),
        "text_area_ocr_eligibility_reason": container.get("ocr_eligibility_reason") or "",
        "text_area_overlap_ratio": round(best[1], 6),
    }


def build_scoped_detection_candidates(
    page_id: str,
    detections: Sequence[Tuple[Sequence[Sequence[float]], float]],
    plan: TextAreaPlan | Mapping[str, Any] | None,
    *,
    detection_source: str = DETECTION_SCOPED,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for idx, (polygon, conf) in enumerate(detections):
        bbox = polygon_to_bbox(polygon)
        assignment = assign_bbox_to_text_area_plan(plan, bbox, detection_source=detection_source)
        candidate = ScopedDetectionCandidate(
            detection_id=f"d{idx:03d}",
            page_id=page_id,
            bbox=bbox,
            polygon=[[float(x), float(y)] for x, y in polygon],
            confidence=float(conf or 0.0),
            text_area_container_id=assignment.get("text_area_container_id"),
            container_type=assignment.get("text_area_container_type") or CONTAINER_UNKNOWN,
            route_intent=assignment.get("text_area_route_intent") or ROUTE_REVIEW_FALLBACK,
            ocr_eligible=bool(assignment.get("text_area_ocr_eligible")),
            detection_source=assignment.get("text_area_detection_source") or DETECTION_COMPATIBILITY_FALLBACK,
            fallback_reason=assignment.get("text_area_fallback_reason"),
            reason_codes=list(assignment.get("text_area_reason_codes") or []),
            conflict_flags=list(assignment.get("text_area_conflict_flags") or []),
            text_area_pre_ocr_authority=bool(assignment.get("text_area_pre_ocr_authority", True)),
            text_area_enriched_from_region=bool(assignment.get("text_area_enriched_from_region", False)),
            cleanup_authorization=assignment.get("text_area_cleanup_authorization") or "",
            must_not_mutate=bool(assignment.get("text_area_must_not_mutate", False)),
            protection_reason=assignment.get("text_area_protection_reason") or "",
            pre_ocr_authority=bool(assignment.get("text_area_pre_ocr_authority", True)),
            source_stage=assignment.get("text_area_authorization_source_stage") or "text_area_plan",
        )
        candidates.append(candidate.to_dict())
    return candidates


def build_scoped_ocr_candidate(
    *,
    page_id: str,
    region_id: str,
    bbox: Sequence[Any],
    assignment: Mapping[str, Any],
    ocr_text: str,
    ocr_confidence: float,
    accepted: bool,
) -> Dict[str, Any]:
    return ScopedOcrCandidate(
        ocr_candidate_id=f"ocr_{region_id}",
        page_id=page_id,
        region_id=region_id,
        bbox=[int(v) for v in bbox[:4]],
        text_area_container_id=assignment.get("text_area_container_id"),
        container_type=assignment.get("text_area_container_type") or CONTAINER_UNKNOWN,
        route_intent=assignment.get("text_area_route_intent") or ROUTE_REVIEW_FALLBACK,
        crop_source=assignment.get("text_area_detection_source") or DETECTION_COMPATIBILITY_FALLBACK,
        fallback_reason=assignment.get("text_area_fallback_reason"),
        ocr_text=ocr_text,
        ocr_confidence=float(ocr_confidence or 0.0),
        accepted=bool(accepted),
        reason_codes=list(assignment.get("text_area_reason_codes") or []),
        text_area_pre_ocr_authority=bool(assignment.get("text_area_pre_ocr_authority", True)),
        text_area_enriched_from_region=bool(assignment.get("text_area_enriched_from_region", False)),
        cleanup_authorization=assignment.get("text_area_cleanup_authorization") or "",
        must_not_mutate=bool(assignment.get("text_area_must_not_mutate", False)),
        protection_reason=assignment.get("text_area_protection_reason") or "",
        pre_ocr_authority=bool(assignment.get("text_area_pre_ocr_authority", True)),
        source_stage=assignment.get("text_area_authorization_source_stage") or "text_area_plan",
    ).to_dict()


def apply_text_area_assignment_to_region(region: Dict[str, Any], assignment: Mapping[str, Any]) -> None:
    if not isinstance(region, dict):
        return
    for key in (
        "text_area_container_id",
        "text_area_container_type",
        "text_area_route_intent",
        "text_area_ocr_eligible",
        "text_area_detection_source",
        "text_area_fallback_reason",
        "text_area_confidence_tier",
        "text_area_container_bbox",
        "text_area_reason_codes",
        "text_area_conflict_flags",
        "text_area_pre_ocr_authority",
        "text_area_enriched_from_region",
        "text_area_ocr_eligibility_reason",
        "text_area_cleanup_authorization",
        "text_area_must_not_mutate",
        "text_area_protection_reason",
        "text_area_authorization_source_stage",
    ):
        region[key] = assignment.get(key)
    render = region.setdefault("render", {})
    render.update({key: assignment.get(key) for key in region if key.startswith("text_area_")})


def write_text_area_plan_artifacts(
    *,
    page_dir: str | Path,
    image_path: str | Path,
    plan: TextAreaPlan | Mapping[str, Any] | None,
    pre_ocr_plan: TextAreaPlan | Mapping[str, Any] | None = None,
    scoped_detection_candidates: Sequence[Mapping[str, Any]] | None = None,
    scoped_ocr_candidates: Sequence[Mapping[str, Any]] | None = None,
    fallback_decisions: Sequence[Mapping[str, Any]] | None = None,
    blocked_text_area_candidates: Sequence[Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    page_dir = Path(page_dir)
    page_dir.mkdir(parents=True, exist_ok=True)
    plan_dict = plan.to_dict() if isinstance(plan, TextAreaPlan) else dict(plan or {})
    paths = {
        "text_area_plan": page_dir / "text_area_plan.json",
        "text_area_plan_pre_ocr": page_dir / "text_area_plan_pre_ocr.json",
        "text_area_plan_enriched": page_dir / "text_area_plan_enriched.json",
        "text_area_plan_overlay": page_dir / "text_area_plan_overlay.jpg",
        "text_area_plan_summary": page_dir / "text_area_plan_summary.md",
        "scoped_detection_candidates": page_dir / "scoped_detection_candidates.json",
        "scoped_ocr_candidates": page_dir / "scoped_ocr_candidates.json",
        "fallback_decisions": page_dir / "fallback_decisions.json",
        "blocked_text_area_candidates": page_dir / "blocked_text_area_candidates.json",
        "caption_localization_candidates": page_dir / "caption_localization_candidates.json",
    }
    pre_ocr_dict = pre_ocr_plan.to_dict() if isinstance(pre_ocr_plan, TextAreaPlan) else dict(pre_ocr_plan or plan_dict or {})
    paths["text_area_plan"].write_text(json.dumps(plan_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["text_area_plan_pre_ocr"].write_text(json.dumps(pre_ocr_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["text_area_plan_enriched"].write_text(json.dumps(plan_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["scoped_detection_candidates"].write_text(json.dumps(list(scoped_detection_candidates or []), ensure_ascii=False, indent=2), encoding="utf-8")
    paths["scoped_ocr_candidates"].write_text(json.dumps(list(scoped_ocr_candidates or []), ensure_ascii=False, indent=2), encoding="utf-8")
    paths["fallback_decisions"].write_text(json.dumps(list(fallback_decisions or []), ensure_ascii=False, indent=2), encoding="utf-8")
    blocked = list(blocked_text_area_candidates or []) + _blocked_containers_from_plan(plan_dict)
    paths["blocked_text_area_candidates"].write_text(json.dumps(blocked, ensure_ascii=False, indent=2), encoding="utf-8")
    caption_candidates = _caption_localization_candidates_from_plan(plan_dict)
    paths["caption_localization_candidates"].write_text(json.dumps(caption_candidates, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_plan_summary(paths["text_area_plan_summary"], plan_dict, scoped_detection_candidates or [], scoped_ocr_candidates or [], fallback_decisions or [], blocked)
    _write_plan_overlay(paths["text_area_plan_overlay"], image_path, plan_dict, scoped_detection_candidates or [])
    return {key: str(path) for key, path in paths.items()}


def polygon_to_bbox(polygon: Sequence[Sequence[Any]]) -> List[int]:
    xs: List[float] = []
    ys: List[float] = []
    for point in polygon or []:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            xs.append(float(point[0]))
            ys.append(float(point[1]))
        except Exception:
            continue
    if not xs or not ys:
        return [0, 0, 0, 0]
    x0 = int(math.floor(min(xs)))
    y0 = int(math.floor(min(ys)))
    x1 = int(math.ceil(max(xs)))
    y1 = int(math.ceil(max(ys)))
    return [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]


def _region_enrichment_update(region: Mapping[str, Any], container: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    semantic = str(region.get("type") or region.get("semantic_class") or "").strip()
    flags = region.get("flags") if isinstance(region.get("flags"), Mapping) else {}
    render = region.get("render") if isinstance(region.get("render"), Mapping) else {}
    cleanup = str(region.get("cleanup_mode") or render.get("cleanup_mode") or "").strip()
    reason = str(region.get("classification_reason") or render.get("classification_reason") or "").strip()
    current_type = str(container.get("container_type") or "")
    current_route = str(container.get("route_intent") or "")
    current_is_translatable = current_route in {ROUTE_TRANSLATE_SPEECH, ROUTE_TRANSLATE_CAPTION}

    if (
        reason in DECORATIVE_REGION_REASONS
        or semantic in {"decorative_text", "sfx"}
        or (cleanup == "preserve" and bool(flags.get("ignore")))
    ):
        if current_is_translatable:
            return None
        return {
            "container_type": CONTAINER_SFX,
            "route_intent": ROUTE_PRESERVE_SFX,
            "ocr_eligible": False,
            "comic_text_detector_scope_eligible": False,
            "confidence_tier": "conflict_preserve_wins",
            "fallback_reason": "post_region_deterministic_preserve_evidence",
            "ocr_eligibility_reason": "blocked_deterministic_preserve_evidence",
            "reason_code": f"text_area_plan:post_region_sfx_decorative:{reason or semantic or cleanup}",
        }

    if reason in CAPTION_REGION_REASONS or semantic == "background_text" or bool(flags.get("bg_text")):
        if current_type == CONTAINER_SPEECH:
            return None
        return {
            "container_type": CONTAINER_CAPTION,
            "route_intent": ROUTE_TRANSLATE_CAPTION,
            "ocr_eligible": True,
            "comic_text_detector_scope_eligible": True,
            "fallback_reason": "post_region_deterministic_caption_evidence",
            "ocr_eligibility_reason": "caption_background_container",
            "reason_code": f"text_area_plan:post_region_caption:{reason or semantic}",
        }

    if semantic == "speech_bubble" or reason in SPEECH_REGION_REASONS:
        if current_type != CONTAINER_UNKNOWN:
            return None
        return {
            "container_type": CONTAINER_SPEECH,
            "route_intent": ROUTE_TRANSLATE_SPEECH,
            "ocr_eligible": True,
            "comic_text_detector_scope_eligible": True,
            "fallback_reason": "post_region_deterministic_speech_evidence",
            "ocr_eligibility_reason": "speech_container_after_region_evidence",
            "reason_code": f"text_area_plan:post_region_speech:{reason or semantic}",
        }

    return None


def _apply_container_enrichment(container: MutableMapping[str, Any], update: Mapping[str, Any]) -> None:
    changed = False
    for key in (
        "container_type",
        "route_intent",
        "ocr_eligible",
        "comic_text_detector_scope_eligible",
        "confidence_tier",
        "fallback_reason",
        "ocr_eligibility_reason",
    ):
        if key in update and container.get(key) != update.get(key):
            container[key] = update.get(key)
            changed = True
    if not changed:
        return
    container["text_area_pre_ocr_authority"] = False
    container["text_area_enriched_from_region"] = True
    reasons = list(container.get("evidence_reason_codes") or [])
    reason_code = str(update.get("reason_code") or "")
    if reason_code and reason_code not in reasons:
        reasons.append(reason_code)
    if "text_area_plan:post_region_enriched_audit_only" not in reasons:
        reasons.append("text_area_plan:post_region_enriched_audit_only")
    container["evidence_reason_codes"] = reasons
    container["human_review_required"] = True


def _summary_for_container_dicts(
    containers: Sequence[Mapping[str, Any]],
    fallback_count: int,
    compatibility_mode: Any,
) -> Dict[str, Any]:
    by_type: Dict[str, int] = {}
    by_intent: Dict[str, int] = {}
    blocked_by_type: Dict[str, int] = {}
    ocr_eligible = 0
    scope_eligible = 0
    review_only_blocked = 0
    for container in containers:
        ctype = str(container.get("container_type") or CONTAINER_UNKNOWN)
        intent = str(container.get("route_intent") or ROUTE_REVIEW_FALLBACK)
        by_type[ctype] = by_type.get(ctype, 0) + 1
        by_intent[intent] = by_intent.get(intent, 0) + 1
        if bool(container.get("ocr_eligible")):
            ocr_eligible += 1
        if bool(container.get("comic_text_detector_scope_eligible")):
            scope_eligible += 1
        if not bool(container.get("ocr_eligible")) or not bool(container.get("comic_text_detector_scope_eligible")):
            blocked_by_type[ctype] = blocked_by_type.get(ctype, 0) + 1
            if ctype == CONTAINER_UNKNOWN:
                review_only_blocked += 1
    return {
        "container_count": len(containers),
        "scope_count": scope_eligible,
        "by_container_type": by_type,
        "by_route_intent": by_intent,
        "ocr_eligible_containers": ocr_eligible,
        "ctd_scope_eligible_containers": scope_eligible,
        "blocked_containers_by_type": blocked_by_type,
        "blocked_sfx_decorative_containers": blocked_by_type.get(CONTAINER_SFX, 0),
        "blocked_review_only_unknown_containers": review_only_blocked,
        "fallback_count": int(fallback_count or 0),
        "compatibility_mode": compatibility_mode,
    }


def _cleanup_authorization_for_container(container: Mapping[str, Any] | TextAreaContainer) -> tuple[str, str, bool]:
    if isinstance(container, TextAreaContainer):
        route = str(container.route_intent or ROUTE_REVIEW_FALLBACK)
        ctype = str(container.container_type or CONTAINER_UNKNOWN)
        reason_values = list(container.evidence_reason_codes or []) + list(container.conflict_flags or [])
        fallback = str(container.fallback_reason or "")
    else:
        route = str(container.get("route_intent") or ROUTE_REVIEW_FALLBACK)
        ctype = str(container.get("container_type") or CONTAINER_UNKNOWN)
        reason_values = list(container.get("evidence_reason_codes") or []) + list(container.get("conflict_flags") or [])
        fallback = str(container.get("fallback_reason") or "")
    marker = " ".join([route, ctype, fallback] + [str(item) for item in reason_values]).lower()
    if any(token in marker for token in ("non_text", "non-text", "art_only", "non_translation_art")):
        return AUTH_PROTECT_ART_OR_NON_TEXT, "explicit_art_or_non_text_authorization", True
    if route == ROUTE_PRESERVE_SFX or ctype == CONTAINER_SFX or any(
        token in marker for token in ("sfx", "decorative", "preserve_sfx_decorative")
    ):
        return AUTH_PROTECT_SFX_DECORATIVE, "explicit_sfx_decorative_authorization", True
    if route == ROUTE_TRANSLATE_SPEECH:
        return AUTH_CLEANUP_TRANSLATE_SPEECH, "", False
    if route == ROUTE_TRANSLATE_CAPTION:
        if "deterministic_vertical_side_caption_search" in marker or "vertical_side_caption_search" in marker:
            return AUTH_REVIEW_UNKNOWN_NOT_CLEANUP, "deterministic_side_caption_requires_review", True
        if "caption" in marker and "background" not in marker:
            return AUTH_CLEANUP_TRANSLATE_CAPTION, "", False
        return AUTH_CLEANUP_TRANSLATE_BACKGROUND, "", False
    if route == ROUTE_REVIEW_FALLBACK or ctype == CONTAINER_UNKNOWN:
        return AUTH_REVIEW_UNKNOWN_NOT_CLEANUP, "review_unknown_not_cleanup", True
    return AUTH_OUTSIDE_CLEANUP_SCOPE, "outside_cleanup_scope", True


def _apply_cleanup_authorization(container: TextAreaContainer) -> None:
    auth, reason, must_not_mutate = _cleanup_authorization_for_container(container)
    container.cleanup_authorization = auth
    container.must_not_mutate = must_not_mutate
    container.protection_reason = reason
    container.pre_ocr_authority = bool(container.text_area_pre_ocr_authority)
    container.source_stage = "text_area_plan_pre_ocr" if container.text_area_pre_ocr_authority else "text_area_plan_region_enriched"
    container.parent_source_evidence = {
        "source_model_ids": list(container.source_model_ids or []),
        "evidence_reason_codes": list(container.evidence_reason_codes or []),
        "conflict_flags": list(container.conflict_flags or []),
    }


def _finish_plan(plan: TextAreaPlan, started: float) -> TextAreaPlan:
    plan.runtime.generated = bool(plan.generated)
    plan.runtime.runtime_sec = round(time.perf_counter() - started, 6)
    by_type: Dict[str, int] = {}
    by_intent: Dict[str, int] = {}
    blocked_by_type: Dict[str, int] = {}
    ocr_eligible = 0
    scope_eligible = 0
    review_only_blocked = 0
    for container in plan.containers:
        _apply_cleanup_authorization(container)
        by_type[container.container_type] = by_type.get(container.container_type, 0) + 1
        by_intent[container.route_intent] = by_intent.get(container.route_intent, 0) + 1
        if container.ocr_eligible:
            ocr_eligible += 1
        if container.comic_text_detector_scope_eligible:
            scope_eligible += 1
        if not container.ocr_eligible or not container.comic_text_detector_scope_eligible:
            blocked_by_type[container.container_type] = blocked_by_type.get(container.container_type, 0) + 1
            if container.container_type == CONTAINER_UNKNOWN:
                review_only_blocked += 1
    plan.summary = {
        "container_count": len(plan.containers),
        "scope_count": len(plan.scopes),
        "by_container_type": by_type,
        "by_route_intent": by_intent,
        "ocr_eligible_containers": ocr_eligible,
        "ctd_scope_eligible_containers": scope_eligible,
        "blocked_containers_by_type": blocked_by_type,
        "blocked_sfx_decorative_containers": blocked_by_type.get(CONTAINER_SFX, 0),
        "blocked_review_only_unknown_containers": review_only_blocked,
        "fallback_count": len(plan.fallback_reasons),
        "compatibility_mode": plan.runtime.compatibility_mode,
    }
    return plan


def _container_from_fused(
    page_id: str,
    fused: Mapping[str, Any],
    image_size: Tuple[int, int],
    *,
    luma_image: Any = None,
) -> TextAreaContainer | None:
    container_id = str(fused.get("fused_container_id") or "")
    if not container_id:
        return None
    fused_type = str(fused.get("fused_container_type") or "")
    confidence = str(fused.get("confidence") or "low")
    reasons = list(fused.get("reason_codes") or [])
    conflicts = list(fused.get("conflict_flags") or [])
    bbox = _bbox_xyxy_to_xywh(fused.get("mask_bbox") or fused.get("bbox"), image_size)
    source_ids = list(fused.get("linked_kitsumed_mask_ids") or []) + list(fused.get("linked_ogkalu_detection_ids") or [])
    confidence_tier = _confidence_tier_from_fused(fused)
    visual = _container_visual_stats(luma_image, bbox, image_size)
    clipped = _is_clipped_or_degenerate_bbox(bbox, image_size)

    if conflicts:
        ctype = CONTAINER_SFX if _has_preserve_conflict(conflicts, reasons) else CONTAINER_UNKNOWN
        route = ROUTE_PRESERVE_SFX if ctype == CONTAINER_SFX else ROUTE_REVIEW_FALLBACK
        return TextAreaContainer(
            container_id=container_id,
            page_id=page_id,
            container_type=ctype,
            bbox=bbox,
            mask_summary={"mask_bbox": _safe_list(fused.get("mask_bbox"))},
            source_model_ids=source_ids,
            confidence=confidence,
            confidence_tier="conflict_preserve_wins" if ctype == CONTAINER_SFX else confidence_tier,
            route_intent=route,
            ocr_eligible=False,
            comic_text_detector_scope_eligible=False,
            fallback_reason="model_conflict_review_required",
            evidence_reason_codes=reasons + ["text_area_plan:conflict_fail_closed"],
            conflict_flags=conflicts,
            human_review_required=True,
            ocr_eligibility_reason="blocked_conflict_fail_closed",
        )

    if fused_type == "speech_bubble":
        has_kitsumed = bool(fused.get("linked_kitsumed_mask_ids"))
        has_ogkalu = bool(fused.get("linked_ogkalu_detection_ids"))
        large_mask_primary = has_kitsumed and _bbox_area_xywh(bbox) >= 45000
        if (confidence == "high" and has_kitsumed) or (confidence == "medium" and large_mask_primary):
            extra_reasons = ["text_area_plan:medium_mask_primary_large_container"] if confidence == "medium" else []
            return TextAreaContainer(
                container_id=container_id,
                page_id=page_id,
                container_type=CONTAINER_SPEECH,
                bbox=bbox,
                mask_summary={"mask_bbox": _safe_list(fused.get("mask_bbox"))},
                source_model_ids=source_ids,
                confidence=confidence,
                confidence_tier=confidence_tier,
                route_intent=ROUTE_TRANSLATE_SPEECH,
                ocr_eligible=True,
                comic_text_detector_scope_eligible=True,
                evidence_reason_codes=reasons + extra_reasons + ["text_area_plan:speech_mask_container"],
                ocr_eligibility_reason="speech_container",
            )
        if has_kitsumed and not has_ogkalu and _mask_primary_speech_activation_completeness_allowed(
            confidence=confidence,
            bbox=bbox,
            reasons=reasons,
            visual=visual,
            image_size=image_size,
            clipped=clipped,
        ):
            return TextAreaContainer(
                container_id=container_id,
                page_id=page_id,
                container_type=CONTAINER_SPEECH,
                bbox=bbox,
                mask_summary={"mask_bbox": _safe_list(fused.get("mask_bbox"))},
                source_model_ids=source_ids,
                confidence=confidence,
                confidence_tier=confidence_tier,
                route_intent=ROUTE_TRANSLATE_SPEECH,
                ocr_eligible=True,
                comic_text_detector_scope_eligible=True,
                evidence_reason_codes=reasons
                + [
                    "text_area_plan:mask_primary_activation_completeness",
                    "text_area_plan:scope_ocr_required_for_visible_speech_root",
                    "text_area_plan:speech_mask_container",
                ],
                ocr_eligibility_reason="speech_activation_completeness_scope_required",
                human_review_required=True,
            )
        if has_kitsumed and not has_ogkalu and _mask_primary_irregular_speech_activation_allowed(
            confidence=confidence,
            bbox=bbox,
            reasons=reasons,
            visual=visual,
            image_size=image_size,
            clipped=clipped,
        ):
            return TextAreaContainer(
                container_id=container_id,
                page_id=page_id,
                container_type=CONTAINER_SPEECH,
                bbox=bbox,
                mask_summary={"mask_bbox": _safe_list(fused.get("mask_bbox"))},
                source_model_ids=source_ids,
                confidence=confidence,
                confidence_tier=confidence_tier,
                route_intent=ROUTE_TRANSLATE_SPEECH,
                ocr_eligible=True,
                comic_text_detector_scope_eligible=True,
                evidence_reason_codes=reasons
                + [
                    "text_area_plan:mask_primary_irregular_speech_activation",
                    "text_area_plan:scope_ocr_required_for_irregular_speech",
                    "text_area_plan:speech_mask_container",
                ],
                ocr_eligibility_reason="irregular_speech_mask_primary_scope_required",
                human_review_required=True,
            )
        review_reason = "mask_primary_container_requires_review" if has_kitsumed else "low_confidence_speech_mask"
        if has_kitsumed and not has_ogkalu:
            reasons = reasons + ["text_area_plan:mask_primary_without_support_review_only"]
        return TextAreaContainer(
            container_id=container_id,
            page_id=page_id,
            container_type=CONTAINER_UNKNOWN,
            bbox=bbox,
            mask_summary={"mask_bbox": _safe_list(fused.get("mask_bbox"))},
            source_model_ids=source_ids,
            confidence=confidence,
            confidence_tier=confidence_tier,
            route_intent=ROUTE_REVIEW_FALLBACK,
            ocr_eligible=not (has_kitsumed and not has_ogkalu),
            comic_text_detector_scope_eligible=not (has_kitsumed and not has_ogkalu),
            fallback_reason=review_reason,
            evidence_reason_codes=reasons + ["text_area_plan:low_confidence_fallback"],
            human_review_required=True,
            ocr_eligibility_reason="blocked_mask_primary_review_only" if has_kitsumed and not has_ogkalu else "speech_mask_review_fallback",
        )

    if (fused_type == "caption_or_background_candidate" or fused_type == "free_text") and _looks_like_caption_background_bbox(bbox, image_size):
        return TextAreaContainer(
            container_id=container_id,
            page_id=page_id,
            container_type=CONTAINER_CAPTION,
            bbox=bbox,
            source_model_ids=source_ids,
            confidence=confidence,
            confidence_tier="text_free_review_only",
            route_intent=ROUTE_TRANSLATE_CAPTION,
            ocr_eligible=True,
            comic_text_detector_scope_eligible=True,
            fallback_reason="caption_background_model_candidate_review",
            evidence_reason_codes=reasons + ["text_area_plan:caption_background_candidate"],
            human_review_required=True,
            ocr_eligibility_reason="caption_background_container",
        )

    if fused_type == "sfx_or_decorative_candidate" or _looks_like_pre_ocr_sfx_or_decorative(fused_type, reasons, bbox, visual, image_size):
        return TextAreaContainer(
            container_id=container_id,
            page_id=page_id,
            container_type=CONTAINER_SFX,
            bbox=bbox,
            source_model_ids=source_ids,
            confidence=confidence,
            confidence_tier="conflict_preserve_wins",
            route_intent=ROUTE_PRESERVE_SFX,
            ocr_eligible=False,
            comic_text_detector_scope_eligible=False,
            fallback_reason="sfx_decorative_model_candidate",
            evidence_reason_codes=reasons + _visual_reason_codes(visual) + ["text_area_plan:sfx_decorative_preserve"],
            conflict_flags=conflicts,
            human_review_required=True,
            ocr_eligibility_reason="blocked_sfx_decorative_art_container",
        )

    safe_unknown = _safe_unknown_ocr_fallback(fused_type, reasons, bbox, visual, image_size)
    if safe_unknown:
        return TextAreaContainer(
            container_id=container_id,
            page_id=page_id,
            container_type=CONTAINER_UNKNOWN,
            bbox=bbox,
            mask_summary={"mask_bbox": _safe_list(fused.get("mask_bbox"))},
            source_model_ids=source_ids,
            confidence=confidence,
            confidence_tier=confidence_tier,
            route_intent=ROUTE_REVIEW_FALLBACK,
            ocr_eligible=True,
            comic_text_detector_scope_eligible=True,
            fallback_reason="safe_unknown_ocr_for_text_conservation",
            evidence_reason_codes=reasons + _visual_reason_codes(visual) + ["text_area_plan:unknown_fallback_ocr_for_text_conservation"],
            conflict_flags=conflicts,
            human_review_required=True,
            ocr_eligibility_reason="unknown_fallback_ocr_for_text_conservation",
        )

    return TextAreaContainer(
        container_id=container_id,
        page_id=page_id,
        container_type=CONTAINER_UNKNOWN,
        bbox=bbox,
        source_model_ids=source_ids,
        confidence=confidence,
        confidence_tier=confidence_tier,
        route_intent=ROUTE_REVIEW_FALLBACK,
        ocr_eligible=False,
        comic_text_detector_scope_eligible=False,
        fallback_reason=(
            "clipped_or_degenerate_model_container_review_only"
            if clipped
            else f"unresolved_fused_container_type:{fused_type or 'unknown'}"
        ),
        evidence_reason_codes=reasons + _visual_reason_codes(visual) + ["text_area_plan:unknown_fallback_blocked_review_only"],
        conflict_flags=conflicts,
        human_review_required=True,
        ocr_eligibility_reason="blocked_unknown_review_only",
    )


def _container_from_unlinked_text_area_evidence(
    page_id: str,
    evidence: Mapping[str, Any],
    result: Mapping[str, Any],
    image_size: Tuple[int, int],
    *,
    luma_image: Any = None,
) -> TextAreaContainer | None:
    evidence_id = str(evidence.get("model_evidence_id") or evidence.get("evidence_id") or "")
    if not evidence_id:
        return None
    if _evidence_is_linked_to_fused_container(evidence_id, result):
        return None
    class_name = str(evidence.get("class_name") or "")
    bbox = _bbox_xyxy_to_xywh(evidence.get("bbox_xyxy") or evidence.get("bbox"), image_size)
    confidence = float(evidence.get("confidence") or 0.0)
    visual = _container_visual_stats(luma_image, bbox, image_size)
    top_caption = class_name == "text_free" and _looks_like_caption_background_bbox(bbox, image_size)
    if top_caption:
        return TextAreaContainer(
            container_id=f"ogkalu_{evidence_id}",
            page_id=page_id,
            container_type=CONTAINER_CAPTION,
            bbox=bbox,
            source_model_ids=[evidence_id],
            confidence=round(confidence, 6),
            confidence_tier="text_free_review_only",
            route_intent=ROUTE_TRANSLATE_CAPTION,
            ocr_eligible=True,
            comic_text_detector_scope_eligible=True,
            fallback_reason="top_row_text_free_caption_candidate",
            evidence_reason_codes=["ogkalu_text_free", "text_area_plan:top_caption_background_candidate"] + _visual_reason_codes(visual),
            human_review_required=True,
            ocr_eligibility_reason="caption_background_container",
        )
    if class_name == "text_free" and _looks_like_pre_ocr_sfx_or_decorative("free_text", [f"ogkalu_class:{class_name}"], bbox, visual, image_size):
        return TextAreaContainer(
            container_id=f"ogkalu_{evidence_id}",
            page_id=page_id,
            container_type=CONTAINER_SFX,
            bbox=bbox,
            source_model_ids=[evidence_id],
            confidence=round(confidence, 6),
            confidence_tier="text_free_review_only",
            route_intent=ROUTE_PRESERVE_SFX,
            ocr_eligible=False,
            comic_text_detector_scope_eligible=False,
            fallback_reason="ogkalu_text_free_non_caption_art_review",
            evidence_reason_codes=[f"ogkalu_class:{class_name}", "text_area_plan:text_free_non_caption_preserve_review"] + _visual_reason_codes(visual),
            human_review_required=True,
            ocr_eligibility_reason="blocked_text_free_non_caption",
        )
    return TextAreaContainer(
        container_id=f"ogkalu_{evidence_id}",
        page_id=page_id,
        container_type=CONTAINER_UNKNOWN,
        bbox=bbox,
        source_model_ids=[evidence_id],
        confidence=round(confidence, 6),
        confidence_tier="text_free_review_only" if class_name == "text_free" else "text_bubble_review_container",
        route_intent=ROUTE_REVIEW_FALLBACK,
        ocr_eligible=False,
        comic_text_detector_scope_eligible=False,
        fallback_reason=f"ogkalu_only_{class_name or 'unknown'}_review",
        evidence_reason_codes=[f"ogkalu_class:{class_name}", "text_area_plan:ogkalu_only_review_blocked"] + _visual_reason_codes(visual),
        human_review_required=True,
        ocr_eligibility_reason="blocked_ogkalu_only_review",
    )


def _append_deterministic_top_band_caption_containers(
    plan: TextAreaPlan,
    page_id: str,
    result: Mapping[str, Any],
    image_size: Tuple[int, int],
    *,
    luma_image: Any = None,
    seen: set[str] | None = None,
) -> None:
    if luma_image is None:
        return
    has_model_signal = _has_top_band_caption_search_signal(result, image_size)
    visual_bboxes = _top_band_visual_caption_search_bboxes(luma_image, image_size)
    width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    search_height = max(120, int(height * 0.22))
    candidates = [
        (
            "det_top_caption_day_center",
            [int(width * 0.56), 0, int(width * 0.09), int(search_height * 0.72)],
            "text_area_plan:deterministic_top_band_day_caption_search",
            "static",
        ),
        (
            "det_top_caption_far_right",
            [int(width * 0.82), 0, int(width * 0.16), search_height],
            "text_area_plan:deterministic_top_band_far_right_caption_search",
            "static",
        ),
        (
            "det_top_caption_center",
            [int(width * 0.48), 0, int(width * 0.18), search_height],
            TOP_BAND_CAPTION_SEARCH_REASON,
            "static",
        ),
        (
            "det_top_caption_right",
            [int(width * 0.68), 0, int(width * 0.20), search_height],
            TOP_BAND_CAPTION_SEARCH_REASON,
            "static",
        ),
    ]
    if not has_model_signal and not visual_bboxes and not any(
        _top_band_static_caption_candidate_has_ink(luma_image, raw_bbox, image_size)
        for _cid, raw_bbox, _reason, _kind in candidates
    ):
        return
    for index, bbox in enumerate(visual_bboxes):
        candidates.append(
            (
                f"det_top_caption_visual_{index:02d}",
                bbox,
                "text_area_plan:deterministic_top_band_visual_text_search",
                "visual",
            )
        )
    source_ids = _top_band_caption_search_source_ids(result, image_size)
    pending: List[Tuple[str, List[int], str, str]] = []
    for container_id, raw_bbox, search_reason, candidate_kind in candidates:
        if seen is not None and container_id in seen:
            continue
        bbox = _normalize_xywh(raw_bbox, image_size)
        if candidate_kind == "static":
            localized = _localize_top_caption_bbox(luma_image, bbox, image_size)
            if localized is not None:
                bbox = localized
                search_reason = f"{search_reason}:localized_ink"
            else:
                # Keep broad top-band boxes as scoped OCR search areas only.
                # Controller/hierarchy must later render and clean against the
                # OCR/glyph evidence, and duplicate partial captions are
                # suppressed before translation.
                search_reason = f"{search_reason}:search_scope_only"
        if _top_band_caption_candidate_is_duplicate(bbox, pending, plan.containers, source_ids):
            continue
        pending.append((container_id, bbox, search_reason, candidate_kind))

    for container_id, bbox, search_reason, _candidate_kind in pending:
        if _caption_search_overlaps_blocking_container(bbox, plan.containers):
            continue
        visual = _container_visual_stats(luma_image, bbox, image_size)
        plan.containers.append(
            TextAreaContainer(
                container_id=container_id,
                page_id=page_id,
                container_type=CONTAINER_CAPTION,
                bbox=bbox,
                source_model_ids=source_ids,
                confidence="deterministic",
                confidence_tier="deterministic_top_band_caption_search",
                route_intent=ROUTE_TRANSLATE_CAPTION,
                ocr_eligible=True,
                comic_text_detector_scope_eligible=True,
                fallback_reason="deterministic_top_band_caption_search",
                evidence_reason_codes=[
                    search_reason,
                    "text_area_plan:caption_background_container",
                ]
                + _visual_reason_codes(visual),
                human_review_required=True,
                ocr_eligibility_reason="caption_background_container_strict_ocr_gate",
            )
        )
        if seen is not None:
            seen.add(container_id)


def _localize_top_caption_bbox(luma_image: Any, search_bbox: Sequence[int], image_size: Tuple[int, int]) -> List[int] | None:
    if luma_image is None:
        return None
    x, y, w, h = _coerce_xywh(search_bbox)
    width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    if w <= 0 or h <= 0:
        return None
    try:
        crop = luma_image.crop((x, y, min(width, x + w), min(height, y + h))).convert("L")
        cw, ch = crop.size
        if cw <= 0 or ch <= 0:
            return None
        pixels = crop.load()
        dark_points: List[Tuple[int, int]] = []
        for cy in range(ch):
            for cx in range(cw):
                if pixels[cx, cy] <= 124:
                    dark_points.append((cx, cy))
        if len(dark_points) < max(24, int(cw * ch * 0.004)):
            return None
        xs = [pt[0] for pt in dark_points]
        ys = [pt[1] for pt in dark_points]
        lx0 = max(0, min(xs) - max(6, int(cw * 0.035)))
        lx1 = min(cw, max(xs) + 1 + max(6, int(cw * 0.035)))
        ly0 = max(0, min(ys) - max(8, int(ch * 0.030)))
        ly1 = min(ch, max(ys) + 1 + max(8, int(ch * 0.030)))
        bw = lx1 - lx0
        bh = ly1 - ly0
        if bw < 12 or bh < 32:
            return None
        localized_area = bw * bh
        search_area = max(1, cw * ch)
        if localized_area > search_area * 0.62:
            return None
        localized = _normalize_xywh([x + lx0, y + ly0, bw, bh], image_size)
        if not _looks_like_top_caption_bbox(localized, image_size):
            return None
        return localized
    except Exception:
        return None


def _top_band_caption_candidate_is_duplicate(
    bbox: Sequence[int],
    pending: Sequence[Tuple[str, Sequence[int], str, str]],
    existing_containers: Sequence[TextAreaContainer],
    source_ids: Sequence[str],
) -> bool:
    for _cid, existing_bbox, _reason, _kind in pending:
        if _intersection_ratio_xywh(bbox, existing_bbox) >= 0.70:
            return True
        if _inside_ratio_xywh(bbox, existing_bbox) >= 0.80 or _inside_ratio_xywh(existing_bbox, bbox) >= 0.80:
            return True
    source_id_set = {str(item) for item in source_ids if str(item)}
    for container in existing_containers:
        if container.container_type != CONTAINER_CAPTION:
            continue
        overlap = _intersection_ratio_xywh(bbox, container.bbox)
        if overlap >= 0.82:
            return True
        existing_source_ids = {str(item) for item in (container.source_model_ids or []) if str(item)}
        if source_id_set and existing_source_ids and source_id_set & existing_source_ids and overlap >= 0.50:
            return True
    return False


def _top_band_visual_caption_search_bboxes(luma_image: Any, image_size: Tuple[int, int]) -> List[List[int]]:
    """Find narrow top-band text columns when model evidence missed captions.

    This stays within TextAreaPlan admission: it only creates caption/background
    OCR scopes, and later OCR quality/SFX gates still decide whether text enters
    translation. Wide impact marks and panel borders are rejected here.
    """
    if luma_image is None:
        return []
    width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    search_height = max(120, int(height * 0.22))
    try:
        crop = luma_image.crop((0, 0, width, min(height, search_height))).convert("L")
        cw, ch = crop.size
        if cw <= 0 or ch <= 0:
            return []
        pixels = crop.load()
    except Exception:
        return []

    col_counts: List[int] = []
    for cx in range(cw):
        count = 0
        for cy in range(ch):
            if pixels[cx, cy] <= 112:
                count += 1
        col_counts.append(count)
    threshold = max(5, int(ch * 0.012))
    active_cols = [idx for idx, count in enumerate(col_counts) if count >= threshold]
    if not active_cols:
        return []

    groups: List[Tuple[int, int]] = []
    start = prev = active_cols[0]
    max_gap = max(8, int(width * 0.012))
    for idx in active_cols[1:]:
        if idx - prev <= max_gap:
            prev = idx
            continue
        groups.append((start, prev))
        start = prev = idx
    groups.append((start, prev))

    candidates: List[List[int]] = []
    for gx0, gx1 in groups:
        gw = gx1 - gx0 + 1
        if gw < 10 or gw > max(150, int(width * 0.14)):
            continue
        band_dark = sum(col_counts[gx0 : gx1 + 1])
        if band_dark < max(70, int(gw * ch * 0.014)):
            continue
        pad_x = max(6, int(gw * 0.20))
        lx0 = max(0, gx0 - pad_x)
        lx1 = min(cw, gx1 + 1 + pad_x)
        row_counts: List[int] = []
        for cy in range(ch):
            count = 0
            for cx in range(lx0, lx1):
                if pixels[cx, cy] <= 124:
                    count += 1
            row_counts.append(count)
        row_threshold = max(2, int((lx1 - lx0) * 0.040))
        active_rows = [idx for idx, count in enumerate(row_counts) if count >= row_threshold]
        if not active_rows:
            continue
        ly0 = max(0, min(active_rows) - max(8, int(ch * 0.018)))
        ly1 = min(ch, max(active_rows) + 1 + max(8, int(ch * 0.018)))
        bh = ly1 - ly0
        bw = lx1 - lx0
        if bh < max(48, int(height * 0.025)):
            continue
        if bh > int(height * 0.24) and bw > int(width * 0.08):
            continue
        area = max(1, bw * bh)
        dark_density = band_dark / float(area)
        if dark_density > 0.42:
            continue
        candidate = _normalize_xywh([lx0, ly0, bw, bh], image_size)
        if _looks_like_top_caption_bbox(candidate, image_size):
            candidates.append(candidate)

    # Deduplicate nested visual candidates, keeping the tighter scope.
    deduped: List[List[int]] = []
    for candidate in candidates:
        if any(_intersection_ratio_xywh(candidate, existing) > 0.72 for existing in deduped):
            continue
        deduped.append(candidate)
        if len(deduped) >= 4:
            break
    return deduped


def _top_band_static_caption_candidate_has_ink(
    luma_image: Any,
    bbox: Sequence[Any],
    image_size: Tuple[int, int],
) -> bool:
    if luma_image is None:
        return False
    try:
        x, y, w, h = _coerce_xywh(bbox)
        width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
        if w <= 0 or h <= 0:
            return False
        crop = luma_image.crop((max(0, x), max(0, y), min(width, x + w), min(height, y + h))).convert("L")
        pixels = list(crop.getdata())
        if not pixels:
            return False
        dark_ratio = sum(1 for value in pixels if value <= 116) / float(len(pixels))
        mid_ratio = sum(1 for value in pixels if value <= 170) / float(len(pixels))
        # These static scopes are still gated by TextAreaPlan ownership and OCR
        # quality. The ink check only avoids adding empty top-band scopes.
        return 0.012 <= dark_ratio <= 0.36 and mid_ratio >= 0.025
    except Exception:
        return False


def _append_deterministic_vertical_side_caption_containers(
    plan: TextAreaPlan,
    page_id: str,
    result: Mapping[str, Any],
    image_size: Tuple[int, int],
    *,
    luma_image: Any = None,
    seen: set[str] | None = None,
) -> None:
    if luma_image is None:
        return
    width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    source_items = _side_caption_signal_items(result, image_size)
    if not source_items:
        return
    added = 0
    added_boxes: list[list[int]] = []
    for item in source_items:
        seed_bbox = _bbox_xyxy_to_xywh(item.get("mask_bbox") or item.get("bbox_xyxy") or item.get("bbox"), image_size)
        search_bbox = _side_caption_search_bbox(seed_bbox, image_size)
        localized = _localize_side_caption_bbox(luma_image, search_bbox, image_size)
        bbox = localized or search_bbox
        if any(_intersection_ratio_xywh(bbox, existing) >= 0.35 for existing in added_boxes):
            continue
        if _caption_search_overlaps_blocking_container(bbox, plan.containers):
            continue
        container_id = f"det_side_caption_right_{added:02d}"
        if seen is not None and container_id in seen:
            continue
        visual = _container_visual_stats(luma_image, bbox, image_size)
        item_id = str(item.get("model_evidence_id") or item.get("evidence_id") or item.get("fused_container_id") or "")
        reason_codes = [
            SIDE_CAPTION_SEARCH_REASON,
            "text_area_plan:caption_background_container",
            "text_area_plan:vertical_side_caption_search",
        ] + _visual_reason_codes(visual)
        if localized:
            reason_codes.append("text_area_plan:vertical_side_caption_localized_ink")
        else:
            reason_codes.append("text_area_plan:vertical_side_caption_seed_scope")
        plan.containers.append(
            TextAreaContainer(
                container_id=container_id,
                page_id=page_id,
                container_type=CONTAINER_CAPTION,
                bbox=bbox,
                source_model_ids=[item_id] if item_id else [],
                confidence="deterministic",
                confidence_tier="deterministic_vertical_side_caption_search",
                route_intent=ROUTE_TRANSLATE_CAPTION,
                ocr_eligible=True,
                comic_text_detector_scope_eligible=True,
                fallback_reason="deterministic_vertical_side_caption_search",
                evidence_reason_codes=reason_codes,
                human_review_required=True,
                ocr_eligibility_reason="caption_background_container_strict_ocr_gate",
            )
        )
        if seen is not None:
            seen.add(container_id)
        added_boxes.append(list(bbox))
        added += 1
        if added >= 1:
            break


def _side_caption_signal_items(result: Mapping[str, Any], image_size: Tuple[int, int]) -> List[Mapping[str, Any]]:
    items: List[Mapping[str, Any]] = []
    for evidence in result.get("text_area_model_evidence", []) or []:
        bbox = _bbox_xyxy_to_xywh(evidence.get("bbox_xyxy") or evidence.get("bbox"), image_size)
        if _looks_like_vertical_side_caption_signal_bbox(bbox, image_size):
            items.append(evidence)
    for fused in result.get("fused_containers", []) or []:
        bbox = _bbox_xyxy_to_xywh(fused.get("mask_bbox") or fused.get("bbox"), image_size)
        if _looks_like_vertical_side_caption_signal_bbox(bbox, image_size):
            items.append(fused)
    return items


def _looks_like_vertical_side_caption_signal_bbox(bbox: Sequence[int], image_size: Tuple[int, int]) -> bool:
    if not bbox or len(bbox) < 4:
        return False
    x, y, w, h = [int(v) for v in bbox[:4]]
    width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    if x < width * 0.72:
        return False
    if y < height * 0.18 or y > height * 0.70:
        return False
    if h < height * 0.12:
        return False
    if w < 1 or w > width * 0.24:
        return False
    return True


def _side_caption_search_bbox(seed_bbox: Sequence[int], image_size: Tuple[int, int]) -> List[int]:
    x, y, w, h = _coerce_xywh(seed_bbox)
    width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    x0 = max(int(width * 0.72), x - max(72, int(max(1, w) * 0.55)))
    x1 = min(width - 1, max(x + max(1, w), int(width * 0.97)))
    y0 = max(int(height * 0.20), y - max(180, int(max(1, h) * 1.35)))
    y1 = min(int(height * 0.78), y + max(1, h) + max(90, int(max(1, h) * 0.28)))
    if x1 <= x0:
        x1 = min(width, x0 + max(80, int(width * 0.14)))
    if y1 <= y0:
        y1 = min(height, y0 + max(220, int(height * 0.25)))
    return _normalize_xywh([x0, y0, x1 - x0, y1 - y0], image_size)


def _localize_side_caption_bbox(luma_image: Any, search_bbox: Sequence[int], image_size: Tuple[int, int]) -> List[int] | None:
    if luma_image is None:
        return None
    x, y, w, h = _coerce_xywh(search_bbox)
    width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    if w <= 0 or h <= 0:
        return None
    try:
        crop = luma_image.crop((x, y, min(width, x + w), min(height, y + h))).convert("L")
        cw, ch = crop.size
        if cw <= 0 or ch <= 0:
            return None
        pixels = crop.load()
        col_counts: List[int] = []
        for cx in range(cw):
            count = 0
            for cy in range(ch):
                if pixels[cx, cy] <= 105:
                    count += 1
            col_counts.append(count)
        threshold = max(4, int(ch * 0.018))
        active_cols = [idx for idx, count in enumerate(col_counts) if count >= threshold]
        if not active_cols:
            return None
        # Group nearby dark columns; side captions are narrow but may have several
        # vertical text columns. Ignore very wide SFX/art bands.
        groups: List[Tuple[int, int]] = []
        start = prev = active_cols[0]
        max_gap = max(8, int(cw * 0.055))
        for idx in active_cols[1:]:
            if idx - prev <= max_gap:
                prev = idx
                continue
            groups.append((start, prev))
            start = prev = idx
        groups.append((start, prev))
        valid: List[Tuple[int, int, int]] = []
        for gx0, gx1 in groups:
            gw = gx1 - gx0 + 1
            if gw < 12 or gw > max(150, int(cw * 0.72)):
                continue
            band_dark = sum(col_counts[gx0 : gx1 + 1])
            if band_dark < max(80, int(gw * ch * 0.018)):
                continue
            valid.append((band_dark, gx0, gx1))
        if not valid:
            return None
        _score, gx0, gx1 = max(valid, key=lambda item: item[0])
        pad_x = max(10, int((gx1 - gx0 + 1) * 0.22))
        lx0 = max(0, gx0 - pad_x)
        lx1 = min(cw, gx1 + 1 + pad_x)
        row_counts: List[int] = []
        for cy in range(ch):
            count = 0
            for cx in range(lx0, lx1):
                if pixels[cx, cy] <= 118:
                    count += 1
            row_counts.append(count)
        row_threshold = max(2, int((lx1 - lx0) * 0.045))
        active_rows = [idx for idx, count in enumerate(row_counts) if count >= row_threshold]
        if not active_rows:
            return None
        ly0 = max(0, min(active_rows) - max(12, int(ch * 0.03)))
        ly1 = min(ch, max(active_rows) + 1 + max(12, int(ch * 0.03)))
        localized = [x + lx0, y + ly0, max(1, lx1 - lx0), max(1, ly1 - ly0)]
        if localized[2] < 28 or localized[3] < 160:
            return None
        return _normalize_xywh(localized, image_size)
    except Exception:
        return None


def _has_top_band_caption_search_signal(result: Mapping[str, Any], image_size: Tuple[int, int]) -> bool:
    for item in _top_band_model_text_items(result, image_size):
        class_name = str(item.get("class_name") or item.get("fused_container_type") or "")
        if class_name in {"text_free", "free_text", "caption_or_background_candidate"}:
            return True
    return False


def _top_band_caption_search_source_ids(result: Mapping[str, Any], image_size: Tuple[int, int]) -> List[str]:
    ids: List[str] = []
    for item in _top_band_model_text_items(result, image_size):
        item_id = str(item.get("model_evidence_id") or item.get("evidence_id") or item.get("fused_container_id") or "")
        if item_id and item_id not in ids:
            ids.append(item_id)
    return ids


def _top_band_model_text_items(result: Mapping[str, Any], image_size: Tuple[int, int]) -> List[Mapping[str, Any]]:
    items: List[Mapping[str, Any]] = []
    for evidence in result.get("text_area_model_evidence", []) or []:
        bbox = _bbox_xyxy_to_xywh(evidence.get("bbox_xyxy") or evidence.get("bbox"), image_size)
        if _looks_like_top_caption_bbox(bbox, image_size):
            items.append(evidence)
    for fused in result.get("fused_containers", []) or []:
        bbox = _bbox_xyxy_to_xywh(fused.get("mask_bbox") or fused.get("bbox"), image_size)
        if _looks_like_top_caption_bbox(bbox, image_size):
            items.append(fused)
    return items


def _caption_search_overlaps_blocking_container(bbox: Sequence[Any], containers: Sequence[TextAreaContainer]) -> bool:
    candidate = _xywh_to_xyxy(bbox)
    if not candidate:
        return True
    candidate_area = max(1.0, _area(candidate))
    for container in containers:
        if container.container_type not in {CONTAINER_SPEECH, CONTAINER_SFX}:
            continue
        cbox = _xywh_to_xyxy(container.bbox)
        if not cbox:
            continue
        if _intersection_area(candidate, cbox) / candidate_area >= 0.18:
            return True
    return False


def _intersection_ratio_xywh(a: Sequence[Any], b: Sequence[Any]) -> float:
    abox = _xywh_to_xyxy(a)
    bbox = _xywh_to_xyxy(b)
    if not abox or not bbox:
        return 0.0
    inter = _intersection_area(abox, bbox)
    if inter <= 0:
        return 0.0
    return inter / max(1.0, min(_area(abox), _area(bbox)))


def _inside_ratio_xywh(inner: Sequence[Any], outer: Sequence[Any]) -> float:
    ibox = _xywh_to_xyxy(inner)
    obox = _xywh_to_xyxy(outer)
    if not ibox or not obox:
        return 0.0
    return _intersection_area(ibox, obox) / max(1.0, _area(ibox))


def _full_page_fallback_container(page_id: str, image_size: Tuple[int, int], reason: str) -> TextAreaContainer:
    width, height = int(image_size[0]), int(image_size[1])
    return TextAreaContainer(
        container_id="fallback_full_page",
        page_id=page_id,
        container_type=CONTAINER_UNKNOWN,
        bbox=[0, 0, max(1, width), max(1, height)],
        confidence="fallback",
        confidence_tier="compatibility_fallback",
        route_intent=ROUTE_REVIEW_FALLBACK,
        ocr_eligible=True,
        comic_text_detector_scope_eligible=True,
        fallback_reason=reason,
        evidence_reason_codes=["text_area_plan:full_page_compatibility_fallback"],
        human_review_required=True,
        ocr_eligibility_reason="full_page_compatibility_fallback",
        text_area_pre_ocr_authority=False,
    )


def _blocked_full_page_review_container(page_id: str, image_size: Tuple[int, int], reason: str) -> TextAreaContainer:
    width, height = int(image_size[0]), int(image_size[1])
    return TextAreaContainer(
        container_id="blocked_full_page_review",
        page_id=page_id,
        container_type=CONTAINER_UNKNOWN,
        bbox=[0, 0, max(1, width), max(1, height)],
        confidence="fallback",
        confidence_tier="compatibility_fallback",
        route_intent=ROUTE_REVIEW_FALLBACK,
        ocr_eligible=False,
        comic_text_detector_scope_eligible=False,
        fallback_reason=reason,
        evidence_reason_codes=[
            "text_area_plan:no_usable_text_area_containers",
            "text_area_plan:blocked_no_full_page_ctd",
        ],
        human_review_required=True,
        ocr_eligibility_reason="blocked_no_usable_text_area_containers",
        text_area_pre_ocr_authority=True,
    )


def _fallback_assignment(reason: str) -> Dict[str, Any]:
    return {
        "text_area_container_id": None,
        "text_area_container_type": CONTAINER_UNKNOWN,
        "text_area_route_intent": ROUTE_REVIEW_FALLBACK,
        "text_area_cleanup_authorization": AUTH_REVIEW_UNKNOWN_NOT_CLEANUP,
        "text_area_must_not_mutate": True,
        "text_area_protection_reason": "compatibility_fallback_not_cleanup_authorized",
        "text_area_authorization_source_stage": "compatibility_fallback",
        "text_area_ocr_eligible": True,
        "text_area_detection_source": DETECTION_COMPATIBILITY_FALLBACK,
        "text_area_fallback_reason": reason,
        "text_area_confidence_tier": "compatibility_fallback",
        "text_area_reason_codes": ["text_area_plan:compatibility_full_page_fallback"],
        "text_area_conflict_flags": [],
        "text_area_pre_ocr_authority": False,
        "text_area_enriched_from_region": False,
        "text_area_ocr_eligibility_reason": "compatibility_fallback_requires_legacy_ocr",
        "text_area_overlap_ratio": 0.0,
    }


def _result_to_mapping(result: Any) -> Dict[str, Any]:
    if result is None:
        return {}
    if isinstance(result, Mapping):
        return dict(result)
    if hasattr(result, "to_dict"):
        try:
            return dict(result.to_dict())
        except Exception:
            return {}
    return {}


def _plan_containers(plan: TextAreaPlan | Mapping[str, Any] | None) -> List[Mapping[str, Any]]:
    if plan is None:
        return []
    if isinstance(plan, TextAreaPlan):
        return [container.to_dict() for container in plan.containers]
    return list(plan.get("containers") or [])


def _confidence_tier_from_fused(fused: Mapping[str, Any]) -> str:
    confidence = str(fused.get("confidence") or "")
    has_kitsumed = bool(fused.get("linked_kitsumed_mask_ids"))
    has_ogkalu = bool(fused.get("linked_ogkalu_detection_ids"))
    if confidence == "high" and has_kitsumed and has_ogkalu:
        return "strong_model_container"
    if has_kitsumed:
        return "mask_primary_container"
    if has_ogkalu:
        return "text_bubble_review_container"
    return confidence or "low"


def _has_preserve_conflict(conflicts: Sequence[Any], reasons: Sequence[Any]) -> bool:
    text = " ".join(str(item).lower() for item in list(conflicts) + list(reasons))
    return any(token in text for token in ("sfx", "decorative", "preserve"))


def _evidence_is_linked_to_fused_container(evidence_id: str, result: Mapping[str, Any]) -> bool:
    for fused in result.get("fused_containers", []) or []:
        if evidence_id in [str(item) for item in fused.get("linked_ogkalu_detection_ids", []) or []]:
            return True
    return False


def _looks_like_top_caption_bbox(bbox: Sequence[int], image_size: Tuple[int, int]) -> bool:
    if not bbox or len(bbox) < 4:
        return False
    _x, y, w, h = [int(v) for v in bbox[:4]]
    width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    return y <= height * 0.24 and h >= height * 0.06 and w <= width * 0.32


def _looks_like_caption_background_bbox(bbox: Sequence[int], image_size: Tuple[int, int]) -> bool:
    if not bbox or len(bbox) < 4:
        return False
    x, y, w, h = [int(v) for v in bbox[:4]]
    width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    if _is_clipped_or_degenerate_bbox(bbox, image_size):
        return False
    top_band = y <= height * 0.24
    narrow_text_column = width * 0.025 <= w <= width * 0.18
    caption_height = height * 0.04 <= h <= height * 0.16
    # Cover/title/logo art can be detected as text_free, but those tall,
    # page-edge columns are not safe caption/background OCR scopes.
    return top_band and narrow_text_column and caption_height and x < width * 0.96


def _load_luma_image(image_path: str | Path) -> Any:
    try:
        from PIL import Image

        path = Path(image_path)
        if not path.exists():
            return None
        return Image.open(path).convert("L")
    except Exception:
        return None


def _container_visual_stats(luma_image: Any, bbox: Sequence[Any], image_size: Tuple[int, int]) -> Dict[str, Any]:
    x, y, w, h = _coerce_xywh(bbox)
    width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    area = max(0, w) * max(0, h)
    stats: Dict[str, Any] = {
        "area_px": area,
        "area_ratio": area / max(1, width * height),
        "mean_luma": None,
        "bright_ratio": None,
        "dark_ratio": None,
        "clipped_or_degenerate": _is_clipped_or_degenerate_bbox(bbox, image_size),
    }
    if luma_image is None or area <= 0:
        return stats
    try:
        crop = luma_image.crop((max(0, x), max(0, y), min(width, x + max(0, w)), min(height, y + max(0, h))))
        pixels = list(crop.getdata())
        if not pixels:
            return stats
        total = float(len(pixels))
        stats["mean_luma"] = round(sum(pixels) / total, 4)
        stats["bright_ratio"] = round(sum(1 for value in pixels if value >= 220) / total, 6)
        stats["dark_ratio"] = round(sum(1 for value in pixels if value <= 90) / total, 6)
    except Exception:
        pass
    return stats


def _looks_like_pre_ocr_sfx_or_decorative(
    fused_type: str,
    reasons: Sequence[Any],
    bbox: Sequence[Any],
    visual: Mapping[str, Any],
    image_size: Tuple[int, int],
) -> bool:
    if _is_clipped_or_degenerate_bbox(bbox, image_size):
        return False
    reason_text = " ".join(str(item).lower() for item in reasons)
    if any(token in reason_text for token in ("sfx", "decorative", "preserve")):
        return True
    if fused_type != "free_text":
        return False
    if _looks_like_caption_background_bbox(bbox, image_size):
        return False
    bright = _optional_float(visual.get("bright_ratio"))
    dark = _optional_float(visual.get("dark_ratio"))
    area_ratio = float(visual.get("area_ratio") or 0.0)
    # Non-caption text_free is not speech evidence. Dark/non-uniform regions are
    # treated as art/SFX review and blocked before normal OCR/translation.
    if dark is not None and dark >= 0.16:
        return True
    if bright is not None and bright < 0.62:
        return True
    if area_ratio < 0.003:
        return True
    return False


def _safe_unknown_ocr_fallback(
    fused_type: str,
    reasons: Sequence[Any],
    bbox: Sequence[Any],
    visual: Mapping[str, Any],
    image_size: Tuple[int, int],
) -> bool:
    if _is_clipped_or_degenerate_bbox(bbox, image_size):
        return False
    reason_text = " ".join(str(item).lower() for item in reasons)
    if "text_free" in reason_text or fused_type == "free_text":
        return False
    if not any(token in reason_text for token in ("ogkalu_bubble_without_kitsumed_mask", "ogkalu_text_bubble_without_kitsumed_mask")):
        return False
    bright = _optional_float(visual.get("bright_ratio"))
    dark = _optional_float(visual.get("dark_ratio"))
    area_ratio = float(visual.get("area_ratio") or 0.0)
    area = float(visual.get("area_px") or 0.0)
    if bright is None or dark is None:
        return False
    return area >= 80000 and area_ratio >= 0.035 and bright >= 0.78 and dark <= 0.14


def _mask_primary_speech_activation_completeness_allowed(
    *,
    confidence: str,
    bbox: Sequence[Any],
    reasons: Sequence[Any],
    visual: Mapping[str, Any],
    image_size: Tuple[int, int],
    clipped: bool,
) -> bool:
    if clipped or confidence != "medium":
        return False
    if _looks_like_pre_ocr_sfx_or_decorative(CONTAINER_SPEECH, reasons, bbox, visual, image_size):
        return False
    x, y, w, h = _coerce_xywh(bbox)
    if w <= 0 or h <= 0:
        return False
    area = w * h
    if area < 36_000:
        return False
    if w < 150 or h < 180:
        return False
    aspect = w / max(1, h)
    if aspect < 0.48 or aspect > 1.15:
        return False
    page_w, page_h = max(1, int(image_size[0])), max(1, int(image_size[1]))
    if w > page_w * 0.28 or h > page_h * 0.18:
        return False
    reason_text = " ".join(str(item).lower() for item in reasons)
    if any(token in reason_text for token in ("sfx", "decorative", "art_only", "non_text", "preserve")):
        return False
    return True


def _mask_primary_irregular_speech_activation_allowed(
    *,
    confidence: str,
    bbox: Sequence[Any],
    reasons: Sequence[Any],
    visual: Mapping[str, Any],
    image_size: Tuple[int, int],
    clipped: bool,
) -> bool:
    if clipped or confidence != "medium":
        return False
    if _looks_like_pre_ocr_sfx_or_decorative(CONTAINER_SPEECH, reasons, bbox, visual, image_size):
        return False
    x, y, w, h = _coerce_xywh(bbox)
    if w <= 0 or h <= 0:
        return False
    area = w * h
    if area < 18_000:
        return False
    if w < 80 or h < 120:
        return False
    aspect = w / max(1, h)
    if aspect < 0.35 or aspect > 1.25:
        return False
    page_w, page_h = max(1, int(image_size[0])), max(1, int(image_size[1]))
    if w > page_w * 0.24 or h > page_h * 0.20:
        return False
    bright = _optional_float(visual.get("bright_ratio"))
    dark = _optional_float(visual.get("dark_ratio"))
    if dark is not None and dark >= 0.24:
        return False
    if bright is not None and bright < 0.58:
        return False
    reason_text = " ".join(str(item).lower() for item in reasons)
    if any(token in reason_text for token in ("sfx", "decorative", "art_only", "non_text", "preserve")):
        return False
    return True


def _visual_reason_codes(visual: Mapping[str, Any]) -> List[str]:
    codes: List[str] = []
    if visual.get("clipped_or_degenerate"):
        codes.append("text_area_plan:clipped_or_degenerate_container")
    bright = _optional_float(visual.get("bright_ratio"))
    dark = _optional_float(visual.get("dark_ratio"))
    if bright is not None and bright >= 0.78:
        codes.append("text_area_plan:bright_container_context")
    if dark is not None and dark >= 0.16:
        codes.append("text_area_plan:dark_or_art_context")
    return codes


def _is_clipped_or_degenerate_bbox(bbox: Sequence[Any], image_size: Tuple[int, int]) -> bool:
    x, y, w, h = _coerce_xywh(bbox)
    width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    if w <= 2 or h <= 2:
        return True
    return x <= -2 or y <= -2 or x + w >= width + 2 or y + h >= height + 2


def _coerce_xywh(value: Sequence[Any]) -> Tuple[int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return 0, 0, 0, 0
    try:
        x, y, w, h = [int(round(float(v or 0))) for v in value[:4]]
        return x, y, w, h
    except Exception:
        return 0, 0, 0, 0


def _optional_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _bbox_xyxy_to_xywh(value: Any, image_size: Tuple[int, int]) -> List[int]:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return [0, 0, max(1, int(image_size[0])), max(1, int(image_size[1]))]
    x0, y0, x1, y1 = [float(v) for v in value[:4]]
    if x1 < x0 or y1 < y0:
        return [int(round(x0)), int(round(y0)), max(1, int(round(x1))), max(1, int(round(y1)))]
    width, height = int(image_size[0]), int(image_size[1])
    x0 = max(0, min(width - 1, int(math.floor(x0))))
    y0 = max(0, min(height - 1, int(math.floor(y0))))
    x1 = max(x0 + 1, min(width, int(math.ceil(x1))))
    y1 = max(y0 + 1, min(height, int(math.ceil(y1))))
    return [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]


def _normalize_xywh(value: Any, image_size: Tuple[int, int]) -> List[int]:
    x, y, w, h = _coerce_xywh(value)
    width, height = max(1, int(image_size[0])), max(1, int(image_size[1]))
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    w = max(1, min(width - x, w))
    h = max(1, min(height - y, h))
    return [x, y, w, h]


def _bbox_area_xywh(bbox: Sequence[Any]) -> int:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return 0
    try:
        return max(0, int(bbox[2])) * max(0, int(bbox[3]))
    except Exception:
        return 0


def _xywh_to_xyxy(value: Any) -> Optional[Tuple[float, float, float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x, y, w, h = [float(v) for v in value[:4]]
    except Exception:
        return None
    return (x, y, x + max(0.0, w), y + max(0.0, h))


def _area(box: Tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection_area(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _center_inside(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float] | None) -> bool:
    if not b:
        return False
    cx = (a[0] + a[2]) / 2.0
    cy = (a[1] + a[3]) / 2.0
    return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]


def _container_priority(container: Mapping[str, Any]) -> int:
    ctype = container.get("container_type")
    if ctype == CONTAINER_SFX:
        return 4
    if ctype == CONTAINER_SPEECH:
        return 3
    if ctype == CONTAINER_CAPTION:
        return 2
    return 1


def _safe_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _write_plan_summary(
    path: Path,
    plan: Mapping[str, Any],
    detections: Sequence[Mapping[str, Any]],
    ocr_candidates: Sequence[Mapping[str, Any]],
    fallbacks: Sequence[Mapping[str, Any]],
    blocked: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# TextAreaPlan Summary",
        "",
        f"- version: `{plan.get('version')}`",
        f"- generated: `{plan.get('generated')}`",
        f"- containers: `{len(plan.get('containers') or [])}`",
        f"- scopes: `{len(plan.get('scopes') or [])}`",
        f"- scoped detection candidates: `{len(detections)}`",
        f"- scoped OCR candidates: `{len(ocr_candidates)}`",
        f"- fallback decisions: `{len(fallbacks)}`",
        f"- blocked text-area candidates: `{len(blocked)}`",
        f"- summary: `{json.dumps(plan.get('summary') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
        "| container_id | type | intent | ocr | tier | fallback | conflicts |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for container in plan.get("containers", []) or []:
        lines.append(
            "| {container_id} | {container_type} | {route_intent} | {ocr_eligible} | {confidence_tier} | {fallback_reason} | {conflicts} |".format(
                container_id=container.get("container_id"),
                container_type=container.get("container_type"),
                route_intent=container.get("route_intent"),
                ocr_eligible=container.get("ocr_eligible"),
                confidence_tier=container.get("confidence_tier"),
                fallback_reason=container.get("fallback_reason") or "",
                conflicts=",".join(str(item) for item in container.get("conflict_flags") or []),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _blocked_containers_from_plan(plan: Mapping[str, Any]) -> List[Dict[str, Any]]:
    blocked: List[Dict[str, Any]] = []
    for container in plan.get("containers", []) or []:
        if bool(container.get("ocr_eligible")) and bool(container.get("comic_text_detector_scope_eligible")):
            continue
        blocked.append(
            {
                "candidate_id": f"blocked_{container.get('container_id')}",
                "container_id": container.get("container_id"),
                "container_type": container.get("container_type"),
                "route_intent": container.get("route_intent"),
                "bbox": container.get("bbox"),
                "ocr_eligible": bool(container.get("ocr_eligible")),
                "comic_text_detector_scope_eligible": bool(container.get("comic_text_detector_scope_eligible")),
                "fallback_reason": container.get("fallback_reason"),
                "ocr_eligibility_reason": container.get("ocr_eligibility_reason"),
                "reason_codes": container.get("evidence_reason_codes") or [],
                "conflict_flags": container.get("conflict_flags") or [],
                "would_change_behavior": False,
            }
        )
    return blocked


def _caption_localization_candidates_from_plan(plan: Mapping[str, Any]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for container in plan.get("containers", []) or []:
        if container.get("container_type") != CONTAINER_CAPTION:
            continue
        candidates.append(
            {
                "candidate_id": f"caption_{container.get('container_id')}",
                "container_id": container.get("container_id"),
                "bbox": container.get("bbox"),
                "route_intent": container.get("route_intent"),
                "ocr_eligible": bool(container.get("ocr_eligible")),
                "comic_text_detector_scope_eligible": bool(container.get("comic_text_detector_scope_eligible")),
                "confidence_tier": container.get("confidence_tier"),
                "fallback_reason": container.get("fallback_reason"),
                "source_model_ids": container.get("source_model_ids") or [],
                "reason_codes": container.get("evidence_reason_codes") or [],
                "conflict_flags": container.get("conflict_flags") or [],
                "would_change_behavior": False,
            }
        )
    return candidates


def _write_plan_overlay(
    path: Path,
    image_path: str | Path,
    plan: Mapping[str, Any],
    detections: Sequence[Mapping[str, Any]],
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return
    image_path = Path(image_path)
    if not image_path.exists():
        return
    base = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(base)
    font = _overlay_font(ImageFont)
    colors = {
        CONTAINER_SPEECH: (40, 120, 255),
        CONTAINER_CAPTION: (0, 170, 120),
        CONTAINER_SFX: (220, 0, 180),
        CONTAINER_UNKNOWN: (245, 150, 0),
    }
    for container in plan.get("containers", []) or []:
        box = _xywh_to_xyxy(container.get("bbox") or [])
        if not box:
            continue
        color = colors.get(container.get("container_type"), (180, 180, 180))
        xyxy = tuple(int(round(v)) for v in box)
        draw.rectangle(xyxy, outline=color, width=4)
        _draw_label(draw, font, (xyxy[0] + 2, max(0, xyxy[1] - 18)), f"{container.get('container_id')} {container.get('container_type')}", color)
    for candidate in detections:
        box = _xywh_to_xyxy(candidate.get("bbox") or [])
        if not box:
            continue
        xyxy = tuple(int(round(v)) for v in box)
        source = candidate.get("detection_source")
        color = (255, 0, 0) if source == DETECTION_BLOCKED else (0, 0, 0)
        draw.rectangle(xyxy, outline=color, width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    base.save(path, quality=92)


def _overlay_font(image_font_module: Any) -> Any:
    for candidate in (
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msgothic.ttc",
    ):
        try:
            if os.path.isfile(candidate):
                return image_font_module.truetype(candidate, 14)
        except Exception:
            pass
    try:
        return image_font_module.load_default()
    except Exception:
        return None


def _draw_label(draw: Any, font: Any, xy: Tuple[int, int], text: str, fill: Tuple[int, int, int]) -> None:
    x, y = xy
    try:
        box = draw.textbbox((x, y), text, font=font)
    except Exception:
        box = (x, y, x + max(1, len(text)) * 8, y + 16)
    draw.rectangle(box, fill=(255, 255, 255))
    draw.text((x, y), text, fill=fill, font=font)
