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
