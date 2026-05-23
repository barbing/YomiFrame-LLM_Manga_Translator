# -*- coding: utf-8 -*-
"""Diagnostic-only container-aware render planner.

Phase 3a consumes existing text-area diagnostics and renderer metadata, then
emits layout suggestions for review. It must not mutate regions, render boxes,
font sizes, wrapping, cleanup masks, or output images.
"""
from __future__ import annotations

import time
from collections import Counter
from typing import Any


RENDER_PLANNER_VERSION = "text_area_render_planner_phase3a_v1"
PHASE3_STATUS = "diagnostic_only"

_SEVERITY_RANK = {"watch": 0, "minor": 1, "serious": 2, "blocker": 3}


def enrich_audit_with_render_planner(audit: dict[str, Any]) -> dict[str, Any]:
    """Attach diagnostic render-plan suggestions to a debug audit."""
    start = time.time()
    enriched = dict(audit)
    enriched["regions"] = [dict(region) for region in audit.get("regions", []) or []]
    enriched["render_planner_version"] = RENDER_PLANNER_VERSION
    try:
        if not enriched.get("diagnostic_generated"):
            raise ValueError("text-area diagnostics unavailable")
        suggestions = build_render_plan_suggestions(enriched)
        _attach_render_plan_suggestions_to_regions(enriched, suggestions)
        enriched["render_plan_suggestions"] = suggestions
        enriched["render_planner_generated"] = True
        enriched["render_planner_error"] = None
    except Exception as exc:  # pragma: no cover - debug isolation
        enriched["render_plan_suggestions"] = []
        enriched["render_planner_generated"] = False
        enriched["render_planner_error"] = str(exc)
        for region in enriched.get("regions", []) or []:
            region["diagnostic_render_plan_suggestions"] = []
            region["diagnostic_render_plan_severity"] = None
            region["diagnostic_render_plan_reason_codes"] = []
    enriched["render_planner_runtime_sec"] = round(time.time() - start, 6)
    return enriched


def build_render_plan_suggestions(audit: dict[str, Any]) -> list[dict[str, Any]]:
    """Build diagnostic-only render/layout suggestions from an enriched audit."""
    regions = [dict(region) for region in audit.get("regions", []) or []]
    regions_by_id = {str(region.get("region_id") or ""): region for region in regions}
    constraints_by_rid = {
        str(item.get("region_id") or ""): item
        for item in audit.get("render_constraints", []) or []
    }
    ownership_by_rid = {
        str(item.get("region_id") or ""): item
        for item in audit.get("text_ownership", []) or []
    }
    containers_by_id = {
        str(item.get("container_id") or ""): item
        for item in audit.get("text_containers", []) or []
    }

    suggestions: list[dict[str, Any]] = []
    for region in regions:
        rid = str(region.get("region_id") or "")
        if not rid:
            continue
        constraint = constraints_by_rid.get(rid) or {}
        ownership = ownership_by_rid.get(rid) or {}
        container = containers_by_id.get(str(ownership.get("container_id") or constraint.get("container_id") or "")) or {}
        for suggestion in _region_render_suggestions(region, constraint, ownership, container):
            suggestion["suggestion_id"] = f"render_plan_{len(suggestions):03d}"
            suggestions.append(suggestion)
    return suggestions


def summarize_render_plan_suggestions(suggestions: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, int] = {}
    by_severity: dict[str, int] = {}
    by_confidence: dict[str, int] = {}
    for item in suggestions:
        by_type[str(item.get("suggestion_type") or "unknown")] = by_type.get(str(item.get("suggestion_type") or "unknown"), 0) + 1
        by_severity[str(item.get("severity") or "unknown")] = by_severity.get(str(item.get("severity") or "unknown"), 0) + 1
        by_confidence[str(item.get("confidence") or "unknown")] = by_confidence.get(str(item.get("confidence") or "unknown"), 0) + 1
    return {
        "total": len(suggestions),
        "by_type": by_type,
        "by_severity": by_severity,
        "by_confidence": by_confidence,
    }


def _region_render_suggestions(
    region: dict[str, Any],
    constraint: dict[str, Any],
    ownership: dict[str, Any],
    container: dict[str, Any],
) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []
    outside = _speech_render_outside_container(region, constraint, ownership, container)
    if outside:
        suggestions.append(outside)
    obstacle = _speech_render_over_preserved_obstacle(region, constraint, ownership, container)
    if obstacle:
        suggestions.append(obstacle)
    fit = _translated_text_fit_risk(region, constraint, ownership, container)
    if fit:
        suggestions.append(fit)
    completeness = _rendered_text_completeness_risk(region, constraint, ownership, container)
    if completeness:
        suggestions.append(completeness)
    uncertain = _render_constraint_uncertain(region, constraint, ownership, container, has_specific=bool(suggestions))
    if uncertain:
        suggestions.append(uncertain)
    return suggestions


def _speech_render_outside_container(
    region: dict[str, Any],
    constraint: dict[str, Any],
    ownership: dict[str, Any],
    container: dict[str, Any],
) -> dict[str, Any] | None:
    if _semantic_class(region) != "speech_bubble":
        return None
    if container.get("container_type") != "speech_bubble":
        return None
    if not _ownership_strong(ownership):
        return None
    reasons = set(str(item) for item in (constraint.get("reason_codes") or []))
    outside = float(constraint.get("outside_allowed_ratio", 0.0) or 0.0)
    if "final_render_bbox_outside_inferred_container" not in reasons or outside < 0.20:
        return None
    reason_codes = _base_reason_codes(region, constraint, ownership, container)
    reason_codes.append("final_render_bbox_outside_inferred_container")
    reason_codes.append(f"outside_allowed_ratio:{outside:.3f}")
    translated_body_len = len(_meaningful_chars(str(region.get("translated_text") or "")))
    reason_codes.append(f"translated_body_length:{translated_body_len}")
    if outside >= 0.45 and translated_body_len >= 8:
        severity = "serious"
        confidence = "high"
        action = "clamp_to_container"
    elif outside >= 0.30:
        severity = "minor"
        confidence = "medium"
        action = "prefer_source_local_box"
    else:
        severity = "watch"
        confidence = "low"
        action = "review_only_no_action"
    return _base_render_suggestion(
        region,
        constraint,
        ownership,
        container,
        suggestion_type="speech_render_outside_container",
        severity=severity,
        confidence=confidence,
        reason_codes=reason_codes,
        proposed_action=action,
    )


def _speech_render_over_preserved_obstacle(
    region: dict[str, Any],
    constraint: dict[str, Any],
    ownership: dict[str, Any],
    container: dict[str, Any],
) -> dict[str, Any] | None:
    if _semantic_class(region) != "speech_bubble":
        return None
    reasons = set(str(item) for item in (constraint.get("reason_codes") or []))
    obstacle_ids = [str(item) for item in (constraint.get("newly_introduced_obstacle_container_ids") or []) if str(item)]
    if "final_render_bbox_newly_overlaps_preserved_obstacle" not in reasons or not obstacle_ids:
        return None
    reason_codes = _base_reason_codes(region, constraint, ownership, container)
    reason_codes.extend(["final_render_bbox_newly_overlaps_preserved_obstacle", "preserved_obstacle_overlap"])
    return _base_render_suggestion(
        region,
        constraint,
        ownership,
        container,
        suggestion_type="speech_render_over_preserved_obstacle",
        severity="serious",
        confidence="high" if _ownership_strong(ownership) else "medium",
        reason_codes=reason_codes,
        proposed_action="avoid_preserved_obstacle",
    )


def _translated_text_fit_risk(
    region: dict[str, Any],
    constraint: dict[str, Any],
    ownership: dict[str, Any],
    container: dict[str, Any],
) -> dict[str, Any] | None:
    if not str(region.get("translated_text") or "").strip():
        return None
    final_box = _xyxy(region.get("final_render_bbox") or constraint.get("current_final_render_bbox"))
    if final_box is None:
        return None
    width = max(1, final_box[2] - final_box[0])
    height = max(1, final_box[3] - final_box[1])
    measured_w = _float_or_none(region.get("measured_rendered_width"))
    measured_h = _float_or_none(region.get("measured_rendered_height"))
    fit_ratio = _float_or_none(region.get("fit_ratio"))
    selected_size = _float_or_none(region.get("selected_font_size"))
    source_hint = _float_or_none(region.get("source_size_hint"))
    reasons = []
    if fit_ratio is not None and fit_ratio >= 0.95:
        reasons.append(f"fit_ratio_high:{fit_ratio:.3f}")
    if measured_w is not None and measured_w / width >= 0.94:
        reasons.append(f"width_occupancy_high:{measured_w / width:.3f}")
    if measured_h is not None and measured_h / height >= 0.94:
        reasons.append(f"height_occupancy_high:{measured_h / height:.3f}")
    if selected_size is not None and source_hint is not None and source_hint > 0 and selected_size / source_hint <= 0.65:
        reasons.append(f"font_size_shrink:{selected_size / source_hint:.3f}")
    if not reasons:
        return None
    severity = "minor" if any(reason.startswith("fit_ratio_high:1.") for reason in reasons) else "watch"
    confidence = "medium" if severity == "minor" else "low"
    reason_codes = _base_reason_codes(region, constraint, ownership, container)
    reason_codes.extend(reasons)
    return _base_render_suggestion(
        region,
        constraint,
        ownership,
        container,
        suggestion_type="translated_text_fit_risk",
        severity=severity,
        confidence=confidence,
        reason_codes=reason_codes,
        proposed_action="require_smaller_font_or_more_columns",
    )


def _rendered_text_completeness_risk(
    region: dict[str, Any],
    constraint: dict[str, Any],
    ownership: dict[str, Any],
    container: dict[str, Any],
) -> dict[str, Any] | None:
    translated = str(region.get("translated_text") or "")
    if not translated.strip():
        return None
    wrapped = region.get("wrapped_lines")
    if not wrapped:
        return None
    if isinstance(wrapped, str):
        wrapped_parts = [wrapped]
    else:
        wrapped_parts = [str(item) for item in wrapped if str(item)]
    if wrapped_parts == ["ellipsis"]:
        return None
    translated_chars = Counter(_meaningful_chars(translated))
    rendered_chars = Counter(_meaningful_chars("".join(wrapped_parts)))
    missing = []
    for ch, count in translated_chars.items():
        if rendered_chars[ch] < count:
            missing.append(ch)
    if not missing:
        return None
    reason_codes = _base_reason_codes(region, constraint, ownership, container)
    reason_codes.append("wrapped_lines_omit_translated_characters")
    reason_codes.append(f"missing_char_count:{sum((translated_chars - rendered_chars).values())}")
    return _base_render_suggestion(
        region,
        constraint,
        ownership,
        container,
        suggestion_type="rendered_text_completeness_risk",
        severity="serious",
        confidence="high",
        reason_codes=reason_codes,
        proposed_action="report_unfit_without_truncation",
    )


def _render_constraint_uncertain(
    region: dict[str, Any],
    constraint: dict[str, Any],
    ownership: dict[str, Any],
    container: dict[str, Any],
    *,
    has_specific: bool,
) -> dict[str, Any] | None:
    if has_specific:
        return None
    status = str(constraint.get("constraint_status") or "")
    if status in {"", "satisfied"}:
        return None
    reason_codes = _base_reason_codes(region, constraint, ownership, container)
    reason_codes.append(f"constraint_status:{status}")
    return _base_render_suggestion(
        region,
        constraint,
        ownership,
        container,
        suggestion_type="render_constraint_uncertain_review_only",
        severity="watch",
        confidence="low",
        reason_codes=reason_codes,
        proposed_action="review_only_no_action",
    )


def _base_render_suggestion(
    region: dict[str, Any],
    constraint: dict[str, Any],
    ownership: dict[str, Any],
    container: dict[str, Any],
    *,
    suggestion_type: str,
    severity: str,
    confidence: str,
    reason_codes: list[str],
    proposed_action: str,
) -> dict[str, Any]:
    allowed = constraint.get("allowed_bbox")
    obstacles = constraint.get("newly_introduced_obstacle_container_ids") or constraint.get("obstacle_container_ids") or []
    model_evidence = _model_backed_render_evidence(region)
    return {
        "region_id": str(region.get("region_id") or ""),
        "current_semantic_class": _semantic_class(region),
        "current_render_bbox": _xyxy_list(region.get("final_render_bbox") or constraint.get("current_final_render_bbox")),
        "source_bbox": region.get("bbox"),
        "inferred_container_id": container.get("container_id") or ownership.get("container_id") or constraint.get("container_id"),
        "inferred_container_bbox": _container_bbox_xyxy(container, constraint),
        "container_type": container.get("container_type"),
        "suggestion_type": suggestion_type,
        "severity": severity,
        "confidence": confidence,
        "reason_codes": sorted(set(str(item) for item in reason_codes if str(item))),
        "would_change_behavior": False,
        "phase3_status": PHASE3_STATUS,
        "human_review_required": True,
        "proposed_allowed_area": _xyxy_list(allowed),
        "proposed_obstacle_ids": [str(item) for item in obstacles],
        "proposed_action": proposed_action,
        "model_backed_container_id": model_evidence.get("container_id"),
        "model_backed_container_type": model_evidence.get("container_type"),
        "model_backed_confidence_tier": model_evidence.get("confidence_tier"),
        "model_backed_membership_type": model_evidence.get("membership_type"),
        "model_backed_supported_actions": model_evidence.get("supported_actions", []),
        "model_backed_container_suggested_actions": model_evidence.get("container_suggested_actions", []),
        "model_backed_blocked_actions": model_evidence.get("blocked_actions", []),
        "model_backed_conflict_flags": model_evidence.get("conflict_flags", []),
        "model_backed_review_only": model_evidence.get("review_only"),
        "model_backed_source_model_ids": model_evidence.get("source_model_ids", []),
        "model_backed_consumer_sources": model_evidence.get("consumer_sources", []),
    }


def _attach_render_plan_suggestions_to_regions(audit: dict[str, Any], suggestions: list[dict[str, Any]]) -> None:
    by_rid: dict[str, list[dict[str, Any]]] = {}
    for item in suggestions:
        by_rid.setdefault(str(item.get("region_id") or ""), []).append(item)
    for region in audit.get("regions", []) or []:
        rid = str(region.get("region_id") or "")
        items = by_rid.get(rid, [])
        region["diagnostic_render_plan_suggestions"] = items
        region["diagnostic_render_plan_severity"] = _max_severity(items)
        reason_codes = []
        for item in items:
            reason_codes.extend(item.get("reason_codes", []) or [])
        region["diagnostic_render_plan_reason_codes"] = sorted(set(str(item) for item in reason_codes))
        if region.get("diagnostic_bubble_confidence_tier") is not None:
            _append_bubble_consumer_source(region, "text_area_render_planner")


def _max_severity(suggestions: list[dict[str, Any]]) -> str | None:
    if not suggestions:
        return None
    return max((str(item.get("severity") or "watch") for item in suggestions), key=lambda value: _SEVERITY_RANK.get(value, -1))


def _model_backed_render_evidence(region: dict[str, Any]) -> dict[str, Any]:
    if region.get("diagnostic_bubble_confidence_tier") is None:
        return {
            "available": False,
            "source": "deterministic_only",
            "would_change_behavior": False,
        }
    return {
        "available": True,
        "source": "bubble_detection_service",
        "container_id": region.get("diagnostic_bubble_container_id"),
        "container_type": region.get("diagnostic_bubble_container_type"),
        "membership_type": region.get("diagnostic_bubble_membership_type"),
        "membership_confidence": region.get("diagnostic_bubble_membership_confidence"),
        "confidence_tier": region.get("diagnostic_bubble_confidence_tier"),
        "supported_actions": _string_list(region.get("diagnostic_bubble_supported_actions")),
        "container_suggested_actions": _string_list(region.get("diagnostic_bubble_container_suggested_actions")),
        "blocked_actions": _string_list(region.get("diagnostic_bubble_blocked_actions")),
        "conflict_flags": _string_list(region.get("diagnostic_bubble_conflict_flags")),
        "source_model_ids": _string_list(region.get("diagnostic_bubble_source_model_ids")),
        "review_only": bool(region.get("diagnostic_bubble_review_only")),
        "consumer_sources": _string_list(region.get("diagnostic_bubble_consumer_sources")),
        "would_change_behavior": False,
    }


def _append_bubble_consumer_source(region: dict[str, Any], source: str) -> None:
    try:
        from app.pipeline.text_area_diagnostics import append_bubble_detection_consumer_source

        append_bubble_detection_consumer_source(region, source)
    except Exception:
        existing = set(_string_list(region.get("diagnostic_bubble_consumer_sources")))
        existing.add(source)
        region["diagnostic_bubble_consumer_sources"] = sorted(existing)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return [str(value)] if str(value) else []


def _base_reason_codes(
    region: dict[str, Any],
    constraint: dict[str, Any],
    ownership: dict[str, Any],
    container: dict[str, Any],
) -> list[str]:
    reasons = []
    rid = str(region.get("region_id") or "")
    if rid:
        reasons.append(f"region:{rid}")
    ctype = str(container.get("container_type") or "")
    if ctype:
        reasons.append(f"container_type:{ctype}")
    relation = str(ownership.get("ownership_type") or "")
    if relation:
        reasons.append(f"ownership:{relation}")
    confidence = ownership.get("confidence")
    if confidence is not None:
        reasons.append(f"ownership_confidence:{float(confidence or 0.0):.3f}")
    if region.get("diagnostic_bubble_confidence_tier") is not None:
        reasons.append("bubble_detection_service_evidence_available")
        reasons.append(f"model_backed_confidence_tier:{region.get('diagnostic_bubble_confidence_tier')}")
        if region.get("diagnostic_bubble_container_id"):
            reasons.append(f"model_backed_container:{region.get('diagnostic_bubble_container_id')}")
        if region.get("diagnostic_bubble_container_type"):
            reasons.append(f"model_backed_container_type:{region.get('diagnostic_bubble_container_type')}")
        for action in _string_list(region.get("diagnostic_bubble_supported_actions")):
            reasons.append(f"model_backed_supported_action:{action}")
        for action in _string_list(region.get("diagnostic_bubble_blocked_actions")):
            reasons.append(f"model_backed_blocked_action:{action}")
        for flag in _string_list(region.get("diagnostic_bubble_conflict_flags")):
            reasons.append(f"model_backed_conflict:{flag}")
    status = str(constraint.get("constraint_status") or "")
    if status:
        reasons.append(f"constraint_status:{status}")
    for reason in ownership.get("reason_codes", []) or []:
        reasons.append(f"ownership_reason:{reason}")
    for reason in constraint.get("reason_codes", []) or []:
        reasons.append(f"constraint_reason:{reason}")
    return reasons


def _semantic_class(region: dict[str, Any]) -> str:
    return str(region.get("semantic_class") or region.get("type") or "unknown")


def _ownership_strong(ownership: dict[str, Any]) -> bool:
    return (
        str(ownership.get("ownership_type") or "") in {"inside", "overlaps"}
        and float(ownership.get("confidence", 0.0) or 0.0) >= 0.70
    )


def _container_bbox_xyxy(container: dict[str, Any], constraint: dict[str, Any]) -> list[int] | None:
    bbox = container.get("bbox")
    if bbox:
        xyxy = _xyxy_from_xywh(bbox)
        return list(xyxy) if xyxy else None
    return _xyxy_list(constraint.get("allowed_bbox"))


def _xyxy_list(value: Any) -> list[int] | None:
    box = _xyxy(value)
    return list(box) if box else None


def _xyxy(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = [int(round(float(v))) for v in value]
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _xyxy_from_xywh(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x, y, w, h = [int(round(float(v))) for v in value]
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return x, y, x + w, y + h


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _meaningful_chars(text: str) -> list[str]:
    chars = []
    for ch in str(text or ""):
        if ch.isspace() or _is_punctuation(ch):
            continue
        chars.append(ch)
    return chars


def _is_punctuation(ch: str) -> bool:
    code = ord(ch)
    return (
        0x2000 <= code <= 0x206F
        or 0x3000 <= code <= 0x303F
        or 0xFF00 <= code <= 0xFF65
        or ch in set(".,!?;:'\"()[]{}<>/\\|-_~`")
    )
