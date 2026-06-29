# -*- coding: utf-8 -*-
"""Parent-keyed execution contract for post-hierarchy pipeline stages.

This module does not build root/parent topology. It converts the finalized
hierarchy view into the downstream execution unit consumed by translation,
cleanup, render eligibility, and rendering.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


PARENT_EXECUTION_BUNDLE_VERSION = "parent_execution_bundle_v1"


@dataclass
class ParentExecutionBundle:
    page_id: str
    bundle_id: str
    root_id: str
    parent_id: str
    graph_parent_id: str
    state: str
    role: str
    source_text: str = ""
    source_quality_state: str = "accepted_for_translation"
    source_quality_action: str = "translate"
    translation_required: bool = False
    cleanup_required: bool = False
    render_required: bool = False
    parent_bbox: list[int] = field(default_factory=list)
    cleanup_target_bbox: list[int] = field(default_factory=list)
    render_allowed_area: list[int] = field(default_factory=list)
    root_bbox: list[int] = field(default_factory=list)
    source_region_ids: list[str] = field(default_factory=list)
    represented_child_ids: list[str] = field(default_factory=list)
    source_candidates: list[dict[str, Any]] = field(default_factory=list)
    semantic_class: str = ""
    route_intent: str = ""
    cleanup_mode: str = ""
    text_area_container_id: str = ""
    text_area_container_type: str = ""
    confidence: float | None = None
    reason_codes: list[str] = field(default_factory=list)
    unresolved_reason: str | None = None
    translated_text: str = ""
    source_glyph_mask_ids: list[str] = field(default_factory=list)
    cleanup_job_ids: list[str] = field(default_factory=list)
    cleanup_mask_ids: list[str] = field(default_factory=list)
    render_decision_id: str = ""
    renderer_audit_id: str = ""
    execution_region: dict[str, Any] = field(default_factory=dict)

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "parent_execution_bundle_version": PARENT_EXECUTION_BUNDLE_VERSION,
            "page_id": self.page_id,
            "bundle_id": self.bundle_id,
            "root_id": self.root_id,
            "parent_id": self.parent_id,
            "graph_parent_id": self.graph_parent_id,
            "state": self.state,
            "role": self.role,
            "source_text": self.source_text,
            "source_quality_state": self.source_quality_state,
            "source_quality_action": self.source_quality_action,
            "translation_required": self.translation_required,
            "cleanup_required": self.cleanup_required,
            "render_required": self.render_required,
            "parent_bbox": list(self.parent_bbox),
            "cleanup_target_bbox": list(self.cleanup_target_bbox),
            "render_allowed_area": list(self.render_allowed_area),
            "root_bbox": list(self.root_bbox),
            "source_region_ids": list(self.source_region_ids),
            "represented_child_ids": list(self.represented_child_ids),
            "source_candidates": [dict(item) for item in self.source_candidates],
            "semantic_class": self.semantic_class,
            "route_intent": self.route_intent,
            "cleanup_mode": self.cleanup_mode,
            "text_area_container_id": self.text_area_container_id,
            "text_area_container_type": self.text_area_container_type,
            "confidence": self.confidence,
            "reason_codes": list(self.reason_codes),
            "unresolved_reason": self.unresolved_reason,
            "translated_text": self.translated_text,
            "source_glyph_mask_ids": list(self.source_glyph_mask_ids),
            "cleanup_job_ids": list(self.cleanup_job_ids),
            "cleanup_mask_ids": list(self.cleanup_mask_ids),
            "render_decision_id": self.render_decision_id,
            "renderer_audit_id": self.renderer_audit_id,
            "execution_region": _copy_region_record(self.execution_region) if self.execution_region else self.to_region_record(),
        }

    def to_region_record(self) -> dict[str, Any]:
        """Return this bundle's parent-owned execution region.

        The record identity is the finalized parent id. Represented child/source
        regions remain evidence only and must not become separate execution
        units after this handoff.
        """

        if self.execution_region:
            record = _copy_region_record(self.execution_region)
            _sync_execution_region_from_bundle(self, record)
            self.execution_region = _copy_region_record(record)
            return _copy_region_record(record)

        bbox = _best_bbox(self.parent_bbox, self.render_allowed_area, self.cleanup_target_bbox, self.root_bbox)
        render_allowed = _best_bbox(self.render_allowed_area, bbox)
        cleanup_target = _best_bbox(self.cleanup_target_bbox, bbox)
        root_bbox = _best_bbox(self.root_bbox, render_allowed)
        semantic_class = self.semantic_class or _semantic_class_for_role(self.role)
        route_intent = self.route_intent or _route_intent_for_role(self.role)
        cleanup_mode = self.cleanup_mode or _cleanup_mode_for_role(self.role)
        container_type = self.text_area_container_type or _container_type_for_role(self.role)
        semantic_kind = _semantic_kind_for_role(self.role)
        cleanup_authorization = _cleanup_authorization_for_role(self.role)
        record = {
            "region_id": self.bundle_id,
            "page_id": self.page_id,
            "type": semantic_class,
            "semantic_class": semantic_class,
            "semantic_kind": semantic_kind,
            "cleanup_authorization": cleanup_authorization,
            "text_area_cleanup_authorization": cleanup_authorization,
            "semantic_authorization_state": cleanup_authorization,
            "text_area_semantic_authorization_state": cleanup_authorization,
            "authorization_explicit": True,
            "text_area_authorization_explicit": True,
            "authorization_field_origin": "parent_execution_bundle",
            "text_area_authorization_field_origin": "parent_execution_bundle",
            "authorization_basis": "finalized_parent_execution_bundle",
            "source_stage": "parent_execution_bundle",
            "execution_region_authority": "parent_execution_bundle",
            "execution_region_role": "parent_execution",
            "legacy_region_execution_authority": False,
            "source_region_evidence_only": True,
            "text_area_authorization_source_stage": "parent_execution_bundle",
            "route_intent": route_intent,
            "text_area_route_intent": route_intent,
            "container_type": container_type,
            "text_area_container_type": container_type,
            "cleanup_mode": cleanup_mode,
            "ocr_text": self.source_text,
            "source_text": self.source_text,
            "translation": self.translated_text,
            "translated_text": self.translated_text,
            "bbox": list(bbox),
            "polygon": _polygon_from_bbox(bbox),
            "order_index": 0,
            "flags": {
                "ignore": not self.translation_required and self.state != "punctuation_identity_parent",
                "bg_text": self.role in {"caption", "background", "caption_background", "background_narration"},
                "needs_review": bool(self.unresolved_reason),
            },
            "parent_execution_bundle_id": self.bundle_id,
            "parent_execution_bundle_version": PARENT_EXECUTION_BUNDLE_VERSION,
            "parent_execution_state": self.state,
            "parent_execution_authoritative": True,
            "text_block_root_id": self.root_id,
            "parent_logical_text_unit_id": self.parent_id,
            "active_translation_unit_id": self.parent_id if self.translation_required else "",
            "logical_text_block_id": self.parent_id,
            "logical_text_ownership_status": "parent_execution_bundle",
            "logical_text_block_source_text": self.source_text,
            "parent_logical_text_unit_source_text": self.source_text,
            "logical_text_block_bbox": list(cleanup_target),
            "parent_logical_text_unit_cleanup_target_bbox": list(cleanup_target),
            "parent_logical_text_unit_render_allowed_area": list(render_allowed),
            "logical_text_block_allowed_bbox": list(render_allowed),
            "logical_text_block_member_region_ids": list(self.source_region_ids),
            "logical_text_block_transferred_region_ids": list(self.source_region_ids),
            "logical_text_block_translation_unit": bool(self.translation_required),
            "child_final_state": "parent_anchor",
            "represented_child_ids": list(self.represented_child_ids),
            "source_region_ids": list(self.source_region_ids),
            "parent_source_coherence_action": self.source_quality_action,
            "logical_text_source_quality_action": self.source_quality_action,
            "source_conservation_status": self.source_quality_state,
            "source_glyph_mask_ids": list(self.source_glyph_mask_ids),
            "cleanup_job_ids": list(self.cleanup_job_ids),
            "cleanup_mask_ids": list(self.cleanup_mask_ids),
            "render_decision_id": self.render_decision_id,
            "renderer_audit_id": self.renderer_audit_id,
            "render": {
                "parent_execution_bundle_id": self.bundle_id,
                "parent_execution_bundle_version": PARENT_EXECUTION_BUNDLE_VERSION,
                "text_block_root_id": self.root_id,
                "parent_logical_text_unit_id": self.parent_id,
                "active_translation_unit_id": self.parent_id if self.translation_required else "",
                "logical_text_block_source_text": self.source_text,
                "parent_logical_text_unit_source_text": self.source_text,
                "source_text": self.source_text,
                "translation": self.translated_text,
                "translated_text": self.translated_text,
                "child_final_state": "parent_anchor",
                "cleanup_mode": cleanup_mode,
                "semantic_class": semantic_class,
                "semantic_kind": semantic_kind,
                "cleanup_authorization": cleanup_authorization,
                "text_area_cleanup_authorization": cleanup_authorization,
                "semantic_authorization_state": cleanup_authorization,
                "text_area_semantic_authorization_state": cleanup_authorization,
                "authorization_explicit": True,
                "text_area_authorization_explicit": True,
                "authorization_field_origin": "parent_execution_bundle",
                "text_area_authorization_field_origin": "parent_execution_bundle",
                "authorization_basis": "finalized_parent_execution_bundle",
                "source_stage": "parent_execution_bundle",
                "execution_region_authority": "parent_execution_bundle",
                "execution_region_role": "parent_execution",
                "legacy_region_execution_authority": False,
                "source_region_evidence_only": True,
                "parent_execution_authoritative": True,
                "text_area_authorization_source_stage": "parent_execution_bundle",
                "text_area_route_intent": route_intent,
                "route_intent": route_intent,
                "container_type": container_type,
                "text_area_container_type": container_type,
                "text_area_container_id": self.text_area_container_id,
                "text_area_container_bbox": list(root_bbox),
                "cleanup_allowed_area": list(render_allowed),
                "allowed_cleanup_area": list(render_allowed),
                "render_allowed_area": list(render_allowed),
                "logical_text_block_bbox": list(cleanup_target),
                "parent_logical_text_unit_cleanup_target_bbox": list(cleanup_target),
                "parent_logical_text_unit_render_allowed_area": list(render_allowed),
                "source_region_ids": list(self.source_region_ids),
                "represented_child_ids": list(self.represented_child_ids),
                "source_glyph_mask_ids": list(self.source_glyph_mask_ids),
                "cleanup_job_ids": list(self.cleanup_job_ids),
                "cleanup_mask_ids": list(self.cleanup_mask_ids),
                "render_decision_id": self.render_decision_id,
                "renderer_audit_id": self.renderer_audit_id,
                "parent_source_coherence_action": self.source_quality_action,
                "logical_text_source_quality_action": self.source_quality_action,
                "wrap_mode": "vertical",
            },
        }
        _sync_execution_region_from_bundle(self, record)
        self.execution_region = _copy_region_record(record)
        return _copy_region_record(record)


@dataclass
class ParentExecutionBundleResult:
    page_id: str
    bundles: list[ParentExecutionBundle] = field(default_factory=list)
    blocked_bundles: list[ParentExecutionBundle] = field(default_factory=list)
    excluded_nonworkflow_children: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def executable_bundles(self) -> list[ParentExecutionBundle]:
        return [
            bundle for bundle in self.bundles
            if bundle.state in {"active_translation_parent", "punctuation_identity_parent"}
        ]

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "parent_execution_bundle_version": PARENT_EXECUTION_BUNDLE_VERSION,
            "page_id": self.page_id,
            "bundle_count": len(self.bundles),
            "blocked_bundle_count": len(self.blocked_bundles),
            "executable_bundle_ids": [bundle.bundle_id for bundle in self.executable_bundles()],
            "bundles": [bundle.to_audit_dict() for bundle in self.bundles],
            "blocked_bundles": [bundle.to_audit_dict() for bundle in self.blocked_bundles],
            "excluded_nonworkflow_children": list(self.excluded_nonworkflow_children),
            "errors": list(self.errors),
        }


def build_parent_execution_bundles(
    *,
    page_id: str,
    hierarchy_result: Any,
    regions: Sequence[Mapping[str, Any]],
) -> ParentExecutionBundleResult:
    finalized = hierarchy_result.finalized_execution_units()
    root_by_id = {
        str(getattr(root, "root_id", "") or ""): root
        for root in getattr(hierarchy_result, "roots", []) or []
        if str(getattr(root, "root_id", "") or "")
    }
    parent_by_id = {
        str(getattr(parent, "parent_id", "") or ""): parent
        for parent in getattr(hierarchy_result, "parent_units", []) or []
        if str(getattr(parent, "parent_id", "") or "")
    }
    region_by_id = {
        str(region.get("region_id") or ""): dict(region)
        for region in regions or []
        if isinstance(region, Mapping) and str(region.get("region_id") or "")
    }

    result = ParentExecutionBundleResult(
        page_id=str(page_id or getattr(finalized, "page_id", "") or ""),
        excluded_nonworkflow_children=[
            child.to_dict() if hasattr(child, "to_dict") else dict(child)
            for child in getattr(finalized, "excluded_nonworkflow_children", []) or []
        ],
    )
    seen_parent_ids: set[str] = set()
    finalized_parents = (
        list(getattr(finalized, "active_translation_parents", []) or [])
        + list(getattr(finalized, "punctuation_parent_obligations", []) or [])
    )
    for parent in finalized_parents:
        parent_id = str(getattr(parent, "parent_id", "") or "")
        if not parent_id:
            result.errors.append("finalized_parent_missing_parent_id")
            continue
        if parent_id in seen_parent_ids:
            result.errors.append(f"duplicate_finalized_parent_id:{parent_id}")
            continue
        seen_parent_ids.add(parent_id)
        bundle = _bundle_from_finalized_parent(
            page_id=result.page_id,
            parent=parent,
            root_by_id=root_by_id,
            parent_by_id=parent_by_id,
            region_by_id=region_by_id,
        )
        result.bundles.append(bundle)

    for parent in getattr(finalized, "blocked_or_unresolved_parents", []) or []:
        bundle = _bundle_from_finalized_parent(
            page_id=result.page_id,
            parent=parent,
            root_by_id=root_by_id,
            parent_by_id=parent_by_id,
            region_by_id=region_by_id,
        )
        result.blocked_bundles.append(bundle)

    _validate_bundle_result(result)
    return result


def parent_execution_region_records(
    bundles: Sequence[ParentExecutionBundle],
) -> list[dict[str, Any]]:
    return [bundle.to_region_record() for bundle in bundles or []]


def sync_bundles_from_region_records(
    bundles: Sequence[ParentExecutionBundle],
    region_records: Sequence[Mapping[str, Any]],
) -> None:
    records_by_id = {
        str(record.get("region_id") or ""): record
        for record in region_records or []
        if isinstance(record, Mapping) and str(record.get("region_id") or "")
    }
    for bundle in bundles or []:
        record = records_by_id.get(bundle.bundle_id)
        if not record:
            continue
        bundle.execution_region = _copy_region_record(record)
        bundle.translated_text = str(record.get("translation") or record.get("translated_text") or "")
        bundle.source_glyph_mask_ids = _list_strings(record.get("source_glyph_mask_ids"))
        bundle.cleanup_job_ids = _list_strings(record.get("cleanup_job_ids"))
        bundle.cleanup_mask_ids = _list_strings(record.get("cleanup_mask_ids"))
        bundle.render_decision_id = str(record.get("render_decision_id") or "")
        bundle.renderer_audit_id = str(record.get("renderer_audit_id") or "")
        bundle.to_region_record()


def parent_execution_bundles_from_audit_records(
    records: Sequence[Mapping[str, Any]],
) -> list[ParentExecutionBundle]:
    """Rehydrate saved parent execution bundle audit records for UI consumers."""

    bundles: list[ParentExecutionBundle] = []
    for record in records or []:
        if not isinstance(record, Mapping):
            continue
        bundle = ParentExecutionBundle(
            page_id=str(record.get("page_id") or ""),
            bundle_id=str(record.get("bundle_id") or record.get("parent_id") or ""),
            root_id=str(record.get("root_id") or ""),
            parent_id=str(record.get("parent_id") or record.get("bundle_id") or ""),
            graph_parent_id=str(record.get("graph_parent_id") or record.get("parent_id") or ""),
            state=str(record.get("state") or ""),
            role=str(record.get("role") or ""),
            source_text=str(record.get("source_text") or ""),
            source_quality_state=str(record.get("source_quality_state") or "accepted_for_translation"),
            source_quality_action=str(record.get("source_quality_action") or "translate"),
            translation_required=bool(record.get("translation_required")),
            cleanup_required=bool(record.get("cleanup_required")),
            render_required=bool(record.get("render_required")),
            parent_bbox=_bbox(record.get("parent_bbox")),
            cleanup_target_bbox=_bbox(record.get("cleanup_target_bbox")),
            render_allowed_area=_bbox(record.get("render_allowed_area")),
            root_bbox=_bbox(record.get("root_bbox")),
            source_region_ids=_list_strings(record.get("source_region_ids")),
            represented_child_ids=_list_strings(record.get("represented_child_ids")),
            source_candidates=[
                dict(item)
                for item in (record.get("source_candidates") or [])
                if isinstance(item, Mapping)
            ],
            semantic_class=str(record.get("semantic_class") or ""),
            route_intent=str(record.get("route_intent") or ""),
            cleanup_mode=str(record.get("cleanup_mode") or ""),
            text_area_container_id=str(record.get("text_area_container_id") or ""),
            text_area_container_type=str(record.get("text_area_container_type") or ""),
            confidence=_float_or_none(record.get("confidence")),
            reason_codes=_list_strings(record.get("reason_codes")),
            unresolved_reason=record.get("unresolved_reason"),
            translated_text=str(record.get("translated_text") or ""),
            source_glyph_mask_ids=_list_strings(record.get("source_glyph_mask_ids")),
            cleanup_job_ids=_list_strings(record.get("cleanup_job_ids")),
            cleanup_mask_ids=_list_strings(record.get("cleanup_mask_ids")),
            render_decision_id=str(record.get("render_decision_id") or ""),
            renderer_audit_id=str(record.get("renderer_audit_id") or ""),
            execution_region=_copy_region_record(record.get("execution_region") or {}),
        )
        if not bundle.execution_region:
            bundle.to_region_record()
        bundles.append(bundle)
    return bundles


def _bundle_from_finalized_parent(
    *,
    page_id: str,
    parent: Any,
    root_by_id: Mapping[str, Any],
    parent_by_id: Mapping[str, Any],
    region_by_id: Mapping[str, Mapping[str, Any]],
) -> ParentExecutionBundle:
    parent_id = str(getattr(parent, "parent_id", "") or "")
    root_id = str(getattr(parent, "root_id", "") or "")
    parent_unit = parent_by_id.get(parent_id)
    root = root_by_id.get(root_id)
    source_region_ids = _list_strings(getattr(parent, "source_region_ids", []))
    source_records = [
        _source_candidate_from_region(region_by_id.get(region_id, {}), region_id)
        for region_id in source_region_ids
    ]
    source_records = [record for record in source_records if record]
    role = str(getattr(parent, "role", "") or getattr(parent_unit, "role", "") or "")
    parent_bbox = _best_bbox(
        getattr(parent, "render_allowed_area", []),
        getattr(parent, "cleanup_target_bbox", []),
        getattr(parent_unit, "parent_visual_group_bbox", []),
        _union_region_bboxes([region_by_id.get(region_id, {}) for region_id in source_region_ids]),
    )
    cleanup_target = _best_bbox(getattr(parent, "cleanup_target_bbox", []), parent_bbox)
    render_allowed = _best_bbox(getattr(parent, "render_allowed_area", []), parent_bbox)
    root_bbox = _best_bbox(getattr(root, "bbox", []), render_allowed)
    primary_region = region_by_id.get(source_region_ids[0], {}) if source_region_ids else {}
    primary_render = primary_region.get("render") if isinstance(primary_region, Mapping) else {}
    if not isinstance(primary_render, Mapping):
        primary_render = {}
    source_action = str(
        getattr(parent_unit, "source_coherence_action", "")
        or primary_region.get("logical_text_source_quality_action")
        or primary_render.get("logical_text_source_quality_action")
        or "translate"
    )
    source_state = str(
        getattr(parent_unit, "source_conservation_status", "")
        or primary_region.get("source_conservation_status")
        or "accepted_for_translation"
    )
    return ParentExecutionBundle(
        page_id=page_id,
        bundle_id=parent_id,
        root_id=root_id,
        parent_id=parent_id,
        graph_parent_id=parent_id,
        state=str(getattr(parent, "state", "") or ""),
        role=role,
        source_text=str(getattr(parent, "source_text", "") or ""),
        source_quality_state=source_state,
        source_quality_action=source_action,
        translation_required=bool(getattr(parent, "translation_required", False)),
        cleanup_required=bool(getattr(parent, "cleanup_required", False)),
        render_required=bool(getattr(parent, "render_required", False)),
        parent_bbox=list(parent_bbox),
        cleanup_target_bbox=list(cleanup_target),
        render_allowed_area=list(render_allowed),
        root_bbox=list(root_bbox),
        source_region_ids=source_region_ids,
        represented_child_ids=_list_strings(getattr(parent, "represented_child_ids", [])),
        source_candidates=source_records,
        semantic_class=_semantic_class_for_role(role),
        route_intent=_route_intent_for_role(role),
        cleanup_mode=_cleanup_mode_for_role(role),
        text_area_container_id=str(
            primary_region.get("text_area_container_id")
            or primary_render.get("text_area_container_id")
            or ""
        ),
        text_area_container_type=str(
            primary_region.get("text_area_container_type")
            or primary_render.get("text_area_container_type")
            or _container_type_for_role(role)
        ),
        confidence=_float_or_none(getattr(parent_unit, "confidence", None)),
        reason_codes=_list_strings(getattr(parent, "reason_codes", [])),
        unresolved_reason=getattr(parent, "unresolved_reason", None),
    )


def _sync_execution_region_from_bundle(
    bundle: ParentExecutionBundle,
    record: dict[str, Any],
) -> None:
    render = record.setdefault("render", {})
    if not isinstance(render, dict):
        render = {}
        record["render"] = render
    record["region_id"] = bundle.bundle_id
    record["parent_execution_bundle_id"] = bundle.bundle_id
    record["parent_execution_bundle_version"] = PARENT_EXECUTION_BUNDLE_VERSION
    record["parent_execution_state"] = bundle.state
    record["parent_execution_authoritative"] = True
    record["text_block_root_id"] = bundle.root_id
    record["parent_logical_text_unit_id"] = bundle.parent_id
    record["active_translation_unit_id"] = bundle.parent_id if bundle.translation_required else ""
    record["logical_text_block_id"] = bundle.parent_id
    record["ocr_text"] = bundle.source_text
    record["source_text"] = bundle.source_text
    record["logical_text_block_source_text"] = bundle.source_text
    record["parent_logical_text_unit_source_text"] = bundle.source_text
    record["translation"] = bundle.translated_text
    record["translated_text"] = bundle.translated_text
    record["source_region_ids"] = list(bundle.source_region_ids)
    record["represented_child_ids"] = list(bundle.represented_child_ids)
    record["source_glyph_mask_ids"] = list(bundle.source_glyph_mask_ids)
    record["cleanup_job_ids"] = list(bundle.cleanup_job_ids)
    record["cleanup_mask_ids"] = list(bundle.cleanup_mask_ids)
    record["render_decision_id"] = bundle.render_decision_id
    record["renderer_audit_id"] = bundle.renderer_audit_id
    record["execution_region_authority"] = "parent_execution_bundle"
    record["execution_region_role"] = "parent_execution"
    record["legacy_region_execution_authority"] = False
    record["source_region_evidence_only"] = True
    render["parent_execution_bundle_id"] = bundle.bundle_id
    render["parent_execution_bundle_version"] = PARENT_EXECUTION_BUNDLE_VERSION
    render["parent_execution_authoritative"] = True
    render["text_block_root_id"] = bundle.root_id
    render["parent_logical_text_unit_id"] = bundle.parent_id
    render["active_translation_unit_id"] = bundle.parent_id if bundle.translation_required else ""
    render["logical_text_block_source_text"] = bundle.source_text
    render["parent_logical_text_unit_source_text"] = bundle.source_text
    render["source_text"] = bundle.source_text
    render["translation"] = bundle.translated_text
    render["translated_text"] = bundle.translated_text
    render["source_region_ids"] = list(bundle.source_region_ids)
    render["represented_child_ids"] = list(bundle.represented_child_ids)
    render["source_glyph_mask_ids"] = list(bundle.source_glyph_mask_ids)
    render["cleanup_job_ids"] = list(bundle.cleanup_job_ids)
    render["cleanup_mask_ids"] = list(bundle.cleanup_mask_ids)
    render["render_decision_id"] = bundle.render_decision_id
    render["renderer_audit_id"] = bundle.renderer_audit_id
    render["execution_region_authority"] = "parent_execution_bundle"
    render["execution_region_role"] = "parent_execution"
    render["legacy_region_execution_authority"] = False
    render["source_region_evidence_only"] = True


def _validate_bundle_result(result: ParentExecutionBundleResult) -> None:
    seen: set[str] = set()
    for bundle in result.bundles:
        if bundle.parent_id in seen:
            result.errors.append(f"duplicate_bundle_parent_id:{bundle.parent_id}")
        seen.add(bundle.parent_id)
        if bundle.state == "active_translation_parent" and not bundle.source_text.strip():
            result.errors.append(f"active_parent_missing_source_text:{bundle.parent_id}")
        if bundle.state == "active_translation_parent" and not _valid_bbox(bundle.parent_bbox):
            result.errors.append(f"active_parent_missing_parent_bbox:{bundle.parent_id}")
        if not bundle.root_id:
            result.errors.append(f"bundle_missing_root_id:{bundle.parent_id}")


def _source_candidate_from_region(region: Mapping[str, Any], region_id: str) -> dict[str, Any]:
    if not isinstance(region, Mapping):
        return {}
    render = region.get("render") or {}
    if not isinstance(render, Mapping):
        render = {}
    return {
        "region_id": region_id,
        "ocr_text": str(region.get("ocr_text") or render.get("ocr_text") or ""),
        "bbox": _bbox(region.get("bbox")),
        "polygon": list(region.get("polygon") or []),
        "child_id": str(region.get("child_recognized_text_segment_id") or render.get("child_recognized_text_segment_id") or ""),
        "confidence": region.get("confidence"),
        "detection_source": str(region.get("detection_source") or render.get("detection_source") or ""),
    }


def _semantic_class_for_role(role: str) -> str:
    lowered = str(role or "").strip().lower()
    if lowered == "speech":
        return "speech_bubble"
    if lowered in {"caption", "background", "caption_background", "background_narration"}:
        return "caption_background"
    if lowered == "review":
        return "review"
    return "speech_bubble" if not lowered else lowered


def _route_intent_for_role(role: str) -> str:
    lowered = str(role or "").strip().lower()
    if lowered == "speech":
        return "translate_speech"
    if lowered in {"caption", "background", "caption_background", "background_narration"}:
        return "translate_caption"
    return "translate"


def _cleanup_mode_for_role(role: str) -> str:
    lowered = str(role or "").strip().lower()
    if lowered == "speech":
        return "bubble"
    if lowered in {"caption", "background", "caption_background", "background_narration"}:
        return "background_box"
    return "bubble"


def _cleanup_authorization_for_role(role: str) -> str:
    lowered = str(role or "").strip().lower()
    if lowered == "speech":
        return "cleanup_translate_speech"
    if lowered in {"caption", "caption_background"}:
        return "cleanup_translate_caption"
    if lowered in {"background", "background_narration"}:
        return "cleanup_translate_background"
    return "cleanup_translate_speech"


def _semantic_kind_for_role(role: str) -> str:
    lowered = str(role or "").strip().lower()
    if lowered == "speech":
        return "speech"
    if lowered in {"caption", "caption_background"}:
        return "caption"
    if lowered in {"background", "background_narration"}:
        return "background_narration"
    if lowered == "review":
        return "unknown"
    return lowered or "speech"


def _container_type_for_role(role: str) -> str:
    lowered = str(role or "").strip().lower()
    if lowered == "speech":
        return "speech_bubble"
    if lowered in {"caption", "background", "caption_background", "background_narration"}:
        return "caption_background"
    return "text_area"


def _best_bbox(*candidates: Any) -> list[int]:
    for candidate in candidates:
        bbox = _bbox(candidate)
        if _valid_bbox(bbox):
            return bbox
    return []


def _bbox(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return []
    try:
        return [int(round(float(value[0]))), int(round(float(value[1]))), int(round(float(value[2]))), int(round(float(value[3])))]
    except Exception:
        return []


def _valid_bbox(value: Any) -> bool:
    bbox = _bbox(value)
    return len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0


def _polygon_from_bbox(bbox: Sequence[int]) -> list[list[int]]:
    box = _bbox(bbox)
    if not _valid_bbox(box):
        return []
    x, y, w, h = box
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def _union_region_bboxes(regions: Sequence[Mapping[str, Any]]) -> list[int]:
    boxes = [_bbox(region.get("bbox")) for region in regions or [] if isinstance(region, Mapping)]
    boxes = [box for box in boxes if _valid_bbox(box)]
    if not boxes:
        return []
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[0] + box[2] for box in boxes)
    y2 = max(box[1] + box[3] for box in boxes)
    return [x1, y1, max(1, x2 - x1), max(1, y2 - y1)]


def _copy_region_record(record: Any) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        return {}
    copied = dict(record)
    render = copied.get("render")
    if isinstance(render, Mapping):
        copied["render"] = dict(render)
    flags = copied.get("flags")
    if isinstance(flags, Mapping):
        copied["flags"] = dict(flags)
    return copied


def _list_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return [str(value)] if str(value) else []


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
