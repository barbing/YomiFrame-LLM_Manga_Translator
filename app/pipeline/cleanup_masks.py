"""Output-neutral CleanupMask contract builder.

This module turns existing CleanupJob candidates and SourceGlyphMask evidence
into CleanupMask contract records. It does not execute cleanup, choose a
backend, run proof, or feed masks into the renderer.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:  # pragma: no cover - exercised in target conda env
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:  # pragma: no cover - exercised in target conda env
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore

from app.pipeline.cleanup_contracts import (
    CleanupClass,
    CleanupJob,
    CleanupMask,
    TextForegroundSegmentationMask,
)


CLEANUP_MASK_CONTRACT_VERSION = "cleanup_masks_phase2"
MAX_ERASE_GROWTH_RATIO = 3.5
MAX_ERASE_PAGE_PIXEL_RATIO = 0.12
MAX_ERASE_BBOX_PAGE_RATIO = 0.25
MAX_ALLOWED_PAGE_RATIO = 0.85
MAX_CAPTION_FLAT_SMALL_EXCEPTION_GROWTH_RATIO = 4.25
MAX_CAPTION_FLAT_SMALL_EXCEPTION_ERASE_PIXELS = 1_200
MAX_CAPTION_FLAT_SMALL_EXCEPTION_ERASE_BBOX_AREA = 18_000
MAX_CAPTION_FLAT_SMALL_EXCEPTION_ALLOWED_AREA = 30_000
MAX_CAPTION_FLAT_SMALL_EXCEPTION_ALLOWED_PAGE_RATIO = 0.025
MAX_CAPTION_FLAT_SMALL_EXCEPTION_ERASE_ALLOWED_RATIO = 0.08
MIN_SEGMENTATION_READY_PIXELS = 32
MIN_SEGMENTATION_READY_COVERAGE_RATIO = 0.05
MIN_SEGMENTATION_READY_SMALL_COVERAGE_RATIO = 0.035
MIN_OWNED_SEGMENTATION_TO_EXECUTABLE_RATIO = 0.75
MIN_OWNED_SEGMENTATION_TO_EXECUTABLE_SMALL_RATIO = 0.55
FRAGMENT_ONLY_MAX_PIXELS = 320
FRAGMENT_ONLY_MAX_COVERAGE_RATIO = 0.025
PROTECTED_DOMINANT_OVERLAP_RATIO = 0.35
COMPONENT_OWNER_MIN_RATIO = 0.10
COMPONENT_OWNER_MIN_PIXELS = 8
COMPONENT_OWNER_STRONG_PIXELS = 64
COMPONENT_PROTECTED_MIN_RATIO = 0.15
COMPONENT_PROTECTED_DOMINANT_RATIO = 0.35
COMPONENT_MULTI_OWNER_DOMINANCE_RATIO = 0.60


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
    if isinstance(value, Mapping):
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
            "module": "app.pipeline.cleanup_masks",
            "stage": stage,
            "event": event,
        }
        payload.update(_cleanup_perf_contract_json_safe(fields))
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception:
        return


@dataclass(frozen=True)
class CleanupMaskBuildResult:
    """CleanupMask build output without renderer consumption."""

    page_id: str
    version: str
    masks: list[CleanupMask] = field(default_factory=list)
    masks_by_job_id: dict[str, list[CleanupMask]] = field(default_factory=dict)
    rejected_records: list[dict[str, Any]] = field(default_factory=list)
    protected_records: list[dict[str, Any]] = field(default_factory=list)
    skipped_records: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    text_foreground_segmentation: dict[str, Any] | None = None

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "page_id": self.page_id,
            "renderer_consumed": False,
            "text_foreground_segmentation": _json_safe(self.text_foreground_segmentation or {}),
            "masks": [_json_safe(mask) for mask in self.masks],
            "masks_by_job_id": {
                str(job_id): [_json_safe(mask) for mask in masks]
                for job_id, masks in self.masks_by_job_id.items()
            },
            "rejected_records": _json_safe(self.rejected_records),
            "protected_records": _json_safe(self.protected_records),
            "skipped_records": _json_safe(self.skipped_records),
            "errors": list(self.errors),
            "summary": {
                "mask_count": len(self.masks),
                "rejected_record_count": len(self.rejected_records),
                "protected_record_count": len(self.protected_records),
                "skipped_record_count": len(self.skipped_records),
                "error_count": len(self.errors),
                "renderer_consumed": False,
            },
        }


@dataclass(frozen=True)
class _SourceEvidence:
    region_id: str
    mask_id: str
    audit: dict[str, Any]
    raw: Any


@dataclass(frozen=True)
class _EffectiveMaskBuild:
    foreground: np.ndarray | None
    erase: np.ndarray | None
    status: str
    failure_reason: str
    audit: dict[str, Any]
    rejected: bool = False


@dataclass(frozen=True)
class _OwnershipBinding:
    owner_mask: np.ndarray | None
    protected_mask: np.ndarray | None
    owned_bbox: list[int] | None
    protected_bbox: list[int] | None
    method: str
    status: str
    protected_records: list[dict[str, Any]]


@dataclass(frozen=True)
class _ComponentOwnershipProjection:
    labels: np.ndarray | None
    components: list[dict[str, Any]]
    component_label_by_id: dict[str, int]


def _segmentation_foreground_mask(segmentation: TextForegroundSegmentationMask | Any | None) -> tuple[np.ndarray | None, dict[str, Any]]:
    if segmentation is None:
        return None, {
            "segmentation_mask_status": "segmentation_mask_missing",
            "segmentation_mask_failure_reason": "text_foreground_segmentation_missing",
        }
    refined = _get_value(segmentation, "refined_mask")
    if refined is None:
        refined = _get_value(segmentation, "mask_refined")
    if refined is None:
        refined = _get_value(segmentation, "mask")
    mask = _to_binary_mask(refined)
    audit = _segmentation_audit(segmentation)
    if mask is None or int(np.count_nonzero(mask)) <= 0:
        audit.update(
            {
                "segmentation_mask_status": "segmentation_mask_missing",
                "segmentation_mask_failure_reason": "refined_segmentation_mask_empty_or_missing",
            }
        )
        return None, audit
    audit.setdefault("segmentation_mask_status", "segmentation_mask_ready")
    audit.setdefault("segmentation_mask_failure_reason", "")
    audit.setdefault("text_pixel_count", int(np.count_nonzero(mask)))
    return mask.astype(np.uint8), audit


def _segmentation_audit(segmentation: Any) -> dict[str, Any]:
    if hasattr(segmentation, "to_audit_dict"):
        raw = segmentation.to_audit_dict()
        if isinstance(raw, Mapping):
            audit = dict(raw)
        else:
            audit = {}
    elif isinstance(segmentation, Mapping):
        audit = dict(segmentation)
    else:
        audit = {}
        for key in (
            "page_id",
            "image_size",
            "raw_mask_ref",
            "refined_mask_ref",
            "threshold_used",
            "provider",
            "backend",
            "runtime_ms",
            "text_pixel_count",
            "connected_component_stats",
            "block_associations",
            "keep_undetected_mask",
            "confidence",
            "provenance",
        ):
            value = _get_value(segmentation, key)
            if value is not None:
                audit[key] = value
    stats = audit.get("connected_component_stats")
    component_count = None
    if isinstance(stats, Mapping):
        component_count = stats.get("component_count")
    return {
        "page_id": audit.get("page_id", ""),
        "image_size": audit.get("image_size"),
        "raw_mask_ref": audit.get("raw_mask_ref", ""),
        "refined_mask_ref": audit.get("refined_mask_ref", ""),
        "threshold_used": audit.get("threshold_used"),
        "provider": audit.get("provider", ""),
        "backend": audit.get("backend", ""),
        "runtime_ms": audit.get("runtime_ms"),
        "text_pixel_count": audit.get("text_pixel_count", 0),
        "connected_component_stats": stats or {},
        "block_associations": list(audit.get("block_associations") or []),
        "keep_undetected_mask": bool(audit.get("keep_undetected_mask", False)),
        "confidence": audit.get("confidence", {}),
        "provenance": audit.get("provenance", {}),
        "segmentation_provider": audit.get("provider", ""),
        "segmentation_mask_ref": audit.get("refined_mask_ref", ""),
        "segmentation_text_pixels": audit.get("text_pixel_count", 0),
        "segmentation_component_count": component_count,
    }


def build_cleanup_masks(
    *,
    page_id: str,
    job_candidates: Sequence[CleanupJob],
    source_glyph_masks: Any,
    image_size: tuple[int, int] | None = None,
    source_image_path: str | Path | None = None,
    source_image: Any | None = None,
    text_foreground_segmentation: TextForegroundSegmentationMask | Any | None = None,
    page_region_records: Sequence[Mapping[str, Any]] | None = None,
) -> CleanupMaskBuildResult:
    """Build CleanupMask contracts from segmentation foreground and source-glyph provenance."""

    started = time.time()
    evidence_records = _index_source_evidence(source_glyph_masks)
    segmentation_mask, segmentation_audit = _segmentation_foreground_mask(text_foreground_segmentation)
    source_np, source_error = _source_image_array(source_image=source_image, source_image_path=source_image_path)
    _cleanup_perf_contract_checkpoint(
        "cleanup_mask_build",
        "module_start",
        page_id=page_id,
        job_count=len(job_candidates or []),
        source_evidence_count=len(evidence_records),
        segmentation_available=segmentation_mask is not None,
        segmentation_text_pixels=segmentation_audit.get("text_pixel_count", 0),
        source_image_available=source_np is not None,
        source_image_error=source_error,
    )
    evidence_by_id: dict[str, _SourceEvidence] = {}
    for evidence in evidence_records:
        if not evidence.mask_id:
            continue
        existing = evidence_by_id.get(evidence.mask_id)
        if existing is None or (_evidence_has_raw_mask(evidence) and not _evidence_has_raw_mask(existing)):
            evidence_by_id[evidence.mask_id] = evidence
    evidence_by_region: dict[str, list[_SourceEvidence]] = {}
    for evidence in evidence_records:
        if evidence.region_id:
            evidence_by_region.setdefault(evidence.region_id, []).append(evidence)
    region_records = _index_region_records(page_region_records)
    component_projection = _build_component_ownership_projection(
        page_id=page_id,
        segmentation_mask=segmentation_mask,
        job_candidates=job_candidates or [],
        region_records=region_records,
        evidence_records=evidence_records,
    )
    segmentation_audit = {
        **segmentation_audit,
        "component_ownership_inventory": component_projection.components,
        "component_ownership_state_counts": _component_ownership_state_counts(component_projection.components),
        "segmentation_component_count": len(component_projection.components),
        "component_projection_method": "segmentation_component_ownership_projection",
    }

    masks: list[CleanupMask] = []
    masks_by_job_id: dict[str, list[CleanupMask]] = {}
    rejected_records: list[dict[str, Any]] = []
    protected_records: list[dict[str, Any]] = []
    skipped_records: list[dict[str, Any]] = []
    errors: list[str] = []

    for job in job_candidates or []:
        job_started = time.time()
        _cleanup_perf_contract_checkpoint(
            "cleanup_mask_job",
            "start",
            page_id=page_id,
            cleanup_job_id=str(getattr(job, "cleanup_job_id", "") or ""),
            target_region_count=len(getattr(job, "target_region_ids", []) or []),
            required_source_glyph_count=len(_required_source_glyph_ids(job)),
        )
        try:
            base = _base_job_record(page_id, job)
            protection_reason = _job_protection_reason(job)
            if protection_reason:
                protected_records.append({**base, "reason": protection_reason})
                continue

            matched = _matching_evidence(job, evidence_by_id, evidence_by_region)
            required_source_ids = _required_source_glyph_ids(job)
            missing_required_ids: list[str] = []
            if required_source_ids:
                matched = [source for source in matched if source.mask_id in required_source_ids]
                matched_by_id = {source.mask_id for source in matched if source.mask_id}
                missing_required_ids = [mask_id for mask_id in required_source_ids if mask_id not in matched_by_id]

            source = _primary_source_evidence(job, matched) if matched else None
            allowed = (
                _valid_bbox(getattr(job, "allowed_cleanup_area", None))
                or _allowed_area_from_regions(job, region_records)
                or _allowed_area_from_evidence(matched)
            )
            if allowed is None:
                rejected_records.append({**base, "reason": "allowed_cleanup_area_missing_or_invalid"})
                continue

            allowed_rejection = _allowed_area_rejection(allowed, image_size)
            if allowed_rejection:
                rejected_records.append({**base, "reason": allowed_rejection, "allowed_area": allowed})
                continue

            source_seed_foreground, _source_seed_erase, foreground_source_keys, erase_source_keys, consumed_source_ids, missing_foreground_source_ids, used_transitional_erase = _union_masks_from_evidence(
                matched,
                allowed=allowed,
            )
            mask_shape = (
                segmentation_mask.shape
                if segmentation_mask is not None
                else (source_seed_foreground.shape if source_seed_foreground is not None else None)
            )
            ownership_binding = _build_ownership_binding(
                job=job,
                allowed=allowed,
                shape=mask_shape,
                region_records=region_records,
            )
            effective = _build_component_projected_text_mask(
                job=job,
                segmentation_mask=segmentation_mask,
                segmentation_audit=segmentation_audit,
                seed_foreground=source_seed_foreground,
                allowed=allowed,
                matched=matched,
                ownership_binding=ownership_binding,
                component_projection=component_projection,
                missing_source_glyph_mask_ids=missing_required_ids,
            )
            if effective.foreground is None:
                if str(effective.audit.get("component_projection_method") or "") == "segmentation_component_ownership_projection":
                    rejected_records.append(
                        {
                            **base,
                            "reason": effective.failure_reason or "cleanup_mask_missing_no_owned_segmentation_components",
                            "required_source_glyph_mask_ids": required_source_ids,
                            "missing_source_glyph_mask_ids": missing_required_ids,
                            **effective.audit,
                        }
                    )
                    continue
                if source_seed_foreground is None:
                    rejected_records.append(
                        {
                            **base,
                            "reason": effective.failure_reason
                            or "segmentation_mask_missing",
                            "required_source_glyph_mask_ids": required_source_ids,
                            "missing_source_glyph_mask_ids": missing_foreground_source_ids,
                            **effective.audit,
                        }
                    )
                    continue
                fallback = _build_effective_text_mask(
                    job=job,
                    source_np=source_np,
                    source_error=source_error,
                    seed_foreground=source_seed_foreground,
                    allowed=allowed,
                    matched=matched,
                )
                fallback_audit = {
                    **fallback.audit,
                    "ownership_binding_status": effective.audit.get("ownership_binding_status", ""),
                    "ownership_binding_method": effective.audit.get("ownership_binding_method", ""),
                    "cleanup_owned_unit_bbox": effective.audit.get("cleanup_owned_unit_bbox"),
                    "cleanup_owned_unit_mask_ref": effective.audit.get("cleanup_owned_unit_mask_ref", ""),
                    "protected_mask_ref": effective.audit.get("protected_mask_ref", ""),
                    "protected_overlap_pixels": effective.audit.get("protected_overlap_pixels"),
                    "segmentation_pixels_before_binding": effective.audit.get("segmentation_pixels_before_binding"),
                    "segmentation_pixels_after_owner_clip": effective.audit.get("segmentation_pixels_after_owner_clip"),
                    "segmentation_pixels_after_protection_subtract": effective.audit.get("segmentation_pixels_after_protection_subtract"),
                    "sourceglyph_overlap_pixels": effective.audit.get("sourceglyph_overlap_pixels"),
                    "sourceglyph_overlap_ratio": effective.audit.get("sourceglyph_overlap_ratio"),
                    "segmentation_outside_sourceglyph_pixels": effective.audit.get("segmentation_outside_sourceglyph_pixels"),
                    "effective_coverage_ratio": fallback.audit.get("text_block_coverage_estimate", 0.0),
                    "effective_coverage_status": effective.audit.get("effective_coverage_status", ""),
                    "segmentation_mask_status": effective.audit.get("segmentation_mask_status", ""),
                    "segmentation_mask_failure_reason": effective.failure_reason
                    or effective.audit.get("segmentation_mask_failure_reason", ""),
                    "mask_completion_method": (
                        f"local_contrast_fallback_after_{effective.audit.get('segmentation_mask_status') or 'segmentation_unavailable'}"
                    ),
                }
                effective = _EffectiveMaskBuild(
                    foreground=fallback.foreground,
                    erase=fallback.erase,
                    status=(
                        "cleanup_mask_unresolved_after_segmentation"
                        if fallback.foreground is not None and fallback.erase is not None
                        else fallback.status
                    ),
                    failure_reason=effective.failure_reason
                    or fallback.failure_reason
                    or "segmentation_unavailable_local_contrast_fallback",
                    audit=fallback_audit,
                    rejected=fallback.rejected,
                )
            if effective.foreground is None or effective.erase is None:
                rejected_records.append(
                    {
                        **base,
                        "reason": effective.failure_reason or "effective_mask_failed_insufficient_evidence",
                        **effective.audit,
                    }
                )
                continue
            foreground = effective.foreground
            erase = effective.erase
            foreground_pixels = int(np.count_nonzero(foreground))
            erase_pixels = int(np.count_nonzero(erase))
            if erase_pixels <= 0:
                rejected_records.append({**base, "reason": "erase_mask_empty_after_allowed_area_clip"})
                continue

            foreground_bbox = _mask_bbox(foreground)
            erase_bbox = _mask_bbox(erase)
            if erase_bbox is None:
                rejected_records.append({**base, "reason": "erase_mask_bbox_empty"})
                continue

            growth_ratio = _growth_ratio(erase_pixels, foreground_pixels)
            artifact_risk = (
                "source_glyph_reported_artifact_risk"
                if any(_truthy(_first_present(item.audit, "cleanup_visual_artifact_risk", default=False)) for item in matched)
                else ""
            )
            exception_reason = _caption_flat_small_mask_growth_exception_reason(
                job=job,
                source=source,
                allowed=allowed,
                erase_bbox=erase_bbox,
                erase_pixels=erase_pixels,
                foreground_pixels=foreground_pixels,
                growth_ratio=growth_ratio,
                image_size=image_size,
                artifact_risk=artifact_risk,
            )
            broad_rejection = _broad_mask_rejection(
                allowed=allowed,
                erase_bbox=erase_bbox,
                erase_pixels=erase_pixels,
                growth_ratio=growth_ratio,
                image_size=image_size,
                allow_growth_exception=bool(exception_reason),
            )
            if broad_rejection:
                rejected_records.append(
                    {
                        **base,
                        "reason": broad_rejection,
                        "allowed_area": allowed,
                        "erase_mask_bbox": erase_bbox,
                        "erase_mask_pixels": erase_pixels,
                        "foreground_mask_pixels": foreground_pixels,
                        "growth_ratio": growth_ratio,
                        "mask_contract_exception_reason": exception_reason,
                        "effective_mask_status": effective.status,
                        "effective_mask_failure_reason": broad_rejection,
                        **effective.audit,
                    }
                )
                continue

            mask_source = (
                "cleanup_mask_from_text_foreground_segmentation"
                if str(effective.audit.get("mask_completion_method") or "").startswith("text_foreground_segmentation")
                else "cleanup_mask_diagnostic_non_segmentation_fallback"
            )
            seed_method = _mask_method_union(matched, erase_source_keys or foreground_source_keys)
            completion_method = str(effective.audit.get("mask_completion_method") or "")
            mask_method = (
                f"{completion_method}|seed:{seed_method}"
                if completion_method
                else seed_method
            )
            visual_scope = _cleanup_visual_scope_for_unit(
                required_source_ids=required_source_ids,
                consumed_source_ids=consumed_source_ids,
            )
            if str(effective.audit.get("component_projection_method") or "") == "segmentation_component_ownership_projection":
                visual_scope = "segmentation_component"
            cleanup_mask = CleanupMask(
                cleanup_mask_id=f"cmask_{_safe_id(page_id)}_{_safe_id(job.cleanup_job_id)}",
                cleanup_job_id=str(job.cleanup_job_id),
                foreground_mask_source_id=consumed_source_ids[0] if consumed_source_ids else None,
                foreground_mask_source_ids=consumed_source_ids,
                consumed_source_glyph_mask_ids=consumed_source_ids,
                missing_source_glyph_mask_ids=[],
                foreground_mask_bbox=foreground_bbox,
                foreground_mask_pixels=foreground_pixels,
                erase_mask_bbox=erase_bbox,
                erase_mask_pixels=erase_pixels,
                allowed_area=allowed,
                growth_ratio=growth_ratio,
                mask_source=mask_source,
                mask_method=mask_method,
                rejection_reason=effective.failure_reason if effective.rejected else "",
                mask_contract_exception_reason=exception_reason,
                artifact_risk=artifact_risk,
                visual_scope=visual_scope,
                protected=False,
                protection_reason="",
                effective_mask_status=effective.status,
                effective_mask_failure_reason=effective.failure_reason,
                seed_foreground_pixels=effective.audit.get("seed_foreground_pixels"),
                completed_foreground_pixels=effective.audit.get("completed_foreground_pixels"),
                component_count_before=effective.audit.get("component_count_before"),
                component_count_after=effective.audit.get("component_count_after"),
                largest_component_pixels_before=effective.audit.get("largest_component_pixels_before"),
                largest_component_pixels_after=effective.audit.get("largest_component_pixels_after"),
                text_block_coverage_estimate=effective.audit.get("text_block_coverage_estimate"),
                bbox_fill_ratio_before=effective.audit.get("bbox_fill_ratio_before"),
                bbox_fill_ratio_after=effective.audit.get("bbox_fill_ratio_after"),
                analysis_scope_bbox=effective.audit.get("analysis_scope_bbox"),
                executable_erase_bbox=effective.audit.get("executable_erase_bbox"),
                mask_completion_method=effective.audit.get("mask_completion_method", ""),
                polarity_mode=effective.audit.get("polarity_mode", ""),
                source_seed_mask_ids=consumed_source_ids,
                recovered_component_count=effective.audit.get("recovered_component_count"),
                rejected_component_count=effective.audit.get("rejected_component_count"),
                rejected_component_reasons=effective.audit.get("rejected_component_reasons", []),
                segmentation_mask_status=effective.audit.get("segmentation_mask_status", ""),
                segmentation_mask_failure_reason=effective.audit.get("segmentation_mask_failure_reason", ""),
                segmentation_provider=effective.audit.get("segmentation_provider", ""),
                segmentation_mask_ref=effective.audit.get("segmentation_mask_ref", ""),
                segmentation_text_pixels=effective.audit.get("segmentation_text_pixels"),
                segmentation_component_count=effective.audit.get("segmentation_component_count"),
                segmentation_binding_method=effective.audit.get("segmentation_binding_method", ""),
                segmentation_block_associations=effective.audit.get("segmentation_block_associations", []),
                ownership_binding_status=effective.audit.get("ownership_binding_status", ""),
                ownership_binding_method=effective.audit.get("ownership_binding_method", ""),
                cleanup_owned_unit_bbox=effective.audit.get("cleanup_owned_unit_bbox"),
                cleanup_owned_unit_mask_ref=effective.audit.get("cleanup_owned_unit_mask_ref", ""),
                protected_mask_ref=effective.audit.get("protected_mask_ref", ""),
                protected_overlap_pixels=effective.audit.get("protected_overlap_pixels"),
                segmentation_pixels_before_binding=effective.audit.get("segmentation_pixels_before_binding"),
                segmentation_pixels_after_owner_clip=effective.audit.get("segmentation_pixels_after_owner_clip"),
                segmentation_pixels_after_protection_subtract=effective.audit.get("segmentation_pixels_after_protection_subtract"),
                sourceglyph_overlap_pixels=effective.audit.get("sourceglyph_overlap_pixels"),
                sourceglyph_overlap_ratio=effective.audit.get("sourceglyph_overlap_ratio"),
                segmentation_outside_sourceglyph_pixels=effective.audit.get("segmentation_outside_sourceglyph_pixels"),
                effective_coverage_ratio=effective.audit.get("effective_coverage_ratio"),
                effective_coverage_status=effective.audit.get("effective_coverage_status", ""),
                component_ownership_status=effective.audit.get("component_ownership_status", ""),
                owned_component_ids=effective.audit.get("owned_component_ids", []),
                protected_component_ids=effective.audit.get("protected_component_ids", []),
                ambiguous_component_ids=effective.audit.get("ambiguous_component_ids", []),
                unowned_component_ids=effective.audit.get("unowned_component_ids", []),
                component_projection_method=effective.audit.get("component_projection_method", ""),
                owned_component_pixel_count=effective.audit.get("owned_component_pixel_count"),
                protected_component_pixel_count=effective.audit.get("protected_component_pixel_count"),
                ambiguous_component_pixel_count=effective.audit.get("ambiguous_component_pixel_count"),
                sourceglyph_overlap_component_ids=effective.audit.get("sourceglyph_overlap_component_ids", []),
                sourceglyph_missing_component_ids=effective.audit.get("sourceglyph_missing_component_ids", []),
                ownership_projection_failure_reason=effective.audit.get("ownership_projection_failure_reason", ""),
                effective_component_coverage_ratio=effective.audit.get("effective_component_coverage_ratio"),
                owned_segmentation_pixels=effective.audit.get("owned_segmentation_pixels"),
                executable_foreground_pixels=effective.audit.get("executable_foreground_pixels"),
                committed_cleanup_mask_pixels=effective.audit.get("committed_cleanup_mask_pixels"),
                owned_segmentation_to_executable_ratio=effective.audit.get("owned_segmentation_to_executable_ratio"),
                owned_segmentation_to_commit_ratio=effective.audit.get("owned_segmentation_to_commit_ratio"),
                ready_but_sparse_violation=bool(effective.audit.get("ready_but_sparse_violation", False)),
                sourceglyph_executable_influence_detected=bool(
                    effective.audit.get("sourceglyph_executable_influence_detected", False)
                ),
                dense_contract_override_detected=bool(effective.audit.get("dense_contract_override_detected", False)),
                foreground_mask=foreground.copy(),
                erase_mask=erase.copy(),
            )
            masks.append(cleanup_mask)
            masks_by_job_id.setdefault(str(job.cleanup_job_id), []).append(cleanup_mask)
            _cleanup_perf_contract_checkpoint(
                "cleanup_mask_job",
                "end",
                page_id=page_id,
                cleanup_job_id=str(job.cleanup_job_id),
                cleanup_mask_id=str(cleanup_mask.cleanup_mask_id),
                matched_evidence_count=len(matched),
                consumed_source_glyph_count=len(consumed_source_ids),
                consumed_source_glyph_mask_ids=consumed_source_ids,
                missing_source_glyph_count=0,
                foreground_pixels=foreground_pixels,
                erase_pixels=erase_pixels,
                allowed_area=allowed,
                visual_scope=visual_scope,
                growth_ratio=growth_ratio,
                elapsed_ms=round((time.time() - job_started) * 1000.0, 3),
            )
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")

    for record in rejected_records:
        _cleanup_perf_contract_checkpoint(
            "cleanup_mask_job",
            "rejected",
            page_id=page_id,
            cleanup_job_id=str(record.get("cleanup_job_id") or ""),
            region_id=str(record.get("region_id") or ""),
            reason=str(record.get("reason") or ""),
            required_source_glyph_mask_ids=record.get("required_source_glyph_mask_ids", []),
            missing_source_glyph_mask_ids=record.get("missing_source_glyph_mask_ids", []),
            foreground_pixels=record.get("foreground_mask_pixels", ""),
            erase_pixels=record.get("erase_mask_pixels", ""),
            allowed_area=record.get("allowed_area", ""),
            erase_mask_bbox=record.get("erase_mask_bbox", ""),
            growth_ratio=record.get("growth_ratio", ""),
        )
    for record in protected_records:
        _cleanup_perf_contract_checkpoint(
            "cleanup_mask_job",
            "protected",
            page_id=page_id,
            cleanup_job_id=str(record.get("cleanup_job_id") or ""),
            region_id=str(record.get("region_id") or ""),
            reason=str(record.get("reason") or ""),
        )
    _cleanup_perf_contract_checkpoint(
        "cleanup_mask_build",
        "module_end",
        page_id=page_id,
        mask_count=len(masks),
        rejected_count=len(rejected_records),
        protected_count=len(protected_records),
        skipped_count=len(skipped_records),
        error_count=len(errors),
        elapsed_ms=round((time.time() - started) * 1000.0, 3),
    )
    return CleanupMaskBuildResult(
        page_id=page_id,
        version=CLEANUP_MASK_CONTRACT_VERSION,
        masks=masks,
        masks_by_job_id=masks_by_job_id,
        rejected_records=rejected_records,
        protected_records=protected_records,
        skipped_records=skipped_records,
        errors=errors,
        text_foreground_segmentation=segmentation_audit,
    )


def _index_source_evidence(source_glyph_masks: Any) -> list[_SourceEvidence]:
    records: list[_SourceEvidence] = []

    masks_by_region = _get_mapping(source_glyph_masks, "masks_by_region")
    if masks_by_region:
        for region_id, value in masks_by_region.items():
            for item in _sequence_or_single(value):
                records.append(_source_evidence(str(region_id or ""), item))

    for key in ("source_glyph_masks", "source_glyph_mask_coverage_records", "coverage_records"):
        value = _get_value(source_glyph_masks, key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                evidence = _source_evidence("", item)
                if evidence.region_id or evidence.mask_id:
                    records.append(evidence)

    deduped: list[_SourceEvidence] = []
    seen: set[tuple[str, str, int]] = set()
    for record in records:
        key = (record.region_id, record.mask_id, id(record.raw))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def _source_evidence(region_id: str, raw: Any) -> _SourceEvidence:
    audit = _audit_mapping(raw)
    if region_id and "region_id" not in audit:
        audit = {"region_id": region_id, **audit}
    source_region_id = str(_first_present(audit, "region_id", "target_region_id", default=region_id) or "")
    mask_id = str(_first_present(audit, "source_glyph_mask_id", "mask_id", "id", default="") or "")
    return _SourceEvidence(region_id=source_region_id, mask_id=mask_id, audit=audit, raw=raw)


def _audit_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_audit_dict"):
        audit = value.to_audit_dict()
        if isinstance(audit, Mapping):
            return dict(audit)
    if is_dataclass(value):
        output: dict[str, Any] = {}
        for name in (
            "region_id",
            "mask_id",
            "source_glyph_mask_id",
            "generation_method",
            "foreground_mask_method",
            "foreground_mask_pixels",
            "foreground_mask_bbox",
            "erase_mask_method",
            "erase_mask_pixels",
            "erase_mask_bbox",
            "erase_mask_growth_ratio",
            "erase_mask_allowed_area",
            "cleanup_allowed_area",
            "cleanup_visual_artifact_risk",
            "erase_mask_artifact_risk",
        ):
            if hasattr(value, name):
                output[name] = getattr(value, name)
        return output
    return {}


def _matching_evidence(
    job: CleanupJob,
    evidence_by_id: Mapping[str, _SourceEvidence],
    evidence_by_region: Mapping[str, list[_SourceEvidence]],
) -> list[_SourceEvidence]:
    matches: list[_SourceEvidence] = []
    seen: set[tuple[str, str]] = set()
    for mask_id in getattr(job, "source_glyph_mask_ids", []) or []:
        evidence = evidence_by_id.get(str(mask_id))
        if evidence is None:
            continue
        key = (evidence.region_id, evidence.mask_id)
        if key not in seen:
            matches.append(evidence)
            seen.add(key)
    for region_id in getattr(job, "target_region_ids", []) or []:
        for evidence in evidence_by_region.get(str(region_id), []):
            key = (evidence.region_id, evidence.mask_id)
            if key not in seen:
                matches.append(evidence)
                seen.add(key)
    return matches


def _required_source_glyph_ids(job: CleanupJob) -> list[str]:
    ids: list[str] = []
    for attr in ("cleanup_unit_required_source_glyph_mask_ids", "source_glyph_mask_ids"):
        for item in getattr(job, attr, []) or []:
            text = str(item or "")
            if text and text not in ids:
                ids.append(text)
    return ids


def _cleanup_visual_scope_for_unit(
    *,
    required_source_ids: Sequence[str],
    consumed_source_ids: Sequence[str],
) -> str:
    ids: list[str] = []
    for item in list(required_source_ids or []) + list(consumed_source_ids or []):
        text = str(item or "")
        if text and text not in ids:
            ids.append(text)
    if len(ids) <= 1:
        return "source_glyph_local"
    return "source_glyph_union"


def _primary_source_evidence(job: CleanupJob, matched: Sequence[_SourceEvidence]) -> _SourceEvidence:
    anchor_region_id = str(getattr(job, "cleanup_unit_anchor_region_id", "") or "")
    for evidence in matched:
        if anchor_region_id and evidence.region_id == anchor_region_id:
            return evidence
    for evidence in matched:
        if evidence.mask_id in getattr(job, "source_glyph_mask_ids", []) or []:
            return evidence
    return matched[0]


def _allowed_area_from_evidence(matched: Sequence[_SourceEvidence]) -> list[int] | None:
    candidates: list[list[int]] = []
    for evidence in matched:
        bbox = _valid_bbox(
            _first_present(
                evidence.audit,
                "cleanup_allowed_area",
                "erase_mask_allowed_area",
                "allowed_cleanup_area",
                "text_area_container_bbox",
                "source_glyph_erasure_bbox",
                "source_glyph_bbox",
                "bbox",
                default=None,
            )
        )
        if bbox is not None and bbox not in candidates:
            candidates.append(bbox)
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        return _union_bboxes(candidates)
    return None


def _allowed_area_from_regions(job: CleanupJob, region_records: Mapping[str, Mapping[str, Any]]) -> list[int] | None:
    candidates: list[list[int]] = []
    for region_id in getattr(job, "target_region_ids", []) or []:
        region = region_records.get(str(region_id))
        bbox = _region_bbox(region) if isinstance(region, Mapping) else None
        if bbox is not None and bbox not in candidates:
            candidates.append(bbox)
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        return _union_bboxes(candidates)
    return None


def _union_masks_from_evidence(
    matched: Sequence[_SourceEvidence],
    *,
    allowed: list[int],
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    list[str],
    list[str],
    list[str],
    list[str],
    bool,
]:
    foreground_union: np.ndarray | None = None
    erase_union: np.ndarray | None = None
    foreground_source_keys: list[str] = []
    erase_source_keys: list[str] = []
    consumed_source_ids: list[str] = []
    missing_foreground_source_ids: list[str] = []
    used_transitional_erase = False

    for evidence in matched:
        foreground, foreground_source = _binary_mask_from_evidence(
            evidence,
            keys=("foreground_mask", "mask"),
        )
        if foreground is None:
            missing_foreground_source_ids.append(evidence.mask_id or evidence.region_id or "unknown_source_glyph")
            continue
        if foreground_union is None:
            foreground_union = np.zeros_like(foreground, dtype=np.uint8)
            erase_union = np.zeros_like(foreground, dtype=np.uint8)
        else:
            foreground = _coerce_mask_shape(foreground, foreground_union.shape)
        foreground = _clip_mask_to_bbox(foreground, allowed)
        foreground_union = np.maximum(foreground_union, foreground)
        if foreground_source and foreground_source not in foreground_source_keys:
            foreground_source_keys.append(foreground_source)

        erase, erase_source = _binary_mask_from_evidence(evidence, keys=("erase_mask",))
        if erase is None:
            erase = foreground.copy()
            erase_source = foreground_source
        else:
            erase = _coerce_mask_shape(erase, foreground_union.shape)
            used_transitional_erase = True
        erase = _clip_mask_to_bbox(erase, allowed)
        erase_union = np.maximum(erase_union, erase)
        if erase_source and erase_source not in erase_source_keys:
            erase_source_keys.append(erase_source)
        if evidence.mask_id and evidence.mask_id not in consumed_source_ids:
            consumed_source_ids.append(evidence.mask_id)

    return (
        foreground_union,
        erase_union,
        foreground_source_keys,
        erase_source_keys,
        consumed_source_ids,
        missing_foreground_source_ids,
        used_transitional_erase,
    )


def _build_component_projected_text_mask(
    *,
    job: CleanupJob,
    segmentation_mask: np.ndarray | None,
    segmentation_audit: Mapping[str, Any],
    seed_foreground: np.ndarray | None,
    allowed: list[int],
    matched: Sequence[_SourceEvidence],
    ownership_binding: _OwnershipBinding,
    component_projection: _ComponentOwnershipProjection,
    missing_source_glyph_mask_ids: Sequence[str],
) -> _EffectiveMaskBuild:
    base_audit = {
        "segmentation_provider": segmentation_audit.get("segmentation_provider") or segmentation_audit.get("provider", ""),
        "segmentation_mask_ref": segmentation_audit.get("segmentation_mask_ref") or segmentation_audit.get("refined_mask_ref", ""),
        "segmentation_text_pixels": segmentation_audit.get("segmentation_text_pixels") or segmentation_audit.get("text_pixel_count", 0),
        "segmentation_component_count": len(component_projection.components),
        "segmentation_block_associations": [],
        "component_projection_method": "segmentation_component_ownership_projection",
    }
    base_audit.update(_ownership_base_audit(ownership_binding))
    if segmentation_mask is None or component_projection.labels is None:
        audit = {
            **base_audit,
            "seed_foreground_pixels": int(np.count_nonzero(seed_foreground > 0)) if seed_foreground is not None else 0,
            "completed_foreground_pixels": 0,
            "component_count_before": 0,
            "component_count_after": 0,
            "largest_component_pixels_before": 0,
            "largest_component_pixels_after": 0,
            "text_block_coverage_estimate": 0.0,
            "bbox_fill_ratio_before": 0.0,
            "bbox_fill_ratio_after": 0.0,
            "analysis_scope_bbox": allowed,
            "executable_erase_bbox": None,
            "mask_completion_method": "segmentation_component_projection_missing",
            "polarity_mode": "segmentation",
            "recovered_component_count": 0,
            "rejected_component_count": 0,
            "rejected_component_reasons": ["segmentation_mask_missing"],
            "segmentation_mask_status": "segmentation_mask_missing",
            "segmentation_mask_failure_reason": str(
                segmentation_audit.get("segmentation_mask_failure_reason") or "text_foreground_segmentation_missing"
            ),
            "segmentation_binding_method": "not_attempted",
            "component_ownership_status": "cleanup_mask_structural_invalid",
            "ownership_projection_failure_reason": "segmentation_mask_missing",
            "owned_component_ids": [],
            "protected_component_ids": [],
            "ambiguous_component_ids": [],
            "unowned_component_ids": [],
            "owned_component_pixel_count": 0,
            "protected_component_pixel_count": 0,
            "ambiguous_component_pixel_count": 0,
            "sourceglyph_overlap_component_ids": [],
            "sourceglyph_missing_component_ids": [],
            "effective_component_coverage_ratio": 0.0,
        }
        return _EffectiveMaskBuild(
            foreground=None,
            erase=None,
            status="cleanup_mask_structural_invalid",
            failure_reason="segmentation_mask_missing",
            audit=audit,
            rejected=True,
        )

    projection = _component_projection_for_job(
        job=job,
        allowed=allowed,
        component_projection=component_projection,
    )
    owned_ids = projection["owned_component_ids"]
    protected_ids = projection["protected_component_ids"]
    ambiguous_ids = projection["ambiguous_component_ids"]
    unowned_ids = projection["unowned_component_ids"]
    foreground = _component_mask_from_ids(component_projection, owned_ids)
    if foreground is not None:
        foreground = _clip_mask_to_bbox(foreground, allowed)
    foreground_pixels = int(np.count_nonzero(foreground > 0)) if foreground is not None else 0
    owned_segmentation_pixels = int(projection["owned_component_pixel_count"])
    owned_to_executable_ratio = round(
        float(foreground_pixels) / float(max(1, owned_segmentation_pixels)),
        4,
    ) if owned_segmentation_pixels > 0 else 0.0
    before_binding = int(np.count_nonzero(_clip_mask_to_bbox((segmentation_mask > 0).astype(np.uint8), allowed)))
    bbox = _mask_bbox(foreground) if foreground is not None else None
    sourceglyph_overlap, sourceglyph_ratio, segmentation_outside_sourceglyph = _sourceglyph_overlap_metrics(
        foreground=foreground if foreground is not None else np.zeros_like(segmentation_mask, dtype=np.uint8),
        seed=seed_foreground,
    )
    sourceglyph_overlap_ids = [
        component_id
        for component_id in owned_ids
        if _component_by_id(component_projection, component_id).get("sourceglyph_overlap_pixels", 0)
    ]
    sourceglyph_missing_ids = [
        component_id
        for component_id in owned_ids
        if not _component_by_id(component_projection, component_id).get("sourceglyph_overlap_pixels", 0)
    ]
    coverage_ratio = _text_block_coverage_estimate(bbox, allowed) if bbox is not None else 0.0
    stats = _component_stats(foreground) if foreground is not None else {"component_count": 0, "largest_component_pixels": 0}
    status = "cleanup_mask_ready_from_owned_segmentation_components"
    failure_reason = ""
    rejected = False
    if not owned_ids or foreground is None or foreground_pixels <= 0:
        if ambiguous_ids:
            status = "cleanup_mask_unresolved_ambiguous_components"
            failure_reason = "cleanup_mask_unresolved_ambiguous_components"
        elif protected_ids:
            status = "cleanup_mask_unresolved_protected_overlap"
            failure_reason = "cleanup_mask_unresolved_protected_overlap"
        elif unowned_ids:
            status = "cleanup_mask_unresolved_unowned_visible_text"
            failure_reason = "cleanup_mask_unresolved_unowned_visible_text"
        else:
            status = "cleanup_mask_missing_no_owned_segmentation_components"
            failure_reason = "cleanup_mask_missing_no_owned_segmentation_components"
        rejected = True
    else:
        coverage_reason = _segmentation_effective_coverage_reason(
            foreground=foreground,
            analysis_scope=allowed,
            job=job,
            protected_overlap_pixels=int(projection["protected_component_pixel_count"]),
            owner_pixels=max(1, int(projection["owned_component_pixel_count"]) + int(projection["protected_component_pixel_count"])),
        )
        owned_ratio_reason = _owned_segmentation_executable_coverage_reason(
            ratio=owned_to_executable_ratio,
            owned_pixels=owned_segmentation_pixels,
            foreground_pixels=foreground_pixels,
            job=job,
        )
        unsafe_reason = _segmentation_foreground_unsafe_reason(foreground, allowed, job)
        if coverage_reason or owned_ratio_reason or unsafe_reason or ambiguous_ids or unowned_ids:
            status = "cleanup_mask_partial_owned_components"
            failure_reason = coverage_reason or owned_ratio_reason or unsafe_reason or "cleanup_mask_partial_owned_components"
            rejected = bool(coverage_reason or owned_ratio_reason or unsafe_reason)

    seed_pixels = int(np.count_nonzero(seed_foreground > 0)) if seed_foreground is not None else 0
    erase = None
    if foreground is not None and foreground_pixels > 0:
        erase = _effective_erase_from_foreground(foreground, allowed, job)
    sourceglyph_executable_influence_detected = False
    ready_but_sparse_violation = bool(
        status == "cleanup_mask_ready_from_owned_segmentation_components"
        and _owned_segmentation_executable_coverage_reason(
            ratio=owned_to_executable_ratio,
            owned_pixels=owned_segmentation_pixels,
            foreground_pixels=foreground_pixels,
            job=job,
        )
    )
    audit = {
        **base_audit,
        "seed_foreground_pixels": seed_pixels,
        "completed_foreground_pixels": foreground_pixels,
        "component_count_before": len(component_projection.components),
        "component_count_after": len(owned_ids),
        "largest_component_pixels_before": _largest_component_pixels(component_projection.components),
        "largest_component_pixels_after": int(stats.get("largest_component_pixels") or 0),
        "text_block_coverage_estimate": coverage_ratio,
        "bbox_fill_ratio_before": 0.0,
        "bbox_fill_ratio_after": _mask_fill_ratio(foreground) if foreground is not None else 0.0,
        "analysis_scope_bbox": allowed,
        "executable_erase_bbox": _mask_bbox(erase) if erase is not None else None,
        "mask_completion_method": "text_foreground_segmentation_component_ownership_projection",
        "polarity_mode": "segmentation",
        "source_seed_mask_ids": [source.mask_id for source in matched if source.mask_id],
        "recovered_component_count": len(owned_ids),
        "rejected_component_count": len(protected_ids) + len(ambiguous_ids) + len(unowned_ids),
        "rejected_component_reasons": projection["unresolved_reasons"],
        "segmentation_mask_status": "segmentation_mask_ready" if segmentation_mask is not None else "segmentation_mask_missing",
        "segmentation_mask_failure_reason": failure_reason if status != "cleanup_mask_ready_from_owned_segmentation_components" else "",
        "segmentation_binding_method": "segmentation_component_ownership_projection",
        "segmentation_pixels_before_binding": before_binding,
        "segmentation_pixels_after_owner_clip": owned_segmentation_pixels,
        "segmentation_pixels_after_protection_subtract": foreground_pixels,
        "protected_overlap_pixels": int(projection["protected_component_pixel_count"]),
        "sourceglyph_overlap_pixels": sourceglyph_overlap,
        "sourceglyph_overlap_ratio": sourceglyph_ratio,
        "segmentation_outside_sourceglyph_pixels": segmentation_outside_sourceglyph,
        "effective_coverage_ratio": coverage_ratio,
        "effective_coverage_status": "effective_coverage_ready" if not rejected else failure_reason,
        "component_ownership_status": status,
        "owned_component_ids": owned_ids,
        "protected_component_ids": protected_ids,
        "ambiguous_component_ids": ambiguous_ids,
        "unowned_component_ids": unowned_ids,
        "owned_component_pixel_count": owned_segmentation_pixels,
        "protected_component_pixel_count": int(projection["protected_component_pixel_count"]),
        "ambiguous_component_pixel_count": int(projection["ambiguous_component_pixel_count"]),
        "sourceglyph_overlap_component_ids": sourceglyph_overlap_ids,
        "sourceglyph_missing_component_ids": sourceglyph_missing_ids,
        "ownership_projection_failure_reason": failure_reason,
        "effective_component_coverage_ratio": coverage_ratio,
        "missing_source_glyph_mask_ids": list(missing_source_glyph_mask_ids or []),
        "owned_segmentation_pixels": owned_segmentation_pixels,
        "executable_foreground_pixels": foreground_pixels,
        "committed_cleanup_mask_pixels": None,
        "owned_segmentation_to_executable_ratio": owned_to_executable_ratio,
        "owned_segmentation_to_commit_ratio": None,
        "ready_but_sparse_violation": ready_but_sparse_violation,
        "sourceglyph_executable_influence_detected": sourceglyph_executable_influence_detected,
        "dense_contract_override_detected": False,
    }
    return _EffectiveMaskBuild(
        foreground=foreground,
        erase=erase,
        status=status,
        failure_reason=failure_reason,
        audit=audit,
        rejected=rejected,
    )


def _build_segmentation_text_mask(
    *,
    job: CleanupJob,
    segmentation_mask: np.ndarray | None,
    segmentation_audit: Mapping[str, Any],
    seed_foreground: np.ndarray | None,
    allowed: list[int],
    matched: Sequence[_SourceEvidence],
    ownership_binding: _OwnershipBinding | None = None,
) -> _EffectiveMaskBuild:
    seed = (
        _clip_mask_to_bbox((seed_foreground > 0).astype(np.uint8), allowed)
        if seed_foreground is not None
        else None
    )
    seed_pixels = int(np.count_nonzero(seed)) if seed is not None else 0
    seed_stats = _component_stats(seed) if seed is not None else {"component_count": 0, "largest_component_pixels": 0}
    base_audit = {
        "seed_foreground_pixels": seed_pixels,
        "component_count_before": seed_stats["component_count"],
        "largest_component_pixels_before": seed_stats["largest_component_pixels"],
        "bbox_fill_ratio_before": _mask_fill_ratio(seed) if seed is not None else 0.0,
        "source_seed_mask_ids": [source.mask_id for source in matched if source.mask_id],
        "segmentation_provider": segmentation_audit.get("segmentation_provider") or segmentation_audit.get("provider", ""),
        "segmentation_mask_ref": segmentation_audit.get("segmentation_mask_ref") or segmentation_audit.get("refined_mask_ref", ""),
        "segmentation_text_pixels": segmentation_audit.get("segmentation_text_pixels") or segmentation_audit.get("text_pixel_count", 0),
        "segmentation_component_count": segmentation_audit.get("segmentation_component_count"),
        "segmentation_block_associations": [],
    }
    if ownership_binding is None:
        fallback_shape = segmentation_mask.shape if segmentation_mask is not None else (seed.shape if seed is not None else None)
        ownership_binding = _build_ownership_binding(
            job=job,
            allowed=allowed,
            shape=fallback_shape,
            region_records={},
        )
    base_audit.update(_ownership_base_audit(ownership_binding))
    if segmentation_mask is None:
        audit = {
            **base_audit,
            "completed_foreground_pixels": 0,
            "component_count_after": 0,
            "largest_component_pixels_after": 0,
            "text_block_coverage_estimate": 0.0,
            "bbox_fill_ratio_after": 0.0,
            "analysis_scope_bbox": allowed,
            "executable_erase_bbox": None,
            "mask_completion_method": "segmentation_mask_missing",
            "polarity_mode": "segmentation",
            "recovered_component_count": 0,
            "rejected_component_count": 1,
            "rejected_component_reasons": ["segmentation_mask_missing"],
            "segmentation_mask_status": "segmentation_mask_missing",
            "segmentation_mask_failure_reason": str(
                segmentation_audit.get("segmentation_mask_failure_reason") or "text_foreground_segmentation_missing"
            ),
            "segmentation_binding_method": "not_attempted",
        }
        return _EffectiveMaskBuild(
            foreground=None,
            erase=None,
            status="segmentation_mask_missing",
            failure_reason="segmentation_mask_missing",
            audit=audit,
            rejected=True,
        )
    segmentation_binary = (segmentation_mask > 0).astype(np.uint8)
    analysis_scope = ownership_binding.owned_bbox or allowed
    before_binding = int(np.count_nonzero(_clip_mask_to_bbox(segmentation_binary, allowed)))
    owner_mask = ownership_binding.owner_mask
    if owner_mask is None:
        owner_mask = _bbox_mask(segmentation_binary.shape, allowed)
    foreground_owner = np.where(owner_mask > 0, segmentation_binary, 0).astype(np.uint8)
    after_owner = int(np.count_nonzero(foreground_owner))
    protected_mask = ownership_binding.protected_mask
    protected_overlap = int(
        np.count_nonzero((foreground_owner > 0) & (protected_mask > 0))
    ) if protected_mask is not None else 0
    if protected_mask is not None:
        foreground = np.where(protected_mask > 0, 0, foreground_owner).astype(np.uint8)
    else:
        foreground = foreground_owner
    after_protection = int(np.count_nonzero(foreground))
    sourceglyph_overlap, sourceglyph_ratio, segmentation_outside_sourceglyph = _sourceglyph_overlap_metrics(
        foreground=foreground,
        seed=seed,
    )
    binding_method = (
        "segmentation_cleanup_ownership_mask_clip_protected_subtract"
        if protected_mask is not None and protected_overlap > 0
        else "segmentation_cleanup_ownership_mask_clip"
    )
    # Ownership binding must not synthesize executable foreground pixels; grouping
    # remains a downstream accounting concern, while erase dilation stays bounded.
    foreground = _clip_mask_to_bbox(foreground, analysis_scope)
    pixels = int(np.count_nonzero(foreground))
    if pixels <= 0:
        audit = {
            **base_audit,
            "completed_foreground_pixels": 0,
            "component_count_after": 0,
            "largest_component_pixels_after": 0,
            "text_block_coverage_estimate": 0.0,
            "bbox_fill_ratio_after": 0.0,
            "analysis_scope_bbox": analysis_scope,
            "executable_erase_bbox": None,
            "mask_completion_method": "text_foreground_segmentation_clipped_empty",
            "polarity_mode": "segmentation",
            "recovered_component_count": 0,
            "rejected_component_count": 1,
            "rejected_component_reasons": ["effective_mask_no_segmentation_in_owned_area"],
            "segmentation_mask_status": "cleanup_mask_unresolved_after_segmentation",
            "segmentation_mask_failure_reason": "effective_mask_no_segmentation_in_owned_area",
            "segmentation_binding_method": binding_method,
            "segmentation_pixels_before_binding": before_binding,
            "segmentation_pixels_after_owner_clip": after_owner,
            "segmentation_pixels_after_protection_subtract": after_protection,
            "protected_overlap_pixels": protected_overlap,
            "sourceglyph_overlap_pixels": sourceglyph_overlap,
            "sourceglyph_overlap_ratio": sourceglyph_ratio,
            "segmentation_outside_sourceglyph_pixels": segmentation_outside_sourceglyph,
            "effective_coverage_ratio": 0.0,
            "effective_coverage_status": "effective_mask_no_segmentation_in_owned_area",
        }
        return _EffectiveMaskBuild(
            foreground=None,
            erase=None,
            status="effective_mask_no_segmentation_in_owned_area",
            failure_reason="effective_mask_no_segmentation_in_owned_area",
            audit=audit,
            rejected=True,
        )
    coverage_reason = _segmentation_effective_coverage_reason(
        foreground=foreground,
        analysis_scope=analysis_scope,
        job=job,
        protected_overlap_pixels=protected_overlap,
        owner_pixels=after_owner,
    )
    unsafe_reason = _segmentation_foreground_unsafe_reason(foreground, allowed, job)
    if coverage_reason or unsafe_reason:
        rejected_reason = coverage_reason or unsafe_reason
        audit = _effective_audit(
            seed=seed if seed is not None else np.zeros_like(foreground, dtype=np.uint8),
            completed=foreground,
            erase=foreground,
            allowed=allowed,
            analysis_scope=analysis_scope,
            method="text_foreground_segmentation_rejected",
            polarity="segmentation",
            rejected_reasons=[rejected_reason],
            recovered_count=0,
        )
        return _EffectiveMaskBuild(
            foreground=foreground,
            erase=foreground.copy(),
            status=rejected_reason,
            failure_reason=rejected_reason,
            audit={
                **base_audit,
                **audit,
                "segmentation_mask_status": "cleanup_mask_unresolved_after_segmentation",
                "segmentation_mask_failure_reason": rejected_reason,
                "segmentation_binding_method": f"{binding_method}_coverage_rejected",
                "segmentation_block_associations": _segmentation_blocks_for_scope(segmentation_audit, analysis_scope),
                "segmentation_pixels_before_binding": before_binding,
                "segmentation_pixels_after_owner_clip": after_owner,
                "segmentation_pixels_after_protection_subtract": after_protection,
                "protected_overlap_pixels": protected_overlap,
                "sourceglyph_overlap_pixels": sourceglyph_overlap,
                "sourceglyph_overlap_ratio": sourceglyph_ratio,
                "segmentation_outside_sourceglyph_pixels": segmentation_outside_sourceglyph,
                "effective_coverage_ratio": audit.get("text_block_coverage_estimate"),
                "effective_coverage_status": rejected_reason,
            },
            rejected=True,
        )
    erase = _effective_erase_from_foreground(foreground, allowed, job)
    if protected_mask is not None:
        erase = np.where(protected_mask > 0, 0, erase).astype(np.uint8)
    audit = _effective_audit(
        seed=seed if seed is not None else np.zeros_like(foreground, dtype=np.uint8),
        completed=foreground,
        erase=erase,
        allowed=allowed,
        analysis_scope=analysis_scope,
        method="text_foreground_segmentation_owned_foreground_clip",
        polarity="segmentation",
        rejected_reasons=[],
        recovered_count=0,
    )
    return _EffectiveMaskBuild(
        foreground=foreground,
        erase=erase,
        status="cleanup_mask_ready_from_segmentation",
        failure_reason="",
        audit={
            **base_audit,
            **audit,
            "segmentation_mask_status": "segmentation_mask_ready",
            "segmentation_mask_failure_reason": "",
            "segmentation_binding_method": f"{binding_method}_sourceglyph_provenance_only",
            "segmentation_block_associations": _segmentation_blocks_for_scope(segmentation_audit, analysis_scope),
            "segmentation_pixels_before_binding": before_binding,
            "segmentation_pixels_after_owner_clip": after_owner,
            "segmentation_pixels_after_protection_subtract": after_protection,
            "protected_overlap_pixels": protected_overlap,
            "sourceglyph_overlap_pixels": sourceglyph_overlap,
            "sourceglyph_overlap_ratio": sourceglyph_ratio,
            "segmentation_outside_sourceglyph_pixels": segmentation_outside_sourceglyph,
            "effective_coverage_ratio": audit.get("text_block_coverage_estimate"),
            "effective_coverage_status": "effective_coverage_ready",
            "owned_segmentation_pixels": after_owner,
            "executable_foreground_pixels": after_protection,
            "committed_cleanup_mask_pixels": None,
            "owned_segmentation_to_executable_ratio": round(float(after_protection) / float(max(1, after_owner)), 4),
            "owned_segmentation_to_commit_ratio": None,
            "ready_but_sparse_violation": False,
            "sourceglyph_executable_influence_detected": False,
            "dense_contract_override_detected": False,
        },
        rejected=False,
    )


def _build_ownership_binding(
    *,
    job: CleanupJob,
    allowed: list[int],
    shape: tuple[int, int] | None,
    region_records: Mapping[str, Mapping[str, Any]] | None,
) -> _OwnershipBinding:
    if shape is None:
        return _OwnershipBinding(
            owner_mask=None,
            protected_mask=None,
            owned_bbox=allowed,
            protected_bbox=None,
            method="ownership_binding_shape_unavailable",
            status="ownership_binding_unresolved_invalid_mask_shape",
            protected_records=[],
        )
    owner_mask = _bbox_mask(shape, allowed)
    protected_mask = np.zeros(shape, dtype=np.uint8)
    protected_records: list[dict[str, Any]] = []
    target_ids = {str(item or "") for item in getattr(job, "target_region_ids", []) or [] if str(item or "")}
    for region_id, region in (region_records or {}).items():
        reason = _region_protection_reason(region)
        if not reason:
            continue
        bbox = _region_bbox(region)
        if bbox is None:
            continue
        region_mask = _bbox_mask(shape, bbox)
        if not np.any((region_mask > 0) & (owner_mask > 0)):
            continue
        protected_mask = np.maximum(protected_mask, region_mask)
        protected_records.append(
            {
                "region_id": region_id,
                "is_target_region": region_id in target_ids,
                "bbox": bbox,
                "reason": reason,
            }
        )
    if not np.any(protected_mask > 0):
        protected_mask_out: np.ndarray | None = None
        protected_bbox = None
    else:
        protected_mask_out = protected_mask
        protected_bbox = _mask_bbox(protected_mask)
    return _OwnershipBinding(
        owner_mask=owner_mask,
        protected_mask=protected_mask_out,
        owned_bbox=allowed,
        protected_bbox=protected_bbox,
        method="cleanup_job_allowed_area_ownership_scope_with_region_protection_subtract",
        status="ownership_binding_ready",
        protected_records=protected_records,
    )


def _ownership_base_audit(binding: _OwnershipBinding) -> dict[str, Any]:
    return {
        "ownership_binding_status": binding.status,
        "ownership_binding_method": binding.method,
        "cleanup_owned_unit_bbox": binding.owned_bbox,
        "cleanup_owned_unit_mask_ref": "",
        "protected_mask_ref": "",
        "protected_overlap_pixels": 0,
        "protected_region_records": list(binding.protected_records or [])[:40],
        "segmentation_pixels_before_binding": 0,
        "segmentation_pixels_after_owner_clip": 0,
        "segmentation_pixels_after_protection_subtract": 0,
        "sourceglyph_overlap_pixels": 0,
        "sourceglyph_overlap_ratio": 0.0,
        "segmentation_outside_sourceglyph_pixels": 0,
        "effective_coverage_ratio": 0.0,
        "effective_coverage_status": "",
    }


def _build_component_ownership_projection(
    *,
    page_id: str,
    segmentation_mask: np.ndarray | None,
    job_candidates: Sequence[CleanupJob],
    region_records: Mapping[str, Mapping[str, Any]],
    evidence_records: Sequence[_SourceEvidence],
) -> _ComponentOwnershipProjection:
    if segmentation_mask is None:
        return _ComponentOwnershipProjection(labels=None, components=[], component_label_by_id={})
    binary = (segmentation_mask > 0).astype(np.uint8)
    if int(np.count_nonzero(binary)) <= 0:
        return _ComponentOwnershipProjection(labels=np.zeros_like(binary, dtype=np.int32), components=[], component_label_by_id={})
    if cv2 is not None:
        label_count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    else:
        label_count = 2
        labels = np.where(binary > 0, 1, 0).astype(np.int32)
        ys, xs = np.where(binary > 0)
        stats = np.array([[0, 0, 0, 0, 0], [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1), int(xs.size)]])
        centroids = np.array([[0.0, 0.0], [float(xs.mean()), float(ys.mean())]])
    shape = binary.shape
    job_scopes = _job_ownership_scope_masks(job_candidates, region_records, shape)
    protected_scopes = _protected_region_scope_masks(region_records, shape)
    sourceglyph_masks = _sourceglyph_projection_masks(evidence_records, shape)
    components: list[dict[str, Any]] = []
    label_by_id: dict[str, int] = {}
    for label in range(1, int(label_count)):
        pixel_count = int(stats[label, cv2.CC_STAT_AREA] if cv2 is not None else stats[label, 4])
        if pixel_count <= 0:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT] if cv2 is not None else stats[label, 0])
        y = int(stats[label, cv2.CC_STAT_TOP] if cv2 is not None else stats[label, 1])
        w = int(stats[label, cv2.CC_STAT_WIDTH] if cv2 is not None else stats[label, 2])
        h = int(stats[label, cv2.CC_STAT_HEIGHT] if cv2 is not None else stats[label, 3])
        centroid_x = float(centroids[label][0])
        centroid_y = float(centroids[label][1])
        component_id = f"segcomp_{_safe_id(page_id)}_{len(components):04d}"
        component_mask = labels == label
        owner_candidates = _component_owner_candidates(component_mask, pixel_count, job_scopes)
        protected_candidates = _component_protected_candidates(
            component_mask=component_mask,
            pixel_count=pixel_count,
            centroid=(centroid_x, centroid_y),
            protected_scopes=protected_scopes,
        )
        sourceglyph_overlap_pixels, sourceglyph_ids = _component_sourceglyph_overlap(component_mask, sourceglyph_masks)
        ownership_state, owner_job_id, ambiguity_reasons = _component_ownership_state(
            owner_candidates=owner_candidates,
            protected_candidates=protected_candidates,
            pixel_count=pixel_count,
        )
        protected_reason = _protected_component_reason(protected_candidates)
        if ownership_state == "protected_sfx_decorative" and protected_reason:
            ambiguity_reasons = [protected_reason] + [item for item in ambiguity_reasons if item != protected_reason]
        scope_job_ids = [item["cleanup_job_id"] for item in owner_candidates if item.get("overlap_pixels", 0) > 0]
        components.append(
            {
                "component_id": component_id,
                "label": int(label),
                "ownership_state": ownership_state,
                "owner_cleanup_job_id": owner_job_id,
                "owner_region_ids": _component_owner_region_ids(owner_candidates, owner_job_id),
                "scope_cleanup_job_ids": scope_job_ids,
                "candidate_cleanup_job_ids": [item["cleanup_job_id"] for item in owner_candidates],
                "protected_region_ids": [item["region_id"] for item in protected_candidates],
                "protected_reason": protected_reason,
                "bbox": [x, y, x + w, y + h],
                "pixel_count": pixel_count,
                "centroid": [round(centroid_x, 3), round(centroid_y, 3)],
                "owner_overlap_pixels": _best_overlap(owner_candidates),
                "owner_overlap_ratio": round(_best_overlap(owner_candidates) / float(max(1, pixel_count)), 4),
                "protected_overlap_pixels": _best_overlap(protected_candidates),
                "protected_overlap_ratio": round(_best_overlap(protected_candidates) / float(max(1, pixel_count)), 4),
                "sourceglyph_overlap_pixels": sourceglyph_overlap_pixels,
                "sourceglyph_overlap_ratio": round(sourceglyph_overlap_pixels / float(max(1, pixel_count)), 4),
                "sourceglyph_overlap_ids": sourceglyph_ids,
                "sourceglyph_missing": not bool(sourceglyph_ids),
                "ambiguity_reasons": ambiguity_reasons,
            }
        )
        label_by_id[component_id] = int(label)
    return _ComponentOwnershipProjection(labels=labels.astype(np.int32), components=components, component_label_by_id=label_by_id)


def _job_ownership_scope_masks(
    job_candidates: Sequence[CleanupJob],
    region_records: Mapping[str, Mapping[str, Any]],
    shape: tuple[int, int],
) -> list[dict[str, Any]]:
    scopes: list[dict[str, Any]] = []
    for job in job_candidates or []:
        if _job_protection_reason(job):
            continue
        job_id = str(getattr(job, "cleanup_job_id", "") or "")
        if not job_id:
            continue
        mask = np.zeros(shape, dtype=np.uint8)
        region_ids: list[str] = []
        for region_id in getattr(job, "target_region_ids", []) or []:
            region = region_records.get(str(region_id))
            bbox = _region_bbox(region) if isinstance(region, Mapping) else None
            if bbox is None:
                continue
            mask = np.maximum(mask, _bbox_mask(shape, bbox))
            region_ids.append(str(region_id))
        if int(np.count_nonzero(mask)) <= 0:
            allowed = _valid_bbox(getattr(job, "allowed_cleanup_area", None))
            if allowed is not None:
                mask = np.maximum(mask, _bbox_mask(shape, allowed))
        scopes.append({"cleanup_job_id": job_id, "region_ids": region_ids, "mask": mask, "job": job})
    return scopes


def _protected_region_scope_masks(
    region_records: Mapping[str, Mapping[str, Any]],
    shape: tuple[int, int],
) -> list[dict[str, Any]]:
    scopes: list[dict[str, Any]] = []
    for region_id, region in (region_records or {}).items():
        reason = _region_protection_reason(region)
        if not reason:
            continue
        bbox = _region_bbox(region)
        if bbox is None:
            continue
        scopes.append(
            {
                "region_id": str(region_id),
                "reason": reason,
                "bbox": bbox,
                "mask": _bbox_mask(shape, bbox),
            }
        )
    return scopes


def _sourceglyph_projection_masks(
    evidence_records: Sequence[_SourceEvidence],
    shape: tuple[int, int],
) -> list[dict[str, Any]]:
    masks: list[dict[str, Any]] = []
    for evidence in evidence_records or []:
        foreground, _source = _binary_mask_from_evidence(evidence, keys=("foreground_mask", "mask"))
        if foreground is None:
            continue
        foreground = _coerce_mask_shape(foreground, shape)
        if int(np.count_nonzero(foreground)) <= 0:
            continue
        masks.append(
            {
                "source_glyph_mask_id": evidence.mask_id or evidence.region_id or "unknown_sourceglyph",
                "mask": foreground.astype(np.uint8),
            }
        )
    return masks


def _component_owner_candidates(
    component_mask: np.ndarray,
    pixel_count: int,
    job_scopes: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    min_pixels = max(COMPONENT_OWNER_MIN_PIXELS, int(pixel_count * COMPONENT_OWNER_MIN_RATIO))
    for scope in job_scopes:
        mask = scope.get("mask")
        if mask is None:
            continue
        overlap = int(np.count_nonzero(component_mask & (mask > 0)))
        if overlap <= 0:
            continue
        strong = overlap >= COMPONENT_OWNER_STRONG_PIXELS
        eligible = overlap >= min_pixels or strong
        candidates.append(
            {
                "cleanup_job_id": str(scope.get("cleanup_job_id") or ""),
                "region_ids": list(scope.get("region_ids") or []),
                "overlap_pixels": overlap,
                "overlap_ratio": overlap / float(max(1, pixel_count)),
                "eligible": eligible,
            }
        )
    candidates.sort(key=lambda item: int(item.get("overlap_pixels") or 0), reverse=True)
    return candidates


def _component_protected_candidates(
    *,
    component_mask: np.ndarray,
    pixel_count: int,
    centroid: tuple[float, float],
    protected_scopes: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for scope in protected_scopes:
        mask = scope.get("mask")
        bbox = scope.get("bbox")
        if mask is None:
            continue
        overlap = int(np.count_nonzero(component_mask & (mask > 0)))
        centroid_inside = bool(_point_in_bbox(centroid, bbox)) if isinstance(bbox, Sequence) else False
        eligible = overlap / float(max(1, pixel_count)) >= COMPONENT_PROTECTED_MIN_RATIO or centroid_inside
        if overlap <= 0 and not centroid_inside:
            continue
        candidates.append(
            {
                "region_id": str(scope.get("region_id") or ""),
                "reason": str(scope.get("reason") or "protected"),
                "overlap_pixels": overlap,
                "overlap_ratio": overlap / float(max(1, pixel_count)),
                "centroid_inside": centroid_inside,
                "eligible": eligible,
            }
        )
    candidates.sort(key=lambda item: int(item.get("overlap_pixels") or 0), reverse=True)
    return candidates


def _component_sourceglyph_overlap(component_mask: np.ndarray, sourceglyph_masks: Sequence[dict[str, Any]]) -> tuple[int, list[str]]:
    total = 0
    ids: list[str] = []
    for item in sourceglyph_masks:
        mask = item.get("mask")
        if mask is None:
            continue
        overlap = int(np.count_nonzero(component_mask & (mask > 0)))
        if overlap <= 0:
            continue
        total += overlap
        mask_id = str(item.get("source_glyph_mask_id") or "")
        if mask_id and mask_id not in ids:
            ids.append(mask_id)
    return total, ids


def _component_ownership_state(
    *,
    owner_candidates: Sequence[dict[str, Any]],
    protected_candidates: Sequence[dict[str, Any]],
    pixel_count: int,
) -> tuple[str, str, list[str]]:
    eligible_owners = [item for item in owner_candidates if item.get("eligible")]
    eligible_protected = [item for item in protected_candidates if item.get("eligible")]
    ambiguity: list[str] = []
    best_owner = eligible_owners[0] if eligible_owners else None
    best_protected = eligible_protected[0] if eligible_protected else None
    if best_owner and best_protected:
        protected_ratio = float(best_protected.get("overlap_pixels") or 0) / float(max(1, pixel_count))
        if protected_ratio >= COMPONENT_PROTECTED_DOMINANT_RATIO or bool(best_protected.get("centroid_inside")):
            return _protected_ownership_state(best_protected), "", [str(best_protected.get("reason") or "protected")]
        return "ambiguous_multi_owner", "", ["cleanup_owned_and_protected_overlap"]
    if best_protected:
        return _protected_ownership_state(best_protected), "", [str(best_protected.get("reason") or "protected")]
    if len(eligible_owners) > 1:
        best_ratio = float(best_owner.get("overlap_pixels") or 0) / float(max(1, pixel_count)) if best_owner else 0.0
        if best_ratio < COMPONENT_MULTI_OWNER_DOMINANCE_RATIO:
            return "ambiguous_multi_owner", "", ["multiple_cleanup_owner_candidates"]
    if best_owner:
        return "cleanup_owned", str(best_owner.get("cleanup_job_id") or ""), []
    if owner_candidates:
        return "unowned_visible_text", "", ["below_cleanup_owner_overlap_threshold"]
    return "outside_cleanup_scope", "", []


def _protected_ownership_state(candidate: Mapping[str, Any]) -> str:
    reason = str(candidate.get("reason") or "").lower()
    if "art" in reason or "non_text" in reason:
        return "protected_art_or_non_text"
    return "protected_sfx_decorative"


def _protected_component_reason(candidates: Sequence[dict[str, Any]]) -> str:
    for item in candidates:
        if item.get("eligible") and item.get("reason"):
            return str(item.get("reason"))
    return ""


def _component_owner_region_ids(candidates: Sequence[dict[str, Any]], owner_job_id: str) -> list[str]:
    for item in candidates:
        if str(item.get("cleanup_job_id") or "") == owner_job_id:
            return list(item.get("region_ids") or [])
    return []


def _best_overlap(candidates: Sequence[dict[str, Any]]) -> int:
    return max([int(item.get("overlap_pixels") or 0) for item in candidates] or [0])


def _point_in_bbox(point: tuple[float, float], bbox: Sequence[int] | None) -> bool:
    box = _valid_bbox(bbox)
    if box is None:
        return False
    x, y = point
    return box[0] <= x < box[2] and box[1] <= y < box[3]


def _component_projection_for_job(
    *,
    job: CleanupJob,
    allowed: list[int],
    component_projection: _ComponentOwnershipProjection,
) -> dict[str, Any]:
    job_id = str(getattr(job, "cleanup_job_id", "") or "")
    scoped_components = [
        item
        for item in component_projection.components
        if item.get("owner_cleanup_job_id") == job_id or job_id in set(item.get("scope_cleanup_job_ids") or [])
    ]
    owned = [item for item in component_projection.components if item.get("ownership_state") == "cleanup_owned" and item.get("owner_cleanup_job_id") == job_id]
    protected = [
        item
        for item in scoped_components
        if str(item.get("ownership_state") or "").startswith("protected")
    ]
    ambiguous = [item for item in scoped_components if item.get("ownership_state") == "ambiguous_multi_owner"]
    unowned = [item for item in scoped_components if item.get("ownership_state") == "unowned_visible_text"]
    return {
        "owned_component_ids": [str(item["component_id"]) for item in owned],
        "protected_component_ids": [str(item["component_id"]) for item in protected],
        "ambiguous_component_ids": [str(item["component_id"]) for item in ambiguous],
        "unowned_component_ids": [str(item["component_id"]) for item in unowned],
        "owned_component_pixel_count": sum(int(item.get("pixel_count") or 0) for item in owned),
        "protected_component_pixel_count": sum(int(item.get("pixel_count") or 0) for item in protected),
        "ambiguous_component_pixel_count": sum(int(item.get("pixel_count") or 0) for item in ambiguous),
        "unresolved_reasons": _component_projection_unresolved_reasons(protected, ambiguous, unowned),
    }


def _component_projection_unresolved_reasons(
    protected: Sequence[Mapping[str, Any]],
    ambiguous: Sequence[Mapping[str, Any]],
    unowned: Sequence[Mapping[str, Any]],
) -> list[str]:
    reasons: list[str] = []
    if protected:
        reasons.append("protected_component_overlap")
    if ambiguous:
        reasons.append("ambiguous_multi_owner_components")
    if unowned:
        reasons.append("unowned_visible_text_components")
    return reasons


def _component_mask_from_ids(
    component_projection: _ComponentOwnershipProjection,
    component_ids: Sequence[str],
) -> np.ndarray | None:
    if component_projection.labels is None or not component_ids:
        return None
    labels = [component_projection.component_label_by_id[item] for item in component_ids if item in component_projection.component_label_by_id]
    if not labels:
        return None
    return np.isin(component_projection.labels, labels).astype(np.uint8)


def _component_by_id(component_projection: _ComponentOwnershipProjection, component_id: str) -> dict[str, Any]:
    for item in component_projection.components:
        if item.get("component_id") == component_id:
            return item
    return {}


def _largest_component_pixels(components: Sequence[Mapping[str, Any]]) -> int:
    return max([int(item.get("pixel_count") or 0) for item in components] or [0])


def _component_ownership_state_counts(components: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in components:
        state = str(item.get("ownership_state") or "unknown")
        counts[state] = counts.get(state, 0) + 1
    return counts


def _bbox_mask(shape: tuple[int, int], bbox: list[int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    height, width = shape[:2]
    x0, y0, x1, y1 = _clip_bbox_to_size(bbox, width, height)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = 1
    return mask


def _index_region_records(records: Sequence[Mapping[str, Any]] | None) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(records or []):
        if not isinstance(record, Mapping):
            continue
        region_id = str(record.get("region_id") or record.get("id") or f"region_{index}")
        if region_id:
            output[region_id] = record
    return output


def _region_bbox(region: Mapping[str, Any]) -> list[int] | None:
    for key in ("bbox", "xyxy", "bounds"):
        bbox = _valid_or_xywh_bbox(region.get(key))
        if bbox is not None:
            return bbox
    render = region.get("render")
    if isinstance(render, Mapping):
        for key in ("bbox", "source_bbox", "text_area_container_bbox"):
            bbox = _valid_or_xywh_bbox(render.get(key))
            if bbox is not None:
                return bbox
    return None


def _valid_or_xywh_bbox(value: Any) -> list[int] | None:
    bbox = _valid_bbox(value)
    if bbox is not None:
        return bbox
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        return None
    try:
        x, y, w, h = [int(round(float(item))) for item in value]
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return [x, y, x + w, y + h]


def _region_protection_reason(region: Mapping[str, Any]) -> str:
    render = region.get("render") if isinstance(region.get("render"), Mapping) else {}
    flags = region.get("flags") if isinstance(region.get("flags"), Mapping) else {}
    marker_parts: list[str] = []
    for value in (
        region.get("type"),
        region.get("semantic_class"),
        region.get("route_intent"),
        region.get("cleanup_mode"),
        region.get("skip_reason"),
        region.get("classification_reason"),
        render.get("cleanup_mode") if isinstance(render, Mapping) else "",
        render.get("classification_reason") if isinstance(render, Mapping) else "",
    ):
        marker_parts.append(str(value or ""))
    for key, value in (flags or {}).items():
        if value:
            marker_parts.append(str(key))
    text = " ".join(marker_parts).lower()
    if "cleanup_mode" in region and str(region.get("cleanup_mode") or "").lower() == "preserve":
        return "explicit_cleanup_mode_preserve"
    if isinstance(render, Mapping) and str(render.get("cleanup_mode") or "").lower() == "preserve":
        return "explicit_render_cleanup_mode_preserve"
    protected_markers = (
        ("preserve_sfx_decorative", "preserve_sfx_decorative"),
        ("decorative_text", "decorative_text"),
        ("non_translation_art", "non_translation_art"),
        ("art_only", "art_only"),
        ("non_text", "non_text"),
        ("sfx", "sfx"),
        ("protected", "protected"),
    )
    for marker, reason in protected_markers:
        if marker in text:
            return reason
    return ""


def _sourceglyph_overlap_metrics(
    *,
    foreground: np.ndarray,
    seed: np.ndarray | None,
) -> tuple[int, float, int]:
    foreground_pixels = int(np.count_nonzero(foreground > 0))
    if seed is None or int(np.count_nonzero(seed > 0)) <= 0:
        return 0, 0.0, foreground_pixels
    seed_mask = _coerce_mask_shape((seed > 0).astype(np.uint8), foreground.shape)
    overlap = int(np.count_nonzero((foreground > 0) & (seed_mask > 0)))
    outside = max(0, foreground_pixels - overlap)
    ratio = round(float(overlap) / float(max(1, foreground_pixels)), 4)
    return overlap, ratio, outside


def _segmentation_effective_coverage_reason(
    *,
    foreground: np.ndarray,
    analysis_scope: list[int],
    job: CleanupJob,
    protected_overlap_pixels: int,
    owner_pixels: int,
) -> str:
    bbox = _mask_bbox(foreground)
    if bbox is None:
        return "effective_mask_no_segmentation_in_owned_area"
    pixels = int(np.count_nonzero(foreground > 0))
    coverage = _text_block_coverage_estimate(bbox, analysis_scope)
    stats = _component_stats(foreground)
    cleanup_value = _enum_value(getattr(job, "cleanup_class", ""))
    small_or_glyph_local = cleanup_value in {
        CleanupClass.SMALL_REACTION.value,
        CleanupClass.SIDE_CAPTION_GLYPH_LOCAL.value,
        CleanupClass.TITLE_OR_SIGN.value,
    }
    min_coverage = MIN_SEGMENTATION_READY_SMALL_COVERAGE_RATIO if small_or_glyph_local else MIN_SEGMENTATION_READY_COVERAGE_RATIO
    if owner_pixels > 0 and protected_overlap_pixels / float(max(1, owner_pixels)) > PROTECTED_DOMINANT_OVERLAP_RATIO:
        return "effective_mask_protected_overlap_removed"
    if pixels < MIN_SEGMENTATION_READY_PIXELS:
        return "effective_mask_fragment_only"
    if pixels <= FRAGMENT_ONLY_MAX_PIXELS and coverage < FRAGMENT_ONLY_MAX_COVERAGE_RATIO:
        return "effective_mask_fragment_only"
    if coverage < min_coverage:
        return "effective_mask_incomplete_under_coverage"
    largest_component = int(stats.get("largest_component_pixels") or 0)
    if largest_component <= 0 or (pixels <= FRAGMENT_ONLY_MAX_PIXELS and largest_component < pixels * 0.45):
        return "effective_mask_fragment_only"
    return ""


def _owned_segmentation_executable_coverage_reason(
    *,
    ratio: float,
    owned_pixels: int,
    foreground_pixels: int,
    job: CleanupJob,
) -> str:
    if owned_pixels <= 0:
        return ""
    cleanup_value = _enum_value(getattr(job, "cleanup_class", ""))
    small_or_glyph_local = cleanup_value in {
        CleanupClass.SMALL_REACTION.value,
        CleanupClass.SIDE_CAPTION_GLYPH_LOCAL.value,
        CleanupClass.TITLE_OR_SIGN.value,
    }
    min_ratio = (
        MIN_OWNED_SEGMENTATION_TO_EXECUTABLE_SMALL_RATIO
        if small_or_glyph_local
        else MIN_OWNED_SEGMENTATION_TO_EXECUTABLE_RATIO
    )
    if foreground_pixels < MIN_SEGMENTATION_READY_PIXELS:
        return "effective_mask_fragment_only"
    if ratio < min_ratio:
        return "effective_mask_incomplete_under_coverage"
    return ""


def _expand_bbox(bbox: list[int], margin: int) -> list[int]:
    return [
        int(bbox[0]) - margin,
        int(bbox[1]) - margin,
        int(bbox[2]) + margin,
        int(bbox[3]) + margin,
    ]


def _intersect_bboxes(a: list[int], b: list[int]) -> list[int] | None:
    box_a = _valid_bbox(a)
    box_b = _valid_bbox(b)
    if box_a is None or box_b is None:
        return None
    x0 = max(box_a[0], box_b[0])
    y0 = max(box_a[1], box_b[1])
    x1 = min(box_a[2], box_b[2])
    y1 = min(box_a[3], box_b[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def _segmentation_blocks_for_scope(segmentation_audit: Mapping[str, Any], scope: list[int]) -> list[dict[str, Any]]:
    blocks = list(segmentation_audit.get("block_associations") or [])
    output: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, Mapping):
            continue
        bbox = _valid_bbox(block.get("line_bbox")) or _valid_bbox(block.get("xyxy"))
        if bbox is None:
            continue
        if _intersect_bboxes(bbox, scope) is not None:
            output.append(dict(block))
    return output[:20]


def _segmentation_foreground_unsafe_reason(
    foreground: np.ndarray,
    allowed: list[int],
    job: CleanupJob,
) -> str:
    bbox = _mask_bbox(foreground)
    if bbox is None:
        return "segmentation_mask_under_coverage"
    pixels = int(np.count_nonzero(foreground > 0))
    allowed_area = max(1, _bbox_area(allowed))
    bbox_area = max(1, _bbox_area(bbox))
    pixel_ratio = pixels / float(allowed_area)
    fill_ratio = pixels / float(bbox_area)
    cleanup_value = _enum_value(getattr(job, "cleanup_class", ""))
    speech_like = cleanup_value in {
        CleanupClass.SPEECH_FLAT_BUBBLE.value,
        CleanupClass.SPEECH_COMPLEX_BUBBLE.value,
        CleanupClass.SMALL_REACTION.value,
    }
    max_pixel_ratio = 0.48 if speech_like else 0.34
    if pixel_ratio > max_pixel_ratio:
        return "segmentation_mask_wrong_owner_broad_background_capture"
    if fill_ratio > 0.72 and bbox_area > 1800:
        return "segmentation_mask_wrong_owner_rectangular_background_capture"
    return ""


def _group_segmentation_text_components(
    foreground: np.ndarray,
    scope: list[int],
    job: CleanupJob,
) -> np.ndarray:
    if cv2 is None or not np.any(foreground > 0):
        return foreground
    stats = _component_stats(foreground)
    if int(stats.get("component_count") or 0) <= 4:
        return foreground
    axis = _text_axis(foreground, scope, job)
    if axis == "vertical":
        kernel = np.ones((5, 13), dtype=np.uint8)
    elif axis == "horizontal":
        kernel = np.ones((13, 5), dtype=np.uint8)
    else:
        kernel = np.ones((5, 5), dtype=np.uint8)
    grouped = cv2.morphologyEx((foreground > 0).astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1)
    grouped = _clip_mask_to_bbox(grouped.astype(np.uint8), scope)
    unsafe = _grouped_segmentation_too_broad(grouped, foreground, scope)
    if unsafe:
        return foreground
    return grouped


def _grouped_segmentation_too_broad(
    grouped: np.ndarray,
    original: np.ndarray,
    scope: list[int],
) -> bool:
    original_pixels = max(1, int(np.count_nonzero(original > 0)))
    grouped_pixels = int(np.count_nonzero(grouped > 0))
    if grouped_pixels <= original_pixels:
        return False
    if grouped_pixels / float(original_pixels) > 2.35:
        return True
    scope_area = max(1, _bbox_area(scope))
    if grouped_pixels / float(scope_area) > 0.42:
        return True
    return False


def _source_image_array(
    *,
    source_image: Any | None,
    source_image_path: str | Path | None,
) -> tuple[np.ndarray | None, str]:
    if source_image is not None:
        try:
            if Image is not None and isinstance(source_image, Image.Image):
                return np.asarray(source_image.convert("RGB")), ""
            arr = np.asarray(source_image)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=2)
            if arr.ndim == 3:
                return arr[:, :, :3].astype(np.uint8), ""
        except Exception as exc:
            return None, f"source_image_array_error:{type(exc).__name__}"
    if source_image_path:
        if Image is None:
            return None, "pillow_unavailable"
        try:
            with Image.open(source_image_path) as img:
                return np.asarray(img.convert("RGB")), ""
        except Exception as exc:
            return None, f"source_image_load_error:{type(exc).__name__}"
    return None, "source_image_unavailable"


def _build_effective_text_mask(
    *,
    job: CleanupJob,
    source_np: np.ndarray | None,
    source_error: str,
    seed_foreground: np.ndarray,
    allowed: list[int],
    matched: Sequence[_SourceEvidence],
) -> _EffectiveMaskBuild:
    seed = _clip_mask_to_bbox((seed_foreground > 0).astype(np.uint8), allowed)
    seed_pixels = int(np.count_nonzero(seed))
    seed_stats = _component_stats(seed)
    seed_bbox = _mask_bbox(seed)
    seed_fill = _mask_fill_ratio(seed)
    base_audit = {
        "seed_foreground_pixels": seed_pixels,
        "component_count_before": seed_stats["component_count"],
        "largest_component_pixels_before": seed_stats["largest_component_pixels"],
        "bbox_fill_ratio_before": seed_fill,
        "source_seed_mask_ids": [source.mask_id for source in matched if source.mask_id],
    }
    if seed_pixels <= 0 or seed_bbox is None:
        audit = {
            **base_audit,
            "completed_foreground_pixels": 0,
            "component_count_after": 0,
            "largest_component_pixels_after": 0,
            "text_block_coverage_estimate": 0.0,
            "bbox_fill_ratio_after": 0.0,
            "analysis_scope_bbox": allowed,
            "executable_erase_bbox": None,
            "mask_completion_method": "sourceglyph_seed_empty",
            "polarity_mode": "unknown",
            "recovered_component_count": 0,
            "rejected_component_count": 0,
            "rejected_component_reasons": ["empty_sourceglyph_seed"],
        }
        return _EffectiveMaskBuild(
            foreground=None,
            erase=None,
            status="effective_mask_failed_insufficient_evidence",
            failure_reason="effective_mask_failed_insufficient_evidence",
            audit=audit,
            rejected=True,
        )

    suspect_reason = _seed_effective_mask_suspect_reason(seed, allowed, job)
    if source_np is None:
        erase = _effective_erase_from_foreground(seed, allowed, job)
        audit = _effective_audit(
            seed=seed,
            completed=seed,
            erase=erase,
            allowed=allowed,
            analysis_scope=seed_bbox,
            method="sourceglyph_seed_used_without_source_image",
            polarity="unknown",
            rejected_reasons=[source_error or "source_image_unavailable"],
            recovered_count=0,
        )
        rejected = bool(suspect_reason)
        return _EffectiveMaskBuild(
            foreground=seed,
            erase=erase,
            status=(
                "effective_mask_failed_insufficient_evidence"
            ),
            failure_reason=(
                "effective_mask_failed_insufficient_evidence"
            ),
            audit={**base_audit, **audit},
            rejected=True,
        )

    analysis_scope = _analysis_scope_bbox(seed_bbox, allowed, job, suspect=bool(suspect_reason))
    reconstructed, reconstruction_audit = _reconstruct_text_strokes(
        source_np=source_np,
        seed=seed,
        allowed=allowed,
        analysis_scope=analysis_scope,
        job=job,
    )
    if reconstructed is None or int(np.count_nonzero(reconstructed)) <= 0:
        erase = _effective_erase_from_foreground(seed, allowed, job)
        audit = _effective_audit(
            seed=seed,
            completed=seed,
            erase=erase,
            allowed=allowed,
            analysis_scope=analysis_scope,
            method="sourceglyph_seed_reconstruction_failed",
            polarity=str(reconstruction_audit.get("polarity_mode") or "unknown"),
            rejected_reasons=list(reconstruction_audit.get("rejected_component_reasons") or [])
            + ["no_text_strokes_found"],
            recovered_count=0,
        )
        rejected = bool(suspect_reason)
        return _EffectiveMaskBuild(
            foreground=seed,
            erase=erase,
            status=(
                "effective_mask_failed_no_text_strokes_found"
            ),
            failure_reason=(
                "effective_mask_failed_no_text_strokes_found"
            ),
            audit={**base_audit, **audit, **reconstruction_audit},
            rejected=True,
        )

    completed = np.maximum(seed, reconstructed.astype(np.uint8))
    completed = _clip_mask_to_bbox(completed, allowed)
    completed = _bridge_text_stroke_gaps(completed, allowed, job)
    completed = _clip_mask_to_bbox(completed, allowed)
    unsafe_reason = _completed_mask_unsafe_reason(completed, allowed, job)
    if unsafe_reason:
        erase = _effective_erase_from_foreground(seed, allowed, job)
        audit = _effective_audit(
            seed=seed,
            completed=completed,
            erase=erase,
            allowed=allowed,
            analysis_scope=analysis_scope,
            method="local_text_stroke_reconstruction_rejected",
            polarity=str(reconstruction_audit.get("polarity_mode") or "unknown"),
            rejected_reasons=list(reconstruction_audit.get("rejected_component_reasons") or [])
            + [unsafe_reason],
            recovered_count=int(reconstruction_audit.get("recovered_component_count") or 0),
        )
        rejected = bool(suspect_reason)
        return _EffectiveMaskBuild(
            foreground=seed if rejected else completed,
            erase=erase,
            status=(
                "effective_mask_failed_unsafe_background_capture"
            ),
            failure_reason=unsafe_reason,
            audit={**base_audit, **audit},
            rejected=True,
        )

    erase = _effective_erase_from_foreground(completed, allowed, job)
    recovered_pixels = int(np.count_nonzero(completed)) - seed_pixels
    method = (
        "local_text_stroke_reconstruction"
        if recovered_pixels > max(24, int(seed_pixels * 0.08)) or suspect_reason
        else "sourceglyph_seed_effective"
    )
    audit = _effective_audit(
        seed=seed,
        completed=completed,
        erase=erase,
        allowed=allowed,
        analysis_scope=analysis_scope,
        method=method,
        polarity=str(reconstruction_audit.get("polarity_mode") or "unknown"),
        rejected_reasons=list(reconstruction_audit.get("rejected_component_reasons") or []),
        recovered_count=int(reconstruction_audit.get("recovered_component_count") or 0),
    )
    return _EffectiveMaskBuild(
        foreground=completed,
        erase=erase,
        status="cleanup_mask_unresolved_after_segmentation",
        failure_reason=suspect_reason or "local_contrast_fallback_diagnostic_only",
        audit={**base_audit, **audit},
        rejected=True,
    )


def _reconstruct_text_strokes(
    *,
    source_np: np.ndarray,
    seed: np.ndarray,
    allowed: list[int],
    analysis_scope: list[int],
    job: CleanupJob,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    height, width = source_np.shape[:2]
    x0, y0, x1, y1 = _clip_bbox_to_size(analysis_scope, width, height)
    if x1 <= x0 or y1 <= y0:
        return None, {
            "polarity_mode": "unknown",
            "recovered_component_count": 0,
            "rejected_component_count": 0,
            "rejected_component_reasons": ["invalid_analysis_scope"],
        }
    crop = source_np[y0:y1, x0:x1]
    seed_crop = seed[y0:y1, x0:x1]
    gray = _gray(crop)
    polarity = _polarity_mode(gray, seed_crop)
    candidate = _local_stroke_candidates(gray, polarity)
    if candidate is None or not np.any(candidate > 0):
        return None, {
            "polarity_mode": polarity,
            "recovered_component_count": 0,
            "rejected_component_count": 0,
            "rejected_component_reasons": ["local_contrast_found_no_strokes"],
        }
    selected, recovered_count, rejected_reasons = _select_textlike_components(
        candidate=candidate,
        seed_crop=seed_crop,
        crop_origin=(x0, y0),
        allowed=allowed,
        job=job,
    )
    if not np.any(selected > 0):
        return None, {
            "polarity_mode": polarity,
            "recovered_component_count": recovered_count,
            "rejected_component_count": len(rejected_reasons),
            "rejected_component_reasons": rejected_reasons,
        }
    output = np.zeros(seed.shape, dtype=np.uint8)
    output[y0:y1, x0:x1] = selected.astype(np.uint8)
    output = _clip_mask_to_bbox(output, allowed)
    return output, {
        "polarity_mode": polarity,
        "recovered_component_count": recovered_count,
        "rejected_component_count": len(rejected_reasons),
        "rejected_component_reasons": rejected_reasons[:40],
    }


def _local_stroke_candidates(gray: np.ndarray, polarity: str) -> np.ndarray | None:
    if gray.size <= 0:
        return None
    if cv2 is not None:
        block = max(15, min(51, (min(gray.shape[:2]) // 2) * 2 + 1))
        dark_adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block,
            7,
        )
        mean = cv2.blur(gray, (max(9, block // 2), max(9, block // 2)))
        dark_local = ((gray.astype(np.int16) + 18) < mean.astype(np.int16)).astype(np.uint8) * 255
        light_local = ((gray.astype(np.int16) - 18) > mean.astype(np.int16)).astype(np.uint8) * 255
        if polarity == "dark_on_light":
            candidate = (dark_adaptive > 0) | (dark_local > 0)
        elif polarity == "light_on_dark":
            candidate = light_local > 0
        else:
            candidate = (dark_local > 0) | (light_local > 0)
        candidate = candidate.astype(np.uint8)
        kernel = np.ones((2, 2), dtype=np.uint8)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel, iterations=1)
        return candidate.astype(np.uint8)
    mean_value = float(np.mean(gray))
    if polarity == "light_on_dark":
        return (gray > mean_value + 18).astype(np.uint8)
    if polarity == "dark_on_light":
        return (gray < mean_value - 18).astype(np.uint8)
    return ((gray < mean_value - 18) | (gray > mean_value + 18)).astype(np.uint8)


def _select_textlike_components(
    *,
    candidate: np.ndarray,
    seed_crop: np.ndarray,
    crop_origin: tuple[int, int],
    allowed: list[int],
    job: CleanupJob,
) -> tuple[np.ndarray, int, list[str]]:
    selected = np.zeros_like(candidate, dtype=np.uint8)
    rejected_reasons: list[str] = []
    if cv2 is None:
        selected = candidate.astype(np.uint8)
        return selected, 1 if np.any(selected > 0) else 0, rejected_reasons
    labels_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (candidate > 0).astype(np.uint8),
        connectivity=8,
    )
    seed_dilated = seed_crop.astype(np.uint8)
    if np.any(seed_dilated > 0):
        kernel = np.ones((19, 19), dtype=np.uint8)
        seed_dilated = cv2.dilate(seed_dilated, kernel, iterations=1)
    seed_box = _mask_bbox(seed_crop)
    axis = _text_axis(seed_crop, allowed, job)
    recovered_count = 0
    for label in range(1, labels_count):
        area_pixels = int(stats[label, cv2.CC_STAT_AREA])
        if area_pixels < 4:
            rejected_reasons.append("component_too_small")
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        fill_ratio = area_pixels / max(1, w * h)
        abs_box = [x + crop_origin[0], y + crop_origin[1], x + w + crop_origin[0], y + h + crop_origin[1]]
        if _component_reject_reason(abs_box, area_pixels, fill_ratio, allowed, job):
            rejected_reasons.append(_component_reject_reason(abs_box, area_pixels, fill_ratio, allowed, job))
            continue
        component = labels == label
        seed_overlap = int(np.count_nonzero(component & (seed_dilated > 0)))
        aligned = _component_aligned_with_seed(
            local_box=[x, y, x + w, y + h],
            seed_box=seed_box,
            axis=axis,
            crop_shape=candidate.shape,
        )
        if seed_overlap <= 0 and not aligned:
            rejected_reasons.append("component_not_seed_connected_or_aligned")
            continue
        selected[component] = 1
        recovered_count += 1
    return selected, recovered_count, rejected_reasons


def _component_reject_reason(
    box: list[int],
    area_pixels: int,
    fill_ratio: float,
    allowed: list[int],
    job: CleanupJob,
) -> str:
    width = max(1, box[2] - box[0])
    height = max(1, box[3] - box[1])
    allowed_area = max(1, _bbox_area(allowed))
    cleanup_value = _enum_value(getattr(job, "cleanup_class", ""))
    speech_like = cleanup_value in {
        CleanupClass.SPEECH_FLAT_BUBBLE.value,
        CleanupClass.SPEECH_COMPLEX_BUBBLE.value,
        CleanupClass.SMALL_REACTION.value,
    }
    if area_pixels > allowed_area * (0.24 if speech_like else 0.16):
        return "component_area_too_large_for_allowed_text_area"
    if fill_ratio > 0.72 and area_pixels > 180:
        return "component_mostly_solid_background_or_patch"
    if width >= height * 12 and height <= 6:
        return "component_panel_or_bubble_border_horizontal"
    if height >= width * 14 and width <= 6:
        return "component_panel_or_bubble_border_vertical"
    border_touch = (
        box[0] <= allowed[0] + 2
        or box[1] <= allowed[1] + 2
        or box[2] >= allowed[2] - 2
        or box[3] >= allowed[3] - 2
    )
    if border_touch and area_pixels > allowed_area * 0.06 and (width <= 8 or height <= 8):
        return "component_touches_allowed_border_like_line"
    return ""


def _component_aligned_with_seed(
    *,
    local_box: list[int],
    seed_box: list[int] | None,
    axis: str,
    crop_shape: tuple[int, int],
) -> bool:
    if seed_box is None:
        return True
    crop_h, crop_w = crop_shape[:2]
    if axis == "vertical":
        pad = max(18, int(crop_w * 0.22))
        return local_box[0] <= seed_box[2] + pad and local_box[2] >= seed_box[0] - pad
    if axis == "horizontal":
        pad = max(18, int(crop_h * 0.22))
        return local_box[1] <= seed_box[3] + pad and local_box[3] >= seed_box[1] - pad
    pad_x = max(20, int(crop_w * 0.20))
    pad_y = max(20, int(crop_h * 0.20))
    return (
        local_box[0] <= seed_box[2] + pad_x
        and local_box[2] >= seed_box[0] - pad_x
        and local_box[1] <= seed_box[3] + pad_y
        and local_box[3] >= seed_box[1] - pad_y
    )


def _analysis_scope_bbox(
    seed_bbox: list[int],
    allowed: list[int],
    job: CleanupJob,
    *,
    suspect: bool,
) -> list[int]:
    cleanup_value = _enum_value(getattr(job, "cleanup_class", ""))
    background_like = cleanup_value in {
        CleanupClass.CAPTION_FLAT_BACKGROUND.value,
        CleanupClass.CAPTION_DARK_OR_SCREENTONE.value,
        CleanupClass.BACKGROUND_ART_TEXT.value,
        CleanupClass.TITLE_OR_SIGN.value,
        CleanupClass.SIDE_CAPTION_GLYPH_LOCAL.value,
    }
    if suspect or background_like:
        return list(allowed)
    pad = max(12, int(max(seed_bbox[2] - seed_bbox[0], seed_bbox[3] - seed_bbox[1]) * 0.20))
    return [
        max(allowed[0], seed_bbox[0] - pad),
        max(allowed[1], seed_bbox[1] - pad),
        min(allowed[2], seed_bbox[2] + pad),
        min(allowed[3], seed_bbox[3] + pad),
    ]


def _polarity_mode(gray: np.ndarray, seed_crop: np.ndarray) -> str:
    if gray.size <= 0:
        return "unknown"
    if np.any(seed_crop > 0):
        seed_values = gray[seed_crop > 0]
        seed_mean = float(np.mean(seed_values))
        if cv2 is not None:
            kernel = np.ones((9, 9), dtype=np.uint8)
            halo = cv2.dilate(seed_crop.astype(np.uint8), kernel, iterations=1)
            background_values = gray[halo <= 0]
        else:
            background_values = gray[seed_crop <= 0]
        background_mean = float(np.mean(background_values)) if background_values.size else float(np.mean(gray))
        if seed_mean < background_mean - 10:
            return "dark_on_light"
        if seed_mean > background_mean + 10:
            return "light_on_dark"
    mean = float(np.mean(gray))
    if mean < 105:
        return "light_on_dark"
    if mean > 175:
        return "dark_on_light"
    return "mixed_or_screentone"


def _gray(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 2:
        return rgb.astype(np.uint8)
    if cv2 is not None:
        return cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return (
        0.299 * rgb[:, :, 0].astype(np.float32)
        + 0.587 * rgb[:, :, 1].astype(np.float32)
        + 0.114 * rgb[:, :, 2].astype(np.float32)
    ).astype(np.uint8)


def _seed_effective_mask_suspect_reason(mask: np.ndarray, allowed: list[int], job: CleanupJob) -> str:
    pixels = int(np.count_nonzero(mask > 0))
    bbox = _mask_bbox(mask)
    if bbox is None:
        return "formal_mask_exists_but_effective_mask_missing"
    bbox_area = _bbox_area(bbox)
    allowed_area = max(1, _bbox_area(allowed))
    width_ratio = (bbox[2] - bbox[0]) / max(1, allowed[2] - allowed[0])
    height_ratio = (bbox[3] - bbox[1]) / max(1, allowed[3] - allowed[1])
    bbox_coverage = bbox_area / allowed_area
    stats = _component_stats(mask)
    largest = int(stats["largest_component_pixels"])
    if pixels < 80:
        return "fragmented_character_only_mask"
    if pixels < 1200 and bbox_coverage < 0.16:
        return "fragmented_character_only_mask"
    if largest < max(24, int(pixels * 0.12)) and int(stats["component_count"]) > 18:
        return "fragmented_character_only_mask"
    if bbox_coverage < 0.10 and (width_ratio < 0.30 or height_ratio < 0.30):
        return "fragmented_character_only_mask"
    cleanup_value = _enum_value(getattr(job, "cleanup_class", ""))
    if cleanup_value in {
        CleanupClass.CAPTION_FLAT_BACKGROUND.value,
        CleanupClass.CAPTION_DARK_OR_SCREENTONE.value,
        CleanupClass.BACKGROUND_ART_TEXT.value,
        CleanupClass.TITLE_OR_SIGN.value,
        CleanupClass.SIDE_CAPTION_GLYPH_LOCAL.value,
    } and bbox_coverage < 0.22:
        return "effective_mask_under_50_percent"
    return ""


def _completed_mask_unsafe_reason(mask: np.ndarray, allowed: list[int], job: CleanupJob) -> str:
    pixels = int(np.count_nonzero(mask > 0))
    bbox = _mask_bbox(mask)
    if bbox is None:
        return "effective_mask_failed_no_text_strokes_found"
    allowed_area = max(1, _bbox_area(allowed))
    bbox_area = max(1, _bbox_area(bbox))
    fill_ratio = pixels / bbox_area
    pixel_ratio = pixels / allowed_area
    cleanup_value = _enum_value(getattr(job, "cleanup_class", ""))
    speech_like = cleanup_value in {
        CleanupClass.SPEECH_FLAT_BUBBLE.value,
        CleanupClass.SPEECH_COMPLEX_BUBBLE.value,
        CleanupClass.SMALL_REACTION.value,
    }
    if pixel_ratio > (0.42 if speech_like else 0.26):
        return "effective_mask_failed_unsafe_background_capture"
    if fill_ratio > 0.66 and bbox_area > 1200:
        return "effective_mask_failed_unsafe_background_capture"
    return ""


def _effective_erase_from_foreground(
    foreground: np.ndarray,
    allowed: list[int],
    job: CleanupJob,
) -> np.ndarray:
    foreground = _clip_mask_to_bbox((foreground > 0).astype(np.uint8), allowed)
    cleanup_value = _enum_value(getattr(job, "cleanup_class", ""))
    speech_like = cleanup_value in {
        CleanupClass.SPEECH_FLAT_BUBBLE.value,
        CleanupClass.SPEECH_COMPLEX_BUBBLE.value,
        CleanupClass.SMALL_REACTION.value,
    }
    if cv2 is not None:
        kernel = np.ones((3, 3), dtype=np.uint8)
        iterations = 2 if speech_like and int(np.count_nonzero(foreground)) < 1400 else 1
        erase = cv2.dilate(foreground, kernel, iterations=iterations).astype(np.uint8)
    else:
        erase = _binary_dilate(foreground, radius=1)
    return _clip_mask_to_bbox(erase.astype(np.uint8), allowed)


def _bridge_text_stroke_gaps(mask: np.ndarray, allowed: list[int], job: CleanupJob) -> np.ndarray:
    if cv2 is None or not np.any(mask > 0):
        return mask
    axis = _text_axis(mask, allowed, job)
    if axis == "vertical":
        kernel = np.ones((3, 7), dtype=np.uint8)
    elif axis == "horizontal":
        kernel = np.ones((7, 3), dtype=np.uint8)
    else:
        kernel = np.ones((3, 3), dtype=np.uint8)
    bridged = cv2.morphologyEx((mask > 0).astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1)
    return _clip_mask_to_bbox(bridged.astype(np.uint8), allowed)


def _binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return (mask > 0).astype(np.uint8)
    padded = np.pad((mask > 0).astype(np.uint8), radius, mode="constant")
    output = np.zeros_like(mask, dtype=np.uint8)
    for dy in range(0, radius * 2 + 1):
        for dx in range(0, radius * 2 + 1):
            output = np.maximum(output, padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]])
    return output.astype(np.uint8)


def _text_axis(mask: np.ndarray, allowed: list[int], job: CleanupJob) -> str:
    bbox = _mask_bbox(mask)
    if bbox is None:
        return "mixed"
    width = max(1, bbox[2] - bbox[0])
    height = max(1, bbox[3] - bbox[1])
    if height >= width * 1.35:
        return "vertical"
    if width >= height * 1.35:
        return "horizontal"
    mode = " ".join(
        str(value or "").lower()
        for value in (
            getattr(job, "cleanup_mode", ""),
            getattr(job, "classification_reason", ""),
            getattr(job, "route_intent", ""),
            getattr(job, "semantic_class", ""),
        )
    )
    if "vertical" in mode or "side_caption" in mode:
        return "vertical"
    return "mixed"


def _effective_audit(
    *,
    seed: np.ndarray,
    completed: np.ndarray,
    erase: np.ndarray,
    allowed: list[int],
    analysis_scope: list[int],
    method: str,
    polarity: str,
    rejected_reasons: Sequence[str],
    recovered_count: int,
) -> dict[str, Any]:
    completed_stats = _component_stats(completed)
    completed_bbox = _mask_bbox(completed)
    coverage = _text_block_coverage_estimate(completed_bbox, allowed)
    return {
        "completed_foreground_pixels": int(np.count_nonzero(completed > 0)),
        "component_count_after": completed_stats["component_count"],
        "largest_component_pixels_after": completed_stats["largest_component_pixels"],
        "text_block_coverage_estimate": coverage,
        "bbox_fill_ratio_after": _mask_fill_ratio(completed),
        "analysis_scope_bbox": list(analysis_scope),
        "executable_erase_bbox": _mask_bbox(erase),
        "mask_completion_method": method,
        "polarity_mode": polarity,
        "recovered_component_count": recovered_count,
        "rejected_component_count": len([reason for reason in rejected_reasons if reason]),
        "rejected_component_reasons": list(dict.fromkeys(str(reason) for reason in rejected_reasons if reason))[:40],
    }


def _component_stats(mask: np.ndarray) -> dict[str, int]:
    if not np.any(mask > 0):
        return {"component_count": 0, "largest_component_pixels": 0}
    if cv2 is None:
        return {
            "component_count": 1,
            "largest_component_pixels": int(np.count_nonzero(mask > 0)),
        }
    labels_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8),
        connectivity=8,
    )
    areas = [int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, labels_count)]
    return {
        "component_count": len(areas),
        "largest_component_pixels": max(areas) if areas else 0,
    }


def _mask_fill_ratio(mask: np.ndarray) -> float:
    bbox = _mask_bbox(mask)
    if bbox is None:
        return 0.0
    return round(float(np.count_nonzero(mask > 0)) / float(max(1, _bbox_area(bbox))), 4)


def _text_block_coverage_estimate(bbox: list[int] | None, allowed: list[int]) -> float:
    if bbox is None:
        return 0.0
    allowed_area = max(1, _bbox_area(allowed))
    return round(float(_bbox_area(bbox)) / float(allowed_area), 4)


def _binary_mask_from_evidence(
    evidence: _SourceEvidence,
    *,
    keys: Sequence[str],
) -> tuple[np.ndarray | None, str]:
    for key in keys:
        value = _get_value(evidence.raw, key)
        if value is None and isinstance(evidence.audit, Mapping):
            value = evidence.audit.get(key)
        mask = _to_binary_mask(value)
        if mask is not None:
            return mask, key
    return None, ""


def _evidence_has_raw_mask(evidence: _SourceEvidence) -> bool:
    for key in ("foreground_mask", "erase_mask", "mask"):
        if _to_binary_mask(_get_value(evidence.raw, key)) is not None:
            return True
    return False


def _to_binary_mask(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    if arr.size == 0:
        return None
    if arr.ndim == 3:
        arr = np.any(arr > 0, axis=2)
    elif arr.ndim != 2:
        return None
    return (arr > 0).astype(np.uint8)


def _coerce_mask_shape(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.copy()
    output = np.zeros(shape, dtype=np.uint8)
    height = min(int(shape[0]), int(mask.shape[0]))
    width = min(int(shape[1]), int(mask.shape[1]))
    if height > 0 and width > 0:
        output[:height, :width] = mask[:height, :width]
    return output


def _clip_mask_to_bbox(mask: np.ndarray, bbox: list[int]) -> np.ndarray:
    clipped = np.zeros_like(mask, dtype=np.uint8)
    height, width = mask.shape[:2]
    x0, y0, x1, y1 = _clip_bbox_to_size(bbox, width, height)
    if x1 <= x0 or y1 <= y0:
        return clipped
    clipped[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    return clipped


def _clip_bbox_to_size(bbox: list[int], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(int(x0), width))
    y0 = max(0, min(int(y0), height))
    x1 = max(0, min(int(x1), width))
    y1 = max(0, min(int(y1), height))
    return x0, y0, x1, y1


def _mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _growth_ratio(erase_pixels: int, foreground_pixels: int) -> float:
    if foreground_pixels <= 0:
        return float("inf")
    return round(float(erase_pixels) / float(foreground_pixels), 4)


def _broad_mask_rejection(
    *,
    allowed: list[int],
    erase_bbox: list[int],
    erase_pixels: int,
    growth_ratio: float,
    image_size: tuple[int, int] | None,
    allow_growth_exception: bool = False,
) -> str:
    if growth_ratio > MAX_ERASE_GROWTH_RATIO and not allow_growth_exception:
        return "erase_mask_growth_too_large"
    allowed_area = _bbox_area(allowed)
    if erase_pixels > allowed_area:
        return "erase_mask_pixels_exceed_allowed_area"
    if image_size and int(image_size[0] or 0) > 0 and int(image_size[1] or 0) > 0:
        page_area = max(1, int(image_size[0]) * int(image_size[1]))
        if erase_pixels / page_area > MAX_ERASE_PAGE_PIXEL_RATIO:
            return "erase_mask_pixels_too_broad_for_page"
        if _bbox_area(erase_bbox) / page_area > MAX_ERASE_BBOX_PAGE_RATIO:
            return "erase_mask_bbox_too_broad_for_page"
    return ""


def _caption_flat_small_mask_growth_exception_reason(
    *,
    job: CleanupJob,
    source: _SourceEvidence | None,
    allowed: list[int],
    erase_bbox: list[int],
    erase_pixels: int,
    foreground_pixels: int,
    growth_ratio: float,
    image_size: tuple[int, int] | None,
    artifact_risk: str,
) -> str:
    if growth_ratio <= MAX_ERASE_GROWTH_RATIO:
        return ""
    if growth_ratio > MAX_CAPTION_FLAT_SMALL_EXCEPTION_GROWTH_RATIO:
        return ""
    cleanup_class = getattr(job, "cleanup_class", "")
    if _enum_value(cleanup_class) != CleanupClass.CAPTION_FLAT_BACKGROUND.value:
        return ""
    if source is not None and _is_side_caption_glyph_local(job, source):
        return ""
    if artifact_risk:
        return ""
    method_text = " ".join(
        str(value or "")
        for value in (
            getattr(job, "classification_reason", ""),
            source.mask_id if source is not None else "",
            _first_present(source.audit, "generation_method", "source_glyph_mask_generation_method", default="")
            if source is not None
            else "",
        )
    ).lower()
    if "glyph_local" not in method_text:
        return ""
    allowed_area = _bbox_area(allowed)
    erase_bbox_area = _bbox_area(erase_bbox)
    if foreground_pixels <= 0 or erase_pixels <= 0:
        return ""
    if erase_pixels > MAX_CAPTION_FLAT_SMALL_EXCEPTION_ERASE_PIXELS:
        return ""
    if erase_bbox_area > MAX_CAPTION_FLAT_SMALL_EXCEPTION_ERASE_BBOX_AREA:
        return ""
    if allowed_area <= 0 or allowed_area > MAX_CAPTION_FLAT_SMALL_EXCEPTION_ALLOWED_AREA:
        return ""
    if erase_pixels / max(1, allowed_area) > MAX_CAPTION_FLAT_SMALL_EXCEPTION_ERASE_ALLOWED_RATIO:
        return ""
    if image_size and int(image_size[0] or 0) > 0 and int(image_size[1] or 0) > 0:
        page_area = max(1, int(image_size[0]) * int(image_size[1]))
        if allowed_area / page_area > MAX_CAPTION_FLAT_SMALL_EXCEPTION_ALLOWED_PAGE_RATIO:
            return ""
    return "caption_flat_small_mask_growth_exception"


def _allowed_area_rejection(allowed: list[int], image_size: tuple[int, int] | None) -> str:
    if _bbox_area(allowed) <= 0:
        return "allowed_cleanup_area_empty"
    if image_size and int(image_size[0] or 0) > 0 and int(image_size[1] or 0) > 0:
        page_area = max(1, int(image_size[0]) * int(image_size[1]))
        if _bbox_area(allowed) / page_area > MAX_ALLOWED_PAGE_RATIO:
            return "allowed_cleanup_area_near_full_page"
    return ""


def _job_protection_reason(job: CleanupJob) -> str:
    if bool(getattr(job, "protected", False)):
        return str(getattr(job, "protection_reason", "") or "job_protected")
    cleanup_class = getattr(job, "cleanup_class", "")
    cleanup_value = cleanup_class.value if isinstance(cleanup_class, Enum) else str(cleanup_class)
    combined = " ".join(
        str(value or "")
        for value in (
            cleanup_value,
            getattr(job, "route_intent", ""),
            getattr(job, "semantic_class", ""),
            getattr(job, "cleanup_mode", ""),
        )
    ).lower()
    if cleanup_value == CleanupClass.PRESERVE_SFX_DECORATIVE.value:
        return "cleanup_class_preserve_sfx_decorative"
    for marker in ("preserve", "sfx", "decorative", "art_only", "non_translation_art"):
        if marker in combined:
            return f"route_or_semantic_{marker}"
    return ""


def _is_side_caption_glyph_local(job: CleanupJob, evidence: _SourceEvidence) -> bool:
    cleanup_class = getattr(job, "cleanup_class", "")
    cleanup_value = cleanup_class.value if isinstance(cleanup_class, Enum) else str(cleanup_class)
    if cleanup_value == CleanupClass.SIDE_CAPTION_GLYPH_LOCAL.value:
        return True
    metadata = " ".join(
        str(value or "")
        for value in (
            cleanup_value,
            getattr(job, "classification_reason", ""),
            evidence.mask_id,
            _first_present(evidence.audit, "generation_method", "source_glyph_mask_generation_method", default=""),
        )
    ).lower()
    return "side_caption_glyph_local" in metadata


def _mask_method(evidence: _SourceEvidence, source_key: str) -> str:
    method = str(
        _first_present(
            evidence.audit,
            "foreground_mask_method",
            "erase_mask_method",
            "generation_method",
            "source_glyph_mask_generation_method",
            default="",
        )
        or ""
    )
    parts = [part for part in (method, source_key, "contract_only_not_renderer_consumed") if part]
    return "|".join(parts)


def _mask_method_union(evidence_records: Sequence[_SourceEvidence], source_keys: Sequence[str]) -> str:
    methods: list[str] = []
    for evidence in evidence_records:
        method = str(
            _first_present(
                evidence.audit,
                "foreground_mask_method",
                "erase_mask_method",
                "generation_method",
                "source_glyph_mask_generation_method",
                default="",
            )
            or ""
        )
        if method and method not in methods:
            methods.append(method)
    for source_key in source_keys or []:
        if source_key and source_key not in methods:
            methods.append(source_key)
    if evidence_records:
        methods.append("source_glyph_union_contract_only_not_renderer_consumed")
    else:
        methods.append("segmentation_component_projection_contract_only_not_renderer_consumed")
    return "|".join(methods)


def _base_job_record(page_id: str, job: CleanupJob) -> dict[str, Any]:
    return {
        "page_id": page_id,
        "cleanup_job_id": str(getattr(job, "cleanup_job_id", "") or ""),
        "target_region_ids": list(getattr(job, "target_region_ids", []) or []),
        "cleanup_class": _enum_value(getattr(job, "cleanup_class", "")),
        "route_intent": str(getattr(job, "route_intent", "") or ""),
        "semantic_class": str(getattr(job, "semantic_class", "") or ""),
        "cleanup_mode": str(getattr(job, "cleanup_mode", "") or ""),
    }


def _get_mapping(source: Any, key: str) -> Mapping[Any, Any] | None:
    value = _get_value(source, key)
    return value if isinstance(value, Mapping) else None


def _get_value(source: Any, key: str) -> Any:
    if isinstance(source, Mapping):
        return source.get(key)
    if hasattr(source, key):
        return getattr(source, key)
    return None


def _first_present(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _sequence_or_single(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


def _valid_bbox(value: Any) -> list[int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = [int(round(float(item))) for item in value]
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def _bbox_area(bbox: list[int]) -> int:
    return max(0, int(bbox[2]) - int(bbox[0])) * max(0, int(bbox[3]) - int(bbox[1]))


def _union_bboxes(bboxes: Sequence[Sequence[int]]) -> list[int] | None:
    valid: list[list[int]] = []
    for bbox in bboxes:
        box = _valid_bbox(bbox)
        if box is not None:
            valid.append(box)
    if not valid:
        return None
    return [
        min(box[0] for box in valid),
        min(box[1] for box in valid),
        max(box[2] for box in valid),
        max(box[3] for box in valid),
    ]


def _safe_id(value: Any) -> str:
    text = str(value or "none")
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_") or "none"


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "risk", "risky"}
    return bool(value)


def _enum_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_audit_dict"):
        return value.to_audit_dict()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(val) for key, val in value.items() if not _looks_like_raw_array(key)}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _looks_like_raw_array(key: Any) -> bool:
    return str(key).lower() in {"mask", "foreground_mask", "erase_mask", "image", "array", "pixels_raw", "bitmap", "crop"}
