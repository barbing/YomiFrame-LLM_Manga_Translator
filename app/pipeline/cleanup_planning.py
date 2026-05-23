"""Cleanup planning, execution, and proof contracts.

This module consumes SourceGlyph-derived cleanup contracts and keeps cleanup as
the pre-render production owner for source-text removal. Runtime execution uses
glyph-local cleanup geometry from CleanupMask foreground evidence; renderer code
remains text/layout only.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from app.pipeline.cleanup_contracts import (
    CleanupClass,
    CleanupJob,
    CleanupMask,
    CleanupPlan,
    CleanupProof,
    CleanupResult,
    ProofStatus,
)
from app.pipeline.cleanup_masks import CleanupMaskBuildResult
from app.pipeline import cleanup_execution
from app.pipeline import cleanup_backend_runner
from app.pipeline.debug_artifacts import mask_stats

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None


CLEANUP_PLAN_CONTRACT_VERSION = "cleanup_plans_phase3c_caption_flat_background_dense_candidates"
PILOT_RUNTIME_VERSION = "cleanup_pilot_phase3c_caption_flat_background_dense_candidates"
CLEANUP_RUNTIME_CONTRACT_VERSION = "cleanup_runtime_phase5_phase6_adaptive_cleanup_inpainting_migration"
CLEANUP_UPSTREAM_COMMIT_VERSION = "cleanup_upstream_commit_phase5_pre_render_working_image"

MAX_PLAN_GROWTH_RATIO = 2.25
MAX_ERASE_MASK_PIXELS = 20_000
MAX_ERASE_BBOX_AREA = 120_000
MAX_ERASE_ALLOWED_RATIO = 0.35
MAX_ALLOWED_PAGE_RATIO = 0.20

PROOF_MASK_SOURCE_OVERLAP_RATIO = 0.90
PROOF_RESIDUAL_RATIO = 0.12
PROOF_RESIDUAL_PIXEL_FRACTION = 0.08
PROOF_CHANGED_OUTSIDE_ALLOWED_PIXELS = 16
PROOF_CHANGED_OUTSIDE_ALLOWED_RATIO = 0.001
PROOF_COLLATERAL_INSIDE_ALLOWED_RATIO = 0.10

RUNTIME_SUPPORTED_CLEANUP_CLASSES = {
    CleanupClass.SPEECH_FLAT_BUBBLE.value,
    CleanupClass.SPEECH_COMPLEX_BUBBLE.value,
    CleanupClass.CAPTION_FLAT_BACKGROUND.value,
    CleanupClass.CAPTION_DARK_OR_SCREENTONE.value,
    CleanupClass.TITLE_OR_SIGN.value,
    CleanupClass.SMALL_REACTION.value,
    CleanupClass.BACKGROUND_ART_TEXT.value,
    CleanupClass.ART_ENTANGLED_AMBIGUOUS.value,
    CleanupClass.SIDE_CAPTION_GLYPH_LOCAL.value,
}
RUNTIME_UNSAFE_CLEANUP_CLASSES = {
    CleanupClass.PRESERVE_SFX_DECORATIVE.value,
}

VALID_CLEANUP_VISUAL_SCOPES = {
    "source_glyph_local",
    "source_glyph_union",
    "source_glyph_union_partition",
}
LEGACY_CLEANUP_VISUAL_SCOPE_MAP = {
    "glyph_local": "source_glyph_local",
}
ARTIFACT_RISK_PARTITION_REQUIRED_REASON = "artifact_risk_partition_required"
MAX_PARTITION_COMPONENTS = 32


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
            "module": "app.pipeline.cleanup_planning",
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
            "module": "app.pipeline.cleanup_planning",
            "stage": stage,
            "event": event,
        }
        payload.update(fields)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception:
        return


@dataclass(frozen=True)
class CleanupPlanBuildResult:
    """Strict caption-flat CleanupPlan contracts."""

    page_id: str
    version: str
    plans: list[CleanupPlan] = field(default_factory=list)
    plans_by_job_id: dict[str, list[CleanupPlan]] = field(default_factory=dict)
    backend_inventory: dict[str, Any] = field(default_factory=dict)
    rejected_records: list[dict[str, Any]] = field(default_factory=list)
    protected_records: list[dict[str, Any]] = field(default_factory=list)
    skipped_records: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "page_id": self.page_id,
            "renderer_consumed": False,
            "plans": [_json_safe(plan) for plan in self.plans],
            "plans_by_job_id": {
                str(job_id): [_json_safe(plan) for plan in plans]
                for job_id, plans in self.plans_by_job_id.items()
            },
            "backend_inventory": _json_safe(self.backend_inventory),
            "rejected_records": _json_safe(self.rejected_records),
            "protected_records": _json_safe(self.protected_records),
            "skipped_records": _json_safe(self.skipped_records),
            "errors": list(self.errors),
            "summary": {
                "pilot_class": CleanupClass.CAPTION_FLAT_BACKGROUND.value,
                "plan_count": len(self.plans),
                "rejected_record_count": len(self.rejected_records),
                "protected_record_count": len(self.protected_records),
                "skipped_record_count": len(self.skipped_records),
                "error_count": len(self.errors),
                "renderer_consumed": False,
                "backend_candidate_count": int(self.backend_inventory.get("candidate_count") or 0),
                "backend_available_count": int(self.backend_inventory.get("available_count") or 0),
            },
        }


@dataclass(frozen=True)
class CaptionFlatPilotRuntimeResult:
    """Renderer-facing decided pilot status and audit payloads."""

    page_id: str
    version: str
    cleaned_image: Any = field(repr=False, compare=False)
    status_records: list[dict[str, Any]] = field(default_factory=list)
    result_records: list[CleanupResult] = field(default_factory=list)
    proof_records: list[CleanupProof] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def committed_region_ids(self) -> set[str]:
        return {
            str(rid)
            for record in self.status_records
            if record.get("pilot_status") == "passed"
            for rid in record.get("target_region_ids", [])
        }

    @property
    def blocked_region_ids(self) -> set[str]:
        return {
            str(rid)
            for record in self.status_records
            if record.get("pilot_status") in {"failed", "inconclusive"}
            for rid in record.get("target_region_ids", [])
        }

    def region_status_by_id(self) -> dict[str, dict[str, Any]]:
        mapping: dict[str, dict[str, Any]] = {}
        for record in self.status_records:
            for rid in record.get("target_region_ids", []):
                mapping[str(rid)] = record
        return mapping

    def to_audit_dict(self) -> dict[str, Any]:
        passed = [record for record in self.status_records if record.get("pilot_status") == "passed"]
        failed = [record for record in self.status_records if record.get("pilot_status") == "failed"]
        inconclusive = [
            record
            for record in self.status_records
            if record.get("pilot_status") == "inconclusive"
        ]
        return {
            "version": self.version,
            "page_id": self.page_id,
            "renderer_consumed": True,
            "pilot_class": CleanupClass.CAPTION_FLAT_BACKGROUND.value,
            "status_records": _json_safe(self.status_records),
            "errors": list(self.errors),
            "summary": {
                "status_count": len(self.status_records),
                "passed_count": len(passed),
                "failed_count": len(failed),
                "inconclusive_count": len(inconclusive),
                "committed_region_ids": sorted(self.committed_region_ids),
                "blocked_region_ids": sorted(self.blocked_region_ids),
            },
        }


@dataclass(frozen=True)
class CleanupRuntimeContractResult:
    """Cleanup-owned Phase 5 runtime result/proof records.

    These records are intentionally pre-render and never consumed by the
    renderer in Phase 5.
    """

    page_id: str
    version: str
    runtime_class: CleanupClass
    status_records: list[dict[str, Any]] = field(default_factory=list)
    result_records: list[CleanupResult] = field(default_factory=list)
    proof_records: list[CleanupProof] = field(default_factory=list)
    parent_cleanup_unit_aggregate_records: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_audit_dict(self) -> dict[str, Any]:
        passed = [
            proof
            for proof in self.proof_records
            if proof.proof_status == ProofStatus.PASSED
        ]
        pixel_changed_passed = [
            proof
            for proof in passed
            for result in self.result_records
            if result.cleanup_result_id == proof.cleanup_result_id and bool(result.pixel_changed)
        ]
        return {
            "version": self.version,
            "page_id": self.page_id,
            "renderer_consumed": False,
            "runtime_class": _enum_value(self.runtime_class),
            "status_records": _json_safe(self.status_records),
            "results": [_json_safe(record) for record in self.result_records],
            "proofs": [_json_safe(record) for record in self.proof_records],
            "parent_cleanup_unit_aggregate_records": _json_safe(
                self.parent_cleanup_unit_aggregate_records
            ),
            "errors": list(self.errors),
            "summary": {
                "status_count": len(self.status_records),
                "result_count": len(self.result_records),
                "proof_count": len(self.proof_records),
                "parent_cleanup_unit_aggregate_count": len(
                    self.parent_cleanup_unit_aggregate_records
                ),
                "proof_passed_count": len(passed),
                "proof_passed_pixel_changed_count": len(pixel_changed_passed),
                "renderer_consumed": False,
            },
        }


@dataclass(frozen=True)
class CleanupUpstreamCommitResult:
    """Cleanup-owned page image committed before renderer invocation."""

    page_id: str
    version: str
    cleaned_image: Any = field(repr=False, compare=False)
    committed_image_ref: str = ""
    commit_diff_ref: str = ""
    commit_mask_ref: str = ""
    commit_records: list[dict[str, Any]] = field(default_factory=list)
    blocked_records: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def committed_region_ids(self) -> list[str]:
        return sorted(
            {
                str(record.get("region_id") or "")
                for record in self.commit_records
                if record.get("cleanup_committed_to_working_image") and str(record.get("region_id") or "")
            }
        )

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "page_id": self.page_id,
            "renderer_consumed": False,
            "cleanup_applied_upstream": bool(self.commit_records),
            "cleanup_committed_to_working_image": bool(self.commit_records),
            "cleanup_upstream_committed_region_ids": self.committed_region_ids,
            "cleanup_upstream_cleaned_image_ref": self.committed_image_ref,
            "cleanup_upstream_diff_ref": self.commit_diff_ref,
            "cleanup_upstream_mask_ref": self.commit_mask_ref,
            "commit_records": _json_safe(self.commit_records),
            "blocked_records": _json_safe(self.blocked_records),
            "errors": list(self.errors),
            "summary": {
                "committed_count": len(self.commit_records),
                "blocked_count": len(self.blocked_records),
                "error_count": len(self.errors),
                "renderer_consumed": False,
            },
        }


def build_cleanup_plans(
    *,
    page_id: str,
    job_candidates: Sequence[CleanupJob] | Any,
    mask_contracts: CleanupMaskBuildResult | Sequence[CleanupMask] | Any,
    image_size: tuple[int, int] | None = None,
    source_image_path: str | None = None,
    source_image: Any | None = None,
    render_eligibility: Any | None = None,
) -> CleanupPlanBuildResult:
    """Build cleanup-unit plans for accepted cleanup-owned masks."""

    plan_started = time.time()
    jobs = _extract_jobs(job_candidates)
    render_eligibility_by_region = _render_eligibility_by_region_id(render_eligibility)
    mask_result = _extract_masks(mask_contracts)
    masks_by_job_id: dict[str, list[CleanupMask]] = {}
    for cleanup_mask in mask_result:
        masks_by_job_id.setdefault(str(cleanup_mask.cleanup_job_id), []).append(cleanup_mask)
    _cleanup_perf_contract_checkpoint(
        "cleanup_plan_build",
        "module_start",
        page_id=page_id,
        job_count=len(jobs),
        mask_count=len(mask_result),
        source_image_path=source_image_path or "",
    )

    plans: list[CleanupPlan] = []
    plans_by_job_id: dict[str, list[CleanupPlan]] = {}
    rejected_records: list[dict[str, Any]] = []
    protected_records: list[dict[str, Any]] = []
    skipped_records: list[dict[str, Any]] = []
    errors: list[str] = []
    source_np, source_error = _source_image_array(source_image=source_image, source_image_path=source_image_path)
    backend_candidates = cleanup_backend_runner.inventory_local_cleanup_backends()
    backend_inventory = cleanup_backend_runner.inventory_to_audit_dict(backend_candidates)

    for job in jobs:
        try:
            base = _job_base_record(page_id, job)
            suppressed_decision = _source_grounding_suppressed_decision(job, render_eligibility_by_region)
            if suppressed_decision is not None:
                skipped_records.append(
                    {
                        **base,
                        "reason": "render_eligibility_suppressed_source_ungrounded",
                        "source_grounding_status": "suppressed_source_ungrounded",
                        "source_grounding_failure_reason": _decision_value(suppressed_decision, "reason"),
                        "legacy_renderer_behavior_preserved": False,
                    }
                )
                continue
            blocked_reason = _job_exclusion_reason(job)
            if blocked_reason:
                protected_records.append({**base, "reason": blocked_reason})
                continue
            candidates = masks_by_job_id.get(str(job.cleanup_job_id), [])
            if len(candidates) != 1:
                _cleanup_perf_contract_checkpoint(
                    "cleanup_plan_mask_filter",
                    "rejected",
                    page_id=page_id,
                    cleanup_job_id=str(job.cleanup_job_id),
                    reason="expected_exactly_one_non_protected_cleanup_mask",
                    matching_mask_count=len(candidates),
                )
                rejected_records.append(
                    {
                        **base,
                        "reason": "expected_exactly_one_non_protected_cleanup_mask",
                        "matching_mask_count": len(candidates),
                    }
                )
                continue
            cleanup_mask = candidates[0]
            mask_rejection = _mask_exclusion_reason(cleanup_mask, image_size)
            if mask_rejection and not _is_partitionable_mask_rejection(mask_rejection):
                _cleanup_perf_contract_checkpoint(
                    "cleanup_plan_mask_filter",
                    "rejected",
                    page_id=page_id,
                    cleanup_job_id=str(job.cleanup_job_id),
                    cleanup_mask_id=str(cleanup_mask.cleanup_mask_id),
                    reason=mask_rejection,
                    visual_scope=str(cleanup_mask.visual_scope or ""),
                    foreground_pixels=cleanup_mask.foreground_mask_pixels,
                    erase_pixels=cleanup_mask.erase_mask_pixels,
                    allowed_area=cleanup_mask.allowed_area,
                    consumed_source_glyph_count=len(cleanup_mask.consumed_source_glyph_mask_ids or []),
                    missing_source_glyph_count=len(cleanup_mask.missing_source_glyph_mask_ids or []),
                )
                rejected_records.append({**base, "cleanup_mask_id": cleanup_mask.cleanup_mask_id, "reason": mask_rejection})
                continue

            flat_rejection, flat_metrics = _flat_background_pre_execution_rejection(
                cleanup_mask=cleanup_mask,
                source_np=source_np,
                source_error=source_error,
            )
            cleanup_tag = _runtime_cleanup_tag_for_class(job.cleanup_class)
            cleanup_class_value = _enum_value(job.cleanup_class)

            for candidate in backend_candidates:
                cleanup_plan_id = (
                    f"cplan_{_safe_id(page_id)}_{_safe_id(job.cleanup_job_id)}_"
                    f"{_safe_id(candidate.candidate_id)}"
                )
                plan = CleanupPlan(
                    cleanup_plan_id=cleanup_plan_id,
                    cleanup_job_id=str(job.cleanup_job_id),
                    cleanup_mask_id=str(cleanup_mask.cleanup_mask_id),
                    cleanup_class=job.cleanup_class,
                    selected_backend=f"app.pipeline.cleanup_backend_runner.{candidate.candidate_id}",
                    cleanup_method=f"{cleanup_class_value}_glyph_local_candidate",
                    backend_parameters={
                        "cleanup_tag": cleanup_tag,
                        "target_region_ids": list(job.target_region_ids or []),
                        "allowed_area": list(cleanup_mask.allowed_area or []),
                        "renderer_consumed_by_phase": "phase5_phase6_cleanup_contract_status_only",
                        "eligibility_reason": "cleanup_unit_sourceglyph_foreground_mask_contract",
                        "mask_contract_exception_reason": cleanup_mask.mask_contract_exception_reason,
                        "pre_execution_risk_reason": mask_rejection or flat_rejection,
                        "partition_strategy_required": bool(mask_rejection),
                        "flatness_metrics": flat_metrics,
                        "backend_candidate_id": candidate.candidate_id,
                        "backend_family": candidate.backend_family,
                        "backend_model_path": candidate.model_path,
                        "backend_adapter_path": candidate.adapter_path,
                        "backend_candidate_available": candidate.available,
                        "backend_candidate_unavailable_reason": candidate.unavailable_reason,
                    },
                    inpaint_mode="glyph_local_candidate",
                    crop_context_bbox=list(cleanup_mask.erase_mask_bbox or cleanup_mask.allowed_area or []),
                    fallback_policy=["runtime_adaptive_cleanup_contract"],
                    expected_runtime_class="local_candidate",
                    proof_thresholds={
                        "mask_source_overlap_ratio_min": PROOF_MASK_SOURCE_OVERLAP_RATIO,
                        "source_residual_ratio_max": PROOF_RESIDUAL_RATIO,
                        "source_residual_pixel_fraction_max": PROOF_RESIDUAL_PIXEL_FRACTION,
                        "changed_outside_allowed_pixels_max": PROOF_CHANGED_OUTSIDE_ALLOWED_PIXELS,
                        "changed_outside_allowed_ratio_max": PROOF_CHANGED_OUTSIDE_ALLOWED_RATIO,
                        "changed_outside_erase_inside_allowed_ratio_max": PROOF_COLLATERAL_INSIDE_ALLOWED_RATIO,
                    },
                )
                plans.append(plan)
                plans_by_job_id.setdefault(str(job.cleanup_job_id), []).append(plan)
                _cleanup_perf_contract_checkpoint(
                    "cleanup_plan_build",
                    "plan_created",
                    page_id=page_id,
                    cleanup_job_id=str(job.cleanup_job_id),
                    cleanup_mask_id=str(cleanup_mask.cleanup_mask_id),
                    cleanup_plan_id=str(plan.cleanup_plan_id),
                    selected_backend=str(plan.selected_backend),
                    visual_scope=str(cleanup_mask.visual_scope or ""),
                    expected_runtime_class=plan.expected_runtime_class,
                )
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")

    _cleanup_perf_contract_checkpoint(
        "cleanup_plan_build",
        "module_end",
        page_id=page_id,
        plan_count=len(plans),
        rejected_count=len(rejected_records),
        protected_count=len(protected_records),
        skipped_count=len(skipped_records),
        error_count=len(errors),
        elapsed_ms=round((time.time() - plan_started) * 1000.0, 3),
    )
    return CleanupPlanBuildResult(
        page_id=page_id,
        version=CLEANUP_PLAN_CONTRACT_VERSION,
        plans=plans,
        plans_by_job_id=plans_by_job_id,
        rejected_records=rejected_records,
        protected_records=protected_records,
        skipped_records=skipped_records,
        errors=errors,
        backend_inventory=backend_inventory,
    )


def execute_cleanup_plan(
    *,
    image: Any,
    cleanup_plan: CleanupPlan,
    cleanup_mask: CleanupMask,
    use_gpu: bool,
    model_id: str,
) -> CleanupResult:
    """Execute one already-eligible pilot plan on a scratch image."""

    params = cleanup_plan.backend_parameters or {}
    candidate = cleanup_backend_runner.LocalCleanupBackendCandidate(
        candidate_id=str(params.get("backend_candidate_id") or cleanup_plan.selected_backend),
        backend_family=str(params.get("backend_family") or ""),
        model_path=str(params.get("backend_model_path") or ""),
        adapter_path=str(params.get("backend_adapter_path") or ""),
        available=bool(params.get("backend_candidate_available")),
        unavailable_reason=str(params.get("backend_candidate_unavailable_reason") or ""),
    )
    candidate_result = cleanup_backend_runner.run_cleanup_backend_candidate(
        image=image,
        mask=cleanup_mask.erase_mask,
        candidate=candidate,
        use_gpu=use_gpu,
    )
    cleaned_image = candidate_result.cleaned_image or image
    operation_bbox = _valid_bbox(candidate_result.crop_bbox) or _valid_bbox(cleanup_mask.erase_mask_bbox)
    return CleanupResult(
        cleanup_result_id=f"cres_{_safe_id(cleanup_plan.cleanup_plan_id)}",
        cleanup_plan_id=str(cleanup_plan.cleanup_plan_id),
        cleanup_job_id=str(cleanup_plan.cleanup_job_id),
        cleanup_mask_id=str(cleanup_plan.cleanup_mask_id),
        operation_bbox=operation_bbox,
        mask_stats=mask_stats(cleanup_mask.erase_mask) or {},
        backend_name=candidate_result.backend_name,
        backend_parameters={
            "backend_detail": candidate_result.detail,
            "backend_candidate_id": candidate_result.candidate_id,
            "backend_family": candidate_result.backend_family,
            "backend_model_path": candidate_result.model_path,
            "backend_adapter_path": candidate_result.adapter_path,
            "candidate_status": candidate_result.status,
            "load_time_ms": candidate_result.load_time_ms,
            "cleanup_tag": "caption_flat_background_phase3c",
            "crop_bbox": candidate_result.crop_bbox,
            "crop_area": candidate_result.crop_area,
            "mask_ratio": candidate_result.mask_ratio,
            "mask_pixels": candidate_result.mask_pixels,
        },
        runtime_ms=candidate_result.runtime_ms,
        fallback_status=candidate_result.status,
        errors=list(candidate_result.errors or []),
        cleaned_image=cleaned_image,
        cleaned_crop=candidate_result.cleaned_crop,
    )


def prove_cleanup_result(
    *,
    source_image: Any,
    before_image: Any,
    cleanup_result: CleanupResult,
    cleanup_plan: CleanupPlan,
    cleanup_mask: CleanupMask,
) -> CleanupProof:
    """Proof one pilot cleanup before translated text drawing."""

    metrics: dict[str, Any] = {}
    failure_reason = "passed"
    proof_status = ProofStatus.PASSED
    try:
        if np is None:
            raise RuntimeError("numpy_unavailable")
        source_np = np.asarray(source_image.convert("RGB") if hasattr(source_image, "convert") else source_image)
        before_np = np.asarray(before_image.convert("RGB") if hasattr(before_image, "convert") else before_image)
        cleaned_image = cleanup_result.cleaned_image
        cleaned_np = np.asarray(cleaned_image.convert("RGB") if hasattr(cleaned_image, "convert") else cleaned_image)
        foreground = _binary_mask(cleanup_mask.foreground_mask, source_np.shape[:2])
        erase = _binary_mask(cleanup_mask.erase_mask, source_np.shape[:2])
        allowed = _bbox_to_mask(cleanup_mask.allowed_area, source_np.shape[:2])
        if foreground is None or not np.any(foreground):
            return _proof(
                cleanup_result,
                cleanup_plan,
                status=ProofStatus.INCONCLUSIVE,
                failure_reason="inconclusive_missing_evidence",
                metrics={"missing": "source_foreground_evidence"},
            )
        if erase is None or not np.any(erase) or allowed is None:
            return _proof(
                cleanup_result,
                cleanup_plan,
                status=ProofStatus.INCONCLUSIVE,
                failure_reason="inconclusive_missing_evidence",
                metrics={"missing": "erase_mask_or_allowed_area"},
            )

        foreground_pixels = int(np.count_nonzero(foreground))
        erase_foreground_overlap = int(np.count_nonzero((erase > 0) & (foreground > 0)))
        mask_source_overlap_ratio = erase_foreground_overlap / max(1, foreground_pixels)

        source_gray = _gray(source_np)
        cleaned_gray = _gray(cleaned_np)
        before_gray = _gray(before_np)
        diff_gray = np.abs(cleaned_gray.astype(np.int16) - before_gray.astype(np.int16))
        changed = diff_gray > 8
        legacy_residual = _source_residual_mask(source_gray, cleaned_gray, foreground)
        legacy_residual_pixels = int(np.count_nonzero(legacy_residual))
        legacy_residual_ratio = legacy_residual_pixels / max(1, foreground_pixels)
        source_dark_foreground = (foreground > 0) & (source_gray < 210)
        source_light_foreground = (foreground > 0) & ~(source_gray < 210)
        source_dark_foreground_pixels = int(np.count_nonzero(source_dark_foreground))
        source_light_foreground_pixels = int(np.count_nonzero(source_light_foreground))
        baseline_residual_dark = _dark_source_residual_mask(source_gray, before_gray, foreground)
        baseline_residual_dark_pixels = int(np.count_nonzero(baseline_residual_dark))
        baseline_residual_dark_ratio = baseline_residual_dark_pixels / max(1, source_dark_foreground_pixels)
        residual_dark = _dark_source_residual_mask(source_gray, cleaned_gray, foreground)
        residual_dark_pixels = int(np.count_nonzero(residual_dark))
        residual_dark_ratio = residual_dark_pixels / max(1, source_dark_foreground_pixels)
        residual_light_unchanged_pixels = int(
            np.count_nonzero(legacy_residual & source_light_foreground)
        )
        params = cleanup_plan.backend_parameters or {}
        proof_scope = _proof_scope_metadata(cleanup_plan, cleanup_mask)
        erasure_box = proof_scope.get("proof_residual_measurement_bbox")
        visual_residual: Any | None = None
        visual_residual_dark_pixels = 0
        visual_residual_pixels = 0
        visual_residual_ratio = 0.0
        visual_residual_baseline_pixels = 0
        visual_residual_baseline_ratio = 0.0
        if erasure_box is not None:
            visual_residual_baseline = _source_erasure_box_residual_mask(
                source_gray,
                before_gray,
                erasure_box,
            )
            visual_residual = _source_erasure_box_residual_mask(
                source_gray,
                cleaned_gray,
                erasure_box,
            )
            visual_erasure_region = _bbox_to_mask(erasure_box, source_np.shape[:2])
            visual_source_dark = (visual_erasure_region > 0) & (source_gray < 135)
            visual_residual_dark_pixels = int(np.count_nonzero(visual_source_dark))
            visual_residual_baseline_pixels = int(np.count_nonzero(visual_residual_baseline))
            visual_residual_pixels = int(np.count_nonzero(visual_residual))
            if visual_residual_dark_pixels >= 20:
                visual_residual_baseline_ratio = (
                    visual_residual_baseline_pixels / max(1, visual_residual_dark_pixels)
                )
                visual_residual_ratio = visual_residual_pixels / max(1, visual_residual_dark_pixels)

        outside_allowed = changed & ~(allowed > 0)
        changed_outside_allowed_pixels = int(np.count_nonzero(outside_allowed))
        changed_outside_allowed_ratio = changed_outside_allowed_pixels / max(1, int(source_np.shape[0] * source_np.shape[1]))
        allowed_changed = changed & (allowed > 0)
        outside_erase_inside_allowed = changed & (allowed > 0) & ~(erase > 0)
        collateral_pixels = int(np.count_nonzero(outside_erase_inside_allowed))
        collateral_ratio = collateral_pixels / max(1, int(np.count_nonzero(allowed_changed)))
        changed_inside_erase = changed & (erase > 0) & (allowed > 0)
        changed_inside_erase_pixels = int(np.count_nonzero(changed_inside_erase))
        white_patch_pixels = int(
            np.count_nonzero(changed_inside_erase & (cleaned_gray > 245) & (source_gray < 180))
        )
        white_patch_ratio = white_patch_pixels / max(1, changed_inside_erase_pixels)
        cleanup_class_value = _enum_value(cleanup_plan.cleanup_class)
        broad_white_patch_risk = (
            cleanup_class_value not in {
                CleanupClass.SPEECH_FLAT_BUBBLE.value,
                CleanupClass.SPEECH_COMPLEX_BUBBLE.value,
            }
            and white_patch_pixels > max(96, int(foreground_pixels * 0.45))
            and white_patch_ratio > 0.70
        )
        residual_pixel_limit = max(24, int(source_dark_foreground_pixels * PROOF_RESIDUAL_PIXEL_FRACTION))
        residual_mask_ref, residual_mask_missing_reason = _write_residual_source_mask(
            cleanup_result=cleanup_result,
            residual_mask=residual_dark,
        )

        backend_name = str(cleanup_result.backend_name or "")
        candidate_status = str((cleanup_result.backend_parameters or {}).get("candidate_status") or "")
        backend_noop = (
            bool(cleanup_result.errors)
            or backend_name in {"", "none", "error"}
            or candidate_status not in {"completed"}
        )
        metrics = {
            "source_foreground_pixels": foreground_pixels,
            "erase_foreground_overlap_pixels": erase_foreground_overlap,
            "mask_source_overlap_ratio": round(mask_source_overlap_ratio, 4),
            "residual_pass_metric": "dark_source_foreground",
            "source_dark_foreground_pixels": source_dark_foreground_pixels,
            "source_light_foreground_pixels": source_light_foreground_pixels,
            "source_residual_baseline_dark_pixels": baseline_residual_dark_pixels,
            "source_residual_baseline_dark_ratio": round(baseline_residual_dark_ratio, 4),
            "residual_dark_source_pixels": residual_dark_pixels,
            "residual_dark_source_ratio": round(residual_dark_ratio, 4),
            "source_residual_delta_dark_pixels": int(baseline_residual_dark_pixels - residual_dark_pixels),
            "source_residual_delta_dark_ratio": round(
                baseline_residual_dark_ratio - residual_dark_ratio,
                4,
            ),
            "legacy_residual_source_pixels": legacy_residual_pixels,
            "legacy_residual_source_ratio": round(legacy_residual_ratio, 4),
            "residual_source_pixels": legacy_residual_pixels,
            "residual_source_ratio": round(legacy_residual_ratio, 4),
            "residual_light_unchanged_foreground_pixels": residual_light_unchanged_pixels,
            "proof_scope": proof_scope.get("proof_scope"),
            "proof_scope_bbox": proof_scope.get("proof_scope_bbox"),
            "proof_scope_uses_parent_expected_bbox": proof_scope.get(
                "proof_scope_uses_parent_expected_bbox"
            ),
            "proof_scope_excludes_parent_expected_bbox": proof_scope.get(
                "proof_scope_excludes_parent_expected_bbox"
            ),
            "parent_cleanup_unit_id": proof_scope.get("parent_cleanup_unit_id"),
            "parent_expected_erasure_bbox": proof_scope.get("parent_expected_erasure_bbox"),
            "source_glyph_erasure_bbox_input": proof_scope.get("source_glyph_erasure_bbox_input"),
            "source_glyph_erasure_expected_area_bbox": proof_scope.get(
                "source_glyph_erasure_expected_area_bbox"
            ),
            "source_glyph_erasure_bbox": erasure_box,
            "source_glyph_erasure_visual_dark_pixels": visual_residual_dark_pixels,
            "source_glyph_erasure_visual_residual_baseline_pixels": visual_residual_baseline_pixels,
            "source_glyph_erasure_visual_residual_baseline_ratio": round(
                visual_residual_baseline_ratio,
                4,
            ),
            "source_glyph_erasure_visual_residual_pixels": visual_residual_pixels,
            "source_glyph_erasure_visual_residual_ratio": round(visual_residual_ratio, 4),
            "source_glyph_erasure_visual_residual_delta_pixels": int(
                visual_residual_baseline_pixels - visual_residual_pixels
            ),
            "source_glyph_erasure_visual_residual_delta_ratio": round(
                visual_residual_baseline_ratio - visual_residual_ratio,
                4,
            ),
            "residual_source_pixel_limit": residual_pixel_limit,
            "changed_outside_allowed_pixels": changed_outside_allowed_pixels,
            "changed_outside_allowed_ratio": round(changed_outside_allowed_ratio, 6),
            "changed_outside_erase_inside_allowed_pixels": collateral_pixels,
            "changed_outside_erase_inside_allowed_ratio": round(collateral_ratio, 4),
            "changed_inside_erase_pixels": changed_inside_erase_pixels,
            "broad_white_patch_pixels": white_patch_pixels,
            "broad_white_patch_ratio": round(white_patch_ratio, 4),
            "broad_white_patch_risk": bool(broad_white_patch_risk),
            "collateral_art_damage_delta": round(collateral_ratio, 4),
            "backend_name": backend_name,
            "backend_candidate_id": (cleanup_result.backend_parameters or {}).get("backend_candidate_id"),
            "candidate_status": candidate_status,
            "pixel_changed": bool(cleanup_result.pixel_changed),
            "changed_pixel_count": int(cleanup_result.changed_pixel_count or 0),
            "effective_mask_status": str(getattr(cleanup_mask, "effective_mask_status", "") or ""),
            "effective_mask_failure_reason": str(getattr(cleanup_mask, "effective_mask_failure_reason", "") or ""),
            "mask_completion_method": str(getattr(cleanup_mask, "mask_completion_method", "") or ""),
            "polarity_mode": str(getattr(cleanup_mask, "polarity_mode", "") or ""),
            "completed_foreground_pixels": getattr(cleanup_mask, "completed_foreground_pixels", None),
            "text_block_coverage_estimate": getattr(cleanup_mask, "text_block_coverage_estimate", None),
        }
        if residual_mask_ref:
            metrics["residual_source_mask_ref"] = residual_mask_ref
        else:
            metrics["residual_source_mask_ref_missing_reason"] = residual_mask_missing_reason
        if backend_noop:
            failure_reason = "backend_noop_or_error"
        elif mask_source_overlap_ratio < PROOF_MASK_SOURCE_OVERLAP_RATIO:
            failure_reason = "mask_missing_source_pixels"
        elif source_dark_foreground_pixels <= 0:
            failure_reason = "inconclusive_missing_dark_source_foreground"
        elif not (
            residual_dark_ratio < PROOF_RESIDUAL_RATIO
            or residual_dark_pixels < residual_pixel_limit
        ):
            failure_reason = "source_residual_remaining"
        elif (
            visual_residual is not None
            and visual_residual_dark_pixels >= 20
            and visual_residual_ratio >= PROOF_RESIDUAL_RATIO
        ):
            failure_reason = (
                "source_residual_remaining_in_component_partition_bbox"
                if proof_scope.get("proof_scope") == "component_partition"
                else "source_residual_remaining_in_source_glyph_erasure_bbox"
            )
        elif (
            changed_outside_allowed_pixels > PROOF_CHANGED_OUTSIDE_ALLOWED_PIXELS
            or changed_outside_allowed_ratio > PROOF_CHANGED_OUTSIDE_ALLOWED_RATIO
        ):
            failure_reason = "changed_outside_allowed_area"
        elif collateral_ratio > PROOF_COLLATERAL_INSIDE_ALLOWED_RATIO:
            failure_reason = "collateral_change_too_broad"
        else:
            failure_reason = "passed"

        if failure_reason == "inconclusive_missing_dark_source_foreground":
            proof_status = ProofStatus.INCONCLUSIVE
        elif failure_reason != "passed":
            proof_status = ProofStatus.FAILED

        source_residual_improved = (
            residual_dark_pixels < baseline_residual_dark_pixels
            or (
                visual_residual_baseline_pixels > 0
                and visual_residual_pixels < visual_residual_baseline_pixels
            )
        )
        source_visible_area_increased = (
            residual_dark_pixels > baseline_residual_dark_pixels + 4
            or (
                visual_residual_baseline_pixels > 0
                and visual_residual_pixels > visual_residual_baseline_pixels + 4
            )
        )
        allowed_area_safe = (
            changed_outside_allowed_pixels <= PROOF_CHANGED_OUTSIDE_ALLOWED_PIXELS
            and changed_outside_allowed_ratio <= PROOF_CHANGED_OUTSIDE_ALLOWED_RATIO
        )
        collateral_safe = collateral_ratio <= PROOF_COLLATERAL_INSIDE_ALLOWED_RATIO
        degraded_commit_allowed = (
            proof_status == ProofStatus.FAILED
            and bool(cleanup_result.pixel_changed)
            and str(failure_reason or "").startswith("source_residual")
            and source_residual_improved
            and not source_visible_area_increased
            and allowed_area_safe
            and collateral_safe
            and not broad_white_patch_risk
            and mask_source_overlap_ratio >= PROOF_MASK_SOURCE_OVERLAP_RATIO
        )
        if failure_reason == "passed":
            proof_quality_state = "proof_passed_clean"
        elif degraded_commit_allowed:
            proof_quality_state = "proof_passed_with_minor_residual"
        elif str(failure_reason or "").startswith("source_residual"):
            proof_quality_state = "proof_failed_residual_warning"
        elif failure_reason in {"changed_outside_allowed_area", "collateral_change_too_broad"}:
            proof_quality_state = "proof_failed_collateral_warning"
        else:
            proof_quality_state = "proof_failed_warning"
        metrics.update(
            {
                "source_residual_improved": bool(source_residual_improved),
                "source_visible_area_increased": bool(source_visible_area_increased),
                "allowed_area_safe": bool(allowed_area_safe),
                "collateral_safe": bool(collateral_safe),
                "degraded_commit_allowed": bool(degraded_commit_allowed),
                "proof_quality_state": proof_quality_state,
                "proof_thresholds_unchanged": True,
                "erase_limits_unchanged": True,
            }
        )
    except Exception as exc:
        proof_status = ProofStatus.INCONCLUSIVE
        failure_reason = "inconclusive_missing_evidence"
        metrics = {"exception": f"{type(exc).__name__}: {exc}"}

    return _proof(
        cleanup_result,
        cleanup_plan,
        status=proof_status,
        failure_reason=failure_reason,
        metrics=metrics,
    )


def _proof_scope_metadata(cleanup_plan: CleanupPlan, cleanup_mask: CleanupMask) -> dict[str, Any]:
    params = cleanup_plan.backend_parameters or {}
    visual_scope = _normalise_cleanup_visual_scope(getattr(cleanup_mask, "visual_scope", ""))
    partitioned = (
        _truthy(params.get("partitioned_cleanup"))
        or str(params.get("proof_scope") or "") == "component_partition"
        or visual_scope == "source_glyph_union_partition"
    )
    source_bbox = _valid_bbox(params.get("source_glyph_erasure_bbox")) or _valid_bbox(
        getattr(cleanup_mask, "erase_mask_bbox", None)
    )
    expected_bbox = _valid_bbox(params.get("source_glyph_erasure_expected_area_bbox"))
    parent_expected = _valid_bbox(params.get("parent_expected_erasure_bbox"))
    if partitioned:
        proof_bbox = _valid_bbox(params.get("proof_scope_bbox")) or source_bbox
        return {
            "proof_scope": "component_partition",
            "proof_scope_bbox": proof_bbox,
            "proof_residual_measurement_bbox": proof_bbox,
            "proof_scope_uses_parent_expected_bbox": False,
            "proof_scope_excludes_parent_expected_bbox": bool(parent_expected or expected_bbox),
            "parent_cleanup_unit_id": str(
                params.get("parent_cleanup_unit_id")
                or params.get("cleanup_unit_id")
                or cleanup_plan.cleanup_job_id
                or ""
            ),
            "parent_expected_erasure_bbox": parent_expected or expected_bbox,
            "source_glyph_erasure_bbox_input": source_bbox,
            "source_glyph_erasure_expected_area_bbox": expected_bbox,
        }
    proof_bbox = _valid_bbox(params.get("proof_scope_bbox")) or _union_valid_bboxes(
        source_bbox,
        expected_bbox,
    )
    return {
        "proof_scope": "cleanup_unit_full",
        "proof_scope_bbox": proof_bbox,
        "proof_residual_measurement_bbox": proof_bbox,
        "proof_scope_uses_parent_expected_bbox": bool(expected_bbox),
        "proof_scope_excludes_parent_expected_bbox": False,
        "parent_cleanup_unit_id": str(
            params.get("parent_cleanup_unit_id")
            or params.get("cleanup_unit_id")
            or cleanup_plan.cleanup_job_id
            or ""
        ),
        "parent_expected_erasure_bbox": parent_expected or expected_bbox,
        "source_glyph_erasure_bbox_input": source_bbox,
        "source_glyph_erasure_expected_area_bbox": expected_bbox,
    }


def run_caption_flat_background_pilot(
    *,
    page_id: str,
    image: Any,
    source_image: Any,
    plan_contracts: CleanupPlanBuildResult | Any,
    mask_contracts: CleanupMaskBuildResult | Any,
    use_gpu: bool,
    model_id: str,
) -> CaptionFlatPilotRuntimeResult:
    """Run only already-decided caption-flat pilot plans."""

    working = image
    status_records: list[dict[str, Any]] = []
    result_records: list[CleanupResult] = []
    proof_records: list[CleanupProof] = []
    parent_cleanup_unit_aggregate_records: list[dict[str, Any]] = []
    errors: list[str] = []
    plans = _extract_plans(plan_contracts)
    masks_by_id = {str(mask.cleanup_mask_id): mask for mask in _extract_masks(mask_contracts)}
    plans_by_job_id: dict[str, list[CleanupPlan]] = {}
    for cleanup_plan in plans:
        plans_by_job_id.setdefault(str(cleanup_plan.cleanup_job_id), []).append(cleanup_plan)

    for cleanup_job_id, candidate_plans in plans_by_job_id.items():
        cleanup_plan = candidate_plans[0]
        target_region_ids = [
            str(rid)
            for rid in (cleanup_plan.backend_parameters or {}).get("target_region_ids", [])
            if str(rid)
        ]
        cleanup_mask = masks_by_id.get(str(cleanup_plan.cleanup_mask_id))
        if cleanup_mask is None:
            status_records.append(
                _status_record(
                    cleanup_plan,
                    target_region_ids=target_region_ids,
                    status="inconclusive",
                    failure_class="inconclusive_missing_evidence",
                    renderer_action="suppress_translated_text_keep_source_visible",
                    detail="cleanup_mask_contract_missing_at_runtime",
                )
            )
            continue
        before = working
        job_results: list[CleanupResult] = []
        job_proofs: list[CleanupProof] = []
        try:
            for candidate_plan in candidate_plans:
                cleanup_result = execute_cleanup_plan(
                    image=before,
                    cleanup_plan=candidate_plan,
                    cleanup_mask=cleanup_mask,
                    use_gpu=use_gpu,
                    model_id=model_id,
                )
                cleanup_proof = prove_cleanup_result(
                    source_image=source_image,
                    before_image=before,
                    cleanup_result=cleanup_result,
                    cleanup_plan=candidate_plan,
                    cleanup_mask=cleanup_mask,
                )
                job_results.append(cleanup_result)
                job_proofs.append(cleanup_proof)
                result_records.append(cleanup_result)
                proof_records.append(cleanup_proof)

            selected_result, selected_proof = _select_best_passed_result(job_results, job_proofs)
            if selected_result is not None and selected_proof is not None:
                working = selected_result.cleaned_image
                status = "passed"
                renderer_action = "commit_pilot_cleanup_draw_translation"
                failure_class = "passed"
                status_plan = _plan_for_result(candidate_plans, selected_result) or cleanup_plan
                cleanup_result_id = selected_result.cleanup_result_id
                cleanup_proof_id = selected_proof.cleanup_proof_id
                proof_metrics = selected_proof.metrics
                selected_backend_candidate_id = (selected_result.backend_parameters or {}).get("backend_candidate_id")
                detail = "selected_proof_passed_backend_candidate"
            else:
                selected_result, selected_proof = _best_failed_attempt(job_results, job_proofs)
                status = "failed" if job_proofs else "inconclusive"
                renderer_action = "discard_pilot_cleanup_suppress_translation"
                failure_class = (
                    selected_proof.failure_reason
                    if selected_proof is not None and selected_proof.failure_reason
                    else "inconclusive_missing_evidence"
                )
                status_plan = _plan_for_result(candidate_plans, selected_result) if selected_result is not None else cleanup_plan
                if status_plan is None:
                    status_plan = cleanup_plan
                cleanup_result_id = selected_result.cleanup_result_id if selected_result is not None else None
                cleanup_proof_id = selected_proof.cleanup_proof_id if selected_proof is not None else None
                proof_metrics = selected_proof.metrics if selected_proof is not None else {}
                selected_backend_candidate_id = (
                    (selected_result.backend_parameters or {}).get("backend_candidate_id")
                    if selected_result is not None
                    else None
                )
                detail = "no_backend_candidate_proof_passed"
            status_records.append(
                _status_record(
                    status_plan,
                    target_region_ids=target_region_ids,
                    status=status,
                    failure_class=failure_class,
                    renderer_action=renderer_action,
                    cleanup_result_id=cleanup_result_id,
                    cleanup_proof_id=cleanup_proof_id,
                    allowed_area=cleanup_mask.allowed_area,
                    erase_mask_bbox=cleanup_mask.erase_mask_bbox,
                    proof_metrics=proof_metrics,
                    detail=detail,
                    candidate_result_count=len(job_results),
                    selected_backend_candidate_id=selected_backend_candidate_id,
                )
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            errors.append(message)
            status_records.append(
                _status_record(
                    cleanup_plan,
                    target_region_ids=target_region_ids,
                    status="inconclusive",
                    failure_class="inconclusive_missing_evidence",
                    renderer_action="suppress_translated_text_keep_source_visible",
                    detail=message,
                )
            )

    return CaptionFlatPilotRuntimeResult(
        page_id=page_id,
        version=PILOT_RUNTIME_VERSION,
        cleaned_image=working,
        status_records=status_records,
        result_records=result_records,
        proof_records=proof_records,
        errors=errors,
    )


def run_cleanup_runtime_contract(
    *,
    page_id: str,
    image: Any,
    source_image: Any,
    job_candidates: Sequence[CleanupJob] | Any,
    mask_contracts: CleanupMaskBuildResult | Sequence[CleanupMask] | Any,
    render_eligibility: Any | None,
    use_gpu: bool,
    model_id: str,
    artifact_dir: str | None = None,
    inpaint_mode: str = "fast",
    runtime_class: CleanupClass | None = None,
) -> CleanupRuntimeContractResult:
    """Run Phase 5 cleanup-owned runtime proof without renderer consumption."""

    runtime_started = time.time()
    status_records: list[dict[str, Any]] = []
    result_records: list[CleanupResult] = []
    proof_records: list[CleanupProof] = []
    parent_cleanup_unit_aggregate_records: list[dict[str, Any]] = []
    errors: list[str] = []
    jobs = _extract_jobs(job_candidates)
    masks = _extract_masks(mask_contracts)
    masks_by_job_id: dict[str, list[CleanupMask]] = {}
    for cleanup_mask in masks:
        masks_by_job_id.setdefault(str(cleanup_mask.cleanup_job_id), []).append(cleanup_mask)
    render_eligibility_by_region = _render_eligibility_by_region_id(render_eligibility)
    _page014_timeout_checkpoint(
        "cleanup_runtime_contract",
        "start",
        page_id=page_id,
        job_count=len(jobs),
        mask_count=len(masks),
        artifact_dir=artifact_dir or "",
    )
    if artifact_dir:
        try:
            os.makedirs(artifact_dir, exist_ok=True)
        except Exception as exc:
            errors.append(f"artifact_dir:{type(exc).__name__}: {exc}")

    for job in jobs:
        job_started = time.time()
        planned_runtime_class = _runtime_cleanup_class_for_job(job, runtime_class)
        base_record = _runtime_base_record(
            page_id,
            job,
            runtime_class=planned_runtime_class or _enum_value(job.cleanup_class),
        )
        target_region_ids = [str(rid) for rid in job.target_region_ids or [] if str(rid)]
        _page014_timeout_checkpoint(
            "cleanup_runtime_job",
            "start",
            page_id=page_id,
            cleanup_job_id=str(job.cleanup_job_id),
            region_id=target_region_ids[0] if target_region_ids else "",
            cleanup_class=_enum_value(job.cleanup_class),
        )
        try:
            suppressed_decision = _source_grounding_suppressed_decision(job, render_eligibility_by_region)
            if suppressed_decision is not None:
                contract_error = _contract_error_for_reason(
                    "render_eligibility_suppressed_source_ungrounded"
                )
                _page014_timeout_checkpoint(
                    "cleanup_runtime_job",
                    "contract_error",
                    page_id=page_id,
                    cleanup_job_id=str(job.cleanup_job_id),
                    region_id=target_region_ids[0] if target_region_ids else "",
                    reason="render_eligibility_suppressed_source_ungrounded",
                    elapsed_ms=round((time.time() - job_started) * 1000.0, 3),
                )
                status_records.append(
                    {
                        **base_record,
                        "runtime_status": "contract_error",
                        "failure_reason": contract_error,
                        "cleanup_outcome_state": contract_error,
                        "contract_error_original_reason": "render_eligibility_suppressed_source_ungrounded",
                        "source_grounding_status": "suppressed_source_ungrounded",
                        "render_consumption_decision_if_consumed": "block_future_renderer_consumption",
            "renderer_consumed": False,
        }
                )
                continue
            if planned_runtime_class is None:
                contract_error = _contract_error_for_reason(
                    "runtime_cleanup_class_not_executable"
                )
                _page014_timeout_checkpoint(
                    "cleanup_runtime_job",
                    "contract_error",
                    page_id=page_id,
                    cleanup_job_id=str(job.cleanup_job_id),
                    region_id=target_region_ids[0] if target_region_ids else "",
                    reason=contract_error,
                    elapsed_ms=round((time.time() - job_started) * 1000.0, 3),
                )
                status_records.append(
                    {
                        **base_record,
                        "runtime_status": "contract_error",
                        "failure_reason": contract_error,
                        "cleanup_outcome_state": contract_error,
                        "contract_error_original_reason": "runtime_cleanup_class_not_executable",
                        "render_consumption_decision_if_consumed": "block_future_renderer_consumption",
                        "renderer_consumed": False,
                    }
                )
                continue
            candidates = masks_by_job_id.get(str(job.cleanup_job_id), [])
            if len(candidates) != 1:
                contract_error = _contract_error_for_reason("expected_exactly_one_cleanup_mask")
                _page014_timeout_checkpoint(
                    "cleanup_runtime_job",
                    "contract_error",
                    page_id=page_id,
                    cleanup_job_id=str(job.cleanup_job_id),
                    region_id=target_region_ids[0] if target_region_ids else "",
                    reason=contract_error,
                    matching_mask_count=len(candidates),
                    elapsed_ms=round((time.time() - job_started) * 1000.0, 3),
                )
                status_records.append(
                    {
                        **base_record,
                        "runtime_status": "contract_error",
                        "failure_reason": contract_error,
                        "cleanup_outcome_state": contract_error,
                        "contract_error_original_reason": "expected_exactly_one_cleanup_mask",
                        "matching_mask_count": len(candidates),
                        "render_consumption_decision_if_consumed": "block_future_renderer_consumption",
                        "renderer_consumed": False,
                    }
                )
                continue
            cleanup_mask = candidates[0]
            exclusion = _runtime_job_exclusion_reason(job, cleanup_mask)
            if exclusion:
                contract_error = _contract_error_for_reason(exclusion)
                _page014_timeout_checkpoint(
                    "cleanup_runtime_job",
                    "contract_error",
                    page_id=page_id,
                    cleanup_job_id=str(job.cleanup_job_id),
                    region_id=target_region_ids[0] if target_region_ids else "",
                    reason=contract_error,
                    elapsed_ms=round((time.time() - job_started) * 1000.0, 3),
                )
                status_records.append(
                    {
                        **base_record,
                        "runtime_status": "contract_error",
                        "failure_reason": contract_error,
                        "cleanup_outcome_state": contract_error,
                        "contract_error_original_reason": exclusion,
                        "render_consumption_decision_if_consumed": "block_future_renderer_consumption",
                        "renderer_consumed": False,
                    }
                )
                continue
            runtime_mask = _runtime_glyph_precision_mask(
                cleanup_mask,
                source_image,
                planned_runtime_class,
            )
            mask_rejection = _mask_exclusion_reason(runtime_mask, None)
            if mask_rejection:
                _cleanup_perf_contract_checkpoint(
                    "cleanup_runtime_mask_filter",
                    "rejected",
                    page_id=page_id,
                    cleanup_job_id=str(job.cleanup_job_id),
                    region_id=target_region_ids[0] if target_region_ids else "",
                    cleanup_mask_id=str(runtime_mask.cleanup_mask_id),
                    reason=mask_rejection,
                    visual_scope=str(runtime_mask.visual_scope or ""),
                    foreground_pixels=runtime_mask.foreground_mask_pixels,
                    erase_pixels=runtime_mask.erase_mask_pixels,
                    allowed_area=runtime_mask.allowed_area,
                    consumed_source_glyph_count=len(runtime_mask.consumed_source_glyph_mask_ids or []),
                    missing_source_glyph_count=len(runtime_mask.missing_source_glyph_mask_ids or []),
                )
                partition_started = time.time()
                _cleanup_perf_contract_checkpoint(
                    "cleanup_partitioning",
                    "start",
                    page_id=page_id,
                    cleanup_job_id=str(job.cleanup_job_id),
                    region_id=target_region_ids[0] if target_region_ids else "",
                    cleanup_mask_id=str(runtime_mask.cleanup_mask_id),
                    parent_rejection_reason=mask_rejection,
                )
                partitioned = _partition_runtime_cleanup_mask(
                    runtime_mask,
                    rejection_reason=mask_rejection,
                    max_components=MAX_PARTITION_COMPONENTS,
                )
                _cleanup_perf_contract_checkpoint(
                    "cleanup_partitioning",
                    "end",
                    page_id=page_id,
                    cleanup_job_id=str(job.cleanup_job_id),
                    region_id=target_region_ids[0] if target_region_ids else "",
                    cleanup_mask_id=str(runtime_mask.cleanup_mask_id),
                    parent_rejection_reason=mask_rejection,
                    partition_count=len(partitioned),
                    elapsed_ms=round((time.time() - partition_started) * 1000.0, 3),
                )
                if partitioned:
                    (
                        partition_results,
                        partition_proofs,
                        partition_statuses,
                        partition_aggregate_records,
                    ) = _run_partitioned_cleanup_attempts(
                        image=image.copy() if hasattr(image, "copy") else image,
                        source_image=source_image,
                        base_plan_kwargs={
                            "page_id": page_id,
                            "job": job,
                            "inpaint_mode": inpaint_mode,
                            "cleanup_class": planned_runtime_class,
                        },
                        partition_masks=partitioned,
                        base_record=base_record,
                        target_region_ids=target_region_ids,
                        parent_rejection_reason=mask_rejection,
                        page_id=page_id,
                        region_id=target_region_ids[0] if target_region_ids else "",
                        use_gpu=use_gpu,
                        model_id=model_id,
                        artifact_dir=artifact_dir,
                    )
                    result_records.extend(partition_results)
                    proof_records.extend(partition_proofs)
                    status_records.extend(partition_statuses)
                    parent_cleanup_unit_aggregate_records.extend(partition_aggregate_records)
                    _page014_timeout_checkpoint(
                        "cleanup_runtime_job",
                        "end",
                        page_id=page_id,
                        cleanup_job_id=str(job.cleanup_job_id),
                        region_id=target_region_ids[0] if target_region_ids else "",
                        runtime_status="partitioned",
                        failure_reason=mask_rejection,
                        partition_count=len(partitioned),
                        elapsed_ms=round((time.time() - job_started) * 1000.0, 3),
                    )
                    continue
                _page014_timeout_checkpoint(
                    "cleanup_runtime_job",
                    "contract_error",
                    page_id=page_id,
                    cleanup_job_id=str(job.cleanup_job_id),
                    region_id=target_region_ids[0] if target_region_ids else "",
                    cleanup_mask_id=str(runtime_mask.cleanup_mask_id),
                    reason=mask_rejection,
                    elapsed_ms=round((time.time() - job_started) * 1000.0, 3),
                )
                contract_error = (
                    "cleanup_contract_error_component_limit"
                    if _is_partitionable_mask_rejection(mask_rejection)
                    else _contract_error_for_reason(mask_rejection)
                )
                status_records.append(
                    {
                        **base_record,
                        "runtime_status": "contract_error",
                        "cleanup_mask_id": runtime_mask.cleanup_mask_id,
                        "failure_reason": contract_error,
                        "cleanup_outcome_state": contract_error,
                        "contract_error_original_reason": mask_rejection,
                        "partitioned_cleanup_attempted": _is_partitionable_mask_rejection(mask_rejection),
                        "partitioned_cleanup_status": "no_valid_component_subplans",
                        "render_consumption_decision_if_consumed": "block_future_renderer_consumption",
                        "renderer_consumed": False,
                    }
                )
                continue

            plan = _runtime_speech_flat_plan(
                page_id=page_id,
                job=job,
                cleanup_mask=runtime_mask,
                inpaint_mode=inpaint_mode,
                cleanup_class=planned_runtime_class,
            )
            before = image.copy() if hasattr(image, "copy") else image
            _page014_timeout_checkpoint(
                "adaptive_cleanup_attempts",
                "start",
                page_id=page_id,
                cleanup_job_id=str(job.cleanup_job_id),
                region_id=target_region_ids[0] if target_region_ids else "",
                cleanup_plan_id=str(plan.cleanup_plan_id),
                cleanup_mask_id=str(runtime_mask.cleanup_mask_id),
            )
            job_results, job_proofs, attempt_records = _run_adaptive_cleanup_attempts(
                image=before,
                source_image=source_image,
                cleanup_plan=plan,
                cleanup_mask=runtime_mask,
                page_id=page_id,
                region_id=target_region_ids[0] if target_region_ids else "",
                use_gpu=use_gpu,
                model_id=model_id,
                artifact_dir=artifact_dir,
            )
            _page014_timeout_checkpoint(
                "adaptive_cleanup_attempts",
                "end",
                page_id=page_id,
                cleanup_job_id=str(job.cleanup_job_id),
                region_id=target_region_ids[0] if target_region_ids else "",
                attempt_count=len(attempt_records),
                result_count=len(job_results),
                proof_count=len(job_proofs),
                elapsed_ms=round((time.time() - job_started) * 1000.0, 3),
            )
            result_records.extend(job_results)
            proof_records.extend(job_proofs)
            cleanup_result, cleanup_proof = _select_best_commit_candidate_result(job_results, job_proofs)
            if cleanup_result is None:
                cleanup_result, cleanup_proof = _best_failed_attempt(job_results, job_proofs)
            proof_passed = bool(cleanup_proof and cleanup_proof.proof_status == ProofStatus.PASSED)
            degraded_commit_allowed = bool(
                cleanup_result
                and cleanup_proof
                and _proof_allows_degraded_commit(cleanup_proof, cleanup_result, {})
            )
            runtime_status = (
                "passed"
                if proof_passed and cleanup_result and cleanup_result.pixel_changed
                else ("warning" if degraded_commit_allowed else "failed")
            )
            failure_reason = (
                "passed"
                if runtime_status == "passed"
                else (cleanup_proof.failure_reason if cleanup_proof else "")
                or (cleanup_result.failure_reason if cleanup_result else "")
                or "cleanup_runtime_proof_failed"
            )
            selected_attempt = _selected_attempt_record(attempt_records, cleanup_result)
            selected_proof_metrics = _json_safe(cleanup_proof.metrics or {}) if cleanup_proof else {}
            status_records.append(
                {
                    **base_record,
                    "runtime_status": runtime_status,
                    "cleanup_plan_id": cleanup_result.cleanup_plan_id if cleanup_result else plan.cleanup_plan_id,
                    "cleanup_mask_id": runtime_mask.cleanup_mask_id,
                    "cleanup_result_id": cleanup_result.cleanup_result_id if cleanup_result else "",
                    "cleanup_proof_id": cleanup_proof.cleanup_proof_id if cleanup_proof else "",
                    "proof_status": _enum_value(cleanup_proof.proof_status) if cleanup_proof else "",
                    "failure_reason": failure_reason,
                    "cleanup_outcome_state": _cleanup_outcome_state_for_failure(failure_reason, runtime_status=runtime_status),
                    "pixel_changed": bool(cleanup_result and cleanup_result.pixel_changed),
                    "changed_pixel_count": int((cleanup_result.changed_pixel_count if cleanup_result else 0) or 0),
                    "cleaned_image_ref": cleanup_result.cleaned_image_ref if cleanup_result else None,
                    "diff_ref": cleanup_result.diff_ref if cleanup_result else None,
                    "mask_ref": cleanup_result.mask_ref if cleanup_result else None,
                    "proof_metrics": selected_proof_metrics,
                    "proof_quality_state": selected_proof_metrics.get("proof_quality_state", ""),
                    "degraded_commit_allowed": bool(
                        selected_proof_metrics.get("degraded_commit_allowed", False)
                    ),
                    "source_residual_delta_dark_pixels": selected_proof_metrics.get(
                        "source_residual_delta_dark_pixels",
                        0,
                    ),
                    "source_residual_delta_dark_ratio": selected_proof_metrics.get(
                        "source_residual_delta_dark_ratio",
                        0,
                    ),
                    "collateral_art_damage_delta": selected_proof_metrics.get(
                        "collateral_art_damage_delta",
                        0,
                    ),
                    "proof_scope": selected_proof_metrics.get("proof_scope", "cleanup_unit_full"),
                    "proof_scope_bbox": selected_proof_metrics.get("proof_scope_bbox", []),
                    "parent_cleanup_unit_id": selected_proof_metrics.get(
                        "parent_cleanup_unit_id",
                        base_record.get("cleanup_unit_id", ""),
                    ),
                    "parent_expected_erasure_bbox": selected_proof_metrics.get(
                        "parent_expected_erasure_bbox",
                        [],
                    ),
                    "adaptive_attempt_count": len(attempt_records),
                    "adaptive_attempts": _json_safe(attempt_records),
                    "selected_attempt_index": selected_attempt.get("attempt_index") if selected_attempt else None,
                    "selected_attempt_strategy": selected_attempt.get("strategy") if selected_attempt else "",
                    "render_consumption_decision_if_consumed": (
                        cleanup_proof.render_consumption_decision_if_consumed
                        if cleanup_proof
                        else "block_future_renderer_consumption"
                    ),
                    "renderer_consumed": False,
                }
            )
            _page014_timeout_checkpoint(
                "cleanup_runtime_job",
                "end",
                page_id=page_id,
                cleanup_job_id=str(job.cleanup_job_id),
                region_id=target_region_ids[0] if target_region_ids else "",
                runtime_status=runtime_status,
                failure_reason=failure_reason,
                attempt_count=len(attempt_records),
                elapsed_ms=round((time.time() - job_started) * 1000.0, 3),
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            _page014_timeout_checkpoint(
                "cleanup_runtime_job",
                "error",
                page_id=page_id,
                cleanup_job_id=str(job.cleanup_job_id),
                region_id=target_region_ids[0] if target_region_ids else "",
                error=message,
                elapsed_ms=round((time.time() - job_started) * 1000.0, 3),
            )
            errors.append(message)
            status_records.append(
                {
                    **base_record,
                    "runtime_status": "inconclusive",
                    "failure_reason": message,
                    "cleanup_outcome_state": "cleanup_partially_completed_with_warnings",
                    "render_consumption_decision_if_consumed": "block_future_renderer_consumption",
                    "renderer_consumed": False,
                }
            )

    _page014_timeout_checkpoint(
        "cleanup_runtime_contract",
        "end",
        page_id=page_id,
        status_count=len(status_records),
        result_count=len(result_records),
        proof_count=len(proof_records),
        error_count=len(errors),
        elapsed_ms=round((time.time() - runtime_started) * 1000.0, 3),
    )
    return CleanupRuntimeContractResult(
        page_id=page_id,
        version=CLEANUP_RUNTIME_CONTRACT_VERSION,
        runtime_class=runtime_class or "phase5_cleanup_owned_general",
        status_records=status_records,
        result_records=result_records,
        proof_records=proof_records,
        parent_cleanup_unit_aggregate_records=parent_cleanup_unit_aggregate_records,
        errors=errors,
    )


def commit_cleanup_runtime_results_to_working_image(
    *,
    page_id: str,
    source_image: Any,
    runtime_contract: CleanupRuntimeContractResult | Any,
    artifact_dir: str | None = None,
    excluded_region_ids: set[str] | None = None,
) -> CleanupUpstreamCommitResult:
    """Commit proof-passed runtime cleanup results to a pre-render working image."""

    commit_started = time.time()
    errors: list[str] = []
    commit_records: list[dict[str, Any]] = []
    blocked_records: list[dict[str, Any]] = []
    excluded = {str(region_id) for region_id in (excluded_region_ids or set()) if str(region_id)}
    working = source_image.copy() if hasattr(source_image, "copy") else source_image
    if Image is None or np is None:
        return CleanupUpstreamCommitResult(
            page_id=page_id,
            version=CLEANUP_UPSTREAM_COMMIT_VERSION,
            cleaned_image=working,
            blocked_records=[
                {
                    "page_id": page_id,
                    "region_id": "",
                    "cleanup_committed_to_working_image": False,
                    "failure_reason": "pillow_or_numpy_unavailable",
                    "renderer_consumed": False,
                }
            ],
        )

    result_records = list(getattr(runtime_contract, "result_records", []) or [])
    proof_records = list(getattr(runtime_contract, "proof_records", []) or [])
    status_records = list(getattr(runtime_contract, "status_records", []) or [])
    _page014_timeout_checkpoint(
        "cleanup_upstream_commit",
        "start",
        page_id=page_id,
        result_count=len(result_records),
        proof_count=len(proof_records),
        status_count=len(status_records),
    )
    proof_by_result_id = {str(proof.cleanup_result_id): proof for proof in proof_records}
    status_by_result_id = {
        str(status.get("cleanup_result_id") or ""): status
        for status in status_records
        if isinstance(status, Mapping) and str(status.get("cleanup_result_id") or "")
    }

    try:
        source_np = np.asarray(source_image.convert("RGB") if hasattr(source_image, "convert") else source_image)
        working_np = np.asarray(working.convert("RGB") if hasattr(working, "convert") else working).copy()
    except Exception as exc:
        return CleanupUpstreamCommitResult(
            page_id=page_id,
            version=CLEANUP_UPSTREAM_COMMIT_VERSION,
            cleaned_image=working,
            blocked_records=[
                {
                    "page_id": page_id,
                    "region_id": "",
                    "cleanup_committed_to_working_image": False,
                    "failure_reason": f"source_image_array_error:{type(exc).__name__}: {exc}",
                    "renderer_consumed": False,
                }
            ],
        )

    for cleanup_result in result_records:
        result_started = time.time()
        result_id = str(cleanup_result.cleanup_result_id)
        proof = proof_by_result_id.get(result_id)
        status = status_by_result_id.get(result_id, {})
        region_id = str(cleanup_result.region_id or status.get("region_id") or "")
        base = {
            "page_id": page_id,
            "region_id": region_id,
            "cleanup_result_id": result_id,
            "cleanup_proof_id": str(getattr(proof, "cleanup_proof_id", "") or ""),
            "cleanup_plan_id": str(cleanup_result.cleanup_plan_id),
            "cleanup_class": _enum_value(cleanup_result.cleanup_class),
            "pixel_changed": bool(cleanup_result.pixel_changed),
            "changed_pixel_count": int(cleanup_result.changed_pixel_count or 0),
            "renderer_consumed": False,
            "runtime_cleaned_image_ref": cleanup_result.cleaned_image_ref or "",
            "runtime_diff_ref": cleanup_result.diff_ref or "",
            "runtime_mask_ref": cleanup_result.mask_ref or "",
        }
        failure_reason = _upstream_commit_block_reason(cleanup_result, proof, status, excluded)
        if failure_reason:
            _page014_timeout_checkpoint(
                "cleanup_upstream_commit_result",
                "blocked",
                page_id=page_id,
                region_id=region_id,
                cleanup_result_id=result_id,
                reason=failure_reason,
                elapsed_ms=round((time.time() - result_started) * 1000.0, 3),
            )
            blocked_records.append(
                {
                    **base,
                    "cleanup_applied_upstream": False,
                    "cleanup_committed_to_working_image": False,
                    "failure_reason": failure_reason,
                }
            )
            continue

        cleaned_image = cleanup_result.cleaned_image or _load_cleanup_image(cleanup_result.cleaned_image_ref)
        if cleaned_image is None:
            _page014_timeout_checkpoint(
                "cleanup_upstream_commit_result",
                "blocked",
                page_id=page_id,
                region_id=region_id,
                cleanup_result_id=result_id,
                reason="runtime_cleaned_image_missing",
                elapsed_ms=round((time.time() - result_started) * 1000.0, 3),
            )
            blocked_records.append(
                {
                    **base,
                    "cleanup_applied_upstream": False,
                    "cleanup_committed_to_working_image": False,
                    "failure_reason": "runtime_cleaned_image_missing",
                }
            )
            continue
        try:
            committed_pixels, commit_mask = _commit_runtime_result_pixels(
                working_np=working_np,
                source_np=source_np,
                cleaned_image=cleaned_image,
                operation_bbox=cleanup_result.operation_bbox,
            )
        except Exception as exc:
            errors.append(f"{region_id}:{type(exc).__name__}: {exc}")
            committed_pixels = 0
            commit_mask = None
        if committed_pixels <= 0:
            _page014_timeout_checkpoint(
                "cleanup_upstream_commit_result",
                "blocked",
                page_id=page_id,
                region_id=region_id,
                cleanup_result_id=result_id,
                reason="no_pixels_committed_to_working_image",
                elapsed_ms=round((time.time() - result_started) * 1000.0, 3),
            )
            blocked_records.append(
                {
                    **base,
                    "cleanup_applied_upstream": False,
                    "cleanup_committed_to_working_image": False,
                    "failure_reason": "no_pixels_committed_to_working_image",
                }
            )
            continue
        proof_metrics = proof.metrics if proof is not None and isinstance(proof.metrics, Mapping) else {}
        degraded_warning_commit = _proof_allows_degraded_commit(proof, cleanup_result, status)
        commit_records.append(
            {
                **base,
                "cleanup_applied_upstream": True,
                "cleanup_committed_to_working_image": True,
                "committed_pixel_count": committed_pixels,
                "commit_mask_pixels": int(np.count_nonzero(commit_mask)) if commit_mask is not None else 0,
                "failure_reason": "passed",
                "cleanup_commit_quality": (
                    "residual_warning_commit"
                    if degraded_warning_commit
                    else "proof_passed_clean_commit"
                ),
                "residual_warning": bool(degraded_warning_commit),
                "proof_quality_state": proof_metrics.get("proof_quality_state", ""),
                "source_residual_delta_dark_pixels": proof_metrics.get(
                    "source_residual_delta_dark_pixels",
                    0,
                ),
                "source_residual_delta_dark_ratio": proof_metrics.get(
                    "source_residual_delta_dark_ratio",
                    0,
                ),
                "collateral_art_damage_delta": proof_metrics.get("collateral_art_damage_delta", 0),
                "broad_white_patch_risk": bool(proof_metrics.get("broad_white_patch_risk", False)),
            }
        )
        _page014_timeout_checkpoint(
            "cleanup_upstream_commit_result",
            "committed",
            page_id=page_id,
            region_id=region_id,
            cleanup_result_id=result_id,
            committed_pixel_count=committed_pixels,
            elapsed_ms=round((time.time() - result_started) * 1000.0, 3),
        )

    committed_image = Image.fromarray(working_np.astype("uint8"), mode="RGB")
    refs = _write_upstream_commit_artifacts(
        artifact_dir=artifact_dir,
        page_id=page_id,
        before=source_image,
        cleaned=committed_image,
        committed_records=commit_records,
    )
    for record in commit_records:
        record["cleanup_upstream_cleaned_image_ref"] = refs.get("committed_image_ref", "")
        record["cleanup_upstream_diff_ref"] = refs.get("commit_diff_ref", "")
        record["cleanup_upstream_mask_ref"] = refs.get("commit_mask_ref", "")

    _page014_timeout_checkpoint(
        "cleanup_upstream_commit",
        "end",
        page_id=page_id,
        committed_count=len(commit_records),
        blocked_count=len(blocked_records),
        error_count=len(errors),
        elapsed_ms=round((time.time() - commit_started) * 1000.0, 3),
    )
    return CleanupUpstreamCommitResult(
        page_id=page_id,
        version=CLEANUP_UPSTREAM_COMMIT_VERSION,
        cleaned_image=committed_image,
        committed_image_ref=refs.get("committed_image_ref", ""),
        commit_diff_ref=refs.get("commit_diff_ref", ""),
        commit_mask_ref=refs.get("commit_mask_ref", ""),
        commit_records=commit_records,
        blocked_records=blocked_records,
        errors=errors,
    )


def _upstream_commit_block_reason(
    cleanup_result: CleanupResult,
    proof: CleanupProof | None,
    status: Mapping[str, Any],
    excluded_region_ids: set[str],
) -> str:
    region_id = str(cleanup_result.region_id or status.get("region_id") or "")
    target_region_ids = [str(rid) for rid in status.get("target_region_ids", []) if str(rid)]
    if region_id in excluded_region_ids or any(rid in excluded_region_ids for rid in target_region_ids):
        return "protected_region_excluded_from_upstream_commit"
    if bool(cleanup_result.renderer_consumed):
        return "runtime_result_already_renderer_consumed"
    if proof is None:
        return "missing_cleanup_proof"
    if proof.proof_status != ProofStatus.PASSED:
        return proof.failure_reason or "cleanup_proof_not_passed"
    if str(status.get("runtime_status") or "") not in {"", "passed", "warning", "failed"}:
        return str(status.get("failure_reason") or "cleanup_runtime_status_not_passed")
    if not cleanup_result.pixel_changed or int(cleanup_result.changed_pixel_count or 0) <= 0:
        return "cleanup_result_not_pixel_changing"
    if proof.source_glyph_removal_passed is False:
        return "source_glyph_removal_proof_failed"
    if proof.mask_containment_passed is False:
        return "mask_containment_proof_failed"
    changed_outside = _int_or_none(proof.changed_outside_allowed_pixels)
    if changed_outside is None and isinstance(proof.metrics, Mapping):
        changed_outside = _int_or_none(proof.metrics.get("changed_outside_allowed_pixels"))
    if changed_outside is not None and changed_outside > PROOF_CHANGED_OUTSIDE_ALLOWED_PIXELS:
        return "changed_outside_allowed_area"
    changed_outside_ratio = None
    if isinstance(proof.metrics, Mapping):
        changed_outside_ratio = _float_or_none(proof.metrics.get("changed_outside_allowed_ratio"))
    if changed_outside_ratio is not None and changed_outside_ratio > PROOF_CHANGED_OUTSIDE_ALLOWED_RATIO:
        return "changed_outside_allowed_ratio"
    decision = str(proof.render_consumption_decision_if_consumed or "")
    if decision and decision not in {
        "allow_stage6_consumption_candidate",
    }:
        return decision
    return ""


def _load_cleanup_image(path: str | None) -> Any | None:
    if not path or Image is None:
        return None
    try:
        if not os.path.isfile(path):
            return None
        with Image.open(path) as img:
            return img.convert("RGB").copy()
    except Exception:
        return None


def _commit_runtime_result_pixels(
    *,
    working_np: Any,
    source_np: Any,
    cleaned_image: Any,
    operation_bbox: Sequence[int] | None,
) -> tuple[int, Any | None]:
    if np is None:
        return 0, None
    cleaned_np = np.asarray(cleaned_image.convert("RGB") if hasattr(cleaned_image, "convert") else cleaned_image)
    if cleaned_np.shape != source_np.shape or working_np.shape != source_np.shape:
        return 0, None
    box = _valid_bbox(operation_bbox)
    if box is None:
        return 0, None
    height, width = source_np.shape[:2]
    x0, y0, x1, y1 = box
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        return 0, None
    before_crop = source_np[y0:y1, x0:x1]
    cleaned_crop = cleaned_np[y0:y1, x0:x1]
    diff = np.abs(cleaned_crop.astype(np.int16) - before_crop.astype(np.int16))
    changed = np.any(diff > 8, axis=2) if diff.ndim == 3 else diff > 8
    changed_pixels = int(np.count_nonzero(changed))
    if changed_pixels <= 0:
        return 0, None
    target_crop = working_np[y0:y1, x0:x1]
    target_crop[changed] = cleaned_crop[changed]
    working_np[y0:y1, x0:x1] = target_crop
    mask = np.zeros(source_np.shape[:2], dtype=np.uint8)
    mask[y0:y1, x0:x1] = changed.astype(np.uint8)
    return changed_pixels, mask


def _write_upstream_commit_artifacts(
    *,
    artifact_dir: str | None,
    page_id: str,
    before: Any,
    cleaned: Any,
    committed_records: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    if not artifact_dir or Image is None or not committed_records:
        return {}
    try:
        os.makedirs(artifact_dir, exist_ok=True)
        stem = f"{_safe_id(page_id)}_phase5_upstream_commit"
        committed_path = os.path.join(artifact_dir, f"{stem}_working_image.png")
        diff_path = os.path.join(artifact_dir, f"{stem}_diff.png")
        mask_path = os.path.join(artifact_dir, f"{stem}_mask.png")
        _save_image(cleaned, committed_path)
        _save_diff(before, cleaned, diff_path)
        _save_commit_diff_mask(before, cleaned, mask_path)
        return {
            "committed_image_ref": committed_path,
            "commit_diff_ref": diff_path,
            "commit_mask_ref": mask_path,
        }
    except Exception:
        return {}


def _save_commit_diff_mask(before: Any, after: Any, path: str) -> None:
    if np is None or Image is None:
        return
    try:
        before_np = np.asarray(before.convert("RGB") if hasattr(before, "convert") else before)
        after_np = np.asarray(after.convert("RGB") if hasattr(after, "convert") else after)
        if before_np.shape != after_np.shape:
            return
        diff = np.abs(after_np.astype(np.int16) - before_np.astype(np.int16))
        changed = np.any(diff > 8, axis=2) if diff.ndim == 3 else diff > 8
        Image.fromarray(changed.astype(np.uint8) * 255).save(path)
    except Exception:
        return


def execute_cleanup_runtime_plan(
    *,
    image: Any,
    cleanup_plan: CleanupPlan,
    cleanup_mask: CleanupMask,
    page_id: str,
    region_id: str,
    use_gpu: bool,
    model_id: str,
    artifact_dir: str | None = None,
) -> CleanupResult:
    """Execute a Phase 5 runtime plan on a scratch image copy."""

    started = time.time()
    params = cleanup_plan.backend_parameters or {}
    debug_info: dict[str, Any] = {}
    mode = str(params.get("inpaint_mode") or cleanup_plan.inpaint_mode or "fast")
    cleanup_tag = str(params.get("cleanup_tag") or "speech_strong")
    _page014_timeout_checkpoint(
        "cleanup_execute_runtime_plan",
        "start",
        page_id=page_id,
        region_id=region_id,
        cleanup_plan_id=str(cleanup_plan.cleanup_plan_id),
        cleanup_mask_id=str(cleanup_mask.cleanup_mask_id),
        cleanup_tag=cleanup_tag,
        inpaint_mode=mode,
    )
    execution = cleanup_execution.apply_local_text_removal(
        image,
        cleanup_mask.erase_mask,
        mode,
        use_gpu,
        model_id=model_id,
        cleanup_tag=cleanup_tag,
        debug_info=debug_info,
    )
    cleaned_image = execution.cleaned_image or image
    runtime_ms = (time.time() - started) * 1000.0
    changed_pixels = _changed_pixel_count(image, cleaned_image)
    pixel_changed = changed_pixels > 0
    execution_status = "completed" if pixel_changed else "completed_no_pixel_change"
    failure_reason = "" if pixel_changed else "no_pixels_changed"
    refs = _write_runtime_artifacts(
        artifact_dir=artifact_dir,
        page_id=page_id,
        region_id=region_id,
        cleanup_plan=cleanup_plan,
        cleanup_mask=cleanup_mask,
        before=image,
        cleaned=cleaned_image,
    )
    backend = str(execution.backend or debug_info.get("backend") or "unknown")
    _page014_timeout_checkpoint(
        "cleanup_execute_runtime_plan",
        "end",
        page_id=page_id,
        region_id=region_id,
        cleanup_plan_id=str(cleanup_plan.cleanup_plan_id),
        cleanup_mask_id=str(cleanup_mask.cleanup_mask_id),
        backend=backend,
        pixel_changed=pixel_changed,
        changed_pixel_count=changed_pixels,
        artifact_ref_count=len([value for value in refs.values() if value]),
        elapsed_ms=round(runtime_ms, 3),
    )
    return CleanupResult(
        cleanup_result_id=f"cres_{_safe_id(cleanup_plan.cleanup_plan_id)}",
        cleanup_plan_id=str(cleanup_plan.cleanup_plan_id),
        cleanup_job_id=str(cleanup_plan.cleanup_job_id),
        cleanup_mask_id=str(cleanup_plan.cleanup_mask_id),
        operation_bbox=_valid_bbox(execution.crop_bbox) or _valid_bbox(cleanup_mask.erase_mask_bbox),
        page=page_id,
        region_id=region_id,
        cleanup_class=cleanup_plan.cleanup_class,
        pixel_changed=pixel_changed,
        changed_pixel_count=changed_pixels,
        cleaned_image_ref=refs.get("cleaned_image_ref"),
        cleaned_crop_ref=refs.get("cleaned_crop_ref"),
        diff_ref=refs.get("diff_ref"),
        mask_ref=refs.get("mask_ref"),
        renderer_consumed=False,
        execution_backend=backend,
        execution_status=execution_status,
        failure_reason=failure_reason,
        mask_stats=mask_stats(cleanup_mask.erase_mask) or {},
        backend_name=backend,
        backend_parameters={
            "backend_detail": execution.backend_detail,
            "candidate_status": "completed" if pixel_changed else "no_pixel_change",
            "cleanup_tag": cleanup_tag,
            "effective_inpaint_mode": execution.effective_inpaint_mode,
            "crop_bbox": execution.crop_bbox,
            "crop_area": execution.crop_area,
            "mask_ratio": execution.mask_ratio,
        },
        runtime_ms=runtime_ms,
        fallback_status=execution_status,
        errors=list(execution.errors or []),
        cleaned_image=cleaned_image,
        cleaned_crop=execution.cleaned_image,
    )


def build_inconclusive_pilot_runtime_result(
    *,
    page_id: str,
    image: Any,
    plan_contracts: CleanupPlanBuildResult | Any,
    detail: str,
) -> CaptionFlatPilotRuntimeResult:
    """Create visible-failure statuses if renderer pilot execution aborts."""

    status_records: list[dict[str, Any]] = []
    grouped: dict[str, CleanupPlan] = {}
    for cleanup_plan in _extract_plans(plan_contracts):
        grouped.setdefault(str(cleanup_plan.cleanup_job_id), cleanup_plan)
    for cleanup_plan in grouped.values():
        target_region_ids = [
            str(rid)
            for rid in (cleanup_plan.backend_parameters or {}).get("target_region_ids", [])
            if str(rid)
        ]
        status_records.append(
            _status_record(
                cleanup_plan,
                target_region_ids=target_region_ids,
                status="inconclusive",
                failure_class="inconclusive_missing_evidence",
                renderer_action="discard_pilot_cleanup_suppress_translation",
                detail=detail,
            )
        )
    return CaptionFlatPilotRuntimeResult(
        page_id=page_id,
        version=PILOT_RUNTIME_VERSION,
        cleaned_image=image,
        status_records=status_records,
        result_records=[],
        proof_records=[],
        errors=[detail] if detail else [],
    )


def _proof(
    cleanup_result: CleanupResult,
    cleanup_plan: CleanupPlan,
    *,
    status: ProofStatus,
    failure_reason: str,
    metrics: dict[str, Any],
) -> CleanupProof:
    pass_residual_ratio = _float_or_none(metrics.get("residual_dark_source_ratio"))
    if pass_residual_ratio is None:
        pass_residual_ratio = _float_or_none(metrics.get("residual_source_ratio"))
    degraded_commit_allowed = bool(metrics.get("degraded_commit_allowed"))
    proof_quality_state = str(metrics.get("proof_quality_state") or "")
    render_decision = (
        "allow_stage6_consumption_candidate"
        if status == ProofStatus.PASSED
        else "block_future_renderer_consumption"
    )
    return CleanupProof(
        cleanup_proof_id=f"cproof_{_safe_id(cleanup_result.cleanup_result_id)}",
        cleanup_result_id=cleanup_result.cleanup_result_id,
        cleanup_job_id=cleanup_result.cleanup_job_id,
        cleanup_plan_id=cleanup_plan.cleanup_plan_id,
        proof_status=status,
        source_glyph_removal_passed=status == ProofStatus.PASSED,
        source_residual_ratio=pass_residual_ratio,
        changed_outside_allowed_pixels=_int_or_none(metrics.get("changed_outside_allowed_pixels")),
        collateral_damage_score=_float_or_none(metrics.get("changed_outside_erase_inside_allowed_ratio")),
        mask_containment_ratio=_float_or_none(metrics.get("mask_source_overlap_ratio")),
        render_consumption_decision_if_consumed=render_decision,
        residual_source_text_risk=(
            "low"
            if status == ProofStatus.PASSED
            else ("warning_residual" if degraded_commit_allowed else "visible_or_unknown")
        ),
        mask_containment_passed=(
            metrics.get("changed_outside_allowed_pixels", 0) <= PROOF_CHANGED_OUTSIDE_ALLOWED_PIXELS
            if metrics
            else None
        ),
        collateral_damage_risk=(
            "low"
            if status == ProofStatus.PASSED or degraded_commit_allowed
            else "possible"
        ),
        backend_failure_visible=failure_reason == "backend_noop_or_error",
        failure_reason=failure_reason,
        metrics={**metrics, "proof_quality_state": proof_quality_state or metrics.get("proof_quality_state", "")},
    )


def _select_best_passed_result(
    result_records: Sequence[CleanupResult],
    proof_records: Sequence[CleanupProof],
) -> tuple[CleanupResult | None, CleanupProof | None]:
    by_result_id = {record.cleanup_result_id: record for record in result_records}
    passed = [
        proof
        for proof in proof_records
        if proof.proof_status == ProofStatus.PASSED and proof.cleanup_result_id in by_result_id
    ]
    if not passed:
        return None, None
    passed.sort(key=lambda proof: _proof_selection_key(proof, by_result_id[proof.cleanup_result_id]))
    proof = passed[0]
    return by_result_id[proof.cleanup_result_id], proof


def _select_best_commit_candidate_result(
    result_records: Sequence[CleanupResult],
    proof_records: Sequence[CleanupProof],
) -> tuple[CleanupResult | None, CleanupProof | None]:
    by_result_id = {record.cleanup_result_id: record for record in result_records}
    candidates = [
        proof
        for proof in proof_records
        if proof.cleanup_result_id in by_result_id
        and _proof_commit_candidate(by_result_id[proof.cleanup_result_id], proof)
    ]
    if not candidates:
        return None, None
    candidates.sort(
        key=lambda proof: (
            0 if proof.proof_status == ProofStatus.PASSED else 1,
            *_proof_selection_key(proof, by_result_id[proof.cleanup_result_id]),
        )
    )
    proof = candidates[0]
    return by_result_id[proof.cleanup_result_id], proof


def _best_failed_attempt(
    result_records: Sequence[CleanupResult],
    proof_records: Sequence[CleanupProof],
) -> tuple[CleanupResult | None, CleanupProof | None]:
    by_result_id = {record.cleanup_result_id: record for record in result_records}
    available = [proof for proof in proof_records if proof.cleanup_result_id in by_result_id]
    if not available:
        return (result_records[0], None) if result_records else (None, None)
    available.sort(key=lambda proof: _proof_selection_key(proof, by_result_id[proof.cleanup_result_id]))
    proof = available[0]
    return by_result_id[proof.cleanup_result_id], proof


def _proof_selection_key(proof: CleanupProof, result: CleanupResult) -> tuple[float, float, float]:
    metrics = proof.metrics or {}
    residual = float(
        metrics.get("residual_dark_source_ratio")
        or metrics.get("residual_source_ratio")
        or 999.0
    )
    collateral = float(metrics.get("changed_outside_erase_inside_allowed_ratio") or 999.0)
    runtime = float(result.runtime_ms if result.runtime_ms is not None else 999_999.0)
    return residual, collateral, runtime


def _plan_for_result(
    plans: Sequence[CleanupPlan],
    cleanup_result: CleanupResult | None,
) -> CleanupPlan | None:
    if cleanup_result is None:
        return None
    for plan in plans:
        if plan.cleanup_plan_id == cleanup_result.cleanup_plan_id:
            return plan
    return None


def _status_record(
    cleanup_plan: CleanupPlan,
    *,
    target_region_ids: Sequence[str],
    status: str,
    failure_class: str,
    renderer_action: str,
    cleanup_result_id: str | None = None,
    cleanup_proof_id: str | None = None,
    allowed_area: Sequence[int] | None = None,
    erase_mask_bbox: Sequence[int] | None = None,
    proof_metrics: Mapping[str, Any] | None = None,
    detail: str = "",
    candidate_result_count: int = 0,
    selected_backend_candidate_id: Any | None = None,
) -> dict[str, Any]:
    return {
        "cleanup_plan_id": cleanup_plan.cleanup_plan_id,
        "cleanup_job_id": cleanup_plan.cleanup_job_id,
        "cleanup_mask_id": cleanup_plan.cleanup_mask_id,
        "cleanup_result_id": cleanup_result_id,
        "cleanup_proof_id": cleanup_proof_id,
        "target_region_ids": list(target_region_ids),
        "cleanup_class": _enum_value(cleanup_plan.cleanup_class),
        "pilot_status": status,
        "failure_class": failure_class,
        "renderer_action": renderer_action,
        "render_suppressed_by_cleanup_proof": status in {"failed", "inconclusive"},
        "pilot_cleanup_committed": status == "passed",
        "allowed_area": list(allowed_area or []),
        "erase_mask_bbox": list(erase_mask_bbox or []),
        "proof_metrics": _json_safe(proof_metrics or {}),
        "detail": detail,
        "candidate_result_count": int(candidate_result_count),
        "selected_backend_candidate_id": selected_backend_candidate_id,
    }


def _runtime_base_record(
    page_id: str,
    job: CleanupJob,
    *,
    runtime_class: CleanupClass,
) -> dict[str, Any]:
    target_region_ids = [str(rid) for rid in job.target_region_ids or [] if str(rid)]
    return {
        "page_id": page_id,
        "region_id": target_region_ids[0] if target_region_ids else "",
        "target_region_ids": target_region_ids,
        "cleanup_job_id": str(job.cleanup_job_id),
        "cleanup_unit_id": str(getattr(job, "cleanup_unit_id", "") or job.cleanup_job_id),
        "cleanup_unit_level": str(getattr(job, "cleanup_unit_level", "") or ""),
        "cleanup_unit_anchor_region_id": str(
            getattr(job, "cleanup_unit_anchor_region_id", "") or ""
        ),
        "cleanup_unit_child_region_ids": list(
            getattr(job, "cleanup_unit_child_region_ids", []) or []
        ),
        "cleanup_unit_required_source_glyph_mask_ids": list(
            getattr(job, "cleanup_unit_required_source_glyph_mask_ids", []) or []
        ),
        "cleanup_unit_missing_source_glyph_mask_ids": list(
            getattr(job, "cleanup_unit_missing_source_glyph_mask_ids", []) or []
        ),
        "cleanup_class": _enum_value(job.cleanup_class),
        "runtime_class": _enum_value(runtime_class),
        "renderer_consumed": False,
    }


def _runtime_cleanup_class_for_job(job: CleanupJob, requested_class: CleanupClass | None) -> CleanupClass | None:
    job_class_value = _enum_value(job.cleanup_class)
    if requested_class is not None and job_class_value != _enum_value(requested_class):
        return None
    if job_class_value in RUNTIME_UNSAFE_CLEANUP_CLASSES:
        return None
    if job_class_value not in RUNTIME_SUPPORTED_CLEANUP_CLASSES:
        return None
    return job.cleanup_class


def _runtime_job_exclusion_reason(job: CleanupJob, cleanup_mask: CleanupMask | None = None) -> str:
    if bool(job.protected):
        return job.protection_reason or "job_protected"
    source_evidence_reason = _runtime_source_evidence_exclusion_reason(job, cleanup_mask)
    if source_evidence_reason:
        return source_evidence_reason
    combined = " ".join(
        str(value or "")
        for value in (
            _enum_value(job.cleanup_class),
            job.route_intent,
            job.semantic_class,
            job.cleanup_mode,
            job.container_type,
            " ".join(job.source_glyph_mask_ids or []),
        )
    ).lower()
    for blocked in (
        "preserve",
        "sfx",
        "decorative",
        "non_text",
        "non_translation_art",
        "art_only",
        "source_grounding_protected",
        "source_ungrounded",
    ):
        if blocked in combined:
            return f"phase5_runtime_excludes_{blocked}"
    if not job.source_text_present:
        return "source_text_missing"
    if not job.translated_text_present:
        return "translated_text_missing"
    return ""


def _runtime_source_evidence_exclusion_reason(
    job: CleanupJob,
    cleanup_mask: CleanupMask | None = None,
) -> str:
    if cleanup_mask is not None:
        consumed_ids = _unique_texts(getattr(cleanup_mask, "consumed_source_glyph_mask_ids", []) or [])
        missing_ids = _unique_texts(getattr(cleanup_mask, "missing_source_glyph_mask_ids", []) or [])
        foreground_pixels = int(getattr(cleanup_mask, "foreground_mask_pixels", 0) or 0)
        has_foreground_array = False
        if np is not None and getattr(cleanup_mask, "foreground_mask", None) is not None:
            try:
                has_foreground_array = bool(np.any(np.asarray(cleanup_mask.foreground_mask) > 0))
            except Exception:
                has_foreground_array = False
        has_source_ids = bool(consumed_ids or getattr(cleanup_mask, "foreground_mask_source_id", None))
        if missing_ids:
            return "phase5_runtime_requires_consumable_sourceglyph_evidence"
        if has_source_ids and (foreground_pixels > 0 or has_foreground_array):
            return ""
    if not job.source_glyph_evidence or not job.source_glyph_mask_ids:
        return "phase5_runtime_requires_sourceglyph_evidence"
    eligible_records = 0
    for item in job.source_glyph_evidence or []:
        if not isinstance(item, Mapping):
            continue
        if _truthy(_first_source_value(item, "source_glyph_mask_review_only")):
            continue
        if _truthy(_first_source_value(item, "preserve", "is_sfx", "is_decorative", "is_art")):
            continue
        if _source_value_false(item, "cleanup_covers_source_glyphs"):
            continue
        coverage = _float_or_none(_first_source_value(item, "source_glyph_erasure_coverage_ratio"))
        if coverage is not None and coverage < PROOF_MASK_SOURCE_OVERLAP_RATIO:
            continue
        if not _valid_bbox(
            _first_source_value(
                item,
                "source_glyph_erasure_bbox",
                "source_glyph_bbox",
                "bbox",
            )
        ):
            continue
        eligible_records += 1
    if eligible_records <= 0:
        return "phase5_runtime_requires_consumable_sourceglyph_evidence"
    return ""


def _job_source_contract_values(job: CleanupJob, key: str) -> set[str]:
    values: set[str] = set()
    for item in job.source_glyph_evidence or []:
        if not isinstance(item, Mapping):
            continue
        value = item.get(key)
        if value is None and isinstance(item.get("compatibility_source_fields"), Mapping):
            value = item.get("compatibility_source_fields", {}).get(key)
        text = str(value or "").strip()
        if text:
            values.add(text)
    return values


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "none", "null"}
    return bool(value)


def _effective_mask_kwargs(cleanup_mask: CleanupMask) -> dict[str, Any]:
    return {
        "effective_mask_status": str(getattr(cleanup_mask, "effective_mask_status", "") or ""),
        "effective_mask_failure_reason": str(getattr(cleanup_mask, "effective_mask_failure_reason", "") or ""),
        "seed_foreground_pixels": getattr(cleanup_mask, "seed_foreground_pixels", None),
        "completed_foreground_pixels": getattr(cleanup_mask, "completed_foreground_pixels", None),
        "component_count_before": getattr(cleanup_mask, "component_count_before", None),
        "component_count_after": getattr(cleanup_mask, "component_count_after", None),
        "largest_component_pixels_before": getattr(cleanup_mask, "largest_component_pixels_before", None),
        "largest_component_pixels_after": getattr(cleanup_mask, "largest_component_pixels_after", None),
        "text_block_coverage_estimate": getattr(cleanup_mask, "text_block_coverage_estimate", None),
        "bbox_fill_ratio_before": getattr(cleanup_mask, "bbox_fill_ratio_before", None),
        "bbox_fill_ratio_after": getattr(cleanup_mask, "bbox_fill_ratio_after", None),
        "analysis_scope_bbox": list(getattr(cleanup_mask, "analysis_scope_bbox", None) or []),
        "executable_erase_bbox": list(getattr(cleanup_mask, "executable_erase_bbox", None) or []),
        "mask_completion_method": str(getattr(cleanup_mask, "mask_completion_method", "") or ""),
        "polarity_mode": str(getattr(cleanup_mask, "polarity_mode", "") or ""),
        "source_seed_mask_ids": list(getattr(cleanup_mask, "source_seed_mask_ids", []) or []),
        "recovered_component_count": getattr(cleanup_mask, "recovered_component_count", None),
        "rejected_component_count": getattr(cleanup_mask, "rejected_component_count", None),
        "rejected_component_reasons": list(getattr(cleanup_mask, "rejected_component_reasons", []) or []),
    }


def _first_source_value(item: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in item and item.get(key) not in (None, ""):
            return item.get(key)
        compat = item.get("compatibility_source_fields")
        if isinstance(compat, Mapping) and key in compat and compat.get(key) not in (None, ""):
            return compat.get(key)
    return None


def _source_value_false(item: Mapping[str, Any], key: str) -> bool:
    value = _first_source_value(item, key)
    if isinstance(value, str):
        return value.strip().lower() in {"0", "false", "no"}
    return value is False


def _runtime_speech_flat_mask(cleanup_mask: CleanupMask, source_image: Any) -> CleanupMask:
    """Compatibility wrapper for older callers."""

    return _runtime_glyph_precision_mask(cleanup_mask, source_image, None)


def _runtime_glyph_precision_mask(
    cleanup_mask: CleanupMask,
    source_image: Any,
    cleanup_class: CleanupClass | None,
) -> CleanupMask:
    """Build runtime geometry from the completed CleanupMask foreground."""

    if np is None:
        return cleanup_mask
    effective_status = str(getattr(cleanup_mask, "effective_mask_status", "") or "")
    if effective_status.startswith("effective_mask_failed_"):
        return _runtime_rejected_mask(
            cleanup_mask,
            reason=effective_status,
            method_suffix=effective_status,
            foreground=cleanup_mask.foreground_mask,
            erase=cleanup_mask.erase_mask,
        )
    allowed = _valid_bbox(cleanup_mask.allowed_area)
    foreground = _binary_cleanup_mask_array(cleanup_mask.foreground_mask)
    if foreground is None:
        foreground = _binary_cleanup_mask_array_from_shape(cleanup_mask.foreground_mask_bbox, source_image)
        if foreground is not None:
            foreground[:] = 0
    if allowed is None or foreground is None:
        return _runtime_rejected_mask(
            cleanup_mask,
            reason="sourceglyph_generation_or_foreground_contract_defect",
            method_suffix="missing_foreground_or_allowed_area",
        )
    foreground = _clip_mask_to_valid_bbox(foreground, allowed)
    defect = _glyph_foreground_geometry_defect_reason(
        foreground=foreground,
        allowed=allowed,
        cleanup_class=cleanup_class,
    )
    if defect:
        return _runtime_rejected_mask(
            cleanup_mask,
            reason="sourceglyph_generation_or_foreground_contract_defect",
            method_suffix=defect,
            foreground=foreground,
        )
    erase = _binary_cleanup_mask_array(cleanup_mask.erase_mask)
    if erase is None or not np.any(erase > 0):
        erase = _glyph_local_erase_from_foreground(foreground, allowed, cleanup_class)
    else:
        erase = _clip_mask_to_valid_bbox(erase, allowed)
    foreground_pixels = int(np.count_nonzero(foreground))
    erase_pixels = int(np.count_nonzero(erase))
    foreground_bbox = _mask_bbox(foreground)
    erase_bbox = _mask_bbox(erase)
    if foreground_pixels <= 0 or erase_pixels <= 0 or foreground_bbox is None or erase_bbox is None:
        return _runtime_rejected_mask(
            cleanup_mask,
            reason="sourceglyph_generation_or_foreground_contract_defect",
            method_suffix="empty_glyph_component_after_clip",
            foreground=foreground,
            erase=erase,
        )
    runtime = CleanupMask(
        cleanup_mask_id=f"{cleanup_mask.cleanup_mask_id}_phase5_runtime",
        cleanup_job_id=str(cleanup_mask.cleanup_job_id),
        foreground_mask_source_id=cleanup_mask.foreground_mask_source_id,
        foreground_mask_source_ids=list(cleanup_mask.foreground_mask_source_ids or []),
        consumed_source_glyph_mask_ids=list(cleanup_mask.consumed_source_glyph_mask_ids or []),
        missing_source_glyph_mask_ids=list(cleanup_mask.missing_source_glyph_mask_ids or []),
        foreground_mask_bbox=foreground_bbox,
        foreground_mask_pixels=foreground_pixels,
        erase_mask_bbox=erase_bbox,
        erase_mask_pixels=erase_pixels,
        allowed_area=allowed,
        growth_ratio=_growth_ratio(erase_pixels, foreground_pixels),
        mask_source="source_glyph_foreground_runtime_precision",
        mask_method="phase5_phase6_runtime_sourceglyph_foreground_glyph_precision",
        rejection_reason="",
        mask_contract_exception_reason=cleanup_mask.mask_contract_exception_reason,
        artifact_risk=cleanup_mask.artifact_risk,
        visual_scope=_normalise_cleanup_visual_scope(cleanup_mask.visual_scope),
        protected=cleanup_mask.protected,
        protection_reason=cleanup_mask.protection_reason,
        **_effective_mask_kwargs(cleanup_mask),
        foreground_mask=foreground,
        erase_mask=erase,
    )
    if _mask_exclusion_reason(runtime, None) in {
        "growth_ratio_too_high",
        "erase_mask_too_large",
        "erase_bbox_too_broad",
        "erase_allowed_ratio_too_high",
    }:
        foreground_only = CleanupMask(
            cleanup_mask_id=f"{cleanup_mask.cleanup_mask_id}_phase5_runtime",
            cleanup_job_id=str(cleanup_mask.cleanup_job_id),
            foreground_mask_source_id=cleanup_mask.foreground_mask_source_id,
            foreground_mask_source_ids=list(cleanup_mask.foreground_mask_source_ids or []),
            consumed_source_glyph_mask_ids=list(cleanup_mask.consumed_source_glyph_mask_ids or []),
            missing_source_glyph_mask_ids=list(cleanup_mask.missing_source_glyph_mask_ids or []),
            foreground_mask_bbox=foreground_bbox,
            foreground_mask_pixels=foreground_pixels,
            erase_mask_bbox=foreground_bbox,
            erase_mask_pixels=foreground_pixels,
            allowed_area=allowed,
            growth_ratio=1.0,
            mask_source="source_glyph_foreground_runtime_precision",
            mask_method="phase5_phase6_runtime_sourceglyph_foreground_minimal_erasure",
            rejection_reason="",
            mask_contract_exception_reason=cleanup_mask.mask_contract_exception_reason,
            artifact_risk=cleanup_mask.artifact_risk,
            visual_scope=_normalise_cleanup_visual_scope(cleanup_mask.visual_scope),
            protected=cleanup_mask.protected,
            protection_reason=cleanup_mask.protection_reason,
            **_effective_mask_kwargs(cleanup_mask),
            foreground_mask=foreground,
            erase_mask=foreground.copy(),
        )
        if not _mask_exclusion_reason(foreground_only, None):
            return foreground_only
    return runtime


def _binary_cleanup_mask_array(value: Any) -> Any | None:
    if np is None or value is None:
        return None
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    if arr.ndim == 3:
        arr = np.any(arr > 0, axis=2)
    elif arr.ndim != 2:
        return None
    if arr.size <= 0:
        return None
    return (arr > 0).astype(np.uint8)


def _binary_cleanup_mask_array_from_shape(bbox: Sequence[int] | None, source_image: Any) -> Any | None:
    if np is None:
        return None
    try:
        source_np = np.asarray(source_image.convert("RGB") if hasattr(source_image, "convert") else source_image)
        if source_np.ndim < 2:
            return None
        return np.zeros(source_np.shape[:2], dtype=np.uint8)
    except Exception:
        box = _valid_bbox(bbox)
        if box is None:
            return None
        return np.zeros((max(1, box[3]), max(1, box[2])), dtype=np.uint8)


def _runtime_rejected_mask(
    cleanup_mask: CleanupMask,
    *,
    reason: str,
    method_suffix: str,
    foreground: Any | None = None,
    erase: Any | None = None,
) -> CleanupMask:
    foreground_arr = foreground if foreground is not None else cleanup_mask.foreground_mask
    erase_arr = erase if erase is not None else cleanup_mask.erase_mask
    return CleanupMask(
        cleanup_mask_id=f"{cleanup_mask.cleanup_mask_id}_phase5_runtime",
        cleanup_job_id=str(cleanup_mask.cleanup_job_id),
        foreground_mask_source_id=cleanup_mask.foreground_mask_source_id,
        foreground_mask_source_ids=list(cleanup_mask.foreground_mask_source_ids or []),
        consumed_source_glyph_mask_ids=list(cleanup_mask.consumed_source_glyph_mask_ids or []),
        missing_source_glyph_mask_ids=list(cleanup_mask.missing_source_glyph_mask_ids or []),
        foreground_mask_bbox=_mask_bbox(foreground_arr) or cleanup_mask.foreground_mask_bbox,
        foreground_mask_pixels=int(np.count_nonzero(foreground_arr)) if np is not None and foreground_arr is not None else cleanup_mask.foreground_mask_pixels,
        erase_mask_bbox=_mask_bbox(erase_arr) or cleanup_mask.erase_mask_bbox,
        erase_mask_pixels=int(np.count_nonzero(erase_arr)) if np is not None and erase_arr is not None else cleanup_mask.erase_mask_pixels,
        allowed_area=_valid_bbox(cleanup_mask.allowed_area),
        growth_ratio=cleanup_mask.growth_ratio,
        mask_source="source_glyph_foreground_runtime_precision",
        mask_method=f"phase5_phase6_runtime_sourceglyph_foreground_rejected:{method_suffix}",
        rejection_reason=reason,
        mask_contract_exception_reason=cleanup_mask.mask_contract_exception_reason,
        artifact_risk=cleanup_mask.artifact_risk,
        visual_scope=_normalise_cleanup_visual_scope(cleanup_mask.visual_scope),
        protected=cleanup_mask.protected,
        protection_reason=cleanup_mask.protection_reason,
        **_effective_mask_kwargs(cleanup_mask),
        foreground_mask=foreground_arr,
        erase_mask=erase_arr,
    )


def _glyph_foreground_geometry_defect_reason(
    *,
    foreground: Any,
    allowed: Sequence[int],
    cleanup_class: CleanupClass | None,
) -> str:
    pixels = int(np.count_nonzero(foreground > 0))
    if pixels <= 0:
        return "empty_sourceglyph_foreground"
    bbox = _mask_bbox(foreground)
    if bbox is None:
        return "empty_sourceglyph_foreground_bbox"
    bbox_area = _bbox_area(bbox)
    allowed_area = _bbox_area(allowed)
    density = float(pixels) / max(1, bbox_area)
    allowed_ratio = float(pixels) / max(1, allowed_area)
    cleanup_value = _enum_value(cleanup_class)
    speech_like = cleanup_value in {
        CleanupClass.SPEECH_FLAT_BUBBLE.value,
        CleanupClass.SPEECH_COMPLEX_BUBBLE.value,
        CleanupClass.SMALL_REACTION.value,
    }
    if pixels < 4:
        return "sourceglyph_foreground_too_small"
    if density > 0.72 and bbox_area > 1600:
        return "sourceglyph_foreground_rectangular_fill"
    if bbox_area > (80_000 if speech_like else 45_000) and density > (0.22 if speech_like else 0.14):
        return "sourceglyph_foreground_broad_background_chunk"
    if allowed_area > 0 and allowed_ratio > (0.55 if speech_like else 0.35) and pixels > 2400:
        return "sourceglyph_foreground_mostly_allowed_background"
    return ""


def _glyph_local_erase_from_foreground(
    foreground: Any,
    allowed: Sequence[int],
    cleanup_class: CleanupClass | None,
) -> Any:
    if cv2 is None:
        return _clip_mask_to_valid_bbox(foreground, allowed)
    cleanup_value = _enum_value(cleanup_class)
    speech_like = cleanup_value in {
        CleanupClass.SPEECH_FLAT_BUBBLE.value,
        CleanupClass.SPEECH_COMPLEX_BUBBLE.value,
        CleanupClass.SMALL_REACTION.value,
    }
    kernel = np.ones((3, 3), dtype=np.uint8)
    iterations = 2 if speech_like and int(np.count_nonzero(foreground > 0)) < 1200 else 1
    erase = cv2.dilate((foreground > 0).astype(np.uint8), kernel, iterations=iterations)
    return _clip_mask_to_valid_bbox(erase, allowed)


def _runtime_speech_flat_plan(
    *,
    page_id: str,
    job: CleanupJob,
    cleanup_mask: CleanupMask,
    inpaint_mode: str,
    cleanup_class: CleanupClass,
) -> CleanupPlan:
    cleanup_plan_id = f"cplan_phase5_runtime_{_safe_id(page_id)}_{_safe_id(job.cleanup_job_id)}"
    cleanup_class_value = _enum_value(cleanup_class)
    cleanup_tag = _runtime_cleanup_tag_for_class(cleanup_class)
    source_bbox = _valid_bbox(job.source_glyph_erasure_bbox) or _valid_bbox(cleanup_mask.erase_mask_bbox)
    expected_bbox = _valid_bbox(job.source_glyph_erasure_expected_area_bbox)
    proof_scope_bbox = _union_valid_bboxes(source_bbox, expected_bbox)
    parent_cleanup_unit_id = str(getattr(job, "cleanup_unit_id", "") or job.cleanup_job_id or "")
    backend_parameters = {
        "cleanup_tag": cleanup_tag,
        "target_region_ids": list(job.target_region_ids or []),
        "allowed_area": list(cleanup_mask.allowed_area or []),
        "source_glyph_erasure_bbox": list(source_bbox or []),
        "source_glyph_erasure_expected_area_bbox": list(expected_bbox or []),
        "proof_scope": "cleanup_unit_full",
        "proof_scope_bbox": list(proof_scope_bbox or []),
        "parent_cleanup_unit_id": parent_cleanup_unit_id,
        "parent_expected_erasure_bbox": list(expected_bbox or []),
        "represented_source_glyph_ids": list(cleanup_mask.consumed_source_glyph_mask_ids or []),
        "consumed_source_glyph_mask_ids": list(cleanup_mask.consumed_source_glyph_mask_ids or []),
        "missing_source_glyph_mask_ids": list(cleanup_mask.missing_source_glyph_mask_ids or []),
        "effective_mask_status": str(getattr(cleanup_mask, "effective_mask_status", "") or ""),
        "effective_mask_failure_reason": str(getattr(cleanup_mask, "effective_mask_failure_reason", "") or ""),
        "mask_completion_method": str(getattr(cleanup_mask, "mask_completion_method", "") or ""),
        "polarity_mode": str(getattr(cleanup_mask, "polarity_mode", "") or ""),
        "completed_foreground_pixels": getattr(cleanup_mask, "completed_foreground_pixels", None),
        "text_block_coverage_estimate": getattr(cleanup_mask, "text_block_coverage_estimate", None),
        "renderer_consumed_by_phase": "phase5_runtime_contract_false",
        "eligibility_reason": "phase5_general_sourceglyph_allowed_area_contract",
        "effectiveness_closeout_reason": "phase5_runtime_source_residual_backend_repair",
    }
    return CleanupPlan(
        cleanup_plan_id=cleanup_plan_id,
        cleanup_job_id=str(job.cleanup_job_id),
        cleanup_mask_id=str(cleanup_mask.cleanup_mask_id),
        cleanup_class=cleanup_class,
        selected_backend="app.pipeline.cleanup_execution.apply_local_text_removal",
        cleanup_method=f"{cleanup_class_value}_local_text_removal",
        backend_parameters=backend_parameters,
        inpaint_mode=str(inpaint_mode or "fast"),
        crop_context_bbox=list(cleanup_mask.erase_mask_bbox or cleanup_mask.allowed_area or []),
        fallback_policy=["record_failure_no_renderer_consumption"],
        expected_runtime_class="cleanup_owned_phase5_runtime",
        proof_thresholds={
            "mask_source_overlap_ratio_min": PROOF_MASK_SOURCE_OVERLAP_RATIO,
            "source_residual_ratio_max": PROOF_RESIDUAL_RATIO,
            "source_residual_pixel_fraction_max": PROOF_RESIDUAL_PIXEL_FRACTION,
            "changed_outside_allowed_pixels_max": PROOF_CHANGED_OUTSIDE_ALLOWED_PIXELS,
            "changed_outside_allowed_ratio_max": PROOF_CHANGED_OUTSIDE_ALLOWED_RATIO,
            "changed_outside_erase_inside_allowed_ratio_max": PROOF_COLLATERAL_INSIDE_ALLOWED_RATIO,
        },
    )


def _runtime_cleanup_tag_for_class(cleanup_class: CleanupClass) -> str:
    cleanup_class_value = _enum_value(cleanup_class)
    if cleanup_class_value in {
        CleanupClass.SPEECH_FLAT_BUBBLE.value,
        CleanupClass.SPEECH_COMPLEX_BUBBLE.value,
        CleanupClass.SMALL_REACTION.value,
    }:
        return "speech_glyph_local"
    if cleanup_class_value in {
        CleanupClass.CAPTION_DARK_OR_SCREENTONE.value,
        CleanupClass.CAPTION_FLAT_BACKGROUND.value,
        CleanupClass.BACKGROUND_ART_TEXT.value,
        CleanupClass.TITLE_OR_SIGN.value,
        CleanupClass.SIDE_CAPTION_GLYPH_LOCAL.value,
        CleanupClass.ART_ENTANGLED_AMBIGUOUS.value,
    }:
        return "glyph_stroke_local"
    return "minimal_erasure"


def _is_partitionable_mask_rejection(reason: str) -> bool:
    reason_text = str(reason or "").lower()
    if reason_text.startswith("mask_artifact_risk_"):
        return True
    return reason_text in {
        ARTIFACT_RISK_PARTITION_REQUIRED_REASON,
        "erase_mask_too_large",
        "erase_bbox_too_broad",
        "erase_allowed_ratio_too_high",
    }


def _partition_runtime_cleanup_mask(
    cleanup_mask: CleanupMask,
    *,
    rejection_reason: str,
    max_components: int,
) -> list[CleanupMask]:
    if np is None or cv2 is None or not _is_partitionable_mask_rejection(rejection_reason):
        return []
    erase = cleanup_mask.erase_mask
    foreground_mask = cleanup_mask.foreground_mask
    if foreground_mask is None:
        return []
    foreground_arr = np.asarray(foreground_mask)
    if foreground_arr.ndim == 3:
        foreground_arr = np.any(foreground_arr > 0, axis=2).astype(np.uint8)
    if foreground_arr.ndim != 2 or not np.any(foreground_arr > 0):
        return []
    allowed = _valid_bbox(cleanup_mask.allowed_area)
    if allowed is None:
        return []
    foreground_arr = _clip_mask_to_valid_bbox((foreground_arr > 0).astype(np.uint8), allowed)
    if not np.any(foreground_arr > 0):
        return []
    labels_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (foreground_arr > 0).astype(np.uint8),
        connectivity=8,
    )
    component_masks: list[Any] = []
    for label in range(1, labels_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        component_masks.append((area, (labels == label).astype(np.uint8)))
    component_masks.sort(key=lambda item: item[0], reverse=True)

    candidates: list[CleanupMask] = []
    for _area, component in component_masks:
        for local in _bounded_component_slices(component):
            foreground = _clip_mask_to_valid_bbox(local, allowed)
            if not np.any(foreground > 0):
                continue
            geometry_defect = _glyph_foreground_geometry_defect_reason(
                foreground=foreground,
                allowed=allowed,
                cleanup_class=None,
            )
            if geometry_defect:
                continue
            local_erase = _glyph_local_erase_from_foreground(foreground, allowed, None)
            erase_pixels = int(np.count_nonzero(local_erase))
            foreground_pixels = int(np.count_nonzero(foreground))
            erase_bbox = _mask_bbox(local_erase)
            foreground_bbox = _mask_bbox(foreground)
            if erase_bbox is None or foreground_bbox is None:
                continue
            partition_index = len(candidates)
            child_artifact_risk = ""
            child_mask_method = f"{cleanup_mask.mask_method}|bounded_connected_component_partition"
            if _is_artifact_risk_partition_reason(rejection_reason):
                child_mask_method = (
                    f"{child_mask_method}|artifact_risk_bounded_component:"
                    f"{_safe_reason_token(cleanup_mask.artifact_risk)}"
                )
            submask = CleanupMask(
                cleanup_mask_id=f"{cleanup_mask.cleanup_mask_id}_part{partition_index:02d}",
                cleanup_job_id=str(cleanup_mask.cleanup_job_id),
                foreground_mask_source_id=cleanup_mask.foreground_mask_source_id,
                foreground_mask_source_ids=list(cleanup_mask.foreground_mask_source_ids or []),
                consumed_source_glyph_mask_ids=list(cleanup_mask.consumed_source_glyph_mask_ids or []),
                missing_source_glyph_mask_ids=list(cleanup_mask.missing_source_glyph_mask_ids or []),
                foreground_mask_bbox=foreground_bbox,
                foreground_mask_pixels=foreground_pixels,
                erase_mask_bbox=erase_bbox,
                erase_mask_pixels=erase_pixels,
                allowed_area=allowed,
                growth_ratio=_growth_ratio(erase_pixels, foreground_pixels),
                mask_source=f"{cleanup_mask.mask_source}|partitioned_cleanup_component",
                mask_method=child_mask_method,
                rejection_reason="",
                mask_contract_exception_reason=cleanup_mask.mask_contract_exception_reason,
                artifact_risk=child_artifact_risk,
                visual_scope=_partition_visual_scope(cleanup_mask.visual_scope),
                protected=cleanup_mask.protected,
                protection_reason=cleanup_mask.protection_reason,
                **_effective_mask_kwargs(cleanup_mask),
                foreground_mask=foreground,
                erase_mask=local_erase.astype(np.uint8),
            )
            if _mask_exclusion_reason(submask, None):
                minimal = CleanupMask(
                    cleanup_mask_id=submask.cleanup_mask_id,
                    cleanup_job_id=submask.cleanup_job_id,
                    foreground_mask_source_id=submask.foreground_mask_source_id,
                    foreground_mask_source_ids=list(submask.foreground_mask_source_ids or []),
                    consumed_source_glyph_mask_ids=list(submask.consumed_source_glyph_mask_ids or []),
                    missing_source_glyph_mask_ids=list(submask.missing_source_glyph_mask_ids or []),
                    foreground_mask_bbox=foreground_bbox,
                    foreground_mask_pixels=foreground_pixels,
                    erase_mask_bbox=foreground_bbox,
                    erase_mask_pixels=foreground_pixels,
                    allowed_area=allowed,
                    growth_ratio=1.0,
                    mask_source=submask.mask_source,
                    mask_method=f"{child_mask_method}|minimal_foreground_erasure",
                    rejection_reason="",
                    mask_contract_exception_reason=submask.mask_contract_exception_reason,
                    artifact_risk=child_artifact_risk,
                    visual_scope=submask.visual_scope,
                    protected=submask.protected,
                    protection_reason=submask.protection_reason,
                    **_effective_mask_kwargs(submask),
                    foreground_mask=foreground,
                    erase_mask=foreground.copy(),
                )
                if _mask_exclusion_reason(minimal, None):
                    continue
                submask = minimal
            if _mask_exclusion_reason(submask, None):
                continue
            candidates.append(submask)
            if len(candidates) >= max(1, int(max_components)):
                return candidates
    return candidates


def _bounded_component_slices(component: Any) -> list[Any]:
    if np is None:
        return []
    bbox = _mask_bbox(component)
    if bbox is None:
        return []
    if (
        int(np.count_nonzero(component > 0)) <= MAX_ERASE_MASK_PIXELS
        and _bbox_area(bbox) <= MAX_ERASE_BBOX_AREA
    ):
        return [component]
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0
    slices: list[Any] = []
    # Keep partitioning deterministic and bounded.  The long axis is split
    # first, but every slice remains clipped to the original component mask.
    split_count = 3
    if height >= width:
        step = max(1, int((height + split_count - 1) / split_count))
        for index in range(split_count):
            sy0 = y0 + index * step
            sy1 = min(y1, sy0 + step)
            if sy1 <= sy0:
                continue
            local = np.zeros_like(component, dtype=np.uint8)
            local[sy0:sy1, x0:x1] = component[sy0:sy1, x0:x1]
            if np.any(local > 0):
                slices.append(local)
    else:
        step = max(1, int((width + split_count - 1) / split_count))
        for index in range(split_count):
            sx0 = x0 + index * step
            sx1 = min(x1, sx0 + step)
            if sx1 <= sx0:
                continue
            local = np.zeros_like(component, dtype=np.uint8)
            local[y0:y1, sx0:sx1] = component[y0:y1, sx0:sx1]
            if np.any(local > 0):
                slices.append(local)
    return slices


def _run_partitioned_cleanup_attempts(
    *,
    image: Any,
    source_image: Any,
    base_plan_kwargs: Mapping[str, Any],
    partition_masks: Sequence[CleanupMask],
    base_record: Mapping[str, Any],
    target_region_ids: Sequence[str],
    parent_rejection_reason: str,
    page_id: str,
    region_id: str,
    use_gpu: bool,
    model_id: str,
    artifact_dir: str | None,
) -> tuple[list[CleanupResult], list[CleanupProof], list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[CleanupResult] = []
    proofs: list[CleanupProof] = []
    statuses: list[dict[str, Any]] = []
    aggregate_records: list[dict[str, Any]] = []
    job = base_plan_kwargs.get("job")
    partition_count = len(partition_masks)
    for index, partition_mask in enumerate(partition_masks):
        plan = _runtime_speech_flat_plan(
            cleanup_mask=partition_mask,
            **base_plan_kwargs,
        )
        plan = _adaptive_plan_variant(
            plan,
            attempt_index=index,
            strategy="partition_component",
            cleanup_tag=str((plan.backend_parameters or {}).get("cleanup_tag") or "cleanup_effectiveness_retry"),
            retry_reason=parent_rejection_reason,
            suffix=f"_part{index:02d}",
        )
        params = dict(plan.backend_parameters or {})
        params["partitioned_cleanup"] = True
        params["partition_index"] = index
        params["partition_count"] = partition_count
        params["partition_parent_rejection_reason"] = parent_rejection_reason
        params["partition_parent_artifact_risk"] = (
            parent_rejection_reason if _is_artifact_risk_partition_reason(parent_rejection_reason) else ""
        )
        parent_expected = (
            _valid_bbox(params.get("parent_expected_erasure_bbox"))
            or _valid_bbox(params.get("source_glyph_erasure_expected_area_bbox"))
            or _valid_bbox(getattr(job, "source_glyph_erasure_expected_area_bbox", None))
        )
        partition_bbox = _valid_bbox(partition_mask.erase_mask_bbox)
        parent_cleanup_unit_id = str(
            params.get("parent_cleanup_unit_id")
            or getattr(job, "cleanup_unit_id", "")
            or getattr(job, "cleanup_job_id", "")
            or ""
        )
        params["source_glyph_erasure_bbox"] = list(partition_bbox or [])
        params["source_glyph_erasure_expected_area_bbox"] = []
        params["parent_expected_erasure_bbox"] = list(parent_expected or [])
        params["proof_scope"] = "component_partition"
        params["proof_scope_bbox"] = list(partition_bbox or [])
        params["proof_scope_excludes_parent_expected_bbox"] = True
        params["parent_cleanup_unit_id"] = parent_cleanup_unit_id
        params["partition_parent_cleanup_job_id"] = str(getattr(job, "cleanup_job_id", "") or plan.cleanup_job_id)
        params["represented_source_glyph_ids"] = list(partition_mask.consumed_source_glyph_mask_ids or [])
        params["consumed_source_glyph_mask_ids"] = list(partition_mask.consumed_source_glyph_mask_ids or [])
        params["missing_source_glyph_mask_ids"] = list(partition_mask.missing_source_glyph_mask_ids or [])
        plan = CleanupPlan(
            cleanup_plan_id=plan.cleanup_plan_id,
            cleanup_job_id=plan.cleanup_job_id,
            cleanup_mask_id=plan.cleanup_mask_id,
            cleanup_class=plan.cleanup_class,
            selected_backend=plan.selected_backend,
            cleanup_method=f"{plan.cleanup_method}_partition_component",
            backend_parameters=params,
            inpaint_mode=plan.inpaint_mode,
            crop_context_bbox=list(partition_mask.erase_mask_bbox or plan.crop_context_bbox or []),
            fallback_policy=list(plan.fallback_policy or []),
            expected_runtime_class=plan.expected_runtime_class,
            proof_thresholds=dict(plan.proof_thresholds or {}),
        )
        partition_results, partition_proofs, attempt_records = _run_adaptive_cleanup_attempts(
            image=image,
            source_image=source_image,
            cleanup_plan=plan,
            cleanup_mask=partition_mask,
            page_id=page_id,
            region_id=region_id,
            use_gpu=use_gpu,
            model_id=model_id,
            artifact_dir=artifact_dir,
        )
        results.extend(partition_results)
        proofs.extend(partition_proofs)
        result, proof = _select_best_commit_candidate_result(partition_results, partition_proofs)
        if result is None:
            result, proof = _best_failed_attempt(partition_results, partition_proofs)
        if result is None or proof is None:
            statuses.append(
                {
                    **dict(base_record),
                    "runtime_status": "contract_error",
                    "cleanup_mask_id": partition_mask.cleanup_mask_id,
                    "failure_reason": "cleanup_contract_error_invalid_input",
                    "cleanup_outcome_state": "cleanup_contract_error_invalid_input",
                    "contract_error_original_reason": "partition_attempt_missing_result_or_proof",
                    "partitioned_cleanup": True,
                    "partition_index": index,
                    "partition_count": partition_count,
                    "partition_parent_rejection_reason": parent_rejection_reason,
                    "target_region_ids": list(target_region_ids),
                    "renderer_consumed": False,
                }
            )
            continue
        proof_passed = proof.proof_status == ProofStatus.PASSED
        degraded_commit_allowed = _proof_allows_degraded_commit(proof, result, {})
        runtime_status = (
            "passed"
            if proof_passed and result.pixel_changed
            else ("warning" if degraded_commit_allowed else "failed")
        )
        failure_reason = "passed" if runtime_status == "passed" else proof.failure_reason or result.failure_reason
        proof_metrics = _json_safe(proof.metrics or {})
        selected_attempt = _selected_attempt_record(attempt_records, result)
        statuses.append(
            {
                **dict(base_record),
                "runtime_status": runtime_status,
                "cleanup_plan_id": result.cleanup_plan_id,
                "cleanup_mask_id": partition_mask.cleanup_mask_id,
                "cleanup_result_id": result.cleanup_result_id,
                "cleanup_proof_id": proof.cleanup_proof_id,
                "proof_status": _enum_value(proof.proof_status),
                "failure_reason": failure_reason,
                "cleanup_outcome_state": _cleanup_outcome_state_for_failure(
                    failure_reason,
                    runtime_status=runtime_status,
                ),
                "pixel_changed": bool(result.pixel_changed),
                "changed_pixel_count": int(result.changed_pixel_count or 0),
                "cleaned_image_ref": result.cleaned_image_ref,
                "diff_ref": result.diff_ref,
                "mask_ref": result.mask_ref,
                "proof_metrics": proof_metrics,
                "proof_quality_state": proof_metrics.get("proof_quality_state", ""),
                "degraded_commit_allowed": bool(proof_metrics.get("degraded_commit_allowed", False)),
                "source_residual_delta_dark_pixels": proof_metrics.get("source_residual_delta_dark_pixels", 0),
                "source_residual_delta_dark_ratio": proof_metrics.get("source_residual_delta_dark_ratio", 0),
                "collateral_art_damage_delta": proof_metrics.get("collateral_art_damage_delta", 0),
                "proof_scope": proof_metrics.get("proof_scope", "component_partition"),
                "proof_scope_bbox": proof_metrics.get("proof_scope_bbox", list(partition_bbox or [])),
                "parent_cleanup_unit_id": parent_cleanup_unit_id,
                "parent_expected_erasure_bbox": list(parent_expected or []),
                "proof_scope_excludes_parent_expected_bbox": bool(
                    proof_metrics.get("proof_scope_excludes_parent_expected_bbox", True)
                ),
                "adaptive_attempt_count": len(attempt_records),
                "adaptive_attempts": _json_safe(attempt_records),
                "selected_attempt_index": selected_attempt.get("attempt_index") if selected_attempt else None,
                "selected_attempt_strategy": selected_attempt.get("strategy") if selected_attempt else "",
                "partitioned_cleanup": True,
                "partition_index": index,
                "partition_count": partition_count,
                "partition_parent_rejection_reason": parent_rejection_reason,
                "target_region_ids": list(target_region_ids),
                "represented_source_glyph_ids": list(partition_mask.consumed_source_glyph_mask_ids or []),
                "consumed_source_glyph_mask_ids": list(partition_mask.consumed_source_glyph_mask_ids or []),
                "missing_source_glyph_mask_ids": list(partition_mask.missing_source_glyph_mask_ids or []),
                "render_consumption_decision_if_consumed": proof.render_consumption_decision_if_consumed,
                "renderer_consumed": False,
            }
        )
    aggregate_record = _parent_cleanup_unit_aggregate_record(
        base_record=base_record,
        statuses=statuses,
        partition_masks=partition_masks,
        parent_rejection_reason=parent_rejection_reason,
        parent_cleanup_unit_id=str(
            getattr(job, "cleanup_unit_id", "")
            or (base_record.get("cleanup_unit_id") if isinstance(base_record, Mapping) else "")
            or (base_record.get("cleanup_job_id") if isinstance(base_record, Mapping) else "")
            or ""
        ),
    )
    if aggregate_record:
        aggregate_records.append(aggregate_record)
        for status in statuses:
            status["parent_cleanup_unit_aggregate_status"] = aggregate_record.get("aggregate_status", "")
            status["parent_cleanup_unit_aggregate_complete"] = bool(
                aggregate_record.get("aggregate_complete", False)
            )
            status["parent_cleanup_unit_component_attempt_count"] = aggregate_record.get(
                "component_attempt_count",
                0,
            )
    return results, proofs, statuses, aggregate_records


def _parent_cleanup_unit_aggregate_record(
    *,
    base_record: Mapping[str, Any],
    statuses: Sequence[Mapping[str, Any]],
    partition_masks: Sequence[CleanupMask],
    parent_rejection_reason: str,
    parent_cleanup_unit_id: str,
) -> dict[str, Any]:
    if not statuses:
        return {}
    component_attempt_ids = [
        str(status.get("cleanup_result_id") or "")
        for status in statuses
        if str(status.get("cleanup_result_id") or "")
    ]
    component_proof_ids = [
        str(status.get("cleanup_proof_id") or "")
        for status in statuses
        if str(status.get("cleanup_proof_id") or "")
    ]
    component_statuses = [
        {
            "cleanup_result_id": status.get("cleanup_result_id", ""),
            "cleanup_proof_id": status.get("cleanup_proof_id", ""),
            "runtime_status": status.get("runtime_status", ""),
            "proof_status": status.get("proof_status", ""),
            "failure_reason": status.get("failure_reason", ""),
            "pixel_changed": bool(status.get("pixel_changed")),
            "partition_index": status.get("partition_index", ""),
            "proof_scope": status.get("proof_scope", ""),
            "proof_scope_bbox": status.get("proof_scope_bbox", []),
        }
        for status in statuses
    ]
    expected_count = len(partition_masks)
    passed = [
        status
        for status in statuses
        if str(status.get("runtime_status") or "") == "passed" and bool(status.get("pixel_changed"))
    ]
    warning = [
        status
        for status in statuses
        if str(status.get("runtime_status") or "") == "warning" and bool(status.get("pixel_changed"))
    ]
    commit_candidates = passed
    failure_reasons = [
        str(status.get("failure_reason") or "")
        for status in statuses
        if str(status.get("failure_reason") or "") and str(status.get("failure_reason") or "") != "passed"
    ]
    aggregate_status = _parent_cleanup_unit_aggregate_status(
        expected_count=expected_count,
        attempted_count=len(statuses),
        passed_count=len(passed),
        warning_count=len(warning),
        failure_reasons=failure_reasons,
    )
    aggregate_complete = expected_count > 0 and len(commit_candidates) == expected_count and len(statuses) == expected_count
    return {
        "page_id": base_record.get("page_id", ""),
        "cleanup_job_id": base_record.get("cleanup_job_id", ""),
        "parent_cleanup_unit_id": parent_cleanup_unit_id,
        "cleanup_unit_id": base_record.get("cleanup_unit_id", parent_cleanup_unit_id),
        "cleanup_unit_level": base_record.get("cleanup_unit_level", ""),
        "cleanup_unit_anchor_region_id": base_record.get("cleanup_unit_anchor_region_id", ""),
        "cleanup_unit_child_region_ids": list(base_record.get("cleanup_unit_child_region_ids", []) or []),
        "target_region_ids": list(base_record.get("target_region_ids", []) or []),
        "required_source_glyph_mask_ids": list(
            base_record.get("cleanup_unit_required_source_glyph_mask_ids", []) or []
        ),
        "component_attempt_ids": component_attempt_ids,
        "component_proof_ids": component_proof_ids,
        "component_attempt_count": len(statuses),
        "component_expected_count": expected_count,
        "component_passed_count": len(passed),
        "component_warning_count": len(warning),
        "component_commit_candidate_count": len(commit_candidates),
        "component_uncommitted_warning_count": len(warning),
        "component_failed_count": max(0, len(statuses) - len(commit_candidates)),
        "component_statuses": component_statuses,
        "aggregate_status": aggregate_status,
        "aggregate_complete": aggregate_complete,
        "aggregate_completion_claimed": aggregate_complete,
        "aggregate_commit_status": (
            "existing_upstream_commit_path_required"
            if aggregate_complete
            else "unresolved_components_after_bounded_attempts"
        ),
        "parent_rejection_reason": parent_rejection_reason,
        "renderer_consumed": False,
    }


def _parent_cleanup_unit_aggregate_status(
    *,
    expected_count: int,
    attempted_count: int,
    passed_count: int,
    warning_count: int = 0,
    failure_reasons: Sequence[str],
) -> str:
    if expected_count <= 0 or attempted_count <= 0:
        return "cleanup_contract_error_invalid_input"
    if passed_count == expected_count and attempted_count == expected_count:
        return "cleanup_completed"
    if passed_count + warning_count == expected_count and attempted_count == expected_count:
        return "cleanup_partially_completed_with_warnings"
    joined = " ".join(str(reason or "").lower() for reason in failure_reasons)
    if "sourceglyph" in joined or "source_glyph" in joined:
        return "cleanup_contract_error_missing_sourceglyph"
    if "unsupported" in joined or "non_phase5" in joined:
        return "cleanup_contract_error_invalid_input"
    if "backend_noop" in joined or "no_pixels_changed" in joined:
        return "cleanup_failed_no_safe_pixel_change"
    if "protected" in joined or "preserve" in joined or "sfx" in joined or "decorative" in joined:
        return "cleanup_contract_error_invalid_input"
    if "source_residual" in joined:
        return "cleanup_partially_completed_with_warnings"
    if "mask" in joined or "scope" in joined or "erase" in joined:
        return "cleanup_partially_completed_with_warnings"
    return "cleanup_partially_completed_with_warnings"


def _run_adaptive_cleanup_attempts(
    *,
    image: Any,
    source_image: Any,
    cleanup_plan: CleanupPlan,
    cleanup_mask: CleanupMask,
    page_id: str,
    region_id: str,
    use_gpu: bool,
    model_id: str,
    artifact_dir: str | None,
) -> tuple[list[CleanupResult], list[CleanupProof], list[dict[str, Any]]]:
    adaptive_started = time.time()
    results: list[CleanupResult] = []
    proofs: list[CleanupProof] = []
    attempts: list[dict[str, Any]] = []

    first_plan = _adaptive_plan_variant(
        cleanup_plan,
        attempt_index=0,
        strategy="current_conservative",
        cleanup_tag=str((cleanup_plan.backend_parameters or {}).get("cleanup_tag") or "cleanup_effectiveness_retry"),
        retry_reason="initial_attempt",
        suffix="",
    )
    _page014_timeout_checkpoint(
        "adaptive_attempt",
        "start",
        page_id=page_id,
        region_id=region_id,
        cleanup_plan_id=str(first_plan.cleanup_plan_id),
        cleanup_mask_id=str(cleanup_mask.cleanup_mask_id),
        attempt_index=0,
        strategy="current_conservative",
        retry_reason="initial_attempt",
    )
    first_result, first_proof = _execute_cleanup_attempt(
        image=image,
        proof_before=image,
        source_image=source_image,
        cleanup_plan=first_plan,
        cleanup_mask=cleanup_mask,
        page_id=page_id,
        region_id=region_id,
        use_gpu=use_gpu,
        model_id=model_id,
        artifact_dir=artifact_dir,
        attempt_index=0,
        strategy="current_conservative",
        retry_reason="initial_attempt",
    )
    results.append(first_result)
    proofs.append(first_proof)
    attempts.append(_adaptive_attempt_record(first_result, first_proof, 0, "current_conservative", "initial_attempt"))
    _page014_timeout_checkpoint(
        "adaptive_attempt",
        "end",
        page_id=page_id,
        region_id=region_id,
        cleanup_plan_id=str(first_plan.cleanup_plan_id),
        attempt_index=0,
        strategy="current_conservative",
        proof_status=_enum_value(first_proof.proof_status),
        failure_reason=first_proof.failure_reason or first_result.failure_reason,
        runtime_ms=first_result.runtime_ms,
    )
    if _proof_passed_with_pixels(first_result, first_proof):
        _page014_timeout_checkpoint(
            "adaptive_cleanup_attempts",
            "early_pass",
            page_id=page_id,
            region_id=region_id,
            attempt_count=len(attempts),
            elapsed_ms=round((time.time() - adaptive_started) * 1000.0, 3),
        )
        return results, proofs, attempts

    last_result = first_result
    last_proof = first_proof
    attempt_index = 1

    if _should_run_backend_noop_fallback(last_result, last_proof):
        for strategy, cleanup_tag in (
            ("backend_noop_deterministic_fallback", "deterministic_local_fill"),
            ("minimal_erasure_fallback", "minimal_erasure"),
        ):
            noop_plan = _adaptive_plan_variant(
                cleanup_plan,
                attempt_index=attempt_index,
                strategy=strategy,
                cleanup_tag=cleanup_tag,
                retry_reason=last_proof.failure_reason or last_result.failure_reason or "backend_noop_or_error",
                suffix=f"_attempt{attempt_index}_{strategy}",
            )
            _page014_timeout_checkpoint(
                "adaptive_attempt",
                "start",
                page_id=page_id,
                region_id=region_id,
                cleanup_plan_id=str(noop_plan.cleanup_plan_id),
                cleanup_mask_id=str(cleanup_mask.cleanup_mask_id),
                attempt_index=attempt_index,
                strategy=strategy,
                retry_reason=last_proof.failure_reason or last_result.failure_reason or "backend_noop_or_error",
            )
            noop_result, noop_proof = _execute_cleanup_attempt(
                image=image,
                proof_before=image,
                source_image=source_image,
                cleanup_plan=noop_plan,
                cleanup_mask=cleanup_mask,
                page_id=page_id,
                region_id=region_id,
                use_gpu=use_gpu,
                model_id=model_id,
                artifact_dir=artifact_dir,
                attempt_index=attempt_index,
                strategy=strategy,
                retry_reason=last_proof.failure_reason or last_result.failure_reason or "backend_noop_or_error",
            )
            results.append(noop_result)
            proofs.append(noop_proof)
            attempts.append(
                _adaptive_attempt_record(
                    noop_result,
                    noop_proof,
                    attempt_index,
                    strategy,
                    last_proof.failure_reason or last_result.failure_reason or "backend_noop_or_error",
                )
            )
            _page014_timeout_checkpoint(
                "adaptive_attempt",
                "end",
                page_id=page_id,
                region_id=region_id,
                cleanup_plan_id=str(noop_plan.cleanup_plan_id),
                attempt_index=attempt_index,
                strategy=strategy,
                proof_status=_enum_value(noop_proof.proof_status),
                failure_reason=noop_proof.failure_reason or noop_result.failure_reason,
                runtime_ms=noop_result.runtime_ms,
            )
            last_result = noop_result
            last_proof = noop_proof
            attempt_index += 1
            if _proof_passed_with_pixels(noop_result, noop_proof):
                _page014_timeout_checkpoint(
                    "adaptive_cleanup_attempts",
                    "early_pass",
                    page_id=page_id,
                    region_id=region_id,
                    attempt_count=len(attempts),
                    elapsed_ms=round((time.time() - adaptive_started) * 1000.0, 3),
                )
                return results, proofs, attempts
            if not _should_run_backend_noop_fallback(noop_result, noop_proof):
                break

    if _should_run_residual_retry(last_proof):
        residual_plan = _adaptive_plan_variant(
            cleanup_plan,
            attempt_index=attempt_index,
            strategy="residual_retry",
            cleanup_tag="cleanup_effectiveness_retry",
            retry_reason=last_proof.failure_reason or "source_residual_remaining",
            suffix=f"_attempt{attempt_index}_residual_retry",
        )
        _page014_timeout_checkpoint(
            "adaptive_attempt",
            "start",
            page_id=page_id,
            region_id=region_id,
            cleanup_plan_id=str(residual_plan.cleanup_plan_id),
            cleanup_mask_id=str(cleanup_mask.cleanup_mask_id),
            attempt_index=attempt_index,
            strategy="residual_retry",
            retry_reason=last_proof.failure_reason or "source_residual_remaining",
        )
        residual_image = last_result.cleaned_image if last_result.cleaned_image is not None else image
        residual_result, residual_proof = _execute_cleanup_attempt(
            image=residual_image,
            proof_before=image,
            source_image=source_image,
            cleanup_plan=residual_plan,
            cleanup_mask=cleanup_mask,
            page_id=page_id,
            region_id=region_id,
            use_gpu=use_gpu,
            model_id=model_id,
            artifact_dir=artifact_dir,
            attempt_index=attempt_index,
            strategy="residual_retry",
            retry_reason=last_proof.failure_reason or "source_residual_remaining",
        )
        results.append(residual_result)
        proofs.append(residual_proof)
        attempts.append(
            _adaptive_attempt_record(
                residual_result,
                residual_proof,
                attempt_index,
                "residual_retry",
                last_proof.failure_reason or "source_residual_remaining",
            )
        )
        _page014_timeout_checkpoint(
            "adaptive_attempt",
            "end",
            page_id=page_id,
            region_id=region_id,
            cleanup_plan_id=str(residual_plan.cleanup_plan_id),
            attempt_index=attempt_index,
            strategy="residual_retry",
            proof_status=_enum_value(residual_proof.proof_status),
            failure_reason=residual_proof.failure_reason or residual_result.failure_reason,
            runtime_ms=residual_result.runtime_ms,
        )
        last_result = residual_result
        last_proof = residual_proof
        attempt_index += 1
        if _proof_passed_with_pixels(residual_result, residual_proof):
            _page014_timeout_checkpoint(
                "adaptive_cleanup_attempts",
                "early_pass",
                page_id=page_id,
                region_id=region_id,
                attempt_count=len(attempts),
                elapsed_ms=round((time.time() - adaptive_started) * 1000.0, 3),
            )
            return results, proofs, attempts

    if _should_run_restoration_retry(last_proof):
        restoration_plan = _adaptive_plan_variant(
            cleanup_plan,
            attempt_index=attempt_index,
            strategy="restoration_scale_back_retry",
            cleanup_tag="speech_strong",
            retry_reason=last_proof.failure_reason or "collateral_or_destructive_fill",
            suffix=f"_attempt{attempt_index}_restoration_retry",
        )
        _page014_timeout_checkpoint(
            "adaptive_attempt",
            "start",
            page_id=page_id,
            region_id=region_id,
            cleanup_plan_id=str(restoration_plan.cleanup_plan_id),
            cleanup_mask_id=str(cleanup_mask.cleanup_mask_id),
            attempt_index=attempt_index,
            strategy="restoration_scale_back_retry",
            retry_reason=last_proof.failure_reason or "collateral_or_destructive_fill",
        )
        restoration_result, restoration_proof = _execute_cleanup_attempt(
            image=image,
            proof_before=image,
            source_image=source_image,
            cleanup_plan=restoration_plan,
            cleanup_mask=cleanup_mask,
            page_id=page_id,
            region_id=region_id,
            use_gpu=use_gpu,
            model_id=model_id,
            artifact_dir=artifact_dir,
            attempt_index=attempt_index,
            strategy="restoration_scale_back_retry",
            retry_reason=last_proof.failure_reason or "collateral_or_destructive_fill",
        )
        results.append(restoration_result)
        proofs.append(restoration_proof)
        attempts.append(
            _adaptive_attempt_record(
                restoration_result,
                restoration_proof,
                attempt_index,
                "restoration_scale_back_retry",
                last_proof.failure_reason or "collateral_or_destructive_fill",
            )
        )
        _page014_timeout_checkpoint(
            "adaptive_attempt",
            "end",
            page_id=page_id,
            region_id=region_id,
            cleanup_plan_id=str(restoration_plan.cleanup_plan_id),
            attempt_index=attempt_index,
            strategy="restoration_scale_back_retry",
            proof_status=_enum_value(restoration_proof.proof_status),
            failure_reason=restoration_proof.failure_reason or restoration_result.failure_reason,
            runtime_ms=restoration_result.runtime_ms,
        )

    _page014_timeout_checkpoint(
        "adaptive_cleanup_attempts",
        "end",
        page_id=page_id,
        region_id=region_id,
        attempt_count=len(attempts),
        elapsed_ms=round((time.time() - adaptive_started) * 1000.0, 3),
    )
    return results, proofs, attempts


def _execute_cleanup_attempt(
    *,
    image: Any,
    proof_before: Any,
    source_image: Any,
    cleanup_plan: CleanupPlan,
    cleanup_mask: CleanupMask,
    page_id: str,
    region_id: str,
    use_gpu: bool,
    model_id: str,
    artifact_dir: str | None,
    attempt_index: int,
    strategy: str,
    retry_reason: str,
) -> tuple[CleanupResult, CleanupProof]:
    attempt_started = time.time()
    _page014_timeout_checkpoint(
        "cleanup_attempt_execute",
        "start",
        page_id=page_id,
        region_id=region_id,
        cleanup_plan_id=str(cleanup_plan.cleanup_plan_id),
        cleanup_mask_id=str(cleanup_mask.cleanup_mask_id),
        attempt_index=attempt_index,
        strategy=strategy,
        retry_reason=retry_reason,
    )
    result = execute_cleanup_runtime_plan(
        image=image,
        cleanup_plan=cleanup_plan,
        cleanup_mask=cleanup_mask,
        page_id=page_id,
        region_id=region_id,
        use_gpu=use_gpu,
        model_id=model_id,
        artifact_dir=artifact_dir,
    )
    result.backend_parameters.setdefault("attempt_index", attempt_index)
    result.backend_parameters.setdefault("strategy", strategy)
    result.backend_parameters.setdefault("retry_reason", retry_reason)
    result.backend_parameters.setdefault("adaptive_cleanup_bounded", True)
    for key in (
        "proof_scope",
        "proof_scope_bbox",
        "parent_cleanup_unit_id",
        "parent_expected_erasure_bbox",
        "source_glyph_erasure_bbox",
        "source_glyph_erasure_expected_area_bbox",
        "partitioned_cleanup",
        "partition_index",
        "partition_count",
        "partition_parent_rejection_reason",
        "partition_parent_cleanup_job_id",
        "represented_source_glyph_ids",
        "consumed_source_glyph_mask_ids",
        "missing_source_glyph_mask_ids",
    ):
        if key in cleanup_plan.backend_parameters:
            result.backend_parameters.setdefault(key, cleanup_plan.backend_parameters.get(key))
    _page014_timeout_checkpoint(
        "cleanup_attempt_proof",
        "start",
        page_id=page_id,
        region_id=region_id,
        cleanup_plan_id=str(cleanup_plan.cleanup_plan_id),
        cleanup_result_id=str(result.cleanup_result_id),
        attempt_index=attempt_index,
        strategy=strategy,
    )
    proof = prove_cleanup_result(
        source_image=source_image,
        before_image=proof_before,
        cleanup_result=result,
        cleanup_plan=cleanup_plan,
        cleanup_mask=cleanup_mask,
    )
    _page014_timeout_checkpoint(
        "cleanup_attempt_execute",
        "end",
        page_id=page_id,
        region_id=region_id,
        cleanup_plan_id=str(cleanup_plan.cleanup_plan_id),
        cleanup_result_id=str(result.cleanup_result_id),
        cleanup_proof_id=str(proof.cleanup_proof_id),
        attempt_index=attempt_index,
        strategy=strategy,
        proof_status=_enum_value(proof.proof_status),
        failure_reason=proof.failure_reason or result.failure_reason,
        elapsed_ms=round((time.time() - attempt_started) * 1000.0, 3),
    )
    return result, proof


def _adaptive_plan_variant(
    base_plan: CleanupPlan,
    *,
    attempt_index: int,
    strategy: str,
    cleanup_tag: str,
    retry_reason: str,
    suffix: str,
) -> CleanupPlan:
    params = dict(base_plan.backend_parameters or {})
    params.update(
        {
            "adaptive_cleanup_bounded": True,
            "attempt_index": attempt_index,
            "strategy": strategy,
            "cleanup_tag": cleanup_tag,
            "retry_reason": retry_reason,
            "residual_retry_max": 1,
            "restoration_retry_max": 1,
            "allowed_area_expanded": False,
            "erase_limits_unchanged": True,
        }
    )
    return CleanupPlan(
        cleanup_plan_id=f"{base_plan.cleanup_plan_id}{suffix}",
        cleanup_job_id=base_plan.cleanup_job_id,
        cleanup_mask_id=base_plan.cleanup_mask_id,
        cleanup_class=base_plan.cleanup_class,
        selected_backend=base_plan.selected_backend,
        cleanup_method=base_plan.cleanup_method,
        backend_parameters=params,
        inpaint_mode=base_plan.inpaint_mode,
        crop_context_bbox=list(base_plan.crop_context_bbox or []),
        fallback_policy=list(base_plan.fallback_policy or []),
        expected_runtime_class=base_plan.expected_runtime_class,
        proof_thresholds=dict(base_plan.proof_thresholds or {}),
    )


def _proof_passed_with_pixels(result: CleanupResult, proof: CleanupProof) -> bool:
    return bool(result.pixel_changed and proof.proof_status == ProofStatus.PASSED)


def _proof_allows_degraded_commit(
    proof: CleanupProof | None,
    result: CleanupResult | None = None,
    status: Mapping[str, Any] | None = None,
) -> bool:
    if proof is None:
        return False
    metrics = proof.metrics or {}
    if not bool(metrics.get("degraded_commit_allowed")):
        return False
    if result is not None and (not result.pixel_changed or int(result.changed_pixel_count or 0) <= 0):
        return False
    if status is not None:
        runtime_status = str(status.get("runtime_status") or "")
        if runtime_status and runtime_status not in {"warning", "failed", "passed"}:
            return False
    if not bool(metrics.get("source_residual_improved")):
        return False
    if bool(metrics.get("source_visible_area_increased")):
        return False
    if not bool(metrics.get("allowed_area_safe")):
        return False
    if not bool(metrics.get("collateral_safe")):
        return False
    if bool(metrics.get("broad_white_patch_risk")):
        return False
    return True


def _proof_commit_candidate(result: CleanupResult, proof: CleanupProof) -> bool:
    return _proof_passed_with_pixels(result, proof)


def _should_run_backend_noop_fallback(result: CleanupResult, proof: CleanupProof | None) -> bool:
    if not result.pixel_changed or int(result.changed_pixel_count or 0) <= 0:
        return True
    candidate_status = str((result.backend_parameters or {}).get("candidate_status") or "")
    if candidate_status in {"no_pixel_change", "completed_no_pixel_change"}:
        return True
    reason = str((proof.failure_reason if proof is not None else "") or result.failure_reason or "")
    return reason in {"backend_noop_or_error", "no_pixels_changed"}


def _should_run_residual_retry(proof: CleanupProof) -> bool:
    reason = str(proof.failure_reason or "")
    if reason not in {
        "source_residual_remaining",
        "source_residual_remaining_in_source_glyph_erasure_bbox",
        "source_residual_remaining_in_component_partition_bbox",
    }:
        return False
    metrics = proof.metrics or {}
    overlap = _float_or_none(metrics.get("mask_source_overlap_ratio"))
    if overlap is not None and overlap < PROOF_MASK_SOURCE_OVERLAP_RATIO:
        return False
    changed_outside = _int_or_none(metrics.get("changed_outside_allowed_pixels")) or 0
    changed_outside_ratio = _float_or_none(metrics.get("changed_outside_allowed_ratio")) or 0.0
    collateral = _float_or_none(metrics.get("changed_outside_erase_inside_allowed_ratio")) or 0.0
    return (
        changed_outside <= PROOF_CHANGED_OUTSIDE_ALLOWED_PIXELS
        and changed_outside_ratio <= PROOF_CHANGED_OUTSIDE_ALLOWED_RATIO
        and collateral <= PROOF_COLLATERAL_INSIDE_ALLOWED_RATIO
    )


def _should_run_restoration_retry(proof: CleanupProof) -> bool:
    reason = str(proof.failure_reason or "")
    if reason in {"changed_outside_allowed_area", "collateral_change_too_broad"}:
        return True
    metrics = proof.metrics or {}
    changed_outside = _int_or_none(metrics.get("changed_outside_allowed_pixels")) or 0
    changed_outside_ratio = _float_or_none(metrics.get("changed_outside_allowed_ratio")) or 0.0
    collateral = _float_or_none(metrics.get("changed_outside_erase_inside_allowed_ratio")) or 0.0
    return (
        changed_outside > PROOF_CHANGED_OUTSIDE_ALLOWED_PIXELS
        or changed_outside_ratio > PROOF_CHANGED_OUTSIDE_ALLOWED_RATIO
        or collateral > PROOF_COLLATERAL_INSIDE_ALLOWED_RATIO
    )


def _adaptive_attempt_record(
    result: CleanupResult,
    proof: CleanupProof,
    attempt_index: int,
    strategy: str,
    retry_reason: str,
) -> dict[str, Any]:
    metrics = proof.metrics or {}
    return {
        "attempt_index": attempt_index,
        "strategy": strategy,
        "backend": result.backend_name or result.execution_backend,
        "cleanup_result_id": result.cleanup_result_id,
        "cleanup_proof_id": proof.cleanup_proof_id,
        "proof_status": _enum_value(proof.proof_status),
        "failure_reason": proof.failure_reason or result.failure_reason,
        "pixel_changed": bool(result.pixel_changed),
        "cleaned_image_ref": result.cleaned_image_ref,
        "diff_ref": result.diff_ref,
        "mask_ref": result.mask_ref,
        "runtime_ms": result.runtime_ms,
        "residual_status": _residual_status(metrics, proof),
        "collateral_status": _collateral_status(metrics, proof),
        "allowed_area_mutation_status": _allowed_area_status(metrics),
        "proof_scope": metrics.get("proof_scope", ""),
        "proof_scope_bbox": metrics.get("proof_scope_bbox", []),
        "parent_cleanup_unit_id": metrics.get("parent_cleanup_unit_id", ""),
        "parent_expected_erasure_bbox": metrics.get("parent_expected_erasure_bbox", []),
        "proof_quality_state": metrics.get("proof_quality_state", ""),
        "degraded_commit_allowed": bool(metrics.get("degraded_commit_allowed", False)),
        "source_residual_delta_dark_pixels": metrics.get("source_residual_delta_dark_pixels", 0),
        "source_residual_delta_dark_ratio": metrics.get("source_residual_delta_dark_ratio", 0),
        "collateral_art_damage_delta": metrics.get("collateral_art_damage_delta", 0),
        "broad_white_patch_risk": bool(metrics.get("broad_white_patch_risk", False)),
        "commit_candidate": _proof_commit_candidate(result, proof),
        "retry_reason": retry_reason,
    }


def _selected_attempt_record(
    attempt_records: Sequence[dict[str, Any]],
    cleanup_result: CleanupResult | None,
) -> dict[str, Any]:
    if cleanup_result is None:
        return {}
    result_id = str(cleanup_result.cleanup_result_id)
    for record in attempt_records:
        if str(record.get("cleanup_result_id") or "") == result_id:
            return dict(record)
    return {}


def _residual_status(metrics: Mapping[str, Any], proof: CleanupProof) -> str:
    if proof.proof_status == ProofStatus.PASSED:
        return "passed"
    reason = str(proof.failure_reason or "")
    if "source_residual" in reason:
        return "failed"
    return "not_primary_failure"


def _collateral_status(metrics: Mapping[str, Any], proof: CleanupProof) -> str:
    reason = str(proof.failure_reason or "")
    if reason in {"changed_outside_allowed_area", "collateral_change_too_broad"}:
        return "failed"
    changed_outside = _int_or_none(metrics.get("changed_outside_allowed_pixels")) or 0
    collateral = _float_or_none(metrics.get("changed_outside_erase_inside_allowed_ratio")) or 0.0
    if changed_outside > PROOF_CHANGED_OUTSIDE_ALLOWED_PIXELS or collateral > PROOF_COLLATERAL_INSIDE_ALLOWED_RATIO:
        return "failed"
    return "passed_or_not_primary"


def _allowed_area_status(metrics: Mapping[str, Any]) -> str:
    changed_outside = _int_or_none(metrics.get("changed_outside_allowed_pixels")) or 0
    changed_outside_ratio = _float_or_none(metrics.get("changed_outside_allowed_ratio")) or 0.0
    if changed_outside > PROOF_CHANGED_OUTSIDE_ALLOWED_PIXELS or changed_outside_ratio > PROOF_CHANGED_OUTSIDE_ALLOWED_RATIO:
        return "changed_outside_allowed_area"
    return "contained"


def _cleanup_outcome_state_for_failure(failure_reason: str, *, runtime_status: str = "blocked") -> str:
    reason = str(failure_reason or "").lower()
    if "sourceglyph_generation_or_foreground_contract_defect" in reason:
        return "sourceglyph_generation_or_foreground_contract_defect"
    if runtime_status == "passed" or reason == "passed":
        return "cleanup_completed"
    if runtime_status == "warning":
        if "source_residual" in reason:
            return "safe_partial_attempt_uncommitted"
        return "cleanup_partially_completed_with_warnings"
    if reason.startswith("cleanup_contract_error_"):
        return reason
    if "component_limit" in reason:
        return "cleanup_contract_error_component_limit"
    if "invalid" in reason or "expected_exactly_one" in reason or "missing_result_or_proof" in reason:
        return "cleanup_contract_error_invalid_input"
    if "source_residual" in reason:
        return "cleanup_restoration_attempted_with_warning"
    if "backend_noop" in reason or "no_pixels_changed" in reason or "no_pixel_change" in reason:
        return "cleanup_failed_no_safe_pixel_change"
    if "sourceglyph" in reason or "source_glyph" in reason or "source_ungrounded" in reason:
        return "cleanup_contract_error_missing_sourceglyph"
    if "erase" in reason or "too_large" in reason:
        return "cleanup_degraded_minimal_erasure_applied"
    if "mask" in reason and "source_residual" not in reason:
        return "cleanup_contract_error_invalid_input"
    if "collateral" in reason or "destructive" in reason or "outside_allowed" in reason:
        return "cleanup_restoration_attempted_with_warning"
    if "commit" in reason:
        return "cleanup_partially_completed_with_warnings"
    if "non_phase5" in reason or "unsupported" in reason or "runtime_supported" in reason:
        return "cleanup_contract_error_invalid_input"
    if "protected" in reason or "preserve" in reason or "sfx" in reason or "decorative" in reason:
        return "cleanup_contract_error_invalid_input"
    if "artifact_risk" in reason:
        return "cleanup_partially_completed_with_warnings"
    return "cleanup_partially_completed_with_warnings"


def _contract_error_for_reason(reason: str) -> str:
    text = str(reason or "").lower()
    if "sourceglyph_generation_or_foreground_contract_defect" in text:
        return "sourceglyph_generation_or_foreground_contract_defect"
    if "component_limit" in text:
        return "cleanup_contract_error_component_limit"
    if (
        "sourceglyph" in text
        or "source_glyph" in text
        or "source_ungrounded" in text
        or "source_text_missing" in text
        or "consumable_sourceglyph" in text
    ):
        return "cleanup_contract_error_missing_sourceglyph"
    if "empty" in text or "missing" in text or "invalid" in text or "expected_exactly_one" in text:
        return "cleanup_contract_error_invalid_input"
    if "bbox" in text or "allowed_area" in text or "dimensions" in text:
        return "cleanup_contract_error_invalid_input"
    if "preserve" in text or "sfx" in text or "decorative" in text or "protected" in text:
        return "cleanup_contract_error_invalid_input"
    if "unsupported" in text or "non_phase5" in text or "runtime_supported" in text:
        return "cleanup_contract_error_invalid_input"
    return "cleanup_contract_error_invalid_input"


def _changed_pixel_count(before: Any, after: Any) -> int:
    if np is None:
        return 0
    try:
        before_np = np.asarray(before.convert("RGB") if hasattr(before, "convert") else before)
        after_np = np.asarray(after.convert("RGB") if hasattr(after, "convert") else after)
        if before_np.shape != after_np.shape:
            return 0
        diff = np.abs(after_np.astype(np.int16) - before_np.astype(np.int16))
        return int(np.count_nonzero(np.any(diff > 8, axis=2) if diff.ndim == 3 else diff > 8))
    except Exception:
        return 0


def _write_runtime_artifacts(
    *,
    artifact_dir: str | None,
    page_id: str,
    region_id: str,
    cleanup_plan: CleanupPlan,
    cleanup_mask: CleanupMask,
    before: Any,
    cleaned: Any,
) -> dict[str, str]:
    if not artifact_dir or Image is None:
        return {}
    try:
        os.makedirs(artifact_dir, exist_ok=True)
        stem = f"{_safe_id(page_id)}_{_safe_id(region_id)}_{_safe_id(cleanup_plan.cleanup_plan_id)}"
        cleaned_path = os.path.join(artifact_dir, f"{stem}_cleaned.png")
        diff_path = os.path.join(artifact_dir, f"{stem}_diff.png")
        mask_path = os.path.join(artifact_dir, f"{stem}_mask.png")
        crop_path = os.path.join(artifact_dir, f"{stem}_cleaned_crop.png")
        _save_image(cleaned, cleaned_path)
        _save_diff(before, cleaned, diff_path)
        _save_mask(cleanup_mask.erase_mask, mask_path)
        _save_crop(cleaned, cleanup_mask.erase_mask_bbox or cleanup_mask.allowed_area, crop_path)
        return {
            "cleaned_image_ref": cleaned_path,
            "cleaned_crop_ref": crop_path,
            "diff_ref": diff_path,
            "mask_ref": mask_path,
        }
    except Exception:
        return {}


def _save_image(image: Any, path: str) -> None:
    if hasattr(image, "save"):
        image.save(path)


def _save_diff(before: Any, after: Any, path: str) -> None:
    if np is None or Image is None:
        return
    before_np = np.asarray(before.convert("RGB") if hasattr(before, "convert") else before)
    after_np = np.asarray(after.convert("RGB") if hasattr(after, "convert") else after)
    if before_np.shape != after_np.shape:
        return
    diff = np.abs(after_np.astype(np.int16) - before_np.astype(np.int16))
    if diff.ndim == 3:
        diff_gray = np.max(diff, axis=2).astype(np.uint8)
    else:
        diff_gray = diff.astype(np.uint8)
    Image.fromarray(diff_gray).save(path)


def _save_mask(mask: Any, path: str) -> None:
    if np is None or Image is None or mask is None:
        return
    arr = np.asarray(mask)
    if arr.ndim != 2:
        return
    Image.fromarray((arr > 0).astype(np.uint8) * 255).save(path)


def _save_crop(image: Any, bbox: Sequence[int] | None, path: str) -> None:
    box = _valid_bbox(bbox)
    if box is None or not hasattr(image, "crop"):
        return
    image.crop(tuple(box)).save(path)


def _clip_mask_to_valid_bbox(mask: Any, bbox: Sequence[int]) -> Any:
    out = np.zeros(mask.shape[:2], dtype=np.uint8)
    box = _valid_bbox(bbox)
    if box is None:
        return out
    x0, y0, x1, y1 = box
    height, width = out.shape[:2]
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = (mask[y0:y1, x0:x1] > 0).astype(np.uint8)
    return out


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _job_exclusion_reason(job: CleanupJob) -> str:
    if bool(job.protected):
        return job.protection_reason or "job_protected"
    combined = " ".join(
        str(value or "")
        for value in (
            _enum_value(job.cleanup_class),
            job.route_intent,
            job.semantic_class,
            job.cleanup_mode,
            job.container_type,
            " ".join(job.source_glyph_mask_ids or []),
        )
    ).lower()
    for blocked in (
        "preserve",
        "sfx",
        "decorative",
        "non_text",
        "non_translation_art",
        "art_only",
        "source_grounding_protected",
        "source_ungrounded",
    ):
        if blocked in combined:
            return f"pilot_excludes_{blocked}"
    return ""


def _mask_exclusion_reason(cleanup_mask: CleanupMask, image_size: tuple[int, int] | None) -> str:
    if cleanup_mask.protected:
        return cleanup_mask.protection_reason or "mask_protected"
    if cleanup_mask.rejection_reason:
        return f"mask_rejected_{cleanup_mask.rejection_reason}"
    protected_conflict = _mask_protected_conflict_reason(cleanup_mask)
    if protected_conflict:
        return protected_conflict
    visual_scope = _normalise_cleanup_visual_scope(cleanup_mask.visual_scope)
    if not visual_scope:
        return "visual_scope_missing"
    if visual_scope not in VALID_CLEANUP_VISUAL_SCOPES:
        return f"unsupported_visual_scope_{_safe_reason_token(visual_scope)}"
    if not str(cleanup_mask.cleanup_job_id or "").strip():
        return "cleanup_job_id_missing"
    consumed_source_ids = _unique_texts(cleanup_mask.consumed_source_glyph_mask_ids)
    missing_source_ids = _unique_texts(cleanup_mask.missing_source_glyph_mask_ids)
    if missing_source_ids:
        return "missing_required_source_glyph_evidence"
    if not consumed_source_ids:
        return "consumed_source_glyph_evidence_missing"
    if visual_scope == "source_glyph_local" and len(consumed_source_ids) != 1:
        return "source_glyph_local_membership_count_mismatch"
    if visual_scope == "source_glyph_union" and len(consumed_source_ids) <= 1:
        return "source_glyph_union_membership_count_mismatch"
    if visual_scope == "source_glyph_union_partition" and len(consumed_source_ids) <= 1:
        return "source_glyph_union_partition_membership_count_mismatch"
    if cleanup_mask.foreground_mask is None or cleanup_mask.erase_mask is None:
        return "mask_raw_arrays_missing"
    if not _valid_bbox(cleanup_mask.allowed_area):
        return "allowed_area_missing_or_invalid"
    if not _valid_bbox(cleanup_mask.erase_mask_bbox):
        return "erase_mask_bbox_missing_or_invalid"
    if cleanup_mask.artifact_risk and not _is_partition_component_mask(cleanup_mask):
        return ARTIFACT_RISK_PARTITION_REQUIRED_REASON
    growth = float(cleanup_mask.growth_ratio if cleanup_mask.growth_ratio is not None else 999.0)
    erase_pixels = int(cleanup_mask.erase_mask_pixels or 0)
    allowed_area = _bbox_area(cleanup_mask.allowed_area)
    erase_bbox_area = _bbox_area(cleanup_mask.erase_mask_bbox)
    growth_exception = str(getattr(cleanup_mask, "mask_contract_exception_reason", "") or "") in {
        "caption_flat_small_mask_growth_exception",
        "speech_flat_runtime_dark_pixel_growth_exception",
    }
    if growth > MAX_PLAN_GROWTH_RATIO and not growth_exception:
        return "growth_ratio_too_high"
    if erase_pixels <= 0:
        return "erase_mask_empty"
    if erase_pixels > MAX_ERASE_MASK_PIXELS:
        return "erase_mask_too_large"
    if erase_bbox_area > MAX_ERASE_BBOX_AREA:
        return "erase_bbox_too_broad"
    if erase_pixels / max(1, allowed_area) > MAX_ERASE_ALLOWED_RATIO:
        return "erase_allowed_ratio_too_high"
    if image_size and image_size[0] > 0 and image_size[1] > 0:
        page_area = int(image_size[0]) * int(image_size[1])
        if allowed_area / max(1, page_area) > MAX_ALLOWED_PAGE_RATIO:
            return "allowed_area_too_broad"
    return ""


def _is_artifact_risk_partition_reason(reason: str) -> bool:
    reason_text = str(reason or "").lower()
    return reason_text == ARTIFACT_RISK_PARTITION_REQUIRED_REASON or reason_text.startswith("mask_artifact_risk_")


def _is_partition_component_mask(cleanup_mask: CleanupMask) -> bool:
    visual_scope = _normalise_cleanup_visual_scope(getattr(cleanup_mask, "visual_scope", ""))
    if visual_scope == "source_glyph_union_partition":
        return True
    method = str(getattr(cleanup_mask, "mask_method", "") or "").lower()
    source = str(getattr(cleanup_mask, "mask_source", "") or "").lower()
    return "bounded_connected_component_partition" in method or "partitioned_cleanup_component" in source


def _normalise_cleanup_visual_scope(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return LEGACY_CLEANUP_VISUAL_SCOPE_MAP.get(text, text)


def _partition_visual_scope(value: Any) -> str:
    visual_scope = _normalise_cleanup_visual_scope(value)
    if visual_scope == "source_glyph_union":
        return "source_glyph_union_partition"
    return visual_scope


def _unique_texts(values: Sequence[Any] | None) -> list[str]:
    output: list[str] = []
    for value in values or []:
        text = str(value or "").strip()
        if text and text not in output:
            output.append(text)
    return output


def _safe_reason_token(value: Any) -> str:
    text = str(value or "").strip().lower()
    output = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in text)
    return output.strip("_") or "unknown"


def _mask_protected_conflict_reason(cleanup_mask: CleanupMask) -> str:
    combined = " ".join(
        str(value or "")
        for value in (
            cleanup_mask.mask_method,
            cleanup_mask.mask_source,
            cleanup_mask.rejection_reason,
            cleanup_mask.protection_reason,
        )
    ).lower()
    protected_markers = (
        ("preserve", "mask_protected_preserve"),
        ("sfx", "mask_protected_sfx"),
        ("decorative", "mask_protected_decorative"),
        ("non_translation_art", "mask_protected_non_translation_art"),
        ("art_only", "mask_protected_art_only"),
    )
    for marker, reason in protected_markers:
        if marker in combined:
            return reason
    return ""


def _source_image_array(*, source_image: Any | None, source_image_path: str | None) -> tuple[Any | None, str]:
    if np is None:
        return None, "numpy_unavailable"
    if source_image is not None:
        try:
            if hasattr(source_image, "convert"):
                return np.asarray(source_image.convert("RGB")), ""
            arr = np.asarray(source_image)
            if arr.ndim == 2:
                return arr, ""
            if arr.ndim == 3:
                return arr[:, :, :3], ""
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"
    if source_image_path:
        if Image is None:
            return None, "pillow_unavailable"
        try:
            with Image.open(source_image_path) as img:
                return np.asarray(img.convert("RGB")), ""
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"
    return None, "source_image_path_missing"


def _flat_background_pre_execution_rejection(
    *,
    cleanup_mask: CleanupMask,
    source_np: Any | None,
    source_error: str,
) -> tuple[str, dict[str, Any]]:
    if source_np is None:
        return "pilot_source_image_evidence_unavailable", {"source_image_error": source_error}
    metrics = _background_flatness_metrics(source_np, cleanup_mask.allowed_area)
    if metrics.get("error"):
        return "pilot_source_image_evidence_unavailable", metrics

    allowed_area = int(metrics.get("allowed_area") or 0)
    edge_density = float(metrics.get("edge_density") or 0.0)
    luma_span = float(metrics.get("luma_span") or 0.0)
    luma_std = float(metrics.get("luma_std") or 0.0)
    page_ratio = float(metrics.get("allowed_page_ratio") or 0.0)

    broad_textured = allowed_area > 60_000 and edge_density >= 0.18 and luma_span >= 160.0
    page_broad_textured = page_ratio > 0.025 and edge_density >= 0.20 and luma_span >= 180.0
    very_broad_variable = allowed_area > 120_000 and (edge_density >= 0.12 or luma_std >= 45.0)
    if broad_textured or page_broad_textured or very_broad_variable:
        return "pilot_background_not_flat_or_art_risk", metrics
    return "", metrics


def _background_flatness_metrics(source_np: Any, allowed_area: Sequence[int] | None) -> dict[str, Any]:
    if np is None:
        return {"error": "numpy_unavailable"}
    allowed = _valid_bbox(allowed_area)
    if allowed is None:
        return {"error": "allowed_area_missing_or_invalid"}
    arr = np.asarray(source_np)
    if arr.ndim == 3:
        gray = _gray(arr[:, :, :3])
    elif arr.ndim == 2:
        gray = arr.astype(np.uint8)
    else:
        return {"error": "unsupported_source_image_shape"}
    height, width = gray.shape[:2]
    x0, y0, x1, y1 = allowed
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        return {"error": "allowed_area_outside_source_image"}
    crop = gray[y0:y1, x0:x1]
    if crop.size <= 0:
        return {"error": "allowed_area_empty_crop"}
    if cv2 is not None:
        edges = cv2.Canny(crop, 48, 128)
        edge_density = float(np.count_nonzero(edges)) / max(1, int(edges.size))
    else:
        gy, gx = np.gradient(crop.astype(np.float32))
        edge_density = float(np.count_nonzero((np.abs(gx) + np.abs(gy)) > 32.0)) / max(1, int(crop.size))
    p10 = float(np.percentile(crop, 10))
    p90 = float(np.percentile(crop, 90))
    allowed_pixels = int(crop.size)
    page_area = max(1, int(width) * int(height))
    return {
        "allowed_area": allowed_pixels,
        "allowed_page_ratio": round(float(allowed_pixels) / float(page_area), 6),
        "luma_mean": round(float(np.mean(crop)), 3),
        "luma_std": round(float(np.std(crop)), 3),
        "luma_p10": round(p10, 3),
        "luma_p90": round(p90, 3),
        "luma_span": round(p90 - p10, 3),
        "edge_density": round(edge_density, 6),
    }


def _source_residual_mask(source_gray: Any, cleaned_gray: Any, foreground: Any) -> Any:
    diff = np.abs(source_gray.astype(np.int16) - cleaned_gray.astype(np.int16))
    source_dark = source_gray < 210
    cleaned_still_dark = cleaned_gray < 230
    source_light = source_gray > 190
    cleaned_still_light = cleaned_gray > 170
    unchanged = diff < 45
    return (foreground > 0) & unchanged & ((source_dark & cleaned_still_dark) | (source_light & cleaned_still_light))


def _dark_source_residual_mask(source_gray: Any, cleaned_gray: Any, foreground: Any) -> Any:
    source_dark = source_gray < 210
    cleaned_still_dark = cleaned_gray < 230
    return (foreground > 0) & source_dark & cleaned_still_dark


def _source_erasure_box_residual_mask(source_gray: Any, cleaned_gray: Any, bbox: Sequence[int]) -> Any:
    residual = np.zeros(source_gray.shape[:2], dtype=bool)
    box = _valid_bbox(bbox)
    if box is None:
        return residual
    height, width = source_gray.shape[:2]
    x0, y0, x1, y1 = box
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        return residual
    source_crop = source_gray[y0:y1, x0:x1]
    cleaned_crop = cleaned_gray[y0:y1, x0:x1]
    source_dark = source_crop < 135
    cleaned_still_dark = cleaned_crop < 150
    unchanged = np.abs(cleaned_crop.astype(np.int16) - source_crop.astype(np.int16)) < 35
    residual[y0:y1, x0:x1] = source_dark & cleaned_still_dark & unchanged
    return residual


def _write_residual_source_mask(
    *,
    cleanup_result: CleanupResult,
    residual_mask: Any,
) -> tuple[str, str]:
    if Image is None or np is None:
        return "", "pillow_or_numpy_unavailable"
    if residual_mask is None:
        return "", "residual_mask_missing"
    base_ref = cleanup_result.mask_ref or cleanup_result.diff_ref or cleanup_result.cleaned_image_ref
    if not base_ref:
        return "", "cleanup_artifact_ref_unavailable"
    try:
        artifact_dir = os.path.dirname(str(base_ref))
        if not artifact_dir or not os.path.isdir(artifact_dir):
            return "", "cleanup_artifact_dir_unavailable"
        filename = f"{_safe_id(cleanup_result.cleanup_result_id)}_residual_source_mask.png"
        path = os.path.join(artifact_dir, filename)
        arr = (np.asarray(residual_mask) > 0).astype(np.uint8) * 255
        Image.fromarray(arr, mode="L").save(path)
        return path, ""
    except Exception as exc:
        return "", f"{type(exc).__name__}: {exc}"


def _gray(image_np: Any) -> Any:
    if image_np.ndim == 2:
        return image_np
    return (
        0.299 * image_np[:, :, 0]
        + 0.587 * image_np[:, :, 1]
        + 0.114 * image_np[:, :, 2]
    ).astype(np.uint8)


def _binary_mask(value: Any, shape: tuple[int, int]) -> Any:
    if np is None or value is None:
        return None
    arr = np.asarray(value)
    if arr.ndim == 3:
        arr = np.any(arr > 0, axis=2)
    elif arr.ndim != 2:
        return None
    output = np.zeros(shape, dtype=np.uint8)
    height = min(shape[0], arr.shape[0])
    width = min(shape[1], arr.shape[1])
    if height <= 0 or width <= 0:
        return output
    output[:height, :width] = (arr[:height, :width] > 0).astype(np.uint8)
    return output


def _bbox_to_mask(bbox: Sequence[int] | None, shape: tuple[int, int]) -> Any:
    box = _valid_bbox(bbox)
    if np is None or box is None:
        return None
    output = np.zeros(shape, dtype=np.uint8)
    x0, y0, x1, y1 = box
    x0 = max(0, min(shape[1], x0))
    x1 = max(0, min(shape[1], x1))
    y0 = max(0, min(shape[0], y0))
    y1 = max(0, min(shape[0], y1))
    if x1 > x0 and y1 > y0:
        output[y0:y1, x0:x1] = 1
    return output


def _mask_bbox(mask: Any) -> list[int] | None:
    if np is None or mask is None:
        return None
    arr = np.asarray(mask)
    if arr.ndim != 2:
        return None
    ys, xs = np.nonzero(arr > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _extract_jobs(value: Any) -> list[CleanupJob]:
    if value is None:
        return []
    if hasattr(value, "jobs"):
        return [item for item in getattr(value, "jobs") or [] if isinstance(item, CleanupJob)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [item for item in value if isinstance(item, CleanupJob)]
    return []


def _extract_masks(value: Any) -> list[CleanupMask]:
    if value is None:
        return []
    if hasattr(value, "masks"):
        return [item for item in getattr(value, "masks") or [] if isinstance(item, CleanupMask)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [item for item in value if isinstance(item, CleanupMask)]
    return []


def _extract_plans(value: Any) -> list[CleanupPlan]:
    if value is None:
        return []
    if hasattr(value, "plans"):
        return [item for item in getattr(value, "plans") or [] if isinstance(item, CleanupPlan)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [item for item in value if isinstance(item, CleanupPlan)]
    return []


def _job_base_record(page_id: str, job: CleanupJob) -> dict[str, Any]:
    return {
        "page_id": page_id,
        "cleanup_job_id": str(job.cleanup_job_id),
        "target_region_ids": list(job.target_region_ids or []),
        "cleanup_class": _enum_value(job.cleanup_class),
        "route_intent": str(job.route_intent or ""),
        "semantic_class": str(job.semantic_class or ""),
        "cleanup_mode": str(job.cleanup_mode or ""),
        "classification_reason": str(job.classification_reason or ""),
    }


def _render_eligibility_by_region_id(render_eligibility: Any | None) -> dict[str, Any]:
    if render_eligibility is None:
        return {}
    decisions = getattr(render_eligibility, "decisions_by_region_id", None)
    if isinstance(decisions, Mapping):
        return {str(key): value for key, value in decisions.items()}
    if isinstance(render_eligibility, Mapping):
        raw = render_eligibility.get("decisions_by_region_id")
        if isinstance(raw, Mapping):
            return {str(key): value for key, value in raw.items()}
        raw_list = render_eligibility.get("decisions")
        if isinstance(raw_list, Sequence) and not isinstance(raw_list, (str, bytes)):
            output: dict[str, Any] = {}
            for item in raw_list:
                rid = str(_decision_value(item, "region_id") or "")
                if rid:
                    output[rid] = item
            return output
    return {}


def _source_grounding_suppressed_decision(
    job: CleanupJob,
    decisions_by_region_id: Mapping[str, Any],
) -> Any | None:
    for region_id in job.target_region_ids or []:
        decision = decisions_by_region_id.get(str(region_id or ""))
        if str(_decision_value(decision, "status") or "") == "suppressed_source_ungrounded":
            return decision
    return None


def _decision_value(decision: Any, key: str) -> Any:
    if decision is None:
        return None
    if isinstance(decision, Mapping):
        return decision.get(key)
    value = getattr(decision, key, None)
    return getattr(value, "value", value)


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


def _union_valid_bboxes(*values: Any) -> list[int] | None:
    boxes = [box for value in values if (box := _valid_bbox(value)) is not None]
    if not boxes:
        return None
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def _bbox_area(value: Sequence[int] | None) -> int:
    box = _valid_bbox(value)
    if box is None:
        return 0
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _growth_ratio(erase_pixels: int, foreground_pixels: int) -> float:
    if foreground_pixels <= 0:
        return float("inf")
    return round(float(erase_pixels) / float(foreground_pixels), 4)


def _safe_id(value: Any) -> str:
    text = str(value or "none")
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_") or "none"


def _enum_value(value: Any) -> str:
    return value.value if isinstance(value, Enum) else str(value or "")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_audit_dict"):
        return value.to_audit_dict()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(val) for key, val in value.items() if not _raw_payload_key(key)}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if np is not None and isinstance(value, np.generic):
        return value.item()
    return str(value)


def _raw_payload_key(key: Any) -> bool:
    return str(key).lower() in {
        "image",
        "cleaned_image",
        "cleaned_crop",
        "mask",
        "foreground_mask",
        "erase_mask",
        "array",
        "bitmap",
        "crop",
    }
