"""Cleanup-owned dense mask construction for the caption-flat pilot.

Phase 3C keeps SourceGlyphMask and Phase 2 CleanupMask records as seed
evidence.  This module builds denser CleanupMask contracts for the single
caption_flat_background pilot class only; renderer behavior is unchanged unless
the later proof-gated pilot explicitly commits a passed result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

try:  # pragma: no cover - exercised in the target conda env
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

from app.pipeline.cleanup_contracts import CleanupClass, CleanupJob, CleanupMask
from app.pipeline.cleanup_masks import CleanupMaskBuildResult


CLEANUP_DENSE_MASK_CONTRACT_VERSION = "cleanup_dense_masks_phase3c_caption_flat_background"

MAX_DENSE_ERASE_PIXELS = 8_000
MAX_DENSE_ERASE_BBOX_AREA = 35_000
MAX_DENSE_ERASE_ALLOWED_RATIO = 0.28
MAX_DENSE_ALLOWED_PAGE_RATIO = 0.20
MAX_DENSE_GROWTH_OVER_SEED = 14.0
MAX_DENSE_ALLOWED_FLAT_EDGE_DENSITY = 0.22
MAX_DENSE_ALLOWED_FLAT_STD = 64.0
MAX_DENSE_ALLOWED_FLAT_LUMA_SPAN = 190
MAX_DENSE_ALLOWED_BORDER_TOUCH_RATIO = 0.55


@dataclass(frozen=True)
class CleanupDenseMaskBuildResult:
    page_id: str
    version: str = CLEANUP_DENSE_MASK_CONTRACT_VERSION
    masks: list[CleanupMask] = field(default_factory=list)
    masks_by_job_id: dict[str, CleanupMask] = field(default_factory=dict)
    rejected_records: list[dict[str, Any]] = field(default_factory=list)
    protected_records: list[dict[str, Any]] = field(default_factory=list)
    skipped_records: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    source_image_loaded: bool = False

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "version": self.version,
            "renderer_consumed": False,
            "summary": {
                "mask_count": len(self.masks),
                "rejected_count": len(self.rejected_records),
                "protected_count": len(self.protected_records),
                "skipped_count": len(self.skipped_records),
                "error_count": len(self.errors),
                "source_image_loaded": bool(self.source_image_loaded),
            },
            "masks": [mask.to_audit_dict() for mask in self.masks],
            "masks_by_job_id": {
                job_id: mask.cleanup_mask_id
                for job_id, mask in sorted(self.masks_by_job_id.items())
            },
            "rejected_records": list(self.rejected_records),
            "protected_records": list(self.protected_records),
            "skipped_records": list(self.skipped_records),
            "errors": list(self.errors),
        }


def build_cleanup_dense_masks(
    *,
    page_id: str,
    cleanup_jobs: Sequence[CleanupJob] | Any,
    cleanup_masks: CleanupMaskBuildResult | Sequence[CleanupMask] | Mapping[str, Any] | Any,
    source_glyph_masks: Any = None,
    source_image_path: str | Path | None = None,
    source_image: Image.Image | np.ndarray | None = None,
    image_size: tuple[int, int] | None = None,
) -> CleanupDenseMaskBuildResult:
    """Build dense CleanupMask records for strict caption-flat candidates."""

    jobs = _coerce_jobs(cleanup_jobs)
    phase2_masks = _coerce_masks(cleanup_masks)
    mask_by_job = {mask.cleanup_job_id: mask for mask in phase2_masks}
    source_records = _source_records_by_region(source_glyph_masks)
    src_image, loaded = _load_source_image(source_image=source_image, source_image_path=source_image_path)

    masks: list[CleanupMask] = []
    masks_by_job_id: dict[str, CleanupMask] = {}
    rejected: list[dict[str, Any]] = []
    protected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    if src_image is None:
        for job in jobs:
            if _job_cleanup_class(job) == CleanupClass.CAPTION_FLAT_BACKGROUND.value:
                rejected.append(
                    _record(job, "source_image_evidence_unavailable", source_image_loaded=False)
                )
        return CleanupDenseMaskBuildResult(
            page_id=page_id,
            masks=masks,
            masks_by_job_id=masks_by_job_id,
            rejected_records=rejected,
            protected_records=protected,
            skipped_records=skipped,
            errors=errors,
            source_image_loaded=False,
        )

    width, height = src_image.size
    page_area = max(1, width * height)

    for job in jobs:
        try:
            job_class = _job_cleanup_class(job)
            if job_class != CleanupClass.CAPTION_FLAT_BACKGROUND.value:
                if _is_protected_job(job):
                    protected.append(_record(job, "non_caption_flat_background_protected"))
                else:
                    skipped.append(_record(job, "non_caption_flat_background_not_phase3c_pilot"))
                continue

            if _job_has_artifact_or_protection(job):
                protected.append(_record(job, "artifact_or_protected_route_blocks_dense_mask"))
                continue

            seed_mask = mask_by_job.get(job.cleanup_job_id)
            if seed_mask is None:
                rejected.append(_record(job, "phase2_cleanup_mask_missing"))
                continue
            if _mask_has_artifact_or_protection(seed_mask):
                protected.append(
                    _record(job, "phase2_mask_artifact_or_protection_blocks_dense_mask", cleanup_mask_id=seed_mask.cleanup_mask_id)
                )
                continue
            if _visual_scope(seed_mask, job) != "source_glyph_local":
                rejected.append(
                    _record(job, "dense_mask_requires_source_glyph_local_scope", cleanup_mask_id=seed_mask.cleanup_mask_id)
                )
                continue

            allowed = _bbox_tuple(seed_mask.allowed_area or job.allowed_area)
            if not _valid_bbox(allowed, width, height):
                rejected.append(
                    _record(job, "allowed_area_missing_or_invalid", cleanup_mask_id=seed_mask.cleanup_mask_id)
                )
                continue
            allowed_area = _bbox_area(allowed)
            if allowed_area / page_area > MAX_DENSE_ALLOWED_PAGE_RATIO:
                rejected.append(
                    _record(
                        job,
                        "allowed_area_too_broad_for_dense_caption_flat",
                        cleanup_mask_id=seed_mask.cleanup_mask_id,
                        allowed_area=allowed_area,
                        allowed_page_ratio=allowed_area / page_area,
                    )
                )
                continue

            foreground_seed = _mask_array(seed_mask.foreground_mask, (height, width))
            erase_seed = _mask_array(seed_mask.erase_mask, (height, width))
            if foreground_seed is None:
                foreground_seed = _mask_array(_source_foreground_record(job, source_records), (height, width))
            if foreground_seed is None:
                rejected.append(
                    _record(job, "foreground_seed_missing", cleanup_mask_id=seed_mask.cleanup_mask_id)
                )
                continue
            if erase_seed is None:
                erase_seed = foreground_seed

            allowed_mask = _bbox_mask((height, width), allowed)
            foreground_seed = (foreground_seed & allowed_mask).astype(np.uint8)
            erase_seed = (erase_seed & allowed_mask).astype(np.uint8)
            if int(foreground_seed.sum()) <= 0:
                rejected.append(
                    _record(job, "foreground_seed_empty_after_allowed_clip", cleanup_mask_id=seed_mask.cleanup_mask_id)
                )
                continue

            dense_foreground, dense_erase, dense_metrics = _build_dense_arrays(
                src_image=src_image,
                allowed=allowed,
                foreground_seed=foreground_seed,
                erase_seed=erase_seed,
                page_area=page_area,
            )
            rejection_reason = dense_metrics.get("rejection_reason")
            if rejection_reason:
                rejected.append(
                    _record(
                        job,
                        str(rejection_reason),
                        cleanup_mask_id=seed_mask.cleanup_mask_id,
                        metrics=_audit_metrics(dense_metrics),
                    )
                )
                continue

            dense_mask = _make_cleanup_mask(
                page_id=page_id,
                job=job,
                seed_mask=seed_mask,
                dense_foreground=dense_foreground,
                dense_erase=dense_erase,
                allowed=allowed,
                metrics=dense_metrics,
            )
            masks.append(dense_mask)
            masks_by_job_id[job.cleanup_job_id] = dense_mask
        except Exception as exc:  # pragma: no cover - defensive audit path
            errors.append(_record(job, "dense_mask_builder_error", error=str(exc)))

    return CleanupDenseMaskBuildResult(
        page_id=page_id,
        masks=masks,
        masks_by_job_id=masks_by_job_id,
        rejected_records=rejected,
        protected_records=protected,
        skipped_records=skipped,
        errors=errors,
        source_image_loaded=loaded,
    )


def _coerce_jobs(cleanup_jobs: Sequence[CleanupJob] | Any) -> list[CleanupJob]:
    if hasattr(cleanup_jobs, "jobs"):
        return [job for job in getattr(cleanup_jobs, "jobs", []) if isinstance(job, CleanupJob)]
    return [job for job in cleanup_jobs or [] if isinstance(job, CleanupJob)]


def _coerce_masks(cleanup_masks: CleanupMaskBuildResult | Sequence[CleanupMask] | Mapping[str, Any] | Any) -> list[CleanupMask]:
    if isinstance(cleanup_masks, CleanupMaskBuildResult):
        return list(cleanup_masks.masks)
    if hasattr(cleanup_masks, "masks"):
        return [mask for mask in getattr(cleanup_masks, "masks", []) if isinstance(mask, CleanupMask)]
    if isinstance(cleanup_masks, Mapping):
        masks = cleanup_masks.get("masks")
        if isinstance(masks, Sequence):
            return [mask for mask in masks if isinstance(mask, CleanupMask)]
    return [mask for mask in cleanup_masks or [] if isinstance(mask, CleanupMask)]


def _load_source_image(
    *,
    source_image: Image.Image | np.ndarray | None,
    source_image_path: str | Path | None,
) -> tuple[Image.Image | None, bool]:
    if isinstance(source_image, Image.Image):
        return source_image.convert("RGB"), True
    if isinstance(source_image, np.ndarray):
        return Image.fromarray(source_image.astype(np.uint8)).convert("RGB"), True
    if source_image_path:
        try:
            return Image.open(source_image_path).convert("RGB"), True
        except Exception:
            return None, False
    return None, False


def _source_records_by_region(source_glyph_masks: Any) -> dict[str, Any]:
    records: list[Any] = []
    if hasattr(source_glyph_masks, "masks"):
        records = list(getattr(source_glyph_masks, "masks", []) or [])
    elif isinstance(source_glyph_masks, Mapping):
        value = source_glyph_masks.get("source_glyph_masks") or source_glyph_masks.get("masks") or []
        if isinstance(value, Sequence):
            records = list(value)
    by_region: dict[str, Any] = {}
    for record in records:
        rid = _get(record, "region_id") or _get(record, "target_region_id")
        if rid and rid not in by_region:
            by_region[str(rid)] = record
    return by_region


def _source_foreground_record(job: CleanupJob, records: Mapping[str, Any]) -> Any:
    record = records.get(_job_region_id(job))
    if record is None:
        return None
    foreground = _get(record, "foreground_mask")
    if foreground is not None:
        return foreground
    return _get(record, "mask")


def _build_dense_arrays(
    *,
    src_image: Image.Image,
    allowed: tuple[int, int, int, int],
    foreground_seed: np.ndarray,
    erase_seed: np.ndarray,
    page_area: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    image = np.asarray(src_image.convert("RGB"))
    height, width = foreground_seed.shape
    gray = _to_gray(image)
    x1, y1, x2, y2 = allowed
    allowed_area = max(1, (x2 - x1) * (y2 - y1))
    allowed_gray = gray[y1:y2, x1:x2]
    seed_union = ((foreground_seed > 0) | (erase_seed > 0)).astype(np.uint8)
    seed_ignore = _dilate_for_metric_ignore(seed_union)
    flat_metrics = _flatness_metrics(allowed_gray, ignore_mask=seed_ignore[y1:y2, x1:x2])
    if _flat_rejection(flat_metrics, allowed_area, page_area):
        return (
            np.zeros((height, width), dtype=np.uint8),
            np.zeros((height, width), dtype=np.uint8),
            {**flat_metrics, "rejection_reason": "dense_background_not_flat_or_art_risk"},
        )

    seed_bbox = _mask_bbox(seed_union)
    if seed_bbox is None:
        return (
            np.zeros((height, width), dtype=np.uint8),
            np.zeros((height, width), dtype=np.uint8),
            {**flat_metrics, "rejection_reason": "dense_seed_empty"},
        )
    focus = _expand_bbox(seed_bbox, allowed, pad_x=max(14, int((seed_bbox[2] - seed_bbox[0]) * 0.55)), pad_y=max(10, int((seed_bbox[3] - seed_bbox[1]) * 0.55)))
    fx1, fy1, fx2, fy2 = focus
    focus_gray = gray[fy1:fy2, fx1:fx2]
    focus_seed = seed_union[fy1:fy2, fx1:fx2]

    candidate_focus = _candidate_text_pixels(focus_gray, focus_seed)
    kept_focus = _keep_seed_connected_components(candidate_focus, focus_seed)
    dense_foreground = np.zeros((height, width), dtype=np.uint8)
    dense_foreground[fy1:fy2, fx1:fx2] = kept_focus
    dense_foreground = (dense_foreground & _bbox_mask((height, width), allowed)).astype(np.uint8)

    if cv2 is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dense_erase = cv2.dilate(dense_foreground, kernel, iterations=1).astype(np.uint8)
        dense_erase = cv2.morphologyEx(dense_erase, cv2.MORPH_CLOSE, kernel, iterations=1).astype(np.uint8)
    else:  # pragma: no cover
        dense_erase = _binary_dilate(dense_foreground, radius=2)
    dense_erase = ((dense_erase > 0) | (erase_seed > 0)).astype(np.uint8)
    dense_erase = (dense_erase & _bbox_mask((height, width), allowed)).astype(np.uint8)

    dense_metrics = _dense_validation_metrics(
        dense_foreground=dense_foreground,
        dense_erase=dense_erase,
        foreground_seed=foreground_seed,
        allowed_area=allowed_area,
        page_area=page_area,
        flat_metrics=flat_metrics,
        focus_bbox=focus,
    )
    return dense_foreground, dense_erase, dense_metrics


def _candidate_text_pixels(focus_gray: np.ndarray, focus_seed: np.ndarray) -> np.ndarray:
    if focus_gray.size == 0:
        return np.zeros_like(focus_seed, dtype=np.uint8)
    p20 = float(np.percentile(focus_gray, 20))
    p80 = float(np.percentile(focus_gray, 80))
    seed_values = focus_gray[focus_seed > 0]
    seed_median = float(np.median(seed_values)) if seed_values.size else float(np.median(focus_gray))

    dark = focus_gray <= max(70.0, min(p20 + 18.0, seed_median + 35.0))
    bright = focus_gray >= min(235.0, max(p80 - 12.0, seed_median + 45.0))
    if cv2 is not None:
        blur = cv2.GaussianBlur(focus_gray, (0, 0), 1.2)
        edge = cv2.Canny(focus_gray.astype(np.uint8), 35, 120) > 0
        edge = cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1) > 0
    else:  # pragma: no cover
        blur = _mean_blur(focus_gray, radius=2)
        gy, gx = np.gradient(focus_gray.astype(np.float32))
        edge = np.hypot(gx, gy) > 22
    local_contrast = np.abs(focus_gray.astype(np.float32) - blur.astype(np.float32)) >= 16
    candidate = (edge & (dark | bright | local_contrast)) | (focus_seed > 0)
    if cv2 is not None:
        candidate_u8 = candidate.astype(np.uint8)
        candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)
        candidate_u8 = cv2.dilate(candidate_u8, np.ones((2, 2), dtype=np.uint8), iterations=1)
        return (candidate_u8 > 0).astype(np.uint8)
    return candidate.astype(np.uint8)


def _keep_seed_connected_components(candidate: np.ndarray, focus_seed: np.ndarray) -> np.ndarray:
    if candidate.size == 0:
        return np.zeros_like(candidate, dtype=np.uint8)
    if cv2 is None:  # pragma: no cover
        return (candidate | focus_seed).astype(np.uint8)
    seed_near = cv2.dilate((focus_seed > 0).astype(np.uint8), np.ones((15, 15), dtype=np.uint8), iterations=1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats((candidate > 0).astype(np.uint8), 8)
    kept = np.zeros_like(candidate, dtype=np.uint8)
    h, w = candidate.shape
    max_component_area = max(450, int(candidate.size * 0.42))
    for idx in range(1, count):
        x, y, bw, bh, area = [int(v) for v in stats[idx]]
        if area <= 0 or area > max_component_area:
            continue
        component = labels == idx
        seed_overlap = int(np.count_nonzero(component & (seed_near > 0)))
        if seed_overlap <= 0:
            continue
        touches = int(x <= 0) + int(y <= 0) + int(x + bw >= w) + int(y + bh >= h)
        touch_ratio = touches / 4.0
        if touch_ratio > MAX_DENSE_ALLOWED_BORDER_TOUCH_RATIO and area > 120:
            continue
        kept[component] = 1
    kept = (kept | (focus_seed > 0)).astype(np.uint8)
    if cv2 is not None:
        kept = cv2.morphologyEx(kept, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return kept.astype(np.uint8)


def _dense_validation_metrics(
    *,
    dense_foreground: np.ndarray,
    dense_erase: np.ndarray,
    foreground_seed: np.ndarray,
    allowed_area: int,
    page_area: int,
    flat_metrics: Mapping[str, Any],
    focus_bbox: tuple[int, int, int, int],
) -> dict[str, Any]:
    foreground_pixels = int(np.count_nonzero(dense_foreground))
    erase_pixels = int(np.count_nonzero(dense_erase))
    seed_pixels = max(1, int(np.count_nonzero(foreground_seed)))
    erase_bbox = _mask_bbox(dense_erase)
    erase_bbox_area = _bbox_area(erase_bbox) if erase_bbox else 0
    seed_coverage = int(np.count_nonzero((dense_erase > 0) & (foreground_seed > 0))) / float(seed_pixels)
    growth_over_seed = erase_pixels / float(seed_pixels)
    metrics = {
        **dict(flat_metrics),
        "foreground_pixels": foreground_pixels,
        "erase_pixels": erase_pixels,
        "erase_bbox": list(erase_bbox) if erase_bbox else None,
        "erase_bbox_area": erase_bbox_area,
        "allowed_area": allowed_area,
        "allowed_page_ratio": allowed_area / float(max(1, page_area)),
        "erase_allowed_ratio": erase_pixels / float(max(1, allowed_area)),
        "seed_coverage_ratio": seed_coverage,
        "growth_over_seed": growth_over_seed,
        "focus_bbox": list(focus_bbox),
    }
    if foreground_pixels <= 0:
        metrics["rejection_reason"] = "dense_foreground_empty"
    elif erase_pixels <= 0:
        metrics["rejection_reason"] = "dense_erase_empty"
    elif seed_coverage < 0.98:
        metrics["rejection_reason"] = "dense_seed_coverage_too_low"
    elif erase_pixels > MAX_DENSE_ERASE_PIXELS:
        metrics["rejection_reason"] = "dense_erase_pixels_too_broad"
    elif erase_bbox_area > MAX_DENSE_ERASE_BBOX_AREA:
        metrics["rejection_reason"] = "dense_erase_bbox_too_broad"
    elif metrics["erase_allowed_ratio"] > MAX_DENSE_ERASE_ALLOWED_RATIO:
        metrics["rejection_reason"] = "dense_erase_allowed_ratio_too_broad"
    elif growth_over_seed > MAX_DENSE_GROWTH_OVER_SEED:
        metrics["rejection_reason"] = "dense_growth_over_seed_too_large"
    return metrics


def _make_cleanup_mask(
    *,
    page_id: str,
    job: CleanupJob,
    seed_mask: CleanupMask,
    dense_foreground: np.ndarray,
    dense_erase: np.ndarray,
    allowed: tuple[int, int, int, int],
    metrics: Mapping[str, Any],
) -> CleanupMask:
    foreground_bbox = _mask_bbox(dense_foreground) or allowed
    erase_bbox = _mask_bbox(dense_erase) or foreground_bbox
    foreground_pixels = int(np.count_nonzero(dense_foreground))
    erase_pixels = int(np.count_nonzero(dense_erase))
    growth_ratio = erase_pixels / float(max(1, foreground_pixels))
    return CleanupMask(
        cleanup_mask_id=f"cdmask_{_safe_id(page_id)}_{_safe_id(job.cleanup_job_id)}",
        cleanup_job_id=job.cleanup_job_id,
        foreground_mask_source_id=seed_mask.cleanup_mask_id,
        foreground_mask_bbox=foreground_bbox,
        foreground_mask_pixels=foreground_pixels,
        erase_mask_bbox=erase_bbox,
        erase_mask_pixels=erase_pixels,
        allowed_area=allowed,
        growth_ratio=float(growth_ratio),
        mask_source="cleanup_dense_mask_from_source_glyph_seed",
        mask_method="caption_flat_background_dense_local_contrast_components",
        mask_contract_exception_reason=seed_mask.mask_contract_exception_reason,
        artifact_risk="",
        visual_scope="source_glyph_local",
        foreground_mask=dense_foreground.astype(np.uint8),
        erase_mask=dense_erase.astype(np.uint8),
    )


def _flatness_metrics(gray: np.ndarray, *, ignore_mask: np.ndarray | None = None) -> dict[str, Any]:
    if gray.size == 0:
        return {
            "allowed_luma_std": 0.0,
            "allowed_luma_span": 0,
            "allowed_edge_density": 0.0,
            "allowed_texture_density": 0.0,
        }
    metric_pixels = gray
    valid_mask = None
    if ignore_mask is not None and ignore_mask.shape == gray.shape:
        valid_mask = ignore_mask <= 0
        if int(np.count_nonzero(valid_mask)) >= max(64, int(gray.size * 0.30)):
            metric_pixels = gray[valid_mask]
        else:
            valid_mask = None
    luma_std = float(np.std(metric_pixels))
    luma_span = int(np.percentile(metric_pixels, 95) - np.percentile(metric_pixels, 5))
    if cv2 is not None:
        edges = cv2.Canny(gray.astype(np.uint8), 45, 140) > 0
        blur = cv2.GaussianBlur(gray, (0, 0), 1.8)
    else:  # pragma: no cover
        gy, gx = np.gradient(gray.astype(np.float32))
        edges = np.hypot(gx, gy) > 28
        blur = _mean_blur(gray, radius=2)
    texture = np.abs(gray.astype(np.float32) - blur.astype(np.float32)) > 22
    if valid_mask is not None:
        edge_density = float(np.mean(edges[valid_mask])) if np.any(valid_mask) else float(np.mean(edges))
        texture_density = float(np.mean(texture[valid_mask])) if np.any(valid_mask) else float(np.mean(texture))
    else:
        edge_density = float(np.mean(edges))
        texture_density = float(np.mean(texture))
    return {
        "allowed_luma_std": round(luma_std, 4),
        "allowed_luma_span": luma_span,
        "allowed_edge_density": round(edge_density, 6),
        "allowed_texture_density": round(texture_density, 6),
        "flatness_ignored_seed_pixels": int(np.count_nonzero(ignore_mask)) if ignore_mask is not None else 0,
    }


def _flat_rejection(metrics: Mapping[str, Any], allowed_area: int, page_area: int) -> bool:
    edge_density = float(metrics.get("allowed_edge_density") or 0.0)
    texture_density = float(metrics.get("allowed_texture_density") or 0.0)
    luma_std = float(metrics.get("allowed_luma_std") or 0.0)
    luma_span = float(metrics.get("allowed_luma_span") or 0.0)
    allowed_page_ratio = allowed_area / float(max(1, page_area))
    if edge_density > MAX_DENSE_ALLOWED_FLAT_EDGE_DENSITY:
        return True
    if luma_std > MAX_DENSE_ALLOWED_FLAT_STD and texture_density > 0.30:
        return True
    if luma_span > MAX_DENSE_ALLOWED_FLAT_LUMA_SPAN and allowed_page_ratio > 0.020:
        return True
    if allowed_area > 60_000 and (edge_density > 0.16 or texture_density > 0.34):
        return True
    return False


def _to_gray(image: np.ndarray) -> np.ndarray:
    rgb = image.astype(np.float32)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    return np.clip(gray, 0, 255).astype(np.uint8)


def _mask_array(value: Any, shape: tuple[int, int]) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.shape != shape:
        return None
    return (arr > 0).astype(np.uint8)


def _bbox_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    mask[y1:y2, x1:x2] = 1
    return mask


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    allowed: tuple[int, int, int, int],
    *,
    pad_x: int,
    pad_y: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    ax1, ay1, ax2, ay2 = allowed
    return max(ax1, x1 - pad_x), max(ay1, y1 - pad_y), min(ax2, x2 + pad_x), min(ay2, y2 + pad_y)


def _bbox_tuple(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in value]
    except Exception:
        return None
    return x1, y1, x2, y2


def _valid_bbox(bbox: tuple[int, int, int, int] | None, width: int, height: int) -> bool:
    if bbox is None:
        return False
    x1, y1, x2, y2 = bbox
    return 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height


def _bbox_area(bbox: tuple[int, int, int, int] | None) -> int:
    if bbox is None:
        return 0
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def _binary_dilate(mask: np.ndarray, *, radius: int) -> np.ndarray:  # pragma: no cover
    padded = np.pad(mask, radius, mode="constant")
    out = np.zeros_like(mask, dtype=np.uint8)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            out |= padded[radius + dy: radius + dy + mask.shape[0], radius + dx: radius + dx + mask.shape[1]]
    return out.astype(np.uint8)


def _dilate_for_metric_ignore(mask: np.ndarray) -> np.ndarray:
    if cv2 is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        return cv2.dilate((mask > 0).astype(np.uint8), kernel, iterations=1).astype(np.uint8)
    return _binary_dilate((mask > 0).astype(np.uint8), radius=5)


def _mean_blur(gray: np.ndarray, *, radius: int) -> np.ndarray:  # pragma: no cover
    padded = np.pad(gray.astype(np.float32), radius, mode="edge")
    out = np.zeros_like(gray, dtype=np.float32)
    count = 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            out += padded[radius + dy: radius + dy + gray.shape[0], radius + dx: radius + dx + gray.shape[1]]
            count += 1
    return (out / max(1, count)).astype(np.float32)


def _job_cleanup_class(job: CleanupJob) -> str:
    value = job.cleanup_class.value if hasattr(job.cleanup_class, "value") else str(job.cleanup_class)
    return value


def _visual_scope(mask: CleanupMask, job: CleanupJob) -> str:
    if mask.visual_scope:
        return str(mask.visual_scope)
    if "source_glyph_local" in job.classification_reason:
        return "source_glyph_local"
    return ""


def _job_has_artifact_or_protection(job: CleanupJob) -> bool:
    text = " ".join(
        str(v)
        for v in [
            job.cleanup_class.value if hasattr(job.cleanup_class, "value") else job.cleanup_class,
            job.route_intent,
            job.cleanup_mode,
            job.classification_reason,
            ",".join(getattr(job, "artifact_risk", []) or []),
        ]
    ).lower()
    protected_terms = (
        "side_caption",
        "speech",
        "sfx",
        "decorative",
        "preserve",
        "artifact_risk",
        "art_risk",
        "art_entangled",
        "screentone",
        "caption_dark",
        "dark_or_screentone",
        "background_art",
        "ambiguous",
    )
    return any(term in text for term in protected_terms)


def _mask_has_artifact_or_protection(mask: CleanupMask) -> bool:
    text = " ".join(
        str(v)
        for v in [
            mask.mask_source,
            mask.mask_method,
            mask.artifact_risk,
            mask.visual_scope,
        ]
    ).lower()
    return any(
        term in text
        for term in (
            "side_caption",
            "speech",
            "sfx",
            "decorative",
            "preserve",
            "art_risk",
            "background_art",
            "screentone",
            "dark",
            "ambiguous",
        )
    )


def _is_protected_job(job: CleanupJob) -> bool:
    value = _job_cleanup_class(job)
    return value in {
        CleanupClass.PRESERVE_SFX_DECORATIVE.value,
        CleanupClass.BACKGROUND_ART_TEXT.value,
        CleanupClass.ART_ENTANGLED_AMBIGUOUS.value,
        CleanupClass.CAPTION_DARK_OR_SCREENTONE.value,
        CleanupClass.SPEECH_FLAT_BUBBLE.value,
        CleanupClass.SPEECH_COMPLEX_BUBBLE.value,
    }


def _record(job: CleanupJob, reason: str, **extra: Any) -> dict[str, Any]:
    record = {
        "cleanup_job_id": job.cleanup_job_id,
        "region_id": _job_region_id(job),
        "cleanup_class": _job_cleanup_class(job),
        "reason": reason,
    }
    record.update({key: _json_safe(value) for key, value in extra.items()})
    return record


def _audit_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _json_safe(value) for key, value in metrics.items() if key != "rejection_reason"}


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return {"omitted_raw_array": True, "shape": list(value.shape)}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return value


def _get(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(key, default)
    return getattr(record, key, default)


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))


def _job_region_id(job: CleanupJob) -> str:
    if job.target_region_ids:
        return str(job.target_region_ids[0])
    return str(job.cleanup_job_id)
