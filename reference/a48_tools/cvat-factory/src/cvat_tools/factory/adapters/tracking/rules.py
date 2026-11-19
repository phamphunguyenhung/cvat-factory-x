"""Tracking validation rules: temporal consistency, keyframe interpolation error, fragmentation, and ID-switch risk.

Never invents or changes track IDs without strong evidence; flags ambiguous cases for REVIEW.
"""
from __future__ import annotations

import math
from typing import Sequence

from cvat_tools.factory.core.models import (
    Proposal,
    QualityVerdict,
    TaskSchema,
    ValidationIssue,
    ValidationReport,
)


def _box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    iw = min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0]))
    ih = min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1]))
    if iw <= 0.0 or ih <= 0.0:
        return 0.0
    inter = iw * ih
    ua = (float(a[2]) - float(a[0])) * (float(a[3]) - float(a[1])) + (float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1])) - inter
    return inter / ua if ua > 0.0 else 0.0


def _extract_track_frames(proposal: Proposal) -> tuple[dict[int, list[float]], list[int]]:
    """Extract {frame_id: [x1,y1,x2,y2]} and sorted keyframe list from a tracking Proposal."""
    geom = proposal.geometry
    frames_raw = geom.get("frames") or {}
    track: dict[int, list[float]] = {int(k): [float(x) for x in v] for k, v in frames_raw.items()}
    kfs = [int(f) for f in (geom.get("keyframes") or ())]
    if not kfs and geom.get("shapes"):
        for s in geom["shapes"]:
            if not s.get("outside", False) and len(s.get("points") or []) == 4:
                f = int(s["frame"])
                kfs.append(f)
                track.setdefault(f, [float(x) for x in s["points"]])
    return track, sorted(set(kfs))


def validate_track_proposal(
    proposal: Proposal,
    *,
    task_schema: TaskSchema | None = None,
    interp_tol: float = 4.0,
    max_jump_ratio: float = 2.5,
    max_scale_log: float = 0.7,
) -> ValidationReport:
    """Validate a single track Proposal on schema, keyframes, interpolation error, and temporal jumps."""
    from cvat_tools.tracking.keyframes import interp_error

    issues: list[ValidationIssue] = []
    geometry_valid = True
    image_alignment_good = True
    schema_valid = True
    confidence_sufficient = True

    if proposal.shape_type != "track":
        geometry_valid = False
        issues.append(
            ValidationIssue(
                code="TRACK_INVALID_SHAPE_TYPE",
                severity="error",
                message=f"Expected track, got {proposal.shape_type!r}",
                region="geometry",
            )
        )

    if proposal.track_id is None:
        schema_valid = False
        issues.append(
            ValidationIssue(
                code="TRACK_MISSING_ID",
                severity="error",
                message="Track proposal missing track_id",
                region="identity",
            )
        )

    track, kfs = _extract_track_frames(proposal)
    if not track or not kfs:
        geometry_valid = False
        issues.append(
            ValidationIssue(
                code="TRACK_EMPTY",
                severity="error",
                message="Track has no valid frames or keyframes",
                region="geometry",
            )
        )
        return ValidationReport(
            proposal_id=proposal.proposal_id,
            verdict=QualityVerdict.REVIEW,
            geometry_valid=False,
            image_alignment_good=False,
            schema_valid=schema_valid,
            confidence_sufficient=False,
            no_regression=True,
            issues=issues,
        )

    err = interp_error(track, kfs)
    if err["max"] > interp_tol:
        image_alignment_good = False
        issues.append(
            ValidationIssue(
                code="TRACK_INTERP_ERROR_EXCEEDED",
                severity="warning",
                message=f"Max linear interp error {err['max']:.2f}px > tol={interp_tol}px at frame {err['worst_frame']}",
                region="keyframes",
                affected_points=[int(err["worst_frame"])] if err["worst_frame"] is not None else [],
                details=err,
            )
        )

    sorted_frames = sorted(track)
    for fa, fb in zip(sorted_frames, sorted_frames[1:]):
        ba, bb = track[fa], track[fb]
        if len(ba) != 4 or len(bb) != 4:
            geometry_valid = False
            continue
        dt = max(1, fb - fa)
        ca = ((ba[0] + ba[2]) / 2.0, (ba[1] + ba[3]) / 2.0)
        cb = ((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0)
        ha = max(1.0, ba[3] - ba[1])
        hb = max(1.0, bb[3] - bb[1])
        dist_norm = math.hypot(cb[0] - ca[0], cb[1] - ca[1]) / (ha * dt)
        if dist_norm > max_jump_ratio:
            image_alignment_good = False
            issues.append(
                ValidationIssue(
                    code="TRACK_TEMPORAL_JUMP",
                    severity="warning",
                    message=f"Sudden jump {dist_norm:.2f}x box_h between frames {fa}->{fb}",
                    region="temporal",
                    affected_points=[fa, fb],
                )
            )
        scale_diff = abs(math.log(max(hb, 1e-3) / ha)) / dt
        if scale_diff > max_scale_log:
            image_alignment_good = False
            issues.append(
                ValidationIssue(
                    code="TRACK_SCALE_JUMP",
                    severity="warning",
                    message=f"Sudden scale jump between frames {fa}->{fb}",
                    region="temporal",
                    affected_points=[fa, fb],
                )
            )

    if task_schema is not None and task_schema.label_by_name(proposal.label) is None:
        schema_valid = False
        issues.append(
            ValidationIssue(
                code="TRACK_UNKNOWN_LABEL",
                severity="error",
                message=f"Label {proposal.label!r} not found in task schema",
                region="schema",
            )
        )

    if not schema_valid or not geometry_valid:
        verdict = QualityVerdict.REVIEW
    elif any(i.code in ("TRACK_TEMPORAL_JUMP", "TRACK_SCALE_JUMP") for i in issues):
        verdict = QualityVerdict.REVIEW
    elif any(i.code == "TRACK_INTERP_ERROR_EXCEEDED" for i in issues):
        verdict = QualityVerdict.REFINE
    else:
        verdict = QualityVerdict.PASS

    return ValidationReport(
        proposal_id=proposal.proposal_id,
        verdict=verdict,
        geometry_valid=geometry_valid,
        image_alignment_good=image_alignment_good,
        schema_valid=schema_valid,
        confidence_sufficient=confidence_sufficient,
        no_regression=True,
        issues=issues,
        metrics={"frame_count": len(track), "keyframe_count": len(kfs), "max_interp_err": round(err["max"], 3)},
    )


def detect_fragmentation_and_id_switches(
    proposals: Sequence[Proposal],
    reports: dict[str, ValidationReport],
    *,
    max_frag_gap: int = 5,
    min_frag_iou: float = 0.30,
    cross_iou: float = 0.35,
) -> None:
    """Detect track fragmentation candidates and crossing ID-switch risks without altering IDs."""
    tracks_data: list[tuple[Proposal, dict[int, list[float]]]] = []
    for p in proposals:
        tr, _ = _extract_track_frames(p)
        if tr:
            tracks_data.append((p, tr))

    n = len(tracks_data)
    for i in range(n):
        pi, ti = tracks_data[i]
        fi_min, fi_max = min(ti), max(ti)
        for j in range(i + 1, n):
            pj, tj = tracks_data[j]
            fj_min, fj_max = min(tj), max(tj)

            shared = sorted(set(ti) & set(tj))
            for f in shared:
                iou = _box_iou(ti[f], tj[f])
                if iou >= cross_iou:
                    for p_target, peer in ((pi, pj), (pj, pi)):
                        rep = reports[p_target.proposal_id]
                        rep.issues.append(
                            ValidationIssue(
                                code="TRACK_ID_SWITCH_RISK",
                                severity="warning",
                                message=f"Track {p_target.track_id} overlaps track {peer.track_id} at frame {f} (IoU={iou:.2f})",
                                region="occlusion_crossing",
                                affected_points=[f],
                                details={"peer_track_id": peer.track_id, "iou": round(iou, 3)},
                            )
                        )
                        if rep.verdict == QualityVerdict.PASS:
                            rep.verdict = QualityVerdict.REVIEW
                    break

            if pi.label == pj.label:
                if 0 < fj_min - fi_max <= max_frag_gap:
                    iou_gap = _box_iou(ti[fi_max], tj[fj_min])
                    if iou_gap >= min_frag_iou:
                        for p_target, peer in ((pi, pj), (pj, pi)):
                            rep = reports[p_target.proposal_id]
                            rep.issues.append(
                                ValidationIssue(
                                    code="TRACK_FRAGMENTATION_CANDIDATE",
                                    severity="info",
                                    message=f"Track {pi.track_id} (end {fi_max}) -> Track {pj.track_id} (start {fj_min}) gap={fj_min - fi_max}, IoU={iou_gap:.2f}",
                                    region="fragmentation",
                                    affected_points=[fi_max, fj_min],
                                    details={"peer_track_id": peer.track_id, "gap": fj_min - fi_max, "iou": round(iou_gap, 3)},
                                )
                            )
                            if rep.verdict == QualityVerdict.PASS:
                                rep.verdict = QualityVerdict.REVIEW
                elif 0 < fi_min - fj_max <= max_frag_gap:
                    iou_gap = _box_iou(tj[fj_max], ti[fi_min])
                    if iou_gap >= min_frag_iou:
                        for p_target, peer in ((pi, pj), (pj, pi)):
                            rep = reports[p_target.proposal_id]
                            rep.issues.append(
                                ValidationIssue(
                                    code="TRACK_FRAGMENTATION_CANDIDATE",
                                    severity="info",
                                    message=f"Track {pj.track_id} (end {fj_max}) -> Track {pi.track_id} (start {fi_min}) gap={fi_min - fj_max}, IoU={iou_gap:.2f}",
                                    region="fragmentation",
                                    affected_points=[fj_max, fi_min],
                                    details={"peer_track_id": peer.track_id, "gap": fi_min - fj_max, "iou": round(iou_gap, 3)},
                                )
                            )
                            if rep.verdict == QualityVerdict.PASS:
                                rep.verdict = QualityVerdict.REVIEW
