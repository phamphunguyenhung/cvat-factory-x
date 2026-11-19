"""Conditional Temporal Association Engine (P2).

Strict rule: Tracking is ONLY enabled when frame continuity is validated.
On independent snapshots (like Job 285), tracking stays disabled.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from harness.contracts import (
    ContinuityReport,
    ElementStatus,
    InstanceCandidate,
    TrackState,
    TrackStateKind,
)
from harness.frame_profiler import assess_continuity


@dataclass
class TemporalConfig:
    iou_match_thresh: float = 0.30
    max_age_frames: int = 4
    min_hits_to_confirm: int = 2
    min_continuity_corr: float = 0.45


def _iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (ua + ub - inter) if (ua + ub - inter) > 0.0 else 0.0


class TemporalAssociationEngine:
    """Conditional ByteTrack-style Hungarian tracker gated by continuity verification."""

    def __init__(self, config: TemporalConfig | None = None):
        self.config = config or TemporalConfig()

    def validate_sequence(
        self,
        frames: list[tuple[int, np.ndarray]],
        *,
        continuity_hint: str = "auto",
    ) -> ContinuityReport:
        return assess_continuity(
            frames,
            continuity_hint=continuity_hint,
            min_hist_corr=self.config.min_continuity_corr,
        )

    def associate(
        self,
        frames: list[tuple[int, np.ndarray]],
        candidates_by_frame: dict[int, list[InstanceCandidate]],
        *,
        continuity_hint: str = "auto",
    ) -> tuple[ContinuityReport, list[TrackState]]:
        """Run multi-frame track association ONLY if sequence continuity is verified."""
        report = self.validate_sequence(frames, continuity_hint=continuity_hint)
        if not report.is_continuous:
            return report, []

        tracks: dict[int, TrackState] = {}
        next_tid = 1

        for fid, _ in sorted(frames, key=lambda x: x[0]):
            cands = [
                c for c in candidates_by_frame.get(fid, [])
                if c.status != ElementStatus.SUPPRESSED
            ]
            active_tids = [
                tid for tid, tr in tracks.items()
                if tr.state != TrackStateKind.TERMINATED
            ]

            matched_cands: set[int] = set()
            matched_tracks: set[int] = set()

            if active_tids and cands:
                cost = np.ones((len(active_tids), len(cands)), dtype=np.float32)
                for r, tid in enumerate(active_tids):
                    tr = tracks[tid]
                    last_box = tr.boxes[tr.last_seen_frame]
                    pred_box = [
                        last_box[0] + tr.velocity_xy[0],
                        last_box[1] + tr.velocity_xy[1],
                        last_box[2] + tr.velocity_xy[0],
                        last_box[3] + tr.velocity_xy[1],
                    ]
                    for c_idx, cand in enumerate(cands):
                        if cand.label != tr.label:
                            continue
                        iou = max(_iou(last_box, cand.bbox_xyxy), _iou(pred_box, cand.bbox_xyxy))
                        if iou >= self.config.iou_match_thresh:
                            cost[r, c_idx] = 1.0 - iou

                r_ind, c_ind = linear_sum_assignment(cost)
                for r, c_idx in zip(r_ind, c_ind):
                    if cost[r, c_idx] <= (1.0 - self.config.iou_match_thresh):
                        tid = active_tids[r]
                        cand = cands[c_idx]
                        tr = tracks[tid]
                        prev_box = tr.boxes[tr.last_seen_frame]
                        dt = max(1, fid - tr.last_seen_frame)
                        vx = ((cand.bbox_xyxy[0] + cand.bbox_xyxy[2]) - (prev_box[0] + prev_box[2])) / (2.0 * dt)
                        vy = ((cand.bbox_xyxy[1] + cand.bbox_xyxy[3]) - (prev_box[1] + prev_box[3])) / (2.0 * dt)
                        tr.velocity_xy = (vx, vy)
                        tr.history[fid] = cand.candidate_id
                        tr.boxes[fid] = list(cand.bbox_xyxy)
                        tr.last_seen_frame = fid
                        tr.misses = 0
                        tr.state = (
                            TrackStateKind.CONFIRMED
                            if len(tr.history) >= self.config.min_hits_to_confirm
                            else TrackStateKind.TENTATIVE
                        )
                        cand.track_id = tid
                        matched_cands.add(c_idx)
                        matched_tracks.add(tid)

            # Age unmatched tracks
            for tid in active_tids:
                if tid not in matched_tracks:
                    tr = tracks[tid]
                    tr.misses += 1
                    if tr.misses > self.config.max_age_frames:
                        tr.state = TrackStateKind.TERMINATED
                    else:
                        tr.state = TrackStateKind.LOST

            # Spawn new tentative tracks for unmatched accepted candidates
            for c_idx, cand in enumerate(cands):
                if c_idx not in matched_cands and cand.status == ElementStatus.ACCEPTED:
                    tid = next_tid
                    next_tid += 1
                    tracks[tid] = TrackState(
                        track_id=tid,
                        label=cand.label,
                        state=TrackStateKind.TENTATIVE,
                        history={fid: cand.candidate_id},
                        boxes={fid: list(cand.bbox_xyxy)},
                        last_seen_frame=fid,
                    )
                    cand.track_id = tid

        return report, list(tracks.values())
