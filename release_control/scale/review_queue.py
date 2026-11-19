"""Risk-prioritised review queue. Never marks 'reviewed' without a real event."""
from __future__ import annotations

import heapq
import random


class ReviewQueue:
    def __init__(self):
        self._heap: list = []
        self._seen: set = set()
        self.assignments: dict = {}   # frame_id -> reviewer
        self.reviewed: dict = {}      # frame_id -> event dict (only real events)

    def add(self, frame_id, risk: float):
        if frame_id in self._seen:
            return False
        self._seen.add(frame_id)
        heapq.heappush(self._heap, (-risk, str(frame_id), frame_id))
        return True

    def pop_batch(self, reviewer: str, n: int) -> list:
        out = []
        while self._heap and len(out) < n:
            _, _, fid = heapq.heappop(self._heap)
            if fid in self.assignments:
                continue
            self.assignments[fid] = reviewer
            out.append(fid)
        return out

    def assign(self, frame_id, reviewer: str):
        cur = self.assignments.get(frame_id)
        if cur is not None and cur != reviewer:
            raise ValueError(f"{frame_id} already assigned to {cur}")
        self.assignments[frame_id] = reviewer

    def record_review(self, frame_id, reviewer: str, event: dict):
        """Only call with a real review event (non-empty). Must be the assignee."""
        if not event:
            raise ValueError("review event required")
        if self.assignments.get(frame_id) != reviewer:
            raise ValueError("reviewer not assigned to frame")
        self.reviewed[frame_id] = {"reviewer": reviewer, **event}

    def is_reviewed(self, frame_id) -> bool:
        return frame_id in self.reviewed


def stratified_sample(items: list, bands: list, per_band: int, seed: int = 0) -> list:
    """items: (frame_id, risk). bands: ascending thresholds, e.g. [0.3, 0.6].
    Samples up to per_band from each stratum, deterministic via seed."""
    rng = random.Random(seed)
    strata: dict = {}
    for fid, r in items:
        strata.setdefault(sum(r >= b for b in bands), []).append(fid)
    out = []
    for k in sorted(strata):
        lst = sorted(strata[k], key=str)
        out += rng.sample(lst, min(per_band, len(lst)))
    return out
