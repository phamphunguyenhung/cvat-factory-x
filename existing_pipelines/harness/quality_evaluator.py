"""Quality Evaluator & Stop-Gate Auditor for Thing, Stuff, Structure, and System metrics."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from harness.contracts import ComposedScene, ElementStatus, LabelCategory, QualityReport


def _box_iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (ua + ub - inter) if (ua + ub - inter) > 0.0 else 0.0


class QualityEvaluator:
    """Evaluates composed scenes against GT shapes and enforces explicit stop gates."""

    def evaluate_things(
        self,
        preds_by_frame: dict[int, list[dict[str, Any]]],
        gt_by_frame: dict[int, list[dict[str, Any]]],
        *,
        iou_thresh: float = 0.5,
        include_review: bool = False,
    ) -> dict[str, Any]:
        tp = fp = fn = 0
        ious: list[float] = []
        per_label: dict[str, dict[str, int]] = defaultdict(lambda: {"gt": 0, "pred": 0, "tp": 0})

        all_frames = sorted(set(preds_by_frame) | set(gt_by_frame))
        for fid in all_frames:
            gts = [g for g in gt_by_frame.get(fid, []) if g.get("type") == "rectangle"]
            preds = [
                p
                for p in preds_by_frame.get(fid, [])
                if p.get("type") == "rectangle"
                and (include_review or p.get("_status", "accepted") == "accepted")
            ]
            for g in gts:
                per_label[g["label"]]["gt"] += 1
            for p in preds:
                per_label[p["label"]]["pred"] += 1

            used_gt: set[int] = set()
            preds_sorted = sorted(preds, key=lambda x: float(x.get("_conf", 1.0)), reverse=True)
            for p in preds_sorted:
                best_iou, best_j = 0.0, -1
                for j, g in enumerate(gts):
                    if j in used_gt:
                        continue
                    iou = _box_iou(p["points"], g["points"])
                    if iou >= iou_thresh and iou > best_iou and p["label"] == g["label"]:
                        best_iou, best_j = iou, j
                if best_j >= 0:
                    used_gt.add(best_j)
                    tp += 1
                    per_label[p["label"]]["tp"] += 1
                    ious.append(best_iou)
                else:
                    fp += 1
            fn += len(gts) - len(used_gt)

        prec = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        return {
            "gt": tp + fn,
            "pred": tp + fp,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "mean_iou": round(sum(ious) / len(ious), 4) if ious else 0.0,
            "per_label": {
                k: {
                    **v,
                    "recall": round(v["tp"] / v["gt"], 4) if v["gt"] else 0.0,
                    "precision": round(v["tp"] / v["pred"], 4) if v["pred"] else 0.0,
                }
                for k, v in sorted(per_label.items())
            },
        }

    def check_stop_gates(
        self,
        scenes: list[ComposedScene],
        *,
        max_latency_s: float = 15.0,
        max_rss_mb: float = 4200.0,
    ) -> QualityReport:
        latencies = [sum(s.timings_s.values()) for s in scenes if s.timings_s]
        mean_lat = sum(latencies) / len(latencies) if latencies else 0.0
        peak_rss = max((s.peak_rss_mb for s in scenes), default=0.0)
        review_total = sum(len(s.review_items) for s in scenes)
        elem_total = sum(len(s.elements) for s in scenes)

        gates = [
            {
                "gate": "cpu_latency_budget",
                "value": round(mean_lat, 3),
                "limit": max_latency_s,
                "passed": mean_lat <= max_latency_s,
            },
            {
                "gate": "cpu_peak_rss_mb",
                "value": round(peak_rss, 1),
                "limit": max_rss_mb,
                "passed": peak_rss <= max_rss_mb,
            },
        ]
        return QualityReport(
            scope=f"{len(scenes)} frames",
            metrics={
                "mean_latency_s_per_frame": round(mean_lat, 3),
                "peak_rss_mb": round(peak_rss, 1),
                "total_elements": elem_total,
                "total_review_items": review_total,
            },
            stop_gates=gates,
            honest_gaps=[
                "No tracking IDF1/IDSW computed on Job 285 because continuity=none.",
                "Partial GT in j285_ann.json omits some distant dark roadside vehicles.",
            ],
        )
