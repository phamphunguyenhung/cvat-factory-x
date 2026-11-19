"""Frame risk scoring from DECLARED features.

NOTE: risk score is used for ROUTING/prioritisation only. It is NOT evidence of
accuracy or label correctness.
"""
from __future__ import annotations

DEFAULT_WEIGHTS = {
    "uncertainty": 0.25, "occlusion": 0.15, "small_far": 0.15, "lighting": 0.10,
    "rare_class": 0.10, "disagreement": 0.15, "severity": 0.10,
}


def _clip(x) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


def score_frame(features: dict, policy: dict | None = None) -> dict:
    """features: confidence, occlusion, small_far, lighting, rare_class,
    disagreement, classes (list). policy: weights, class_severity {cls: 0..1}."""
    policy = policy or {}
    w = {**DEFAULT_WEIGHTS, **policy.get("weights", {})}
    sev_map = policy.get("class_severity", {})
    sev = max((_clip(sev_map.get(c, 0.0)) for c in features.get("classes", [])), default=0.0)
    comps = {
        "uncertainty": 1.0 - _clip(features.get("confidence", 0.0)),
        "occlusion": _clip(features.get("occlusion", 0.0)),
        "small_far": _clip(features.get("small_far", 0.0)),
        "lighting": _clip(features.get("lighting", 0.0)),
        "rare_class": _clip(features.get("rare_class", 0.0)),
        "disagreement": _clip(features.get("disagreement", 0.0)),
        "severity": sev,
    }
    total_w = sum(w.get(k, 0) for k in comps) or 1.0
    reasons = {k: {"value": round(v, 4), "weight": w.get(k, 0),
                   "contribution": round(v * w.get(k, 0) / total_w, 4)} for k, v in comps.items()}
    score = round(sum(r["contribution"] for r in reasons.values()), 4)
    return {"score": score, "reasons": reasons,
            "note": "routing score only; not accuracy evidence"}
