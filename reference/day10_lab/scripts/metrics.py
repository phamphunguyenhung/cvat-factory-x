"""Scoring contract for one delivery (detection + panoptic cùng một lô).

A finding is the pair (type, target). scene_leak targets a scene id.
Every other type targets an image stem. Semantic faults are scored apart
from rule faults so a single F1 cannot hide a gate that only catches hashes.
"""

from __future__ import annotations

RULE_TYPES = {
    # detection: nhìn file giao là thấy (luật)
    "missing_image",
    "wrong_hash",
    "invalid_class",
    "bbox_out_of_bounds",
    "duplicate_box",
    "scene_leak",
    "guideline_mismatch",
    # panoptic: luật
    "invalid_category_id",
    "thing_overlap",
}
# Không có nhãn gốc trong delivery: phải đối chiếu detection <-> panoptic rồi xem ảnh.
SEMANTIC_TYPES = {"class_swap", "bbox_drift", "missing_box", "category_swap", "instance_merge"}
HOLD_TYPES = RULE_TYPES | SEMANTIC_TYPES

GROUPS = {
    "rule_detectable": RULE_TYPES,
    "cross_check_review": SEMANTIC_TYPES,
}


def pairs(items: list[dict]) -> set[tuple[str, str]]:
    return {(item["type"], item["target"]) for item in items}


def score(truth: set[tuple[str, str]], pred: set[tuple[str, str]]) -> dict[str, float | int]:
    tp = len(truth & pred)
    fp = len(pred - truth)
    fn = len(truth - pred)
    # No claim and no fault: precision is 1. No claim while faults exist: precision
    # does not apply. A 0 there looks like a false alarm.
    if tp + fp == 0:
        precision = 1.0 if not truth else None
    else:
        precision = round(tp / (tp + fp), 4)
    recall = 1.0 if tp + fn == 0 else round(tp / (tp + fn), 4)
    if precision is None or precision + recall == 0:
        f1 = None
    else:
        f1 = round(2 * precision * recall / (precision + recall), 4)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def by_type(truth: set[tuple[str, str]], pred: set[tuple[str, str]]) -> list[dict]:
    rows = []
    for kind in sorted({item[0] for item in truth | pred}):
        true = {item for item in truth if item[0] == kind}
        flagged = {item for item in pred if item[0] == kind}
        row = score(true, flagged)
        row["type"] = kind
        if kind in SEMANTIC_TYPES:
            row["group"] = "cross_check_review"
        elif kind in RULE_TYPES:
            row["group"] = "rule_detectable"
        else:
            row["group"] = "other"
        rows.append(row)
    return rows


def group_scores(truth: set[tuple[str, str]], pred: set[tuple[str, str]]) -> dict[str, dict]:
    out = {}
    for name, kinds in GROUPS.items():
        out[name] = score(
            {item for item in truth if item[0] in kinds},
            {item for item in pred if item[0] in kinds},
        )
    return out


def decision(findings: list[dict]) -> str:
    """HOLD when a known fault is reported. Empty findings are not proof of a clean dataset."""
    kinds = {item["type"] for item in findings}
    if kinds & HOLD_TYPES:
        return "HOLD"
    return "RELEASE_CANDIDATE"
