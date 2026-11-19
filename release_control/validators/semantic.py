"""Semantic validator: T5 bbox drift, T6 class swap, T7 scene leak, T8 guideline mismatch."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

from .models import (
    NormalizedBox,
    ValidationIssue,
    compute_iou,
    load_boxes_for_entry,
    resolve_image_id,
    resolve_image_relpath,
    resolve_label_relpath,
)

DEFAULT_DRIFT_IOU_THRESHOLD = 0.75
DEFAULT_DRIFT_CENTER_SHIFT = 0.04
DEFAULT_SWAP_PAIRS = (("car", "truck"), ("bus", "train"))
DEFAULT_DISTRIBUTION_SHIFT_THRESHOLD = 0.25


def _parse_ref_box_list(raw_boxes: Any) -> list[NormalizedBox]:
    """Parse a list of reference box dicts or sequences into NormalizedBox objects."""
    parsed: list[NormalizedBox] = []
    if not isinstance(raw_boxes, list):
        return parsed
    for idx, item in enumerate(raw_boxes, start=1):
        if isinstance(item, dict):
            cls_id = item.get("class_id")
            if cls_id is None:
                cls_id = item.get("category_id")
            if isinstance(cls_id, int):
                if all(k in item for k in ("cx", "cy", "w", "h")):
                    parsed.append(
                        NormalizedBox(
                            class_id=cls_id,
                            cx=float(item["cx"]),
                            cy=float(item["cy"]),
                            w=float(item["w"]),
                            h=float(item["h"]),
                            line_number=idx,
                        )
                    )
                elif isinstance(item.get("norm_xywh"), (list, tuple)) and len(item["norm_xywh"]) == 4:
                    cx, cy, w, h = item["norm_xywh"]
                    parsed.append(
                        NormalizedBox(
                            class_id=cls_id,
                            cx=float(cx),
                            cy=float(cy),
                            w=float(w),
                            h=float(h),
                            line_number=idx,
                        )
                    )
        elif isinstance(item, (list, tuple)) and len(item) == 5:
            cls_id, cx, cy, w, h = item
            parsed.append(
                NormalizedBox(
                    class_id=int(cls_id),
                    cx=float(cx),
                    cy=float(cy),
                    w=float(w),
                    h=float(h),
                    line_number=idx,
                )
            )
    return parsed


def _load_reference_boxes(
    reference_dir: str | Path | None,
    qc_evidence: dict[str, Any] | None,
) -> dict[str, list[NormalizedBox]]:
    """Load reference boxes keyed by image_id from reference_dir or qc_evidence."""
    ref_map: dict[str, list[NormalizedBox]] = {}

    if reference_dir is not None:
        ref_root = Path(reference_dir)
        ref_json = ref_root / "reference_boxes.json"
        if ref_json.is_file():
            data = json.loads(ref_json.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                container = data.get("images") or data.get("boxes") or data
                if isinstance(container, dict):
                    for img_key, raw_list in container.items():
                        ref_map[str(img_key)] = _parse_ref_box_list(raw_list)
        else:
            ref_manifest = ref_root / "manifest.json"
            if ref_manifest.is_file():
                m_data = json.loads(ref_manifest.read_text(encoding="utf-8"))
                for idx, entry in enumerate(m_data.get("images", [])):
                    if isinstance(entry, dict):
                        img_id = resolve_image_id(entry, idx)
                        ref_map[img_id] = load_boxes_for_entry(ref_root, entry)

    if qc_evidence and isinstance(qc_evidence.get("reference_boxes"), dict):
        for img_key, raw_list in qc_evidence["reference_boxes"].items():
            ref_map[str(img_key)] = _parse_ref_box_list(raw_list)

    return ref_map


def _build_class_name_to_id(
    manifest: dict[str, Any],
    contract: dict[str, Any],
) -> tuple[dict[str, int], dict[int, str]]:
    """Build bidirectional mappings between class names and class IDs."""
    name_to_id: dict[str, int] = {}
    id_to_name: dict[int, str] = {}

    for source in (manifest.get("classes"), contract.get("classes")):
        if isinstance(source, list):
            for idx, name in enumerate(source):
                if isinstance(name, str):
                    name_to_id[name.lower()] = idx
                    id_to_name[idx] = name
        elif isinstance(source, dict):
            for k, v in source.items():
                if isinstance(k, (int, str)) and str(k).isdigit() and isinstance(v, str):
                    cid = int(k)
                    name_to_id[v.lower()] = cid
                    id_to_name[cid] = v
                elif isinstance(k, str) and isinstance(v, int):
                    name_to_id[k.lower()] = v
                    id_to_name[v] = k

    return name_to_id, id_to_name


def _resolve_swap_pairs(
    contract: dict[str, Any],
    name_to_id: dict[str, int],
) -> list[tuple[int, int]]:
    """Resolve configured or default swap pairs into class_id tuples."""
    raw_pairs = contract.get("swap_pairs", DEFAULT_SWAP_PAIRS)
    resolved: list[tuple[int, int]] = []
    for pair in raw_pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        a, b = pair
        cid_a = a if isinstance(a, int) else name_to_id.get(str(a).lower())
        cid_b = b if isinstance(b, int) else name_to_id.get(str(b).lower())
        if cid_a is not None and cid_b is not None and cid_a != cid_b:
            resolved.append((cid_a, cid_b))
    return resolved


def _match_boxes_greedy(
    work_boxes: list[NormalizedBox],
    ref_boxes: list[NormalizedBox],
    same_class_only: bool = True,
) -> list[tuple[NormalizedBox, NormalizedBox, float]]:
    """Deterministically match working boxes to reference boxes by descending IoU."""
    candidates: list[tuple[float, int, int]] = []
    for wi, wb in enumerate(work_boxes):
        for ri, rb in enumerate(ref_boxes):
            if same_class_only and wb.class_id != rb.class_id:
                continue
            iou = compute_iou(wb, rb)
            candidates.append((iou, wi, ri))

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_w: set[int] = set()
    used_r: set[int] = set()
    matches: list[tuple[NormalizedBox, NormalizedBox, float]] = []

    for iou, wi, ri in candidates:
        if wi in used_w or ri in used_r:
            continue
        used_w.add(wi)
        used_r.add(ri)
        matches.append((work_boxes[wi], ref_boxes[ri], iou))

    return matches


def validate_semantic(
    dataset_dir: str | Path,
    manifest: dict[str, Any],
    contract: dict[str, Any] | None = None,
    reference_dir: str | Path | None = None,
    qc_evidence: dict[str, Any] | None = None,
) -> list[ValidationIssue]:
    """Run semantic checks: T5 bbox drift, T6 class swap, T7 scene leak, T8 guideline mismatch."""
    contract = contract or {}
    qc_evidence = qc_evidence or {}
    dataset_root = Path(dataset_dir)
    issues: list[ValidationIssue] = []

    images = manifest.get("images", [])
    if not isinstance(images, list):
        return issues

    # T8: Guideline mismatch (manifest-level & per-image)
    expected_guideline = contract.get("expected_guideline_version")
    if isinstance(expected_guideline, str) and expected_guideline.strip():
        manifest_guideline = manifest.get("guideline_version")
        if manifest_guideline != expected_guideline:
            issues.append(
                ValidationIssue(
                    code="T8_GUIDELINE_MISMATCH",
                    fault_type="T8",
                    severity="CRITICAL",
                    message=(
                        f"Manifest guideline_version '{manifest_guideline}' does not match "
                        f"contract expected_guideline_version '{expected_guideline}'."
                    ),
                    details={
                        "actual_guideline_version": manifest_guideline,
                        "expected_guideline_version": expected_guideline,
                    },
                )
            )

        for idx, entry in enumerate(images):
            if not isinstance(entry, dict):
                continue
            img_guideline = entry.get("guideline_version")
            if img_guideline is not None and img_guideline != expected_guideline:
                image_id = resolve_image_id(entry, idx)
                issues.append(
                    ValidationIssue(
                        code="T8_GUIDELINE_MISMATCH",
                        fault_type="T8",
                        severity="CRITICAL",
                        message=(
                            f"Image '{image_id}' guideline_version '{img_guideline}' does not match "
                            f"contract expected_guideline_version '{expected_guideline}'."
                        ),
                        file_path=resolve_image_relpath(entry),
                        image_id=image_id,
                        details={
                            "actual_guideline_version": img_guideline,
                            "expected_guideline_version": expected_guideline,
                        },
                    )
                )

    # T7: Scene leak across train and val splits
    train_scenes: set[str] = set()
    val_scenes: set[str] = set()
    for entry in images:
        if not isinstance(entry, dict):
            continue
        scene_id = entry.get("scene_id")
        split = str(entry.get("split", "")).strip().lower()
        if isinstance(scene_id, str) and scene_id.strip():
            if split == "train":
                train_scenes.add(scene_id.strip())
            elif split in ("val", "valid", "validation"):
                val_scenes.add(scene_id.strip())

    leaked_scenes = sorted(train_scenes & val_scenes)
    for scene_id in leaked_scenes:
        issues.append(
            ValidationIssue(
                code="T7_SCENE_LEAK",
                fault_type="T7",
                severity="CRITICAL",
                message=f"Scene '{scene_id}' appears in both train and val splits (data leakage).",
                details={"scene_id": scene_id, "splits": ["train", "val"]},
            )
        )

    # T5 & T6: Reference-based drift and class swap anomaly detection
    ref_map = _load_reference_boxes(reference_dir, qc_evidence)
    iou_threshold = float(contract.get("drift_iou_threshold", DEFAULT_DRIFT_IOU_THRESHOLD))
    shift_threshold = float(contract.get("drift_center_shift_threshold", DEFAULT_DRIFT_CENTER_SHIFT))
    qc_confirmed_drift = bool(qc_evidence.get("qc_confirmed_drift", qc_evidence.get("qc_confirmed", False)))
    qc_confirmed_swaps = bool(qc_evidence.get("qc_confirmed_swaps", qc_evidence.get("qc_confirmed", False)))

    name_to_id, id_to_name = _build_class_name_to_id(manifest, contract)
    swap_pairs = _resolve_swap_pairs(contract, name_to_id)
    swap_pair_set = {(min(a, b), max(a, b)) for a, b in swap_pairs}

    work_class_counts: Counter[int] = Counter()
    ref_class_counts: Counter[int] = Counter()

    for idx, entry in enumerate(images):
        if not isinstance(entry, dict):
            continue
        image_id = resolve_image_id(entry, idx)
        label_rel = resolve_label_relpath(entry)
        work_boxes = load_boxes_for_entry(dataset_root, entry)
        for wb in work_boxes:
            work_class_counts[wb.class_id] += 1

        # Resolve reference boxes by image_id, file_name, or relative image path
        ref_boxes = ref_map.get(image_id)
        if ref_boxes is None and entry.get("file_name"):
            ref_boxes = ref_map.get(str(entry["file_name"]))
        if ref_boxes is None:
            img_rel = resolve_image_relpath(entry)
            if img_rel:
                ref_boxes = (
                    ref_map.get(img_rel)
                    or ref_map.get(Path(img_rel).name)
                    or ref_map.get(Path(img_rel).stem)
                )
        if not ref_boxes:
            continue

        for rb in ref_boxes:
            ref_class_counts[rb.class_id] += 1

        # T5: Same-class box drift check
        same_matches = _match_boxes_greedy(work_boxes, ref_boxes, same_class_only=True)
        for wb, rb, iou in same_matches:
            center_shift = math.hypot(wb.cx - rb.cx, wb.cy - rb.cy)
            if iou < iou_threshold or center_shift > shift_threshold:
                issues.append(
                    ValidationIssue(
                        code="T5_BBOX_DRIFT",
                        fault_type="T5",
                        severity="WARNING",
                        message=(
                            f"Bbox drift anomaly in '{image_id}' (class_id={wb.class_id}, "
                            f"IoU={iou:.3f}, center_shift={center_shift:.4f}). "
                            "Anomaly detection only; NOT ground-truth proof unless confirmed by QC."
                        ),
                        file_path=label_rel,
                        image_id=image_id,
                        line_number=wb.line_number,
                        requires_human_review=not qc_confirmed_drift,
                        details={
                            "class_id": wb.class_id,
                            "iou": round(iou, 4),
                            "center_shift": round(center_shift, 4),
                            "qc_confirmed": qc_confirmed_drift,
                            "note": "Anomaly detection only; NOT ground-truth proof unless confirmed by QC.",
                        },
                    )
                )

        # T6: Box-level class swap check (high spatial overlap with different class)
        any_matches = _match_boxes_greedy(work_boxes, ref_boxes, same_class_only=False)
        for wb, rb, iou in any_matches:
            if wb.class_id == rb.class_id or iou < 0.5:
                continue
            pair_key = (min(wb.class_id, rb.class_id), max(wb.class_id, rb.class_id))
            is_known_pair = pair_key in swap_pair_set
            if is_known_pair or iou >= 0.75:
                w_name = id_to_name.get(wb.class_id, str(wb.class_id))
                r_name = id_to_name.get(rb.class_id, str(rb.class_id))
                issues.append(
                    ValidationIssue(
                        code="T6_CLASS_SWAP",
                        fault_type="T6",
                        severity="WARNING",
                        message=(
                            f"Anomalous class swap in '{image_id}': '{r_name}' -> '{w_name}' "
                            f"(IoU={iou:.3f}). Requires human review unless confirmed by QC."
                        ),
                        file_path=label_rel,
                        image_id=image_id,
                        line_number=wb.line_number,
                        requires_human_review=not qc_confirmed_swaps,
                        details={
                            "working_class_id": wb.class_id,
                            "reference_class_id": rb.class_id,
                            "working_class_name": w_name,
                            "reference_class_name": r_name,
                            "known_swap_pair": is_known_pair,
                            "iou": round(iou, 4),
                            "qc_confirmed": qc_confirmed_swaps,
                        },
                    )
                )

    # T6: Dataset-level distribution swap check for known swap pairs
    if isinstance(qc_evidence.get("reference_class_counts"), dict):
        for k, v in qc_evidence["reference_class_counts"].items():
            cid = int(k) if str(k).isdigit() else name_to_id.get(str(k).lower())
            if cid is not None:
                ref_class_counts[cid] = int(v)

    if ref_class_counts and work_class_counts:
        dist_threshold = float(
            contract.get("swap_distribution_shift_threshold", DEFAULT_DISTRIBUTION_SHIFT_THRESHOLD)
        )
        for cid_a, cid_b in swap_pairs:
            ref_pair_total = ref_class_counts[cid_a] + ref_class_counts[cid_b]
            work_pair_total = work_class_counts[cid_a] + work_class_counts[cid_b]
            if ref_pair_total == 0 or work_pair_total == 0:
                continue
            ref_share_a = ref_class_counts[cid_a] / ref_pair_total
            work_share_a = work_class_counts[cid_a] / work_pair_total
            if abs(work_share_a - ref_share_a) >= dist_threshold:
                name_a = id_to_name.get(cid_a, str(cid_a))
                name_b = id_to_name.get(cid_b, str(cid_b))
                issues.append(
                    ValidationIssue(
                        code="T6_CLASS_SWAP",
                        fault_type="T6",
                        severity="WARNING",
                        message=(
                            f"Anomalous class distribution shift between '{name_a}' and '{name_b}' "
                            f"(ref share {ref_share_a:.2f} -> work share {work_share_a:.2f}). "
                            "Requires human review unless confirmed by QC."
                        ),
                        requires_human_review=not qc_confirmed_swaps,
                        details={
                            "pair": [name_a, name_b],
                            "ref_counts": {name_a: ref_class_counts[cid_a], name_b: ref_class_counts[cid_b]},
                            "work_counts": {name_a: work_class_counts[cid_a], name_b: work_class_counts[cid_b]},
                            "qc_confirmed": qc_confirmed_swaps,
                        },
                    )
                )

    return issues
