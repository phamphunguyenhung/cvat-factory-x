"""5-Layer Pose17 Validator, Guideline/Rule Registry, and Case Engine.

Read-only on CVAT. Never imports torch/ultralytics/transformers at module top-level.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cvat_tools.core import paths

VALIDATOR_VERSION = "2.0.0"

DEFAULT_RULES_PATH = paths.WORKSPACE / "docs" / "guidelines" / "pose17" / "pose17_rules.csv"
DEFAULT_CASES_PATH = paths.WORKSPACE / "docs" / "guidelines" / "pose17" / "difficult_cases.csv"
DEFAULT_SNAPSHOT_DIR = paths.WORKSPACE / "artifacts" / "pose17_research" / "snapshots"
DEFAULT_OUTPUT_DIR = paths.WORKSPACE / "artifacts" / "pose17_research"
DEFAULT_VLM_DIR = paths.WORKSPACE / "reports" / "w2_vlm"

KEYPOINT_NAMES: dict[int, str] = {
    1: "nose",
    2: "eye_R",
    3: "eye_L",
    4: "ear_R",
    5: "ear_L",
    6: "shoulder_R",
    7: "shoulder_L",
    8: "elbow_R",
    9: "elbow_L",
    10: "wrist_R",
    11: "wrist_L",
    12: "hip_R",
    13: "hip_L",
    14: "knee_R",
    15: "knee_L",
    16: "ankle_R",
    17: "ankle_L",
}

R_L_PAIRS: list[tuple[int, int, str]] = [
    (2, 3, "eye"),
    (4, 5, "ear"),
    (6, 7, "shoulder"),
    (8, 9, "elbow"),
    (10, 11, "wrist"),
    (12, 13, "hip"),
    (14, 15, "knee"),
    (16, 17, "ankle"),
]

FACE_TORSO_IDS: set[int] = {1, 2, 3, 4, 5, 6, 7, 12, 13}
LIMB_IDS: set[int] = {8, 9, 10, 11, 14, 15, 16, 17}

# 18 SVG skeleton edges in CVAT person_vf17 (no [2, 3] eye-to-eye edge)
POSE17_SVG_EDGES: list[tuple[int, int]] = [
    (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7),
    (6, 7), (6, 8), (8, 10), (7, 9), (9, 11),
    (6, 12), (7, 13), (12, 13),
    (12, 14), (14, 16), (13, 15), (15, 17),
]

BONE_DEFS: list[tuple[str, int, int]] = [
    ("upper_arm_R", 6, 8),
    ("upper_arm_L", 7, 9),
    ("forearm_R", 8, 10),
    ("forearm_L", 9, 11),
    ("thigh_R", 12, 14),
    ("thigh_L", 13, 15),
    ("shank_R", 14, 16),
    ("shank_L", 15, 17),
]

RULE_CSV_FIELDS = [
    "rule_id", "rule_version", "scope", "category", "condition",
    "expected_action", "prohibited_action", "source_type", "source_ref",
    "evidence_frame", "confidence", "status",
]

OVERLAP_MIN_PX = 3.0
BORDER_IN_FRAME_PX = 8.0
BONE_MIN_TORSO_RATIO = 0.15
BONE_MAX_MEDIAN_MULT = 2.5
BONE_MIN_MEDIAN_MULT = 0.15
LR_MIN_TORSO_SEP = 0.12
BTC_BASE_W, BTC_BASE_H = 960, 540


def get_structured_logger(name: str = "cvat_tools.pose17") -> logging.Logger:
    """Return a logger configured with structured JSON formatting if no handlers exist."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()

        class _JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                payload = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "level": record.levelname,
                    "logger": record.name,
                    "msg": record.getMessage(),
                }
                extra = getattr(record, "structured", None)
                if isinstance(extra, dict):
                    payload.update(extra)
                return json.dumps(payload, ensure_ascii=False)

        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


@dataclass
class ValidationFinding:
    job_id: int
    frame_index: int
    image_name: str
    skeleton_id: str
    keypoint_id: str
    layer: str
    issue_type: str
    severity: str
    rule_id: str
    message: str
    advisory_only: bool = False
    is_ground_truth: bool = False
    auto_promoted: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)

    def dedup_key(self) -> tuple[Any, ...]:
        return (
            self.job_id,
            self.frame_index,
            self.skeleton_id,
            self.keypoint_id,
            self.layer,
            self.issue_type,
        )

    def to_csv_row(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "frame_index": self.frame_index,
            "image_name": self.image_name,
            "skeleton_id": self.skeleton_id,
            "keypoint_id": self.keypoint_id,
            "layer": self.layer,
            "issue_type": self.issue_type,
            "severity": self.severity,
            "rule_id": self.rule_id,
            "message": self.message,
            "advisory_only": str(self.advisory_only).lower(),
            "is_ground_truth": str(self.is_ground_truth).lower(),
            "auto_promoted": str(self.auto_promoted).lower(),
            "metrics_json": json.dumps(self.metrics, ensure_ascii=False, sort_keys=True),
        }


# ---------------------------------------------------------------------------
# Rule & Case Registry Engine
# ---------------------------------------------------------------------------

def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def lookup_rule(rule_id: str, rules_path: Path | None = None) -> dict[str, str] | None:
    """Return rule metadata dict for rule_id, or None if not found."""
    p = Path(rules_path) if rules_path else DEFAULT_RULES_PATH
    for row in _read_csv_rows(p):
        if row.get("rule_id") == rule_id:
            return dict(row)
    return None


def cases_for_rule(rule_id: str, cases_path: Path | None = None) -> list[dict[str, str]]:
    """Return all difficult cases referencing rule_id."""
    p = Path(cases_path) if cases_path else DEFAULT_CASES_PATH
    out: list[dict[str, str]] = []
    for row in _read_csv_rows(p):
        rids = [r.strip() for r in (row.get("rule_ids") or "").split(";") if r.strip()]
        if rule_id in rids:
            out.append(dict(row))
    return out


def open_case(
    case_id: str,
    cases_path: Path | None = None,
    workspace: Path | None = None,
) -> dict[str, Any] | None:
    """Return case metadata with verified evidence_path and linked rule details."""
    cp = Path(cases_path) if cases_path else DEFAULT_CASES_PATH
    ws = Path(workspace) if workspace else paths.WORKSPACE
    for row in _read_csv_rows(cp):
        if row.get("case_id") == case_id:
            ev_rel = (row.get("evidence_path") or "").strip()
            ev_abs = (ws / ev_rel).resolve() if ev_rel else None
            ev_exists = bool(ev_abs and ev_abs.exists())
            rids = [r.strip() for r in (row.get("rule_ids") or "").split(";") if r.strip()]
            return {
                **row,
                "evidence_abs_path": str(ev_abs) if ev_abs else "",
                "evidence_exists": ev_exists,
                "rule_ids_list": rids,
            }
    return None


def propose_rule(rule: dict[str, Any], rules_path: Path | None = None) -> dict[str, Any]:
    """Idempotently append or update a proposed rule without ever overwriting approved rules."""
    p = Path(rules_path) if rules_path else DEFAULT_RULES_PATH
    rule_id = str(rule.get("rule_id") or "").strip()
    if not rule_id:
        raise ValueError("rule_id is required")

    rows = _read_csv_rows(p)
    normalized = {k: str(rule.get(k, "")).strip() for k in RULE_CSV_FIELDS}
    normalized["rule_id"] = rule_id
    if not normalized["rule_version"]:
        normalized["rule_version"] = "2.0"
    if not normalized["confidence"]:
        normalized["confidence"] = "medium"
    normalized["status"] = "proposed"

    for idx, existing in enumerate(rows):
        if existing.get("rule_id") == rule_id:
            if (existing.get("status") or "").strip().lower() == "approved":
                return {"action": "preserved_approved", "rule": dict(existing)}
            if all(str(existing.get(k, "")).strip() == normalized[k] for k in RULE_CSV_FIELDS):
                return {"action": "unchanged", "rule": dict(existing)}
            rows[idx] = normalized
            _write_csv_rows(p, RULE_CSV_FIELDS, rows)
            return {"action": "updated_proposed", "rule": normalized}

    rows.append(normalized)
    _write_csv_rows(p, RULE_CSV_FIELDS, rows)
    return {"action": "appended_proposed", "rule": normalized}


# ---------------------------------------------------------------------------
# Snapshot Loading & Normalization (Offline / Online Read-Only)
# ---------------------------------------------------------------------------

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_job_snapshot(
    job_id: int,
    offline: bool = True,
    profile: str = "edufun",
    snapshot_path: Path | None = None,
) -> dict[str, Any]:
    """Load job data into canonical keypoint_mapping structure + snapshot_sha256."""
    if snapshot_path is not None:
        sp = Path(snapshot_path)
        raw_bytes = sp.read_bytes()
        data = json.loads(raw_bytes.decode("utf-8"))
        data["snapshot_sha256"] = _sha256_bytes(raw_bytes)
        data["snapshot_path"] = str(sp)
        return data

    if offline:
        km_path = DEFAULT_SNAPSHOT_DIR / "keypoint_mapping.json"
        if km_path.exists():
            raw_bytes = km_path.read_bytes()
            data = json.loads(raw_bytes.decode("utf-8"))
            if int(data.get("job_id", -1)) == int(job_id):
                data["snapshot_sha256"] = _sha256_bytes(raw_bytes)
                data["snapshot_path"] = str(km_path)
                return data

        jdir = paths.job_dir(profile, job_id)
        if jdir.exists() and (jdir / "job_info.json").exists():
            info = json.loads((jdir / "job_info.json").read_text(encoding="utf-8"))
            labels = json.loads((jdir / "labels.json").read_text(encoding="utf-8")) if (jdir / "labels.json").exists() else []
            ann_path = jdir / "annotations.json"
            anns = json.loads(ann_path.read_text(encoding="utf-8")) if ann_path.exists() else {"shapes": []}
            return _normalize_cvat_payload(job_id, info, labels, anns, source=str(jdir))

        raise FileNotFoundError(f"No offline snapshot found for job {job_id}")

    # Online read-only CVAT
    from cvat_tools.core.client import Cvat
    cv = Cvat.from_env(profile=profile, write_enabled=False)
    job_info = cv.job(job_id)
    meta = cv.meta(job_id)
    labels = cv.labels(job_id)
    anns = cv.annotations(job_id)
    return _normalize_cvat_payload(job_id, {"job": job_info, "meta": meta}, labels, anns, source="cvat_readonly")


def _normalize_cvat_payload(
    job_id: int,
    info: dict[str, Any],
    labels: list[dict[str, Any]],
    anns: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    meta = info.get("meta", info)
    job_meta = info.get("job", info)
    start_f = int(job_meta.get("start_frame", meta.get("start_frame", 0)))
    stop_f = int(job_meta.get("stop_frame", meta.get("stop_frame", start_f + len(meta.get("frames", [])) - 1)))
    sub_map: dict[int, str] = {}
    label_map: dict[int, str] = {}
    for lab in labels:
        label_map[int(lab["id"])] = lab["name"]
        for sl in lab.get("sublabels", []):
            sub_map[int(sl["id"])] = sl["name"]

    frames_list = meta.get("frames", [])
    frames_out: dict[str, Any] = {}
    for idx, fmeta in enumerate(frames_list):
        fidx = start_f + idx
        name = Path(fmeta.get("name", f"frame_{fidx:06d}.jpg")).name
        frames_out[str(fidx)] = {
            "frame_index": fidx,
            "image_name": name,
            "meta_name": fmeta.get("name", name),
            "width": int(fmeta.get("width", 0)),
            "height": int(fmeta.get("height", 0)),
            "image_hash": "",
            "skeleton_count": 0,
            "skeletons": {},
        }

    for sh in anns.get("shapes", []):
        fidx = int(sh["frame"])
        fkey = str(fidx)
        if fkey not in frames_out:
            frames_out[fkey] = {
                "frame_index": fidx,
                "image_name": f"frame_{fidx:06d}.jpg",
                "width": 0,
                "height": 0,
                "skeleton_count": 0,
                "skeletons": {},
            }
        sid = str(sh.get("id", f"tmp_{len(frames_out[fkey]['skeletons'])}"))
        kps = []
        for el in sh.get("elements", []):
            pts = el.get("points", [0.0, 0.0])
            kps.append({
                "keypoint_id": el.get("id"),
                "sublabel_id": el.get("label_id"),
                "sublabel_name": sub_map.get(int(el.get("label_id", -1)), str(el.get("label_id", ""))),
                "x": float(pts[0]),
                "y": float(pts[1]),
                "occluded": bool(el.get("occluded", False)),
                "outside": bool(el.get("outside", False)),
                "hidden": bool(el.get("hidden", False)),
                "source": el.get("source", sh.get("source", "manual")),
            })
        frames_out[fkey]["skeletons"][sid] = {
            "skeleton_id": sh.get("id", sid),
            "label_id": sh.get("label_id"),
            "label_name": label_map.get(int(sh.get("label_id", -1)), "person_vf17"),
            "type": sh.get("type", "skeleton"),
            "source": sh.get("source", "manual"),
            "occluded": bool(sh.get("occluded", False)),
            "outside": bool(sh.get("outside", False)),
            "hidden": bool(sh.get("hidden", False)),
            "keypoints": kps,
        }
        frames_out[fkey]["skeleton_count"] = len(frames_out[fkey]["skeletons"])

    payload = {
        "job_id": job_id,
        "start_frame": start_f,
        "stop_frame": stop_f,
        "frame_count": len(frames_out),
        "source_snapshot": source,
        "frames": frames_out,
    }
    payload["snapshot_sha256"] = _sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return payload


# ---------------------------------------------------------------------------
# Geometry Helpers
# ---------------------------------------------------------------------------

def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _kp_map(skeleton: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for kp in skeleton.get("keypoints", []):
        name = str(kp.get("sublabel_name", "")).strip()
        if name.isdigit():
            out[int(name)] = kp
    return out


def _is_active(kp: dict[str, Any] | None) -> bool:
    return kp is not None and not bool(kp.get("outside", False))


def _torso_frame_metrics(kpm: dict[int, dict[str, Any]]) -> dict[str, Any] | None:
    req = (6, 7, 12, 13)
    if not all(_is_active(kpm.get(i)) for i in req):
        return None
    s_r = (float(kpm[6]["x"]), float(kpm[6]["y"]))
    s_l = (float(kpm[7]["x"]), float(kpm[7]["y"]))
    h_r = (float(kpm[12]["x"]), float(kpm[12]["y"]))
    h_l = (float(kpm[13]["x"]), float(kpm[13]["y"]))
    mid_s = ((s_r[0] + s_l[0]) / 2.0, (s_r[1] + s_l[1]) / 2.0)
    mid_h = ((h_r[0] + h_l[0]) / 2.0, (h_r[1] + h_l[1]) / 2.0)
    vx = mid_s[0] - mid_h[0]
    vy = mid_s[1] - mid_h[1]
    torso_h = math.hypot(vx, vy)
    if torso_h < 1e-6:
        return None
    # Perpendicular unit vector pointing to body-aligned image right when upright (vy < 0)
    ux, uy = (-vy / torso_h, vx / torso_h)
    is_horizontal = abs(vx) > abs(vy)
    return {
        "torso_height": torso_h,
        "vx": vx,
        "vy": vy,
        "u_right": (ux, uy),
        "is_horizontal": is_horizontal,
    }


def _ccw(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(
    a1: tuple[float, float], a2: tuple[float, float],
    b1: tuple[float, float], b2: tuple[float, float],
) -> bool:
    d1 = _ccw(a1, a2, b1)
    d2 = _ccw(a1, a2, b2)
    d3 = _ccw(b1, b2, a1)
    d4 = _ccw(b1, b2, a2)
    return ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))


# ---------------------------------------------------------------------------
# 5-Layer Validation Engine
# ---------------------------------------------------------------------------

def validate_frame(
    frame: dict[str, Any],
    job_id: int = 0,
    bone_medians: dict[str, float] | None = None,
    vlm_entry: dict[str, Any] | None = None,
    image_metrics: dict[str, float] | None = None,
) -> list[ValidationFinding]:
    """Run L1 (Structural), L2 (Geometric), L3 (Semantic), and L5 (Evidence QA) on one frame."""
    findings: list[ValidationFinding] = []
    fidx = int(frame.get("frame_index", 0))
    img_name = str(frame.get("image_name", f"frame_{fidx}.jpg"))
    w = int(frame.get("width", 0))
    h = int(frame.get("height", 0))
    skeletons = frame.get("skeletons") or {}
    skel_count = int(frame.get("skeleton_count", len(skeletons)))

    # L1: Frame mapping & missing/multiple skeletons
    if w <= 0 or h <= 0:
        findings.append(ValidationFinding(
            job_id, fidx, img_name, "", "all", "L1_STRUCTURAL",
            "INVALID_FRAME_MAPPING", "error", "P17-G02",
            f"Invalid frame dimensions ({w}x{h})",
        ))
    if skel_count == 0 or len(skeletons) == 0:
        findings.append(ValidationFinding(
            job_id, fidx, img_name, "", "1..17", "L1_STRUCTURAL",
            "MISSING_SKELETON", "error", "P17-G01",
            "Frame has 0 skeletons (missing driver annotation)",
        ))
        return findings

    if len(skeletons) > 1:
        findings.append(ValidationFinding(
            job_id, fidx, img_name, "", "all", "L1_STRUCTURAL",
            "MULTIPLE_SKELETONS", "warning", "P17-G01",
            f"Frame has {len(skeletons)} skeletons; guideline requires 1 driver skeleton",
        ))

    for sid, skel in skeletons.items():
        sid_str = str(sid)
        label_name = str(skel.get("label_name", ""))
        if label_name and label_name != "person_vf17":
            findings.append(ValidationFinding(
                job_id, fidx, img_name, sid_str, "all", "L1_STRUCTURAL",
                "INVALID_LABEL", "error", "P17-G02",
                f"Expected label person_vf17, got {label_name}",
            ))

        if bool(skel.get("hidden", False)):
            findings.append(ValidationFinding(
                job_id, fidx, img_name, sid_str, "skeleton", "L1_STRUCTURAL",
                "FORBIDDEN_HIDDEN_FLAG", "error", "P17-G10",
                "Skeleton has forbidden hidden=True flag",
            ))

        kps = skel.get("keypoints") or []
        names = [str(kp.get("sublabel_name", "")).strip() for kp in kps]
        expected = [str(i) for i in range(1, 18)]
        missing = [e for e in expected if e not in names]
        if missing or len(kps) != 17:
            findings.append(ValidationFinding(
                job_id, fidx, img_name, sid_str, ",".join(missing) or "all", "L1_STRUCTURAL",
                "INVALID_KEYPOINT_SCHEMA", "error", "P17-G02",
                f"Expected 17 keypoints 1..17, got {len(kps)} (missing={missing})",
            ))

        # L1/L3 per-keypoint checks
        for kp in kps:
            kid = str(kp.get("sublabel_name", "")).strip()
            x, y = float(kp.get("x", 0.0)), float(kp.get("y", 0.0))
            occ = bool(kp.get("occluded", False))
            out = bool(kp.get("outside", False))
            hid = bool(kp.get("hidden", False))

            if hid:
                findings.append(ValidationFinding(
                    job_id, fidx, img_name, sid_str, kid, "L1_STRUCTURAL",
                    "FORBIDDEN_HIDDEN_FLAG", "error", "P17-G10",
                    f"Keypoint {kid} has forbidden hidden=True flag",
                ))

            # L3: Semantic visibility rules
            if occ and out:
                findings.append(ValidationFinding(
                    job_id, fidx, img_name, sid_str, kid, "L3_SEMANTIC",
                    "VISIBILITY_CONFLICT_OCC_OUT", "error", "P17-G11",
                    f"Keypoint {kid} has occluded=True AND outside=True simultaneously",
                    metrics={"x": x, "y": y},
                ))

            kid_int = int(kid) if kid.isdigit() else -1
            in_bounds = (w > 0 and h > 0 and 0.0 <= x <= w and 0.0 <= y <= h)
            min_border = min(x, w - x, y, h - y) if in_bounds else -1.0

            if out and kid_int in FACE_TORSO_IDS and in_bounds:
                findings.append(ValidationFinding(
                    job_id, fidx, img_name, sid_str, kid, "L3_SEMANTIC",
                    "IN_FRAME_FACE_TORSO_OUTSIDE", "error", "P17-G15",
                    f"Face/torso keypoint {kid} ({KEYPOINT_NAMES.get(kid_int, '')}) marked outside inside frame (border={min_border:.1f}px)",
                    metrics={"x": x, "y": y, "min_border_px": round(min_border, 2)},
                ))
            elif out and min_border >= BORDER_IN_FRAME_PX:
                findings.append(ValidationFinding(
                    job_id, fidx, img_name, sid_str, kid, "L3_SEMANTIC",
                    "OUTSIDE_DEEP_IN_FRAME", "warning", "P17-G13",
                    f"Keypoint {kid} marked outside while {min_border:.1f}px inside frame",
                    metrics={"x": x, "y": y, "min_border_px": round(min_border, 2)},
                ))

            # L2: Bounds check for visible / occluded (non-outside)
            if not out and w > 0 and h > 0 and not in_bounds:
                findings.append(ValidationFinding(
                    job_id, fidx, img_name, sid_str, kid, "L2_GEOMETRIC",
                    "OUT_OF_BOUNDS_COORD", "error", "P17-G23",
                    f"Non-outside keypoint {kid} at ({x:.1f}, {y:.1f}) is outside [0,{w}]x[0,{h}]",
                    metrics={"x": x, "y": y, "width": w, "height": h},
                ))

        # L3: Skeleton bulk occluded
        if bool(skel.get("occluded", False)):
            findings.append(ValidationFinding(
                job_id, fidx, img_name, sid_str, "skeleton", "L3_SEMANTIC",
                "SKELETON_BULK_OCCLUDED", "error", "P17-G12",
                "Skeleton-level occluded=True (bulk-occlude forbidden)",
            ))

        # L2: Keypoint overlap (< 3 px)
        kpm = _kp_map(skel)
        active_ids = sorted(i for i, kp in kpm.items() if _is_active(kp))
        for idx_a in range(len(active_ids)):
            for idx_b in range(idx_a + 1, len(active_ids)):
                ia, ib = active_ids[idx_a], active_ids[idx_b]
                pa = (float(kpm[ia]["x"]), float(kpm[ia]["y"]))
                pb = (float(kpm[ib]["x"]), float(kpm[ib]["y"]))
                d = _dist(pa, pb)
                if d < OVERLAP_MIN_PX:
                    findings.append(ValidationFinding(
                        job_id, fidx, img_name, sid_str, f"{ia};{ib}", "L2_GEOMETRIC",
                        "KEYPOINT_OVERLAP", "error", "P17-G22",
                        f"Keypoints {ia} and {ib} overlap at {d:.2f}px (< {OVERLAP_MIN_PX}px)",
                        metrics={"dist_px": round(d, 3), "kp_a": ia, "kp_b": ib},
                    ))

        # L2: Torso metrics, bone lengths, R/L image check, chain crossings
        tmet = _torso_frame_metrics(kpm)
        if tmet:
            torso_h = tmet["torso_height"]
            ux, uy = tmet["u_right"]
            is_horiz = tmet["is_horizontal"]

            # Bone lengths
            for bname, ia, ib in BONE_DEFS:
                if _is_active(kpm.get(ia)) and _is_active(kpm.get(ib)):
                    blen = _dist((float(kpm[ia]["x"]), float(kpm[ia]["y"])),
                                 (float(kpm[ib]["x"]), float(kpm[ib]["y"])))
                    ratio = blen / torso_h
                    med = (bone_medians or {}).get(bname)
                    low_thresh = max(BONE_MIN_TORSO_RATIO, BONE_MIN_MEDIAN_MULT * med) if med else BONE_MIN_TORSO_RATIO
                    high_thresh = (BONE_MAX_MEDIAN_MULT * med) if med else 2.5
                    if ratio < low_thresh or ratio > high_thresh:
                        findings.append(ValidationFinding(
                            job_id, fidx, img_name, sid_str, f"{ia};{ib}", "L2_GEOMETRIC",
                            "BONE_LENGTH_OUTLIER", "warning", "P17-G24",
                            f"Bone {bname} ratio={ratio:.3f}x torso outside [{low_thresh:.3f}, {high_thresh:.3f}]",
                            metrics={"bone": bname, "len_px": round(blen, 2), "torso_ratio": round(ratio, 4)},
                        ))

            # R/L check (image-based upright vs body-aligned horizontal)
            for ir, il, pname in R_L_PAIRS:
                if _is_active(kpm.get(ir)) and _is_active(kpm.get(il)):
                    dx = float(kpm[ir]["x"]) - float(kpm[il]["x"])
                    dy = float(kpm[ir]["y"]) - float(kpm[il]["y"])
                    proj = dx * ux + dy * uy
                    if not is_horiz and proj < 0:
                        findings.append(ValidationFinding(
                            job_id, fidx, img_name, sid_str, f"{ir};{il}", "L2_GEOMETRIC",
                            "LEFT_RIGHT_INVERSION", "error" if ir in (6, 7, 12, 13) else "warning", "P17-G06",
                            f"Upright R/L inversion on {pname} ({ir}/{il}): proj={proj:.1f}px",
                            metrics={"pair": pname, "proj_px": round(proj, 2), "horizontal_frame": False},
                        ))
                    elif is_horiz and proj < -LR_MIN_TORSO_SEP * torso_h:
                        findings.append(ValidationFinding(
                            job_id, fidx, img_name, sid_str, f"{ir};{il}", "L2_GEOMETRIC",
                            "BODY_ALIGNED_LR_INVERSION", "warning", "P17-G08",
                            f"Horizontal/rotated frame R/L body-aligned inversion on {pname} ({ir}/{il}): proj={proj:.1f}px",
                            metrics={"pair": pname, "proj_px": round(proj, 2), "horizontal_frame": True},
                        ))

            # Upright leg chain inversion (ankle higher than knee)
            if not is_horiz:
                for knee, ankle, side in ((14, 16, "R"), (15, 17, "L")):
                    if _is_active(kpm.get(knee)) and _is_active(kpm.get(ankle)):
                        if float(kpm[ankle]["y"]) < float(kpm[knee]["y"]):
                            findings.append(ValidationFinding(
                                job_id, fidx, img_name, sid_str, f"{knee};{ankle}", "L2_GEOMETRIC",
                                "LIMB_CHAIN_INVERSION", "warning", "P17-G25",
                                f"Ankle_{side} ({ankle}) is above Knee_{side} ({knee}) in upright frame",
                                metrics={"knee_y": kpm[knee]["y"], "ankle_y": kpm[ankle]["y"]},
                            ))

        # L2: Chain crossings (upper limbs vs upper limbs, lower limbs vs lower limbs)
        upper_segs = [(6, 8), (8, 10), (7, 9), (9, 11)]
        for i in range(len(upper_segs)):
            for j in range(i + 1, len(upper_segs)):
                a, b = upper_segs[i]
                c, d = upper_segs[j]
                if len({a, b, c, d}) < 4:
                    continue
                if all(_is_active(kpm.get(k)) for k in (a, b, c, d)):
                    pa = (float(kpm[a]["x"]), float(kpm[a]["y"]))
                    pb = (float(kpm[b]["x"]), float(kpm[b]["y"]))
                    pc = (float(kpm[c]["x"]), float(kpm[c]["y"]))
                    pd = (float(kpm[d]["x"]), float(kpm[d]["y"]))
                    if _segments_intersect(pa, pb, pc, pd):
                        findings.append(ValidationFinding(
                            job_id, fidx, img_name, sid_str, f"{a}-{b};{c}-{d}", "L2_GEOMETRIC",
                            "CHAIN_CROSSING", "warning", "P17-G07",
                            f"Upper-limb segments ({a}-{b}) and ({c}-{d}) cross",
                        ))

    # L5: Evidence-based QA (strictly advisory, never auto-promoted)
    if vlm_entry:
        status = str(vlm_entry.get("status", "")).lower()
        vlm_issues = vlm_entry.get("issues") or []
        if status in ("refine", "fail", "reject") or vlm_issues:
            findings.append(ValidationFinding(
                job_id, fidx, img_name, "", "all", "L5_EVIDENCE_QA",
                "SUSPICIOUS_QA_VLM", "info", "P17-G03",
                f"VLM advisory ({status}): {len(vlm_issues)} flagged items (never auto-promoted to ground truth)",
                advisory_only=True, is_ground_truth=False, auto_promoted=False,
                metrics={"vlm_status": status, "vlm_issue_count": len(vlm_issues)},
            ))
    if image_metrics:
        mean_l = image_metrics.get("mean_luma")
        lap_var = image_metrics.get("lap_var")
        if mean_l is not None and mean_l < 30.0:
            findings.append(ValidationFinding(
                job_id, fidx, img_name, "", "all", "L5_EVIDENCE_QA",
                "SUSPICIOUS_QA_LOW_LIGHT", "info", "P17-G05",
                f"Low-light IR frame (mean_luma={mean_l:.1f} < 30)",
                advisory_only=True, is_ground_truth=False, auto_promoted=False,
                metrics={"mean_luma": round(mean_l, 2)},
            ))
        if lap_var is not None and lap_var < 25.0:
            findings.append(ValidationFinding(
                job_id, fidx, img_name, "", "all", "L5_EVIDENCE_QA",
                "SUSPICIOUS_QA_BLUR", "info", "P17-G05",
                f"Low-sharpness/blurry frame (lap_var={lap_var:.1f} < 25)",
                advisory_only=True, is_ground_truth=False, auto_promoted=False,
                metrics={"lap_var": round(lap_var, 2)},
            ))

    return findings


def _load_vlm_reports(job_id: int, vlm_dir: Path | None = None) -> dict[int, dict[str, Any]]:
    vd = Path(vlm_dir) if vlm_dir else DEFAULT_VLM_DIR
    out: dict[int, dict[str, Any]] = {}
    if not vd.exists():
        return out
    for p in sorted(vd.glob(f"{job_id}_*.json")):
        try:
            items = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and "frame" in item:
                        out[int(item["frame"])] = item
        except Exception:
            continue
    return out


def _compute_image_metrics(job_id: int, image_name: str, profile: str = "edufun") -> dict[str, float] | None:
    fdir = paths.job_dir(profile, job_id) / "frames"
    img_path = fdir / image_name
    if not img_path.exists():
        return None
    try:
        import cv2
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        mean_luma = float(img.mean())
        lap_var = float(cv2.Laplacian(img, cv2.CV_64F).var())
        return {"mean_luma": mean_luma, "lap_var": lap_var}
    except Exception:
        return None


def _compute_job_bone_medians(frames: list[dict[str, Any]]) -> dict[str, float]:
    ratios: dict[str, list[float]] = {b[0]: [] for b in BONE_DEFS}
    for fr in frames:
        for skel in (fr.get("skeletons") or {}).values():
            kpm = _kp_map(skel)
            tmet = _torso_frame_metrics(kpm)
            if not tmet:
                continue
            th = tmet["torso_height"]
            for bname, ia, ib in BONE_DEFS:
                if _is_active(kpm.get(ia)) and _is_active(kpm.get(ib)):
                    blen = _dist((float(kpm[ia]["x"]), float(kpm[ia]["y"])),
                                 (float(kpm[ib]["x"]), float(kpm[ib]["y"])))
                    ratios[bname].append(blen / th)
    medians: dict[str, float] = {}
    for bname, vals in ratios.items():
        if vals:
            s = sorted(vals)
            medians[bname] = s[len(s) // 2]
    return medians


def validate_job(
    job_data: dict[str, Any],
    profile: str = "edufun",
    vlm_dir: Path | None = None,
    compute_img_metrics: bool = True,
) -> dict[str, Any]:
    """Run all 5 validation layers (L1-L5) on a loaded job snapshot."""
    job_id = int(job_data.get("job_id", 0))
    frames_dict = job_data.get("frames") or {}
    frames = [frames_dict[k] for k in sorted(frames_dict.keys(), key=lambda x: int(x) if str(x).isdigit() else 0)]

    resolutions = sorted({(int(fr.get("width", 0)), int(fr.get("height", 0))) for fr in frames})
    is_independent_snapshots = len(resolutions) > 1
    bone_medians = _compute_job_bone_medians(frames)
    vlm_by_frame = _load_vlm_reports(job_id, vlm_dir=vlm_dir)

    all_findings: list[ValidationFinding] = []
    coverage_rows: list[dict[str, Any]] = []

    # L4: Cross-frame / Cross-case mode check
    if is_independent_snapshots:
        all_findings.append(ValidationFinding(
            job_id, -1, "", "", "all", "L4_CROSS_CASE",
            "INDEPENDENT_SNAPSHOT_COLLECTION", "info", "P17-G26",
            f"Mixed resolutions {resolutions}; treating frames as independent snapshots (temporal continuity disabled)",
            advisory_only=True,
            metrics={"resolutions": [f"{w}x{h}" for w, h in resolutions]},
        ))

    seen_keys: set[tuple[Any, ...]] = set()
    for fr in frames:
        fidx = int(fr.get("frame_index", 0))
        img_name = str(fr.get("image_name", ""))
        w, h = int(fr.get("width", 0)), int(fr.get("height", 0))
        img_met = _compute_image_metrics(job_id, img_name, profile=profile) if compute_img_metrics else None
        fr_findings = validate_frame(
            fr,
            job_id=job_id,
            bone_medians=bone_medians,
            vlm_entry=vlm_by_frame.get(fidx),
            image_metrics=img_met,
        )

        # L4 per-frame resolution mismatch vs BTC 960x540
        if w > 0 and h > 0 and (w, h) != (BTC_BASE_W, BTC_BASE_H):
            diag_scale = math.hypot(w, h) / math.hypot(BTC_BASE_W, BTC_BASE_H)
            fr_findings.append(ValidationFinding(
                job_id, fidx, img_name, "", "all", "L4_CROSS_CASE",
                "RESOLUTION_MISMATCH_BTC", "info", "P17-G26",
                f"Resolution {w}x{h} differs from BTC {BTC_BASE_W}x{BTC_BASE_H} (diag_scale={diag_scale:.3f})",
                advisory_only=True,
                metrics={"width": w, "height": h, "diag_scale": round(diag_scale, 3)},
            ))

        dedup_fr: list[ValidationFinding] = []
        for fnd in fr_findings:
            dk = fnd.dedup_key()
            if dk not in seen_keys:
                seen_keys.add(dk)
                dedup_fr.append(fnd)
                all_findings.append(fnd)

        err_cnt = sum(1 for f in dedup_fr if f.severity == "error")
        warn_cnt = sum(1 for f in dedup_fr if f.severity == "warning")
        info_cnt = sum(1 for f in dedup_fr if f.severity == "info")
        coverage_rows.append({
            "job_id": job_id,
            "frame_index": fidx,
            "image_name": img_name,
            "width": w,
            "height": h,
            "skeleton_count": int(fr.get("skeleton_count", len(fr.get("skeletons") or {}))),
            "error_count": err_cnt,
            "warning_count": warn_cnt,
            "info_count": info_cnt,
            "status": "fail" if err_cnt > 0 else ("warn" if warn_cnt > 0 else "pass"),
            "snapshot_sha256": job_data.get("snapshot_sha256", ""),
        })

    # Deduplicate L4 job-level findings too
    final_findings: list[ValidationFinding] = []
    final_keys: set[tuple[Any, ...]] = set()
    for fnd in all_findings:
        dk = fnd.dedup_key()
        if dk not in final_keys:
            final_keys.add(dk)
            final_findings.append(fnd)

    return {
        "job_id": job_id,
        "validator_version": VALIDATOR_VERSION,
        "snapshot_sha256": job_data.get("snapshot_sha256", ""),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sequence_mode": "independent_snapshots" if is_independent_snapshots else "uniform_sequence",
        "resolutions": [f"{w}x{h}" for w, h in resolutions],
        "frame_count": len(frames),
        "finding_count": len(final_findings),
        "findings": [asdict(f) for f in final_findings],
        "findings_csv_rows": [f.to_csv_row() for f in final_findings],
        "coverage_rows": coverage_rows,
    }


def write_validation_outputs(
    result: dict[str, Any],
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Idempotently write VALIDATION_RESULTS.csv and FRAME_COVERAGE.csv."""
    od = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    val_path = od / "VALIDATION_RESULTS.csv"
    cov_path = od / "FRAME_COVERAGE.csv"

    val_fields = [
        "job_id", "frame_index", "image_name", "skeleton_id", "keypoint_id",
        "layer", "issue_type", "severity", "rule_id", "message",
        "advisory_only", "is_ground_truth", "auto_promoted", "metrics_json",
    ]
    cov_fields = [
        "job_id", "frame_index", "image_name", "width", "height",
        "skeleton_count", "error_count", "warning_count", "info_count",
        "status", "snapshot_sha256",
    ]
    _write_csv_rows(val_path, val_fields, result["findings_csv_rows"])
    _write_csv_rows(cov_path, cov_fields, result["coverage_rows"])
    return val_path, cov_path


def render_before_after_cases(
    job_data: dict[str, Any],
    frames_to_render: list[int] | None = None,
    corrections_path: Path | None = None,
    output_dir: Path | None = None,
    profile: str = "edufun",
) -> list[Path]:
    """Render side-by-side before/after comparison images for key proposed_corrections cases."""
    import cv2

    target_frames = frames_to_render or [93, 101, 113, 114, 116, 117]
    corr_p = Path(corrections_path) if corrections_path else (DEFAULT_OUTPUT_DIR / "proposed_corrections.csv")
    out_dir = Path(output_dir) if output_dir else (DEFAULT_OUTPUT_DIR / "evidence" / "before_after")
    out_dir.mkdir(parents=True, exist_ok=True)

    job_id = int(job_data.get("job_id", 1623))
    fdir = paths.job_dir(profile, job_id) / "frames"
    corr_rows = _read_csv_rows(corr_p)

    rendered: list[Path] = []
    for fidx in target_frames:
        fr = (job_data.get("frames") or {}).get(str(fidx))
        if not fr:
            continue
        img_name = str(fr.get("image_name", ""))
        img_path = fdir / img_name
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        f_corrs = [r for r in corr_rows if str(r.get("frame_index", "")).strip() == str(fidx)]
        bad_kps: set[int] = set()
        for r in f_corrs:
            for part in str(r.get("keypoint_id", "")).split(";"):
                if part.strip().isdigit():
                    bad_kps.add(int(part.strip()))

        before = img.copy()
        after = img.copy()
        h, w = img.shape[:2]

        for skel in (fr.get("skeletons") or {}).values():
            kpm = _kp_map(skel)
            # Draw edges
            for ia, ib in POSE17_SVG_EDGES:
                if ia in kpm and ib in kpm:
                    pa = (int(round(kpm[ia]["x"])), int(round(kpm[ia]["y"])))
                    pb = (int(round(kpm[ib]["x"])), int(round(kpm[ib]["y"])))
                    col_b = (0, 0, 255) if (ia in bad_kps or ib in bad_kps) else (0, 200, 255)
                    col_a = (0, 220, 0) if (ia in bad_kps or ib in bad_kps) else (200, 200, 0)
                    cv2.line(before, pa, pb, col_b, 2)
                    if ia not in bad_kps and ib not in bad_kps:
                        cv2.line(after, pa, pb, col_a, 2)

            for kid, kp in kpm.items():
                px, py = int(round(kp["x"])), int(round(kp["y"]))
                is_bad = kid in bad_kps
                col_b = (0, 0, 255) if is_bad else (0, 255, 255)
                cv2.circle(before, (px, py), 6 if is_bad else 4, col_b, -1)
                cv2.putText(before, str(kid), (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col_b, 1)

                if is_bad:
                    cv2.circle(after, (px, py), 8, (0, 255, 0), 2)
                    cv2.putText(after, f"{kid}->FIX", (px + 6, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                else:
                    cv2.circle(after, (px, py), 4, (255, 220, 0), -1)

        # Header banners
        cv2.rectangle(before, (0, 0), (w, 30), (20, 20, 40), -1)
        cv2.putText(before, f"BEFORE f{fidx:03d} ({img_name}) - {len(f_corrs)} issues", (10, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 255), 2)
        cv2.rectangle(after, (0, 0), (w, 30), (20, 40, 20), -1)
        rules_txt = ",".join(sorted({r.get("rule_id", "") for r in f_corrs if r.get("rule_id")}))
        cv2.putText(after, f"AFTER TARGET f{fidx:03d} [{rules_txt}]", (10, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 255, 80), 2)

        canvas = cv2.hconcat([before, after])
        out_file = out_dir / f"f{fidx:03d}_{img_name}"
        cv2.imwrite(str(out_file), canvas)
        rendered.append(out_file)
    return rendered
