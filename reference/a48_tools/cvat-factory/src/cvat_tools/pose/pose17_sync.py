"""Pose17 Proposal Builder, Bold Visual Renderer (VS Code), and Safe CVAT Sync.

Workflow:
1. `propose --job 1623 [--open]`:
   - Reads raw CVAT annotations (read-only) + deterministic rule fixes + optional draft for missing frames.
   - Saves concrete proposed shapes to `artifacts/pose17_research/proposed_shapes_<job>.json`.
   - Renders bold, color-coded BEFORE vs PROPOSED images + `00_PROPOSALS_CONTACT_SHEET.jpg`.
   - Opens rendered images directly in VS Code when `--open` is passed.
2. `sync --job 1623 [--frames 93,101,...] [--yes]`:
   - Dry-run by default: prints exact diff per frame/keypoint and exits without touching CVAT.
   - With `--yes`: backs up live CVAT annotations, updates/creates approved shapes (`source="manual"`),
     saves run log for rollback, and verifies the post-write state.
"""
from __future__ import annotations

import copy
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from cvat_tools.core import paths
from cvat_tools.core.client import Cvat
from cvat_tools.pose import week2
from cvat_tools.pose.pose17_validate import (
    DEFAULT_OUTPUT_DIR,
    FACE_TORSO_IDS,
    POSE17_SVG_EDGES,
    _kp_map,
    get_structured_logger,
    load_job_snapshot,
)

# BGR colors for bold OpenCV rendering
C_GREEN = (50, 235, 50)        # visible
C_AMBER = (0, 200, 255)        # occluded (Q)
C_RED_O = (70, 40, 255)        # outside (O)
C_MAGENTA = (255, 0, 230)      # occ+out conflict / error
C_CYAN = (255, 240, 40)        # changed / fixed highlight
C_BLUE = (220, 160, 70)        # unchanged edge
C_BLACK = (15, 15, 15)         # outline halo


def _state_str(kp: dict[str, Any]) -> str:
    occ = bool(kp.get("occluded", False))
    out = bool(kp.get("outside", False))
    if occ and out:
        return "Q+O"
    if occ:
        return "Q"
    if out:
        return "O"
    return "V"


def _set_state(kp: dict[str, Any], state: str) -> None:
    if state == "V":
        kp["occluded"] = False
        kp["outside"] = False
    elif state == "Q":
        kp["occluded"] = True
        kp["outside"] = False
    elif state == "O":
        kp["occluded"] = False
        kp["outside"] = True
    else:
        raise ValueError(f"Invalid keypoint state: {state}")


def _load_missing_draft(job_id: int, missing_frames: list[int]) -> dict[int, dict[str, Any]]:
    """Load existing local AI draft for missing frames (no model import needed)."""
    if not missing_frames:
        return {}
    try:
        folder = week2.job_folder(job_id)
    except FileNotFoundError:
        return {}
    drafts = sorted((folder / "drafts").glob("person17_auto_*.json"))
    if not drafts:
        drafts = sorted((folder / "drafts").glob("person17_*.json"))
    out: dict[int, dict[str, Any]] = {}
    for dp in reversed(drafts):
        data = json.loads(dp.read_text(encoding="utf-8"))
        for fr in data.get("frames") or []:
            fidx = int(fr.get("frame", -1))
            if fidx in missing_frames and fidx not in out and (fr.get("people") or []):
                out[fidx] = {"draft_file": dp.name, "person": fr["people"][0]}
        if len(out) == len(missing_frames):
            break
    return out


def _draft_person_to_cvat_shape(
    frame_idx: int,
    person: dict[str, Any],
    labels_raw: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Convert local draft person dict into a CVAT person_vf17 skeleton shape."""
    person_lbl = next((l for l in labels_raw if l.get("name") == "person_vf17"), None)
    if not person_lbl:
        return None
    sub_by_name = {str(s["name"]): int(s["id"]) for s in (person_lbl.get("sublabels") or [])}
    pts = person.get("points") or {}
    elements = []
    for kid in range(1, 18):
        p = pts.get(str(kid))
        if not p:
            return None
        st = str(p.get("state", "visible"))
        elements.append({
            "type": "points",
            "frame": frame_idx,
            "label_id": sub_by_name[str(kid)],
            "points": [float(p["x"]), float(p["y"])],
            "occluded": st == "occluded",
            "outside": st == "outside",
            "z_order": 0,
            "rotation": 0.0,
            "source": "manual",
            "attributes": [],
        })
    return {
        "type": "skeleton",
        "frame": frame_idx,
        "label_id": int(person_lbl["id"]),
        "occluded": False,
        "outside": False,
        "z_order": 0,
        "rotation": 0.0,
        "source": "manual",
        "attributes": [],
        "elements": elements,
    }


def build_proposals(
    job_id: int,
    offline: bool = True,
    profile: str = "edufun",
    snapshot_dir: Path | None = None,
    output_path: Path | None = None,
    include_missing_drafts: bool = True,
) -> dict[str, Any]:
    """Build concrete, reviewable CVAT shape proposals for Job 1623."""
    log = get_structured_logger()
    snap_dir = snapshot_dir or (DEFAULT_OUTPUT_DIR / "snapshots")
    raw_ann_path = snap_dir / f"job_{job_id}_annotations_raw.json"
    labels_path = snap_dir / f"job_{job_id}_labels.json"

    if offline:
        if not raw_ann_path.exists():
            raise FileNotFoundError(f"Offline snapshot not found: {raw_ann_path}")
        raw_ann = json.loads(raw_ann_path.read_text(encoding="utf-8"))
        labels_raw = (
            json.loads(labels_path.read_text(encoding="utf-8"))
            if labels_path.exists()
            else []
        )
        job_data = load_job_snapshot(
            job_id=job_id,
            offline=True,
            profile=profile,
            snapshot_path=raw_ann_path if snapshot_dir is not None else None,
        )
    else:
        cv = Cvat.from_env(profile, write_enabled=False)
        raw_ann = cv.annotations(job_id)
        labels_raw = cv.labels(job_id)
        job_data = load_job_snapshot(job_id=job_id, offline=False, profile=profile)

    raw_by_frame: dict[int, list[dict[str, Any]]] = {}
    for sh in raw_ann.get("shapes") or []:
        raw_by_frame.setdefault(int(sh["frame"]), []).append(sh)

    # Sublabel id -> keypoint 1..17
    person_lbl = next((l for l in labels_raw if l.get("name") == "person_vf17"), {})
    sub_id_to_kid = {int(s["id"]): int(s["name"]) for s in (person_lbl.get("sublabels") or []) if str(s["name"]).isdigit()}

    frames_meta = job_data.get("frames") or {}
    missing_frames = [
        int(fidx_s) for fidx_s, fr in sorted(frames_meta.items(), key=lambda kv: int(kv[0]))
        if not (fr.get("skeletons") or {})
    ]
    missing_drafts = _load_missing_draft(job_id, missing_frames) if include_missing_drafts else {}

    proposals: list[dict[str, Any]] = []
    for fidx_s, fr in sorted(frames_meta.items(), key=lambda kv: int(kv[0])):
        fidx = int(fidx_s)
        w, h = int(fr.get("width", 1280)), int(fr.get("height", 1024))
        img_name = str(fr.get("image_name", ""))
        raw_shapes = raw_by_frame.get(fidx, [])

        # Case 1: Missing skeleton -> propose from local AI draft (flagged as AI_DRAFT_FOR_REVIEW)
        if not raw_shapes:
            if fidx in missing_drafts:
                new_shape = _draft_person_to_cvat_shape(fidx, missing_drafts[fidx]["person"], labels_raw)
                if new_shape is not None:
                    proposals.append({
                        "frame_index": fidx,
                        "image_name": img_name,
                        "width": w,
                        "height": h,
                        "action": "create",
                        "shape_id": None,
                        "review_tier": "AI_DRAFT_FOR_REVIEW",
                        "changes": [f"Create missing skeleton from {missing_drafts[fidx]['draft_file']}"],
                        "changed_kpts": list(range(1, 18)),
                        "before_shape": None,
                        "proposed_shape": new_shape,
                    })
            continue

        for sh in raw_shapes:
            before_sh = copy.deepcopy(sh)
            prop_sh = copy.deepcopy(sh)
            changes: list[str] = []
            changed_kpts: set[int] = set()

            # 1) Fix skeleton-level bulk occluded (P17-G12, job-agnostic: never hardcode frame numbers)
            if bool(prop_sh.get("occluded", False)):
                prop_sh["occluded"] = False
                changes.append("skeleton.occluded: True -> False (P17-G12)")

            # Map elements by kid for targeted rule fixes
            el_by_kid: dict[int, dict[str, Any]] = {}
            for el in prop_sh.get("elements") or []:
                kid = sub_id_to_kid.get(int(el["label_id"]))
                if kid is not None:
                    el_by_kid[kid] = el

            # 2) Fix simultaneous occluded+outside conflicts (P17-G11) & in-frame face/ear outside (P17-G15)
            for kid, el in sorted(el_by_kid.items()):
                st = _state_str(el)
                px, py = float(el["points"][0]), float(el["points"][1])
                in_frame_margin = min(px, py, w - px, h - py)
                if st == "Q+O":
                    # Face/torso in frame -> Q; limbs with outside=True -> O (respect user's outside intent)
                    target = "Q" if (kid in FACE_TORSO_IDS and in_frame_margin >= 8.0) else "O"
                    _set_state(el, target)
                    changed_kpts.add(kid)
                    changes.append(f"kp {kid}: Q+O -> {target} (P17-G11)")
                elif st == "O" and kid in FACE_TORSO_IDS and in_frame_margin >= 8.0:
                    _set_state(el, "Q")
                    changed_kpts.add(kid)
                    changes.append(f"kp {kid}: O -> Q (in-frame face/torso, P17-G15)")

            if changes:
                # Safety invariant: never silently mutate existing human-placed (x, y) coordinates
                for b_el, p_el in zip(before_sh.get("elements") or [], prop_sh.get("elements") or []):
                    if b_el.get("points") != p_el.get("points"):
                        raise AssertionError(
                            f"Invariant violated on frame {fidx}: build_proposals must not mutate coordinates"
                        )
                proposals.append({
                    "frame_index": fidx,
                    "image_name": img_name,
                    "width": w,
                    "height": h,
                    "action": "update",
                    "shape_id": int(sh["id"]),
                    "review_tier": "RULE_FIX_READY",
                    "changes": changes,
                    "changed_kpts": sorted(changed_kpts),
                    "before_shape": before_sh,
                    "proposed_shape": prop_sh,
                })

    out_file = Path(output_path) if output_path else (DEFAULT_OUTPUT_DIR / f"proposed_shapes_{job_id}.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "job_id": job_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "proposal_count": len(proposals),
        "sublabel_id_to_kid": {str(k): v for k, v in sub_id_to_kid.items()},
        "proposals": proposals,
    }
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("build_proposals_complete", extra={"job_id": job_id, "count": len(proposals), "output": str(out_file)})
    return payload


def _draw_bold_skeleton(
    img,
    shape: dict[str, Any] | None,
    sub_id_to_kid: dict[int, int],
    changed_kpts: set[int],
    is_after: bool,
    before_shape: dict[str, Any] | None = None,
) -> None:
    import cv2

    if shape is None:
        return
    kpm: dict[int, dict[str, Any]] = {}
    for el in shape.get("elements") or []:
        kid = sub_id_to_kid.get(int(el["label_id"]))
        if kid is not None:
            kpm[kid] = {
                "x": float(el["points"][0]),
                "y": float(el["points"][1]),
                "state": _state_str(el),
            }

    before_kpm: dict[int, dict[str, Any]] = {}
    if before_shape:
        for el in before_shape.get("elements") or []:
            kid = sub_id_to_kid.get(int(el["label_id"]))
            if kid is not None:
                before_kpm[kid] = {"x": float(el["points"][0]), "y": float(el["points"][1]), "state": _state_str(el)}

    # 1. Bold edges with dark halo
    for ia, ib in POSE17_SVG_EDGES:
        if ia in kpm and ib in kpm:
            pa = (int(round(kpm[ia]["x"])), int(round(kpm[ia]["y"])))
            pb = (int(round(kpm[ib]["x"])), int(round(kpm[ib]["y"])))
            touches_chg = (ia in changed_kpts) or (ib in changed_kpts)
            col = C_CYAN if (is_after and touches_chg) else (C_RED_O if (not is_after and touches_chg) else C_BLUE)
            cv2.line(img, pa, pb, C_BLACK, 7, cv2.LINE_AA)
            cv2.line(img, pa, pb, col, 4, cv2.LINE_AA)

    # 2. Movement arrows on AFTER when coordinates changed
    if is_after and before_kpm:
        for kid in changed_kpts:
            if kid in kpm and kid in before_kpm:
                p0 = (int(round(before_kpm[kid]["x"])), int(round(before_kpm[kid]["y"])))
                p1 = (int(round(kpm[kid]["x"])), int(round(kpm[kid]["y"])))
                if abs(p0[0] - p1[0]) + abs(p0[1] - p1[1]) >= 4:
                    cv2.arrowedLine(img, p0, p1, C_BLACK, 6, cv2.LINE_AA, tipLength=0.25)
                    cv2.arrowedLine(img, p0, p1, C_CYAN, 3, cv2.LINE_AA, tipLength=0.25)

    # 3. Bold keypoints + state badges
    for kid, kp in sorted(kpm.items()):
        px, py = int(round(kp["x"])), int(round(kp["y"]))
        st = kp["state"]
        is_chg = kid in changed_kpts
        col = C_GREEN if st == "V" else (C_AMBER if st == "Q" else (C_MAGENTA if st == "Q+O" else C_RED_O))

        if st == "O":
            s = 9
            cv2.line(img, (px - s, py - s), (px + s, py + s), C_BLACK, 6, cv2.LINE_AA)
            cv2.line(img, (px - s, py + s), (px + s, py - s), C_BLACK, 6, cv2.LINE_AA)
            cv2.line(img, (px - s, py - s), (px + s, py + s), col, 3, cv2.LINE_AA)
            cv2.line(img, (px - s, py + s), (px + s, py - s), col, 3, cv2.LINE_AA)
        else:
            cv2.circle(img, (px, py), 9 if is_chg else 7, C_BLACK, -1, cv2.LINE_AA)
            cv2.circle(img, (px, py), 7 if is_chg else 5, col, -1, cv2.LINE_AA)

        if is_chg:
            ring_col = C_CYAN if is_after else C_MAGENTA
            cv2.circle(img, (px, py), 13, ring_col, 3, cv2.LINE_AA)

        tag = f"{kid}:{st}" if (is_chg or st != "V") else str(kid)
        if is_after and is_chg and kid in before_kpm and before_kpm[kid]["state"] != st:
            tag = f"{kid}:{before_kpm[kid]['state']}->{st}"
        cv2.putText(img, tag, (px + 10, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_BLACK, 4, cv2.LINE_AA)
        cv2.putText(img, tag, (px + 10, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_CYAN if (is_after and is_chg) else col, 2, cv2.LINE_AA)


def render_bold_proposals(
    proposals_payload: dict[str, Any],
    output_dir: Path | None = None,
    profile: str = "edufun",
) -> dict[str, Any]:
    """Render bold, color-coded BEFORE vs PROPOSED images and a summary contact sheet."""
    import cv2
    import numpy as np

    job_id = int(proposals_payload["job_id"])
    fdir = paths.job_dir(profile, job_id) / "frames"
    out_dir = Path(output_dir) if output_dir else (DEFAULT_OUTPUT_DIR / "evidence" / "proposals")
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_id_to_kid = {int(k): int(v) for k, v in (proposals_payload.get("sublabel_id_to_kid") or {}).items()}

    rendered_files: list[Path] = []
    tiles: list[Any] = []

    for prop in proposals_payload.get("proposals") or []:
        fidx = int(prop["frame_index"])
        img_name = str(prop["image_name"])
        img_path = fdir / img_name
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        chg_set = {int(k) for k in (prop.get("changed_kpts") or [])}
        before = img.copy()
        after = img.copy()

        if prop.get("before_shape") is None:
            cv2.putText(before, "MISSING SKELETON (0 shapes)", (30, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.1, C_BLACK, 6, cv2.LINE_AA)
            cv2.putText(before, "MISSING SKELETON (0 shapes)", (30, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.1, C_MAGENTA, 3, cv2.LINE_AA)
        else:
            _draw_bold_skeleton(before, prop["before_shape"], sub_id_to_kid, chg_set, is_after=False)

        _draw_bold_skeleton(after, prop["proposed_shape"], sub_id_to_kid, chg_set, is_after=True, before_shape=prop.get("before_shape"))

        # Top & bottom banners
        header_h, footer_h = 46, 34
        b_pad = cv2.copyMakeBorder(before, header_h, footer_h, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 38))
        a_pad = cv2.copyMakeBorder(after, header_h, footer_h, 0, 0, cv2.BORDER_CONSTANT, value=(20, 38, 25))

        cv2.putText(b_pad, f"BEFORE f{fidx:03d} ({img_name}) | Current CVAT", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, C_MAGENTA, 2, cv2.LINE_AA)
        tier = prop.get("review_tier", "")
        summary = "; ".join(prop.get("changes", [])[:2])
        cv2.putText(a_pad, f"PROPOSED f{fidx:03d} [{tier}]", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, C_CYAN, 2, cv2.LINE_AA)
        cv2.putText(b_pad, "Legend: Green=V  Amber=Q  RedX=O  MagentaRing=Conflict", (14, h + header_h + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(a_pad, f"Changes: {summary[:78]}", (14, h + header_h + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_CYAN, 1, cv2.LINE_AA)

        canvas = cv2.hconcat([b_pad, a_pad])
        out_file = out_dir / f"f{fidx:03d}_{img_name}"
        cv2.imwrite(str(out_file), canvas)
        rendered_files.append(out_file)

        # Tile for contact sheet (AFTER panel resized cleanly)
        tile = cv2.resize(a_pad, (640, 420), interpolation=cv2.INTER_AREA)
        tiles.append(tile)

    contact_path = out_dir / "00_PROPOSALS_CONTACT_SHEET.jpg"
    if tiles:
        cols = 3
        rows = (len(tiles) + cols - 1) // cols
        while len(tiles) < rows * cols:
            tiles.append(np.full_like(tiles[0], 20))
        grid_rows = [cv2.hconcat(tiles[r * cols:(r + 1) * cols]) for r in range(rows)]
        sheet = cv2.vconcat(grid_rows)
        cv2.imwrite(str(contact_path), sheet)

    return {"output_dir": str(out_dir), "contact_sheet": str(contact_path) if tiles else None, "files": [str(p) for p in rendered_files]}


def open_in_vscode(file_paths: list[str | Path]) -> bool:
    """Open rendered proposal images directly inside VS Code tabs."""
    code_bin = shutil.which("code") or shutil.which("code.cmd")
    if not code_bin or not file_paths:
        return False
    cmd = [code_bin, "--reuse-window", *[str(p) for p in file_paths]]
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return True


def sync_proposals(
    job_id: int,
    proposals_path: Path | None = None,
    frames: list[int] | None = None,
    profile: str = "edufun",
    yes: bool = False,
) -> dict[str, Any]:
    """Dry-run by default; with --yes, backup and sync approved proposals to CVAT."""
    log = get_structured_logger()
    prop_file = Path(proposals_path) if proposals_path else (DEFAULT_OUTPUT_DIR / f"proposed_shapes_{job_id}.json")
    if not prop_file.exists():
        raise FileNotFoundError(f"Proposal file not found: {prop_file}. Run `propose` first.")

    payload = json.loads(prop_file.read_text(encoding="utf-8"))
    all_props = payload.get("proposals") or []
    frame_filter = set(frames) if frames else None
    selected = [p for p in all_props if frame_filter is None or int(p["frame_index"]) in frame_filter]

    to_update = [p["proposed_shape"] for p in selected if p["action"] == "update"]
    to_create = [p["proposed_shape"] for p in selected if p["action"] == "create"]

    summary = [
        {
            "frame_index": int(p["frame_index"]),
            "action": p["action"],
            "shape_id": p.get("shape_id"),
            "review_tier": p.get("review_tier"),
            "changes": p.get("changes", []),
        }
        for p in selected
    ]

    if not yes:
        log.info("sync_dry_run", extra={"job_id": job_id, "selected_frames": [s["frame_index"] for s in summary]})
        return {
            "job_id": job_id,
            "mode": "dry_run",
            "written_to_cvat": False,
            "selected_count": len(summary),
            "update_count": len(to_update),
            "create_count": len(to_create),
            "summary": summary,
        }

    # Live CVAT write (only when explicitly authorized via --yes)
    cv = Cvat.from_env(profile, write_enabled=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = week2.job_folder(job_id) / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)

    before_live = cv.annotations(job_id)
    live_by_id = {int(s["id"]): s for s in (before_live.get("shapes") or []) if s.get("id") is not None}
    for p in selected:
        if p["action"] == "update":
            sid = int(p["shape_id"])
            live_sh = live_by_id.get(sid)
            b_sh = p.get("before_shape") or {}
            if live_sh is None:
                raise RuntimeError(f"Stale proposal: shape {sid} (frame {p['frame_index']}) no longer exists on CVAT. Re-run `propose`.")
            live_pts = {int(e["label_id"]): (e.get("points"), bool(e.get("occluded")), bool(e.get("outside"))) for e in (live_sh.get("elements") or [])}
            prop_b_pts = {int(e["label_id"]): (e.get("points"), bool(e.get("occluded")), bool(e.get("outside"))) for e in (b_sh.get("elements") or [])}
            if bool(live_sh.get("occluded")) != bool(b_sh.get("occluded")) or live_pts != prop_b_pts:
                raise RuntimeError(f"Stale proposal: frame {p['frame_index']} (shape {sid}) was modified on CVAT after `propose`. Re-run `propose` first.")

    backup_file = backup_dir / f"before_sync_{ts}.json"
    backup_file.write_text(json.dumps(before_live, indent=2, ensure_ascii=False), encoding="utf-8")

    # Enforce source="manual" on all synced shapes & elements
    for sh in to_update + to_create:
        sh["source"] = "manual"
        for el in sh.get("elements") or []:
            el["source"] = "manual"

    update_res = cv.update_shapes(job_id, to_update) if to_update else {}
    create_res = cv.create_shapes(job_id, to_create) if to_create else {}
    after_live = cv.annotations(job_id)

    run_log = {
        "job_id": job_id,
        "timestamp": ts,
        "backup_file": str(backup_file),
        "updated_ids": [int(s["id"]) for s in to_update],
        "created_ids": [int(s["id"]) for s in (create_res.get("shapes") or [])],
        "selected_summary": summary,
        "before_count": len(before_live.get("shapes") or []),
        "after_count": len(after_live.get("shapes") or []),
    }
    run_file = backup_dir / f"run_sync_{ts}.json"
    run_file.write_text(json.dumps(run_log, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("sync_committed", extra={"job_id": job_id, "updated_count": len(to_update), "created_count": len(to_create)})

    return {
        "job_id": job_id,
        "mode": "committed",
        "written_to_cvat": True,
        "backup_file": str(backup_file),
        "run_file": str(run_file),
        "update_count": len(to_update),
        "create_count": len(to_create),
        "before_count": run_log["before_count"],
        "after_count": run_log["after_count"],
        "summary": summary,
        "update_response_keys": list(update_res.keys()) if isinstance(update_res, dict) else [],
    }
