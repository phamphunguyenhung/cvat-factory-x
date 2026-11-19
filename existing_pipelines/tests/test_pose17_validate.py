"""Unit & CLI smoke tests for Pose17 5-layer validator and Guideline/Case Engine."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from cvat_tools.core import paths
from cvat_tools.pose.pose17_validate import (
    RULE_CSV_FIELDS,
    cases_for_rule,
    load_job_snapshot,
    lookup_rule,
    open_case,
    propose_rule,
    validate_frame,
    validate_job,
    write_validation_outputs,
)


def _make_upright_kps(
    overrides: dict[int, dict] | None = None,
) -> list[dict]:
    """Build a valid upright 17-keypoint person_vf17 skeleton inside 1280x1024."""
    base = {
        1: (640.0, 200.0),  # nose
        2: (660.0, 185.0),  # eye_R (x > eye_L)
        3: (620.0, 185.0),  # eye_L
        4: (685.0, 195.0),  # ear_R
        5: (595.0, 195.0),  # ear_L
        6: (720.0, 300.0),  # shoulder_R
        7: (560.0, 300.0),  # shoulder_L
        8: (740.0, 420.0),  # elbow_R
        9: (540.0, 420.0),  # elbow_L
        10: (730.0, 520.0), # wrist_R
        11: (550.0, 520.0), # wrist_L
        12: (700.0, 550.0), # hip_R
        13: (580.0, 550.0), # hip_L
        14: (710.0, 720.0), # knee_R
        15: (570.0, 720.0), # knee_L
        16: (715.0, 900.0), # ankle_R
        17: (565.0, 900.0), # ankle_L
    }
    ov = overrides or {}
    out = []
    for i in range(1, 18):
        x, y = base[i]
        item = {
            "sublabel_name": str(i),
            "x": x,
            "y": y,
            "occluded": False,
            "outside": False,
            "hidden": False,
        }
        if i in ov:
            item.update(ov[i])
        out.append(item)
    return out


def _make_frame(
    frame_index: int = 90,
    width: int = 1280,
    height: int = 1024,
    kps: list[dict] | None = None,
    skeleton_occluded: bool = False,
    skeleton_hidden: bool = False,
    empty: bool = False,
) -> dict:
    if empty:
        return {
            "frame_index": frame_index,
            "image_name": f"f{frame_index:03d}.jpg",
            "width": width,
            "height": height,
            "skeleton_count": 0,
            "skeletons": {},
        }
    return {
        "frame_index": frame_index,
        "image_name": f"f{frame_index:03d}.jpg",
        "width": width,
        "height": height,
        "skeleton_count": 1,
        "skeletons": {
            "1001": {
                "skeleton_id": 1001,
                "label_name": "person_vf17",
                "type": "skeleton",
                "occluded": skeleton_occluded,
                "outside": False,
                "hidden": skeleton_hidden,
                "keypoints": kps if kps is not None else _make_upright_kps(),
            }
        },
    }


def test_l1_structural_checks():
    # Missing skeleton
    f_empty = _make_frame(empty=True)
    res_empty = validate_frame(f_empty, job_id=1)
    assert any(f.issue_type == "MISSING_SKELETON" and f.layer == "L1_STRUCTURAL" for f in res_empty)

    # Missing keypoint
    kps_missing = [kp for kp in _make_upright_kps() if kp["sublabel_name"] != "17"]
    res_miss = validate_frame(_make_frame(kps=kps_missing), job_id=1)
    assert any(f.issue_type == "INVALID_KEYPOINT_SCHEMA" and f.layer == "L1_STRUCTURAL" for f in res_miss)

    # Forbidden hidden flag
    kps_hidden = _make_upright_kps({2: {"hidden": True}})
    res_hid = validate_frame(_make_frame(kps=kps_hidden), job_id=1)
    assert any(f.issue_type == "FORBIDDEN_HIDDEN_FLAG" and f.rule_id == "P17-G10" for f in res_hid)


def test_l2_geometric_oob_overlap_and_lr():
    # Out-of-bounds visible coord
    kps_oob = _make_upright_kps({10: {"x": 1350.0, "y": 500.0}})
    res_oob = validate_frame(_make_frame(kps=kps_oob), job_id=1)
    assert any(f.issue_type == "OUT_OF_BOUNDS_COORD" and f.rule_id == "P17-G23" for f in res_oob)

    # < 3px overlap (2 and 4 at 0.5 px)
    kps_ov = _make_upright_kps({4: {"x": 660.4, "y": 185.2}})
    res_ov = validate_frame(_make_frame(kps=kps_ov), job_id=1)
    assert any(f.issue_type == "KEYPOINT_OVERLAP" and f.rule_id == "P17-G22" for f in res_ov)

    # Upright R/L inversion (swap shoulder 6 and 7)
    kps_inv = _make_upright_kps({6: {"x": 560.0}, 7: {"x": 720.0}})
    res_inv = validate_frame(_make_frame(kps=kps_inv), job_id=1)
    assert any(f.issue_type == "LEFT_RIGHT_INVERSION" and f.rule_id == "P17-G06" for f in res_inv)


def test_l2_horizontal_torso_exemption_vs_real_wrist_swap():
    # Horizontal person: head on right (x=900, y=300), hips on left (x=400, y=300)
    # Torso vector v_torso = (500, 0) -> |vx| > |vy| (horizontal=True).
    # Body-aligned u_right = (-vy, vx)/|v| = (0, 1) (pointing down in image).
    # So R side (even) has larger y than L side (odd), even though x_R == x_L (or x_R < x_L)!
    # Convert tuple overrides to dicts
    horiz_kps = _make_upright_kps({k: {"x": v[0], "y": v[1]} for k, v in {
        1: (920.0, 300.0),
        2: (940.0, 315.0), 3: (940.0, 285.0),
        4: (900.0, 335.0), 5: (900.0, 265.0),
        6: (800.0, 380.0), 7: (800.0, 220.0),
        8: (680.0, 390.0), 9: (680.0, 210.0),
        10: (580.0, 385.0), 11: (580.0, 215.0),
        12: (500.0, 360.0), 13: (500.0, 240.0),
        14: (330.0, 370.0), 15: (330.0, 230.0),
        16: (180.0, 375.0), 17: (180.0, 225.0),
    }.items()})
    res_ok = validate_frame(_make_frame(kps=horiz_kps), job_id=1)
    assert not any(f.issue_type in ("LEFT_RIGHT_INVERSION", "BODY_ALIGNED_LR_INVERSION") for f in res_ok)

    # Now swap wrists 10 and 11 along body-aligned axis -> triggers BODY_ALIGNED_LR_INVERSION
    horiz_swap = _make_upright_kps({k: {"x": v[0], "y": v[1]} for k, v in {
        1: (920.0, 300.0),
        2: (940.0, 315.0), 3: (940.0, 285.0),
        4: (900.0, 335.0), 5: (900.0, 265.0),
        6: (800.0, 380.0), 7: (800.0, 220.0),
        8: (680.0, 390.0), 9: (680.0, 210.0),
        10: (580.0, 215.0), 11: (580.0, 385.0),  # real wrist swap!
        12: (500.0, 360.0), 13: (500.0, 240.0),
        14: (330.0, 370.0), 15: (330.0, 230.0),
        16: (180.0, 375.0), 17: (180.0, 225.0),
    }.items()})
    res_swap = validate_frame(_make_frame(kps=horiz_swap), job_id=1)
    assert any(f.issue_type == "BODY_ALIGNED_LR_INVERSION" and f.rule_id == "P17-G08" for f in res_swap)


def test_l3_semantic_rules():
    kps = _make_upright_kps({
        16: {"occluded": True, "outside": True},   # occ + out conflict
        2: {"outside": True},                      # in-frame eye marked outside
    })
    res = validate_frame(_make_frame(kps=kps, skeleton_occluded=True), job_id=1)
    types = {f.issue_type for f in res}
    assert "VISIBILITY_CONFLICT_OCC_OUT" in types
    assert "IN_FRAME_FACE_TORSO_OUTSIDE" in types
    assert "SKELETON_BULK_OCCLUDED" in types


def test_l4_cross_case_mixed_resolutions_no_temporal_assumption():
    # Two frames with very different resolutions and shifted positions should NOT trigger temporal jump errors
    f1 = _make_frame(frame_index=90, width=1280, height=1024)
    f2 = _make_frame(frame_index=91, width=512, height=424, kps=_make_upright_kps({
        i: {"x": x * 0.4, "y": y * 0.4}
        for i, (x, y) in {
            1: (640, 200), 2: (660, 185), 3: (620, 185), 4: (685, 195), 5: (595, 195),
            6: (720, 300), 7: (560, 300), 8: (740, 420), 9: (540, 420),
            10: (730, 520), 11: (550, 520), 12: (700, 550), 13: (580, 550),
            14: (710, 720), 15: (570, 720), 16: (715, 900), 17: (565, 900),
        }.items()
    }))
    job_data = {"job_id": 999, "snapshot_sha256": "abc", "frames": {"90": f1, "91": f2}}
    res = validate_job(job_data, compute_img_metrics=False)
    assert res["sequence_mode"] == "independent_snapshots"
    assert not any("TEMPORAL" in f["issue_type"] for f in res["findings"])
    assert any(f["issue_type"] == "INDEPENDENT_SNAPSHOT_COLLECTION" for f in res["findings"])


def test_l5_vlm_never_auto_promoted_to_ground_truth():
    fr = _make_frame(frame_index=91)
    vlm_entry = {"job": 1, "frame": 91, "status": "refine", "issues": [{"keypoint": "4", "problem": "hair"}]}
    res = validate_frame(fr, job_id=1, vlm_entry=vlm_entry, image_metrics={"mean_luma": 15.0, "lap_var": 10.0})
    l5 = [f for f in res if f.layer == "L5_EVIDENCE_QA"]
    assert len(l5) == 3
    for f in l5:
        assert f.advisory_only is True
        assert f.is_ground_truth is False
        assert f.auto_promoted is False


def test_rule_and_case_registry_idempotency(tmp_path: Path):
    # Real lookup_rule & cases_for_rule & open_case
    r11 = lookup_rule("P17-G11")
    assert r11 is not None and r11["status"] == "approved"
    cases = cases_for_rule("P17-G11")
    assert any(c["case_id"] == "CASE-1623-F093" for c in cases)
    c93 = open_case("CASE-1623-F093")
    assert c93 is not None and c93["evidence_exists"] is True

    # Idempotent propose_rule on temp CSV preserving approved rules
    tmp_rules = tmp_path / "rules.csv"
    with tmp_rules.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RULE_CSV_FIELDS)
        w.writeheader()
        w.writerow(r11)

    # Attempt to overwrite approved rule -> preserved!
    out1 = propose_rule({"rule_id": "P17-G11", "condition": "hacked"}, rules_path=tmp_rules)
    assert out1["action"] == "preserved_approved"
    assert lookup_rule("P17-G11", rules_path=tmp_rules)["condition"] == r11["condition"]

    # Append new proposed rule twice -> idempotent, no duplicates
    new_rule = {"rule_id": "P17-TEST-99", "scope": "general", "category": "QA", "condition": "c1"}
    out2 = propose_rule(new_rule, rules_path=tmp_rules)
    out3 = propose_rule(new_rule, rules_path=tmp_rules)
    assert out2["action"] == "appended_proposed"
    assert out3["action"] == "unchanged"
    with tmp_rules.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2


def test_idempotent_validation_output_and_cli_smoke(tmp_path: Path):
    snap = load_job_snapshot(1623, offline=True)
    res1 = validate_job(snap, profile="edufun")
    res2 = validate_job(snap, profile="edufun")
    assert res1["finding_count"] == res2["finding_count"]

    p1, _ = write_validation_outputs(res1, output_dir=tmp_path)
    p2, _ = write_validation_outputs(res2, output_dir=tmp_path)
    assert p1.read_text(encoding="utf-8") == p2.read_text(encoding="utf-8")

    # CLI smoke tests via subprocess
    for cmd in (
        [sys.executable, "-m", "tools.pose17", "validate", "--job", "1623", "--offline", "--out-dir", str(tmp_path), "--json"],
        [sys.executable, "-m", "tools.pose17", "lookup-rule", "--rule-id", "P17-G11", "--json"],
        [sys.executable, "-m", "tools.pose17", "open-case", "--case-id", "CASE-1623-F093", "--json"],
    ):
        proc = subprocess.run(cmd, cwd=str(paths.WORKSPACE), capture_output=True, text=True, check=True)
        assert json.loads(proc.stdout)


def test_propose_render_and_sync_dry_run_and_mocked_commit(tmp_path: Path, monkeypatch):
    import pytest
    from cvat_tools.pose.pose17_sync import build_proposals, render_bold_proposals, sync_proposals

    prop_file = tmp_path / "proposed_shapes_1623.json"
    payload = build_proposals(job_id=1623, offline=True, output_path=prop_file)
    assert payload["proposal_count"] >= 15
    assert prop_file.exists()

    # Render subset for speed in unit test
    sub_payload = dict(payload)
    sub_payload["proposals"] = payload["proposals"][:2]
    rendered = render_bold_proposals(sub_payload, output_dir=tmp_path / "vis")
    assert len(rendered["files"]) == 2
    assert Path(rendered["contact_sheet"]).exists()
    for f in rendered["files"]:
        assert Path(f).exists()

    # Dry-run default never writes to CVAT
    dry = sync_proposals(job_id=1623, proposals_path=prop_file, frames=[93, 103], yes=False)
    assert dry["mode"] == "dry_run"
    assert dry["written_to_cvat"] is False
    assert dry["update_count"] == 2

    # Mocked --yes commit verifies backup + update_shapes + source="manual"
    calls = {"update": [], "create": []}
    prop_data = json.loads(prop_file.read_text(encoding="utf-8"))
    matching_live = [p["before_shape"] for p in prop_data["proposals"] if p["frame_index"] in {93, 103}]

    class _FakeCvat:
        def annotations(self, jid):
            return {"shapes": matching_live}

        def update_shapes(self, jid, shapes):
            calls["update"].extend(shapes)
            return {"shapes": shapes}

        def create_shapes(self, jid, shapes):
            calls["create"].extend(shapes)
            return {"shapes": shapes}

    monkeypatch.setattr("cvat_tools.pose.pose17_sync.Cvat.from_env", lambda *a, **kw: _FakeCvat())
    monkeypatch.setattr("cvat_tools.pose.pose17_sync.week2.job_folder", lambda jid: tmp_path)

    res = sync_proposals(job_id=1623, proposals_path=prop_file, frames=[93, 103], yes=True)
    assert res["mode"] == "committed"
    assert res["written_to_cvat"] is True
    assert res["update_count"] == 2
    assert len(calls["update"]) == 2
    assert all(s["source"] == "manual" for s in calls["update"])
    assert Path(res["backup_file"]).exists()
    assert Path(res["run_file"]).exists()

    # Invariant: build_proposals NEVER mutates existing (x, y) coordinates
    for p in prop_data["proposals"]:
        if p["action"] == "update":
            for b_el, p_el in zip(p["before_shape"]["elements"], p["proposed_shape"]["elements"]):
                assert b_el["points"] == p_el["points"]

    # Staleness guard: refuse to overwrite if live CVAT changed after `propose`
    class _StaleLiveCvat(_FakeCvat):
        def annotations(self, jid):
            return {"shapes": [{"id": 1320580, "frame": 93, "occluded": False, "elements": []}]}

    monkeypatch.setattr("cvat_tools.pose.pose17_sync.Cvat.from_env", lambda *a, **kw: _StaleLiveCvat())
    with pytest.raises(RuntimeError, match="Stale proposal"):
        sync_proposals(job_id=1623, proposals_path=prop_file, frames=[93], yes=True)
