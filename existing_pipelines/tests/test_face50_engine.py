from __future__ import annotations

import json
from pathlib import Path

import pytest

from cvat_tools.pose.face50_engine import (
    FACE50_GROUPS,
    align_face50_clahe_fan,
    build_face50_proposals,
    render_face50_proposals,
    sync_face50_eligible,
    validate_face50_frame,
)


def _make_clean_face50_shapes(frame: int = 0) -> tuple[list[dict], dict[int, int], dict[int, str]]:
    """Build a valid synthetic 50-point Face50 frame."""
    coords: dict[int, tuple[float, float]] = {}
    # Brows: left 0..4, right 5..9
    for i, k in enumerate(range(0, 5)):
        coords[k] = (140.0 + i * 15.0, 140.0 - (2 - abs(i - 2)) * 2.0)
    for i, k in enumerate(range(5, 10)):
        coords[k] = (240.0 + i * 15.0, 140.0 - (2 - abs(i - 2)) * 2.0)
    # Nose bridge 10..13
    for i, k in enumerate(range(10, 14)):
        coords[k] = (220.0, 165.0 + i * 15.0)
    # Left eye 14..21 (15,16,17 upper; 19,20,21 lower)
    eye_l = [(145, 175), (158, 168), (170, 166), (182, 168), (195, 175), (182, 182), (170, 184), (158, 182)]
    for k, xy in zip(range(14, 22), eye_l):
        coords[k] = (float(xy[0]), float(xy[1]))
    # Right eye 22..29 (23,24,25 upper; 27,28,29 lower)
    eye_r = [(245, 175), (258, 168), (270, 166), (282, 168), (295, 175), (282, 182), (270, 184), (258, 182)]
    for k, xy in zip(range(22, 30), eye_r):
        coords[k] = (float(xy[0]), float(xy[1]))
    # Outer lip 30..41 (clean convex-ish ring around (220, 260))
    outer = [
        (185, 260), (197, 250), (210, 246), (220, 248), (230, 246), (243, 250),
        (255, 260), (243, 272), (230, 276), (220, 278), (210, 276), (197, 272),
    ]
    for k, xy in zip(range(30, 42), outer):
        coords[k] = (float(xy[0]), float(xy[1]))
    # Inner lip 42..49 (strictly inside outer lip)
    inner = [
        (195, 260), (208, 254), (220, 253), (232, 254),
        (245, 260), (232, 267), (220, 268), (208, 267),
    ]
    for k, xy in zip(range(42, 50), inner):
        coords[k] = (float(xy[0]), float(xy[1]))

    sub_id_to_kp = {1000 + k: k for k in range(50)}
    lbl_id_to_name = {100 + idx: name for idx, name in enumerate(FACE50_GROUPS)}
    shapes = []
    for idx, (name, kp_range) in enumerate(FACE50_GROUPS.items()):
        shapes.append({
            "id": 500 + idx,
            "frame": frame,
            "label_id": 100 + idx,
            "type": "skeleton",
            "occluded": False,
            "elements": [
                {
                    "id": 2000 + k,
                    "label_id": 1000 + k,
                    "type": "points",
                    "points": [coords[k][0], coords[k][1]],
                    "occluded": False,
                    "outside": False,
                }
                for k in kp_range
            ],
        })
    return shapes, sub_id_to_kp, lbl_id_to_name


def test_face50_rotation_aware_and_surgical_proposals(tmp_path: Path):
    prop_file = tmp_path / "prop_1975.json"
    payload = build_face50_proposals(job_id=1975, offline=True, output_path=prop_file)
    assert prop_file.exists()
    assert len(payload["validation_summary"]) == 30

    # f092 is rotated ~90 deg; rotation-aware validator must NOT flag lip inversion
    f92 = next(r for r in payload["validation_summary"] if r["frame_index"] == 92)
    assert "LIP_INVERSION" not in f92["codes"]

    # Missing frames 102, 106 and extreme profiles 100, 118 must stay in review_queue
    rq_frames = {r["frame_index"] for r in payload["review_queue"]}
    assert {100, 102, 106, 116, 118} <= rq_frames

    # Eligible proposals must have 0 post-fix inner-lip errors and preserve untouched points
    ren = render_face50_proposals(payload, output_dir=tmp_path / "ba")
    assert Path(ren["contact_sheet"]).exists()


def test_face50_sync_dry_run_and_staleness_guard(tmp_path: Path, monkeypatch):
    prop_file = tmp_path / "prop_1975.json"
    payload = build_face50_proposals(job_id=1975, offline=True, output_path=prop_file)

    dry = sync_face50_eligible(job_id=1975, proposals_path=prop_file, yes=False)
    assert dry["mode"] == "dry_run"
    assert dry["written_to_cvat"] is False

    # Staleness guard triggers if live CVAT changed
    class _StaleCvat:
        def annotations(self, jid):
            return {"shapes": []}

    monkeypatch.setattr("cvat_tools.pose.face50_engine.Cvat.from_env", lambda *a, **kw: _StaleCvat())
    with pytest.raises(RuntimeError, match="Stale proposal"):
        sync_face50_eligible(job_id=1975, proposals_path=prop_file, yes=True)


def test_face50_validator_blind_spots():
    # Baseline clean frame has 0 findings
    shapes, sub_map, lbl_map = _make_clean_face50_shapes(frame=10)
    res_clean = validate_face50_frame(10, shapes, sub_map, lbl_map, 640, 480)
    assert res_clean["findings"] == []

    # 1) Sub-pixel inner-lip escape (-0.15 px < -0.05 px threshold)
    shapes_lip, sub_map, lbl_map = _make_clean_face50_shapes(frame=11)
    for sh in shapes_lip:
        for el in sh["elements"]:
            if sub_map[el["label_id"]] == 42:
                el["points"] = [184.85, 260.0]  # 0.15 px outside kp 30 (185, 260)
    res_lip = validate_face50_frame(11, shapes_lip, sub_map, lbl_map, 640, 480)
    assert any(f["code"] == "INNER_LIP_OUTSIDE_OUTER" for f in res_lip["findings"])

    # 2) Self-intersecting lip polygon (swap kp 43 and kp 47 so moitrong crosses)
    shapes_cross, sub_map, lbl_map = _make_clean_face50_shapes(frame=12)
    el_by_kp = {sub_map[el["label_id"]]: el for sh in shapes_cross for el in sh["elements"]}
    el_by_kp[43]["points"], el_by_kp[47]["points"] = el_by_kp[47]["points"], el_by_kp[43]["points"]
    res_cross = validate_face50_frame(12, shapes_cross, sub_map, lbl_map, 640, 480)
    assert any(f["code"] == "LIP_SELF_INTERSECT" for f in res_cross["findings"])

    # 3) Brow below upper eyelid & brow non-monotonic in face-local coordinates
    shapes_brow, sub_map, lbl_map = _make_clean_face50_shapes(frame=13)
    el_by_kp = {sub_map[el["label_id"]]: el for sh in shapes_brow for el in sh["elements"]}
    el_by_kp[9]["points"] = [260.0, 175.0]  # below upper lid 23..25 (y~166..168) AND u < kp 8 (285)
    res_brow = validate_face50_frame(13, shapes_brow, sub_map, lbl_map, 640, 480)
    codes = {f["code"] for f in res_brow["findings"]}
    assert "BROW_BELOW_EYE" in codes
    assert "BROW_NON_MONOTONIC_LOCAL" in codes


def test_align_face50_clahe_fan_with_mock_landmarker():
    import numpy as np

    # Build synthetic 68-point iBUG landmarks with kp60 (inner lip left) poking outside kp48 (outer lip left)
    p68 = np.zeros((68, 2), dtype=np.float64)
    for i in range(17, 22):
        p68[i] = [140.0 + (i - 17) * 15.0, 140.0]
    for i in range(22, 27):
        p68[i] = [240.0 + (i - 22) * 15.0, 140.0]
    for i in range(27, 31):
        p68[i] = [220.0, 165.0 + (i - 27) * 15.0]
    p68[36:42] = [[145, 175], [158, 168], [182, 168], [195, 175], [182, 182], [158, 182]]
    p68[42:48] = [[245, 175], [258, 168], [282, 168], [295, 175], [282, 182], [258, 182]]
    p68[48:60] = [
        [185, 260], [197, 250], [210, 246], [220, 248], [230, 246], [243, 250],
        [255, 260], [243, 272], [230, 276], [220, 278], [210, 276], [197, 272],
    ]
    p68[60:68] = [
        [183, 260], [208, 254], [220, 253], [232, 254],  # kp60 -> kp42 is 2px outside kp30!
        [245, 260], [232, 267], [220, 268], [208, 267],
    ]

    class _MockLandmarker:
        def landmarks68(self, img_bgr, try_rotations=False):
            return {"p68": p68.copy(), "score68": np.ones(68), "face_score": 0.99, "rot": 0}

    dark_img = np.full((480, 640, 3), 20, dtype=np.uint8)
    res = align_face50_clahe_fan(dark_img, try_rotations=True, landmarker=_MockLandmarker())
    assert res is not None
    assert res["p50"].shape == (50, 2)
    assert res["lip_clamped"] >= 1
    # kp 42 must now be clamped inside moingoai (x > 185)
    assert res["p50"][42, 0] > 185.0


def test_face50_sync_action_create_and_readback(tmp_path: Path, monkeypatch):
    shapes, _, _ = _make_clean_face50_shapes(frame=102)
    # Strip IDs as new create payloads have no shape/element IDs yet
    create_shapes = []
    for sh in shapes:
        s_copy = {
            "frame": sh["frame"],
            "label_id": sh["label_id"],
            "type": "skeleton",
            "occluded": False,
            "source": "ai",
            "elements": [
                {
                    "label_id": el["label_id"],
                    "type": "points",
                    "points": list(el["points"]),
                    "occluded": False,
                    "outside": False,
                    "source": "ai",
                }
                for el in sh["elements"]
            ],
        }
        create_shapes.append(s_copy)

    prop_file = tmp_path / "prop_create.json"
    prop_file.write_text(json.dumps({
        "job_id": 1975,
        "eligible_proposals": [{
            "frame_index": 102,
            "image_name": "G06_103.jpg",
            "action": "create",
            "changes": [{"kp": 0, "type": "create"}],
            "before_shapes": [],
            "proposed_shapes": create_shapes,
        }],
        "review_queue": [],
    }), encoding="utf-8")

    dry = sync_face50_eligible(job_id=1975, proposals_path=prop_file, yes=False)
    assert dry["create_shapes_count"] == 7
    assert dry["patch_shapes_count"] == 0

    class _MockCreateCvat:
        def __init__(self):
            self.live_shapes: list[dict] = []
            self.created_payloads: list[dict] = []

        def annotations(self, jid):
            return {"shapes": list(self.live_shapes)}

        def create_shapes(self, jid, new_shapes):
            self.created_payloads.extend(new_shapes)
            created = []
            for idx, sh in enumerate(new_shapes):
                s_saved = {**sh, "id": 9000 + idx}
                self.live_shapes.append(s_saved)
                created.append(s_saved)
            return {"shapes": created}

    mock_cv = _MockCreateCvat()
    monkeypatch.setattr("cvat_tools.pose.face50_engine.Cvat.from_env", lambda *a, **kw: mock_cv)
    monkeypatch.setattr("cvat_tools.pose.week2.job_folder", lambda jid: tmp_path / f"job_{jid}")

    res = sync_face50_eligible(job_id=1975, proposals_path=prop_file, yes=True)
    assert res["mode"] == "committed"
    assert res["written_to_cvat"] is True
    assert res["create_shapes_count"] == 7
    assert res["readback_mismatches"] == 0
    assert all(sh["source"] == "manual" for sh in mock_cv.created_payloads)
    assert all(el["source"] == "manual" for sh in mock_cv.created_payloads for el in sh["elements"])

    # Re-running sync on same proposal must trip staleness guard because frame 102 now exists on CVAT
    with pytest.raises(RuntimeError, match="Stale proposal"):
        sync_face50_eligible(job_id=1975, proposals_path=prop_file, yes=True)
