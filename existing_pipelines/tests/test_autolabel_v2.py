from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from cvat_tools.autolabel.cache import load_or_fetch_snapshot
from cvat_tools.autolabel.cli import run_pipeline
from cvat_tools.autolabel.qc import evaluate_frame_qc
from cvat_tools.autolabel.refine import selective_refine_frame
from cvat_tools.autolabel.sync import ConcurrentEditConflict, safe_auto_sync


def _load_cached_1975():
    state = load_or_fetch_snapshot(1975, offline=True)
    labels = state["labels"]
    lbl_name = {int(l["id"]): str(l["name"]) for l in labels}
    sub_kp = {int(s["id"]): int(s["name"]) for l in labels for s in (l.get("sublabels") or []) if str(s["name"]).isdigit()}
    by_f = {}
    for sh in state["annotations"]["shapes"]:
        by_f.setdefault(int(sh["frame"]), []).append(sh)
    return state, lbl_name, sub_kp, by_f


def test_f090_eyebrow_and_lip_regression_and_f102_f106_present():
    state, lbl_name, sub_kp, by_f = _load_cached_1975()
    meta = {int(state["meta"]["start_frame"]) + i: fm for i, fm in enumerate(state["meta"]["frames"])}

    # f090 eyebrow & lip must now pass with 0 errors and 0 warnings
    qc90 = evaluate_frame_qc(90, by_f[90], sub_kp, lbl_name, meta[90]["width"], meta[90]["height"])
    assert qc90["error_count"] == 0
    assert qc90["warning_count"] == 0

    # f102 and f106 missing-shape regression: both must have 7 shapes & 0 errors
    for fidx in (102, 106):
        assert len(by_f[fidx]) == 7
        qc = evaluate_frame_qc(fidx, by_f[fidx], sub_kp, lbl_name, meta[fidx]["width"], meta[fidx]["height"])
        assert qc["error_count"] == 0
        assert qc["warning_count"] == 0


def test_f100_f111_f112_difficult_case_qa():
    state, lbl_name, sub_kp, by_f = _load_cached_1975()
    meta = {int(state["meta"]["start_frame"]) + i: fm for i, fm in enumerate(state["meta"]["frames"])}

    # f112 (already synced) passes cleanly with 0 errors
    qc112 = evaluate_frame_qc(112, by_f[112], sub_kp, lbl_name, meta[112]["width"], meta[112]["height"])
    assert qc112["error_count"] == 0

    # f100 and f111 remain in review (not auto-synced)
    for fidx in (100, 111):
        res = selective_refine_frame(fidx, by_f[fidx], sub_kp, lbl_name, meta[fidx]["width"], meta[fidx]["height"])
        assert res["status"] == "review"


def test_idempotent_offline_run():
    rep1 = run_pipeline(1975, mode="dry-run", offline=True, yes=False)
    rep2 = run_pipeline(1975, mode="dry-run", offline=True, yes=False)
    assert rep1["eligible_frames"] == []
    assert rep2["eligible_frames"] == []
    assert rep1["qc_errors_after"] == rep2["qc_errors_after"]


def test_concurrent_cvat_edit_and_network_failure_rollback(tmp_path: Path, monkeypatch):
    _, _, _, by_f = _load_cached_1975()
    sh_b = copy.deepcopy(by_f[90][0])
    sh_p = copy.deepcopy(sh_b)
    sh_p["elements"][0]["points"][0] += 1.0

    eligible = [{
        "frame": 90,
        "changed_kpts": [10],
        "before_shapes": [sh_b],
        "proposed_shapes": [sh_p],
    }]

    # 1) Concurrent edit conflict
    class _ConflictCvat:
        def annotations(self, jid):
            mod = copy.deepcopy(sh_b)
            mod["elements"][0]["points"][0] += 9.0
            return {"shapes": [mod]}

    monkeypatch.setattr("cvat_tools.autolabel.sync.Cvat.from_env", lambda *a, **kw: _ConflictCvat())
    with pytest.raises(ConcurrentEditConflict):
        safe_auto_sync(1975, eligible, yes=True)

    # 2) Network failure mid-sync -> automatic rollback
    calls = {"update": 0}

    class _FailMidSyncCvat:
        def annotations(self, jid):
            return {"shapes": [copy.deepcopy(sh_b)]}

        def update_shapes(self, jid, shapes):
            calls["update"] += 1
            if calls["update"] == 1:
                raise ConnectionError("Simulated network drop")
            return {"shapes": shapes}

    monkeypatch.setattr("cvat_tools.autolabel.sync.Cvat.from_env", lambda *a, **kw: _FailMidSyncCvat())
    monkeypatch.setattr("cvat_tools.autolabel.sync.week2.job_folder", lambda jid: tmp_path)
    with pytest.raises(RuntimeError, match="rolled_back=True"):
        safe_auto_sync(1975, eligible, yes=True)
    assert calls["update"] == 2
