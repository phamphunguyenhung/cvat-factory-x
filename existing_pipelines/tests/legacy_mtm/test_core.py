from mtm import core as C


def walk(f0, f1, x0=0.0, dx=5.0):
    return {f: [x0 + dx * (f - f0), 50.0, x0 + dx * (f - f0) + 40, 150.0] for f in range(f0, f1 + 1)}


def test_straight_walk_two_keyframes():
    assert C.simplify_keyframes(walk(0, 149), 1.0) == [0, 149]


def test_turn_adds_keyframe_and_error_within_tol():
    tr = walk(0, 49)
    tr.update({f: [250.0 - 5 * (f - 49), 50.0, 290.0 - 5 * (f - 49), 150.0] for f in range(50, 100)})
    kf = C.simplify_keyframes(tr, 2.0)
    assert any(45 <= k <= 54 for k in kf)
    for f in tr:
        a, b = max(k for k in kf if k <= f), min(k for k in kf if k >= f)
        assert C.max_dev(C.lerp_box(f, a, tr[a], b, tr[b]), tr[f]) <= 2.0 + 1e-6


def test_cvat_roundtrip_outside_and_source_manual():
    tr = walk(10, 40)
    trk = C.to_cvat_track(7, tr, [10, 25, 40], 149)
    assert trk["source"] == "manual" and trk["shapes"][-1] == {**trk["shapes"][-1], "outside": True, "frame": 41}
    assert sorted(C.expand_cvat_track(trk, 149)) == list(range(10, 41))
    assert not any(s["outside"] for s in C.to_cvat_track(7, walk(100, 149), [100, 149], 149)["shapes"])


def test_fill_gaps_and_select_longest():
    tr = walk(0, 20); del tr[10], tr[11]
    assert sorted(C.fill_gaps(tr, 3)) == list(range(21))
    assert list(C.select_longest({1: walk(0, 99), 2: walk(0, 3), 3: walk(0, 79)}, 2)) == [1, 3]


def test_anchor_selection_and_conflicts():
    tracks = {1: walk(0, 50, 0), 2: walk(0, 50, 500)}
    pick, w = C.select_by_anchors(tracks, {"P1": (20, 100), "P2": (520, 100)}, 0)
    assert pick == {"P1": 1, "P2": 2} and not w
    pick, w = C.select_by_anchors(tracks, {"P1": (20, 100), "P2": (25, 100), "P3": (900, 900)}, 0)
    assert any("trùng" in x for x in w) and any("không track" in x for x in w)


def test_qa_kinds():
    a, b = walk(0, 49), walk(53, 99, 265.0)
    frag = walk(0, 40); del frag[20]
    dup = {f: [v + 1 for v in bx] for f, bx in a.items()}
    k = lambda d: {i["kind"] for i in C.qa_tracks(d)}
    assert "id_switch" in k({1: a, 2: b}) and "fragmentation" in k({1: frag})
    assert "false_track" in k({1: walk(0, 3)}) and "merge" in k({1: a, 2: dup})
    assert k({1: walk(0, 99)}) == set()


def test_qa_keyframes():
    def trk(fr, out=None):
        sh = [{"frame": f, "points": [0, 0, 1, 1], "outside": False, "keyframe": True} for f in fr]
        if out is not None:
            sh.append({"frame": out, "points": [0, 0, 1, 1], "outside": True, "keyframe": True})
        return {"shapes": sh}
    k = lambda t: {i["kind"] for i in C.qa_keyframes(t)}
    assert "drift_risk" in k([trk([0, 100])]) and "keyframe_dense" in k([trk(range(40))])
    gap = {"shapes": [{"frame": 0, "outside": False, "points": [0, 0, 1, 1]}, {"frame": 10, "outside": True, "points": [0, 0, 1, 1]},
                      {"frame": 90, "outside": False, "points": [0, 0, 1, 1]}]}
    assert "gap_over_2s" in k([gap]) and k([trk(range(0, 150, 10))]) == set()


def test_write_mot_one_based(tmp_path):
    p = tmp_path / "gt.txt"
    assert C.write_mot({9: {10: [0, 0, 10, 20]}, 3: {10: [5, 5, 15, 25], 11: [6, 5, 16, 25]}}, p) == 3
    L = p.read_text().splitlines()
    assert L[0] == "1,1,5.00,5.00,10.00,20.00,1,1,1" and any(x.startswith("1,2,") for x in L)


def test_anchor_per_person_frame_and_same_person():
    tracks = {1: walk(0, 50, 0), 2: walk(40, 90, 500)}
    pick, w = C.select_by_anchors(tracks, {"P1": (20, 100, 0), "P2": (620, 100, 60)}, 0)
    assert pick == {"P1": 1, "P2": 2} and not w
    assert C.same_person(walk(0, 30), walk(0, 30)) and not C.same_person(walk(0, 30), walk(0, 30, 800))
