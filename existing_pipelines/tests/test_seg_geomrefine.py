import numpy as np
import pytest
from shapely.geometry import Polygon

from cvat_tools.segmentation import geomrefine as G


def jitter_rect(seed=0, amp=0.8):
    rng = np.random.default_rng(seed)
    pts = []
    for x in range(20, 120, 3):
        pts.append((x, 30))
    for y in range(30, 90, 3):
        pts.append((120, y))
    for x in range(120, 20, -3):
        pts.append((x, 90))
    for y in range(90, 30, -3):
        pts.append((20, y))
    a = np.array(pts, float) + rng.normal(0, amp, (len(pts), 2))
    return a.reshape(-1).tolist()


def test_artificial_straightens_and_reduces_points():
    p = jitter_rect()
    r = G.refine(p, "artificial")
    m0, m1 = G.metrics(p), G.metrics(r)
    assert m1["n"] <= 8 < m0["n"]
    assert m1["straight"] >= 0.9
    assert m1["jag"] <= m0["jag"]
    assert G.iou(p, r) > 0.95
    # canh gan ngang/doc -> dung ngang/doc
    xy = np.asarray(r).reshape(-1, 2)
    d = np.abs(np.diff(np.vstack([xy, xy[:1]]), axis=0))
    assert np.all(np.minimum(d[:, 0], d[:, 1]) < 0.05)


def test_organic_fewer_points_valid():
    t = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    rng = np.random.default_rng(1)
    rr = 40 + rng.normal(0, 0.8, len(t))
    p = np.stack([100 + rr * np.cos(t), 100 + rr * np.sin(t)], 1).reshape(-1).tolist()
    r = G.refine(p, "organic")
    assert G.metrics(r)["n"] <= 40
    assert G.metrics(r)["valid"]
    assert G.iou(p, r) > 0.95


def test_self_intersecting_becomes_valid():
    bow = [0, 0, 50, 50, 50, 0, 0, 50]  # no
    r = G.refine(bow, "artificial")
    assert Polygon(np.asarray(r).reshape(-1, 2)).is_valid


def test_degenerate_returns_input():
    assert G.refine([1, 1, 2, 2], "lane") == [1.0, 1.0, 2.0, 2.0]


def test_max_points_respected():
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    p = np.stack([200 + 90 * np.cos(t), 200 + 90 * np.sin(t)], 1).reshape(-1).tolist()
    for kind in ("artificial", "organic", "lane"):
        assert G.metrics(G.refine(p, kind, max_points=20))["n"] <= 20


def test_edge_snap_pulls_to_real_edge():
    img = np.full((140, 160, 3), 40, np.uint8)
    img[30:100, 30:110] = 220  # canh that x=30..110, y=30..100
    sloppy = [32.5, 32.5, 107.5, 32.5, 107.5, 97.5, 32.5, 97.5]  # lech ~2.5px vao trong
    r = G.refine(sloppy, "artificial", img, snap_k=4)
    b = Polygon(np.asarray(r).reshape(-1, 2)).bounds
    assert abs(b[0] - 30) < 1.6 and abs(b[1] - 30) < 1.6 and abs(b[2] - 110) < 1.6 and abs(b[3] - 100) < 1.6
    assert abs(b[0] - 30) < abs(32.5 - 30)


def test_edge_snap_stays_in_band():
    img = np.full((140, 160, 3), 40, np.uint8)
    img[30:100, 30:110] = 220
    far = [40, 40, 100, 40, 100, 90, 40, 90]  # canh that cach 10px > k
    r = G.refine(far, "artificial", img, snap_k=3)
    b = Polygon(np.asarray(r).reshape(-1, 2)).bounds
    assert b[0] > 36 and b[2] < 104


def test_shared_boundary_no_gap_no_overlap():
    rng = np.random.default_rng(3)
    ys = np.arange(20, 121, 2)
    xs = 100 + 4 * np.sin(ys / 9.0) + rng.normal(0, 1.0, len(ys))  # bien chung rang cua
    A = [(20, 20)] + [(x - 1, y) for x, y in zip(xs, ys)] + [(20, 120)]  # chong/ho ngau nhien
    B = [(180, 20)] + [(x + 1.5, y) for x, y in zip(xs, ys)] + [(180, 120)]
    a, b = G.smooth_shared_boundary(np.array(A).reshape(-1).tolist(), np.array(B).reshape(-1).tolist())
    ov, gap = G.overlap_gap(a, b)
    ov0, gap0 = G.overlap_gap(np.array(A).reshape(-1), np.array(B).reshape(-1))
    assert ov < 0.5 and gap < 0.5
    assert ov0 + gap0 > ov + gap
    assert G.metrics(a)["valid"] and G.metrics(b)["valid"]
    assert G.iou(A, a) > 0.95 and G.iou(B, b) > 0.95
    # bien chung muot hon: so dinh giam
    assert G.metrics(a)["n"] < G.metrics(np.array(A).reshape(-1))["n"]


def test_kind_for_label():
    assert G.kind_for_label("vegetation") == "organic"
    assert G.kind_for_label("pole") == "artificial"
    assert G.kind_for_label("lane/single white") == "lane"
    assert G.kind_for_label("lane/crosswalk") == "artificial"


def test_sam_mask_to_polygon_and_lazy_import():
    from cvat_tools.segmentation import sam_refine as S
    m = np.zeros((80, 80), np.uint8)
    m[10:50, 20:60] = 1
    m[70:75, 70:75] = 1  # manh nho bi bo
    poly = S.mask_to_polygon(m)
    assert poly and len(poly) // 2 <= 8
    assert S._MODELS == {}  # chua nap model
    with pytest.raises(ValueError):
        S.load("nope")


def test_sam_real_mobilesam_box():
    pytest.importorskip("ultralytics")
    import cv2
    from cvat_tools.segmentation import sam_refine as S
    try:
        S._weights_dir()
    except FileNotFoundError:
        pytest.skip("khong co weights")
    img = np.full((240, 320, 3), 60, np.uint8)
    cv2.circle(img, (160, 120), 50, (230, 230, 230), -1)
    poly = S.refine_with_sam(img, {"box": [100, 60, 220, 180]}, backend="mobilesam")
    assert poly is not None
    area = Polygon(np.asarray(poly).reshape(-1, 2)).area
    assert 0.6 * np.pi * 50**2 < area < 1.5 * np.pi * 50**2
