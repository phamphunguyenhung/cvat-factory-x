"""Sinh nhap lane/* + area/* TUNG FRAME doc lap (khong dung thong tin frame lan can).

Mac dinh cac frame KHONG lien tuc -> khong noi suy/truyen anchor. Duong noi suy (laneinterp.interpolate_*)
chi dung rieng khi job khai bao continuity="video" (keyframe moi 10 anh), xem generate_frame(continuity=...).

Dau vao moi frame: anh BGR + polygon road nguoi ve (doc tu client chi-doc hoac cache offline). CPU nhe (cv2/numpy).

  1. Diem tu (VP): bien trai/phai cua road (fit ben vung) -> tinh chinh bang binh chon tia san:
     luoi nho quanh VP, chon VP cho tong diem cua 3 tia son tot nhat lon nhat (co phat do lech VP goc).
  2. Mask son: anh sang rieng theo frame (gamma ve trung vi ROI road + CLAHE, chi tac dong khi anh toi),
     top-hat mong, nguong theo phan vi cua road -> khong dung nguong cung cho anh sang.
  3. Tia hoi tu: moi vach la mot TIA qua VP (x_ref tai hang y_ref). Diem = so mau son doc theo tia
     (khong quet Hough tu do) -> rang buoc hoi tu ve VP la bat buoc; vach dut = mot polygon lien (noi gap).
     double = hai tia sat nhau cung mau -> MOT polygon om cap. Mau trang/vang: do am (R+G)/2-B tuong doi voi nen road.
  4. Be rong hep theo phoi canh: half(y) = h0 * (y - vy) / (y_ref - vy), tam 1.2 px o xa.
  5. crosswalk: cum soc song song/gan nhau tren mask son -> bao loi 4-6 diem.
  6. area/direct: lan xe minh tu capo toi VP; hai bien = tia son gan tam anh nhat moi ben (hoac mep road,
     hoac bien theo be rong tien nghiem), cat voi road. area/alternative: CHI lan canh cung chieu (co vach
     trang phan lan o bien va road con tiep tuc), khong phai road - direct.

Moi shape: {label, polygon (phang), conf, _issues, _src}. Khong co GT cho vach/curb: chi bao so vach/frame.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from cvat_tools.lanes.areadirect import mask_polys

W0, H0 = 1280, 720
LANE_W_PRIOR = 0.42        # be rong lan xe minh tai y_ref, theo W (tien nghiem, chi dung khi thieu bang chung)
Y_REF = 0.80               # hang tham chieu (theo H) de tham so hoa tia
CFG: Dict[str, float] = {
    "dark_mean": 85.0, "ray_step": 4.0, "min_hits": 14, "min_cov": 0.18, "nms": 10,
    "gap_samples": 18, "min_chain_hits": 12, "double_min": 10, "double_max": 46,
    "yellow_rel_warmth": 24.0, "min_span_h": 0.09, "double_gap_fill": 0.88, "max_lines": 6, "paint_half0": 6.0, "min_alt_area": 15000.0,
}


# ------------------------------------------------------------------ io / hinh hoc
def road_mask(roads: Any, W: int, H: int) -> np.ndarray:
    polys = [roads] if len(roads) and not hasattr(roads[0], "__len__") else list(roads)
    m = np.zeros((H, W), np.uint8)
    for p in polys:
        cv2.fillPoly(m, [np.round(np.asarray(p, np.float32)).astype(np.int32).reshape(-1, 2)], 1)
    return m


def _flat(a) -> List[float]:
    return [round(float(v), 1) for v in np.asarray(a).reshape(-1)]


def road_extent(rm: np.ndarray) -> Tuple[float, float]:
    ys = np.nonzero(rm.any(1))[0]
    return (float(ys.min()), float(ys.max())) if ys.size else (0.0, 0.0)


def brighten_for_paint(img: np.ndarray, rm: np.ndarray) -> Tuple[np.ndarray, bool, float]:
    """Anh da sang rieng cho frame toi: gamma dua trung vi road ve ~110, roi CLAHE. Anh sang giu nguyen."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vals = gray[rm > 0]
    med = float(np.median(vals)) if vals.size else float(gray.mean())
    dark = float(gray.mean()) < CFG["dark_mean"] or med < 80
    if not dark:
        return img, False, med
    g = float(np.clip(np.log(110 / 255.0) / np.log(max(med, 3.0) / 255.0), 0.3, 1.0))
    out = (255.0 * (img.astype(np.float32) / 255.0) ** g).clip(0, 255).astype(np.uint8)
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), True, med


_LANE_SEG_CACHE: Optional[Tuple[Any, Any]] = None
_LANE_SEG_FAILED: bool = False


def _get_lane_seg():
    """Lazy-load 8bits-ai/SegFormer-Lane-Marking from local HF cache (returns None if unavailable)."""
    global _LANE_SEG_CACHE, _LANE_SEG_FAILED
    if _LANE_SEG_FAILED:
        return None
    if _LANE_SEG_CACHE is not None:
        return _LANE_SEG_CACHE
    try:
        import torch
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        mid = "8bits-ai/SegFormer-Lane-Marking"
        proc = SegformerImageProcessor(size={"height": 224, "width": 384})
        model = SegformerForSemanticSegmentation.from_pretrained(mid, local_files_only=True)
        model.eval()
        _LANE_SEG_CACHE = (proc, model, torch)
        return _LANE_SEG_CACHE
    except Exception:
        _LANE_SEG_FAILED = True
        return None


def _segformer_lane_mask(img_bgr: np.ndarray, rm: np.ndarray) -> Optional[np.ndarray]:
    """Run SegFormer-Lane-Marking on BGR image and return binary uint8 mask inside road."""
    loaded = _get_lane_seg()
    if loaded is None:
        return None
    proc, model, torch = loaded
    from PIL import Image

    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    inp = proc(images=pil, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inp).logits
    up = torch.nn.functional.interpolate(logits, size=img_bgr.shape[:2], mode="bilinear", align_corners=False)
    prob = torch.softmax(up, dim=1)[0, 1].numpy()
    allow = cv2.dilate(rm, np.ones((9, 9), np.uint8)) > 0
    return ((prob > 0.30) & allow).astype(np.uint8)


def paint_mask(enh: np.ndarray, rm: np.ndarray, dark: bool) -> np.ndarray:
    """Mask son vach ke duong: uu tien SegFormer-Lane-Marking (chong loa den dem), fallback top-hat."""
    H, W = enh.shape[:2]
    seg_m = _segformer_lane_mask(enh, rm)
    if seg_m is not None and int(seg_m.sum()) >= 150:
        return cv2.morphologyEx(seg_m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    s = H / 720.0
    g = cv2.cvtColor(enh, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    k = int(35 * s) | 1
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    allow = cv2.dilate(rm, np.ones((9, 9), np.uint8)) > 0
    vals = th[allow]
    if vals.size == 0:
        return np.zeros((H, W), np.uint8)
    thr = max(float(np.percentile(vals, 93 if dark else 94)), 22.0 if dark else 28.0)
    sat_ok = cv2.cvtColor(enh, cv2.COLOR_BGR2HSV)[..., 1] < 255   # de mo rong (mau xu ly o buoc tia)
    m = ((th > thr) & allow & sat_ok).astype(np.uint8)
    if dark:
        # Boost yellow paint channel (Lab b*) so dimmer yellow lines are not starved by white/glare
        lab_b = cv2.cvtColor(enh, cv2.COLOR_BGR2LAB)[..., 2].astype(np.float32)
        th_b = cv2.morphologyEx(lab_b, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        b_vals = th_b[allow]
        if b_vals.size:
            b_thr = max(float(np.percentile(b_vals, 92)), 10.0)
            m = m | ((th_b > b_thr) & allow).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


# ------------------------------------------------------------------ diem tu
def _edge_fit(rm: np.ndarray, side: str, y0: float, y1: float):
    H, W = rm.shape
    pts = []
    for y in range(int(y0), int(y1), 3):
        xs = np.nonzero(rm[y])[0]
        if xs.size < 3:
            continue
        x = xs.min() if side == "L" else xs.max()
        if (side == "L" and x <= 3) or (side == "R" and x >= W - 4):
            continue
        pts.append((x, y))
    if len(pts) < 12:
        return None
    vx, vy, x0, yy0 = cv2.fitLine(np.float32(pts), cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
    return np.array([x0, yy0]), np.array([vx, vy])


def vp_from_road(rm: np.ndarray) -> Optional[Tuple[float, float]]:
    H, W = rm.shape
    t, b = road_extent(rm)
    if b - t < 40:
        return None
    y0, y1 = t + 0.08 * (b - t), b - 0.05 * (b - t)
    L, R = _edge_fit(rm, "L", y0, y1), _edge_fit(rm, "R", y0, y1)
    if L is None or R is None:
        return None
    (c1, d1), (c2, d2) = L, R
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-6:
        return None
    tt = ((c2[0] - c1[0]) * d2[1] - (c2[1] - c1[1]) * d2[0]) / den
    vp = c1 + tt * d1
    if not (-0.3 * W < vp[0] < 1.3 * W and t - 0.25 * H < vp[1] < t + 0.12 * H):
        return None
    return float(vp[0]), float(vp[1])


def _ray_xy(xr: np.ndarray, ys: np.ndarray, vp, yref: float):
    vx, vy = vp
    return vx + (xr[:, None] - vx) * (ys[None, :] - vy) / (yref - vy)


def ray_scores(m: np.ndarray, rm_d: np.ndarray, vp, yref: float, ytop: float, ybot: float):
    """-> (xr, hits[n_ray,n_y], valid[n_ray,n_y], ys). hits = mask son doc tia, valid = trong anh va trong road."""
    H, W = m.shape
    xr = np.arange(-0.7 * W, 1.7 * W, CFG["ray_step"])
    ys = np.arange(ytop, ybot, 3.0)
    if ys.size < 8:
        return xr, np.zeros((xr.size, 0), bool), np.zeros((xr.size, 0), bool), ys
    X = _ray_xy(xr, ys, vp, yref)
    inb = (X >= 0) & (X < W - 1)
    xi = np.clip(np.round(X).astype(int), 0, W - 1)
    yi = np.broadcast_to(np.clip(ys.astype(int), 0, H - 1)[None, :], xi.shape)
    valid = inb & (rm_d[yi, xi] > 0)
    md = cv2.dilate(m, np.ones((3, 5), np.uint8))
    return xr, (md[yi, xi] > 0) & valid, valid, ys


def _peaks(xr, hits, valid, nms: int) -> List[int]:
    sc = hits.sum(1).astype(float)
    step = CFG["ray_step"]
    w = max(1, int(nms / step))
    out = []
    for i in range(len(xr)):
        lo, hi = max(0, i - w), min(len(xr), i + w + 1)
        v = valid[i].sum()
        if sc[i] >= CFG["min_hits"] and v > 0 and sc[i] / v >= CFG["min_cov"] and sc[i] >= sc[lo:hi].max() \
                and (sc[lo:i] < sc[i]).all():
            out.append(i)
    return out


def refine_vp(m: np.ndarray, rm_d: np.ndarray, vp0, yref: float, ytop: float, ybot: float, rng: Tuple[float, float]):
    """Luoi nho quanh VP goc: chon VP cho tong diem 3 tia tot nhat lon nhat (tru phat do lech)."""
    best, bv = vp0, -1e9
    base = None
    for dy in np.arange(-rng[1], rng[1] + 1, 12):
        for dx in np.arange(-rng[0], rng[0] + 1, 24):
            vp = (vp0[0] + dx, vp0[1] + dy)
            xr, hits, valid, _ = ray_scores(m, rm_d, vp, yref, max(ytop, vp[1] + 0.03 * m.shape[0]), ybot)
            if hits.shape[1] == 0:
                continue
            pk = _peaks(xr, hits, valid, CFG["nms"])
            sc = sorted((hits[i].sum() for i in pk), reverse=True)[:3]
            v = float(sum(sc)) - 0.04 * (abs(dx) + 2 * abs(dy))
            if dx == 0 and dy == 0:
                base = float(sum(sc))
            if v > bv:
                best, bv = vp, v
    return best, (base or 0.0), bv


# ------------------------------------------------------------------ son -> vach
def _chain(hit_row: np.ndarray, gap: int):
    """Chuoi hit noi gap <= gap; tra (i0, i1, so hit) cua chuoi nhieu hit nhat."""
    idx = np.nonzero(hit_row)[0]
    if idx.size == 0:
        return None
    best, cur = None, [idx[0], idx[0], 1]
    for i in idx[1:]:
        if i - cur[1] <= gap:
            cur[1], cur[2] = i, cur[2] + 1
        else:
            best = cur if best is None or cur[2] > best[2] else best
            cur = [i, i, 1]
    best = cur if best is None or cur[2] > best[2] else best
    return tuple(best)


def _taper(vp, yref: float, y_a: float, y_b: float, xr: float, half_ref: float, W: int, H: int) -> List[float]:
    """Polygon hep tu y_a (xa) den y_b (gan) doc tia qua VP, half(y)=half_ref*(y-vy)/(yref-vy), tam 1.2."""
    vx, vy = vp
    ys = np.array([y_a, y_b], float)
    xs = vx + (xr - vx) * (ys - vy) / (yref - vy)
    hf = np.maximum(1.2, half_ref * (ys - vy) / (yref - vy))
    q = np.array([[xs[0] - hf[0], ys[0]], [xs[0] + hf[0], ys[0]], [xs[1] + hf[1], ys[1]], [xs[1] - hf[1], ys[1]]])
    q[:, 0] = q[:, 0].clip(0, W - 1)
    q[:, 1] = q[:, 1].clip(0, H - 1)
    return _flat(q)


def _warm(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float32)
    return (f[..., 2] + f[..., 1]) / 2 - f[..., 0]


def detect_ray_lanes(img: np.ndarray, enh: np.ndarray, m: np.ndarray, rm: np.ndarray, vp, dark: bool,
                     ytop: float, ybot: float) -> List[Dict[str, Any]]:
    """Cac vach (khong gom crosswalk). Moi phan tu: label, polygon, conf, _issues, xr (tia), y0,y1, color, double."""
    H, W = m.shape
    yref = Y_REF * H
    rm_d = cv2.dilate(rm, np.ones((11, 11), np.uint8))
    xr, hits, valid, ys = ray_scores(m, rm_d, vp, yref, ytop, ybot)
    if hits.shape[1] == 0:
        return []
    pk = _peaks(xr, hits, valid, CFG["nms"])
    # Compute yellow warmth on original `img` (gamma+CLAHE on `enh` washes out yellow toward white)
    warm = cv2.GaussianBlur(_warm(img), (0, 0), 1.5)
    bg = float(np.median(warm[rm > 0])) if (rm > 0).any() else 0.0
    lab_b = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 2].astype(np.float32) - 128.0, (0, 0), 1.5)
    bg_b = float(np.median(lab_b[rm > 0])) if (rm > 0).any() else 0.0
    cands = []
    for i in pk:
        ch = _chain(hits[i], int(CFG["gap_samples"]))
        if ch is None or ch[2] < CFG["min_chain_hits"]:
            continue
        i0, i1, nh = ch
        if ys[i1] - ys[i0] < CFG["min_span_h"] * H:
            continue    # vach qua ngan (thuong la nhieu o gan VP / mep xe)
        X = _ray_xy(xr[i:i + 1], ys[i0:i1 + 1], vp, yref)[0]
        yy = ys[i0:i1 + 1]
        sel = hits[i, i0:i1 + 1]
        xi = np.clip(np.round(X[sel]).astype(int), 0, W - 1)
        yi = np.clip(yy[sel].astype(int), 0, H - 1)
        # Sample +/-2px around ray and take p75 so thin yellow paint is not washed out by asphalt
        w_vals = np.maximum.reduce([warm[yi, np.clip(xi + dx, 0, W - 1)] for dx in (-2, -1, 0, 1, 2)])
        b_vals = np.maximum.reduce([lab_b[yi, np.clip(xi + dx, 0, W - 1)] for dx in (-2, -1, 0, 1, 2)])
        rel = float(np.percentile(w_vals, 75)) - bg
        rel_b = float(np.percentile(b_vals, 75)) - bg_b
        cov = nh / max(1, int(valid[i, i0:i1 + 1].sum()))
        is_yellow = (rel > 12.5 and rel_b > 5.0) or (rel > 16.0) or (rel_b > 8.0)
        cands.append({"i": i, "xr": float(xr[i]), "y0": float(ys[i0]), "y1": float(ys[i1]), "hits": int(nh),
                      "cov": float(cov), "span": float(ys[i1] - ys[i0]), "yellow": is_yellow, "rel": rel})
    cands.sort(key=lambda c: -c["hits"])
    cands = cands[:16]
    used, out = set(), []
    # double = hai tia sat nhau cung mau
    for a in range(len(cands)):
        for b in range(a + 1, len(cands)):
            A, B = cands[a], cands[b]
            sep = abs(A["xr"] - B["xr"])
            if a in used or b in used or A["yellow"] != B["yellow"] or not CFG["double_min"] <= sep <= CFG["double_max"]:
                continue
            if min(A["y1"], B["y1"]) - max(A["y0"], B["y0"]) < 0.4 * min(A["span"], B["span"]):
                continue
            mid = int(round(((A["xr"] + B["xr"]) / 2 - xr[0]) / CFG["ray_step"]))
            if 0 <= mid < hits.shape[0] and hits[mid].sum() > CFG["double_gap_fill"] * min(A["hits"], B["hits"]):
                continue    # giua khong trong = mot vach day, khong phai cap vach
            used.update((a, b))
            out.append(_emit([A, B], vp, yref, W, H, dark, ytop, ybot))
    for a, A in enumerate(cands):
        if a not in used:
            out.append(_emit([A], vp, yref, W, H, dark, ytop, ybot))
    out.sort(key=lambda d: -d["conf"])
    deduped_out: List[Dict[str, Any]] = []
    min_sep = max(36.0, 0.028 * W)
    for d in out:
        if any(k["yellow"] == d["yellow"] and abs(k["xr"] - d["xr"]) < min_sep for k in deduped_out):
            continue
        deduped_out.append(d)
    return deduped_out[:int(CFG["max_lines"])]


def _emit(cs: List[Dict[str, Any]], vp, yref: float, W: int, H: int, dark: bool, ytop: float, ybot: float) -> Dict[str, Any]:
    dbl = len(cs) == 2
    xr = float(np.mean([c["xr"] for c in cs]))
    y0, y1 = min(c["y0"] for c in cs), max(c["y1"] for c in cs)
    yellow = cs[0]["yellow"]
    half = CFG["paint_half0"] + (abs(cs[0]["xr"] - cs[1]["xr"]) / 2 + 3 if dbl else 0)
    hits = float(np.mean([c["hits"] for c in cs]))
    cov = float(np.mean([c["cov"] for c in cs]))
    conf = 0.45 * min(1.0, hits / 70) + 0.3 * min(1.0, cov / 0.5) + 0.25 * min(1.0, (y1 - y0) / (0.25 * H))
    iss = []
    if dark:
        conf *= 0.85
        iss.append("anh_toi")
    if yellow and dark:
        iss.append("mau_vang_tren_anh_toi_can_kiem")
        conf *= 0.85
    if hits < 25:
        iss.append("vach_ngan")
    color = "yellow" if yellow else "white"
    return {"label": f"lane/{'double' if dbl else 'single'} {color}", "polygon": _taper(vp, yref, y0, y1, xr, half, W, H),
            "conf": round(float(min(conf, 1.0)), 3), "_issues": iss, "_src": "ray_vp", "xr": xr, "y0": y0, "y1": y1,
            "yellow": yellow, "double": dbl}


# ------------------------------------------------------------------ crosswalk
def detect_crosswalk_stripes(
    img: np.ndarray,
    enh: np.ndarray,
    rm: np.ndarray,
    dark: bool,
    paint_m: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """Cum >=4 soc ngan, gan song song, xep canh nhau tren mask son -> polygon 4-6 diem."""
    H, W = rm.shape
    s = H / 720.0
    allow = cv2.dilate(rm, np.ones((9, 9), np.uint8)) > 0
    if not allow.any():
        return []
    if paint_m is not None:
        m = (paint_m > 0).astype(np.uint8) & allow.astype(np.uint8)
    else:
        g = cv2.GaussianBlur(cv2.cvtColor(enh, cv2.COLOR_BGR2GRAY), (0, 0), 0.8)
        k = int(31 * s) | 1
        th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        thr = max(float(np.percentile(th[allow], 90)), 20.0 if dark else 30.0)
        m = ((th > thr) & allow).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    bl = []
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        if a < 30 * s * s or a > 9000 * s * s:
            continue
        pts = np.column_stack(np.nonzero(lab[st[i, 1]:st[i, 1] + st[i, 3], st[i, 0]:st[i, 0] + st[i, 2]] == i))[:, ::-1] + st[i, :2]
        (cx, cy), (w, h), ang = cv2.minAreaRect(pts.astype(np.float32))
        L, S = max(w, h), min(w, h)
        if S < 2 or L / S < 2.0 or L < 12 * s or S > 45 * s:
            continue
        th_ = np.deg2rad(ang if w >= h else ang + 90)
        bl.append({"c": np.array([cx, cy]), "L": L, "S": S, "u": np.array([np.cos(th_), np.sin(th_)]), "pts": pts})
    if len(bl) < 4:
        return []
    par = list(range(len(bl)))

    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for a in range(len(bl)):
        for b in range(a + 1, len(bl)):
            A, B = bl[a], bl[b]
            cosang = abs(float(A["u"] @ B["u"]))
            if cosang < np.cos(np.deg2rad(16)):
                continue
            d = B["c"] - A["c"]
            u = (A["u"] + np.sign(A["u"] @ B["u"]) * B["u"])
            u /= np.linalg.norm(u) + 1e-9
            du, dv = abs(float(d @ u)), abs(float(d @ np.array([-u[1], u[0]])))
            Lm, Sm = (A["L"] + B["L"]) / 2, max(A["S"], B["S"])
            if du < 0.75 * Lm and dv < 2.2 * Sm + 8 * s:
                par[f(a)] = f(b)
    groups: Dict[int, List[int]] = {}
    for i in range(len(bl)):
        groups.setdefault(f(i), []).append(i)
    out = []
    for g_ in groups.values():
        if len(g_) < 4:
            continue
        pts = np.vstack([bl[i]["pts"] for i in g_]).astype(np.float32)
        hull = cv2.convexHull(pts)
        eps = 2.0
        ap = cv2.approxPolyDP(hull, eps, True)
        while len(ap) > 6:
            eps *= 1.4
            ap = cv2.approxPolyDP(hull, eps, True)
        if len(ap) < 4:
            ap = cv2.boxPoints(cv2.minAreaRect(pts)).reshape(-1, 1, 2)
        poly = ap.reshape(-1, 2)
        _, _, cw_w, cw_h = cv2.boundingRect(poly.astype(np.int32))
        if cw_w <= cw_h * 1.15 or cw_h > 0.35 * H or cv2.contourArea(poly.astype(np.float32)) < 250 * s * s:
            continue
        cen = np.array([bl[i]["c"] for i in g_])
        u = np.mean([bl[i]["u"] * np.sign(bl[i]["u"][0] + 1e-9) for i in g_], 0)
        u /= np.linalg.norm(u) + 1e-9
        vv = np.sort(cen @ np.array([-u[1], u[0]]))
        gaps = np.diff(vv)
        reg = 1.0 - min(1.0, float(np.std(gaps) / (np.mean(gaps) + 1e-6))) if len(gaps) >= 2 else 0.3
        if reg < 0.45:
            continue
        conf = 0.55 * min(1.0, len(g_) / 8) + 0.45 * reg
        iss = ["anh_toi"] if dark else []
        if len(g_) < 5:
            iss.append("it_soc")
        if cv2.contourArea(poly.astype(np.float32)) > 0.12 * H * W:
            iss.append("polygon_rong_bat_thuong")
            conf *= 0.6
        out.append({"label": "lane/crosswalk", "polygon": _flat(poly), "conf": round(float(conf * (0.9 if dark else 1)), 3),
                    "_issues": iss, "_src": "stripes", "n_stripes": len(g_)})
    out.sort(key=lambda d: -d["conf"])
    return out[:3]


# ------------------------------------------------------------------ area
def _wedge(vp, yref: float, xl: float, xr: float, ya: float, yb: float) -> np.ndarray:
    vx, vy = vp

    def x(xx, y):
        return vx + (xx - vx) * (y - vy) / (yref - vy)
    return np.array([[x(xl, ya), ya], [x(xr, ya), ya], [x(xr, yb), yb], [x(xl, yb), yb]], np.float32)


def _poly_mask(poly: np.ndarray, W: int, H: int) -> np.ndarray:
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32).reshape(-1, 2)], 1)
    return m


def _road_xr_at(rm: np.ndarray, vp, yref: float) -> Tuple[Optional[float], Optional[float]]:
    """Mep trai/phai cua road tai y_ref (tranh goc cut o day anh)."""
    H, W = rm.shape
    y = int(min(H - 1, yref))
    best: Optional[Tuple[float, float]] = None
    best_w = -1.0
    for yy in range(y, max(0, y - 160), -4):
        xs = np.nonzero(rm[yy])[0]
        if xs.size:
            x0, x1 = float(xs.min()), float(xs.max())
            w = x1 - x0
            if x0 < W * 0.5 < x1 and w >= W * 0.25:
                return x0, x1
            if w > best_w:
                best_w = w
                best = (x0, x1)
    return best if best is not None else (None, None)


def build_area(rm: np.ndarray, vp, lanes: Sequence[Dict[str, Any]], dark: bool,
               xc: Optional[float] = None, crosswalks: Sequence[Sequence[float]] = ()) -> List[Dict[str, Any]]:
    H, W = rm.shape
    yref = Y_REF * H
    t, b = road_extent(rm)
    if b - t < 20:
        return []
    yb = b + 3
    ya = max(t, vp[1] + 0.02 * H)
    xc = W / 2.0 if xc is None else xc
    prior_half = LANE_W_PRIOR * W / 2
    marks = []
    for L in lanes:
        # tia o y_ref; chi tin neu day phu qua vung gan y_ref (hoac keo dai hop ly)
        marks.append({"xr": L["xr"], "yellow": L["yellow"], "double": L["double"], "conf": L["conf"]})
    rl, rr = _road_xr_at(rm, vp, yref)
    issues_d: List[str] = ["anh_toi"] if dark else []
    left = [mk for mk in marks if mk["xr"] < xc - 0.12 * prior_half]
    right = [mk for mk in marks if mk["xr"] > xc + 0.12 * prior_half]
    lmk = max(left, key=lambda k: k["xr"]) if left else None
    rmk = min(right, key=lambda k: k["xr"]) if right else None
    # nhin bien: ve phai 1 tia son neu < 1.85*nua be rong tien nghiem, nguoc lai dung mep road / tien nghiem
    xl = xr_ = None
    if lmk is not None and xc - lmk["xr"] <= 1.85 * prior_half:
        xl = lmk["xr"]
    elif rl is not None and rl < xc - 0.15 * prior_half and xc - rl <= 2.4 * prior_half:
        xl, lmk = rl, None
        issues_d.append("bien_trai=mep_road")
    else:
        xl, lmk = xc - prior_half, None
        issues_d.append("bien_trai=tien_nghiem")
    if rmk is not None and rmk["xr"] - xc <= 1.85 * prior_half:
        xr_ = rmk["xr"]
    elif rr is not None and rr > xc + 0.15 * prior_half and rr - xc <= 2.4 * prior_half:
        xr_, rmk = rr, None
        issues_d.append("bien_phai=mep_road")
    else:
        xr_, rmk = xc + prior_half, None
        issues_d.append("bien_phai=tien_nghiem")
    if xr_ - xl < 0.35 * prior_half:
        xl, xr_ = xc - prior_half, xc + prior_half
    wedge = _wedge(vp, yref, xl, xr_, ya, yb)
    direct = cv2.morphologyEx((rm & _poly_mask(wedge, W, H)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    shapes: List[Dict[str, Any]] = []
    dp = _largest_poly(direct, 2.5, 600)
    if dp is not None:
        c = 0.5 + 0.25 * (lmk is not None) + 0.25 * (rmk is not None)
        if dark:
            c *= 0.8
        shapes.append({"label": "area/direct", "polygon": dp, "conf": round(c * 0.9, 3), "_issues": issues_d, "_src": "road+edges"})
    # alternative: chi lan canh cung chieu, can vach trang phan lan o bien va road con tiep
    wlane = xr_ - xl
    for side, mk in (("trai", lmk), ("phai", rmk)):
        if mk is None or mk["yellow"] or mk["conf"] < 0.35:
            continue    # bien la mep road, hoac vach vang (nguoc chieu) -> khong ve alternative
        x0, x1 = (mk["xr"] - wlane, mk["xr"]) if side == "trai" else (mk["xr"], mk["xr"] + wlane)
        w2 = _wedge(vp, yref, x0, x1, ya, yb)
        alt = cv2.morphologyEx((rm & _poly_mask(w2, W, H) & (1 - cv2.dilate(direct, np.ones((5, 5), np.uint8)))).astype(np.uint8),
                               cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        ap = _largest_poly(alt, 2.5, CFG["min_alt_area"])
        if ap is not None and any((_poly_mask(np.float32(c).reshape(-1, 2), W, H) & alt).sum() > 0.15 * alt.sum() for c in crosswalks):
            ap = None   # nga tu / vach qua duong chen vao lan canh: khong doan
        if ap is not None:
            shapes.append({"label": "area/alternative", "polygon": ap, "conf": round(0.4 * (0.8 if dark else 1), 3),
                           "_issues": ["cung_chieu_chua_xac_minh", f"ben_{side}"] + (["anh_toi"] if dark else []), "_src": "road+edges"})
    return shapes


def _largest_poly(m: np.ndarray, eps: float, min_area: float) -> Optional[List[float]]:
    ps = mask_polys(m, eps, min_area)
    if not ps:
        return None
    ps.sort(key=lambda p: -cv2.contourArea(np.float32(p).reshape(-1, 2)))
    return ps[0]


# ------------------------------------------------------------------ vao chinh
def generate_frame(img: np.ndarray, roads: Any, continuity: str = "independent") -> Dict[str, Any]:
    """-> {"shapes": [...], "vp": (x,y), "vp_src": str, "dark": bool, "n_lines": int}.

    continuity="independent" (mac dinh): moi frame tu than. continuity="video" KHONG lam gi them o day: nguoi goi
    dung laneinterp.interpolate_lanes/interpolate_quad voi anchor keyframe (10 anh/lan) roi tu quyet dinh."""
    if continuity not in ("independent", "video"):
        raise ValueError(continuity)
    H, W = img.shape[:2]
    rm = road_mask(roads, W, H)
    if not rm.any():
        return {"shapes": [], "vp": None, "vp_src": "no_road", "dark": False, "n_lines": 0}
    enh, dark, med = brighten_for_paint(img, rm)
    # Union SegFormer masks on raw `img` + `enh` when dark (fixes f64/f72 while keeping f68/f71/f79 recall)
    seg_raw = _segformer_lane_mask(img, rm)
    seg_enh = _segformer_lane_mask(enh, rm) if dark else None
    if seg_raw is not None and seg_enh is not None:
        seg_u = (seg_raw > 0) | (seg_enh > 0)
        m = cv2.morphologyEx(seg_u.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) if int(seg_u.sum()) >= 150 else paint_mask(enh, rm, dark)
    elif seg_raw is not None and int(seg_raw.sum()) >= 150:
        m = cv2.morphologyEx(seg_raw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    else:
        m = paint_mask(enh, rm, dark)
    t, b = road_extent(rm)
    vp0 = vp_from_road(rm)
    src = "road_edges"
    if vp0 is None:
        vp0, src = (W / 2.0, max(t, 0.38 * H)), "prior"
    ybot = min(b, 0.86 * H)
    yref = Y_REF * H
    vp, base, best = refine_vp(m, cv2.dilate(rm, np.ones((11, 11), np.uint8)), vp0, yref,
                               max(t, vp0[1] + 0.03 * H), ybot, (120, 36))
    if best > base + 6:
        src += "+son"
    else:
        vp = vp0
    vp = (float(np.clip(vp[0], -0.3 * W, 1.3 * W)), float(np.clip(vp[1], 0.15 * H, 0.7 * H)))
    ytop = max(t, vp[1] + 0.03 * H)
    lanes = detect_ray_lanes(img, enh, m, rm, vp, dark, ytop, ybot)
    shapes: List[Dict[str, Any]] = []
    for L in lanes:
        d = {k: v for k, v in L.items() if k not in ("xr", "y0", "y1", "yellow", "double")}
        if src.startswith("prior"):
            d["_issues"] = d["_issues"] + ["vp_prior"]
        shapes.append(d)
    # Crosswalk needs zebra-stripe top-hat rather than longitudinal lane-line mask
    cw = detect_crosswalk_stripes(img, enh, rm, dark, paint_m=None)
    shapes += cw
    rl_y, rr_y = _road_xr_at(rm, vp, yref)
    xc_est = 0.5 * vp[0] + 0.5 * ((rl_y + rr_y) / 2.0) if (rl_y is not None and rr_y is not None) else None
    if xc_est is not None:
        xc_est = float(np.clip(xc_est, 0.32 * W, 0.68 * W))
    shapes += build_area(rm, vp, lanes, dark, xc=xc_est, crosswalks=[c["polygon"] for c in cw])
    return {"shapes": shapes, "vp": vp, "vp_src": src, "dark": dark, "n_lines": len(lanes)}
