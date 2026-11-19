"""Lam sac / muot polygon segmentation (numpy + cv2 + shapely, CPU, khong model).

refine(polygon, kind, img=None)      -> polygon phang [x1,y1,x2,y2,...] luon hop le
smooth_shared_boundary(a, b)         -> (a', b') bien chung lam muot mot lan, khong ho / chong
metrics(polygon)                     -> so do khach quan (dinh, % canh thang, rang cua, hop le)

Khong co GT: chi co the noi "sac/muot hon theo chi so", khong noi "dung hon".
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import shapely
from shapely import make_valid
from shapely.geometry import MultiPolygon, Polygon

Poly = Union[Sequence[float], np.ndarray]

KIND_PARAMS = {
    # eps: DP (px); snap: ep song song/vuong goc; axis: them 0/90 do; chaikin: so vong; max_points
    "building": dict(eps=1.5, snap=True, axis=True, chaikin=0, max_points=20),
    "artificial": dict(eps=1.5, snap=True, axis=True, chaikin=0, max_points=60),
    "organic": dict(eps=0.8, snap=False, axis=False, chaikin=1, max_points=60),
    "lane": dict(eps=1.0, snap=True, axis=False, chaikin=0, max_points=40),
}
MIN_AREA = 4.0


def kind_for_label(label: str) -> str:
    l = (label or "").lower()
    if l in ("building", "wall"):
        return "building"
    if l in ("vegetation", "terrain", "sky", "person"):
        return "organic"
    if l.startswith("lane/crosswalk") or l in ("pole", "road", "sidewalk", "fence",
                                                 "traffic sign", "traffic light"):
        return "artificial"
    return "lane"  # lane/*, area/*, khac: bao thu


# ----------------------------------------------------------------- io / hop le
def _to_xy(p: Poly) -> np.ndarray:
    a = np.asarray(p, dtype=float)
    return a.reshape(-1, 2) if a.ndim == 1 else a[:, :2]


def _flat(xy: np.ndarray) -> List[float]:
    return [round(float(v), 2) for v in np.asarray(xy).reshape(-1)]


def _largest(g) -> Optional[Polygon]:
    """Lay manh Polygon lon nhat (bo mang, manh nho); bo lo."""
    if g is None or g.is_empty:
        return None
    if isinstance(g, Polygon):
        parts = [g]
    elif hasattr(g, "geoms"):
        parts = []
        for s in g.geoms:
            if isinstance(s, Polygon):
                parts.append(s)
            elif hasattr(s, "geoms"):
                parts.extend(x for x in s.geoms if isinstance(x, Polygon))
    else:
        return None
    parts = [p for p in parts if p.area >= MIN_AREA]
    if not parts:
        return None
    best = max(parts, key=lambda p: p.area)
    return Polygon(best.exterior)


def valid_polygon(poly: Union[Polygon, np.ndarray]) -> Optional[Polygon]:
    if hasattr(poly, "geom_type") and not isinstance(poly, Polygon):
        poly = make_valid(poly) if not poly.is_valid else poly
        poly = _largest(poly)
        return poly if poly is not None and poly.is_valid else None
    if not isinstance(poly, Polygon):
        xy = _to_xy(poly)
        if len(xy) < 3:
            return None
        poly = Polygon(xy)
    if not poly.is_valid:
        poly = make_valid(poly)
    p = _largest(poly)
    if p is None or not p.is_valid:
        return None
    return p


def _ring(p: Polygon) -> np.ndarray:
    return np.asarray(p.exterior.coords)[:-1]


def _cap_points(p: Polygon, eps: float, max_points: int) -> Polygon:
    cur = p
    for _ in range(12):
        if len(cur.exterior.coords) - 1 <= max_points:
            break
        eps *= 1.3
        s = valid_polygon(cur.simplify(eps, preserve_topology=True))
        if s is None:
            break
        cur = s
    return cur


# ----------------------------------------------------------------- bam canh
def snap_to_edges(p: Polygon, img: np.ndarray, k: float = 3.0, spacing: float = 3.0,
                  min_grad: float = 6.0) -> Tuple[Polygon, float]:
    """Keo bien ve canh that trong dai +-k px quanh bien (chi cat vung quanh polygon, khong xu ly ca anh)."""
    import cv2

    H, W = img.shape[:2]
    dens = shapely.segmentize(p, spacing)
    P = _ring(dens)
    n = len(P)
    if n < 6:
        return p, 0.0
    x0 = int(max(0, math.floor(P[:, 0].min() - k - 4)))
    y0 = int(max(0, math.floor(P[:, 1].min() - k - 4)))
    x1 = int(min(W, math.ceil(P[:, 0].max() + k + 5)))
    y1 = int(min(H, math.ceil(P[:, 1].max() + k + 5)))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return p, 0.0
    crop = img[y0:y1, x0:x1]
    g = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    t = np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)
    t /= (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)
    nrm = np.stack([t[:, 1], -t[:, 0]], axis=1)
    ds = np.arange(-k, k + 0.01, 0.5, dtype=np.float32)
    sx = (P[:, 0:1] - x0 + 0.5 * 0)[:, :1] + nrm[:, 0:1] * ds[None, :]
    sy = (P[:, 1:2] - y0)[:, :1] + nrm[:, 1:2] * ds[None, :]
    mx = cv2.remap(gx, sx.astype(np.float32), sy.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    my = cv2.remap(gy, sx.astype(np.float32), sy.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mag = np.abs(mx * nrm[:, 0:1] + my * nrm[:, 1:2])
    w = np.exp(-0.5 * (ds / (0.6 * k)) ** 2)[None, :]
    best = np.argmax(mag * w, axis=1)
    off = ds[best].astype(float)
    conf = mag[np.arange(n), best] >= min_grad
    off[~conf] = 0.0
    # lam tron offset doc bien (trung vi vong + gauss) de khong tao rang cua moi
    def circ(a, fn):
        return fn(np.concatenate([a[-8:], a, a[:8]]))[8:-8]
    from numpy.lib.stride_tricks import sliding_window_view as swv
    def med(a, wd=5):
        pad = np.pad(a, (wd // 2, wd // 2), mode="wrap")
        return np.median(swv(pad, wd), axis=1)
    off = med(off, 5)
    ker = np.exp(-0.5 * (np.arange(-6, 7) / 2.5) ** 2); ker /= ker.sum()
    off = np.convolve(np.pad(off, (6, 6), mode="wrap"), ker, mode="valid")
    off = np.clip(off, -k, k)
    border = (P[:, 0] < 1.5) | (P[:, 1] < 1.5) | (P[:, 0] > W - 1.5) | (P[:, 1] > H - 1.5)
    off[border] = 0.0
    Q = P + nrm * off[:, None]
    q = valid_polygon(Q)
    if q is None:
        return p, 0.0
    return q, float(np.mean(np.abs(off)))


# ----------------------------------------------------------------- fit duong thang
def _ang(d: np.ndarray) -> float:
    return math.degrees(math.atan2(d[1], d[0])) % 180.0


def _adiff(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def straighten(ring: np.ndarray, axis: bool, min_len: float = 12.0, tol: float = 5.0,
               axis_tol: float = 3.0, max_shift: float = 6.0) -> np.ndarray:
    """Ep canh dai gan song song / vuong goc voi cac canh chu dao (va truc 0/90) ve dung huong."""
    n = len(ring)
    if n < 3:
        return ring
    nxt = np.roll(ring, -1, axis=0)
    vec = nxt - ring
    L = np.linalg.norm(vec, axis=1)
    ang = np.array([_ang(v) for v in vec])
    order = np.argsort(-L)
    dom: List[float] = []
    for i in order:
        if L[i] < min_len or len(dom) >= 3:
            break
        if all(_adiff(ang[i], d) > tol and _adiff(ang[i] + 90, d) > tol for d in dom):
            dom.append(float(ang[i]))
    if axis:  # canh chu dao gan truc -> ep han ve truc
        dom = [0.0 if _adiff(d, 0.0) <= axis_tol else 90.0 if _adiff(d, 90.0) <= axis_tol else d for d in dom]
    cands = list(dom)
    if axis:
        cands += [0.0, 90.0]
    cands += [(d + 90.0) % 180.0 for d in dom]
    new_ang = ang.copy()
    snapped = np.zeros(n, bool)
    for i in range(n):
        if L[i] < min_len or not cands:
            continue
        j = min(range(len(cands)), key=lambda c: _adiff(ang[i], cands[c]))
        d = _adiff(ang[i], cands[j])
        lim = axis_tol if (axis and cands[j] in (0.0, 90.0) and cands[j] not in dom) else tol
        if d <= lim:
            new_ang[i] = cands[j]
            snapped[i] = True
    if not snapped.any():
        return ring
    mid = (ring + nxt) / 2.0
    dirs = np.stack([np.cos(np.radians(new_ang)), np.sin(np.radians(new_ang))], axis=1)
    # chieu canh goc de giu huong
    sgn = np.sign(np.sum(dirs * vec, axis=1)); sgn[sgn == 0] = 1
    dirs *= sgn[:, None]
    out = ring.copy()
    for i in range(n):
        j = (i - 1) % n  # canh j=(i-1)->i ; canh i = i->i+1
        if not (snapped[i] or snapped[j]):
            continue
        a, da = mid[j], dirs[j]
        b, db = mid[i], dirs[i]
        den = da[0] * db[1] - da[1] * db[0]
        if abs(den) < math.sin(math.radians(10)):
            # gan song song: chieu dinh len duong da ep (neu co)
            ref, dr = (a, da) if snapped[j] else (b, db)
            v = ring[i] - ref
            cand = ref + dr * (v @ dr)
        else:
            s = ((b[0] - a[0]) * db[1] - (b[1] - a[1]) * db[0]) / den
            cand = a + da * s
        if np.linalg.norm(cand - ring[i]) <= max_shift:
            out[i] = cand
    return out


def _rdp_idx(P: np.ndarray, i0: int, i1: int, eps: float, out: list):
    a, b = P[i0], P[i1]
    ab = b - a
    la = np.linalg.norm(ab)
    seg = P[i0 + 1:i1]
    if len(seg) == 0:
        return
    d = np.linalg.norm(seg - a, axis=1) if la < 1e-9 else np.abs(ab[0] * (seg - a)[:, 1] - ab[1] * (seg - a)[:, 0]) / la
    k = int(np.argmax(d))
    if d[k] > eps:
        m = i0 + 1 + k
        _rdp_idx(P, i0, m, eps, out)
        out.append(m)
        _rdp_idx(P, m, i1, eps, out)


def fit_lines(P: np.ndarray, coarse_eps: float = 3.0, max_shift: float = 8.0) -> np.ndarray:
    """Chia vong thanh doan (DP tho), fit duong TLS tren diem goc cua moi doan, giao cac duong ke nhau."""
    n = len(P)
    if n < 8:
        return P
    j = int(np.argmax(np.linalg.norm(P - P[0], axis=1)))
    ids = [0]
    _rdp_idx(P, 0, j, coarse_eps, ids)
    ids.append(j)
    Q = np.vstack([P, P[:1]])
    sub: list = []
    _rdp_idx(Q, j, n, coarse_eps, sub)
    ids += sub
    if len(ids) < 3:
        return P
    m = len(ids)
    lines = []
    for t in range(m):
        i0, i1 = ids[t], ids[(t + 1) % m]
        pts = Q[i0:i1 + 1] if i1 > i0 else Q[i0:n + 1 + i1] if i1 else Q[i0:n + 1]
        if len(pts) > 4:
            pts = pts[1:-1] if len(pts) > 6 else pts
        c = pts.mean(0)
        if len(pts) >= 3 and np.linalg.norm(Q[i1] - Q[i0]) >= 6:
            u, sv, vt = np.linalg.svd(pts - c)
            d = vt[0]
        else:
            d = Q[i1] - Q[i0]
            d = d / (np.linalg.norm(d) + 1e-9)
            c = (Q[i0] + Q[i1]) / 2
        lines.append((c, d))
    out = []
    for t in range(m):
        v = P[ids[t]]
        (c0, d0), (c1, d1) = lines[t - 1], lines[t]
        den = d0[0] * d1[1] - d0[1] * d1[0]
        if abs(den) < math.sin(math.radians(10)):
            out.append(v)
            continue
        sct = ((c1[0] - c0[0]) * d1[1] - (c1[1] - c0[1]) * d1[0]) / den
        cand = c0 + d0 * sct
        out.append(cand if np.linalg.norm(cand - v) <= max_shift else v)
    return np.array(out)


def merge_collinear(ring: np.ndarray, max_turn: float = 8.0, max_dev: float = 2.0) -> np.ndarray:
    """Bo dinh gan thang hang (goc re nho va lech < max_dev px so voi day noi 2 dinh ke)."""
    r = [tuple(v) for v in ring]
    changed = True
    while changed and len(r) > 4:
        changed = False
        for i in range(len(r)):
            a, b, c = np.array(r[i - 1]), np.array(r[i]), np.array(r[(i + 1) % len(r)])
            ac = c - a
            la = np.linalg.norm(ac)
            if la < 1e-6:
                continue
            dev = abs(ac[0] * (b - a)[1] - ac[1] * (b - a)[0]) / la
            t = abs((_ang(b - a) - _ang(c - b) + 90) % 180 - 90)
            if dev <= max_dev and t <= max_turn:
                r.pop(i)
                changed = True
                break
    return np.array(r)


def chaikin(ring: np.ndarray, it: int = 2, w: float = 0.25) -> np.ndarray:
    r = ring
    for _ in range(it):
        nx = np.roll(r, -1, axis=0)
        q = (1 - w) * r + w * nx
        rr = w * r + (1 - w) * nx
        r = np.stack([q, rr], axis=1).reshape(-1, 2)
    return r


# ----------------------------------------------------------------- refine
def refine_poly(poly: Poly, kind: str = "artificial", img: Optional[np.ndarray] = None,
                snap_k: float = 3.0, max_points: Optional[int] = None,
                image_size: Optional[Tuple[int, int]] = None) -> Optional[Polygon]:
    """Tra shapely Polygon hop le (hoac None neu dau vao suy bien)."""
    kp = KIND_PARAMS.get(kind) or KIND_PARAMS["lane"]
    mp = max_points or kp["max_points"]
    p0 = valid_polygon(poly)
    if p0 is None:
        return None
    p = p0
    wd = 2.0 * p0.area / max(p0.length, 1e-6)  # be rong hieu dung: vat mong (cot, vach) dich it hon
    eps_e = min(kp["eps"], max(0.3, 0.12 * wd))
    k_e = min(snap_k, 0.3 * wd)
    if img is not None and k_e >= 0.75:
        p, _ = snap_to_edges(p, img, k=k_e)
    if kp["snap"] and kind == "artificial":
        f = valid_polygon(fit_lines(_ring(shapely.segmentize(p, 3.0)), coarse_eps=min(3.0, max(0.5, 0.3 * wd)),
                                    max_shift=min(8.0, 0.8 * wd)))
        if f is not None and f.intersection(p).area / max(f.union(p).area, 1e-9) > 0.9:
            p = f
    s = valid_polygon(p.simplify(eps_e, preserve_topology=True)) or p
    ring = _ring(s)
    if kp["snap"]:
        q = valid_polygon(merge_collinear(straighten(ring, axis=kp["axis"], max_shift=min(6.0, 0.6 * wd)),
                                          max_dev=min(2.0, 0.2 * wd)))
        if q is not None:
            s = q
    if kp["chaikin"]:
        q = valid_polygon(chaikin(_ring(s), kp["chaikin"]))
        if q is not None:
            s = valid_polygon(q.simplify(min(0.7, eps_e), preserve_topology=True)) or q
    s = valid_polygon(s.simplify(min(0.4, eps_e), preserve_topology=True)) or s
    s = _cap_points(s, kp["eps"], mp)
    W = H = None
    if img is not None:
        H, W = img.shape[:2]
    elif image_size:
        W, H = image_size
    if W:
        c = valid_polygon(s.intersection(shapely.box(0, 0, W, H)))
        if c is not None:
            s = c
    return s if s.is_valid and not s.is_empty else p0


def refine(polygon: Poly, kind: str = "artificial", img: Optional[np.ndarray] = None, **kw) -> List[float]:
    """polygon phang vao -> polygon phang ra (luon hop le; suy bien thi tra nguyen ban)."""
    r = refine_poly(polygon, kind, img, **kw)
    if r is None:
        return _flat(_to_xy(polygon))
    return _flat(_ring(r))


# ----------------------------------------------------------------- bien chung
def smooth_shared_boundary(poly_a: Poly, poly_b: Poly, sigma: float = 2.0, band: float = 6.0,
                           gap: float = 3.0, simplify_tol: float = 0.8, scale: int = 3,
                           max_points: int = 80) -> Tuple[List[float], List[float]]:
    """Lam muot bien chung cua 2 label ke nhau MOT lan roi chia lai: khong ho, khong chong.

    Raster cuc bo (chi bbox cua hop 2 polygon) o do phan giai x`scale`; trong dai +-band quanh
    bien chung, nhan gan theo argmax cua 2 mat na da lam mo (sigma px); ngoai dai giu nguyen.
    B = U - A nen khop tuyet doi; coverage_simplify giu canh chung dong nhat.
    """
    import cv2

    A0, B0 = valid_polygon(poly_a), valid_polygon(poly_b)
    if A0 is None or B0 is None:
        return _flat(_to_xy(poly_a)), _flat(_to_xy(poly_b))
    S = scale
    pad = band + gap + 4
    minx, miny, maxx, maxy = A0.union(B0).bounds
    ox, oy = math.floor(minx - pad), math.floor(miny - pad)
    w = int(math.ceil(maxx + pad) - ox)
    h = int(math.ceil(maxy + pad) - oy)

    def mask(p: Polygon) -> np.ndarray:
        m = np.zeros((h * S, w * S), np.uint8)
        pts = np.round((_ring(p) - [ox, oy]) * S).astype(np.int32)
        cv2.fillPoly(m, [pts], 1)
        return m

    ma, mb = mask(A0), mask(B0)
    k = int(2 * gap * S) | 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    U = cv2.morphologyEx(ma | mb, cv2.MORPH_CLOSE, ker)
    kb = int(2 * band * S) | 1
    kerb = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kb, kb))
    inband = (cv2.dilate(ma, kerb) & cv2.dilate(mb, kerb)).astype(bool)
    sa = cv2.GaussianBlur(ma.astype(np.float32), (0, 0), sigma * S)
    sb = cv2.GaussianBlur(mb.astype(np.float32), (0, 0), sigma * S)
    pick_a = sa >= sb
    lab_a = np.where(inband, pick_a, (ma > 0) | (((ma == 0) & (mb == 0)) & pick_a)) & (U > 0)
    lab_a = lab_a.astype(np.uint8)

    def vec(m: np.ndarray) -> Optional[Polygon]:
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        best = None
        for c in cs:
            if len(c) < 3:
                continue
            pg = valid_polygon((c.reshape(-1, 2) + 0.5) / S + [ox, oy])
            if pg is not None and (best is None or pg.area > best.area):
                best = pg
        return best.buffer(0.5 / S, join_style=2) if best is not None else None

    Ar = vec(lab_a)
    Ur = vec(U)
    if Ar is None or Ur is None:
        return _flat(_ring(A0)), _flat(_ring(B0))
    Br = valid_polygon(Ur.difference(Ar))
    if Br is None:
        return _flat(_ring(A0)), _flat(_ring(B0))
    A1 = B1 = None
    tol = simplify_tol
    for _ in range(10):  # coverage_simplify giu canh chung dong nhat; tang tol neu vuot max_points
        try:
            sa_, sb_ = shapely.coverage_simplify([Ar, Br], tol)
            A1, B1 = valid_polygon(sa_), valid_polygon(sb_)
        except Exception:
            A1 = valid_polygon(Ar.simplify(tol))
            B1 = valid_polygon(Br.simplify(tol))
            if A1 is not None and B1 is not None:
                B1 = valid_polygon(B1.difference(A1)) or B1
        if A1 is None or B1 is None:
            break
        if len(A1.exterior.coords) - 1 <= max_points and len(B1.exterior.coords) - 1 <= max_points:
            break
        tol *= 1.3
    if A1 is None or B1 is None:
        return _flat(_ring(A0)), _flat(_ring(B0))
    if A1.intersection(B1).area > 0.01:
        B1 = valid_polygon(B1.difference(A1)) or B1
    return _flat(_ring(A1)), _flat(_ring(B1))


def overlap_gap(a: Poly, b: Poly) -> Tuple[float, float]:
    """(dien tich chong, dien tich ho do duoc bang closing 3px)."""
    A, B = valid_polygon(a), valid_polygon(b)
    ov = A.intersection(B).area
    u = A.union(B)
    gap = u.buffer(3, join_style=2).buffer(-3, join_style=2).difference(u).area
    return float(ov), float(gap)


# ----------------------------------------------------------------- do
def metrics(poly: Poly, straight_len: float = 20.0, straight_tol: float = 3.0) -> Dict[str, float]:
    xy = _to_xy(poly)
    raw_valid = bool(len(xy) >= 3 and Polygon(xy).is_valid)
    p = valid_polygon(xy)
    if p is None:
        return dict(n=len(xy), valid=raw_valid, straight=0.0, jag=0.0, perim=0.0)
    R = _ring(p)
    nxt = np.roll(R, -1, axis=0)
    vec_ = nxt - R
    L = np.linalg.norm(vec_, axis=1)
    keep = L > 1e-6
    vec_, L = vec_[keep], L[keep]
    ang = np.degrees(np.arctan2(vec_[:, 1], vec_[:, 0]))
    # run: canh lien tiep gan thang hang (doi huong < tol) -> nhom, tinh do dai; run >= straight_len => "thang"
    m = len(L)
    turn = np.abs((ang - np.roll(ang, 1) + 180) % 360 - 180)
    breaks = np.where(turn > straight_tol)[0]
    tot = L.sum()
    st = 0.0
    if len(breaks) == 0:
        st = tot if tot >= straight_len else 0.0
    else:
        for bi in range(len(breaks)):
            s = breaks[bi]
            e = breaks[(bi + 1) % len(breaks)]
            idx = np.arange(s, e if e > s else e + m) % m
            ln = L[idx].sum()
            if ln >= straight_len:
                st += ln
    hull = p.convex_hull.exterior.length
    return dict(n=int(len(R)), valid=raw_valid, straight=float(st / tot) if tot else 0.0,
                jag=float(tot / hull) if hull else 0.0, perim=float(tot))


def iou(a: Poly, b: Poly) -> float:
    A, B = valid_polygon(a), valid_polygon(b)
    if A is None or B is None:
        return 0.0
    u = A.union(B).area
    return float(A.intersection(B).area / u) if u else 0.0


# ----------------------------------------------------------------- bench job 285 (doc-only)
def _find_root():
    from pathlib import Path
    for p in Path(__file__).resolve().parents:
        if (p / "tools" / "cvat-seg-copilot").exists():
            return p
    raise FileNotFoundError("repo root")


def bench(frames_show=(61, 63, 70)):
    import json, csv, cv2
    root = _find_root()
    drafts = root / "cvat/week01/job285-bbox/drafts"
    fr_dir = root / "tools/cvat-seg-copilot/frames"
    out = root / "cvat/data/benchmarks/refine_job285"
    out.mkdir(parents=True, exist_ok=True)
    items = []  # (frame, label, pts, src)
    for n in ["bboxfull", "bboxbright", "bboxbright2", "bboxbright3", "lanearea", "lanearea61"]:
        d = json.load(open(drafts / f"LATEST_{n}.json", encoding="utf-8"))
        for s in d["shapes"]:
            if s["type"] == "polygon" and len(s["points"]) >= 6:
                items.append((s["frame"], s.get("_label", "?"), s["points"], n))
    imgs: Dict[int, np.ndarray] = {}
    rows, agg = [], {}
    refined: Dict[int, List] = {}
    for f, lab, pts, src in items:
        if f not in imgs:
            imgs[f] = cv2.imread(str(fr_dir / f"j285_f{f}.jpg"))
        kind = kind_for_label(lab)
        new = refine(pts, kind, imgs[f])
        m0, m1 = metrics(pts), metrics(new)
        i = iou(pts, new)
        rows.append(dict(frame=f, label=lab, src=src, kind=kind, n0=m0["n"], n1=m1["n"], st0=round(m0["straight"], 3),
                         st1=round(m1["straight"], 3), jag0=round(m0["jag"], 3), jag1=round(m1["jag"], 3),
                         valid0=m0["valid"], valid1=m1["valid"], iou=round(i, 4)))
        rows[-1]["iou_geom"] = round(iou(pts, refine(pts, kind)), 4)
        refined.setdefault(f, []).append((pts, new, lab))
    with open(out / "refine_metrics.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # tong hop
    def mean(k, rs):
        return float(np.mean([r[k] for r in rs])) if rs else 0.0
    summ = {}
    for kind in ["all", "artificial", "organic", "lane"]:
        rs = rows if kind == "all" else [r for r in rows if r["kind"] == kind]
        summ[kind] = dict(count=len(rs), n0=mean("n0", rs), n1=mean("n1", rs), straight0=mean("st0", rs),
                          straight1=mean("st1", rs), jag0=mean("jag0", rs), jag1=mean("jag1", rs),
                          valid0=sum(bool(r["valid0"]) for r in rs), valid1=sum(bool(r["valid1"]) for r in rs),
                          iou_mean=mean("iou", rs), iou_min=min([r["iou"] for r in rs] or [0]),
                          iou_lt_097=sum(r["iou"] < 0.97 for r in rs))
    # bien chung: cap ke nhau cung frame
    pairs, seen = [], set()
    for f, lst in refined.items():
        by = {}
        for pts, new, lab in lst:
            by.setdefault(lab, []).append(pts)
        for la, lb in [("area/direct", "area/alternative"), ("road", "sidewalk")]:
            for pa in by.get(la, []):
                for pb in by.get(lb, []):
                    A, B = valid_polygon(pa), valid_polygon(pb)
                    if A is None or B is None or A.distance(B) > 3:
                        continue
                    if A.intersection(B).area > 0.2 * min(A.area, B.area):
                        continue  # chong nhau lon = khong phai 2 label ke nhau
                    shared = A.buffer(3).intersection(B.boundary).length
                    if shared < 20:
                        continue
                    key = (f, tuple(np.round(pa, 1)), tuple(np.round(pb, 1)))
                    if key in seen:
                        continue
                    seen.add(key)
                    a2, b2 = smooth_shared_boundary(pa, pb)
                    o0, g0 = overlap_gap(pa, pb)
                    o1, g1 = overlap_gap(a2, b2)
                    pairs.append(dict(frame=f, pair=f"{la}|{lb}", shared_px=round(shared, 1),
                                      ov0=round(o0, 1), gap0=round(g0, 1), ov1=round(o1, 1), gap1=round(g1, 1),
                                      iouA=round(iou(pa, a2), 4), iouB=round(iou(pb, b2), 4),
                                      nA=(metrics(pa)["n"], metrics(a2)["n"]), nB=(metrics(pb)["n"], metrics(b2)["n"])))
    pa = lambda k: float(np.sum([p[k] for p in pairs])) if pairs else 0.0
    summ["shared_pairs_total"] = dict(count=len(pairs), ov0=pa("ov0"), gap0=pa("gap0"), ov1=pa("ov1"), gap1=pa("gap1"),
                                      iou_min=min([min(p["iouA"], p["iouB"]) for p in pairs] or [0]),
                                      pairs_ov1_gt_5px2=sum(p["ov1"] > 5 for p in pairs))
    summ["iou_geom_only_mean"] = float(np.mean([r["iou_geom"] for r in rows]))
    summ["iou_geom_only_lt_097"] = int(sum(r["iou_geom"] < 0.97 for r in rows))
    json.dump(dict(summary=summ, shared_pairs=pairs), open(out / "summary.json", "w", encoding="utf-8"), indent=1)
    # anh before/after
    for f in frames_show:
        img = imgs.get(f)
        if img is None:
            img = cv2.imread(str(fr_dir / f"j285_f{f}.jpg"))
        L, R = img.copy(), img.copy()
        for pts, new, lab in refined.get(f, []):
            cv2.polylines(L, [np.round(_to_xy(pts)).astype(np.int32)], True, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.polylines(R, [np.round(_to_xy(new)).astype(np.int32)], True, (0, 200, 0), 1, cv2.LINE_AA)
        cv2.putText(L, f"before f{f}", (8, 24), 0, 0.7, (0, 0, 255), 2)
        cv2.putText(R, f"after f{f}", (8, 24), 0, 0.7, (0, 160, 0), 2)
        cv2.imwrite(str(out / f"f{f}_before_after.jpg"), np.hstack([L, R]), [cv2.IMWRITE_JPEG_QUALITY, 90])
        # zoom: polygon doi nhieu nhat (IoU thap nhat, dien tich du lon)
        cand = [(iou(a, b), a, b) for a, b, _ in refined.get(f, []) if (valid_polygon(a) is not None and valid_polygon(a).area > 400)]
        if cand:
            _, a, b = min(cand, key=lambda t: t[0])
            xy = _to_xy(a); x0, y0 = np.maximum(0, xy.min(0) - 10).astype(int); x1, y1 = (xy.max(0) + 10).astype(int)
            c0, c1 = img[y0:y1, x0:x1].copy(), img[y0:y1, x0:x1].copy()
            z = 4
            c0 = cv2.resize(c0, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
            c1 = cv2.resize(c1, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
            cv2.polylines(c0, [np.round((xy - [x0, y0] + 0.5) * z).astype(np.int32)], True, (0, 0, 255), 1)
            cv2.polylines(c1, [np.round((_to_xy(b) - [x0, y0] + 0.5) * z).astype(np.int32)], True, (0, 200, 0), 1)
            cv2.imwrite(str(out / f"f{f}_zoom_worst_iou.jpg"), np.hstack([c0, c1]))
    return summ, pairs


if __name__ == "__main__":
    import json
    s, p = bench()
    print(json.dumps(s, indent=1))
