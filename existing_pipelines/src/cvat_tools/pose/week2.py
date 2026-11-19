"""Nháp skeleton cho job Week 2 (face 50 điểm / person_vf17) bằng công cụ Day 4 `tools/skeleton-pose-manager` (spm).

CHỈ SINH NHÁP LOCAL: không bao giờ ghi lên CVAT. Nháp lưu ở week02/<task>/<job>/drafts + ảnh xem ở review/.
- person_vf17: YOLO pose (spm.draft.person_drafts), sublabel '1'..'17' khớp thứ tự COCO.
- face 50 điểm: không có model landmark 50 điểm; đặt template (lấy từ svg của label trên CVAT) theo mắt + mũi do YOLO tìm
  (similarity fit, giống spm.face_draft). Chỉ ở mức gần đúng, người gán PHẢI chỉnh từng điểm.
Trái/phải của mặt theo quy ước guideline: `trai` = bên TRÁI ảnh (mắt có x nhỏ hơn).
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from cvat_tools.core import paths
from cvat_tools.core.client import Cvat

SPM_SRC = paths.REPO / "tools" / "skeleton-pose-manager" / "src"
FACE_LABELS = ("longmaytrai", "longmayphai", "songmui", "mattrai", "matphai", "moingoai", "moitrong")
LOW_CONF_Q = 0.5      # pose: điểm YOLO conf < 0.5 được đánh occluded (Q) trong nháp
CHUNK = 10


def load_policy(folder: Path) -> dict:
    """push_policy.json cạnh SPEC.md: skip_frames (vẽ tay, không sinh nháp), q_labels {frame: [label,...]} (đánh Q cả label)."""
    f = folder / "push_policy.json"
    return json.loads(f.read_text(encoding="utf8")) if f.exists() else {}


def _spm():
    if str(SPM_SRC) not in sys.path:
        sys.path.insert(0, str(SPM_SRC))
    from spm import draft as D          # noqa: WPS433 (lazy: nặng, chỉ nạp khi chạy)
    from spm.transforms import apply_similarity, similarity_fit
    return D, similarity_fit, apply_similarity


def job_folder(job_id: int) -> Path:
    hits = sorted((paths.WORKSPACE).glob(f"week*/*/job{job_id}-*"))
    if not hits:
        raise FileNotFoundError(f"Chưa có thư mục week*/<task>/job{job_id}-* (xem week02/README.md).")
    return hits[-1]


def template_from_label(label: dict) -> dict:
    """{tên sublabel: (x, y)} chuẩn hóa [0,1], lấy từ svg của label skeleton trên CVAT."""
    nodes = {int(e): (float(x) / 100, float(y) / 100)
             for e, x, y in re.findall(r'data-element-id="(\d+)"[^>]*?cx="([\d.]+)" cy="([\d.]+)"', label["svg"])}
    if not nodes:
        nodes = {int(e): (float(x) / 100, float(y) / 100)
                 for x, y, e in re.findall(r'cx="([\d.]+)" cy="([\d.]+)"[^>]*?data-element-id="(\d+)"', label["svg"])}
    names = [s["name"] for s in label["sublabels"]]
    return {names[i - 1]: xy for i, xy in nodes.items()}


def detect_kind(labels: list) -> str:
    names = {l["name"] for l in labels}
    if "person_vf17" in names:
        return "person17"
    if set(FACE_LABELS) <= names:
        return "face50"
    raise ValueError(f"Job không có bộ label Week 2 quen thuộc: {sorted(names)}")


def _clip(x, y, w, h):
    return float(min(max(x, 0.0), w)), float(min(max(y, 0.0), h))


def _state(x, y, w, h):
    return "outside" if x < 0 or y < 0 or x > w or y > h else "visible"


def face_frame(model, img, frame, tpls: dict) -> dict:
    """7 skeleton mặt cho 1 ảnh. tpls: {label: {sub: (x,y)}}."""
    D, similarity_fit, apply_similarity = _spm()
    h, w = img.shape[:2]
    eyeL = np.mean(list(tpls["mattrai"].values()), axis=0)    # mắt bên trái ảnh (tâm template)
    eyeR = np.mean(list(tpls["matphai"].values()), axis=0)
    nose_name = list(tpls["songmui"])[-1]                      # điểm cuối sống mũi ~ chóp mũi
    nose_t = np.array(tpls["songmui"][nose_name])
    flags, anchor, fit, nose_xy = [], None, None, None
    people = [d for d in D.detect_people(model, img, conf=0.15) if d["score"] >= 0.2]
    if people:
        d = max(people, key=lambda d: d["score"])
        kc, xy = d["kconf"], d["xy"]
        if kc[0] > 0.5 and kc[1] > 0.5 and kc[2] > 0.5:
            e1, e2 = xy[1], xy[2]
            left, right = (e1, e2) if e1[0] <= e2[0] else (e2, e1)   # theo vị trí trên ảnh, không theo giải phẫu
            fit = similarity_fit([eyeL, eyeR, nose_t], [left, right, xy[0]])
            anchor = "yolo(mắt+mũi)"
    if fit is None:
        box = D._haar_face(img)
        if box is not None:
            x, y, fw, fh = box
            fit, anchor = (fw, 0.0, x, y), "haar(hộp mặt)"
    if fit is None:
        side = 0.6 * min(w, h)
        fit, anchor, flags = (side, 0.0, (w - side) / 2, (h - side) / 2), "mặc định(giữa ảnh)", ["không_tìm_thấy_mặt"]
    out = []
    for lab in FACE_LABELS:
        placed = apply_similarity({n: {"x": x, "y": y} for n, (x, y) in tpls[lab].items()}, fit)
        pts = {}
        for n, p in placed.items():
            cx, cy = _clip(p["x"], p["y"], w, h)
            pts[n] = {"x": cx, "y": cy, "state": _state(p["x"], p["y"], w, h)}
        out.append({"label": lab, "points": pts, "anchor": anchor, "flags": flags + ["template_gần_đúng"]})
    return {"frame": frame, "width": w, "height": h, "people": out}


def person_frame(model, img, frame, sub_names: list) -> dict:
    D, _, _ = _spm()
    fr = D.person_drafts(model, img, frame)
    coco = list(fr["people"][0]["points"]) if fr["people"] else []
    out = []
    for p in fr["people"][:1]:                                   # guideline: 1 skeleton chính / ảnh
        pts = {}
        for i in range(17):
            q = dict(p["points"][coco[i]])
            if q["state"] == "visible" and (q.get("conf") or 0.0) < LOW_CONF_Q:
                q["state"] = "occluded"      # conf thấp: đánh Q để bạn nhìn thấy điểm đáng ngờ khi duyệt
            pts[sub_names[i]] = q
        out.append({"label": "person_vf17", "points": pts, "score": p["score"], "flags": p["flags"], "anchor": "yolo"})
    return {"frame": frame, "width": fr["width"], "height": fr["height"], "people": out, "skipped": fr["skipped"]}


class _OfflineCvat:
    """Mất mạng: đọc cache local data/jobs/<profile>/<job>/{job_info.json,labels.json} và ảnh frames_dir (do draft online ghi lại)."""

    def __init__(self, job_id: int, profile: str):
        self.d = paths.DATA / "jobs" / profile / str(job_id)
        self.info = json.loads((self.d / "job_info.json").read_text(encoding="utf8"))
        self.frames_dir = paths.WORKSPACE / self.info["frames_dir"]

    def labels(self, job_id): return json.loads((self.d / "labels.json").read_text(encoding="utf8"))
    def job(self, job_id): return {"start_frame": self.info["start_frame"], "assignee": None}
    def annotations(self, job_id): return {"shapes": [], "tracks": [], "tags": []}
    def meta(self, job_id): return {"frames": [{"name": n} for n in self.info["names"]]}

    def frame(self, job_id, n):
        return (self.frames_dir / self.info["names"][n - self.info["start_frame"]].split("/")[-1]).read_bytes()


def _cache_job(c, job_id: int, profile: str, labels: list, job: dict, meta: list) -> Path:
    """Ghi cache job (labels, thông tin frame) để lần sau chạy --offline được."""
    d = paths.DATA / "jobs" / profile / str(job_id)
    (d / "frames").mkdir(parents=True, exist_ok=True)
    (d / "labels.json").write_text(json.dumps(labels, ensure_ascii=False), encoding="utf8")
    info = {"start_frame": job["start_frame"], "names": [m["name"] for m in meta], "frames_dir": f"data/jobs/{profile}/{job_id}/frames"}
    old = d / "job_info.json"
    if old.exists():                                   # giữ frames_dir đã trỏ tay (vd cache dùng chung của task)
        info["frames_dir"] = json.loads(old.read_text(encoding="utf8")).get("frames_dir", info["frames_dir"])
    old.write_text(json.dumps(info, ensure_ascii=False), encoding="utf8")
    return d / "frames"


def _person17_frame(img, n, backend, sub_names) -> dict:
    """Pose 17 điểm: xoay 4 hướng nếu cần, chọn tài xế, kiểm hình học; vô lý/không thấy người -> people rỗng (vẽ tay)."""
    from cvat_tools.pose import person17
    h, w = img.shape[:2]
    det = person17.detect(backend, img)
    if det is None:
        return {"frame": n, "width": w, "height": h, "people": [], "reject": ["không thấy người"]}
    bad = person17.plausibility_pose(det["xy"], det["conf"])
    if bad:
        return {"frame": n, "width": w, "height": h, "people": [], "reject": bad}
    return {"frame": n, "width": w, "height": h, "people": person17.frame_people(det, sub_names, w, h)}


def _face50_frame(img, n, fl, rot_frames, model, tpls) -> dict:
    """FAN (landmark thật). Không thấy mặt hoặc hình học vô lý -> people rỗng (frame để vẽ tay, KHÔNG đoán bừa bằng template)."""
    from cvat_tools.pose import face50
    h, w = img.shape[:2]
    det = fl.landmarks68(img, try_rotations=n in rot_frames)
    if det is None:
        return {"frame": n, "width": w, "height": h, "people": [], "reject": ["không thấy mặt"]}
    bad = face50.plausibility(face50.map68_to_50(det["p68"]))
    if bad:
        return {"frame": n, "width": w, "height": h, "people": [], "reject": bad}
    return {"frame": n, "width": w, "height": h, "people": face50.frame_people(det, w, h)}


def draft_job(job_id: int, profile: str = "edufun", include_existing: bool = False, backend: str = "auto", honor_skip: bool = True, offline: bool = False) -> dict:
    import cv2
    D, _, _ = _spm()
    c = _OfflineCvat(job_id, profile) if offline else Cvat.from_env(profile)
    labels, job = c.labels(job_id), c.job(job_id)
    include_existing = include_existing or offline
    kind = detect_kind(labels)
    have = {s["frame"] for s in c.annotations(job_id)["shapes"]}
    meta = c.meta(job_id)["frames"]
    cache = None if offline else _cache_job(c, job_id, profile, labels, job, meta)
    model, mname = D.load_model()
    by = {l["name"]: l for l in labels}
    tpls = {n: template_from_label(by[n]) for n in FACE_LABELS} if kind == "face50" else None
    sub = [s["name"] for s in by["person_vf17"]["sublabels"]] if kind == "person17" else None
    folder = job_folder(job_id)
    (folder / "drafts").mkdir(exist_ok=True)
    (folder / "review" / "draft_preview").mkdir(parents=True, exist_ok=True)
    policy = load_policy(folder)
    fl, rot_frames = None, set()
    if kind == "face50":
        from cvat_tools.pose import face50
        fl = face50.FaceLandmarker()
        cj = folder.parent / "cases.json"       # frame fisheye/nằm ngang -> thử cả 4 hướng
        if cj.exists():
            rot_frames = {int(k) for k, v in json.loads(cj.read_text(encoding="utf8")).items() if any("Fisheye" in c["case"] for c in v)}
    skip_hand = set(policy.get("skip_frames", [])) if honor_skip else set()
    pbe = None
    if kind == "person17":
        from cvat_tools.pose import person17
        pbe = person17.make_backend(backend)
    qmap = {int(k): v for k, v in policy.get("q_labels", {}).items()}
    frames, done, skipped, by_hand, rejected = [], 0, [], [], {}
    for i, mf in enumerate(meta):
        n = job["start_frame"] + i
        if n in have and not include_existing:
            skipped.append(n)
            continue
        if n in skip_hand:
            by_hand.append(n)
            continue
        raw = c.frame(job_id, n)
        if cache is not None and not (cache / meta[i]["name"].split("/")[-1]).exists():
            (cache / meta[i]["name"].split("/")[-1]).write_bytes(raw)
        img = D.decode(raw)
        if kind == "face50":
            fr = _face50_frame(img, n, fl, rot_frames, model, tpls)
        else:
            fr = _person17_frame(img, n, pbe, sub)
        for p in fr["people"]:
            if p["label"] in qmap.get(n, []):
                for pt in p["points"].values():
                    if pt["state"] == "visible":
                        pt["state"] = "occluded"
        if not fr["people"]:
            by_hand.append(n)
            rejected[n] = fr.get("reject", [])
            continue
        fr["name"] = mf["name"]
        frames.append(fr)
        _preview(img, fr, folder / "review" / "draft_preview" / f"{mf['name'].split('/')[-1][:-4]}.jpg", kind, cv2)
        done += 1
    ts = datetime.now()
    tag = f"{kind}_{backend}" if kind == "person17" else kind
    out = folder / "drafts" / f"{tag}_{ts:%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({"job": job_id, "profile": profile, "kind": kind, "backend": backend if kind == "person17" else "fan2d", "model": mname, "created": ts.isoformat(timespec="seconds"),
                               "status": "ai_draft_local_only", "frames": frames}, ensure_ascii=False), encoding="utf8")
    return {"job": job_id, "kind": kind, "drafted": done, "skipped_existing": skipped, "by_hand": by_hand, "rejected": rejected, "file": str(out), "model": mname}


def _shape(label: dict, p: dict, frame: int) -> dict:
    """Skeleton CVAT. source='auto' vì là nháp AI CHƯA duyệt (không giả làm nhãn tay)."""
    ids = {s["name"]: s["id"] for s in label["sublabels"]}
    els = []
    for s in label["sublabels"]:
        pt = p["points"][s["name"]]
        if not (np.isfinite(pt["x"]) and np.isfinite(pt["y"])):
            raise ValueError(f"tọa độ không hữu hạn ở điểm {s['name']}")
        els.append({"type": "points", "frame": frame, "label_id": ids[s["name"]], "group": 0, "source": "auto",
                    "occluded": pt["state"] == "occluded", "outside": pt["state"] == "outside", "z_order": 0, "rotation": 0,
                    "points": [round(pt["x"], 2), round(pt["y"], 2)], "attributes": []})
    return {"type": "skeleton", "frame": frame, "label_id": label["id"], "group": 0, "source": "auto", "occluded": False,
            "outside": False, "z_order": 0, "rotation": 0, "points": [], "attributes": [], "elements": els}


def _my_auto_status(c: Cvat, job_id: int, folder: Path, tol: float = 1.5):
    """Skeleton auto do công cụ này tạo (theo mọi run_*.json) và frame BẠN đã sửa (lệch >tol px hoặc đổi cờ Q/O so với nháp lúc đẩy).
    Trả (mine {shape_id: frame}, protected_frames). Frame trong metrics/gold.json luôn được bảo vệ."""
    import math
    labels = c.labels(job_id)
    subn = {s["id"]: s["name"] for l in labels for s in l["sublabels"]}
    lname = {l["id"]: l["name"] for l in labels}
    shapes = c.annotations(job_id)["shapes"]
    mine, edited = {}, set()
    for rf in sorted((folder / "backups").glob("run_*.json")):
        run = json.loads(rf.read_text(encoding="utf8"))
        dd = json.loads((folder / "drafts" / run["draft"]).read_text(encoding="utf8"))
        pts = {(fr["frame"], p["label"], n): pt for fr in dd["frames"] for p in fr["people"] for n, pt in p["points"].items()}
        ids = set(run["created_ids"])
        for s in shapes:
            if s["id"] not in ids:
                continue
            mine[s["id"]] = s["frame"]
            for e in s["elements"]:
                k = (s["frame"], lname[s["label_id"]], subn[e["label_id"]])
                if k not in pts:
                    continue
                q = pts[k]
                moved = math.hypot(e["points"][0] - q["x"], e["points"][1] - q["y"]) > tol
                if moved or e["occluded"] != (q["state"] == "occluded") or e["outside"] != (q["state"] == "outside"):
                    edited.add(s["frame"])
    gf = folder / "metrics" / "gold.json"
    gold = {int(k) for k in json.loads(gf.read_text(encoding="utf8"))} if gf.exists() else set()
    return mine, edited | gold


def apply_draft(job_id: int, profile: str = "edufun", file: str | None = None, yes: bool = False, replace_auto: bool = False) -> dict:
    """Ghi nháp lên CVAT: dry-run mặc định. Chỉ job được gán cho chính mình; backup trước; đọc lại xác minh; run log để rollback.
    Mặc định chỉ THÊM và bỏ qua frame đã có skeleton. --replace-auto: xóa skeleton auto cũ của công cụ trên frame BẠN CHƯA SỬA rồi tạo lại
    từ nháp mới (frame bạn đã sửa, frame source=file và nhãn tay luôn giữ nguyên)."""
    c = Cvat.from_env(profile, write_enabled=yes)
    me, job = c.me(), c.job(job_id)
    if (job.get("assignee") or {}).get("username") != me["username"]:
        raise PermissionError(f"Job {job_id} không được gán cho bạn ({me['username']}): từ chối ghi.")
    folder = job_folder(job_id)
    f = Path(file) if file else max((folder / "drafts").glob("*.json"), key=lambda p: p.stat().st_mtime)   # nháp MỚI NHẤT theo thời gian
    draft = json.loads(f.read_text(encoding="utf8"))
    labels = {l["name"]: l for l in c.labels(job_id)}
    existing = c.annotations(job_id)
    protected, mine, del_ids = set(), {}, []
    if replace_auto:
        mine, protected = _my_auto_status(c, job_id, folder)
        del_ids = [sid for sid, fr in mine.items() if fr not in protected]
    gone = set(del_ids)
    have = {s["frame"] for s in existing["shapes"] if s["id"] not in gone}
    new, skipped = [], []
    for fr in draft["frames"]:
        if fr["frame"] in have or fr["frame"] in protected:
            skipped.append(fr["frame"])
            continue
        for p in fr["people"]:
            new.append(_shape(labels[p["label"]], p, fr["frame"]))
    summary = {"job": job_id, "draft": f.name, "to_create": len(new), "frames": sorted({s["frame"] for s in new}),
               "to_delete": len(del_ids), "delete_frames": sorted({mine[x] for x in del_ids}),
               "protected_frames": sorted(protected), "skipped_have_shapes": skipped, "before": len(existing["shapes"]), "applied": False}
    if not yes or not (new or del_ids):
        return summary
    bk = folder / "backups" / f"before_{datetime.now():%Y%m%d_%H%M%S}.json"
    bk.parent.mkdir(exist_ok=True)
    bk.write_text(json.dumps(existing, ensure_ascii=False), encoding="utf8")
    byid = {s["id"]: s for s in existing["shapes"]}
    for i in range(0, len(del_ids), CHUNK):
        c.delete_shapes(job_id, [byid[x] for x in del_ids[i:i + CHUNK]])
    for i in range(0, len(new), CHUNK):
        c.create_shapes(job_id, new[i:i + CHUNK])
    after = c.annotations(job_id)
    old_ids = {s["id"] for s in existing["shapes"]}
    created = [s for s in after["shapes"] if s["id"] not in old_ids]
    ok = len(after["shapes"]) == len(existing["shapes"]) - len(del_ids) + len(new) and all(s.get("elements") for s in created)
    run = folder / "backups" / f"run_{datetime.now():%Y%m%d_%H%M%S}.json"
    run.write_text(json.dumps({"job": job_id, "profile": profile, "backup": str(bk), "draft": f.name, "created_ids": [s["id"] for s in created],
                               "deleted_ids": del_ids, "before": len(existing["shapes"]), "after": len(after["shapes"]), "verified": ok},
                              ensure_ascii=False), encoding="utf8")
    return {**summary, "applied": True, "after": len(after["shapes"]), "verified": ok, "run": str(run)}


def rollback(run_file: str, yes: bool = False) -> dict:
    """Hoàn tác đúng một lần chạy: xóa skeleton lần đó tạo (theo id) rồi dựng lại skeleton nó đã xóa (lấy từ file backup)."""
    run = json.loads(Path(run_file).read_text(encoding="utf8"))
    c = Cvat.from_env(run.get("profile", "edufun"), write_enabled=yes)
    ann = c.annotations(run["job"])
    mine = [s for s in ann["shapes"] if s["id"] in set(run["created_ids"])]
    restore = []
    if run.get("deleted_ids"):
        bk = json.loads(Path(run["backup"]).read_text(encoding="utf8"))
        want = set(run["deleted_ids"])
        for s in bk["shapes"]:
            if s["id"] in want:
                t = {k: v for k, v in s.items() if k != "id"}
                t["elements"] = [{k: v for k, v in e.items() if k != "id"} for e in s.get("elements", [])]
                restore.append(t)
    if not yes:
        return {"job": run["job"], "would_delete": len(mine), "would_restore": len(restore), "applied": False}
    for i in range(0, len(mine), CHUNK):
        c.delete_shapes(run["job"], mine[i:i + CHUNK])
    for i in range(0, len(restore), CHUNK):
        c.create_shapes(run["job"], restore[i:i + CHUNK])
    return {"job": run["job"], "deleted": len(mine), "restored": len(restore),
            "remaining": len(c.annotations(run["job"])["shapes"]), "expected": run["before"], "applied": True}


_COLORS = [(255, 100, 100), (100, 220, 100), (60, 60, 255), (0, 200, 255), (255, 0, 255), (255, 160, 0), (160, 160, 0)]


def _preview(img, fr, path: Path, kind: str, cv2):
    im = img.copy()
    for k, p in enumerate(fr["people"]):
        col = _COLORS[k % len(_COLORS)]
        for n, pt in p["points"].items():
            cv2.circle(im, (int(pt["x"]), int(pt["y"])), 3, col, -1)
            if kind == "person17":
                cv2.putText(im, n, (int(pt["x"]) + 4, int(pt["y"]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    cv2.imwrite(str(path), im)
