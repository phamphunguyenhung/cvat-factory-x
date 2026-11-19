"""Bang nhan -> cach sinh nhap hien co. Chi LIET KE (doc file/regex/json), KHONG import model nang."""
from __future__ import annotations

import json
import re

from cvat_tools.core import paths
from cvat_tools.job import ruleset as RS

MEASURED = "có model + đã đo"
MODEL_UNMEASURED = "có model, chưa đo"
RULE = "có luật chưa đo"
MANUAL = "chỉ có thể vẽ tay"
MISSING = "THIẾU"
ORDER = [MEASURED, MODEL_UNMEASURED, RULE, MANUAL, MISSING]
NEED_MODEL = {MISSING, MANUAL}

COPILOT = paths.REPO / "tools" / "cvat-seg-copilot"
LANE_INIT = COPILOT / "src" / "cvat_seg" / "lane_detectors" / "__init__.py"
BBOX_DRAFT = COPILOT / "scripts" / "bbox_all_draft.py"
YOLO_PT = paths.CHECKPOINTS / "yolo26l.pt"

YOLO_CLASSES = {"pedestrian", "car", "truck", "bus", "train", "motorcycle", "bicycle", "traffic light"}
CS_SEG = {"road", "sidewalk", "building", "wall", "fence", "pole", "vegetation", "terrain", "sky"}
AREA_RULE = {"area/direct", "area/alternative"}


def _lane_registry() -> set[str]:
    if not LANE_INIT.exists():
        return set()
    txt = LANE_INIT.read_text(encoding="utf8", errors="replace")
    blk = txt.split("REGISTRY", 1)[-1]
    return set(re.findall(r'"(lane/[a-z ]+)"\s*:', blk))


def _seg_bench(job_id) -> dict:
    """label -> ['road/night m2f IoU 0.69', ...] tu data/benchmarks/seg_model_bench_job<N>.json."""
    f = paths.DATA / "benchmarks" / f"seg_model_bench_job{job_id}.json"
    if not f.exists():
        return {}
    d = json.loads(f.read_text(encoding="utf8"))
    out: dict = {}
    for entries in d.get("best", {}).values():
        for key, v in entries.items():
            if isinstance(v, dict) and "iou" in v:
                out.setdefault(key.split("/")[0], []).append(f"{key} {v.get('best', '?')} IoU {v['iou']}")
    return out


def _obj_metrics(job_id) -> dict:
    """label -> (matched, total) tu confusion YOLO x GT cua job."""
    out: dict = {}
    for f in (paths.WORKSPACE / "week01").glob(f"job{job_id}-*/metrics/metrics_job{job_id}.json"):
        d = json.loads(f.read_text(encoding="utf8"))
        for gt, row in d.get("confusion_gt_x_yolo", {}).items():
            out[gt] = (row.get(gt, 0), sum(row.values()))
    return out


def scan(job_id, rs: dict, assigned: list[str] | None = None) -> dict:
    """-> {label: {group,status,generator,measured,note}} cho nhan quy uoc + nhan duoc giao."""
    lanes, bench, objm = _lane_registry(), _seg_bench(job_id), _obj_metrics(job_id)
    al = rs.get("aliases", {})
    labels = list(dict.fromkeys(RS.all_known_labels(rs) + [al.get(a, a) for a in (assigned or [])]))
    out = {}
    for lab in labels:
        g = RS.group_of(rs, lab)
        e = {"group": g, "status": MISSING, "generator": "-", "measured": "-", "note": ""}
        if g is None:
            e.update(status=MANUAL, note="không có trong quy ước chung; không có generator")
        elif lab in YOLO_CLASSES or lab == "rider":
            e["generator"] = "YOLO26l-seg + SAM (scripts/bbox_all_draft.py)" + ("; rider = người trên xe 2 bánh (luật)" if lab == "rider" else "")
            if not (BBOX_DRAFT.exists() and YOLO_PT.exists()):
                e.update(status=MISSING, note="thiếu bbox_all_draft.py hoặc yolo26l.pt")
            elif lab in objm:
                m, t = objm[lab]
                e.update(status=MEASURED, measured=f"recall {m}/{t} = {m / max(t, 1):.0%} (job {job_id})",
                         note="recall thấp" if m / max(t, 1) < 0.5 else "")
            else:
                e.update(status=MODEL_UNMEASURED, note="chưa có GT/đo cho nhãn này trong job")
        elif lab == "traffic sign":
            if lab in objm:
                m, t = objm[lab]
                e["measured"] = f"YOLO recall {m}/{t} (COCO không có lớp biển)"
            e.update(status=MODEL_UNMEASURED if BBOX_DRAFT.exists() else MISSING,
                     generator="Cityscapes SceneParser (cvat_seg.scene) -> box từ mask; signs.py chỉ có luật biển cấm đỏ",
                     note="chưa đo bằng model có lớp biển; YOLO không phát hiện được")
        elif lab in CS_SEG:
            e["generator"] = "Cityscapes seg: SegFormer-B2/B5, Mask2Former (segmentation/evidence.py)"
            if lab in bench:
                e.update(status=MEASURED, measured="; ".join(bench[lab]))
            else:
                e.update(status=MODEL_UNMEASURED, note="Cityscapes có lớp này nhưng bench job chưa có số đo")
        elif lab in lanes:
            e.update(status=RULE, generator=f"lane_detectors.REGISTRY['{lab}'] (luật ảnh, cvat_seg)",
                     note="đầu ra polygon hẹp; chưa đo riêng nhãn này")
        elif lab in AREA_RULE:
            e.update(status=RULE, generator="dẫn xuất từ road + 2 vạch biên (areadirect.py / lane_area_from_road.py)",
                     note="chưa đo; phụ thuộc chất lượng road và lane")
        else:
            e.update(status=MISSING, note="không có model/luật" + ("; bbox_all_draft ghi 'không có model'" if g == "lane" else ""))
        out[lab] = e
    return out
