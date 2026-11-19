"""Route: rank frames for human review using release_control.scale.risk.

The risk score is a ROUTING/prioritisation signal derived from declared
features (draft confidence, box size, overlap). It is NOT evidence of label
accuracy or correctness.
"""
from __future__ import annotations

from release_control.scale.review_queue import ReviewQueue
from release_control.scale.risk import score_frame

from .prelabel import ACTOR, _audit, _image_size, resolve_image

NOTE = "risk score = routing priority only; NOT evidence of accuracy or correctness"


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _features(shapes: list[dict], size) -> dict:
    if not shapes:
        # nothing drafted: model is silent, so confidence is 0 (maximal uncertainty)
        return {"confidence": 0.0, "classes": []}
    confs = [float(s.get("confidence", 0.0)) for s in shapes]
    area = size[0] * size[1] or 1
    small = sum(1 for s in shapes
                if (s["points"][2] - s["points"][0]) * (s["points"][3] - s["points"][1]) / area < 0.01)
    occ = max((_iou(a["points"], b["points"]) for i, a in enumerate(shapes) for b in shapes[i + 1:]),
              default=0.0)
    return {"confidence": min(confs), "small_far": small / len(shapes),
            "occlusion": occ, "classes": [s.get("label") for s in shapes]}


def route(session: dict, client=None, policy: dict | None = None) -> dict:
    if client is None:
        from release_control.demo_session.session import CvatDemoClient
        client = CvatDemoClient()
    shapes = client.get_job_annotations()
    if isinstance(shapes, dict):
        shapes = shapes.get("shapes", [])
    by_frame: dict[int, list] = {}
    for s in shapes:
        by_frame.setdefault(int(s.get("frame", 0)), []).append(s)
    queue = ReviewQueue()
    info = {}
    for frame, name in enumerate(session.get("image_files", [])):
        size = _image_size(resolve_image(name))
        sc = score_frame(_features(by_frame.get(frame, []), size), policy)
        top = sorted(sc["reasons"].items(), key=lambda kv: -kv[1]["contribution"])[:3]
        info[frame] = (sc["score"], [f"{k}={v['value']} (+{v['contribution']})" for k, v in top])
        queue.add(frame, sc["score"])
    order = queue.pop_batch("router", len(info))
    items = [{"frame": f, "risk": info[f][0], "reasons": info[f][1]} for f in order]
    result = {"job_id": session.get("job_id"), "queue": items, "note": NOTE}
    _audit(session, "route", f"task:{session.get('task_id')}/job:{session.get('job_id')}",
           outputs={"frames": len(items), "top": [i["frame"] for i in items[:3]]})
    return result
