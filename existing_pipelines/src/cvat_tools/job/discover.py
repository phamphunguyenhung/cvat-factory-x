"""Chi doc CVAT: task/job, frame id that, label (id,ten,type,attributes), shape hien co theo label x frame."""
from __future__ import annotations

import re
from collections import defaultdict


def norm_key(name: str) -> str:
    return re.sub(r"[\s_]+", " ", name.strip().lower())


def duplicate_groups(names) -> list[list[str]]:
    """Ten trung nghia sau khi chuan hoa (person/pedestrian xu ly rieng qua aliases)."""
    g = defaultdict(list)
    for n in names:
        g[norm_key(n)].append(n)
    return [sorted(v) for v in g.values() if len(v) > 1]


def alias_pairs(names, aliases: dict) -> list[list[str]]:
    s = set(names)
    return [sorted([a, b]) for a, b in aliases.items() if a in s and b in s]


def frame_ids(job: dict, meta: dict) -> list[int]:
    inc = meta.get("included_frames")
    if inc:
        return sorted(int(x) for x in inc)
    return list(range(int(job["start_frame"]), int(job["stop_frame"]) + 1))


def discover(cv, job_id: int) -> dict:
    """cv: doi tuong co job/task/meta/labels/annotations (core.client.Cvat hoac gia)."""
    job = cv.job(job_id)
    try:
        task = cv.task(job["task_id"])
    except Exception as e:  # quyen doc task co the bi 403; khong chan preflight
        task = {"name": f"(khong doc duoc task: {str(e)[:40]})"}
    meta = cv.meta(job_id)
    ids = frame_ids(job, meta)
    frames = meta.get("frames", [])
    wh = [frames[0].get("width"), frames[0].get("height")] if frames else None
    labels = []
    for l in cv.labels(job_id):
        labels.append({"id": l["id"], "name": l["name"], "type": l.get("type", "any"),
                       "attributes": [{"name": a["name"], "input_type": a.get("input_type")} for a in l.get("attributes", [])]})
    by_id = {l["id"]: l["name"] for l in labels}
    ann = cv.annotations(job_id)
    cnt: dict = defaultdict(lambda: defaultdict(int))     # label -> frame -> n
    types: dict = defaultdict(lambda: defaultdict(int))   # label -> shape type -> n
    for s in ann.get("shapes", []):
        n = by_id.get(s.get("label_id"), f"#{s.get('label_id')}")
        cnt[n][int(s["frame"])] += 1
        types[n][s.get("type", "?")] += 1
    for t in ann.get("tracks", []):
        n = by_id.get(t.get("label_id"), f"#{t.get('label_id')}")
        for s in t.get("shapes", []):
            if not s.get("outside"):
                cnt[n][int(s["frame"])] += 1
                types[n][s.get("type", "?")] += 1
    per_label = {}
    for l in labels:
        n = l["name"]
        have = sorted(cnt[n])
        per_label[n] = {"total": sum(cnt[n].values()), "frames_with": have,
                        "frames_without": [f for f in ids if f not in cnt[n]],
                        "shape_types": dict(types[n])}
    return {"job_id": job_id, "task_id": job["task_id"], "task_name": task.get("name"),
            "start_frame": job["start_frame"], "stop_frame": job["stop_frame"],
            "n_frames": len(ids), "frame_ids": ids, "frame_wh": wh,
            "labels": labels, "per_label": per_label,
            "totals": {"shapes": len(ann.get("shapes", [])), "tracks": len(ann.get("tracks", []))}}
