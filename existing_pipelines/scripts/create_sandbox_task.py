"""Create a separate sandbox Task + Job on CVAT via PAT and upload Job 285 sample frames (f60-f79).

Does NOT touch Job 285 or any existing cohort task. Uses personal workspace by default
(or --org cohort-5a if requested).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

CVAT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CVAT_ROOT / "src"))

from cvat_tools.core.client import Cvat  # noqa: E402

FRAMES_DIR = CVAT_ROOT.parent / "tools" / "cvat-seg-copilot" / "frames"


def _copy_label_spec(lbl: dict) -> dict:
    attrs = []
    for a in lbl.get("attributes") or []:
        attrs.append({
            "name": a["name"],
            "mutable": bool(a.get("mutable", False)),
            "input_type": a.get("input_type", "select"),
            "default_value": a.get("default_value", ""),
            "values": list(a.get("values") or []),
        })
    return {
        "name": lbl["name"],
        "color": lbl["color"],
        "type": lbl.get("type", "any"),
        "attributes": attrs,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Create a separate CVAT sandbox task+job and upload sample frames.")
    ap.add_argument("--name", default="cvat_tools_sandbox_j285_f60_79")
    ap.add_argument("--source-job", type=int, default=285)
    ap.add_argument("--org", default="", help="Empty = personal workspace; or e.g. cohort-5a")
    ap.add_argument("--image-quality", type=int, default=95)
    args = ap.parse_args()

    src_cv = Cvat.from_env(write_enabled=False)
    pat = src_cv.h["Authorization"].split(" ", 1)[1]
    labels = [_copy_label_spec(l) for l in src_cv.labels(args.source_job)]
    print(f"Loaded {len(labels)} labels from job {args.source_job}.")

    cv = Cvat(src_cv.host, pat, org=args.org, write_enabled=True)
    me = cv.me()
    print(f"Authenticated as {me.get('username')} (org={args.org or 'personal'}) on {cv.host}")

    frames = sorted(FRAMES_DIR.glob("j285_f*.jpg"))
    if not frames:
        raise SystemExit(f"No j285_f*.jpg found in {FRAMES_DIR}")
    print(f"Found {len(frames)} sample frames in {FRAMES_DIR}")

    # 1. Create task
    r_task = requests.post(
        f"{cv.host}/api/tasks",
        headers=cv.h,
        json={"name": args.name, "labels": labels},
        timeout=60,
    )
    if r_task.status_code >= 400:
        raise SystemExit(f"POST /api/tasks failed ({r_task.status_code}): {r_task.text[:400]}")
    task = r_task.json()
    task_id = task["id"]
    print(f"Created task id={task_id} name={task.get('name')}")

    # 2. Upload sample frames
    files = [("client_files[{}]".format(i), (p.name, p.read_bytes(), "image/jpeg")) for i, p in enumerate(frames)]
    data = {
        "image_quality": str(args.image_quality),
        "sorting_method": "lexicographical",
        "use_zip_chunks": "true",
        "use_cache": "true",
    }
    r_data = requests.post(
        f"{cv.host}/api/tasks/{task_id}/data",
        headers=cv.h,
        data=data,
        files=files,
        timeout=180,
    )
    if r_data.status_code >= 400:
        raise SystemExit(f"POST /api/tasks/{task_id}/data failed ({r_data.status_code}): {r_data.text[:400]}")

    rq_id = (r_data.json() or {}).get("rq_id")
    print(f"Upload accepted (rq_id={rq_id}). Waiting for CVAT data processing...")
    if rq_id:
        for _ in range(60):
            rq = cv.get(f"/api/requests/{rq_id}")
            st = rq.get("status")
            if st == "finished":
                break
            if st == "failed":
                raise SystemExit(f"CVAT data processing failed: {rq.get('message')}")
            time.sleep(2)
        else:
            raise SystemExit("Timed out waiting for CVAT data processing.")

    # 3. Fetch created job
    jobs = cv.all("/api/jobs", task_id=task_id)
    job_ids = [j["id"] for j in jobs]
    meta = cv.meta(job_ids[0]) if job_ids else {}
    out = {
        "host": cv.host,
        "org": args.org or "personal",
        "task_id": task_id,
        "job_ids": job_ids,
        "task_url": f"{cv.host}/tasks/{task_id}",
        "job_url": f"{cv.host}/tasks/{task_id}/jobs/{job_ids[0]}" if job_ids else None,
        "n_frames": len(meta.get("frames") or []),
        "frames_uploaded": [p.name for p in frames],
    }
    out_path = CVAT_ROOT / "data" / "sandbox_task.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
