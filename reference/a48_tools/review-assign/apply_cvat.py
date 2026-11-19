"""Chia reviewer tự động rồi đẩy lên CVAT: đặt stage=validation + assignee=reviewer cho các job ĐÃ XONG.

Chỉ động vào job nêu trong --jobs. Job không nêu: chỉ in bảng chia, không gọi lệnh ghi.
Mặc định dry-run; ghi thật khi có --yes. Backup trạng thái cũ trước khi ghi, có --rollback.

  python -X utf8 apply_cvat.py tuan-02-task473.json --task 473 --jobs 1975 1976          # dry-run
  python -X utf8 apply_cvat.py tuan-02-task473.json --task 473 --jobs 1975 1976 --yes    # ghi thật
  python -X utf8 apply_cvat.py --rollback backups/<file>.json --yes                      # hoàn tác

Tên -> user id CVAT: lấy từ assignee hiện tại của job do chính người đó làm (annotator),
hoặc khai báo "cvat_users": {"Tên": id} trong file JSON đầu vào.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.environ.get("CVAT_TOOLS_SRC", r"D:\06-Learning\AIA-VINGROUP\learning\cvat\src"))
from assign_reviewers import assign, to_markdown  # noqa: E402

HERE = Path(__file__).parent


def _client(write: bool):
    from cvat_tools.core.client import Cvat

    return Cvat.from_env("edufun", write_enabled=write)


def resolve_users(data: dict, cv_jobs: list[dict]) -> dict[str, int]:
    users = {k: int(v) for k, v in (data.get("cvat_users") or {}).items()}
    by_id = {j["id"]: j for j in cv_jobs}
    for j in data["jobs"]:
        a = (by_id.get(j["id"]) or {}).get("assignee")
        if a and j["annotator"] not in users:
            users[j["annotator"]] = int(a["id"])
    return users


def plan(data: dict, cv_jobs: list[dict], only: list[int]) -> list[dict]:
    rows = assign(data["members"], data["jobs"], set(data.get("exclude_reviewers", [])))
    users = resolve_users(data, cv_jobs)
    by_id = {j["id"]: j for j in cv_jobs}
    out = []
    for r in rows:
        if r.job_id not in only:
            continue
        cj = by_id.get(r.job_id)
        if cj is None:
            raise SystemExit(f"job {r.job_id} không thuộc task trên CVAT")
        if r.reviewer not in users:
            raise SystemExit(f"không biết user id CVAT của {r.reviewer!r}; thêm vào cvat_users")
        if users[r.reviewer] == users.get(r.annotator):
            raise SystemExit(f"job {r.job_id}: reviewer trùng annotator")
        out.append({"job": r.job_id, "annotator": r.annotator, "reviewer": r.reviewer,
                    "reviewer_id": users[r.reviewer],
                    "before": {"stage": cj["stage"], "state": cj["state"], "assignee": (cj.get("assignee") or {}).get("id")},
                    "after": {"stage": "validation", "assignee": users[r.reviewer]}})
    return out


def apply(items: list[dict], cv) -> None:
    for it in items:
        cv._need_write()
        cv._req("PATCH", f"/api/jobs/{it['job']}", json={"stage": "validation", "assignee": it["reviewer_id"]})
        got = cv.job(it["job"])
        ok = got["stage"] == "validation" and (got.get("assignee") or {}).get("id") == it["reviewer_id"]
        print(f"  job {it['job']}: stage={got['stage']} state={got['state']} assignee={(got.get('assignee') or {}).get('username')} -> {'OK' if ok else 'LỆCH'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", type=Path)
    ap.add_argument("--task", type=int)
    ap.add_argument("--jobs", type=int, nargs="*", default=[], help="job đã xong, sẽ ghi lên CVAT")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--rollback", type=Path)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    if a.rollback:
        items = json.loads(a.rollback.read_text(encoding="utf-8"))["items"]
        print("Rollback:", [(i["job"], i["before"]) for i in items])
        if not a.yes:
            print("DRY-RUN. Thêm --yes để hoàn tác.")
            return 0
        cv = _client(True)
        for i in items:
            b = i["before"]
            cv._req("PATCH", f"/api/jobs/{i['job']}", json={"stage": b["stage"], "assignee": b["assignee"]})
            print(f"  job {i['job']} đã trả về {b}")
        return 0

    data = json.loads(a.input.read_text(encoding="utf-8"))
    cv = _client(False)
    cv_jobs = cv.all("/api/jobs", task_id=a.task)
    rows = assign(data["members"], data["jobs"], set(data.get("exclude_reviewers", [])))
    print(to_markdown(rows, data["members"]))
    items = plan(data, cv_jobs, a.jobs)
    print("\nSẽ ghi lên CVAT:" if items else "\nKhông job nào để ghi (chỉ chia reviewer).")
    for it in items:
        print(f"  job {it['job']}: {it['before']} -> {it['after']}  (reviewer {it['reviewer']})")
    if not items or not a.yes:
        print("\nDRY-RUN, chưa ghi gì." if items else "")
        return 0
    bdir = HERE / "backups"
    bdir.mkdir(exist_ok=True)
    bf = bdir / f"task{a.task}_{datetime.now():%Y%m%d_%H%M%S}.json"
    bf.write_text(json.dumps({"task": a.task, "items": items}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nBackup: {bf}")
    apply(items, _client(True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
