"""Đọc trạng thái review và Issue của các job CVAT, ghi ra file để làm báo cáo.

Đọc qua REST API, không tải ảnh. Mặc định chỉ đọc; `--apply` chỉ chuyển job `new` sang `in progress` khi đã có shape.

Cần biến môi trường CVAT_TOKEN (lấy theo bước 1 của cvat-seg-autolabel/HUONG_DAN_CHAY.md).

    python lay_review.py --jobs 984-988
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import requests


def parse_jobs(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return out


class Api:
    def __init__(self, url: str, token: str, org: str | None, scheme: str):
        self.url = url.rstrip("/")
        self.s = requests.Session()
        self.s.headers["Authorization"] = f"{scheme} {token}"
        if org:
            self.s.headers["X-Organization"] = org

    def get(self, path: str, **params):
        r = self.s.get(f"{self.url}{path}", params=params, timeout=60)
        r.raise_for_status()
        return r.json()

    def patch(self, path: str, body: dict):
        r = self.s.patch(f"{self.url}{path}", json=body, timeout=60)
        r.raise_for_status()
        return r.json()

    def get_all(self, path: str, **params) -> list[dict]:
        out, page = [], 1
        while True:
            d = self.get(path, page=page, page_size=100, **params)
            out.extend(d.get("results", []))
            if not d.get("next"):
                return out
            page += 1


def user_name(u: dict | None) -> str:
    if not u:
        return "—"
    full = f"{u.get('first_name', '')} {u.get('last_name', '')}".strip()
    return f"{full} ({u.get('username')})" if full else u.get("username", "—")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default=None, help="Danh sách job, ví dụ 1972-1976,1623")
    ap.add_argument("--task", default=None, help="Danh sách task, ví dụ 473,402. Tự lấy mọi job của task")
    ap.add_argument("--url", default="https://cvat.edufun.ai")
    ap.add_argument("--org", default="cohort-5a")
    ap.add_argument("--auth-scheme", default="Bearer")
    ap.add_argument("--apply", action="store_true",
                    help="Ghi thay đổi lên CVAT. Mặc định chỉ đề xuất, không ghi.")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "du-lieu-review"))
    args = ap.parse_args()

    token = os.environ.get("CVAT_TOKEN")
    if not token:
        sys.exit("Thiếu CVAT_TOKEN. Làm bước 1 trong HUONG_DAN_CHAY.md trong cửa sổ này trước.")
    api = Api(args.url, token, args.org, args.auth_scheme)

    report: list[dict] = []
    job_ids: list[int] = []
    if args.task:
        for tid in parse_jobs(args.task):
            job_ids += [j["id"] for j in api.get_all("/api/jobs", task_id=tid)]
    if args.jobs:
        job_ids += parse_jobs(args.jobs)
    if not job_ids:
        sys.exit("Cần --task (ví dụ --task 473) hoặc --jobs (ví dụ --jobs 1972-1976).")
    for jid in dict.fromkeys(job_ids):
        rec: dict = {"job": jid}
        try:
            j = api.get(f"/api/jobs/{jid}")
        except requests.HTTPError as e:
            rec["loi"] = f"Không đọc được job ({e.response.status_code})"
            print(f"Job {jid}: {rec['loi']}")
            report.append(rec)
            continue
        rec.update({
            "task": j.get("task_id"),
            "frames": f"{j.get('start_frame')}-{j.get('stop_frame')}",
            "nguoi_duoc_giao": user_name(j.get("assignee")),
            "stage": j.get("stage"),
            "state": j.get("state"),
            "cap_nhat": j.get("updated_date"),
        })

        issues = []
        for it in api.get_all("/api/issues", job_id=jid):
            comments = api.get_all("/api/comments", issue_id=it["id"])
            issues.append({
                "id": it["id"],
                "frame": it.get("frame"),
                "nguoi_mo": user_name(it.get("owner")),
                "da_resolve": it.get("resolved"),
                "ngay": it.get("created_date"),
                "link": f"{args.url}/jobs/{jid}?frame={it.get('frame')}",
                "binh_luan": [{"nguoi": user_name(c.get("owner")), "ngay": c.get("created_date"),
                               "noi_dung": c.get("message")} for c in comments],
            })
        rec["issues"] = issues

        try:
            ann = api.get(f"/api/jobs/{jid}/annotations")
            shapes = ann.get("shapes", [])
            src = Counter(s.get("source", "?") for s in shapes)
            auto_frames = sorted({s["frame"] for s in shapes if s.get("source") == "auto"})
            rec["shape"] = {"tong": len(shapes), "theo_nguon": dict(src),
                            "frame_co_shape": len({s["frame"] for s in shapes}),
                            "frame_con_shape_auto": auto_frames}
        except requests.HTTPError as e:
            rec["shape"] = {"loi": f"Không đọc được annotation ({e.response.status_code})"}

        # Tự chuyển new -> in progress khi job đã có shape. Không tự đặt completed:
        # người gán phải tự bấm complete khi đã gán đủ frame.
        rec["de_xuat"] = ""
        if rec.get("state") == "new" and rec.get("shape", {}).get("frame_co_shape", 0) > 0:
            rec["de_xuat"] = "new -> in progress"
            if args.apply:
                try:
                    api.patch(f"/api/jobs/{jid}", {"state": "in progress"})
                    rec["de_xuat"] += " (đã ghi)"
                except requests.HTTPError as e:
                    rec["de_xuat"] += f" (lỗi ghi: {e.response.status_code})"

        print(f"Job {jid}: {rec['state']}/{rec['stage']}, {len(issues)} issue, đề xuất: {rec['de_xuat'] or '-'}")
        report.append(rec)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "review_cvat.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Review CVAT — dữ liệu thô", "",
             "| Job | Frame | Người được giao | Stage | State | Issue (mở/tổng) | Shape auto | Frame còn shape auto |",
             "|---|---|---|---|---|---|---|---|"]
    for r in report:
        if "loi" in r:
            lines.append(f"| {r['job']} | | {r['loi']} | | | | | |")
            continue
        iss = r["issues"]
        open_n = sum(1 for i in iss if not i["da_resolve"])
        sh = r.get("shape", {})
        lines.append(f"| {r['job']} | {r['frames']} | {r['nguoi_duoc_giao']} | {r['stage']} | {r['state']} | "
                     f"{open_n}/{len(iss)} | {sh.get('theo_nguon', {}).get('auto', 0)} | "
                     f"{len(sh.get('frame_con_shape_auto', []))} |")
    lines += ["", "## Issue", ""]
    for r in report:
        for i in r.get("issues", []):
            first = i["binh_luan"][0]["noi_dung"] if i["binh_luan"] else ""
            mark = "✅" if i["da_resolve"] else "🔴"
            lines.append(f"- {mark} Job {r['job']} frame {i['frame']} · {i['nguoi_mo']} · {first} · {i['link']}")
    (out / "review_cvat.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Đã ghi {out / 'review_cvat.json'} và review_cvat.md")


if __name__ == "__main__":
    main()
