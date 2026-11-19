"""Chia reviewer chéo cho các job CVAT. Không gọi mạng, không ghi lên CVAT.

Quy tắc cứng:
  - Reviewer của một job KHÔNG BAO GIỜ là annotator của job đó.
  - Ai trong `exclude_reviewers` (ví dụ Lead) không được nhận việc review.
Quy tắc mềm (ưu tiên theo thứ tự):
  - Cân bằng số job review giữa các thành viên.
  - Khi hoà, chọn người đứng kế tiếp annotator trong danh sách thành viên (xoay vòng).

Đầu vào (JSON):
  {
    "members": ["Hùng", "Thịnh", "Việt", "Long", "Hưng"],
    "exclude_reviewers": [],
    "jobs": [{"id": 1972, "annotator": "Hùng"}, {"id": 1973, "annotator": "Thịnh"}]
  }

Chạy:
  python -X utf8 assign_reviewers.py input.json                  # in bảng markdown
  python -X utf8 assign_reviewers.py input.json --out out.json   # ghi thêm JSON
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Assignment:
    job_id: int
    annotator: str
    reviewer: str


def assign(members: list[str], jobs: list[dict], exclude: set[str] | None = None) -> list[Assignment]:
    exclude = exclude or set()
    if len(set(members)) != len(members):
        raise ValueError("members có tên trùng nhau")
    for j in jobs:
        if j["annotator"] not in members:
            raise ValueError(f"annotator {j['annotator']!r} của job {j['id']} không có trong members")

    load = {m: 0 for m in members}
    result: list[Assignment] = []
    for j in sorted(jobs, key=lambda x: x["id"]):
        ann = j["annotator"]
        start = members.index(ann)
        # Thứ tự xoay vòng bắt đầu từ người kế tiếp annotator.
        order = [members[(start + k) % len(members)] for k in range(1, len(members))]
        candidates = [m for m in order if m != ann and m not in exclude]
        if not candidates:
            raise ValueError(f"job {j['id']}: không còn reviewer hợp lệ (cần ít nhất 1 người khác annotator)")
        reviewer = min(candidates, key=lambda m: (load[m], order.index(m)))
        load[reviewer] += 1
        result.append(Assignment(j["id"], ann, reviewer))
    return result


def to_markdown(rows: list[Assignment], members: list[str]) -> str:
    lines = ["| Job | Annotator | Reviewer |", "|---|---|---|"]
    lines += [f"| {r.job_id} | {r.annotator} | {r.reviewer} |" for r in rows]
    counts = {m: 0 for m in members}
    for r in rows:
        counts[r.reviewer] += 1
    lines += ["", "Số job review mỗi người: " + ", ".join(f"{m} {c}" for m, c in counts.items())]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="file JSON đầu vào")
    ap.add_argument("--out", type=Path, help="ghi kết quả JSON ra file này")
    args = ap.parse_args(argv)

    data = json.loads(args.input.read_text(encoding="utf-8"))
    members = data["members"]
    rows = assign(members, data["jobs"], set(data.get("exclude_reviewers", [])))
    print(to_markdown(rows, members))

    if args.out:
        payload = [r.__dict__ for r in rows]
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nĐã ghi {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
