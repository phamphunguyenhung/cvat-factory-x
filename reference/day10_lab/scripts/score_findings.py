#!/usr/bin/env python3
"""Chấm danh sách lỗi (errors.json) và phần truy vết (trace.json) của một lô.

  python scripts/score_findings.py --run runs/r11
  python scripts/score_findings.py --run nop/nhom3 --truth-dir <đề ẩn>/_private   # giảng viên chấm đề ẩn (in đầy đủ)

Chế độ học viên (không có --truth-dir) chỉ in F1 làm tròn 1 chữ số, không in bảng theo loại lỗi
và không in số lỗi gài: đây là công cụ tự đo, không phải nơi dò đáp án từng ảnh (xem RULES.md).

errors.json:  {"errors": [{"type": "class_swap", "target": "000000000139"}]}
trace.json:   {"tier_a": [...ảnh...], "tier_b": [...ảnh...], "notify": ["A01", "Q02"]}

Điểm xếp hạng là F1 của toàn bộ cặp (type, target): báo thừa bị trừ như bỏ sót.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics  # noqa: E402


def load_errors(path: Path) -> list[dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = document.get("errors")
    if not isinstance(rows, list):
        raise SystemExit(f"{path} cần khóa errors là một list")
    for row in rows:
        if not isinstance(row, dict) or "type" not in row or "target" not in row:
            raise SystemExit(f"{path} mỗi phần tử cần type và target")
    return rows


def fmt(value) -> str:
    return "—" if value is None else str(value)


def line(name: str, row: dict) -> str:
    return (f"- {name}: precision {fmt(row['precision'])} · recall {fmt(row['recall'])} · F1 {fmt(row['f1'])} "
            f"(TP {row['tp']} / FP {row['fp']} / FN {row['fn']})")


def score_trace(truth: dict, path: Path) -> dict | None:
    if not path.is_file():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for key in ("tier_a", "tier_b", "notify"):
        out[key] = metrics.score({("x", v) for v in truth[key]}, {("x", v) for v in doc.get(key, [])})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Chấm errors.json và trace.json so với ground truth")
    ap.add_argument("--run", required=True, help="thư mục có errors.json (và trace.json)")
    ap.add_argument("--truth-dir", help="mặc định <run>/_private")
    args = ap.parse_args()
    run = Path(args.run)
    coach = bool(args.truth_dir)
    tdir = Path(args.truth_dir) if args.truth_dir else run / "_private"
    if not (tdir / "truth.json").is_file():
        raise SystemExit(f"Thiếu {tdir / 'truth.json'}. Chạy inject_release.py trước.")
    if not (run / "errors.json").is_file():
        raise SystemExit(f"Thiếu {run / 'errors.json'}. Ghi danh sách lỗi của nhóm vào file này.")
    truth_doc = json.loads((tdir / "truth.json").read_text(encoding="utf-8"))
    predicted = load_errors(run / "errors.json")
    truth, pred = metrics.pairs(truth_doc["errors"]), metrics.pairs(predicted)
    overall = metrics.score(truth, pred)
    groups = metrics.group_scores(truth, pred)
    trace = score_trace(json.loads((tdir / "trace_truth.json").read_text(encoding="utf-8")), run / "trace.json") \
        if (tdir / "trace_truth.json").is_file() else None
    report = {"seed": truth_doc.get("seed"), "decision": metrics.decision(predicted), "error_count": len(pred),
              "truth_count": len(truth), "score": {"overall": overall, "groups": groups,
                                                   "by_type": metrics.by_type(truth, pred)}, "trace": trace}
    if coach:
        lines = [f"# Score — seed {report['seed']}", "",
                 f"- Quyết định từ danh sách nộp: **{report['decision']}**",
                 f"- Số lỗi nộp: {len(pred)} (lô có {len(truth)} lỗi gài)",
                 line("Overall", overall)]
        lines += [line(name, row) for name, row in groups.items()]
        lines += ["", "| type | group | TP | FP | FN |", "|---|---|---:|---:|---:|"]
        lines += [f"| {r['type']} | {r['group']} | {r['tp']} | {r['fp']} | {r['fn']} |" for r in report["score"]["by_type"]]
        if trace:
            lines += ["", "## Truy vết (ticket team AI)"] + [line(k, v) for k, v in trace.items()]
        else:
            lines += ["", "## Truy vết: chưa có trace.json"]
        lines += ["", f"**rank F1 = {fmt(overall['f1'])}** (F1 toàn bộ lỗi; báo thừa bị trừ như bỏ sót)", ""]
        (run / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        def r1(v):
            return "—" if v is None else f"{v:.1f}"
        lines = [f"# Tự đo — seed {report['seed']}", "",
                 f"- Số lỗi nộp: {len(pred)}",
                 f"- F1 tổng ≈ {r1(overall['f1'])} · lỗi luật ≈ {r1(groups['rule_detectable']['f1'])} · lỗi gán nhãn ≈ {r1(groups['cross_check_review']['f1'])}"]
        if trace:
            lines.append("- Truy vết F1 ≈ " + " · ".join(f"{k} {r1(v['f1'])}" for k, v in trace.items()))
        lines += ["", "Chỉ in làm tròn: dùng để biết validator tốt lên hay kém đi, không để dò từng ảnh.", ""]
    text = "\n".join(lines)
    (run / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
