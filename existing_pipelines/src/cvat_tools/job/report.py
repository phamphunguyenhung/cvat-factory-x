"""GAP REPORT (Markdown + JSON). KHONG tu choi job: thieu thi bao, phan con lai van chay."""
from __future__ import annotations

import json
from pathlib import Path

from cvat_tools.core import paths
from cvat_tools.job import capability as CAP
from cvat_tools.job import discover as D
from cvat_tools.job import ruleset as RS


def _ranges(ids) -> str:
    if not ids:
        return "-"
    out, a, b = [], ids[0], ids[0]
    for x in ids[1:]:
        if x == b + 1:
            b = x
            continue
        out.append(f"{a}-{b}" if a != b else str(a))
        a = b = x
    out.append(f"{a}-{b}" if a != b else str(a))
    return ",".join(out)


def build(job_id: int, disc: dict | None, rs: dict, error: str | None = None) -> dict:
    assigned = [l["name"] for l in disc["labels"]] if disc else []
    caps = CAP.scan(job_id, rs, assigned)
    warnings = list(rs.get("_warnings", []))
    contradictions, rows = [], []
    if error:
        warnings.append(f"Không đọc được CVAT ({error}); bảng chỉ dựa trên quy ước + registry, không có nhãn được giao.")
    if disc:
        seen = set()
        for grp in D.duplicate_groups(assigned) + D.alias_pairs(assigned, rs.get("aliases", {})):
            if tuple(grp) in seen:
                continue
            seen.add(tuple(grp))
            warnings.append(f"Nhãn trùng nghĩa cùng tồn tại: {grp}; quy ước dùng bản có dấu cách / tên chuẩn.")
        for l in disc["labels"]:
            n = l["name"]
            if n in rs.get("aliases", {}):
                contradictions.append(f"Job giao nhãn '{n}' (bản gạch dưới/Segmentation) trong khi quy ước dùng '{rs['aliases'][n]}'.")
            want = RS.expected_shape(rs, n)
            if want and l["type"] not in ("any", want):
                contradictions.append(f"'{n}': CVAT khai type={l['type']} nhưng quy ước là {want} "
                                      f"(vd lane: Sổ tay W1 polyline vs quy ước polygon hẹp).")
            bad = [t for t in disc["per_label"][n]["shape_types"] if want and t != want]
            if bad:
                contradictions.append(f"'{n}': đã có shape kiểu {bad} khác quy ước {want}.")
            if RS.group_of(rs, n) == "object":
                have = {a["name"] for a in l["attributes"]}
                miss = [a for a in rs.get("attributes", {}).get("object", []) if a not in have]
                if miss:
                    warnings.append(f"'{n}': thiếu thuộc tính {miss} theo quy ước.")
            if RS.group_of(rs, n) is None:
                warnings.append(f"'{n}': không có trong quy ước chung (chưa có shape/z_order chuẩn).")
        ids = disc["frame_ids"]
        gaps = [(a, b) for a, b in zip(ids, ids[1:]) if b != a + 1]
        if rs["continuity"] == "none":
            warnings.append("Frame KHÔNG liên tục (mặc định/khai báo job): không truyền/nội suy theo thời gian, mỗi frame sinh nháp độc lập. "
                            f"Nếu thực ra là video liên tục, khai báo continuity: video (keyframe_every=10) trong config/rulesets/jobs/{job_id}.yaml.")
        else:
            warnings.append(f"Job khai báo video liên tục: keyframe mỗi {rs['keyframe_every']} ảnh.")
            if gaps:
                warnings.append(f"continuity=video nhưng frame id có khoảng nhảy: {gaps[:5]}")
        if not ids:
            warnings.append("Job không có frame nào.")
    for lab, c in caps.items():
        d = disc["per_label"].get(lab) if disc else None
        rows.append({"label": lab, "assigned": bool(disc) and lab in assigned, "group": c["group"],
                     "shape": RS.expected_shape(rs, lab), "z": RS.expected_z(rs, lab),
                     "status": c["status"], "generator": c["generator"], "measured": c["measured"], "note": c["note"],
                     "existing": d["total"] if d else None,
                     "frames_with": d["frames_with"] if d else None,
                     "frames_without": d["frames_without"] if d else None})
    need = [r["label"] for r in rows if r["status"] in CAP.NEED_MODEL and (r["assigned"] or not disc)]
    return {"job": job_id, "error": error,
            "task": {k: disc[k] for k in ("task_id", "task_name", "start_frame", "stop_frame", "n_frames", "frame_wh", "totals")} if disc else None,
            "frame_ids": disc["frame_ids"] if disc else None,
            "continuity": rs["continuity"], "keyframe_every": rs["keyframe_every"],
            "occluded_threshold": rs.get("occluded_threshold"), "ruleset_sources": rs["_sources"],
            "labels": rows, "need_model": need, "contradictions": contradictions, "warnings": warnings,
            "assigned_not_in_rules": [a for a in assigned if RS.group_of(rs, a) is None]}


def to_markdown(rep: dict) -> str:
    L = [f"# GAP REPORT — job {rep['job']}", ""]
    t = rep["task"]
    if t:
        wh = t["frame_wh"] or ["?", "?"]
        L.append(f"- Task {t['task_id']} `{t['task_name']}` · {t['n_frames']} frame ({t['start_frame']}..{t['stop_frame']}) · "
                 f"ảnh {wh[0]}x{wh[1]} · shape hiện có {t['totals']['shapes']}, track {t['totals']['tracks']}")
    L += [f"- Liên tục: **{rep['continuity']}**" + (f" (keyframe mỗi {rep['keyframe_every']})" if rep["keyframe_every"] else ""),
          f"- Ngưỡng occluded: {rep['occluded_threshold']:.0%} · ruleset: {', '.join(Path(s).name for s in rep['ruleset_sources'])}",
          "- Preflight không từ chối job; mục thiếu chỉ để báo, phần còn lại vẫn chạy.", "",
          "## 1. Nhãn × khả năng sinh nháp", "",
          "| label | giao | kiểu hình | z | trạng thái | generator | đã đo | shape hiện có | ghi chú |",
          "|---|:-:|---|---:|---|---|---|---:|---|"]
    order = {s: i for i, s in enumerate(CAP.ORDER)}
    for r in sorted(rep["labels"], key=lambda r: (not r["assigned"], order[r["status"]], r["label"])):
        ex = "-" if r["existing"] is None else r["existing"]
        L.append(f"| {r['label']} | {'x' if r['assigned'] else ''} | {r['shape'] or '-'} | {'' if r['z'] is None else r['z']} | "
                 f"{r['status']} | {r['generator']} | {r['measured']} | {ex} | {r['note']} |")
    L += ["", "## 2. Shape hiện có theo frame (nhãn đã có shape)", ""]
    ex = [r for r in rep["labels"] if r["existing"]]
    if ex:
        L += ["| label | n | frame đã có | frame chưa có |", "|---|---:|---|---|"]
        L += [f"| {r['label']} | {r['existing']} | {_ranges(r['frames_with'])} | {_ranges(r['frames_without'])} |" for r in ex]
    else:
        L.append("(chưa có shape nào)")
    L += ["", "## 3. Cần bạn tìm model / đề xuất hướng", ""]
    nm = [r for r in rep["labels"] if r["label"] in rep["need_model"]]
    L += [f"- **{r['label']}** ({r['status']}): {r['note'] or 'chưa có generator'}" for r in nm] or ["(không có nhãn THIẾU model)"]
    weak = [r["label"] for r in rep["labels"] if r["status"] == CAP.MODEL_UNMEASURED and (r["assigned"] or not rep["task"])]
    if weak:
        L += ["", "Có model nhưng chưa đo (cần GT/đo trước khi tin): " + ", ".join(weak)]
    L += ["", "## 4. Mâu thuẫn quy ước", ""]
    L += [f"- {c}" for c in rep["contradictions"]] or ["(không phát hiện)"]
    L += ["", "## 5. Cảnh báo", ""]
    L += [f"- {w}" for w in rep["warnings"]] or ["(không)"]
    return "\n".join(L) + "\n"


def write(rep: dict, out_dir=None):
    out_dir = Path(out_dir) if out_dir else paths.DATA / "jobs" / str(rep["job"]) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    md, js = out_dir / "gap_report.md", out_dir / "gap_report.json"
    md.write_text(to_markdown(rep), encoding="utf8")
    js.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf8")
    return md, js


def run(job_id: int, cv, root=None, out_dir=None) -> dict:
    rs = RS.load_ruleset(job_id, root)
    disc, err = None, None
    try:
        disc = D.discover(cv, job_id)
    except Exception as e:  # khong tu choi job
        err = f"{type(e).__name__}: {str(e)[:120]}"
    rep = build(job_id, disc, rs, err)
    rep["files"] = [str(p) for p in write(rep, out_dir)]
    return rep
