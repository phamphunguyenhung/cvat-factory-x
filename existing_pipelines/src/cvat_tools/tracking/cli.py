"""CLI: python -m mtm <inspect|draft|bench|apply|check|export|rollback> ..."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cvat_tools.tracking import geometry as core
from cvat_tools.tracking import draft as D
from cvat_tools.core.client import Cvat, load_env
from cvat_tools.core import paths


def _job(cv: Cvat, job: int):
    j = cv.job(job)
    return j, cv.meta(job)["frames"]


def _label(cv: Cvat, job: int, name: str) -> int:
    labels = {l["name"].lower(): l["id"] for l in cv.labels(job)}
    if name.lower() not in labels:
        raise SystemExit(f"Label '{name}' không có trong job {job}: {list(labels)}")
    return labels[name.lower()]


def _images(cv: Cvat, job: int, j: dict, frames: list, limit: int | None):
    out = {}
    for i, _ in enumerate(frames[:limit]):
        out[j["start_frame"] + i] = D.decode(cv.frame(job, j["start_frame"] + i))
    return out


def _anchors(a) -> dict | None:
    if not a.anchors:
        return None
    data = json.loads(Path(a.anchors).read_text(encoding="utf8"))
    return {k: tuple(v) for k, v in data.items()}


def _guard(a) -> None:
    if not a.allow_ai_draft:
        print("DỪNG:", D.NOTICE, "\nThêm --allow-ai-draft nếu đã được phép.", file=sys.stderr)
        raise SystemExit(2)
    print("LƯU Ý:", D.NOTICE)


def cmd_inspect(a):
    cv = Cvat.from_env()
    print("CVAT", cv.about()["version"], "· user", cv.me()["username"])
    j, frames = _job(cv, a.job)
    t = cv.task(j["task_id"])
    print(f"job {a.job} · task {j['task_id']} {t['name']} · mode {t['mode']} · {len(frames)} frame "
          f"({j['start_frame']}..{j['stop_frame']}) · {frames[0]['width']}x{frames[0]['height']}")
    print("  label:", {l["name"]: l["id"] for l in cv.labels(a.job)})
    ann = cv.annotations(a.job)
    print(f"  annotation hiện có: shapes {len(ann['shapes'])} · tracks {len(ann['tracks'])}")
    return 0


def cmd_draft(a):
    _guard(a)
    cv = Cvat.from_env()
    j, frames = _job(cv, a.job)
    label_id = _label(cv, a.job, a.label)
    imgs = _images(cv, a.job, j, frames, a.limit)
    model, mname = D.load_model(load_env().get("MOT_MODEL"))
    print(f"{len(imgs)} frame · model {mname} · tracker {a.tracker}{'+reid' if a.reid else ''}")
    raw = D.run_tracker(model, imgs, a.tracker, a.reid, a.conf, a.imgsz)
    print(f"tracker tạo {len(raw)} track thô")
    ref = j["start_frame"] + a.ref_frame
    result, warns = D.build_draft(raw, label_id, j["stop_frame"], a.people, _anchors(a), ref, a.tol, a.max_gap, a.min_len)
    for name, i in result["info"].items():
        print(f"  {name}: tracker_id {i['tracker_id']} · f{i['first']}..f{i['last']} · {i['boxes']} box -> {i['keyframes']} keyframe")
    for w in warns:
        print("  [CẢNH BÁO]", w)
    if not result["tracks"]:
        print("Không chọn được track nào (hạ --conf, hoặc kiểm tra điểm neo).")
        return 1
    draft = {"job": a.job, "model": mname, "tracker": a.tracker + ("+reid" if a.reid else ""), "notice": D.NOTICE,
             "params": {"conf": a.conf, "imgsz": a.imgsz, "tol": a.tol, "max_gap": a.max_gap}, **result}
    print("nháp:", D.save_draft(a.job, draft).relative_to(paths.WORKSPACE))
    print(f"Tiếp: python -m mtm apply --job {a.job}   (dry-run)")
    return 0


def cmd_bench(a):
    """So các tracker trên CÙNG frame bằng chỉ số tự nhất quán (không có ground truth nên KHÔNG phải MOTA/IDF1)."""
    _guard(a)
    cv = Cvat.from_env()
    j, frames = _job(cv, a.job)
    imgs = _images(cv, a.job, j, frames, a.limit)
    model, mname = D.load_model(load_env().get("MOT_MODEL"))
    print(f"model {mname} · {len(imgs)} frame · mục tiêu {a.people} người\n")
    print(f"{'tracker':<16}{'track thô':>10}{'dài>=30':>9}{'id_switch?':>11}{'frag':>6}{'s/frame':>9}")
    import time
    for name in a.trackers:
        for reid in ([False, True] if a.reid and name in ("botsort", "deepocsort", "tracktrack") else [False]):
            t0 = time.time()
            raw = D.run_tracker(model, imgs, name, reid, a.conf, a.imgsz)
            dt = (time.time() - t0) / max(1, len(imgs))
            top = core.select_longest(raw, a.people, 10)
            q = core.qa_tracks(top)
            kinds = lambda k: sum(1 for x in q if x["kind"] == k)
            print(f"{name + ('+reid' if reid else ''):<16}{len(raw):>10}{sum(len(t) >= 30 for t in raw.values()):>9}"
                  f"{kinds('id_switch'):>11}{kinds('fragmentation'):>6}{dt:>9.2f}")
    print("\nÍt track thô + ít id_switch/frag thường tốt hơn, nhưng chỉ khi số track dài gần bằng số người thật.")
    return 0


def cmd_apply(a):
    from cvat_tools.tracking.writer import apply

    cv = Cvat.from_env(write_enabled=bool(a.yes))  # chi doc tru khi --yes
    draft = D.load_draft(a.job, a.draft)
    if a.only:
        want = {x.strip().upper() for x in a.only.split(",")}
        draft["tracks"] = [t for t in draft["tracks"] if t["name"].upper() in want]
    r = apply(cv, a.job, draft, yes=a.yes)
    print(f"job {a.job}: tạo {r['to_create']} track ({r['keyframes']} keyframe) · track hiện có {r['before_tracks']}")
    for e in r["errors"]:
        print("  [LỖI]", e)
    for e in r["skipped"]:
        print("  [BỎ QUA]", e)
    if r["applied"]:
        print(f"ĐÃ GHI · đọc lại {r['after_tracks']} track · xác minh {r['verified']}\n  backup {r['backup']}\n  run    {r['run']}")
        print("Tải lại CVAT (F5), rồi sửa keyframe/occluded/outside bằng tay.")
    else:
        print("DRY-RUN: chưa ghi gì. Thêm --yes để ghi." if not r["errors"] else "Không ghi do lỗi trên.")
    return 1 if r["errors"] else 0


def cmd_check(a):
    cv = Cvat.from_env()
    j, _ = _job(cv, a.job)
    trks = cv.annotations(a.job)["tracks"]
    if not trks:
        print("Job chưa có track.")
        return 1
    tracks = {t.get("id", i): core.expand_cvat_track(t, j["stop_frame"]) for i, t in enumerate(trks, 1)}
    print(f"job {a.job}: {len(tracks)} track")
    for tid, tr in tracks.items():
        print(f"  track {tid}: f{min(tr)}..f{max(tr)} · {len(tr)} box")
    issues = core.qa_tracks(tracks) + core.qa_keyframes(trks)
    if len(tracks) != a.expect:
        issues.append({"kind": "count", "msg": f"có {len(tracks)} track, đề yêu cầu {a.expect}"})
    for it in issues:
        print(f"  [{it['kind']}] {it['msg']}")
    print("QA sạch (chưa chứng minh đúng: vẫn tự review 2 lượt)." if not issues else f"{len(issues)} gợi ý, kiểm tra lại trong CVAT.")
    return 0


def cmd_export(a):
    cv = Cvat.from_env()
    j, _ = _job(cv, a.job)
    trks = cv.annotations(a.job)["tracks"]
    tracks = {t["id"]: core.expand_cvat_track(t, j["stop_frame"]) for t in trks}
    n = core.write_mot(tracks, Path(a.out), base_frame=j["start_frame"])
    print(f"MOT 1.1: {n} dòng -> {a.out} (gt.txt tự tính từ nội suy, KHÔNG thay file xuất chính thức của CVAT để nộp)")
    return 0


def cmd_rollback(a):
    from cvat_tools.tracking.writer import rollback

    r = rollback(Cvat.from_env(write_enabled=True), a.run)
    print(f"job {r['job']}: xóa {r['deleted']} track của lần chạy này · còn {r['remaining']} track")
    return 0


def main(argv=None) -> int:
    for st in (sys.stdout, sys.stderr):
        if hasattr(st, "reconfigure"):
            st.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="mtm", description="MOT track manager: nháp track cho CVAT (Lab Ngày 5)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, **kw):
        s = sub.add_parser(name, **kw)
        s.set_defaults(fn=fn)
        return s

    def common(s, ai=False):
        s.add_argument("--job", type=int, required=True)
        if ai:
            s.add_argument("--allow-ai-draft", action="store_true", help="Xác nhận đã được giảng viên cho phép dùng nháp AI")
            s.add_argument("--conf", type=float, default=0.25)
            s.add_argument("--imgsz", type=int, default=1920, help="MOT17-09 là 1920x1080; người nhỏ cần imgsz lớn")
            s.add_argument("--people", type=int, default=5)
            s.add_argument("--limit", type=int, default=None, help="Chỉ N frame đầu (chạy thử)")

    s = add("inspect", cmd_inspect, help="Chỉ đọc: CVAT, job, label, annotation hiện có"); common(s)
    s = add("draft", cmd_draft, help="Sinh nháp ra drafts/ (chưa đụng CVAT)"); common(s, True)
    s.add_argument("--tracker", choices=D.TRACKERS, default="botsort")
    s.add_argument("--reid", action="store_true", help="Bật ReID (botsort/deepocsort/tracktrack): bớt id_switch khi bị che")
    s.add_argument("--label", default="person")
    s.add_argument("--anchors", help='JSON {"P1":[x,y],...}: điểm neo từng người ở --ref-frame (theo bảng nhận diện của đề)')
    s.add_argument("--ref-frame", type=int, default=0, help="Frame tương đối của ảnh nhận diện (mặc định 0)")
    s.add_argument("--tol", type=float, default=4.0, help="Sai số nội suy tối đa (px) để bỏ keyframe")
    s.add_argument("--max-gap", type=int, default=5, help="Lấp lỗ detector <= N frame")
    s.add_argument("--min-len", type=int, default=10)
    s = add("bench", cmd_bench, help="So các tracker bằng chỉ số tự nhất quán"); common(s, True)
    s.add_argument("--trackers", nargs="+", default=["bytetrack", "botsort", "tracktrack"], choices=D.TRACKERS)
    s.add_argument("--reid", action="store_true", help="Thêm biến thể +reid")
    s = add("apply", cmd_apply, help="Dry-run; --yes mới ghi"); common(s)
    s.add_argument("--yes", action="store_true"); s.add_argument("--only", help="Chỉ ghi các người này, ví dụ P4,P5"); s.add_argument("--draft", help="Mặc định drafts/LATEST_job<ID>.json")
    s = add("check", cmd_check, help="QA track thật trên CVAT"); common(s)
    s.add_argument("--expect", type=int, default=5)
    s = add("export", cmd_export, help="Ghi gt.txt MOT 1.1 tính từ track trên CVAT"); common(s)
    s.add_argument("--out", default="out/gt.txt")
    s = add("rollback", cmd_rollback, help="Xóa đúng các track của một lần apply"); s.add_argument("--run", required=True)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
