"""python -m cvat_tools <doctor|projects|models|inspect|detect|track|export> ...

Mac dinh chi doc / dry-run. KHONG ghi len CVAT tu day (ghi that chi qua `python -m mtm apply --yes`).
Lenh sinh du doan AI (detect, track bench/review/keyframes) can --allow-ai-draft (luat RULES Day 5: khong nop box AI nhu nhan cua minh).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cvat_tools.core import paths
from cvat_tools.core.config import list_projects, load_models, load_profiles, load_project

NOTICE = ("Du doan AI (ai_draft). RULES.md Day 5 cam nop box/track AI nhu nhan cua minh; "
          "chi dung khi giang vien cho phep va ghi vao bao cao. Khong ghi len CVAT.")


def _guard(a):
    if not getattr(a, "allow_ai_draft", False):
        print("DUNG:", NOTICE, "\nThem --allow-ai-draft neu da duoc phep (hoac chi de do hieu nang model, khong dung lam nhan).", file=sys.stderr)
        raise SystemExit(2)
    print("LUU Y:", NOTICE)


def _project_frames(proj: dict, limit=None, start=0):
    from cvat_tools.detection.frames import DirSource
    fd = (paths.REPO / proj["dataset"]["frames_dir"]).resolve()
    if not fd.exists():
        raise SystemExit(f"BLOCKED: khong thay frames {fd}")
    return DirSource(fd, start, limit)


def _profile(a, proj):
    prof = load_profiles()[a.profile or proj.get("profile", "balanced")]
    d = dict(prof["detector"])
    for k in ("model", "backend", "imgsz"):
        if getattr(a, k, None):
            d[k] = getattr(a, k)
    return prof, d


def _detector(d, prof):
    from cvat_tools.detection.detector import Detector
    return Detector(d["model"], d["backend"], d["imgsz"], d.get("conf", 0.25), prof.get("limits", {}).get("threads"))


def cmd_doctor(a):
    import importlib.metadata as md
    import platform

    import psutil
    print("Python", platform.python_version(), "·", platform.platform())
    for pkg in ("ultralytics", "torch", "openvino", "onnxruntime", "onnx", "opencv-python", "numpy", "psutil"):
        try:
            print(f"  {pkg}: {md.version(pkg)}")
        except md.PackageNotFoundError:
            print(f"  {pkg}: (chua cai)")
    vm = psutil.virtual_memory()
    print(f"RAM {vm.total / 2**30:.1f} GB (con trong {vm.available / 2**30:.1f}) · CPU {psutil.cpu_count(logical=False)}c/{psutil.cpu_count()}t · tai CPU {psutil.cpu_percent(0.5):.0f}%")
    print("Checkpoint:", {n: (paths.CHECKPOINTS / s["checkpoint"]).exists() for n, s in load_models().items()})
    from cvat_tools.core.client import Cvat, CvatError, load_env
    env = load_env()
    print("CVAT_PAT:", "co" if env.get("CVAT_PAT") else "THIEU", "· host", env.get("CVAT_HOST", "(mac dinh local)"))
    try:
        cv = Cvat.from_env()
        print("CVAT", cv.about()["version"], "· user", cv.me()["username"], "· che do", "ghi" if cv.write_enabled else "chi doc")
    except (CvatError, Exception) as e:
        print("CVAT: khong ket noi duoc /", type(e).__name__, str(e)[:120])
    for p in list_projects():
        fd = p.get("dataset", {}).get("frames_dir")
        if fd:
            q = (paths.REPO / fd).resolve()
            print(f"  project {p['name']}: frames {'OK' if q.exists() else 'THIEU'} ({len(list(q.glob('*.jpg'))) if q.exists() else 0})")
    return 0


def cmd_projects(a):
    for p in list_projects():
        print(f"{p['name']:<22} {p.get('description', '')}  [{p.get('status', 'active')}]")
    return 0


def cmd_models_list(a):
    from cvat_tools.core.model_registry import ModelRegistry
    reg = ModelRegistry.from_config()
    if getattr(a, "json", False):
        print(json.dumps({"models": [m.to_metadata() for m in reg.list()], "status": reg.status()}, ensure_ascii=False, indent=1))
        return 0
    prof = load_profiles()
    for n, s in load_models().items():
        ckpt = s.get("checkpoint", "")
        have = (paths.CHECKPOINTS / ckpt).exists() or (paths.WORKSPACE / ckpt).exists()
        sha = s.get("sha256_16", "-")
        backends = ", ".join(f"{b}={v}" for b, v in (s.get("backends") or {}).items())
        print(f"{n}: {s.get('task', '?')} · ckpt {'co' if have else 'THIEU/HF'} ({sha}) · {backends}")
    print("profiles:", {k: f"{v['detector']['model']}/{v['detector']['backend']}/{v['detector']['imgsz']}+{v['tracker']['name']}" for k, v in prof.items()})
    return 0


def cmd_models_smoke(a):
    """Khoi tao + suy luan 1 frame; cap nhat trang thai backend trong models.yaml (chi 'supported' khi dat)."""
    import yaml

    from cvat_tools.detection.detector import Detector
    proj = load_project(a.project)
    img = _project_frames(proj, 1).image(0)
    f = paths.CONFIG / "models.yaml"
    cfg = yaml.safe_load(f.read_text(encoding="utf8"))
    for m in a.models:
        for b in a.backends:
            try:
                r = Detector(m, b, a.imgsz).smoke(img)
                ok = r["n_det"] >= 0
                print(f"{m} {b}: {'supported' if ok else 'FAIL'} {r}")
                cfg["models"][m]["backends"][b] = f"supported (smoke {a.imgsz}px, {r['n_det']} det)" if ok else "failed"
            except Exception as e:
                print(f"{m} {b}: BLOCKED {repr(e)[:160]}")
                cfg["models"][m]["backends"][b] = f"blocked: {repr(e)[:80]}"
    f.write_text("# Registry model (trang thai backend do `models smoke` cap nhat)\n" + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf8")
    return 0


def cmd_inspect(a):
    from cvat_tools.core.client import Cvat
    proj = load_project(a.project)
    cv = Cvat.from_env(proj["cvat"].get("profile", "local"))
    job = a.job or proj["cvat"]["job_id"]
    print("CVAT", cv.about()["version"], "· user", cv.me()["username"])
    j, m = cv.job(job), cv.meta(job)["frames"]
    t = cv.task(j["task_id"])
    print(f"job {job} · task {j['task_id']} {t['name']} · {len(m)} frame ({j['start_frame']}..{j['stop_frame']}) · {m[0]['width']}x{m[0]['height']}")
    print("  label:", {l["name"]: l["id"] for l in cv.labels(job)})
    ann = cv.annotations(job)
    print(f"  annotation hien co: shapes {len(ann['shapes'])} · tracks {len(ann['tracks'])}")
    return 0


def cmd_detect(a):
    _guard(a)
    from cvat_tools.detection.cached import detect_cached
    proj = load_project(a.project)
    prof, d = _profile(a, proj)
    src = _project_frames(proj, a.limit, a.start)
    det = _detector(d, prof)
    r = detect_cached(src, det, force=a.force)
    tm = r["timing"]
    print(f"{d['model']}/{d['backend']}/{d['imgsz']} · {tm['n']} frame · {'CACHE HIT' if r['hit'] else 'do moi'} key {r['key']}")
    print(f"  detector: median {tm['median_s'] * 1000:.0f} ms · P95 {tm['p95_s'] * 1000:.0f} ms · {tm['fps']:.2f} fps · tong {tm['total_s']:.1f}s (thoi gian cua lan do dau)")
    print("  so nguoi/frame:", round(sum(len(v) for v in r["dets"].values()) / max(1, tm["n"]), 1))
    return 0


def _get_dets(a, proj, prof, d):
    from cvat_tools.detection.cached import detect_cached
    src = _project_frames(proj, a.limit, a.start)
    return src, detect_cached(src, _detector(d, prof))


def _anchors(proj):
    f = proj.get("anchors_file")
    if not f:
        return None
    p = (paths.REPO / f).resolve()
    return {k: tuple(v) for k, v in json.loads(p.read_text(encoding="utf8")).items()} if p.exists() else None


def cmd_track_bench(a):
    _guard(a)
    from cvat_tools.benchmark.tracker_bench import run_all
    proj = load_project(a.project)
    prof, d = _profile(a, proj)
    src, r = _get_dets(a, proj, prof, d)
    print(f"detection cache {r['key']} ({'hit' if r['hit'] else 'moi'}) · {len(r['dets'])} frame · {d['model']}/{d['backend']}/{d['imgsz']}")
    rows = run_all(r["dets"], src, a.trackers, r["key"], anchors=_anchors(proj), reid=a.reid,
                   save_dir=paths.DATA / "predictions" / a.project / r["key"])
    print(f"{'tracker':<16}{'assoc ms(med/p95)':>20}{'fps':>8}{'tracks':>7}{'>=30':>6}{'frag':>6}{'idsw?':>6}{'anchor cont':>12}{'shared':>7}  status")
    for x in rows:
        print(f"{x['tracker']:<16}{str(x.get('assoc_median_ms', '-')) + '/' + str(x.get('assoc_p95_ms', '-')):>20}{x.get('assoc_fps', '-'):>8}"
              f"{x.get('raw_tracks', '-'):>7}{x.get('long_tracks_ge30', '-'):>6}{x.get('fragmentation', '-'):>6}{x.get('id_switch_candidates', '-'):>6}"
              f"{x.get('anchor_continuity_mean', '-'):>12}{x.get('anchors_shared_track', '-'):>7}  {x['status']}{' NHIEU' if x['noisy'] else ''} {x['note']}")
    print("Chi so la HEURISTIC (khong co GT) - khong phai IDF1/MOTA/HOTA.")
    return 0


def _load_pred(proj_name, key, tracker):
    f = paths.DATA / "predictions" / proj_name / key / f"{tracker}.json"
    d = json.loads(f.read_text(encoding="utf8"))
    return {int(t): {int(k): v for k, v in fr.items()} for t, fr in d["tracks"].items()}, \
           {int(t): {int(k): v for k, v in fr.items()} for t, fr in d["scores"].items()}


def cmd_track_keyframes(a):
    _guard(a)
    from cvat_tools.tracking import geometry as G
    from cvat_tools.tracking import keyframes as K
    proj = load_project(a.project)
    prof, d = _profile(a, proj)
    src, r = _get_dets(a, proj, prof, d)
    tracks, _ = _load_pred(a.project, r["key"], a.tracker)
    anchors = _anchors(proj)
    pick, warns = G.select_by_anchors(tracks, anchors, 0) if anchors else ({f"T{i+1}": t for i, t in enumerate(G.select_longest(tracks, 5))}, [])
    tot = {"all": 0, "dp": 0, "opt": 0, "ev": 0}
    out = {}
    print(f"tracker {a.tracker} · tol {a.tol}px · {len(pick)} nguoi")
    for name, tid in pick.items():
        t = G.fill_gaps(tracks[tid], a.max_gap)
        others = {o: G.fill_gaps(tracks[o], a.max_gap) for o in tracks if o != tid and len(tracks[o]) >= 10}
        p = K.plan_track(t, a.tol, others)
        out[name] = {"tracker_id": tid, **{k: p[k] for k in ("keyframes", "n_min", "n_with_events", "error", "baseline", "reduction_vs_all", "reduction_vs_dp")},
                     "events": {str(f): r_ for f, r_ in p["events"].items()}}
        tot["all"] += len(t); tot["dp"] += p["baseline"]["douglas_peucker"]; tot["opt"] += p["n_min"]; tot["ev"] += p["n_with_events"]
        print(f"  {name}: tracker#{tid} {len(t)} box -> DP {p['baseline']['douglas_peucker']} · toi uu {p['n_min']} · +su kien {p['n_with_events']} · sai so max {p['error']['max']:.2f}px")
    for w in warns:
        print("  [CANH BAO]", w)
    print(f"TONG: {tot['all']} box · Douglas-Peucker {tot['dp']} · toi uu {tot['opt']} ({1 - tot['opt'] / max(1, tot['dp']):.0%} it hon DP, {1 - tot['opt'] / max(1, tot['all']):.0%} it hon moi frame) · giu them su kien {tot['ev']}")
    outf = paths.DATA / "exports" / a.project / f"keyframes_{a.tracker}.json"
    outf.parent.mkdir(parents=True, exist_ok=True)
    outf.write_text(json.dumps({"tol": a.tol, "tracker": a.tracker, "provenance": "ai_draft", "tracks": out, "totals": tot}, ensure_ascii=False, indent=1), encoding="utf8")
    print("->", outf.relative_to(paths.WORKSPACE))
    return 0


def cmd_track_review(a):
    _guard(a)
    from cvat_tools.quality import review as R
    from cvat_tools.tracking import geometry as G
    proj = load_project(a.project)
    prof, d = _profile(a, proj)
    src, r = _get_dets(a, proj, prof, d)
    trk, sc = {}, {}
    for name in a.trackers:
        try:
            trk[name], sc[name] = _load_pred(a.project, r["key"], name)
        except FileNotFoundError:
            print(f"  bo qua {name}: chua co du doan (chay `track bench` truoc)")
    # chi giu track du dai de hang doi khong ngap
    trk = {n: {t: tr for t, tr in ts.items() if len(tr) >= 10} for n, ts in trk.items()}
    dets_sets = {f"{d['model']}-{d['backend']}": r["dets"]}
    q = R.build_queue(dets_sets, trk, sc)
    out = Path(a.out) if a.out else paths.DATA / "exports" / a.project / "review.html"
    n = R.write_html(out, q, src, trk, r["dets"], f"Review {a.project}", a.top, tuple(proj.get("identities", [])),
                     {"det_cache": r["key"], "trackers": list(trk), "provenance": "ai_draft"})
    print(f"hang doi {len(q)} frame co rui ro; ghi {n} muc -> {out}")
    for it in q[:5]:
        print(f"  f{it['frame']} risk {it['risk']}: " + "; ".join(sorted({x['kind'] for x in it['reasons']})))
    return 0


def cmd_track_assist(a):
    from cvat_tools.quality import assist
    if a.json:
        tracks, stop = assist.load_tracks(a.json)
    else:
        from cvat_tools.core.client import Cvat
        proj = load_project(a.project)
        cv = Cvat.from_env(proj["cvat"].get("profile", "local"))  # chi doc
        tracks, stop = assist.load_tracks(cv, a.job or proj["cvat"]["job_id"])
    if not tracks:
        print("Job chua co track nguoi dung da gan; khong co gi de kiem tra.")
        return 1
    rep = assist.analyze(tracks, stop)
    md = assist.to_markdown(rep)
    print(md)
    out = paths.DATA / "exports" / (a.project or "adhoc")
    out.mkdir(parents=True, exist_ok=True)
    (out / "assist_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf8")
    (out / "assist_report.md").write_text(md, encoding="utf8")
    print("->", (out / "assist_report.md").relative_to(paths.WORKSPACE))
    return 0


def cmd_export(a):
    from cvat_tools.core.client import Cvat
    from cvat_tools.tracking import geometry as G
    proj = load_project(a.project)
    cv = Cvat.from_env(proj["cvat"].get("profile", "local"))
    job = proj["cvat"]["job_id"]
    j, trks = cv.job(job), cv.annotations(job)["tracks"]
    if a.format == "mot":
        tracks = {t["id"]: G.expand_cvat_track(t, j["stop_frame"]) for t in trks}
        out = Path(a.out) if a.out else paths.DATA / "exports" / a.project / "gt.txt"
        n = G.write_mot(tracks, out, base_frame=j["start_frame"])
        print(f"MOT 1.1: {n} dong -> {out} (tu noi suy, KHONG thay ban xuat chinh thuc cua CVAT de nop)")
    else:
        out = Path(a.out) if a.out else paths.DATA / "exports" / a.project / "cvat_annotations.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"job": job, "stop_frame": j["stop_frame"], "tracks": trks}, ensure_ascii=False), encoding="utf8")
        print(f"CVAT annotations (chi doc tu job {job}) -> {out}")
    return 0


def cmd_job_preflight(a):
    from cvat_tools.core.client import Cvat
    from cvat_tools.job import report
    rep = report.run(a.job, Cvat.from_env(a.profile), out_dir=a.out)
    if getattr(a, "json", False):
        print(json.dumps(rep, ensure_ascii=False, indent=1))
    else:
        print(report.to_markdown(rep))
        print("Da luu:", *rep["files"], sep="\n  ")
    return 0  # preflight KHONG tu choi job


def cmd_tools_list(a):
    from cvat_tools.core.registry import build_default_registry
    reg = build_default_registry()
    if getattr(a, "json", False):
        print(json.dumps([t.to_metadata() for t in reg.list()], ensure_ascii=False, indent=1))
        return 0
    for t in reg.list():
        perms = ",".join(p.value for p in t.permissions)
        print(f"{t.tool_id:<24} v{t.version:<6} [{perms}]  {t.description}")
    return 0


def cmd_tools_info(a):
    from cvat_tools.core.registry import build_default_registry
    spec = build_default_registry().get(a.tool_id)
    if getattr(a, "json", False):
        print(json.dumps(spec.to_metadata(), ensure_ascii=False, indent=1))
    else:
        meta = spec.to_metadata()
        print(f"{meta['tool_id']} ({meta['name']} v{meta['version']})")
        print(f"  desc: {meta['description']}")
        print(f"  permissions: {meta['permissions']}")
        print(f"  resources: {meta['resources']}")
        print(f"  input_schema: {meta['input_schema']}")
    return 0


def cmd_full_bbox(a):
    import subprocess
    script = paths.WORKSPACE / "scripts" / "export_full_bbox_job285.py"
    if not script.exists():
        raise FileNotFoundError(f"Khong tim thay script: {script}")
    if a.job != 285:
        raise ValueError(f"full-bbox hien tai ho tro --job 285 (nhan duoc --job {a.job}).")
    rc = subprocess.call([sys.executable, str(script)])
    if rc != 0:
        return rc
    out_dir = paths.WORKSPACE / "week01" / "job285-bbox" / "previews_full_bbox"
    out_json = out_dir / "full_bbox_job285.json"
    if getattr(a, "json", False) and out_json.exists():
        data = json.loads(out_json.read_text(encoding="utf-8"))
        print(json.dumps({"job": data.get("job"), "frames": len(data.get("frames", {})), "label_totals": data.get("label_totals", {}), "out_dir": str(out_dir)}, ensure_ascii=False, indent=1))
    if a.open:
        grid = out_dir / "summary_grid_20frames.jpg"
        if grid.exists():
            subprocess.Popen(["code", str(grid)], shell=(sys.platform == "win32"))
    return 0


def cmd_week(a):
    from cvat_tools.job import weekly
    if a.wcmd == "list":
        print("\n".join(weekly.list_weeks()) or "(chua co tuan nao)")
    elif a.wcmd == "new":
        print("Da tao", weekly.new_week(a.week))
    else:
        wk = a.week or weekly.current_week()
        rows = weekly.status(wk)
        print(f"== {wk}: {len(rows)} job")
        for r in rows:
            tail = r.get("error") or f"state={r['state']} stage={r['stage']} task={r['task']} frames={r['frames']} dir={r['dir']}"
            print(f"  {r['profile']:<6} job {r['id']:>5} [{r['kind']}] due={r['due'] or '-'} local={r['local']}  {tail}")


def cmd_pose(a):
    from cvat_tools.pose import week2
    if a.pcmd == "draft":
        r = week2.draft_job(a.job, a.profile, a.all, backend=a.backend, honor_skip=not a.no_skip, offline=a.offline)
        print(f"job {r['job']} [{r['kind']}] nhap {r['drafted']} frame (co nhan san {len(r['skipped_existing'])}, ve tay {r['by_hand']}) · model {r['model']}")
        print("nhap:", r["file"])
    elif a.pcmd == "apply":
        r = week2.apply_draft(a.job, a.profile, a.file, yes=a.yes, replace_auto=a.replace_auto)
        print(json.dumps(r, ensure_ascii=False, indent=1))
        if not a.yes:
            print("DRY-RUN: chua ghi gi. Them --yes de ghi (them skeleton source=auto, co backup + rollback).")
    else:
        r = week2.rollback(a.run, yes=a.yes)
        print(json.dumps(r, ensure_ascii=False, indent=1))
        if not a.yes:
            print("DRY-RUN: chua xoa gi. Them --yes de xoa.")


def main(argv=None) -> int:
    for st in (sys.stdout, sys.stderr):
        if hasattr(st, "reconfigure"):
            st.reconfigure(encoding="utf-8")
    epilog = (
        "Vi du:\n"
        "  cvat-tools doctor\n"
        "  cvat-tools store capture --job 285\n"
        "  cvat-tools store push --job 285            # dry-run mac dinh\n"
        "  cvat-tools store rollback --latest --job 285\n"
        "  cvat-tools full-bbox --job 285 --open\n"
    )
    p = argparse.ArgumentParser(
        prog="cvat-tools",
        description="CVAT Engineering Toolkit (dry-run mac dinh; chi ghi CVAT khi co --yes)",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--json", action="store_true", help="xuat JSON sach (cho script/agent)")
    p.add_argument("--traceback", action="store_true", help="in full traceback khi co loi")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(parent, name, fn, **kw):
        s = parent.add_parser(name, **kw)
        s.set_defaults(fn=fn)
        return s

    def proj(s, ai=False):
        s.add_argument("--project", required=ai or True)
        if ai:
            s.add_argument("--allow-ai-draft", action="store_true")
            s.add_argument("--profile", choices=list(load_profiles()) or None)
            s.add_argument("--model"); s.add_argument("--backend"); s.add_argument("--imgsz", type=int)
            s.add_argument("--limit", type=int, default=None); s.add_argument("--start", type=int, default=0)

    add(sub, "doctor", cmd_doctor, help="Chan doan moi truong + ket noi CVAT (khong in PAT)")
    s = add(sub, "projects", cmd_projects, help="Liet ke project"); s.add_argument("action", nargs="?", default="list")
    ms = add(sub, "models", lambda a: 0, help="list | smoke").add_subparsers(dest="mcmd", required=True)
    add(ms, "list", cmd_models_list)
    s = add(ms, "smoke", cmd_models_smoke); s.add_argument("--project", default="day05-mot17")
    s.add_argument("--models", nargs="+", default=["yolo26n", "yolo26s"]); s.add_argument("--backends", nargs="+", default=["pytorch", "onnx", "openvino"]); s.add_argument("--imgsz", type=int, default=640)
    s = add(sub, "inspect", cmd_inspect, help="Chi doc: job/label/annotation"); proj(s); s.add_argument("--job", type=int)
    s = add(sub, "detect", cmd_detect, help="Detection + cache"); proj(s, True); s.add_argument("--force", action="store_true")
    tr = add(sub, "track", lambda a: 0, help="bench | keyframes | review | assist").add_subparsers(dest="tcmd", required=True)
    s = add(tr, "bench", cmd_track_bench); proj(s, True)
    s.add_argument("--trackers", nargs="+", default=["bytetrack", "tracktrack", "ocsort", "fasttrack", "botsort", "deepocsort"]); s.add_argument("--reid", action="store_true")
    s = add(tr, "keyframes", cmd_track_keyframes); proj(s, True)
    s.add_argument("--tracker", default="bytetrack"); s.add_argument("--tol", type=float, default=4.0); s.add_argument("--max-gap", type=int, default=5)
    s = add(tr, "review", cmd_track_review); proj(s, True)
    s.add_argument("--trackers", nargs="+", default=["bytetrack", "tracktrack", "botsort"]); s.add_argument("--top", type=int, default=30); s.add_argument("--out")
    s = add(tr, "assist", cmd_track_assist, help="manual-assist: bao cao chat luong track nguoi dung tu gan (chi doc)")
    s.add_argument("--project", default=None); s.add_argument("--job", type=int); s.add_argument("--json", help="JSON annotation thay vi doc CVAT")
    s = add(sub, "export", cmd_export, help="Xuat tu CVAT (chi doc)"); proj(s); s.add_argument("--format", choices=["cvat", "mot"], default="cvat"); s.add_argument("--out")

    jb = add(sub, "job", lambda a: 0, help="preflight").add_subparsers(dest="jcmd", required=True)
    s = add(jb, "preflight", cmd_job_preflight, help="GAP REPORT cho job (chi doc CVAT, khong tu choi job)")
    s.add_argument("--job", type=int, required=True); s.add_argument("--profile", default="local"); s.add_argument("--out")

    wk = add(sub, "week", lambda a: 0, help="Job theo tuan: new | status | list").add_subparsers(dest="wcmd", required=True)
    for n in ("new", "status", "list"):
        s = add(wk, n, cmd_week); s.set_defaults(wcmd=n)
        if n != "list":
            s.add_argument("--week", default=None, help="vd 2026-W47 (mac dinh: tuan hien tai)")

    ps = add(sub, "pose", lambda a: 0, help="Week 2 face50/person17: draft (local) | apply (ghi CVAT, can --yes) | rollback").add_subparsers(dest="pcmd", required=True)
    s = add(ps, "draft", cmd_pose)
    s.add_argument("--job", type=int, required=True); s.add_argument("--profile", default="edufun")
    s.add_argument("--all", action="store_true", help="ca frame da co nhan (mac dinh chi frame trong)")
    s.add_argument("--backend", default="auto", choices=["yolo", "vitpose", "auto"], help="pose17: yolo | vitpose | auto=ca hai, bat dong -> Q (face50 luon dung FAN)")
    s.add_argument("--no-skip", action="store_true", help="bo qua skip_frames trong push_policy.json (de so sanh backend)")
    s.add_argument("--offline", action="store_true", help="mat mang: doc cache local (labels, frames) thay vi CVAT; ngam --all")
    s = add(ps, "apply", cmd_pose)
    s.add_argument("--job", type=int, required=True); s.add_argument("--profile", default="edufun")
    s.add_argument("--file", default=None, help="file nhap (mac dinh: moi nhat trong drafts/)"); s.add_argument("--yes", action="store_true")
    s.add_argument("--replace-auto", action="store_true", help="xoa skeleton auto cu cua tool tren frame ban CHUA sua roi tao lai tu nhap moi")
    s = add(ps, "rollback", cmd_pose)
    s.add_argument("--run", required=True, help="backups/run_*.json"); s.add_argument("--yes", action="store_true")

    from cvat_tools.sync.cli_store import register as _register_store
    _register_store(sub, add)
    from cvat_tools.pipeline.cli_run import register as _register_run
    _register_run(sub, add)
    from harness.cli_v3 import register as _register_harness
    _register_harness(sub, add)
    from cvat_tools.pose.pose17_cli import register as _register_pose17
    _register_pose17(sub, add)

    fb = add(sub, "full-bbox", cmd_full_bbox, help="xuat preview FULL BBOX (things + lanes + areas + stuff) cho job")
    fb.add_argument("--job", type=int, default=285, help="CVAT job ID (mac dinh 285)")
    fb.add_argument("--open", action="store_true", help="mo summary_grid_20frames.jpg trong VS Code sau khi xong")

    tl = add(sub, "tools", lambda a: 0, help="Tool kernel registry: list | info").add_subparsers(dest="tcmd", required=True)
    add(tl, "list", cmd_tools_list, help="liet ke cac tool da dang ky trong kernel")
    ti = add(tl, "info", cmd_tools_info, help="xem schema va quyen cua 1 tool")
    ti.add_argument("tool_id", help="vi du: segmentation.prelabel, store.push")

    a = p.parse_args(argv)
    try:
        return a.fn(a) or 0
    except KeyboardInterrupt:
        print("ERROR: Da huy boi nguoi dung.", file=sys.stderr)
        return 130
    except Exception as exc:
        if getattr(a, "traceback", False):
            raise
        from cvat_tools.core.client import CvatError, WriteNotAuthorized
        if isinstance(exc, (CvatError, WriteNotAuthorized, FileNotFoundError, ValueError, KeyError)):
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        print(f"ERROR ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
