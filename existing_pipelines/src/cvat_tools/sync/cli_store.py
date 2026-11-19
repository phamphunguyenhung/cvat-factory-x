"""Lenh `python -m cvat_tools store <import|list|capture|stats|push|rollback>` (dang ky tu cli.py)."""
from __future__ import annotations

import json
from pathlib import Path


def _store(a):
    from cvat_tools.core.shapestore import ShapeStore
    return ShapeStore(a.db)


def _cv(a, write=False):
    from cvat_tools.core.client import Cvat
    return Cvat.from_env(a.profile, write_enabled=write)


def cmd_import(a):
    """File JSON: list cac {frame,label,shape_type,points,[z_order,occluded,attributes,conf,issues,provenance]}."""
    st = _store(a)
    items = json.loads(Path(a.file).read_text(encoding="utf8"))
    for it in items:
        st.add(a.job, it["frame"], it["label"], it["shape_type"], it["points"], a.stage, z_order=it.get("z_order", 0),
               occluded=it.get("occluded", False), attributes=it.get("attributes"), conf=it.get("conf"), issues=it.get("issues"),
               provenance=it.get("provenance") or {"file": Path(a.file).name})
    if getattr(a, "json", False):
        print(json.dumps({"added": len(items), "stage": a.stage, "job": a.job, "db": str(st.path)}, ensure_ascii=False))
    else:
        print(f"da them {len(items)} shape stage={a.stage} job={a.job} -> {st.path}")
    return 0


def cmd_list(a):
    rows = _store(a).list(job=a.job, frame=a.frame, stage=a.stage, label=a.label)
    if getattr(a, "json", False):
        print(json.dumps(rows, ensure_ascii=False, indent=1))
        return 0
    if not rows:
        print("0 shapes found.")
        return 0
    for r in rows:
        print(f"#{r['id']} f{r['frame']} {r['label']} {r['shape_type']} {r['stage']}/{r['status']} cvat={r['cvat_id']} parent={r['parent_id']} pts={len(r['points']) // 2}")
    return 0


def cmd_capture(a):
    from cvat_tools.sync import capture
    res = capture.capture(_cv(a), _store(a), a.job)     # client chi doc
    if getattr(a, "json", False):
        print(json.dumps(res, ensure_ascii=False, indent=1))
    else:
        print(f"capture job={a.job}: total={res.get('total', '?')} inserted={res.get('inserted', res.get('added', '?'))} unchanged={res.get('unchanged', '?')}")
    return 0


def cmd_stats(a):
    from cvat_tools.sync import capture
    st = capture.stats(_store(a), a.job)
    if getattr(a, "json", False):
        print(json.dumps(st, ensure_ascii=False, indent=1))
    else:
        print(capture.format_stats(st))
    return 0


def cmd_push(a):
    from cvat_tools.sync import push
    r = push.push(_cv(a, write=a.yes), _store(a), a.job, yes=a.yes)
    if getattr(a, "json", False):
        print(json.dumps(r, ensure_ascii=False, indent=1))
    else:
        mode = "DRY-RUN" if r["dry_run"] else "PUSHED"
        print(f"[{mode}] job={r['job']} existing={r['n_existing']} to_add={r['n_to_add']} skipped={len(r['skipped'])}")
        if not r["dry_run"]:
            print(f"  added={r.get('added', 0)} verified={r.get('verified')} run_log={r.get('run_log')}")
        else:
            print("  Khong ghi gi. Them --yes de ghi that (backup + run log + rollback).")
    return 0


def _resolve_run_file(a) -> Path:
    if a.run:
        p = Path(a.run)
        if not p.exists():
            raise FileNotFoundError(f"Khong tim thay run log: {p}")
        return p
    if not a.latest or a.job is None:
        raise ValueError("Can chi dinh --run <file> hoac --latest --job <id>.")
    from cvat_tools.core import paths
    runs_dir = paths.DATA / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(f"Chua co thu muc {runs_dir}.")
    for p in sorted(runs_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf8"))
        except Exception:
            continue
        if data.get("job") == a.job:
            return p
    raise FileNotFoundError(f"Khong tim thay run log nao cho job={a.job} trong data/runs.")


def cmd_rollback(a):
    from cvat_tools.sync import push
    run_file = _resolve_run_file(a)
    if not a.yes:
        run = json.loads(run_file.read_text(encoding="utf8"))
        job = run["job"]
        st = _store(a)
        created = run.get("created_ids", [])
        edited = {cid for cid in created if any(r["stage"] == "human_final" for r in st.by_cvat_id(job, cid))}
        res = {"dry_run": True, "job": job, "run": str(run_file), "would_delete": len(created) - len(edited), "would_keep_human_final": len(edited)}
        if getattr(a, "json", False):
            print(json.dumps(res, ensure_ascii=False, indent=1))
        else:
            print(f"[DRY-RUN] rollback job={job} run={run_file.name}: se xoa {res['would_delete']} shape, giu {res['would_keep_human_final']} human_final.")
            print("  Them --yes de thuc hien rollback that.")
        return 0
    res = push.rollback(_cv(a, write=True), _store(a), run_file)
    if getattr(a, "json", False):
        print(json.dumps(res, ensure_ascii=False, indent=1))
    else:
        print(f"[ROLLBACK] run={run_file.name}: deleted={res.get('deleted', 0)} kept={res.get('kept', 0)}")
    return 0


def register(sub, add):
    st = add(sub, "store", lambda a: 0, help="kho shape cuc bo: import | list | capture | stats | push | rollback").add_subparsers(dest="scmd", required=True)

    def common(s, job=True, required=True):
        s.add_argument("--db", default=None, help="mac dinh data/store/shapes.sqlite")
        s.add_argument("--profile", default="local", help="profile CVAT trong configs/profiles.yaml")
        if job:
            s.add_argument("--job", type=int, required=required, help="CVAT job ID")
        return s
    s = common(add(st, "import", cmd_import, help="nhap shape JSON vao kho cuc bo")); s.add_argument("--file", required=True); s.add_argument("--stage", default="rule_draft", choices=["rule_draft", "llm_edit"])
    s = common(add(st, "list", cmd_list, help="liet ke shape trong kho")); s.add_argument("--frame", type=int); s.add_argument("--stage"); s.add_argument("--label")
    common(add(st, "capture", cmd_capture, help="doc CVAT (chi doc) -> human_final trong kho"))
    common(add(st, "stats", cmd_stats, help="thong ke shape theo stage/label"))
    s = common(add(st, "push", cmd_push, help="dry-run mac dinh; --yes de ghi len CVAT")); s.add_argument("--yes", action="store_true", help="xac nhan ghi that len CVAT")
    s = common(add(st, "rollback", cmd_rollback, help="dry-run mac dinh; hoan tac 1 lan push theo --run hoac --latest --job"), job=True, required=False)
    s.add_argument("--run", default=None, help="run log data/runs/*.json")
    s.add_argument("--latest", action="store_true", help="tu chon run log moi nhat cua --job")
    s.add_argument("--yes", action="store_true", help="xac nhan xoa that tren CVAT")
