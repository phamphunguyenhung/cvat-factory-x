"""Dashboard FastAPI app goi truc tiep release_control. Chay: python -X utf8 -m uvicorn dashboard.app:app --port 8765"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from release_control.db import ControlTowerDB
from release_control.fault_lab.inject import inject_faults
from release_control.lineage.impact import ImpactAnalyzer
from release_control.models import FaultType
from release_control.models import ValidationReport as ModelReport
from release_control.release.gate import ReleaseGate
from release_control.validators import ValidationEngine
from release_control.versions import RestoreManager, SnapshotManager

ROOT = Path(__file__).resolve().parent.parent
DATASET = Path(os.environ.get("DASHBOARD_DATASET", ROOT / "demo" / "coco128_clean"))
WORK = Path(os.environ.get("DASHBOARD_WORK", ROOT / "artifacts" / "dashboard"))
FAULTY = WORK / "faulty"
RESTORED = WORK / "restored"
STATIC = Path(__file__).resolve().parent / "static"
LADDER = ["DRAFT", "VALIDATED", "HOLD", "REWORK", "APPROVED", "PUBLISHED", "ACCEPTED"]

app = FastAPI(title="Release Control Dashboard")


def _db() -> ControlTowerDB:
    WORK.mkdir(parents=True, exist_ok=True)
    return ControlTowerDB(WORK / "dash.sqlite")


def _snap() -> SnapshotManager:
    return SnapshotManager(snapshots_root=WORK / "snapshots")


def _has_dataset() -> bool:
    return DATASET.is_dir() and (DATASET / "manifest.json").exists()


def _dir(which: str) -> Path:
    if which == "faulty":
        if not FAULTY.is_dir():
            raise HTTPException(400, "chua inject, chua co ban sao loi")
        return FAULTY
    if not _has_dataset():
        raise HTTPException(400, "chua co dataset")
    return DATASET


class Inject(BaseModel):
    faults: list[str]
    seed: int = 42


class Snap(BaseModel):
    source: str = "clean"
    tag: str = "v1"


class Val(BaseModel):
    source: str = "clean"


class Restore(BaseModel):
    digest: str


class Rel(BaseModel):
    digest: str
    source: str = "clean"


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/state")
def state():
    db = _db()
    return {
        "has_dataset": _has_dataset(),
        "dataset_path": str(DATASET),
        "message": None if _has_dataset() else "chua co dataset",
        "faulty_exists": FAULTY.is_dir(),
        "fault_types": [f.value for f in FaultType],
        "snapshots": [s.model_dump(mode="json") for s in db.list_snapshots()],
        "releases": [r.model_dump(mode="json") for r in db.list_releases()],
        "ladder": LADDER,
    }


@app.post("/api/inject")
def inject(body: Inject):
    src = _dir("clean")
    if not body.faults:
        raise HTTPException(400, "chon it nhat 1 loi")
    try:
        m = inject_faults(src, FAULTY, body.faults, seed=body.seed)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"inject loi: {e}")
    return m.model_dump(mode="json")


@app.post("/api/snapshot")
def snapshot(body: Snap):
    d = _dir(body.source)
    rec = _snap().create_snapshot(d, DATASET.name, body.tag, db=_db())
    return rec.model_dump(mode="json")


@app.post("/api/validate")
def validate(body: Val):
    d = _dir(body.source)
    ref = DATASET if body.source == "faulty" else None
    return ValidationEngine.validate_dataset(d, reference_dir=ref)


@app.get("/api/diff")
def diff(a: str, b: str):
    try:
        return _snap().diff_snapshots(a, b)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, str(e))


@app.post("/api/restore")
def restore(body: Restore):
    try:
        res = RestoreManager(snapshots_root=WORK / "snapshots").restore_snapshot(
            body.digest, RESTORED, db=_db()
        )
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "match": False, "error": str(e)}
    got, _ = SnapshotManager.compute_directory_digest(RESTORED)
    return {"ok": True, "match": got == body.digest, "expected": body.digest, "restored": got, "detail": res}


@app.get("/api/lineage")
def lineage(node: str):
    r = ImpactAnalyzer(_db()).get_blast_radius(node, include_unverified=True)
    if isinstance(r, dict):
        for e in r.get("edges", []):
            e["label"] = "simulated" if e.get("evidence_status", "verified") != "verified" else "verified"
    return r


@app.post("/api/release")
def release(body: Rel):
    d = _dir(body.source)
    db = _db()
    rec = db.get_snapshot(body.digest)
    if rec is None:
        raise HTTPException(400, "snapshot chua ghi nhan")
    gate = ReleaseGate(db)
    rid = f"REL-{body.digest[:8]}-{uuid.uuid4().hex[:4]}"
    gate.create_release(rid, rec.dataset_name, rec.version, body.digest, snapshot_id=rec.snapshot_id)
    rep = ValidationEngine.validate_dataset(d, reference_dir=DATASET if body.source == "faulty" else None)
    sc = rep["severity_counts"]
    mr = ModelReport(
        report_id=f"rep-{uuid.uuid4().hex[:8]}",
        release_id=rid,
        snapshot_id=rec.snapshot_id,
        passed=rep["passed"],
        critical_count=sc.get("CRITICAL", 0),
        warning_count=sc.get("WARNING", 0),
        info_count=sc.get("INFO", 0),
        metrics={"requires_human_review": rep["requires_human_review"]},
    )
    out = gate.submit_validation(rid, mr, requires_human_review=rep["requires_human_review"])
    return {"release": out.model_dump(mode="json"), "report": rep}


# ---------------- Agent Console (goi release_control.cli_live qua subprocess) ----------------
DEMO_DIR = Path(os.environ.get("RC_DEMO_DIR", ROOT / "artifacts" / "release_control_demo"))
CVAT_BASE = os.environ.get("RC_CVAT_BASE", "http://localhost:8080")
AGENT_CMDS = ("ingest", "prelabel", "route", "pull", "validate", "inject", "trace", "restore")
AGENT_TIMEOUT = int(os.environ.get("RC_AGENT_TIMEOUT", "300"))
NOT_READY = "chua san sang"


def _cli_ready() -> bool:
    try:
        return importlib.util.find_spec("release_control.cli_live") is not None
    except (ImportError, ValueError):
        return False


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _session() -> dict | None:
    s = _read_json(DEMO_DIR / "session.json")
    return s if isinstance(s, dict) else None


def _job_url(sess: dict | None) -> str | None:
    if not sess or sess.get("task_id") is None or sess.get("job_id") is None:
        return None
    return f"{CVAT_BASE}/tasks/{sess['task_id']}/jobs/{sess['job_id']}"


def _shape_count(sess: dict) -> tuple[int | None, str | None]:
    """So shape hien tai qua release_control.demo_session.session (neu import duoc)."""
    try:
        mod = importlib.import_module("release_control.demo_session.session")
        ann = mod.CvatDemoClient().get_job_annotations()
        if isinstance(ann, dict):
            return len(ann.get("shapes", [])), None
        return len(ann), None
    except Exception as e:  # noqa: BLE001
        return None, f"{NOT_READY}: {type(e).__name__}: {e}"


@app.get("/api/agent/status")
def agent_status():
    sess = _session()
    ready = _cli_ready()
    return {
        "cli_ready": ready,
        "session_ready": sess is not None,
        "commands": list(AGENT_CMDS),
        "job_url": _job_url(sess),
        "message": None if ready else f"{NOT_READY}: chua co release_control.cli_live",
    }


@app.get("/api/job")
def job_state():
    sess = _session()
    if sess is None:
        return {"ready": False, "message": f"{NOT_READY}: chua co session.json (chay ingest)"}
    shapes, err = _shape_count(sess)
    return {
        "ready": True,
        "task_id": sess.get("task_id"),
        "job_id": sess.get("job_id"),
        "task_name": sess.get("task_name"),
        "frames": len(sess.get("image_files") or []),
        "shapes": shapes,
        "shapes_message": err,
        "job_url": _job_url(sess),
    }


@app.get("/api/audit")
def audit_state():
    try:
        mod = importlib.import_module("release_control.audit")
        log = mod.AuditLog(DEMO_DIR / "audit.jsonl")
        entries = log.read()
        return {"ready": True, "entries": entries, "chain_ok": bool(log.verify_chain())}
    except Exception as e:  # noqa: BLE001
        return {"ready": False, "entries": [], "chain_ok": None,
                "message": f"{NOT_READY}: {type(e).__name__}: {e}"}


@app.get("/api/queue")
def review_queue():
    r = _read_json(DEMO_DIR / "route.json")
    if r is None:
        return {"ready": False, "message": f"{NOT_READY}: chua co route.json (chay route)"}
    return {"ready": True, "route": r}


@app.post("/api/agent/{cmd}")
def agent_run(cmd: str):
    if cmd not in AGENT_CMDS:
        raise HTTPException(404, "lenh khong hop le")
    if not _cli_ready():
        return {"ok": False, "ready": False, "message": f"{NOT_READY}: chua co release_control.cli_live"}
    if cmd != "ingest" and _session() is None:
        return {"ok": False, "ready": False, "message": f"{NOT_READY}: chua co session.json (chay ingest truoc)"}
    argv = [sys.executable, "-X", "utf8", "-m", "release_control.cli_live", cmd]
    try:
        p = subprocess.run(argv, cwd=str(ROOT), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=AGENT_TIMEOUT, shell=False)
    except subprocess.TimeoutExpired:
        return {"ok": False, "ready": True, "message": f"timeout sau {AGENT_TIMEOUT}s", "stdout": "", "stderr": ""}
    except OSError as e:
        return {"ok": False, "ready": True, "message": f"khong chay duoc: {e}", "stdout": "", "stderr": ""}
    try:
        data = json.loads(p.stdout)
    except ValueError:
        data = None
    return {"ok": p.returncode == 0, "ready": True, "returncode": p.returncode, "stdout": p.stdout,
            "stderr": p.stderr, "json": data, "job_url": _job_url(_session())}
