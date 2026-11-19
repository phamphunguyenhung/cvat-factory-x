"""Kho CUC BO (SQLite) luu phien ban shape de biet provenance, hoc tu lan nguoi sua, dong bo an toan.

Shape tool ghi len CVAT co source='manual' nen CVAT khong phan biet nhap tool va nhan nguoi; kho nay giu su that do.
Chi them dong (append-only): khong co API xoa; cvat_id la truong duy nhat duoc cap nhat (khi dong bo/rollback),
moi cap nhat ghi them vao bang events.
"""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from cvat_tools.core import paths

STORE_DIR = paths.DATA / "store"
DEFAULT_DB = STORE_DIR / "shapes.sqlite"
STAGES = ("rule_draft", "llm_edit", "human_final")
TOOL_STAGES = ("rule_draft", "llm_edit")
STATUSES = ("active", "rejected")

SCHEMA = """
CREATE TABLE IF NOT EXISTS shapes(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job INTEGER NOT NULL, frame INTEGER NOT NULL, label TEXT NOT NULL, shape_type TEXT NOT NULL,
  points TEXT NOT NULL, z_order INTEGER NOT NULL DEFAULT 0, occluded INTEGER NOT NULL DEFAULT 0,
  attributes TEXT NOT NULL DEFAULT '[]',
  stage TEXT NOT NULL CHECK(stage IN ('rule_draft','llm_edit','human_final')),
  status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','rejected')),
  parent_id INTEGER REFERENCES shapes(id),
  conf REAL, issues TEXT NOT NULL DEFAULT '[]',
  provenance TEXT NOT NULL DEFAULT '{}',
  cvat_id INTEGER, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_shapes_jfs ON shapes(job, frame, stage);
CREATE INDEX IF NOT EXISTS ix_shapes_cvat ON shapes(job, cvat_id);
CREATE INDEX IF NOT EXISTS ix_shapes_parent ON shapes(parent_id);
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, kind TEXT NOT NULL, job INTEGER, detail TEXT NOT NULL
);
-- migration cong them (CREATE IF NOT EXISTS): bang quyet dinh cua tang decision/. Khong dung toi bang shapes/events.
CREATE TABLE IF NOT EXISTS decisions(
  id INTEGER PRIMARY KEY AUTOINCREMENT, job INTEGER NOT NULL, frame INTEGER NOT NULL,
  shape_ids TEXT NOT NULL, action TEXT NOT NULL, reason_code TEXT NOT NULL, evidence_refs TEXT NOT NULL DEFAULT '[]',
  params TEXT NOT NULL DEFAULT '{}', rule_id TEXT, model_id TEXT, model_version TEXT,
  confidence REAL, calibrated INTEGER NOT NULL DEFAULT 0, priority_score REAL,
  before_revision INTEGER REFERENCES shapes(id), after_revision INTEGER REFERENCES shapes(id),
  validation_status TEXT NOT NULL DEFAULT 'pending',
  approval_status TEXT NOT NULL DEFAULT 'pending' CHECK(approval_status IN ('pending','approved','rejected')),
  schema_version INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_decisions_jf ON decisions(job, frame);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _pts(points) -> list[tuple[float, float]]:
    return [(float(points[i]), float(points[i + 1])) for i in range(0, len(points) - 1, 2)]


def vertex_deviation(a, b) -> float:
    """Do lech dinh trung binh (px). Cung so dinh: tung cap; khac so dinh: chamfer doi xung."""
    pa, pb = _pts(a), _pts(b)
    if not pa or not pb:
        return float("nan")
    if len(pa) == len(pb):
        return sum(math.dist(p, q) for p, q in zip(pa, pb)) / len(pa)

    def one(x, y):
        return sum(min(math.dist(p, q) for q in y) for p in x) / len(x)
    return (one(pa, pb) + one(pb, pa)) / 2


class ShapeStore:
    def __init__(self, db_path=None):
        self.path = Path(db_path) if db_path else DEFAULT_DB
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.commit()

    def close(self):
        self.db.close()

    # ---- ghi
    def add(self, job: int, frame: int, label: str, shape_type: str, points, stage: str, *, z_order: int = 0,
            occluded: bool = False, attributes=None, parent_id: int | None = None, conf: float | None = None,
            issues=None, provenance=None, cvat_id: int | None = None, status: str = "active") -> int:
        if stage not in STAGES:
            raise ValueError(f"stage phai thuoc {STAGES}")
        if status not in STATUSES:
            raise ValueError(f"status phai thuoc {STATUSES}")
        if parent_id is not None and self.get(parent_id) is None:
            raise ValueError(f"parent_id {parent_id} khong ton tai")
        cur = self.db.execute(
            "INSERT INTO shapes(job,frame,label,shape_type,points,z_order,occluded,attributes,stage,status,parent_id,conf,issues,provenance,cvat_id,created_at)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (job, frame, label, shape_type, json.dumps([float(x) for x in points]), int(z_order), int(bool(occluded)),
             json.dumps(attributes or [], ensure_ascii=False), stage, status, parent_id, conf,
             json.dumps(issues or [], ensure_ascii=False), json.dumps(provenance or {}, ensure_ascii=False), cvat_id, _now()))
        self.db.commit()
        return cur.lastrowid

    def set_cvat_id(self, row_id: int, cvat_id: int | None, note: str = "") -> None:
        old = self.db.execute("SELECT job,cvat_id FROM shapes WHERE id=?", (row_id,)).fetchone()
        if old is None:
            raise KeyError(row_id)
        self.db.execute("UPDATE shapes SET cvat_id=? WHERE id=?", (cvat_id, row_id))
        self.db.commit()
        self.log("set_cvat_id", old["job"], {"row": row_id, "old": old["cvat_id"], "new": cvat_id, "note": note})

    def log(self, kind: str, job: int | None, detail: dict) -> None:
        self.db.execute("INSERT INTO events(ts,kind,job,detail) VALUES(?,?,?,?)", (_now(), kind, job, json.dumps(detail, ensure_ascii=False)))
        self.db.commit()

    def events(self, job: int | None = None, kind: str | None = None) -> list[dict]:
        q, a = "SELECT * FROM events WHERE 1=1", []
        if job is not None:
            q += " AND job=?"
            a.append(job)
        if kind:
            q += " AND kind=?"
            a.append(kind)
        return [dict(r, detail=json.loads(r["detail"])) for r in self.db.execute(q + " ORDER BY id", a)]

    # ---- quyet dinh (decision/)
    def add_decision(self, rec: dict) -> int:
        cur = self.db.execute(
            "INSERT INTO decisions(job,frame,shape_ids,action,reason_code,evidence_refs,params,rule_id,model_id,model_version,"
            "confidence,calibrated,priority_score,before_revision,after_revision,validation_status,approval_status,schema_version,created_at)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (rec["job_id"], rec["frame_id"], json.dumps(rec["shape_ids"]), rec["proposed_action"], rec["reason_code"],
             json.dumps(rec.get("evidence_refs", []), ensure_ascii=False), json.dumps(rec.get("params", {}), ensure_ascii=False),
             rec.get("rule_id"), rec.get("model_id"), rec.get("model_version"), rec.get("confidence"),
             int(bool(rec.get("calibrated", False))), rec.get("priority_score"), rec.get("before_revision"),
             rec.get("after_revision"), rec.get("validation_status", "pending"), rec.get("approval_status", "pending"),
             rec.get("schema_version", 1), _now()))
        self.db.commit()
        return cur.lastrowid

    def decisions(self, job: int | None = None, frame: int | None = None, approval: str | None = None) -> list[dict]:
        q, a = "SELECT * FROM decisions WHERE 1=1", []
        for col, v in (("job", job), ("frame", frame), ("approval_status", approval)):
            if v is not None:
                q += f" AND {col}=?"
                a.append(v)
        out = []
        for r in self.db.execute(q + " ORDER BY id", a):
            d = dict(r)
            for k in ("shape_ids", "evidence_refs", "params"):
                d[k] = json.loads(d[k])
            d["calibrated"] = bool(d["calibrated"])
            out.append(d)
        return out

    def set_approval(self, decision_id: int, status: str, note: str = "") -> None:
        if status not in ("pending", "approved", "rejected"):
            raise ValueError("approval_status khong hop le")
        row = self.db.execute("SELECT job FROM decisions WHERE id=?", (decision_id,)).fetchone()
        if row is None:
            raise KeyError(decision_id)
        self.db.execute("UPDATE decisions SET approval_status=? WHERE id=?", (status, decision_id))
        self.db.commit()
        self.log("decision_approval", row["job"], {"decision": decision_id, "status": status, "note": note})

    # ---- doc
    @staticmethod
    def _row(r) -> dict | None:
        if r is None:
            return None
        d = dict(r)
        for k in ("points", "attributes", "issues", "provenance"):
            d[k] = json.loads(d[k])
        d["occluded"] = bool(d["occluded"])
        return d

    def get(self, row_id: int) -> dict | None:
        return self._row(self.db.execute("SELECT * FROM shapes WHERE id=?", (row_id,)).fetchone())

    def list(self, job: int | None = None, frame: int | None = None, stage: str | None = None, label: str | None = None,
             status: str | None = None, synced: bool | None = None) -> list[dict]:
        q, a = "SELECT * FROM shapes WHERE 1=1", []
        for col, v in (("job", job), ("frame", frame), ("stage", stage), ("label", label), ("status", status)):
            if v is not None:
                q += f" AND {col}=?"
                a.append(v)
        if synced is not None:
            q += " AND cvat_id IS NOT NULL" if synced else " AND cvat_id IS NULL"
        return [self._row(r) for r in self.db.execute(q + " ORDER BY id", a)]

    def by_cvat_id(self, job: int, cvat_id: int) -> list[dict]:
        return [self._row(r) for r in self.db.execute("SELECT * FROM shapes WHERE job=? AND cvat_id=? ORDER BY id", (job, cvat_id))]

    def children(self, row_id: int) -> list[dict]:
        return [self._row(r) for r in self.db.execute("SELECT * FROM shapes WHERE parent_id=? ORDER BY id", (row_id,))]

    def descendants(self, row_id: int) -> list[dict]:
        out, todo = [], [row_id]
        while todo:
            for c in self.children(todo.pop()):
                out.append(c)
                todo.append(c["id"])
        return out

    def latest(self, job: int, frame: int | None = None, label: str | None = None) -> list[dict]:
        """Phien ban moi nhat cua tung dong doi (la: dong khong la parent cua dong nao khac)."""
        q = "SELECT * FROM shapes s WHERE job=? AND NOT EXISTS(SELECT 1 FROM shapes c WHERE c.parent_id=s.id)"
        a: list = [job]
        if frame is not None:
            q += " AND frame=?"
            a.append(frame)
        if label is not None:
            q += " AND label=?"
            a.append(label)
        return [self._row(r) for r in self.db.execute(q + " ORDER BY id", a)]

    def diff(self, a_id: int, b_id: int) -> dict:
        a, b = self.get(a_id), self.get(b_id)
        if a is None or b is None:
            raise KeyError("id khong ton tai")
        keys = ("frame", "label", "shape_type", "z_order", "occluded", "attributes", "stage", "status")
        d = {k: (a[k], b[k]) for k in keys if a[k] != b[k]}
        dev = vertex_deviation(a["points"], b["points"]) if a["points"] != b["points"] else 0.0
        return {"fields": d, "vertex_dev_px": dev, "points_changed": a["points"] != b["points"],
                "n_vertices": (len(a["points"]) // 2, len(b["points"]) // 2)}
