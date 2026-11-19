"""Audit log hash-chained (JSONL) + write-guard cho Agent Console.

Hop dong: AuditLog(path).append/read/verify_chain, guard_write(host, task_id, session) -> WriteDenied.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

GENESIS_HASH = "0" * 64
_SECRET_MARKERS = ("token", "password", "passwd", "pat", "secret", "api_key", "apikey", "authorization", "cookie")
_SAFE_PAT_KEYS = {"path", "patch", "pattern", "patches", "pattern_id"}
_LOCAL_HOSTS = {"localhost", "127.0.0.1"}
REDACTED = "[REDACTED]"


class WriteDenied(PermissionError):
    """Hanh dong ghi bi write-guard tu choi."""


def _is_secret_key(key: Any) -> bool:
    k = str(key).lower()
    if k in _SAFE_PAT_KEYS:
        return False
    if k == "pat" or k.endswith(("_pat", "-pat")) or k.startswith(("pat_", "pat-")):
        return True
    return any(m in k for m in _SECRET_MARKERS if m != "pat")


def redact(obj: Any) -> Any:
    """Loc de quy moi khoa chua token/password/pat/secret..."""
    if isinstance(obj, dict):
        return {str(k): (REDACTED if _is_secret_key(k) else redact(v)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [redact(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _digest(entry: dict[str, Any]) -> str:
    body = {k: v for k, v in entry.items() if k != "hash"}
    payload = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AuditLog:
    """JSONL append-only; moi dong co prev_hash + hash (sha256 cua noi dung dong)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def _last_hash(self) -> str:
        if not self.path.is_file():
            return GENESIS_HASH
        last = GENESIS_HASH
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    last = json.loads(line).get("hash", last)
                except json.JSONDecodeError:
                    last = hashlib.sha256(line.encode("utf-8")).hexdigest()
        return last

    def append(
        self,
        actor: str,
        action: str,
        target: Any,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        evidence: Any = None,
    ) -> dict[str, Any]:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            entry: dict[str, Any] = {
                "seq": 0,
                "ts": datetime.now(timezone.utc).isoformat(),
                "actor": str(actor),
                "action": str(action),
                "target": redact(target),
                "inputs": redact(inputs or {}),
                "outputs": redact(outputs or {}),
                "evidence": redact(evidence),
                "prev_hash": self._last_hash(),
            }
            entry["seq"] = len(self._read_all()) + 1
            entry["hash"] = _digest(entry)
            with self.path.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
            return entry

    def _read_all(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        rows.append({"_corrupt": line})
        return rows

    def read(self, limit: int | None = None) -> list[dict[str, Any]]:
        rows = self._read_all()
        return rows[-limit:] if limit else rows

    def verify_chain(self) -> bool:
        prev = GENESIS_HASH
        for row in self._read_all():
            if "_corrupt" in row or row.get("prev_hash") != prev or row.get("hash") != _digest(row):
                return False
            prev = row["hash"]
        return True


def _host_of(host: str) -> str:
    h = (host or "").strip().lower()
    if "://" in h:
        h = urlparse(h).hostname or ""
    else:
        h = h.split("/")[0]
        if h.startswith("["):
            h = h.split("]")[0].lstrip("[")
        else:
            h = h.split(":")[0]
    return h


def guard_write(
    host: str,
    task_id: Any,
    session: dict[str, Any],
    audit: AuditLog | None = None,
    actor: str = "release_control",
    action: str = "cvat_write",
) -> None:
    """Cho phep ghi CVAT khi: host local, task_id thuoc session, RC_READ_ONLY != 1. Nguoc lai raise + audit DENIED."""
    reason = None
    if _host_of(host) not in _LOCAL_HOSTS:
        reason = f"host khong phai localhost: {_host_of(host) or host!r}"
    elif task_id is None or str(task_id) != str((session or {}).get("task_id")):
        reason = f"task_id {task_id!r} khong nam trong session"
    elif os.environ.get("RC_READ_ONLY", "").strip().lower() in {"1", "true", "yes", "on"}:
        reason = "RC_READ_ONLY=1"
    if reason is None:
        return
    if audit is not None:
        audit.append(actor, "DENIED", {"host": host, "task_id": task_id},
                     inputs={"attempted_action": action}, outputs={"reason": reason})
    raise WriteDenied(reason)
