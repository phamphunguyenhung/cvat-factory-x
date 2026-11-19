"""Cache dinh danh theo noi dung: khoa = hash(nguon, anh xa frame, hash checkpoint, cau hinh suy luan)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cvat_tools.core import paths


def file_hash(path: str | Path, n: int = 16, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()[:n]


def make_key(**parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:20]


class Cache:
    def __init__(self, kind: str, root: Path | None = None):
        self.dir = (root or paths.CACHE) / kind
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str):
        p = self.path(key)
        return json.loads(p.read_text(encoding="utf8")) if p.exists() else None

    def put(self, key: str, value, meta: dict | None = None) -> Path:
        p = self.path(key)
        p.write_text(json.dumps({"meta": meta or {}, "value": value}, ensure_ascii=False), encoding="utf8")
        return p
