#!/usr/bin/env python3
"""Đóng gói một thư mục lô thành zip tái lập được (thứ tự cố định, thời gian cố định) + file .sha256.

  python scripts/pack_release.py runs/r11/handover/lab250_xxxxxxxxxxxx_v2

Ghi <thư mục>.zip và <thư mục>.sha256 cạnh thư mục. Cùng nội dung thì cùng mã băm, trên mọi máy.
"""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

SKIP = {".DS_Store", "Thumbs.db"}


def build_zip(folder: Path) -> bytes:
    """Zip tái lập: thứ tự file cố định, thời gian cố định, quyền cố định."""
    import io
    buf = io.BytesIO()
    files = sorted(p for p in folder.rglob("*") if p.is_file() and p.name not in SKIP)
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            info = zipfile.ZipInfo(f"{folder.name}/{p.relative_to(folder).as_posix()}", date_time=(2026, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, p.read_bytes())
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(description="Zip tái lập + sha256")
    ap.add_argument("folder")
    args = ap.parse_args()
    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        raise SystemExit(f"Không thấy thư mục {folder}")
    out = folder.with_suffix(".zip")
    data = build_zip(folder)
    out.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    folder.with_suffix(".sha256").write_text(f"{digest}  {out.name}\n", encoding="utf-8")
    n = sum(1 for p in folder.rglob("*") if p.is_file() and p.name not in SKIP)
    print(f"{out.name}: {n} file, sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
