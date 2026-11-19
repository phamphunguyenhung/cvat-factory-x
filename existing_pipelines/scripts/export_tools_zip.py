"""Đóng gói công cụ CVAT (Week 2) thành 1 file zip để chép sang máy khác. Không kèm secret, dữ liệu, model nặng.

  python scripts/export_tools_zip.py                 # -> dist/cvat-tools-week2_<ngày>.zip
  python scripts/export_tools_zip.py --open          # mở File Explorer chọn sẵn file zip (Windows)

Cấu trúc trong zip giữ nguyên đường dẫn tương đối để `paths.REPO` vẫn đúng:
  learning/cvat/...                         toolkit cvat_tools + config + scripts + tests + weeks + week02 (chỉ tài liệu)
  learning/tools/skeleton-pose-manager/...  spm (YOLO pose Day 4) mà cvat_tools.pose.week2 tái dùng (không kèm .env, model, nháp)
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import zipfile
from datetime import date
from pathlib import Path

CVAT = Path(__file__).resolve().parents[1]
REPO = CVAT.parent
SPM = REPO / "tools" / "skeleton-pose-manager"

SKIP_DIRS = {"__pycache__", ".pytest_cache", ".git", ".pi", "dist", "node_modules", ".venv", "venv", "thumbs"}
# thư mục dữ liệu/sinh ra: chỉ loại ở đúng vị trí (KHÔNG loại package src/cvat_tools/review, ...)
SKIP_TOP = {"cvat": {"data", "models", "reports", "week01"}, "spm": {"models", "drafts", "backups", "runs"}}
SKIP_WEEK_SUB = {"drafts", "backups", "review", "metrics"}
SKIP_NAMES = {".env", "kernel.errors.txt", "download.log"}
SKIP_SUFFIX = {".pt", ".pth", ".onnx", ".bin", ".pyc", ".log", ".sqlite", ".db", ".zip", ".jpg", ".jpeg", ".png"}
KEEP_PNG = {"face50_template.png", "pose17_template.png"}          # ảnh khung mẫu của guideline
SECRET = re.compile(r"^[ 	]*[A-Z_]*(?:PAT|TOKEN|PASSWORD|SECRET|API_KEY)[A-Z_]*[ 	]*=[ 	]*[A-Za-z0-9._\-]{16,}[ 	]*$", re.M)

INSTALL = """# Cài công cụ CVAT Week 2 (máy mới)

Giải nén giữ nguyên thư mục `learning/`. Cần Python >= 3.11.

```powershell
cd <thư mục giải nén>\\learning\\cvat
pip install requests numpy pyyaml scipy opencv-python ultralytics face-alignment
pip install -e . --no-deps
copy .env.example .env          # rồi điền dòng CVAT_EDUFUN_PAT=<PAT edufun của bạn>   (KHÔNG chép .env cũ qua máy khác)
python -m cvat_tools week status
python -m cvat_tools pose draft --job <JOB_ID>            # sinh nháp local (lần đầu tải trọng số FAN và YOLO)
python -m cvat_tools pose apply --job <JOB_ID> --replace-auto          # dry-run; thêm --yes để ghi lên CVAT (có backup + rollback)
python scripts/eval_face50.py --job <JOB_ID> --refresh-gold            # đo nháp so với nhãn bạn đã sửa tay
```

- Model YOLO pose `yolo11l-pose.pt` do ultralytics tự tải vào `learning/tools/skeleton-pose-manager/models/` lần đầu.
- Thư mục `week02/<task>/<job>/` cần có sẵn (SPEC.md) để `pose draft` biết nơi lưu nháp/backup; tạo thủ công nếu job mới.
- Guideline nội bộ: `weeks/guidelines/w2-face50-pose17.md`. Quy tắc Q (occluded) và O (outside) ở mục 5.
- Nháp AI có `source=auto`. Chỉ dùng khi BTC cho phép (guideline mục 8 câu 4).
"""


def keep(p: Path, root: Path, tag: str) -> bool:
    rel = p.relative_to(root)
    parts = rel.parts
    if any(x in SKIP_DIRS for x in parts[:-1]) or p.name in SKIP_NAMES:
        return False
    if parts[0] in SKIP_TOP[tag]:
        return False
    if tag == "cvat" and parts[0] == "week02" and any(x in SKIP_WEEK_SUB for x in parts[1:-1]):
        return False
    if p.suffix.lower() in SKIP_SUFFIX and p.name not in KEEP_PNG:
        return False
    return p.is_file()


def collect() -> list[tuple[Path, str]]:
    files = []
    for root, prefix, tag in ((CVAT, "learning/cvat", "cvat"), (SPM, "learning/tools/skeleton-pose-manager", "spm")):
        for p in sorted(root.rglob("*")):
            if keep(p, root, tag):
                files.append((p, f"{prefix}/{p.relative_to(root).as_posix()}"))
    # week02: chỉ tài liệu/cấu hình (md/json), bỏ drafts, backups, review, metrics (đã lọc ở SKIP_DIRS) và các json nặng
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None); ap.add_argument("--open", action="store_true")
    a = ap.parse_args()
    out = Path(a.out) if a.out else CVAT / "dist" / f"cvat-tools-week2_{date.today():%Y%m%d}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    files = collect()
    leaks = []
    for p, arc in files:
        if p.suffix.lower() in {".py", ".md", ".yaml", ".yml", ".json", ".txt", ".toml", ".example", ""} or p.name.endswith(".example"):
            m = SECRET.search(p.read_text(encoding="utf8", errors="ignore"))
            if m:
                leaks.append((arc, m.group(0)[:12] + "..."))
    if leaks:
        sys.exit("DỪNG: nghi lộ secret trong " + "; ".join(f"{a} ({s})" for a, s in leaks))
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p, arc in files:
            z.write(p, arc)
        z.writestr("INSTALL.md", INSTALL)
    print(f"{len(files) + 1} file, {out.stat().st_size / 1024:.0f} KB -> {out}")
    if a.open and sys.platform == "win32":
        subprocess.Popen(["explorer", "/select,", str(out)])


if __name__ == "__main__":
    main()
