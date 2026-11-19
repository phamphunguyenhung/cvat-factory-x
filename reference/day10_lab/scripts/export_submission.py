#!/usr/bin/env python3
"""Chép bài làm từ runs/ (không commit) sang submission/ (commit), và ngược lại cho người chấm.

  python scripts/export_submission.py --run runs/r3007            # nhóm: runs/r3007 -> submission/r3007
  python scripts/export_submission.py --import submission/r3007   # người chấm: tạo lại lô cùng seed rồi chép bài vào

Chép: errors.json, candidates.json, review_log.csv, trace.json, security.md, handover/ (trừ .zip),
và runs/lineage.jsonl -> submission/lineage.jsonl. Không chép delivery/ và _private/.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["errors.json", "candidates.json", "review_log.csv", "trace.json", "security.md"]


def copy(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        if (src / name).is_file():
            shutil.copyfile(src / name, dst / name)
    if (src / "handover").is_dir():
        shutil.rmtree(dst / "handover", ignore_errors=True)
        shutil.copytree(src / "handover", dst / "handover", ignore=shutil.ignore_patterns("*.zip", ".DS_Store"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Chép bài làm giữa runs/ và submission/")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="runs/r<seed> -> submission/r<seed>")
    g.add_argument("--import", dest="imp", help="submission/r<seed> -> runs/r<seed> (người chấm)")
    args = ap.parse_args()
    if args.run:
        run = Path(args.run)
        dst = ROOT / "submission" / run.name
        copy(run, dst)
        if (run.parent / "lineage.jsonl").is_file():
            shutil.copyfile(run.parent / "lineage.jsonl", ROOT / "submission" / "lineage.jsonl")
        print(f"đã chép {run} -> {dst}")
    else:
        sub = Path(args.imp)
        seed = sub.name.lstrip("r")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "inject_release.py"), "--seed", seed,
                        "--out", str(ROOT / "runs" / sub.name)], check=True)
        copy(sub, ROOT / "runs" / sub.name)
        if (sub.parent / "lineage.jsonl").is_file():
            shutil.copyfile(sub.parent / "lineage.jsonl", ROOT / "runs" / "lineage.jsonl")
        print(f"đã tạo lại lô seed {seed} và chép bài vào runs/{sub.name}. Chạy: python scripts/check_handover.py --run runs/{sub.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
