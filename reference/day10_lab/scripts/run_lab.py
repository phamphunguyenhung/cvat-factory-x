#!/usr/bin/env python3
"""Tạo lô có lỗi gài cho một hoặc nhiều seed (detection + panoptic cùng một lô).

  python scripts/download_public.py
  python scripts/run_lab.py --seeds 11
  python scripts/score_findings.py --run runs/r11

Seed đầu tiên của lần chạy ĐẦU được gắn với training run R-021 trong runs/lineage.jsonl (chỉ một lần).
Muốn thêm seed để thử validator mà không đụng lineage: python scripts/inject_release.py --seed X --out runs/rX
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def python(script: str, *args: str) -> None:
    cmd = [sys.executable, str(SCRIPTS / script), *args]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Tạo lô có lỗi gài cho các seed")
    parser.add_argument("--seeds", default="11,22,33")
    args = parser.parse_args()
    if not (ROOT / "data_source" / "coco_panoptic250" / "panoptic.json").is_file():
        raise SystemExit("Chưa có dữ liệu. Chạy: python scripts/download_public.py")
    seeds = [str(int(part)) for part in args.seeds.split(",") if part.strip()]
    for index, seed in enumerate(seeds):
        out = ROOT / "runs" / f"r{seed}"
        python("inject_release.py", "--seed", seed, "--out", str(out))
        release = json.loads((out / "delivery" / "release.json").read_text(encoding="utf-8"))
        python("lineage.py", "record", "--event", "inject", "--seed", seed, "--release-id", release["release_id"], "--detail", f"r{seed}")
        lp = ROOT / "runs" / "lineage.jsonl"
        has_train = lp.is_file() and any(json.loads(x).get("event") == "train" for x in lp.read_text(encoding="utf-8").splitlines() if x.strip())
        if index == 0 and not has_train:
            python("lineage.py", "record", "--event", "train", "--release-id", release["release_id"], "--run-id", "R-021")
    print()
    print("Đã tạo delivery và ground truth. Chưa có danh sách lỗi của nhóm.")
    for seed in seeds:
        print(f"python scripts/score_findings.py --run runs/r{seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
