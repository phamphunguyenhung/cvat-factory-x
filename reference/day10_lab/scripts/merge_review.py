#!/usr/bin/env python3
"""Gộp lỗi luật của validator với các dòng confirmed trong review_log.csv thành errors.json cuối.

  python scripts/merge_review.py --rules runs/r11/rules.json --review runs/r11/review_log.csv --out runs/r11/errors.json

--rules: errors.json validator ghi ra (lỗi luật; lỗi gán nhãn trong file này bị bỏ, chỉ lấy từ review).
--review: image,type,verdict,note. Chỉ dòng verdict=confirmed được đưa vào.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Gộp lỗi luật + review đã xác nhận")
    ap.add_argument("--rules", required=True)
    ap.add_argument("--review", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rules = [e for e in json.loads(Path(args.rules).read_text(encoding="utf-8"))["errors"] if e["type"] in metrics.RULE_TYPES]
    with open(args.review, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    bad = [r for r in rows if r.get("verdict", "").strip() == "confirmed" and r.get("type", "").strip() not in metrics.SEMANTIC_TYPES]
    if bad:
        raise SystemExit(f"review_log có type không phải lỗi gán nhãn: {[r.get('type') for r in bad][:3]}")
    conf = [{"type": r["type"].strip(), "target": r["image"].strip()} for r in rows if r.get("verdict", "").strip() == "confirmed"]
    out = sorted({(e["type"], e["target"]) for e in rules + conf})
    Path(args.out).write_text(json.dumps({"errors": [{"type": t, "target": g} for t, g in out]}, indent=1) + "\n", encoding="utf-8")
    print(f"{len(rules)} lỗi luật + {len(conf)} lỗi gán nhãn đã xác nhận -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
