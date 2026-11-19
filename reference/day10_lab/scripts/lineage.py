#!/usr/bin/env python3
"""Append-only lineage for the classroom case, and an impact query.

Events live in runs/lineage.jsonl. A training run that names a release is affected
when that release is later held. This does not replace DVC or OpenLineage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def log_path(root: Path) -> Path:
    path = root / "lineage.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def append(root: Path, event: dict) -> None:
    with log_path(root).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def read(root: Path) -> list[dict]:
    path = log_path(root)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def impact(root: Path, release_id: str) -> list[dict]:
    events = read(root)
    held = any(row.get("event") == "decision" and row.get("release_id") == release_id and row.get("decision") == "HOLD" for row in events)
    runs = [row for row in events if row.get("event") == "train" and row.get("release_id") == release_id]
    return [{"run_id": row["run_id"], "release_id": release_id, "affected": held, "action": "review_or_retrain" if held else "none"} for row in runs]


def main() -> int:
    parser = argparse.ArgumentParser(description="Record lineage or query blast radius")
    parser.add_argument("--root", default="runs")
    sub = parser.add_subparsers(dest="cmd", required=True)

    record = sub.add_parser("record")
    record.add_argument("--event", required=True, choices=["source", "inject", "validate", "decision", "release", "train"])
    record.add_argument("--release-id")
    record.add_argument("--seed", type=int)
    record.add_argument("--run-id")
    record.add_argument("--decision")
    record.add_argument("--detail")
    record.add_argument("--parent", help="release cha (dùng với --event release)")

    query = sub.add_parser("impact")
    query.add_argument("--release-id", required=True)
    args = parser.parse_args()
    root = Path(args.root)

    if args.cmd == "impact":
        rows = impact(root, args.release_id)
        if not rows:
            print(f"no training run recorded for {args.release_id}")
            return 0
        for row in rows:
            print(f"{row['run_id']}: affected={row['affected']} action={row['action']}")
        return 0

    event = {"event": args.event}
    for key in ("release_id", "parent", "seed", "run_id", "decision", "detail"):
        value = getattr(args, key)
        if value is not None:
            event[key] = value
    append(root, event)
    print("recorded", event)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
