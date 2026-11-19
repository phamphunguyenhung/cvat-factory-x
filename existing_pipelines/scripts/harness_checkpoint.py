"""Lightweight task checkpoint helper for post-compaction resume."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / ".agents" / "state"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _git_branch() -> str:
    res = subprocess.run(
        ["git", "-C", str(ROOT), "branch", "--show-current"],
        capture_output=True,
        text=True,
        check=False,
    )
    return res.stdout.strip() or "unknown"


def _resolve_rel(rel_path: str) -> Path:
    p = Path(rel_path)
    return p if p.is_absolute() else (ROOT / p)


def cmd_save(args: argparse.Namespace) -> int:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    out = STATE_DIR / f"{args.task_id}.ckpt.json"
    existing: dict = {}
    if out.exists():
        existing = json.loads(out.read_text(encoding="utf-8"))

    inspected: dict[str, str] = dict(existing.get("files_inspected", {}))
    for rel in args.inspected or []:
        p = _resolve_rel(rel)
        if p.exists() and p.is_file():
            inspected[rel.replace("\\", "/")] = _sha256(p)

    modified = list(dict.fromkeys([*existing.get("files_modified", []), *(args.modified or [])]))
    passed = list(dict.fromkeys([*existing.get("tests_passed", []), *(args.passed or [])]))
    failed = list(args.failed) if args.failed is not None else existing.get("tests_failed", [])

    payload = {
        "task_id": args.task_id,
        "mode": args.mode or existing.get("mode", "FAST"),
        "objective": args.objective or existing.get("objective", ""),
        "branch": _git_branch(),
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files_inspected": inspected,
        "files_modified": modified,
        "tests_passed": passed,
        "tests_failed": failed,
        "blockers": args.blocker if args.blocker is not None else existing.get("blockers", []),
        "next_action": args.next_action or existing.get("next_action", ""),
        "key_decisions": list(dict.fromkeys([*existing.get("key_decisions", []), *(args.decision or [])])),
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved checkpoint: {out.relative_to(ROOT)}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    ckpt_path = _resolve_rel(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 2

    data = json.loads(ckpt_path.read_text(encoding="utf-8"))
    current_branch = _git_branch()
    stale: list[str] = []
    missing: list[str] = []

    for rel, recorded_hash in data.get("files_inspected", {}).items():
        p = _resolve_rel(rel)
        if not p.exists():
            missing.append(rel)
        elif _sha256(p) != recorded_hash:
            stale.append(rel)

    branch_match = data.get("branch") == current_branch
    print(f"task_id={data.get('task_id')} mode={data.get('mode')} branch_match={branch_match} ({current_branch})")
    print(f"next_action={data.get('next_action')}")
    print(f"stale={stale}")
    print(f"missing={missing}")
    return 1 if (stale or missing or not branch_match) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage compact task checkpoints.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_save = sub.add_parser("save", help="Create or update a task checkpoint.")
    p_save.add_argument("task_id")
    p_save.add_argument("--mode", choices=["FAST", "STANDARD", "DEEP"])
    p_save.add_argument("--objective")
    p_save.add_argument("--next-action")
    p_save.add_argument("--inspected", nargs="*", help="Inspected file paths (hashed)")
    p_save.add_argument("--modified", nargs="*", help="Modified file paths")
    p_save.add_argument("--passed", nargs="*", help="Passed test commands/files")
    p_save.add_argument("--failed", nargs="*", help="Failed test commands/files")
    p_save.add_argument("--blocker", nargs="*", help="Unresolved blockers")
    p_save.add_argument("--decision", nargs="*", help="Brief key decisions")
    p_save.set_defaults(func=cmd_save)

    p_val = sub.add_parser("validate", help="Validate checkpoint hashes & branch against disk.")
    p_val.add_argument("checkpoint")
    p_val.set_defaults(func=cmd_validate)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
