"""CLI live: python -X utf8 -m release_control.cli_live <lenh>. Hien noi: ingest, cleanup."""
from __future__ import annotations

import argparse
import json
import sys

COMMANDS = ["ingest", "prelabel", "route", "pull", "validate", "inject", "trace", "restore", "live", "cleanup"]


def cmd_ingest(args) -> int:
    from release_control.demo_session.session import create_session, job_url
    s = create_session(args.n)
    print(json.dumps({"task_id": s["task_id"], "job_id": s["job_id"], "task_name": s["task_name"],
                      "n_images": len(s["image_files"]), "job_url": job_url(s)}, ensure_ascii=False, indent=2))
    return 0


def cmd_cleanup(args) -> int:
    from release_control.demo_session.session import cleanup_session
    cleanup_session()
    print("Da xoa task cua session.")
    return 0


def _not_wired(name: str):
    def run(args) -> int:
        print(f"{name}: chua noi")
        return 0
    return run


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="release_control.cli_live")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest", help="Tao task/job CVAT local moi")
    p.add_argument("--n", type=int, default=20)
    p.set_defaults(func=cmd_ingest)
    sub.add_parser("cleanup", help="Xoa task cua session").set_defaults(func=cmd_cleanup)
    for c in COMMANDS:
        if c not in ("ingest", "cleanup"):
            sub.add_parser(c).set_defaults(func=_not_wired(c))
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except Exception as e:  # noqa: BLE001
        print(f"LOI: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
