"""Unified `cvat-factory` CLI parser and command dispatcher."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import os
import platform
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from cvat_tools.factory import __version__
from cvat_tools.factory.cli.output import emit_json, print_table, redact_secrets

_HEAVY_MODULES = ("torch", "ultralytics", "onnxruntime", "transformers")


def _shared_parent_parser() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="Emit structured JSON output")
    parent.add_argument("--verbose", "-v", action="store_true", default=argparse.SUPPRESS, help="Enable verbose logging")
    parent.add_argument("--host", default=argparse.SUPPRESS, help="CVAT server URL override")
    return parent


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level `cvat-factory` argument parser with all 11 commands."""
    shared = _shared_parent_parser()
    parser = argparse.ArgumentParser(
        prog="cvat-factory",
        description="CVAT Autolabel Factory V3 — Universal CLI & Model-Driven Annotation",
        parents=[shared],
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # 1. doctor
    p_doc = sub.add_parser("doctor", parents=[shared], help="Check Python, RAM/CPU, CVAT auth, permissions, and cached models")
    p_doc.add_argument("--offline", action="store_true", help="Skip live CVAT server connectivity check")
    p_doc.set_defaults(handler=cmd_doctor)

    # 2. setup
    p_setup = sub.add_parser("setup", parents=[shared], help="Zero-friction credential discovery and secure PAT configuration")
    p_setup.add_argument("--non-interactive", action="store_true", help="Do not prompt for missing PAT")
    p_setup.add_argument("--force", action="store_true", help="Re-prompt and overwrite stored PAT")
    p_setup.add_argument("--verify", action="store_true", help="Verify server connectivity and role permissions")
    p_setup.set_defaults(handler=cmd_setup)

    # 3. discover
    p_disc = sub.add_parser("discover", parents=[shared], help="Discover accessible CVAT tasks and jobs with suitability summary")
    p_disc.add_argument("--task", type=int, default=None, help="Filter by CVAT task ID")
    p_disc.add_argument("--status", default=None, help="Filter by job status (e.g. annotation, validation, completed)")
    p_disc.add_argument("--mine", action="store_true", help="Only show jobs assigned to current user")
    p_disc.add_argument("--limit", type=int, default=25, help="Maximum jobs to display")
    p_disc.set_defaults(handler=cmd_discover)

    # 4. models list
    p_models = sub.add_parser("models", parents=[shared], help="Manage and inspect the versioned local model registry")
    models_sub = p_models.add_subparsers(dest="models_action", metavar="<action>")
    p_mlist = models_sub.add_parser("list", parents=[shared], help="List registered/discovered models without loading weights")
    p_mlist.add_argument("--task", default=None, help="Filter by task type (pose17, face50, bbox, segmentation, tracking)")
    p_mlist.add_argument("--scan", action="store_true", help="Rescan repo and local framework caches before listing")
    p_mlist.set_defaults(handler=cmd_models_list)
    p_models.set_defaults(handler=cmd_models_list)

    # 5. jobs
    p_jobs = sub.add_parser("jobs", parents=[shared], help="List CVAT jobs with local run state and resume-first summary")
    p_jobs.add_argument("--task", type=int, default=None, help="Filter by CVAT task ID")
    p_jobs.add_argument("--status", default=None, help="Filter by job status")
    p_jobs.add_argument("--mine", action="store_true", help="Only show jobs assigned to current user")
    p_jobs.add_argument("--limit", type=int, default=25, help="Maximum jobs to display")
    p_jobs.add_argument("--select", action="store_true", help="Interactively choose a job")
    p_jobs.set_defaults(handler=cmd_jobs)

    # 6. inspect
    p_insp = sub.add_parser("inspect", parents=[shared], help="Inspect job schema, labels, geometry types, and suitability")
    p_insp.add_argument("--job", type=int, required=False, default=None, help="CVAT job ID (prompts/resumes if omitted)")
    p_insp.add_argument("--offline", action="store_true", help="Inspect from local snapshot/store without network")
    p_insp.set_defaults(handler=cmd_inspect)

    # 7. run
    p_run = sub.add_parser("run", parents=[shared], help="Run adaptive preprocessing, inference/proposal, QC, and local draft storage")
    p_run.add_argument("--job", type=int, required=False, default=None, help="CVAT job ID")
    p_run.add_argument(
        "--mode",
        default="auto",
        choices=("auto", "pose17", "face50", "bbox", "segmentation", "tracking", "rule", "deterministic"),
        help="Pipeline mode (default: auto from schema)",
    )
    p_run.add_argument("--model", default=None, help="Explicit model ID/name override")
    p_run.add_argument("--profile", default="balanced", help="Execution profile (fast, balanced, accurate)")
    p_run.add_argument("--start", type=int, default=0, help="Start frame index")
    p_run.add_argument("--limit", type=int, default=None, help="Max frames to process")
    p_run.add_argument("--offline", action="store_true", help="Use local cached frames/snapshot")
    p_run.add_argument("--agent-mode", default="none", choices=("none", "claude", "antigravity"), help="Optional agent escalation mode")
    p_run.add_argument("--resume", action="store_true", help="Resume latest incomplete run for this job")
    p_run.set_defaults(handler=cmd_run)

    # 8. review
    p_rev = sub.add_parser("review", parents=[shared], help="Render BEFORE/AFTER/OVERLAY evidence and open VS Code review dashboard")
    p_rev.add_argument("--job", type=int, required=False, default=None, help="CVAT job ID")
    p_rev.add_argument("--run-id", default=None, help="Specific local run ID to review")
    p_rev.add_argument("--max-cases", type=int, default=5, help="Max high-priority cases per review batch (default: 5)")
    p_rev.add_argument("--no-open", action="store_true", help="Generate evidence without launching VS Code")
    p_rev.set_defaults(handler=cmd_review)

    # 9. sync
    p_sync = sub.add_parser("sync", parents=[shared], help="Safely sync approved proposals to CVAT (dry-run by default)")
    p_sync.add_argument("--job", type=int, required=False, default=None, help="CVAT job ID")
    p_sync.add_argument("--run-id", default=None, help="Run ID to sync")
    p_sync.add_argument("--yes", action="store_true", help="Execute live CVAT write (requires write permission & clean staleness check)")
    p_sync.add_argument("--rollback", action="store_true", help="Rollback a previous sync instead of pushing")
    p_sync.add_argument("--latest", action="store_true", help="Target latest sync when rolling back")
    p_sync.set_defaults(handler=cmd_sync)

    # 10. benchmark
    p_bench = sub.add_parser("benchmark", parents=[shared], help="Benchmark V2 vs V3 latency, RAM, cache hits, and geometry failures")
    p_bench.add_argument("--job", type=int, required=False, default=None, help="CVAT job ID")
    p_bench.add_argument("--limit", type=int, default=20, help="Number of frames to benchmark")
    p_bench.add_argument("--compare-v2", action="store_true", help="Include V2 baseline comparison")
    p_bench.add_argument("--output", default=None, help="Write JSON benchmark report to path")
    p_bench.set_defaults(handler=cmd_benchmark)

    # 11. resume
    p_res = sub.add_parser("resume", parents=[shared], help="Resume the latest interrupted run or sync for a job")
    p_res.add_argument("--job", type=int, required=False, default=None, help="CVAT job ID")
    p_res.add_argument("--run-id", default=None, help="Specific run ID to resume")
    p_res.add_argument("--yes", action="store_true", help="Allow write if resuming an approved sync")
    p_res.set_defaults(handler=cmd_resume)

    return parser


def _import_attr(module_path: str, *attr_names: str) -> Any:
    """Import a module lazily and return the first available attribute."""
    mod = importlib.import_module(module_path)
    for name in attr_names:
        if hasattr(mod, name):
            return getattr(mod, name)
    raise AttributeError(f"{module_path} has none of {attr_names}")


def _resolve_job_id(args: argparse.Namespace) -> int | None:
    """Return explicit --job or attempt resume-first / interactive discovery."""
    if getattr(args, "job", None) is not None:
        return int(args.job)
    try:
        from cvat_tools.factory.discovery import resolve_default_job  # type: ignore[attr-defined]

        return resolve_default_job(host=getattr(args, "host", None))
    except (ImportError, AttributeError):
        return None


def cmd_doctor(args: argparse.Namespace) -> int:
    """Environment, dependency, auth, permission, and lazy-load health check."""
    import psutil

    pkgs: dict[str, str] = {}
    for pkg in ("requests", "numpy", "pyyaml", "opencv-python", "psutil", "cvat-sdk", "ultralytics", "torch", "onnxruntime", "mediapipe"):
        try:
            pkgs[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pkgs[pkg] = "not-installed"

    vm = psutil.virtual_memory()
    loaded_heavy = [m for m in _HEAVY_MODULES if m in sys.modules]

    auth_info: dict[str, Any] = {"status": "unavailable"}
    if not getattr(args, "offline", False):
        try:
            from cvat_tools.factory.auth import check_auth_status  # type: ignore[attr-defined]

            auth_info = check_auth_status(host=getattr(args, "host", None))
        except ImportError:
            auth_info = {"status": "auth-module-pending"}
        except Exception as exc:
            auth_info = {"status": "error", "error": redact_secrets(f"{type(exc).__name__}: {exc}")}
    else:
        auth_info = {"status": "skipped-offline"}

    model_summary: dict[str, Any] = {"status": "registry-module-pending"}
    try:
        from cvat_tools.factory.registry import ModelRegistry  # type: ignore[attr-defined]

        reg = ModelRegistry()
        models = reg.list_models() if hasattr(reg, "list_models") else reg.list()
        model_summary = {"status": "ok", "count": len(models)}
    except ImportError:
        pass
    except Exception as exc:
        model_summary = {"status": "error", "error": redact_secrets(str(exc))}

    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "ram_total_gb": round(vm.total / (1024**3), 2),
        "ram_available_gb": round(vm.available / (1024**3), 2),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "lazy_load_ok": len(loaded_heavy) == 0,
        "heavy_modules_loaded": loaded_heavy,
        "packages": pkgs,
        "auth": auth_info,
        "models": model_summary,
    }

    if getattr(args, "json", False):
        emit_json(report)
        return 0

    print(f"cvat-factory v{__version__} | Python {report['python']} | {report['platform']}")
    print(
        f"RAM: {report['ram_total_gb']} GB total ({report['ram_available_gb']} GB free) | "
        f"CPU: {report['cpu_cores_physical']}c/{report['cpu_cores_logical']}t"
    )
    print(f"Lazy load: {'OK (0 heavy imports)' if report['lazy_load_ok'] else f'WARN loaded={loaded_heavy}'}")
    print("Auth:", redact_secrets(str(auth_info)))
    print("Models:", model_summary)
    return 0


def cmd_setup(args: argparse.Namespace) -> int:
    """Run zero-friction credential discovery and secure PAT setup."""
    try:
        setup_fn = _import_attr("cvat_tools.factory.auth", "run_setup", "setup_credentials")
    except (ImportError, AttributeError) as exc:
        return _missing_backend("setup", exc, args)

    result = setup_fn(
        host=getattr(args, "host", None),
        non_interactive=getattr(args, "non_interactive", False),
        force=getattr(args, "force", False),
        verify=getattr(args, "verify", True),
    )
    if getattr(args, "json", False):
        emit_json(result)
    else:
        print(redact_secrets(f"Setup complete: {result}"))
    return 0


def cmd_discover(args: argparse.Namespace) -> int:
    """Discover CVAT tasks and jobs with permissions and model suitability."""
    try:
        discover_fn = _import_attr("cvat_tools.factory.discovery", "discover_jobs", "list_jobs")
    except (ImportError, AttributeError) as exc:
        return _missing_backend("discover", exc, args)

    jobs = discover_fn(
        host=getattr(args, "host", None),
        task_id=getattr(args, "task", None),
        status=getattr(args, "status", None),
        mine=getattr(args, "mine", False),
        limit=getattr(args, "limit", 25),
    )
    if getattr(args, "json", False):
        emit_json({"jobs": jobs})
        return 0

    if not jobs:
        print("No matching CVAT jobs found.")
        return 0

    rows = []
    for j in jobs:
        d = j if isinstance(j, dict) else vars(j)
        rows.append([
            d.get("job_id", d.get("id", "?")),
            d.get("task_id", "?"),
            str(d.get("task_name", ""))[:32],
            d.get("status", "?"),
            d.get("frame_count", "?"),
            d.get("annotation_type", "?"),
            d.get("suitable_model", "-"),
            d.get("last_run_status", "-"),
        ])
    print_table(["Job", "Task", "Task Name", "Status", "Frames", "Type", "Model", "Local Run"], rows)
    return 0


def cmd_models_list(args: argparse.Namespace) -> int:
    """List registered and cached models without loading weights."""
    try:
        from cvat_tools.factory.registry import ModelRegistry
    except (ImportError, AttributeError) as exc:
        return _missing_backend("models list", exc, args)

    reg = ModelRegistry()
    if getattr(args, "scan", False):
        reg.scan()
    else:
        existing = reg.list_models(task=getattr(args, "task", None))
        if not existing:
            reg.sync_config()

    models = reg.list_models(task=getattr(args, "task", None))

    if getattr(args, "json", False):
        emit_json({"models": models})
        return 0

    if not models:
        print("No models registered. Run `cvat-factory models list --scan` to scan local caches.")
        return 0

    rows = []
    for m in models:
        d = m if isinstance(m, dict) else vars(m)
        tasks = d.get("supported_tasks") or d.get("supported_task") or d.get("task") or "?"
        if isinstance(tasks, (list, tuple)):
            tasks = ",".join(str(t) for t in tasks)
        size_mb = d.get("size_mb")
        if size_mb is None and d.get("size_bytes"):
            size_mb = round(float(d["size_bytes"]) / (1024 * 1024), 1)
        device = d.get("device") or d.get("compute")
        if not device:
            devs = []
            if d.get("cpu_supported", True):
                devs.append("cpu")
            if d.get("gpu_supported"):
                devs.append("gpu")
            device = "/".join(devs) or "cpu"
        rows.append([
            d.get("model_id", d.get("name", "?")),
            d.get("framework", "?"),
            tasks,
            size_mb if size_mb is not None else "-",
            device,
            str(d.get("sha256", ""))[:12] or "-",
            d.get("benchmark_status", "-"),
        ])
    print_table(["Model", "Framework", "Task", "Size(MB)", "Device", "SHA256", "Bench"], rows)
    return 0


def cmd_jobs(args: argparse.Namespace) -> int:
    """List jobs with local run history (alias/focused view of discover)."""
    return cmd_discover(args)


def cmd_inspect(args: argparse.Namespace) -> int:
    """Inspect a CVAT job's schema, labels, geometry types, and model match."""
    job_id = _resolve_job_id(args)
    if job_id is None:
        print("ERROR: --job <id> is required (no resumable job found).", file=sys.stderr)
        return 2

    try:
        inspect_fn = _import_attr("cvat_tools.factory.discovery", "inspect_job")
    except (ImportError, AttributeError) as exc:
        return _missing_backend("inspect", exc, args)

    info = inspect_fn(
        job_id=job_id,
        host=getattr(args, "host", None),
        offline=getattr(args, "offline", False),
    )
    if getattr(args, "json", False):
        emit_json(info)
    else:
        print(redact_secrets(f"Job {job_id} inspection: {info}"))
    return 0


def _build_gateway(host: str | None, *, write_enabled: bool = False) -> Any:
    """Construct a read-only-by-default CvatGateway via zero-friction credential discovery."""
    from cvat_tools.factory.auth import discover_credentials
    from cvat_tools.factory.cvat import CvatGateway

    creds = discover_credentials(host=host, allow_prompt=False)
    if not creds.pat:
        raise RuntimeError("Missing CVAT PAT. Run `cvat-factory setup` first.")
    return CvatGateway(host=creds.host, pat=creds.pat, write_enabled=write_enabled)


def _find_run_log(job_id: int, run_id: str | None = None) -> Path | None:
    """Locate the latest (or matching run_id) sync run log in data/runs/."""
    from cvat_tools.core.paths import DATA

    runs_dir = DATA / "runs"
    if not runs_dir.exists():
        return None
    candidates = sorted(runs_dir.glob(f"*{job_id}*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if run_id:
        for p in candidates:
            if run_id in p.name:
                return p
    return candidates[0] if candidates else None


def cmd_run(args: argparse.Namespace) -> int:
    """Execute the end-to-end local-first annotation pipeline (dry-run / local SQLite)."""
    job_id = _resolve_job_id(args)
    if job_id is None:
        print("ERROR: --job <id> is required.", file=sys.stderr)
        return 2

    from cvat_tools.core.paths import DATA
    from cvat_tools.factory.orchestration import run_job_pipeline
    from cvat_tools.factory.storage import FactoryDraftStore

    store = FactoryDraftStore(DATA / "store" / "factory_drafts.sqlite")
    provider_map = {"none": "none", "claude": "claude", "antigravity": "antigravity"}
    result = run_job_pipeline(
        job_id=job_id,
        mode=getattr(args, "mode", "auto"),
        provider=provider_map.get(getattr(args, "agent_mode", "none"), "none"),
        dry_run=True,
        model=getattr(args, "model", None),
        profile=getattr(args, "profile", "balanced"),
        start=getattr(args, "start", 0),
        limit=getattr(args, "limit", None),
        offline=getattr(args, "offline", False),
        resume=getattr(args, "resume", False),
        host=getattr(args, "host", None),
    )
    run_id = result.get("run_id")
    if run_id and hasattr(store, "create_run"):
        try:
            store.create_run(run_id=str(run_id), job_id=job_id, mode=str(result.get("mode", "auto")))
        except Exception:
            pass
    if getattr(args, "json", False):
        emit_json(result)
    else:
        print(redact_secrets(f"Run completed for job {job_id}: {result}"))
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    """Generate BEFORE/AFTER/OVERLAY evidence and open VS Code review dashboard."""
    job_id = _resolve_job_id(args)
    if job_id is None:
        print("ERROR: --job <id> is required.", file=sys.stderr)
        return 2

    from cvat_tools.factory.review import open_review

    result = open_review(
        job_id=job_id,
        run_id=getattr(args, "run_id", None),
        max_cases=max(1, min(5, int(getattr(args, "max_cases", 5)))),
        auto_open=not getattr(args, "no_open", False),
    )

    if getattr(args, "json", False):
        emit_json(result)
    else:
        print(redact_secrets(f"Review prepared for job {job_id}: {result}"))
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    """Sync approved proposals to CVAT (dry-run by default; --yes to write)."""
    job_id = _resolve_job_id(args)
    if job_id is None:
        print("ERROR: --job <id> is required.", file=sys.stderr)
        return 2

    from cvat_tools.factory.sync import rollback_sync, sync_job

    yes = bool(getattr(args, "yes", False))
    gateway = _build_gateway(getattr(args, "host", None), write_enabled=yes)

    if getattr(args, "rollback", False):
        run_log = _find_run_log(job_id, getattr(args, "run_id", None))
        if run_log is None:
            raise FileNotFoundError(f"No sync run log found for job {job_id} in data/runs/.")
        result = rollback_sync(gateway, run_log, yes=yes)
    else:
        result = sync_job(gateway, job_id, operations=(), yes=yes)

    if getattr(args, "json", False):
        emit_json(result)
    else:
        print(redact_secrets(f"Sync ({'LIVE' if yes else 'DRY-RUN'}) for job {job_id}: {result}"))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run real V2 vs V3 benchmark metrics on representative frames."""
    job_id = _resolve_job_id(args)
    try:
        bench_fn = _import_attr("cvat_tools.factory.benchmark", "run_benchmark")
        result = bench_fn(
            job_id=job_id,
            limit=getattr(args, "limit", 20),
            compare_v2=getattr(args, "compare_v2", False),
            output_path=getattr(args, "output", None),
        )
    except (ImportError, AttributeError):
        try:
            script_mod = _import_attr("scripts.benchmark_factory_v3", "run_benchmark")
            result = script_mod(
                job_id=job_id,
                limit=getattr(args, "limit", 20),
                compare_v2=getattr(args, "compare_v2", False),
                output_path=getattr(args, "output", None),
            )
        except (ImportError, AttributeError) as exc:
            return _missing_backend("benchmark", exc, args)

    if getattr(args, "json", False):
        emit_json(result)
    else:
        print(redact_secrets(f"Benchmark completed: {result}"))
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    """Resume the latest interrupted run or sync for a job."""
    job_id = _resolve_job_id(args)
    if job_id is None:
        print("ERROR: --job <id> is required.", file=sys.stderr)
        return 2

    from cvat_tools.factory.orchestration import resume_job

    run_log = _find_run_log(job_id, getattr(args, "run_id", None))
    result = resume_job(
        job_id=job_id,
        run_log_path=str(run_log) if run_log else None,
        dry_run=not bool(getattr(args, "yes", False)),
    )
    if getattr(args, "json", False):
        emit_json(result)
    else:
        print(redact_secrets(f"Resume completed for job {job_id}: {result}"))
    return 0


def _missing_backend(command: str, exc: Exception, args: argparse.Namespace) -> int:
    msg = f"Command '{command}' backend not available yet: {type(exc).__name__}: {exc}"
    if getattr(args, "json", False):
        emit_json({"ok": False, "command": command, "error": msg})
    else:
        print(f"ERROR: {msg}", file=sys.stderr)
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for `cvat-factory`."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    handler: Callable[[argparse.Namespace], int] | None = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    try:
        return int(handler(args))
    except SystemExit:
        raise
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        err = redact_secrets(f"{type(exc).__name__}: {exc}")
        if getattr(args, "json", False):
            emit_json({"ok": False, "error": err})
        else:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1
