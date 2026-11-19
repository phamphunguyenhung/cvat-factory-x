"""Output formatting and secret redaction helpers for `cvat-factory`."""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any

_PAT_PATTERNS = (
    re.compile(r"(Token\s+)([A-Za-z0-9._\-]{8,})", re.IGNORECASE),
    re.compile(r"(Bearer\s+)([A-Za-z0-9._\-]{8,})", re.IGNORECASE),
    re.compile(r"(CVAT_PAT\s*[=:]\s*)([^\s\"']+)", re.IGNORECASE),
)


def redact_secrets(text: str) -> str:
    """Mask any token or PAT-like substring before printing to stdout/stderr."""
    masked = text
    for pat in _PAT_PATTERNS:
        masked = pat.sub(r"\1***REDACTED***", masked)
    return masked


def to_jsonable(obj: Any) -> Any:
    """Convert dataclasses, Paths, and nested objects into JSON-serializable structures."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: to_jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        return to_jsonable(obj.model_dump())
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return to_jsonable(obj.to_dict())
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(item) for item in obj]
    return str(obj)


def emit_json(payload: Any) -> None:
    """Print JSON payload with secrets redacted."""
    raw = json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2)
    print(redact_secrets(raw))


def print_table(headers: list[str], rows: list[list[Any]]) -> None:
    """Print an aligned plain-text table for terminal output."""
    str_rows = [[redact_secrets(str(cell)) for cell in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in str_rows:
        for idx, cell in enumerate(row):
            if idx < len(widths):
                widths[idx] = max(widths[idx], len(cell))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths)))
    for row in str_rows:
        padded = row + [""] * (len(headers) - len(row))
        print(fmt.format(*padded[: len(headers)]))
