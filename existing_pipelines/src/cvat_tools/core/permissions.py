"""Permission gates for CVAT Tools (read-only by default, explicit approval for remote writes)."""
from __future__ import annotations

from enum import Enum


class Permission(str, Enum):
    READ_ONLY = "read_only"
    LOCAL_WRITE = "local_write"
    REMOTE_WRITE = "remote_write"


class PermissionDeniedError(PermissionError):
    """Raised when a tool requires a permission not authorized by the caller."""


DEFAULT_ALLOWED: frozenset[Permission] = frozenset({Permission.READ_ONLY, Permission.LOCAL_WRITE})


def check_permissions(required: tuple[Permission, ...] | list[Permission], allowed: frozenset[Permission] | set[Permission]) -> None:
    missing = [p.value for p in required if p not in allowed]
    if missing:
        raise PermissionDeniedError(
            f"Quyen bi tu choi: can {missing}. Mac dinh chi cho phep read_only/local_write; "
            "ghi len CVAT can duyet ro rang (--yes / write_enabled)."
        )
