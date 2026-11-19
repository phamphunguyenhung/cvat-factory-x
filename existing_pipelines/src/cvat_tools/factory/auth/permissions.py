"""Real CVAT API role & permission detection with automatic read-only fallback."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cvat_tools.factory.cvat.gateway import CvatGateway


@dataclass(frozen=True)
class UserIdentity:
    id: int | None
    username: str
    email: str = ""
    first_name: str = ""
    last_name: str = ""
    is_superuser: bool = False
    is_staff: bool = False
    groups: tuple[str, ...] = ()


@dataclass(frozen=True)
class OrgMembership:
    org_slug: str
    org_name: str = ""
    role: str = ""  # owner, maintainer, supervisor, worker, observer, or ""


@dataclass(frozen=True)
class PermissionReport:
    identity: UserIdentity
    membership: OrgMembership | None
    job_id: int | None = None
    task_id: int | None = None
    is_assignee: bool = False
    is_task_owner: bool = False
    can_read: bool = True
    can_create_annotations: bool = False
    can_update_annotations: bool = False
    can_delete_annotations: bool = False
    effective_mode: str = "read_only"  # "read_only" | "read_write"
    read_only_reason: str | None = "default_read_only"
    allowed_methods: frozenset[str] = field(default_factory=frozenset)


def _extract_username(obj: Any) -> str | None:
    if isinstance(obj, dict):
        return obj.get("username") or obj.get("name")
    if isinstance(obj, str):
        return obj
    return None


def _extract_user_id(obj: Any) -> int | None:
    if isinstance(obj, dict) and obj.get("id") is not None:
        return int(obj["id"])
    if isinstance(obj, int):
        return obj
    return None


def _matches_user(obj: Any, identity: UserIdentity) -> bool:
    if obj is None:
        return False
    uid = _extract_user_id(obj)
    if uid is not None and identity.id is not None and uid == identity.id:
        return True
    uname = _extract_username(obj)
    return bool(uname and uname == identity.username)


def detect_permissions(
    gateway: CvatGateway,
    *,
    job_id: int | None = None,
    requested_write: bool = False,
) -> PermissionReport:
    """Inspect real CVAT user identity, org membership, job assignee, and annotation permissions.

    Automatically switches `gateway` to read-only mode if write is not permitted.
    """
    me = gateway.me()
    identity = UserIdentity(
        id=me.get("id"),
        username=me.get("username") or "unknown",
        email=me.get("email") or "",
        first_name=me.get("first_name") or "",
        last_name=me.get("last_name") or "",
        is_superuser=bool(me.get("is_superuser", False)),
        is_staff=bool(me.get("is_staff", False)),
        groups=tuple(str(g) for g in (me.get("groups") or [])),
    )

    membership: OrgMembership | None = None
    if gateway.org:
        org_name = ""
        try:
            for o in gateway.organizations():
                if o.get("slug") == gateway.org:
                    org_name = o.get("name") or ""
                    break
        except Exception:
            pass
        role = ""
        try:
            for m in gateway.memberships(org=gateway.org):
                if _matches_user(m.get("user"), identity):
                    role = str(m.get("role") or "").lower()
                    break
        except Exception:
            pass
        membership = OrgMembership(org_slug=gateway.org, org_name=org_name, role=role)

    task_id: int | None = None
    is_assignee = False
    is_task_owner = False
    job_state = ""
    allowed_methods: set[str] = set()
    can_read = True

    if job_id is not None:
        try:
            job = gateway.job(job_id)
            task_id = job.get("task_id")
            job_state = str(job.get("state") or "").lower()
            is_assignee = _matches_user(job.get("assignee"), identity)
        except Exception:
            can_read = False
            job = {}

        if task_id is not None:
            try:
                task = gateway.task(task_id)
                is_task_owner = _matches_user(task.get("owner"), identity)
                if not is_assignee:
                    is_assignee = _matches_user(task.get("assignee"), identity)
            except Exception:
                pass

        allowed_methods = gateway.probe_annotation_methods(job_id)

    # Evaluate write capability
    role_val = membership.role if membership else ""
    write_permitted = False
    deny_reason: str | None = None

    if not can_read:
        write_permitted = False
        deny_reason = "cannot_read_job"
    elif identity.is_superuser or role_val in {"owner", "maintainer"}:
        write_permitted = True
    elif role_val == "supervisor":
        write_permitted = True
    elif role_val == "worker":
        if job_id is None or is_assignee or is_task_owner:
            write_permitted = True
        else:
            write_permitted = False
            deny_reason = "worker_not_assigned_to_job"
    elif role_val == "observer":
        write_permitted = False
        deny_reason = "observer_role_read_only"
    elif not gateway.org:
        # Personal workspace without org: owner/assignee can write
        if job_id is None or is_assignee or is_task_owner:
            write_permitted = True
        else:
            write_permitted = False
            deny_reason = "not_job_assignee_or_owner"
    else:
        # Unknown org role when memberships endpoint is restricted: rely on assignee/OPTIONS
        if is_assignee or is_task_owner or ("PATCH" in allowed_methods):
            write_permitted = True
        else:
            write_permitted = False
            deny_reason = "insufficient_org_permissions"

    if write_permitted and allowed_methods and not ({"PATCH", "PUT", "POST"} & allowed_methods):
        write_permitted = False
        deny_reason = "server_disallows_patch_on_annotations"

    if write_permitted and job_state == "completed" and not (identity.is_superuser or role_val in {"owner", "maintainer"}):
        write_permitted = False
        deny_reason = "job_completed_read_only"

    if requested_write and write_permitted:
        gateway.set_read_only(False)
        effective_mode = "read_write"
        read_only_reason = None
    else:
        gateway.set_read_only(True)
        effective_mode = "read_only"
        read_only_reason = deny_reason if not write_permitted else "default_read_only"

    return PermissionReport(
        identity=identity,
        membership=membership,
        job_id=job_id,
        task_id=task_id,
        is_assignee=is_assignee,
        is_task_owner=is_task_owner,
        can_read=can_read,
        can_create_annotations=write_permitted,
        can_update_annotations=write_permitted,
        can_delete_annotations=write_permitted and (identity.is_superuser or role_val in {"owner", "maintainer", ""}),
        effective_mode=effective_mode,
        read_only_reason=read_only_reason,
        allowed_methods=frozenset(allowed_methods),
    )
