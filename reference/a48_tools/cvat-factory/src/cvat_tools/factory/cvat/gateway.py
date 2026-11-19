"""CVAT API gateway wrapping core.client.Cvat with safe probing, org/membership queries, and PAT redaction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from cvat_tools.core.client import Cvat, CvatError, WriteNotAuthorized


def redact_secret(text: str, secret: str | None) -> str:
    """Replace any occurrence of a secret token in text with [REDACTED]."""
    if not secret or not text:
        return text
    return text.replace(secret, "[REDACTED]")


@dataclass(frozen=True)
class ConnectivityResult:
    """Result of verifying CVAT server reachability and PAT authentication."""

    reachable: bool
    authenticated: bool
    host: str
    server_version: str | None = None
    username: str | None = None
    user_id: int | None = None
    error: str | None = None


class CvatGateway:
    """Read-only-by-default CVAT REST gateway reusing cvat_tools.core.client.Cvat."""

    def __init__(
        self,
        host: str,
        pat: str,
        org: str = "",
        write_enabled: bool = False,
        timeout: float = 30.0,
        client: Any | None = None,
    ) -> None:
        if not pat and client is None:
            raise CvatError("Missing CVAT PAT.")
        self.host = host.rstrip("/")
        self._pat = pat or ""
        self.org = org or ""
        self.write_enabled = bool(write_enabled)
        self.timeout = float(timeout)
        self._client = client if client is not None else Cvat(
            self.host, self._pat, org=self.org, write_enabled=self.write_enabled
        )

    def __repr__(self) -> str:
        return (
            f"CvatGateway(host={self.host!r}, org={self.org!r}, "
            f"write_enabled={self.write_enabled!r}, pat='[REDACTED]')"
        )

    @property
    def raw_client(self) -> Any:
        return self._client

    def set_read_only(self, read_only: bool = True) -> None:
        self.write_enabled = not read_only
        if hasattr(self._client, "write_enabled"):
            self._client.write_enabled = self.write_enabled

    def _safe_error(self, exc: Exception) -> str:
        return redact_secret(str(exc), self._pat)

    def test_connection(self) -> ConnectivityResult:
        """Verify server reachability (/api/server/about) and token validity (/api/users/self)."""
        version: str | None = None
        try:
            about_data = self.about()
            if isinstance(about_data, dict):
                version = str(about_data.get("version") or "") or None
        except Exception as exc:
            return ConnectivityResult(
                reachable=False,
                authenticated=False,
                host=self.host,
                error=self._safe_error(exc),
            )

        try:
            user = self.me()
            return ConnectivityResult(
                reachable=True,
                authenticated=True,
                host=self.host,
                server_version=version,
                username=user.get("username") if isinstance(user, dict) else None,
                user_id=user.get("id") if isinstance(user, dict) else None,
            )
        except Exception as exc:
            return ConnectivityResult(
                reachable=True,
                authenticated=False,
                host=self.host,
                server_version=version,
                error=self._safe_error(exc),
            )

    # ---- Read operations ----
    def about(self) -> dict[str, Any]:
        return self._client.about()

    def me(self) -> dict[str, Any]:
        return self._client.me()

    def organizations(self) -> list[dict[str, Any]]:
        if hasattr(self._client, "organizations"):
            return self._client.organizations()
        return self._client.all("/api/organizations")

    def memberships(self, org: str | None = None) -> list[dict[str, Any]]:
        if hasattr(self._client, "memberships"):
            return self._client.memberships(org=org)
        params: dict[str, Any] = {}
        if org:
            params["org"] = org
        return self._client.all("/api/memberships", **params)

    def job(self, job_id: int) -> dict[str, Any]:
        return self._client.job(job_id)

    def task(self, task_id: int) -> dict[str, Any]:
        return self._client.task(task_id)

    def tasks(self) -> list[dict[str, Any]]:
        return self._client.tasks()

    def jobs(self) -> list[dict[str, Any]]:
        return self._client.jobs()

    def labels(self, job_id: int) -> list[dict[str, Any]]:
        return self._client.labels(job_id)

    def meta(self, job_id: int) -> dict[str, Any]:
        return self._client.meta(job_id)

    def annotations(self, job_id: int) -> dict[str, Any]:
        return self._client.annotations(job_id)

    def frame(self, job_id: int, frame_idx: int) -> bytes:
        return self._client.frame(job_id, frame_idx)

    def probe_annotation_methods(self, job_id: int) -> set[str]:
        """Non-mutating OPTIONS probe on /api/jobs/{job_id}/annotations to inspect allowed HTTP methods."""
        if hasattr(self._client, "probe_annotation_methods"):
            return set(self._client.probe_annotation_methods(job_id))
        headers = getattr(self._client, "h", None)
        if not headers:
            return {"GET"}
        try:
            resp = requests.request(
                "OPTIONS",
                f"{self.host}/api/jobs/{job_id}/annotations",
                headers=headers,
                timeout=self.timeout,
            )
            allow = resp.headers.get("Allow", "")
            methods = {m.strip().upper() for m in allow.split(",") if m.strip()}
            return methods or ({"GET"} if resp.status_code < 400 else set())
        except Exception:
            return set()

    # ---- Write operations (strictly gated by write_enabled) ----
    def _ensure_write(self) -> None:
        if not self.write_enabled:
            raise WriteNotAuthorized(
                "Gateway is in read-only mode. Enable write_enabled=True only after authorization."
            )

    def create_shapes(self, job_id: int, shapes: list[dict[str, Any]]) -> dict[str, Any]:
        self._ensure_write()
        return self._client.create_shapes(job_id, shapes)

    def update_shapes(self, job_id: int, shapes: list[dict[str, Any]]) -> dict[str, Any]:
        self._ensure_write()
        return self._client.update_shapes(job_id, shapes)

    def delete_shapes(self, job_id: int, shapes: list[dict[str, Any]]) -> None:
        self._ensure_write()
        self._client.delete_shapes(job_id, shapes)

    def create_tracks(self, job_id: int, tracks: list[dict[str, Any]]) -> dict[str, Any]:
        self._ensure_write()
        return self._client.create_tracks(job_id, tracks)

    def delete_tracks(self, job_id: int, tracks: list[dict[str, Any]]) -> None:
        self._ensure_write()
        self._client.delete_tracks(job_id, tracks)

    def to_sdk_client(self) -> Any:
        """Lazy-create a cvat_sdk.Client using the same host, org, and PAT."""
        import cvat_sdk

        return cvat_sdk.make_client(
            host=self.host,
            credentials=(None, self._pat),
        )
