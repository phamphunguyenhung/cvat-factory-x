"""Test Matrix Items 1-6: PAT config, invalid PAT, role detection, read-only user, job discovery, taxonomy mismatch."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from cvat_tools.factory.auth import (
    MemoryCredentialStore,
    credential_target,
    detect_permissions,
    discover_credentials,
)
from cvat_tools.factory.cvat import CvatGateway
from cvat_tools.factory.discovery import discover_job, discover_jobs
from cvat_tools.factory.registry import (
    CVATLabelDef,
    HardwareProfile,
    ModelRegistryStore,
    ModelSelectionContext,
    ModelSelector,
)


class _FakeCvatClient:
    """In-memory mock of cvat_tools.core.client.Cvat for auth & discovery tests."""

    def __init__(
        self,
        *,
        me_payload: dict[str, Any] | None = None,
        me_error: Exception | None = None,
        memberships: list[dict[str, Any]] | None = None,
        job_payload: dict[str, Any] | None = None,
        task_payload: dict[str, Any] | None = None,
        labels_payload: list[dict[str, Any]] | None = None,
        annotations_payload: dict[str, Any] | None = None,
        allowed_methods: tuple[str, ...] = ("GET",),
    ) -> None:
        self.write_enabled = False
        self._me_payload = me_payload or {"id": 7, "username": "annotator", "email": "a@example.com"}
        self._me_error = me_error
        self._memberships = memberships if memberships is not None else []
        self._job_payload = job_payload or {
            "id": 1623,
            "task_id": 472,
            "stage": "annotation",
            "state": "in progress",
            "assignee": {"id": 7, "username": "annotator"},
            "start_frame": 90,
            "stop_frame": 119,
        }
        self._task_payload = task_payload or {
            "id": 472,
            "name": "week2-human-pose",
            "size": 30,
            "owner": {"id": 1, "username": "lead"},
        }
        self._labels_payload = labels_payload if labels_payload is not None else [
            {"id": 12322, "name": "person_vf17", "type": "skeleton", "sublabels": [{"id": 12323, "name": "1"}]}
        ]
        self._annotations_payload = annotations_payload or {"shapes": [], "tracks": [], "tags": []}
        self._allowed_methods = allowed_methods

    def about(self) -> dict[str, Any]:
        return {"version": "2.11.0"}

    def me(self) -> dict[str, Any]:
        if self._me_error is not None:
            raise self._me_error
        return self._me_payload

    def memberships(self, org: str | None = None) -> list[dict[str, Any]]:
        return list(self._memberships)

    def job(self, job_id: int) -> dict[str, Any]:
        return dict(self._job_payload)

    def jobs(self) -> list[dict[str, Any]]:
        return [dict(self._job_payload)]

    def task(self, task_id: int) -> dict[str, Any]:
        return dict(self._task_payload)

    def tasks(self) -> list[dict[str, Any]]:
        return [dict(self._task_payload)]

    def labels(self, job_id: int) -> list[dict[str, Any]]:
        return list(self._labels_payload)

    def annotations(self, job_id: int) -> dict[str, Any]:
        return dict(self._annotations_payload)

    def probe_annotation_methods(self, job_id: int) -> set[str]:
        return set(self._allowed_methods)


def test_01_pat_configuration_and_redaction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Item 1: PAT discovery via env/store without leaking secret in repr."""
    secret = "cvat_pat_super_secret_token_98765"
    monkeypatch.setenv("CVAT_PAT", secret)
    monkeypatch.setenv("CVAT_HOST", "https://cvat.example.local")

    store = MemoryCredentialStore()
    creds = discover_credentials(
        profile="local",
        store=store,
        verify=False,
    )
    assert creds.pat == secret
    assert creds.host == "https://cvat.example.local"
    assert creds.source.startswith("env:") or creds.source in {"env", "explicit"}
    assert secret not in repr(creds)

    target = credential_target("local", creds.host)
    store.set(target, "annotator", secret)
    assert store.get(target, "annotator") == secret


def test_02_expired_or_invalid_pat_redacts_secret() -> None:
    """Item 2: Expired/invalid PAT reports authenticated=False and never leaks PAT in error."""
    secret = "expired_pat_xyz_12345"
    fake_client = _FakeCvatClient(me_error=RuntimeError(f"401 Unauthorized for token {secret}"))
    gw = CvatGateway("https://cvat.example.local", secret, client=fake_client)

    res = gw.test_connection()
    assert res.reachable is True
    assert res.authenticated is False
    assert res.error is not None
    assert secret not in res.error
    assert secret not in repr(gw)


def test_03_role_detection_extracts_identity_org_and_permissions() -> None:
    """Item 3: Role detection inspects identity, org membership, assignee, and write flags."""
    fake_client = _FakeCvatClient(
        me_payload={"id": 7, "username": "annotator", "email": "a@example.com", "is_staff": False},
        memberships=[{
            "user": {"id": 7, "username": "annotator"},
            "organization": {"slug": "edufun", "name": "EduFun"},
            "role": "maintainer",
        }],
        allowed_methods=("GET", "POST", "PATCH", "PUT"),
    )
    gw = CvatGateway("https://cvat.example.local", "pat_ok", org="edufun", client=fake_client)
    report = detect_permissions(gw, job_id=1623, requested_write=True)

    assert report.identity.username == "annotator"
    assert report.membership is not None
    assert report.membership.org_slug == "edufun"
    assert report.membership.role == "maintainer"
    assert report.is_assignee is True
    assert report.can_read is True
    assert report.can_create_annotations is True
    assert report.can_update_annotations is True
    assert report.effective_mode == "read_write"


def test_04_read_only_user_auto_switches_to_read_only() -> None:
    """Item 4: Observer / read-only user cannot enable write; gateway forced to read_only."""
    fake_client = _FakeCvatClient(
        me_payload={"id": 99, "username": "observer_user"},
        memberships=[{"organization": {"slug": "edufun"}, "role": "observer"}],
        allowed_methods=("GET",),
    )
    gw = CvatGateway("https://cvat.example.local", "pat_ro", org="edufun", write_enabled=True, client=fake_client)
    report = detect_permissions(gw, job_id=1623, requested_write=True)

    assert report.effective_mode == "read_only"
    assert report.can_create_annotations is False
    assert gw.write_enabled is False


def test_05_job_discovery_returns_rich_metadata(tmp_path: Path) -> None:
    """Item 5: Job discovery returns Job ID, Task ID, labels, frame count, coverage, suitable model."""
    fake_client = _FakeCvatClient(
        annotations_payload={"shapes": [{"frame": 90, "type": "skeleton", "label_id": 12322}], "tracks": [], "tags": []}
    )
    gw = CvatGateway("https://cvat.example.local", "pat_ok", client=fake_client)

    job = discover_job(gw, 1623, db_path=tmp_path / "factory.sqlite")
    assert job.job_id == 1623
    assert job.task_id == 472
    assert job.task_name == "week2-human-pose"
    assert job.frame_count == 30
    assert job.annotation_type == "skeleton"
    assert len(job.labels) == 1
    assert job.coverage.annotated_frames == 1
    assert job.suitable_model != ""

    res = discover_jobs(gw, db_path=tmp_path / "factory.sqlite")
    assert len(res.jobs) >= 1


def test_06_taxonomy_mismatch_stops_at_review(tmp_path: Path) -> None:
    """Item 6: Unknown/mismatched taxonomy must not guess or silently force an incompatible model."""
    store = ModelRegistryStore(tmp_path / "registry.sqlite")
    selector = ModelSelector(store)
    ctx = ModelSelectionContext(
        labels=[CVATLabelDef(name="vertebra_c4", geometry_type="cuboid")],
        hardware=HardwareProfile(cpu_cores=4, ram_mb=8192, gpu_available=False),
    )
    decision = selector.select(ctx)
    assert decision.selected_model_id is None
    assert decision.status in {"inspect_required", "needs_review"}
