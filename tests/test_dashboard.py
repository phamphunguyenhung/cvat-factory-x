import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import dashboard.app as d  # noqa: E402

pytestmark = pytest.mark.skipif(not d._has_dataset(), reason="chua co dataset")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "WORK", tmp_path)
    monkeypatch.setattr(d, "FAULTY", tmp_path / "faulty")
    monkeypatch.setattr(d, "RESTORED", tmp_path / "restored")
    return TestClient(d.app)


def test_flow(client):
    assert client.get("/").status_code == 200
    assert client.get("/api/state").json()["has_dataset"]
    v0 = client.post("/api/validate", json={"source": "clean"}).json()
    assert v0["recommended_status"] in ("VALIDATED", "HOLD")
    s = client.post("/api/snapshot", json={"source": "clean", "tag": "t"}).json()
    r = client.post("/api/restore", json={"digest": s["snapshot_id"]}).json()
    assert r["match"] is True
    assert client.post("/api/inject", json={"faults": ["T1_MISSING_IMAGE"]}).status_code == 200
    v = client.post("/api/validate", json={"source": "faulty"}).json()
    assert v["recommended_status"] == "HOLD" and v["issues"]
    assert client.get("/api/lineage", params={"node": s["snapshot_id"]}).status_code == 200
    rel = client.post("/api/release", json={"digest": s["snapshot_id"]})
    assert rel.status_code == 200, rel.text
