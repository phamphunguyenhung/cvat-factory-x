from release_control.agent.prelabel import SimulatedDetector, run_prelabel
from release_control.agent.routing import route


class FakeClient:
    def __init__(self):
        self.shapes = []

    def put_annotations(self, shapes):
        self.shapes = list(shapes)  # replace semantics, like CVAT PUT

    def get_job_annotations(self):
        return {"shapes": self.shapes}


class FixedDetector:
    simulated = False
    model_version = "fake-1"

    def detect(self, image):
        n = sum(map(ord, image.name)) % 3  # 0..2 boxes
        return [{"label": "person", "box": [10 + i * 5, 10, 200, 200], "confidence": 0.9 - 0.3 * i}
                for i in range(n)]


SESSION = {"task_id": 1, "job_id": 2, "label_map": {"person": 7},
           "image_files": [f"img{i}.jpg" for i in range(6)]}


def test_prelabel_idempotent_and_auto():
    c = FakeClient()
    r1 = run_prelabel(SESSION, client=c, detector=FixedDetector())
    first = list(c.shapes)
    r2 = run_prelabel(SESSION, client=c, detector=FixedDetector())
    assert c.shapes == first and r1["shapes_written"] == r2["shapes_written"] == len(first)
    assert first and all(s["source"] == "auto" and s["label_id"] == 7 for s in first)
    assert {a["name"] for a in first[0]["attributes"]} == {"model_version", "confidence"}


def test_simulated_flagged():
    c = FakeClient()
    r = run_prelabel(SESSION, client=c, detector=SimulatedDetector())
    assert r["simulated"] is True and all(s["simulated"] for s in c.shapes)


def test_unknown_label_skipped():
    c = FakeClient()
    run_prelabel({**SESSION, "label_map": {"car": 1}}, client=c, detector=FixedDetector())
    assert c.shapes == []


def test_route_sorted_with_note():
    c = FakeClient()
    run_prelabel(SESSION, client=c, detector=FixedDetector())
    out = route(SESSION, client=c)
    risks = [i["risk"] for i in out["queue"]]
    assert len(risks) == 6 and risks == sorted(risks, reverse=True)
    assert set(out["queue"][0]) == {"frame", "risk", "reasons"}
    assert "NOT evidence" in out["note"]
