import pytest

from release_control.scale import capacity
from release_control.scale.review_queue import ReviewQueue, stratified_sample
from release_control.scale.risk import score_frame
from release_control.scale.sharding import iter_shards, run_batches


def test_risk_reasons_and_order():
    pol = {"class_severity": {"ped": 1.0}}
    hi = score_frame({"confidence": 0.1, "occlusion": 0.9, "classes": ["ped"]}, pol)
    lo = score_frame({"confidence": 0.99, "classes": ["car"]}, pol)
    assert hi["score"] > lo["score"] and "uncertainty" in hi["reasons"]


def test_resume_after_crash(tmp_path):
    cp = str(tmp_path / "cp.json")
    seen = []

    def proc(sid, fr):
        if sid == "shard-000002" and "boom" not in seen:
            seen.append("boom")
            raise KeyboardInterrupt  # simulated mid-run crash
        seen.append(sid)

    with pytest.raises(KeyboardInterrupt):
        run_batches(range(50), 10, proc, cp)
    assert [s for s in seen if s != "boom"] == ["shard-000000", "shard-000001"]
    st = run_batches(range(50), 10, proc, cp)
    assert st["skipped"] == 2 and st["processed"] == 3
    done = [s for s in seen if s != "boom"]
    assert len(done) == len(set(done)) == 5


def test_idempotent(tmp_path):
    cp = str(tmp_path / "cp.json")
    calls = []
    run_batches(range(20), 5, lambda s, f: calls.append(s), cp)
    st = run_batches(range(20), 5, lambda s, f: calls.append(s), cp)
    assert len(calls) == 4 and st["processed"] == 0 and st["skipped"] == 4


def test_dead_letter_bounded_retry(tmp_path):
    cp = str(tmp_path / "cp.json")
    n = {"c": 0}

    def proc(sid, fr):
        if sid == "shard-000001":
            n["c"] += 1
            raise RuntimeError("bad")

    st = run_batches(range(30), 10, proc, cp, max_retries=3)
    assert n["c"] == 3 and st["dead"] == 1 and st["processed"] == 2
    st2 = run_batches(range(30), 10, proc, cp, max_retries=3)
    assert n["c"] == 3 and st2["skipped"] == 3


def test_generator_lazy():
    g = iter_shards((i for i in range(10**9)), 100)
    assert len(next(g)[1]) == 100


def test_priority_order_and_single_assignment():
    q = ReviewQueue()
    for fid, r in [("a", 0.2), ("b", 0.9), ("c", 0.5)]:
        q.add(fid, r)
    assert q.add("a", 0.99) is False
    assert q.pop_batch("u1", 2) == ["b", "c"]
    assert q.pop_batch("u2", 5) == ["a"]
    with pytest.raises(ValueError):
        q.assign("b", "u2")
    assert not q.is_reviewed("b")
    with pytest.raises(ValueError):
        q.record_review("b", "u1", {})
    q.record_review("b", "u1", {"verdict": "ok"})
    assert q.is_reviewed("b")


def test_stratified_and_capacity():
    s = stratified_sample([(i, i / 10) for i in range(10)], [0.3, 0.6], 2)
    assert len(s) == 6
    c = capacity.scenario()
    assert c["label"] == "target, not achieved"
    assert c["frames_per_day"] == pytest.approx(142857.143)
    assert c["frames_per_person_per_day"] == pytest.approx(14285.714)
    b = capacity.benchmark_throughput(20000)
    assert b["frames"] == 20000 and b["frames_per_sec"] > 0
    print("THROUGHPUT", b)
