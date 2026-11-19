"""Chạy: python -m pytest -q source-tool/review-assign"""
import pytest

from assign_reviewers import assign

MEMBERS = ["Hùng", "Thịnh", "Việt", "Long", "Hưng"]


def test_reviewer_never_equals_annotator():
    jobs = [{"id": i, "annotator": m} for i, m in enumerate(MEMBERS * 2, start=1)]
    for r in assign(MEMBERS, jobs):
        assert r.reviewer != r.annotator


def test_load_is_balanced_when_one_job_each():
    jobs = [{"id": i, "annotator": m} for i, m in enumerate(MEMBERS, start=1)]
    rows = assign(MEMBERS, jobs)
    counts = {m: 0 for m in MEMBERS}
    for r in rows:
        counts[r.reviewer] += 1
    assert set(counts.values()) == {1}


def test_excluded_member_never_reviews():
    jobs = [{"id": i, "annotator": m} for i, m in enumerate(MEMBERS, start=1)]
    rows = assign(MEMBERS, jobs, exclude={"Hưng"})
    assert all(r.reviewer != "Hưng" for r in rows)
    assert all(r.reviewer != r.annotator for r in rows)


def test_person_with_two_jobs_gets_different_reviewers_when_possible():
    # Hưng gán 2 job (như Task 473: Job 1975 và Job 1623)
    jobs = [
        {"id": 1972, "annotator": "Hùng"},
        {"id": 1973, "annotator": "Thịnh"},
        {"id": 1974, "annotator": "Việt"},
        {"id": 1975, "annotator": "Hưng"},
        {"id": 1976, "annotator": "Long"},
        {"id": 1623, "annotator": "Hưng"},
    ]
    rows = {r.job_id: r for r in assign(MEMBERS, jobs)}
    assert rows[1975].reviewer != "Hưng"
    assert rows[1623].reviewer != "Hưng"


def test_only_one_candidate_raises():
    with pytest.raises(ValueError):
        assign(["A", "B"], [{"id": 1, "annotator": "A"}], exclude={"B"})


def test_unknown_annotator_raises():
    with pytest.raises(ValueError):
        assign(["A", "B"], [{"id": 1, "annotator": "C"}])
