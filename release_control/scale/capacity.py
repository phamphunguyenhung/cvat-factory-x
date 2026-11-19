"""Capacity arithmetic (sanity check only) + measured throughput benchmark."""
from __future__ import annotations

import time

LABEL = "target, not achieved"


def scenario(people: int = 10, frames: int = 2_000_000, days: int = 14) -> dict:
    return {"label": LABEL, "people": people, "frames": frames, "days": days,
            "frames_per_day": round(frames / days, 3),
            "frames_per_person_per_day": round(frames / days / people, 3)}


def benchmark_throughput(n: int = 100_000, shard_size: int = 1000) -> dict:
    """Measured on N synthetic frames: risk scoring + sharding. Not a production claim."""
    from .risk import score_frame
    from .sharding import iter_shards
    feats = {"confidence": 0.7, "occlusion": 0.2, "small_far": 0.1, "classes": ["car"]}
    pol = {"class_severity": {"car": 0.5}}
    t0 = time.perf_counter()
    cnt = 0
    for _, frames in iter_shards(range(n), shard_size):
        for _f in frames:
            score_frame(feats, pol)
            cnt += 1
    dt = time.perf_counter() - t0
    return {"frames": cnt, "seconds": round(dt, 4), "frames_per_sec": round(cnt / dt, 1),
            "scope": "synthetic risk-scoring+sharding only; no human review"}
