"""Sharding + checkpointed, idempotent, bounded-retry batch processing."""
from __future__ import annotations

import json
import os
from typing import Callable, Iterable, Iterator


def iter_shards(frame_ids: Iterable, shard_size: int) -> Iterator[tuple[str, list]]:
    """Lazy generator: yields (shard_id, [frames]) without loading all in RAM."""
    buf: list = []
    n = 0
    for f in frame_ids:
        buf.append(f)
        if len(buf) >= shard_size:
            yield f"shard-{n:06d}", buf
            buf, n = [], n + 1
    if buf:
        yield f"shard-{n:06d}", buf


def _load(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {"done_shards": [], "dead_letter": [], "attempts": {}}


def _save(path, st):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(st, fh)
    os.replace(tmp, path)


def run_batches(frame_ids: Iterable, shard_size: int, process: Callable[[str, list], None],
                checkpoint_path: str, max_retries: int = 3) -> dict:
    """Process shards; skip those already done (resume/idempotent); retry up to
    max_retries per shard, then dead-letter. Checkpoint after every shard.
    BaseException (e.g. KeyboardInterrupt) propagates; state saved so far
    allows resume."""
    st = _load(checkpoint_path)
    done = set(st["done_shards"])
    dead = {d["shard_id"] for d in st["dead_letter"]}
    stats = {"processed": 0, "skipped": 0, "dead": 0}
    for sid, frames in iter_shards(frame_ids, shard_size):
        if sid in done or sid in dead:
            stats["skipped"] += 1
            continue
        err = None
        while st["attempts"].get(sid, 0) < max_retries:
            st["attempts"][sid] = st["attempts"].get(sid, 0) + 1
            try:
                process(sid, frames)
                err = None
                break
            except Exception as e:  # bounded retry
                err = repr(e)
        if err is None:
            st["done_shards"].append(sid)
            stats["processed"] += 1
        else:
            st["dead_letter"].append({"shard_id": sid, "error": err, "frames": len(frames)})
            stats["dead"] += 1
        _save(checkpoint_path, st)
    return stats
