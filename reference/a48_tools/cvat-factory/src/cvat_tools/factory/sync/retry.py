"""Bounded network retry helper for CVAT sync operations."""
from __future__ import annotations

import time
from typing import Callable, TypeVar

import requests

from .models import RetryPolicy, SyncError

T = TypeVar("T")


def is_transient_error(exc: BaseException, policy: RetryPolicy) -> bool:
    """Determine if an exception is transient and safe to retry."""
    if isinstance(exc, SyncError):
        return False
    if isinstance(exc, (requests.Timeout, requests.ConnectionError, TimeoutError, ConnectionError)):
        return True
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code in policy.retry_statuses
    msg = str(exc)
    for code in policy.retry_statuses:
        if f"-> {code}:" in msg or f"HTTP {code}" in msg:
            return True
    return False


def retry_call(
    fn: Callable[[], T],
    policy: RetryPolicy | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> T:
    """Execute fn() with bounded exponential backoff on transient network errors."""
    pol = policy or RetryPolicy()
    attempts = max(1, pol.max_attempts)
    delay = pol.base_delay
    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except BaseException as exc:
            last_exc = exc
            if attempt >= attempts or not is_transient_error(exc, pol):
                raise
            sleep_fn(min(delay, pol.max_delay))
            delay *= pol.backoff_factor
    assert last_exc is not None
    raise last_exc
