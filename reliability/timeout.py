"""Timeout wrapper for LLM API calls."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout


def with_timeout(fn, args=(), kwargs=None, timeout_seconds: float = 120.0):
    """
    Call fn(*args, **kwargs) with a deadline.
    Raises TimeoutError if the call exceeds timeout_seconds.
    """
    kwargs = kwargs or {}
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeout:
            raise TimeoutError(
                f"LLM API call timed out after {timeout_seconds}s"
            )
