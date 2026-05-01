"""Retry with exponential backoff for transient LLM API failures."""
from __future__ import annotations

import dataclasses
import functools
import logging
import random
import time

import openai

log = logging.getLogger(__name__)


@dataclasses.dataclass
class RetryPolicy:
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    retryable_exceptions: tuple = (
        openai.APITimeoutError,
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.InternalServerError,
    )


def retry_with_backoff(policy: RetryPolicy):
    """Decorator that retries on transient failures with exponential backoff + jitter."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, policy.max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except policy.retryable_exceptions as e:
                    last_exc = e
                    if attempt == policy.max_attempts:
                        break
                    delay = min(
                        policy.base_delay * (2 ** (attempt - 1)),
                        policy.max_delay,
                    )
                    jitter = random.uniform(0, delay * 0.3)
                    total_delay = delay + jitter
                    log.warning(
                        "Retry %d/%d after %.1fs (error: %s)",
                        attempt, policy.max_attempts, total_delay, e,
                    )
                    time.sleep(total_delay)
            raise last_exc
        return wrapper
    return decorator
