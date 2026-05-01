"""Track token usage and estimate costs across LLM API calls."""
from __future__ import annotations

import dataclasses
import threading
import time

from config.settings import DEFAULT_SETTINGS


@dataclasses.dataclass
class UsageRecord:
    model_id: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    timestamp: float


class CostTracker:
    def __init__(self):
        self._records: list[UsageRecord] = []
        self._lock = threading.Lock()

    def record(self, model_id: str, usage: dict | None) -> None:
        if not usage:
            return
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

        prompt_cost_per_m, completion_cost_per_m = DEFAULT_SETTINGS.get_pricing(model_id)
        cost = (
            prompt_tokens * prompt_cost_per_m / 1_000_000
            + completion_tokens * completion_cost_per_m / 1_000_000
        )

        record = UsageRecord(
            model_id=model_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=cost,
            timestamp=time.time(),
        )
        with self._lock:
            self._records.append(record)

    def total_cost(self) -> float:
        with self._lock:
            return sum(r.estimated_cost_usd for r in self._records)

    def total_tokens(self) -> int:
        with self._lock:
            return sum(r.total_tokens for r in self._records)

    def summary(self) -> dict:
        with self._lock:
            return {
                "total_cost_usd": round(sum(r.estimated_cost_usd for r in self._records), 6),
                "total_tokens": sum(r.total_tokens for r in self._records),
                "total_prompt_tokens": sum(r.prompt_tokens for r in self._records),
                "total_completion_tokens": sum(r.completion_tokens for r in self._records),
                "call_count": len(self._records),
            }
