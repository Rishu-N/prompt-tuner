"""In-memory run storage (default, same as original behavior)."""
from __future__ import annotations

import time
import threading


class InMemoryRunStorage:
    def __init__(self):
        self._store: dict[str, dict] = {}
        self._lock = threading.Lock()

    def save_run(self, run_id: str, status: str, data: dict | None = None) -> None:
        with self._lock:
            self._store[run_id] = {
                "run_id": run_id,
                "status": status,
                "data": data,
                "created_at": self._store.get(run_id, {}).get("created_at", time.time()),
                "updated_at": time.time(),
            }

    def get_run(self, run_id: str) -> dict | None:
        with self._lock:
            return self._store.get(run_id)

    def list_runs(self) -> list[dict]:
        with self._lock:
            return [
                {"run_id": v["run_id"], "status": v["status"], "created_at": v["created_at"]}
                for v in self._store.values()
            ]

    def delete_run(self, run_id: str) -> None:
        with self._lock:
            self._store.pop(run_id, None)

    def cleanup_expired(self, ttl_seconds: int) -> int:
        if ttl_seconds <= 0:
            return 0
        cutoff = time.time() - ttl_seconds
        with self._lock:
            expired = [k for k, v in self._store.items() if v["updated_at"] < cutoff]
            for k in expired:
                del self._store[k]
            return len(expired)
