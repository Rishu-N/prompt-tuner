"""SQLite-backed persistent run storage."""
from __future__ import annotations

import json
import sqlite3
import time
import threading


class SQLiteRunStorage:
    def __init__(self, db_path: str = "runs.db"):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    data TEXT,
                    created_at REAL,
                    updated_at REAL
                )
            """)
            conn.commit()
            conn.close()

    def _connect(self):
        return sqlite3.connect(self._db_path)

    def save_run(self, run_id: str, status: str, data: dict | None = None) -> None:
        now = time.time()
        data_json = json.dumps(data) if data else None
        with self._lock:
            conn = self._connect()
            conn.execute("""
                INSERT INTO runs (run_id, status, data, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    status = excluded.status,
                    data = excluded.data,
                    updated_at = excluded.updated_at
            """, (run_id, status, data_json, now, now))
            conn.commit()
            conn.close()

    def get_run(self, run_id: str) -> dict | None:
        with self._lock:
            conn = self._connect()
            cur = conn.execute("SELECT run_id, status, data, created_at, updated_at FROM runs WHERE run_id = ?", (run_id,))
            row = cur.fetchone()
            conn.close()
        if not row:
            return None
        return {
            "run_id": row[0],
            "status": row[1],
            "data": json.loads(row[2]) if row[2] else None,
            "created_at": row[3],
            "updated_at": row[4],
        }

    def list_runs(self) -> list[dict]:
        with self._lock:
            conn = self._connect()
            cur = conn.execute("SELECT run_id, status, created_at FROM runs ORDER BY created_at DESC")
            rows = cur.fetchall()
            conn.close()
        return [{"run_id": r[0], "status": r[1], "created_at": r[2]} for r in rows]

    def delete_run(self, run_id: str) -> None:
        with self._lock:
            conn = self._connect()
            conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            conn.commit()
            conn.close()

    def cleanup_expired(self, ttl_seconds: int) -> int:
        if ttl_seconds <= 0:
            return 0
        cutoff = time.time() - ttl_seconds
        with self._lock:
            conn = self._connect()
            cur = conn.execute("DELETE FROM runs WHERE updated_at < ?", (cutoff,))
            count = cur.rowcount
            conn.commit()
            conn.close()
            return count
