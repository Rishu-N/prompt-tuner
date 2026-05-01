"""LRU response cache for deterministic LLM calls."""
from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict

log = logging.getLogger(__name__)


class ResponseCache:
    def __init__(self, max_size: int = 256, cache_all_temperatures: bool = False):
        self._max_size = max_size
        self._cache_all = cache_all_temperatures
        self._store: OrderedDict[str, str] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def _cache_key(self, model_id: str, messages: list[dict], temperature: float) -> str:
        raw = json.dumps({"model": model_id, "messages": messages, "temperature": temperature}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def should_cache(self, temperature: float) -> bool:
        return self._cache_all or temperature == 0.0

    def get(self, model_id: str, messages: list[dict], temperature: float) -> str | None:
        if not self.should_cache(temperature):
            return None
        key = self._cache_key(model_id, messages, temperature)
        if key in self._store:
            self._hits += 1
            self._store.move_to_end(key)
            log.debug("Cache HIT (hits=%d)", self._hits)
            return self._store[key]
        self._misses += 1
        return None

    def put(self, model_id: str, messages: list[dict], temperature: float, response: str) -> None:
        if not self.should_cache(temperature):
            return
        key = self._cache_key(model_id, messages, temperature)
        self._store[key] = response
        self._store.move_to_end(key)
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)

    def stats(self) -> dict:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._store),
            "max_size": self._max_size,
        }
