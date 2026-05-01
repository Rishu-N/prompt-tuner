"""
Generic strategy registry.  Each domain (evaluators, optimizers, updaters)
gets its own singleton registry.  Strategy modules register themselves at
import time via `registry.register(key, factory)`.
"""
from __future__ import annotations

from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class StrategyRegistry(Generic[T]):
    def __init__(self, name: str):
        self.name = name
        self._factories: dict[str, Callable[..., T]] = {}

    def register(self, key: str, factory: Callable[..., T]) -> None:
        self._factories[key] = factory

    def get(self, key: str) -> Callable[..., T]:
        if key not in self._factories:
            available = ", ".join(sorted(self._factories)) or "(none)"
            raise KeyError(
                f"Unknown {self.name} strategy '{key}'. Available: {available}"
            )
        return self._factories[key]

    def list_keys(self) -> list[str]:
        return sorted(self._factories)

    def __contains__(self, key: str) -> bool:
        return key in self._factories

    def __repr__(self) -> str:
        return f"StrategyRegistry({self.name!r}, keys={self.list_keys()})"


# Singleton registries — populated by each strategy module on import
evaluator_registry: StrategyRegistry = StrategyRegistry("evaluator")
optimizer_registry: StrategyRegistry = StrategyRegistry("optimizer")
updater_registry: StrategyRegistry = StrategyRegistry("updater")
