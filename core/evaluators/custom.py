"""Wraps a user-provided callable as an evaluator."""
from __future__ import annotations

from typing import Callable

from core.evaluator import EvalResult
from utils.registry import evaluator_registry


class CallableEvaluator:
    def __init__(self, fn: Callable[[str, str], EvalResult]):
        self._fn = fn

    def evaluate(self, target_output: str, expected_output: str, **_kw) -> EvalResult:
        return self._fn(target_output, expected_output)


evaluator_registry.register("custom", CallableEvaluator)
