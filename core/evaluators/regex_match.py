"""Regex-based evaluator — expected_output is treated as a regex pattern."""
from __future__ import annotations

import re

from core.evaluator import EvalResult
from utils.registry import evaluator_registry


class RegexMatchEvaluator:
    def __init__(self, flags: int = re.IGNORECASE | re.DOTALL):
        self.flags = flags

    def evaluate(self, target_output: str, expected_output: str, **_kw) -> EvalResult:
        try:
            pattern = re.compile(expected_output, self.flags)
        except re.error as e:
            return EvalResult(
                passed=False,
                feedback=f"Invalid regex in expected_output: {e}",
                reasoning="Could not compile expected_output as regex.",
            )

        if pattern.search(target_output):
            return EvalResult(passed=True, feedback="", reasoning="Regex pattern matched.")
        return EvalResult(
            passed=False,
            feedback=f"Output did not match pattern: {expected_output}",
            reasoning=f"No match found in: {target_output[:200]}",
        )


evaluator_registry.register("regex", RegexMatchEvaluator)
