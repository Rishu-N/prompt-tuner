"""Exact / substring match evaluator — deterministic, no LLM needed."""
from __future__ import annotations

from core.evaluator import EvalResult
from utils.registry import evaluator_registry


class ExactMatchEvaluator:
    def __init__(self, case_sensitive: bool = False, strip_whitespace: bool = True):
        self.case_sensitive = case_sensitive
        self.strip_whitespace = strip_whitespace

    def evaluate(self, target_output: str, expected_output: str, **_kw) -> EvalResult:
        actual = target_output
        expected = expected_output
        if self.strip_whitespace:
            actual = actual.strip()
            expected = expected.strip()
        if not self.case_sensitive:
            actual = actual.lower()
            expected = expected.lower()

        if actual == expected:
            return EvalResult(passed=True, feedback="", reasoning="Exact match.")
        if expected in actual:
            return EvalResult(passed=True, feedback="", reasoning="Expected found as substring.")

        return EvalResult(
            passed=False,
            feedback=f"Output does not match expected. Got: {target_output[:200]}",
            reasoning="Neither exact nor substring match.",
        )


evaluator_registry.register("exact_match", ExactMatchEvaluator)
