"""Majority-vote evaluator — runs multiple evaluators and decides by vote."""
from __future__ import annotations

from typing import Literal

from core.models import ModelClient
from core.evaluator import EvalResult
from core.interfaces.evaluator import EvaluatorProtocol
from utils.registry import evaluator_registry


class CompositeEvaluator:
    def __init__(
        self,
        evaluators: list[EvaluatorProtocol],
        strategy: Literal["any_pass", "all_pass", "majority"] = "majority",
    ):
        self.evaluators = evaluators
        self.strategy = strategy

    def evaluate(
        self,
        target_output: str,
        expected_output: str,
        *,
        supervisor: ModelClient | None = None,
    ) -> EvalResult:
        results = [
            e.evaluate(target_output, expected_output, supervisor=supervisor)
            for e in self.evaluators
        ]

        passes = [r for r in results if r.passed]
        fails = [r for r in results if not r.passed]

        if self.strategy == "any_pass":
            passed = len(passes) > 0
        elif self.strategy == "all_pass":
            passed = len(fails) == 0
        else:  # majority
            passed = len(passes) > len(fails)

        if passed:
            return EvalResult(
                passed=True,
                feedback="",
                reasoning=f"Composite ({self.strategy}): {len(passes)}/{len(results)} passed.",
            )

        all_feedback = "; ".join(r.feedback for r in fails if r.feedback)
        all_reasoning = "; ".join(r.reasoning for r in fails if r.reasoning)
        return EvalResult(
            passed=False,
            feedback=all_feedback or "Composite evaluation failed.",
            reasoning=all_reasoning,
        )


evaluator_registry.register("composite", CompositeEvaluator)
