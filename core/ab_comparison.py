"""A/B prompt comparison — run two prompts against the same test suite."""
from __future__ import annotations

from dataclasses import dataclass, field

from .models import ModelClient
from .prompt import Prompt
from .test_case import TestCase
from .evaluator import evaluate, EvalResult
from .runner import TestResult


@dataclass
class ABComparisonResult:
    results_a: list[TestResult] = field(default_factory=list)
    results_b: list[TestResult] = field(default_factory=list)

    @property
    def pass_rate_a(self) -> float:
        if not self.results_a:
            return 0.0
        return sum(1 for r in self.results_a if r.passed) / len(self.results_a)

    @property
    def pass_rate_b(self) -> float:
        if not self.results_b:
            return 0.0
        return sum(1 for r in self.results_b if r.passed) / len(self.results_b)

    @property
    def winner(self) -> str:
        if self.pass_rate_a > self.pass_rate_b:
            return "A"
        elif self.pass_rate_b > self.pass_rate_a:
            return "B"
        return "tie"


def _run_prompt(
    prompt: Prompt,
    test_cases: list[TestCase],
    target: ModelClient,
    supervisor: ModelClient,
) -> list[TestResult]:
    results = []
    for tc in test_cases:
        messages = [
            {"role": "system", "content": prompt.render()},
            {"role": "user", "content": tc.input_text},
        ]
        actual = target.chat(messages)
        eval_result = evaluate(actual, tc.expected_output, supervisor)
        results.append(TestResult(
            test_name=tc.name,
            input_text=tc.input_text,
            expected_output=tc.expected_output,
            actual_output=actual,
            eval_result=eval_result,
        ))
    return results


def run_ab_comparison(
    prompt_a: Prompt,
    prompt_b: Prompt,
    test_cases: list[TestCase],
    target: ModelClient,
    supervisor: ModelClient,
) -> ABComparisonResult:
    results_a = _run_prompt(prompt_a, test_cases, target, supervisor)
    results_b = _run_prompt(prompt_b, test_cases, target, supervisor)
    return ABComparisonResult(results_a=results_a, results_b=results_b)
