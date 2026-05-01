"""Multi-model evaluation — run the same prompt across N target models."""
from __future__ import annotations

from dataclasses import dataclass, field

from .models import ModelConfig, ModelClient
from .prompt import Prompt
from .test_case import TestCase
from .evaluator import evaluate
from .runner import TestResult


@dataclass
class MultiModelResult:
    model_id: str
    test_results: list[TestResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if not self.test_results:
            return 0.0
        return sum(1 for r in self.test_results if r.passed) / len(self.test_results)


def run_multi_model_eval(
    prompt: Prompt,
    test_cases: list[TestCase],
    target_configs: list[ModelConfig],
    supervisor: ModelClient,
) -> list[MultiModelResult]:
    results = []
    for cfg in target_configs:
        target = ModelClient(cfg)
        test_results = []
        for tc in test_cases:
            messages = [
                {"role": "system", "content": prompt.render()},
                {"role": "user", "content": tc.input_text},
            ]
            actual = target.chat(messages)
            eval_result = evaluate(actual, tc.expected_output, supervisor)
            test_results.append(TestResult(
                test_name=tc.name,
                input_text=tc.input_text,
                expected_output=tc.expected_output,
                actual_output=actual,
                eval_result=eval_result,
            ))
        results.append(MultiModelResult(model_id=cfg.model_id, test_results=test_results))
    return results
