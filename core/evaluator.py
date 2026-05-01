"""
Evaluator module — backward-compatible interface.

The EvalResult dataclass lives here (canonical location).
The evaluate() function delegates to the active evaluator strategy.
"""
from dataclasses import dataclass

from .models import ModelClient


@dataclass
class EvalResult:
    passed: bool
    feedback: str       # what went wrong (empty string when passed)
    reasoning: str      # why it went wrong / what the model did differently

    def to_dict(self) -> dict:
        return {"passed": self.passed, "feedback": self.feedback, "reasoning": self.reasoning}


def evaluate(
    target_output: str,
    expected_output: str,
    supervisor: ModelClient,
) -> EvalResult:
    """
    Backward-compatible evaluate() — delegates to LLMJudgeEvaluator.
    For strategy-aware evaluation, use the registry via the runner.
    """
    from .evaluators.llm_judge import LLMJudgeEvaluator
    return LLMJudgeEvaluator().evaluate(
        target_output, expected_output, supervisor=supervisor,
    )
