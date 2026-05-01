from __future__ import annotations

from typing import Callable, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from core.models import ModelClient
    from core.prompt import Prompt
    from core.test_case import TestCase
    from core.evaluator import EvalResult
    from core.optimizer import OptimizationResult


class OptimizerProtocol(Protocol):
    def optimize(
        self,
        prompt: Prompt,
        test_case: TestCase,
        eval_result: EvalResult,
        target: ModelClient,
        supervisor: ModelClient,
        max_iterations: int = 3,
        log_callback: Callable[[str], None] | None = None,
    ) -> OptimizationResult: ...
