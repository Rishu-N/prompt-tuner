from __future__ import annotations

from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from core.models import ModelClient
    from core.evaluator import EvalResult


class EvaluatorProtocol(Protocol):
    def evaluate(
        self,
        target_output: str,
        expected_output: str,
        *,
        supervisor: ModelClient | None = None,
    ) -> EvalResult: ...
