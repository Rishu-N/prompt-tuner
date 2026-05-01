"""
Optimizer module — backward-compatible interface.

The IterationLog and OptimizationResult dataclasses live here (canonical location).
The optimize() function delegates to the active optimizer strategy.
"""
from dataclasses import dataclass

from .models import ModelClient
from .prompt import Prompt
from .test_case import TestCase
from .evaluator import EvalResult


@dataclass
class IterationLog:
    iteration: int
    proposed_mutable_text: str
    supervisor_feedback: str
    approved: bool


@dataclass
class OptimizationResult:
    updated_prompt: Prompt
    iterations: list[IterationLog]
    converged: bool  # True if supervisor approved within max_iterations


def optimize(
    prompt: Prompt,
    test_case: TestCase,
    eval_result: EvalResult,
    target: ModelClient,
    supervisor: ModelClient,
    max_iterations: int = 3,
    log_callback=None,
) -> OptimizationResult:
    """
    Backward-compatible optimize() — delegates to DiscussionLoopOptimizer.
    For strategy-aware optimization, use the registry via the runner.
    """
    from .optimizers.discussion_loop import DiscussionLoopOptimizer
    return DiscussionLoopOptimizer().optimize(
        prompt, test_case, eval_result, target, supervisor,
        max_iterations=max_iterations, log_callback=log_callback,
    )
