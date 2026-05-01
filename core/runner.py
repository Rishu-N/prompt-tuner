"""
Epoch orchestrator: runs all test cases, triggers optimization on failure,
tracks history across epochs.  Supports pluggable evaluators/optimizers via
the strategy registry when FeatureFlags are provided.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional
import time

from .models import ModelClient
from .prompt import Prompt
from .test_case import TestCase
from .evaluator import evaluate as _default_evaluate, EvalResult
from .optimizer import optimize as _default_optimize, OptimizationResult


# -----------------------------------------------------------------------
# Result data structures
# -----------------------------------------------------------------------

@dataclass
class TestResult:
    test_name: str
    input_text: str
    expected_output: str
    actual_output: str
    eval_result: EvalResult
    optimization: Optional[OptimizationResult] = None

    @property
    def passed(self) -> bool:
        return self.eval_result.passed


@dataclass
class EpochResult:
    epoch: int
    prompt_before: Prompt
    prompt_after: Prompt
    test_results: list[TestResult] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.test_results if r.passed)

    @property
    def total_count(self) -> int:
        return len(self.test_results)

    @property
    def pass_rate(self) -> float:
        if not self.test_results:
            return 0.0
        return self.pass_count / self.total_count


@dataclass
class OptimizationHistory:
    epoch_results: list[EpochResult] = field(default_factory=list)

    @property
    def final_prompt(self) -> Optional[Prompt]:
        if self.epoch_results:
            return self.epoch_results[-1].prompt_after
        return None

    def pass_rates(self) -> list[float]:
        return [r.pass_rate for r in self.epoch_results]


# -----------------------------------------------------------------------
# Strategy resolution helpers
# -----------------------------------------------------------------------

def _resolve_strategies(flags):
    """
    Given FeatureFlags, return (evaluator_instance, optimizer_instance).
    If flags is None, returns None (caller should use legacy defaults).
    """
    if flags is None:
        return None, None

    # Lazy import to avoid circular imports and to only load registries when needed
    from utils.registry import evaluator_registry, optimizer_registry
    # Trigger registration of all strategy modules
    import core.evaluators  # noqa: F401
    import core.optimizers  # noqa: F401
    import core.updaters    # noqa: F401

    evaluator = evaluator_registry.get(flags.evaluator_strategy)()
    optimizer = optimizer_registry.get(flags.optimizer_strategy)()
    return evaluator, optimizer


# -----------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------

def run_epoch(
    prompt: Prompt,
    test_cases: list[TestCase],
    target: ModelClient,
    supervisor: ModelClient,
    max_iterations: int = 3,
    log_callback: Optional[Callable[[str], None]] = None,
    flags=None,
) -> EpochResult:
    """
    Run one epoch:
    - For each test case: execute prompt, evaluate, optimize on failure.
    - Returns an EpochResult with the prompt as it evolved through this epoch.
    """
    start = time.time()
    current_prompt = prompt.clone()
    test_results: list[TestResult] = []

    evaluator_inst, optimizer_inst = _resolve_strategies(flags)

    for tc in test_cases:
        if log_callback:
            log_callback(f"Running test: {tc.name}")

        # Run target model
        messages = [
            {"role": "system", "content": current_prompt.render()},
            {"role": "user", "content": tc.input_text},
        ]
        actual_output = target.chat(messages)

        # Evaluate
        if evaluator_inst is not None:
            eval_result = evaluator_inst.evaluate(
                actual_output, tc.expected_output, supervisor=supervisor,
            )
        else:
            eval_result = _default_evaluate(actual_output, tc.expected_output, supervisor)

        opt_result: Optional[OptimizationResult] = None

        if not eval_result.passed:
            if log_callback:
                log_callback(
                    f"  FAIL — {tc.name}: {eval_result.feedback[:100]}\n"
                    f"  Optimizing prompt..."
                )
            if optimizer_inst is not None:
                opt_result = optimizer_inst.optimize(
                    prompt=current_prompt,
                    test_case=tc,
                    eval_result=eval_result,
                    target=target,
                    supervisor=supervisor,
                    max_iterations=max_iterations,
                    log_callback=log_callback,
                )
            else:
                opt_result = _default_optimize(
                    prompt=current_prompt,
                    test_case=tc,
                    eval_result=eval_result,
                    target=target,
                    supervisor=supervisor,
                    max_iterations=max_iterations,
                    log_callback=log_callback,
                )
            current_prompt = opt_result.updated_prompt
            if log_callback:
                converged = "converged" if opt_result.converged else "did not converge"
                log_callback(f"  Optimization {converged} after {len(opt_result.iterations)} iteration(s).")
        else:
            if log_callback:
                log_callback(f"  PASS — {tc.name}")

        test_results.append(TestResult(
            test_name=tc.name,
            input_text=tc.input_text,
            expected_output=tc.expected_output,
            actual_output=actual_output,
            eval_result=eval_result,
            optimization=opt_result,
        ))

    return EpochResult(
        epoch=0,
        prompt_before=prompt,
        prompt_after=current_prompt,
        test_results=test_results,
        duration_seconds=time.time() - start,
    )


def _check_convergence(history: OptimizationHistory, patience: int) -> bool:
    """Return True if pass rate has not improved for `patience` consecutive epochs."""
    if len(history.epoch_results) < patience + 1:
        return False
    recent = history.epoch_results[-patience:]
    best_before = max(r.pass_rate for r in history.epoch_results[:-patience])
    return all(r.pass_rate <= best_before for r in recent)


def run_optimization(
    prompt: Prompt,
    test_cases: list[TestCase],
    target: ModelClient,
    supervisor: ModelClient,
    epochs: int = 5,
    max_iterations: int = 3,
    log_callback: Optional[Callable[[str], None]] = None,
    stop_flag: Optional[Callable[[], bool]] = None,
    flags=None,
) -> OptimizationHistory:
    """
    Run full prompt optimization over `epochs` epochs.

    flags: FeatureFlags instance (optional). When provided, uses the strategy
           registry for evaluator/optimizer selection and enables convergence detection.
    stop_flag(): if provided, returning True will halt the run early.
    """
    history = OptimizationHistory()
    current_prompt = prompt.clone()

    for epoch in range(1, epochs + 1):
        if stop_flag and stop_flag():
            if log_callback:
                log_callback("Stopped by user.")
            break

        if log_callback:
            log_callback(f"\n=== Epoch {epoch}/{epochs} ===")

        result = run_epoch(
            prompt=current_prompt,
            test_cases=test_cases,
            target=target,
            supervisor=supervisor,
            max_iterations=max_iterations,
            log_callback=log_callback,
            flags=flags,
        )
        result.epoch = epoch
        history.epoch_results.append(result)
        current_prompt = result.prompt_after

        if log_callback:
            log_callback(
                f"Epoch {epoch} done — pass rate: "
                f"{result.pass_count}/{result.total_count} "
                f"({result.pass_rate:.0%})"
            )

        # Early stopping: all tests pass
        if result.pass_count == result.total_count:
            if log_callback:
                log_callback("All tests passing — stopping early.")
            break

        # Convergence detection (feature toggle)
        if flags and getattr(flags, "convergence_detection_enabled", False):
            patience = getattr(flags, "convergence_patience", 3)
            if _check_convergence(history, patience):
                if log_callback:
                    log_callback(
                        f"Pass rate stagnated for {patience} epochs — stopping."
                    )
                break

    return history
