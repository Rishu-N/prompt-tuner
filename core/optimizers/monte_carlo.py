"""
Monte Carlo optimizer — generates N candidate prompt modifications in parallel
(high temperature), evaluates each, and picks the best.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from core.models import ModelClient
from core.prompt import Prompt
from core.test_case import TestCase
from core.evaluator import EvalResult
from core.optimizer import OptimizationResult, IterationLog
from core.evaluators.llm_judge import LLMJudgeEvaluator
from utils.registry import optimizer_registry

_PROPOSE_SYSTEM = """\
You are a prompt engineer. Rewrite ONLY the MUTABLE_SECTION below to fix the \
failure described. Be creative and make targeted sentence-level changes.

Respond ONLY with the updated mutable text (no explanations, no JSON, no fences).
"""


class MonteCarloOptimizer:
    def __init__(
        self,
        n_samples: int = 5,
        temperature: float = 0.9,
    ):
        self.n_samples = n_samples
        self.temperature = temperature

    def optimize(
        self,
        prompt: Prompt,
        test_case: TestCase,
        eval_result: EvalResult,
        target: ModelClient,
        supervisor: ModelClient,
        max_iterations: int = 3,
        log_callback: Callable[[str], None] | None = None,
    ) -> OptimizationResult:
        failure_feedback = eval_result.feedback + "\n" + eval_result.reasoning
        iteration_logs: list[IterationLog] = []
        evaluator = LLMJudgeEvaluator()

        if log_callback:
            log_callback(f"  [MonteCarlo] generating {self.n_samples} candidates...")

        user_msg = (
            f"FULL_PROMPT:\n{prompt.render()}\n\n"
            f"MUTABLE_SECTION:\n{prompt.render_mutable_only()}\n\n"
            f"USER_INPUT:\n{test_case.input_text}\n\n"
            f"EXPECTED_OUTPUT:\n{test_case.expected_output}\n\n"
            f"FAILURE_FEEDBACK:\n{failure_feedback}"
        )
        messages = [
            {"role": "system", "content": _PROPOSE_SYSTEM},
            {"role": "user", "content": user_msg},
        ]

        # Generate N candidates in parallel
        candidates: list[str] = []
        with ThreadPoolExecutor(max_workers=min(self.n_samples, 5)) as pool:
            futures = [
                pool.submit(target.chat, messages, self.temperature)
                for _ in range(self.n_samples)
            ]
            for f in as_completed(futures):
                try:
                    candidates.append(f.result())
                except Exception:
                    pass

        if not candidates:
            return OptimizationResult(
                updated_prompt=prompt.clone(),
                iterations=[],
                converged=False,
            )

        if log_callback:
            log_callback(f"  [MonteCarlo] evaluating {len(candidates)} candidates...")

        # Evaluate each candidate
        best_candidate = candidates[0]
        best_passed = False

        for idx, candidate in enumerate(candidates):
            candidate_prompt = prompt.apply_mutable_update(candidate)
            test_messages = [
                {"role": "system", "content": candidate_prompt.render()},
                {"role": "user", "content": test_case.input_text},
            ]
            output = target.chat(test_messages)
            result = evaluator.evaluate(output, test_case.expected_output, supervisor=supervisor)

            iteration_logs.append(IterationLog(
                iteration=idx + 1,
                proposed_mutable_text=candidate,
                supervisor_feedback=result.feedback,
                approved=result.passed,
            ))

            if result.passed and not best_passed:
                best_candidate = candidate
                best_passed = True
                if log_callback:
                    log_callback(f"  [MonteCarlo] candidate {idx+1} PASSED")
                break

            if log_callback:
                log_callback(f"  [MonteCarlo] candidate {idx+1} failed")

        updated = prompt.apply_mutable_update(best_candidate)
        return OptimizationResult(
            updated_prompt=updated,
            iterations=iteration_logs,
            converged=best_passed,
        )


optimizer_registry.register("monte_carlo", MonteCarloOptimizer)
