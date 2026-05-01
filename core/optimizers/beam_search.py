"""
Beam search optimizer — maintains top-k candidate prompts, expands each
via the target model, evaluates all expansions, and keeps the best.
"""
from __future__ import annotations

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
failure described. Make targeted, sentence-level changes.

Respond ONLY with the updated mutable text (no explanations, no JSON, no fences).
"""


class BeamSearchOptimizer:
    def __init__(self, beam_width: int = 3, temperature: float = 0.8):
        self.beam_width = beam_width
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
        evaluator = LLMJudgeEvaluator()
        iteration_logs: list[IterationLog] = []
        failure_feedback = eval_result.feedback + "\n" + eval_result.reasoning

        # Start with a single beam (the current mutable text)
        beams: list[tuple[str, float]] = [(prompt.render_mutable_only(), 0.0)]

        for depth in range(1, max_iterations + 1):
            if log_callback:
                log_callback(f"  [BeamSearch] depth {depth}/{max_iterations}, beams={len(beams)}")

            candidates: list[tuple[str, float, bool]] = []  # (text, score, passed)

            for beam_text, _ in beams:
                # Generate a proposal from this beam
                user_msg = (
                    f"FULL_PROMPT:\n{prompt.render()}\n\n"
                    f"MUTABLE_SECTION:\n{beam_text}\n\n"
                    f"USER_INPUT:\n{test_case.input_text}\n\n"
                    f"EXPECTED_OUTPUT:\n{test_case.expected_output}\n\n"
                    f"FAILURE_FEEDBACK:\n{failure_feedback}"
                )
                messages = [
                    {"role": "system", "content": _PROPOSE_SYSTEM},
                    {"role": "user", "content": user_msg},
                ]
                proposed = target.chat(messages, temperature=self.temperature)

                # Evaluate the proposal
                candidate_prompt = prompt.apply_mutable_update(proposed)
                test_msgs = [
                    {"role": "system", "content": candidate_prompt.render()},
                    {"role": "user", "content": test_case.input_text},
                ]
                output = target.chat(test_msgs)
                result = evaluator.evaluate(output, test_case.expected_output, supervisor=supervisor)

                score = 1.0 if result.passed else 0.0
                candidates.append((proposed, score, result.passed))

                iteration_logs.append(IterationLog(
                    iteration=depth,
                    proposed_mutable_text=proposed,
                    supervisor_feedback=result.feedback,
                    approved=result.passed,
                ))

                # Early exit if we found a passing candidate
                if result.passed:
                    if log_callback:
                        log_callback(f"  [BeamSearch] found passing candidate at depth {depth}")
                    updated = prompt.apply_mutable_update(proposed)
                    return OptimizationResult(
                        updated_prompt=updated,
                        iterations=iteration_logs,
                        converged=True,
                    )

            # Keep top-k beams
            candidates.sort(key=lambda x: -x[1])
            beams = [(text, score) for text, score, _ in candidates[:self.beam_width]]

            if not beams:
                break

        # Return best beam even if not passing
        best_text = beams[0][0] if beams else prompt.render_mutable_only()
        updated = prompt.apply_mutable_update(best_text)
        return OptimizationResult(
            updated_prompt=updated,
            iterations=iteration_logs,
            converged=False,
        )


optimizer_registry.register("beam_search", BeamSearchOptimizer)
