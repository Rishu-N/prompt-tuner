"""Discussion-loop optimizer — the original target↔supervisor strategy."""
from __future__ import annotations

import json
import re
from typing import Callable, Optional

from core.models import ModelClient
from core.prompt import Prompt
from core.test_case import TestCase
from core.evaluator import EvalResult
from core.optimizer import OptimizationResult, IterationLog
from utils.registry import optimizer_registry

_TARGET_PROPOSE_SYSTEM = """\
You are a prompt engineer. Your task is to improve the MUTABLE part of a system \
prompt so that it produces the expected output for a given user input.

You will be given:
- FULL_PROMPT: the current complete prompt (immutable + mutable sections combined)
- MUTABLE_SECTION: the specific text you are allowed to change
- USER_INPUT: the input that will be sent to the model
- EXPECTED_OUTPUT: what the model should produce
- FAILURE_FEEDBACK: what was wrong with the previous output and why

Instructions:
1. Propose an updated version of MUTABLE_SECTION only.
2. Make changes at the sentence level — add, remove, or rewrite individual sentences.
3. Keep the changes minimal; do not rewrite the whole section unless necessary.
4. Respond ONLY with the updated mutable text (no explanations, no JSON, no fences).
"""

_SUPERVISOR_REVIEW_SYSTEM = """\
You are a senior prompt engineer reviewing a proposed change to a system prompt.

You will be given:
- USER_INPUT: the input sent to the model
- EXPECTED_OUTPUT: what the model should produce
- ORIGINAL_MUTABLE: the mutable section before the proposed change
- PROPOSED_MUTABLE: the proposed updated mutable section
- FAILURE_FEEDBACK: what was wrong previously

Your job:
1. Decide if the proposed change is likely to fix the problem and produce the expected output.
2. If yes, respond with JSON: {"approved": true, "feedback": ""}
3. If no, respond with JSON: {"approved": false, "feedback": "<specific revision guidance>"}

Respond ONLY with the JSON object (no markdown fences).
"""


class DiscussionLoopOptimizer:
    def __init__(
        self,
        target_propose_prompt: str = _TARGET_PROPOSE_SYSTEM,
        supervisor_review_prompt: str = _SUPERVISOR_REVIEW_SYSTEM,
        target_temperature: float = 0.7,
        supervisor_temperature: float = 0.0,
    ):
        self.target_propose_prompt = target_propose_prompt
        self.supervisor_review_prompt = supervisor_review_prompt
        self.target_temperature = target_temperature
        self.supervisor_temperature = supervisor_temperature

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
        current_prompt = prompt.clone()
        iteration_logs: list[IterationLog] = []
        failure_feedback = eval_result.feedback + "\n" + eval_result.reasoning

        for i in range(1, max_iterations + 1):
            if log_callback:
                log_callback(f"  [Optimizer] iteration {i}/{max_iterations}")

            proposed_mutable = self._target_propose(
                full_prompt=current_prompt.render(),
                mutable_section=current_prompt.render_mutable_only(),
                user_input=test_case.input_text,
                expected_output=test_case.expected_output,
                failure_feedback=failure_feedback,
                target=target,
            )

            if log_callback:
                log_callback(f"  [Target] proposed: {proposed_mutable[:120]}...")

            approved, supervisor_feedback = self._supervisor_review(
                user_input=test_case.input_text,
                expected_output=test_case.expected_output,
                original_mutable=current_prompt.render_mutable_only(),
                proposed_mutable=proposed_mutable,
                failure_feedback=failure_feedback,
                supervisor=supervisor,
            )

            iteration_logs.append(IterationLog(
                iteration=i,
                proposed_mutable_text=proposed_mutable,
                supervisor_feedback=supervisor_feedback,
                approved=approved,
            ))

            if log_callback:
                status = "APPROVED" if approved else f"REJECTED — {supervisor_feedback[:80]}"
                log_callback(f"  [Supervisor] {status}")

            if approved:
                current_prompt = current_prompt.apply_mutable_update(proposed_mutable)
                return OptimizationResult(
                    updated_prompt=current_prompt,
                    iterations=iteration_logs,
                    converged=True,
                )

            failure_feedback = supervisor_feedback

        if iteration_logs:
            last_proposed = iteration_logs[-1].proposed_mutable_text
            current_prompt = current_prompt.apply_mutable_update(last_proposed)

        return OptimizationResult(
            updated_prompt=current_prompt,
            iterations=iteration_logs,
            converged=False,
        )

    def _target_propose(self, full_prompt, mutable_section, user_input,
                         expected_output, failure_feedback, target) -> str:
        user_msg = (
            f"FULL_PROMPT:\n{full_prompt}\n\n"
            f"MUTABLE_SECTION:\n{mutable_section}\n\n"
            f"USER_INPUT:\n{user_input}\n\n"
            f"EXPECTED_OUTPUT:\n{expected_output}\n\n"
            f"FAILURE_FEEDBACK:\n{failure_feedback}"
        )
        messages = [
            {"role": "system", "content": self.target_propose_prompt},
            {"role": "user", "content": user_msg},
        ]
        return target.chat(messages, temperature=self.target_temperature)

    def _supervisor_review(self, user_input, expected_output, original_mutable,
                            proposed_mutable, failure_feedback, supervisor) -> tuple[bool, str]:
        user_msg = (
            f"USER_INPUT:\n{user_input}\n\n"
            f"EXPECTED_OUTPUT:\n{expected_output}\n\n"
            f"ORIGINAL_MUTABLE:\n{original_mutable}\n\n"
            f"PROPOSED_MUTABLE:\n{proposed_mutable}\n\n"
            f"FAILURE_FEEDBACK:\n{failure_feedback}"
        )
        messages = [
            {"role": "system", "content": self.supervisor_review_prompt},
            {"role": "user", "content": user_msg},
        ]
        raw = supervisor.chat(messages, temperature=self.supervisor_temperature)
        return _parse_review(raw)


def _parse_review(raw: str) -> tuple[bool, str]:
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    try:
        data = json.loads(raw)
        return bool(data.get("approved", False)), data.get("feedback", "")
    except json.JSONDecodeError:
        approved = "true" in raw.lower()
        return approved, raw


optimizer_registry.register("discussion_loop", DiscussionLoopOptimizer)
