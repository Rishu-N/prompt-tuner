"""LLM-as-judge evaluator — the original evaluation strategy."""
from __future__ import annotations

import json
import re
from typing import Optional

from core.models import ModelClient
from core.evaluator import EvalResult
from utils.registry import evaluator_registry

DEFAULT_SYSTEM_PROMPT = """\
You are an impartial evaluator. Your job is to decide whether a language model's \
output satisfies an expected output.

You will be given:
- ACTUAL_OUTPUT: what the model produced
- EXPECTED_OUTPUT: what it should have produced

Rules:
1. Focus on semantic equivalence, not word-for-word matching.
2. If the actual output conveys the same meaning / achieves the same goal as the \
expected output, mark it as passed.
3. If it fails, provide clear, actionable feedback explaining WHAT is wrong and WHY \
the model may have produced the wrong answer.

Respond ONLY with a JSON object (no markdown fences) in this exact format:
{"passed": <true|false>, "feedback": "<short description of what is wrong, or empty>", \
"reasoning": "<explanation of why the model likely produced this output>"}
"""


class LLMJudgeEvaluator:
    def __init__(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.0,
    ):
        self.system_prompt = system_prompt
        self.temperature = temperature

    def evaluate(
        self,
        target_output: str,
        expected_output: str,
        *,
        supervisor: ModelClient | None = None,
    ) -> EvalResult:
        if supervisor is None:
            raise ValueError("LLMJudgeEvaluator requires a supervisor ModelClient")
        user_msg = (
            f"ACTUAL_OUTPUT:\n{target_output}\n\n"
            f"EXPECTED_OUTPUT:\n{expected_output}"
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg},
        ]
        raw = supervisor.chat(messages, temperature=self.temperature)
        return _parse_eval_result(raw)


def _parse_eval_result(raw: str) -> EvalResult:
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    try:
        data = json.loads(raw)
        return EvalResult(
            passed=bool(data.get("passed", False)),
            feedback=data.get("feedback", ""),
            reasoning=data.get("reasoning", ""),
        )
    except json.JSONDecodeError:
        first_line = raw.splitlines()[0].lower() if raw else ""
        passed = "true" in first_line
        return EvalResult(passed=passed, feedback=raw, reasoning="")


evaluator_registry.register("llm_judge", LLMJudgeEvaluator)
