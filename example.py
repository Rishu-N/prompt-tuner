"""
example.py — Plug-and-play prompt optimization.

This single file is the minimum you need to use the prompt-optimizer library
from another project. Two ways to use it:

  A) Run it as-is from this repo:
        python3 example.py

  B) Drop it into a different project:
        1. Copy the `core/`, `config/`, `reliability/`, `storage/`, `utils/`
           folders from this repo alongside this script (or install this repo
           as a package — there's no setup.py yet, so copying is easiest).
        2. Put your OPENAI_API_KEY in a `.env` file in the same directory.
        3. Edit the PROMPT and TEST_CASES at the bottom of this file.
        4. Run: python3 example.py

The `optimize_prompt(...)` function does all the work. Edit nothing above it
unless you want to change the model or endpoint defaults.
"""
from __future__ import annotations

import os
import sys
from typing import Callable, Optional

# Make sibling `core/` importable when this file is dropped into a project root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional — env vars from the shell work too

from core.models import ModelConfig, ModelClient
from core.prompt import Prompt, PromptSection
from core.test_case import TestCase
from core.runner import run_optimization, OptimizationHistory


# -----------------------------------------------------------------------------
# Defaults — override per-call or via env vars.
# -----------------------------------------------------------------------------
DEFAULT_MODEL = os.environ.get("PROMPT_OPTIMIZER_MODEL", "gpt-4o-mini")
DEFAULT_BASE_URL = os.environ.get("PROMPT_OPTIMIZER_BASE_URL", "https://api.openai.com/v1")
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "")


def optimize_prompt(
    immutable_text: str,
    mutable_text: str,
    test_cases: list[tuple[str, str, str]],
    *,
    target_model: str = DEFAULT_MODEL,
    supervisor_model: str = DEFAULT_MODEL,
    api_key: str = DEFAULT_API_KEY,
    base_url: str = DEFAULT_BASE_URL,
    epochs: int = 3,
    max_iterations: int = 3,
    log: Optional[Callable[[str], None]] = print,
) -> OptimizationHistory:
    """Optimize a prompt against a set of test cases.

    Args:
        immutable_text:  The fixed part of the prompt (role, task) — never edited.
        mutable_text:    The part the optimizer is allowed to rewrite.
        test_cases:      A list of (name, input_text, expected_output) tuples.
                         `expected_output` is a natural-language description of
                         what a correct answer looks like — it is judged
                         semantically by an LLM, not compared as a literal string.
        target_model:    The model whose prompt is being optimized.
        supervisor_model:The model judging outputs and reviewing proposed edits.
        api_key:         OpenAI-compatible API key. Defaults to $OPENAI_API_KEY.
        base_url:        OpenAI-compatible endpoint.
        epochs:          How many times to sweep through all test cases.
        max_iterations:  Per-test optimization attempts before giving up.
        log:             Where each runner log line goes. Default: print().
                         Pass `log=None` to suppress all output.

    Returns:
        OptimizationHistory — `.final_prompt`, `.pass_rates()`, `.epoch_results`
    """
    if not api_key:
        raise RuntimeError(
            "No API key. Set OPENAI_API_KEY in your environment or pass "
            "api_key=... explicitly."
        )

    prompt = Prompt([
        PromptSection(text=immutable_text, mutable=False),
        PromptSection(text=mutable_text, mutable=True),
    ])
    cases = [TestCase(name=n, input_text=i, expected_output=e) for n, i, e in test_cases]

    target = ModelClient(ModelConfig(model_id=target_model, api_key=api_key, base_url=base_url))
    supervisor = ModelClient(ModelConfig(model_id=supervisor_model, api_key=api_key, base_url=base_url))

    return run_optimization(
        prompt=prompt,
        test_cases=cases,
        target=target,
        supervisor=supervisor,
        epochs=epochs,
        max_iterations=max_iterations,
        log_callback=(log or (lambda _: None)),
    )


# =============================================================================
# EDIT BELOW — your prompt and test cases.
# =============================================================================

IMMUTABLE = """\
You are a customer support agent for a SaaS product called Acme.
You respond to user messages in chat.\
"""

MUTABLE = """\
Reply to the user.\
"""

TEST_CASES = [
    # (test_name, user_message, what_a_correct_reply_looks_like)
    (
        "greeting",
        "Hi! I just signed up — what do I do first?",
        "A warm, friendly welcome that points to a clear next step "
        "(e.g. completing onboarding or visiting the dashboard). Should use "
        "contractions and feel human, not corporate.",
    ),
    (
        "bug_report",
        "The export button is broken, nothing happens when I click it.",
        "An empathetic acknowledgement of the bug, a request for one or two "
        "specific details that would help diagnose (browser, what they were "
        "exporting), and a commitment to follow up. Avoids blame and avoids "
        "asking the user to re-do work they've already done.",
    ),
    (
        "pricing_question",
        "How much does the Pro plan cost?",
        "A direct answer that either states the price or, if unknown, points "
        "the user to the pricing page with a working URL pattern (e.g. "
        "'visit /pricing'). Does NOT make up a number.",
    ),
]


if __name__ == "__main__":
    print("=" * 60)
    print("INITIAL PROMPT")
    print("=" * 60)
    print(f"[IMMUTABLE]\n{IMMUTABLE}\n\n[MUTABLE]\n{MUTABLE}\n")

    history = optimize_prompt(
        immutable_text=IMMUTABLE,
        mutable_text=MUTABLE,
        test_cases=TEST_CASES,
        epochs=3,
        max_iterations=3,
    )

    print()
    print("=" * 60)
    print("FINAL PROMPT")
    print("=" * 60)
    print(history.final_prompt.render_annotated())
    print()
    print("Pass rates per epoch:", [f"{r:.0%}" for r in history.pass_rates()])
