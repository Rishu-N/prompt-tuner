"""
Walkthrough Example 1 — Regex Explainer (format/structure)
==========================================================
Scenario: The model is asked to explain regular expressions to a developer.
The initial mutable section just says "explain the regex" without dictating
any structure, so the model produces free-form prose that mixes the pattern,
its purpose, and examples into a single paragraph. The optimizer should
discover that it needs to enforce a structured output format with labeled
sections (Pattern / Plain-English meaning / Token breakdown / Example match)
so the explanation is consistently parseable.

Run:
  python examples/walkthrough_1.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

from core.models import ModelConfig, ModelClient
from core.prompt import Prompt, PromptSection
from core.test_case import TestCase
from core.runner import run_optimization

API_KEY = os.environ["OPENAI_API_KEY"]
BASE_URL = "https://api.openai.com/v1"
TARGET_MODEL = "gpt-4o-mini"
SUPERVISOR_MODEL = "gpt-4o-mini"

prompt = Prompt([
    PromptSection(
        text=(
            "You are a regex tutor. The user will paste a regular expression "
            "and you must help them understand it."
        ),
        mutable=False,
    ),
    PromptSection(
        text="Explain what the regex does.",
        mutable=True,
    ),
])

test_cases = [
    TestCase(
        name="email_like_pattern",
        input_text=r"^[\w.+-]+@[\w-]+\.[A-Za-z]{2,}$",
        expected_output=(
            "A response that uses four clearly labeled sections in this order: "
            "'Pattern:' (echoes the regex), 'Plain-English meaning:' (one-sentence "
            "summary, here something like matching an email-style string), "
            "'Token breakdown:' (a bullet list breaking down each metacharacter or "
            "character class), and 'Example match:' (a concrete string that the "
            "regex would match, e.g. user.name+tag@example.com). The four section "
            "headers must all be present."
        ),
    ),
    TestCase(
        name="us_phone_number",
        input_text=r"\(\d{3}\) \d{3}-\d{4}",
        expected_output=(
            "A response with exactly four labeled sections: 'Pattern:', "
            "'Plain-English meaning:' (it matches a US-style phone number with "
            "parentheses around the area code), 'Token breakdown:' (bulleted "
            "explanation of \\d{3}, the literal parentheses, the space, and the "
            "hyphen), and 'Example match:' (a sample string like '(415) 555-1212' "
            "that the regex would accept). All four headers must be present."
        ),
    ),
    TestCase(
        name="hex_color_code",
        input_text=r"^#(?:[0-9a-fA-F]{3}){1,2}$",
        expected_output=(
            "A response with the four labeled sections 'Pattern:', "
            "'Plain-English meaning:' (it matches a CSS hex color code, either "
            "3-digit or 6-digit form), 'Token breakdown:' (bullets covering the "
            "leading #, the character class [0-9a-fA-F], the {3} quantifier, and "
            "the {1,2} repetition of the group), and 'Example match:' (something "
            "like '#fff' or '#1A2B3C'). All four headers must appear."
        ),
    ),
]

target_cfg     = ModelConfig(model_id=TARGET_MODEL,     api_key=API_KEY, base_url=BASE_URL)
supervisor_cfg = ModelConfig(model_id=SUPERVISOR_MODEL, api_key=API_KEY, base_url=BASE_URL)
target     = ModelClient(target_cfg)
supervisor = ModelClient(supervisor_cfg)

print("=" * 60)
print("INITIAL PROMPT")
print("=" * 60)
print(prompt.render_annotated())
print()

def log(msg): print(msg)

history = run_optimization(
    prompt=prompt,
    test_cases=test_cases,
    target=target,
    supervisor=supervisor,
    epochs=3,
    max_iterations=3,
    log_callback=log,
)

print()
print("=" * 60)
print("FINAL PROMPT")
print("=" * 60)
print(history.final_prompt.render_annotated())
print()
print("Pass rates per epoch:", [f"{r:.0%}" for r in history.pass_rates()])
