"""
Example 1 — Tone Fixer
======================
Scenario:
  A customer support bot prompt originally responds too formally.
  We want it to respond in a friendly, casual tone.

The immutable section defines the bot's role (cannot be changed).
The mutable section defines tone guidance (optimizer can refine it).

Run:
  python examples/example_tone_fixer.py
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
TARGET_MODEL  = "gpt-4o-mini"
SUPERVISOR_MODEL = "gpt-4o-mini"

# ── Prompt ─────────────────────────────────────────────────────────────
# Immutable: the bot's core identity
# Mutable:   tone / style instructions (optimizer will refine these)
prompt = Prompt([
    PromptSection(
        text="You are a customer support agent for a software company called Acme.",
        mutable=False,
    ),
    PromptSection(
        text=(
            "Respond in a professional and formal manner. "
            "Use complete sentences and avoid contractions."
        ),
        mutable=True,
    ),
])

# ── Test cases ──────────────────────────────────────────────────────────
# We want friendly, casual replies — not stiff corporate speak.
test_cases = [
    TestCase(
        name="greeting",
        input_text="Hey! I can't log in to my account.",
        expected_output=(
            "A casual, friendly reply that acknowledges the issue and "
            "offers to help, using contractions and a warm tone."
        ),
    ),
    TestCase(
        name="refund_request",
        input_text="I want a refund please",
        expected_output=(
            "A warm, empathetic reply that starts a refund process in a "
            "relaxed, conversational tone — not corporate speak."
        ),
    ),
    TestCase(
        name="thanks",
        input_text="Thanks, got it working!",
        expected_output=(
            "A short, cheerful response celebrating with the user — "
            "feels human and friendly, not stiff."
        ),
    ),
]

# ── Run ─────────────────────────────────────────────────────────────────
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
