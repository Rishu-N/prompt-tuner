"""
Example 2 — JSON Output Enforcer
=================================
Scenario:
  An extraction prompt that should always return a JSON object
  with keys "name", "email", and "issue".  The base prompt
  is missing clear JSON formatting instructions — the optimizer
  adds them.

The immutable section defines the task.
The mutable section (initially weak) will be strengthened by the optimizer.

Run:
  python examples/example_json_output.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

from core.models import ModelConfig, ModelClient
from core.prompt import Prompt, PromptSection
from core.test_case import TestCase
from core.runner import run_optimization

API_KEY = os.environ["OPENAI_API_KEY"]
BASE_URL = "https://api.openai.com/v1"
TARGET_MODEL     = "gpt-4o-mini"
SUPERVISOR_MODEL = "gpt-4o-mini"

# ── Prompt ──────────────────────────────────────────────────────────────
prompt = Prompt([
    PromptSection(
        text="You extract contact and issue information from customer messages.",
        mutable=False,
    ),
    PromptSection(
        text="Give the name, email, and issue.",  # deliberately weak — optimizer will fix this
        mutable=True,
    ),
])

# ── Test cases ───────────────────────────────────────────────────────────
test_cases = [
    TestCase(
        name="basic_extraction",
        input_text=(
            "Hi, I'm Sarah (sarah@example.com) and my account "
            "keeps getting locked out every morning."
        ),
        expected_output=(
            'A JSON object with exactly these keys: '
            '"name": "Sarah", "email": "sarah@example.com", '
            '"issue": "Account keeps getting locked out every morning"'
        ),
    ),
    TestCase(
        name="missing_email",
        input_text=(
            "This is Tom. I can't upload files larger than 10MB."
        ),
        expected_output=(
            'A JSON object with "name": "Tom", "email": null, '
            '"issue": "Cannot upload files larger than 10MB"'
        ),
    ),
    TestCase(
        name="verbose_message",
        input_text=(
            "Good morning, my name is Priya Sharma, reach me at priya@corp.io. "
            "The dashboard is showing wrong revenue figures for Q1."
        ),
        expected_output=(
            'A JSON object with "name": "Priya Sharma", '
            '"email": "priya@corp.io", '
            '"issue": "Dashboard showing wrong revenue figures for Q1"'
        ),
    ),
]

# ── Run ──────────────────────────────────────────────────────────────────
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
