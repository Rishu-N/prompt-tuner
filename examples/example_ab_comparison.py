"""
Example 4 — A/B Prompt Comparison
==================================
Compare a formal vs casual customer support prompt head-to-head.

Run:
  python examples/example_ab_comparison.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

from core.models import ModelConfig, ModelClient
from core.prompt import Prompt, PromptSection
from core.test_case import TestCase
from core.ab_comparison import run_ab_comparison

API_KEY = os.environ["OPENAI_API_KEY"]
BASE_URL = "https://api.openai.com/v1"
MODEL = "gpt-4o-mini"

# ── Two prompts to compare ──────────────────────────────────────────────
prompt_a = Prompt([
    PromptSection(text="You are a customer support agent for Acme.", mutable=False),
    PromptSection(text="Respond in a professional and formal manner.", mutable=True),
])

prompt_b = Prompt([
    PromptSection(text="You are a customer support agent for Acme.", mutable=False),
    PromptSection(text="Respond in a casual, friendly, warm tone. Use contractions.", mutable=True),
])

# ── Test cases ──────────────────────────────────────────────────────────
test_cases = [
    TestCase(name="greeting", input_text="Hey! I can't log in.",
             expected_output="A casual, friendly reply offering to help."),
    TestCase(name="thanks", input_text="Thanks, all fixed!",
             expected_output="A short, cheerful, human response."),
]

# ── Run ─────────────────────────────────────────────────────────────────
cfg = ModelConfig(model_id=MODEL, api_key=API_KEY, base_url=BASE_URL)
target = ModelClient(cfg)
supervisor = ModelClient(cfg)

result = run_ab_comparison(prompt_a, prompt_b, test_cases, target, supervisor)

print("=" * 60)
print("A/B COMPARISON RESULTS")
print("=" * 60)
print(f"\nPrompt A pass rate: {result.pass_rate_a:.0%}")
print(f"Prompt B pass rate: {result.pass_rate_b:.0%}")
print(f"Winner: {result.winner}")
print()

for label, results in [("A", result.results_a), ("B", result.results_b)]:
    print(f"--- Prompt {label} ---")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  {r.test_name}: {status}")
        if not r.passed:
            print(f"    Feedback: {r.eval_result.feedback[:100]}")
