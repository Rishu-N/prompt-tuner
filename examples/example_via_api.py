"""
Example 3 — Run via REST API
=============================
This script submits the tone-fixer scenario to the running Prompt Optimizer
API server, polls for completion, and prints results.

Prerequisites:
  uvicorn api:app --reload --port 8000

Run:
  python examples/example_via_api.py
"""
import os, sys, time, json
import requests
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

API_BASE = "http://localhost:8000"
OPENAI_KEY = os.environ["OPENAI_API_KEY"]
MODEL = "gpt-4o-mini"
BASE_URL = "https://api.openai.com/v1"

# ── Build request payload ────────────────────────────────────────────────
payload = {
    "prompt_sections": [
        {
            "text": "You are a customer support agent for a software company called Acme.",
            "mutable": False,
        },
        {
            "text": (
                "Respond in a professional and formal manner. "
                "Use complete sentences and avoid contractions."
            ),
            "mutable": True,
        },
    ],
    "test_cases": [
        {
            "name": "greeting",
            "input_text": "Hey! I can't log in to my account.",
            "expected_output": (
                "A casual, friendly reply that acknowledges the issue and "
                "offers to help, using contractions and a warm tone."
            ),
        },
        {
            "name": "refund_request",
            "input_text": "I want a refund please",
            "expected_output": (
                "A warm, empathetic reply that starts a refund process in a "
                "relaxed, conversational tone."
            ),
        },
        {
            "name": "thanks",
            "input_text": "Thanks, got it working!",
            "expected_output": (
                "A short, cheerful, human response celebrating with the user."
            ),
        },
    ],
    "target_model": {
        "model_id": MODEL,
        "api_key": OPENAI_KEY,
        "base_url": BASE_URL,
        "name": "Target (gpt-4o-mini)",
    },
    "supervisor_model": {
        "model_id": MODEL,
        "api_key": OPENAI_KEY,
        "base_url": BASE_URL,
        "name": "Supervisor (gpt-4o-mini)",
    },
    "epochs": 3,
    "max_iterations": 3,
}

# ── Submit run ───────────────────────────────────────────────────────────
print("Submitting optimization run to API...")
resp = requests.post(f"{API_BASE}/optimize", json=payload)
resp.raise_for_status()
data = resp.json()
run_id = data["run_id"]
print(f"Run started — ID: {run_id}")
print()

# ── Poll until complete ──────────────────────────────────────────────────
while True:
    r = requests.get(f"{API_BASE}/history/{run_id}")
    r.raise_for_status()
    result = r.json()
    status = result["status"]
    if status == "running":
        print("  ... still running, polling again in 10s")
        time.sleep(10)
    elif status.startswith("error"):
        print(f"Run failed: {status}")
        sys.exit(1)
    else:
        break

# ── Print results ─────────────────────────────────────────────────────────
print("=" * 60)
print("OPTIMIZATION COMPLETE")
print("=" * 60)
print()

for epoch in result["epochs"]:
    print(
        f"Epoch {epoch['epoch']}: "
        f"{epoch['pass_count']}/{epoch['total_count']} passed "
        f"({epoch['pass_rate']:.0%})"
    )

print()
print("FINAL PROMPT SECTIONS:")
for section in result["final_prompt"]:
    label = "MUTABLE" if section["mutable"] else "IMMUTABLE"
    print(f"[{label}]\n{section['text']}\n")
