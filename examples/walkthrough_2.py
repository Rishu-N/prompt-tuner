"""
Walkthrough Example 2 — Error-Message Rewriter (constraint/refusal)
====================================================================
Scenario: The model rewrites raw developer error messages (often with stack
traces, internal paths, and accusatory language like "you did X wrong") into
user-friendly messages suitable for an end-user-facing UI. The initial mutable
section is permissive — it just says "make the error friendly" — and the
model leaks stack-trace fragments, internal file paths, and second-person
blame ("you forgot to..."). The optimizer should discover hard rules: never
include stack traces, file paths, exception class names, or second-person
blame; always state the problem in neutral terms and suggest a next step.

Run:
  python examples/walkthrough_2.py
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
            "You rewrite raw developer error messages into a single short "
            "message that will be shown to a non-technical end user inside a "
            "web application."
        ),
        mutable=False,
    ),
    PromptSection(
        text="Make the error friendly and clear.",
        mutable=True,
    ),
])

test_cases = [
    TestCase(
        name="python_traceback",
        input_text=(
            "Traceback (most recent call last):\n"
            '  File "/srv/app/handlers/upload.py", line 142, in handle_upload\n'
            "    parsed = json.loads(body)\n"
            "json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)"
        ),
        expected_output=(
            "A single short user-facing message (1-2 sentences) that says the "
            "uploaded data could not be read because it was not valid, and "
            "suggests a concrete next step (such as checking the file format "
            "or trying again). The message must NOT contain any file path, "
            "any stack trace, the words 'Traceback', 'JSONDecodeError', "
            "'json.loads', or any second-person blame ('you did', 'you forgot', "
            "'you should have'). It should read naturally to a non-technical user."
        ),
    ),
    TestCase(
        name="database_constraint",
        input_text=(
            "psycopg2.errors.UniqueViolation: duplicate key value violates "
            'unique constraint "users_email_key"\n'
            "DETAIL:  Key (email)=(test@example.com) already exists.\n"
            "  File \"/srv/app/db/users.py\", line 88, in create_user"
        ),
        expected_output=(
            "A single short user-facing message saying the email address is "
            "already in use and suggesting the user sign in or use a different "
            "email. Must NOT contain 'psycopg2', 'UniqueViolation', any SQL "
            "constraint name like 'users_email_key', any file path, or any "
            "second-person blame phrasing."
        ),
    ),
    TestCase(
        name="network_timeout",
        input_text=(
            "requests.exceptions.ConnectTimeout: HTTPSConnectionPool("
            "host='api.payments.internal', port=443): "
            "Max retries exceeded with url: /v2/charge (Caused by "
            "ConnectTimeoutError(...))"
        ),
        expected_output=(
            "A short user-facing message explaining that the action could not "
            "be completed because a connection timed out, and suggesting the "
            "user try again in a moment. Must NOT mention the internal host "
            "name (api.payments.internal), the URL path /v2/charge, "
            "'requests.exceptions', 'HTTPSConnectionPool', or any second-person "
            "blame."
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
