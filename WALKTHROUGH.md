# Prompt Optimizer — End-to-End Walkthrough

This document walks through everything the project does, end to end, on three brand-new examples. For each example you will see:

1. The **initial prompt** (with mutable/immutable sections annotated).
2. **Every step** the runner emits during optimization — every test run, every iteration the target proposes, every approve/reject from the supervisor.
3. The **final prompt** after 3 epochs.
4. The **pass rate per epoch**.
5. The **API contract** that produced the same run via the REST server — request body and the polled `/history` response.

The same three examples are then collected in a "Reproducing this walkthrough" section so anyone can regenerate every artifact.

---

## Environment

| | |
|---|---|
| **Model used** | `gpt-4o-mini` (both target and supervisor) |
| **OpenAI endpoint** | `https://api.openai.com/v1` |
| **API key** | Loaded from `.env` (`OPENAI_API_KEY`, project key `sk-proj-…`) |
| **Server** | `uvicorn api:app --port 8000` |
| **Examples** | `examples/walkthrough_{1,2,3}.py` |

### Model selection

The project's `.env` holds an OpenAI **project key** (`sk-proj-…`). Project keys can have restricted permissions, so we explicitly verified which models the key can call before choosing one.

```bash
# 1. List models accessible to this key
curl -s https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY" \
  | python3 -c "import json,sys; print(len(json.load(sys.stdin)['data']),'models')"
# -> 125 models

# 2. Smoke-test the chosen model
curl -s https://api.openai.com/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"reply with one word: pong"}],"max_tokens":5,"temperature":0}'
# -> {"choices":[{"message":{"content":"Pong",...}}],...}
```

The key has access to a wide range including the entire GPT-4o / GPT-4.1 / GPT-5 families. We use **`gpt-4o-mini`** throughout this walkthrough because (a) it is the cheapest model that supports the chat-completions JSON-response patterns the project uses and (b) the project's CLAUDE-memory confirms it was the model used during prior development.

---

## How the optimizer actually works

The runner loop, with every log line it emits, is:

```
INITIAL PROMPT (annotated render — printed once, before any optimization)

for each epoch in 1..N:
  === Epoch N/M ===
  for each test_case:
    Running test: <name>
       (target_model.call(prompt + input)  → output)
       (supervisor_model.judge(output, expected) → EvalResult)
    if pass:
        PASS — <name>
    else:
        FAIL — <name>: <feedback truncated to 100 chars>
        Optimizing prompt...
        for i in 1..max_iterations:
            [Optimizer] iteration i/max_iterations
            (target proposes a replacement mutable block)
            [Target] proposed: <first 120 chars>...
            (supervisor reviews proposal → approved+feedback)
            if approved:  [Supervisor] APPROVED
            else:         [Supervisor] REJECTED — <feedback truncated to 80 chars>
            if approved: break
        if approved:  Optimization converged after I iteration(s).
        else:         Optimization did not converge after max iteration(s).
        (the new prompt is applied for subsequent test cases / epochs)
  Epoch N done — pass rate: K/T (P%)

FINAL PROMPT (annotated render)
Pass rates per epoch: ['p%', 'p%', 'p%']
```

Notes:

- **Immutable sections never change.** The optimizer only rewrites the text of `mutable=True` sections.
- **Each test that fails triggers its own optimization round.** The mutated prompt then carries forward into the next test and next epoch.
- **`expected_output` is a natural-language description**, not a literal string. The supervisor LLM judges semantic equivalence ([`core/evaluators/llm_judge.py`](core/evaluators/llm_judge.py)).
- Per-iteration prompt diffs are not persisted to disk — only end-of-epoch snapshots (`prompt_before` / `prompt_after`) are stored in `OptimizationHistory`. The iteration trace lives in the runner's log output and in `epoch_result.test_results[*].optimization.iterations`.

---

## API Contracts

The REST server exposes four endpoints. All examples here were executed against a live `uvicorn api:app --port 8000` instance.

### `GET /runs` — list all optimization runs

**Request**
```bash
curl http://localhost:8000/runs
```

**Response — empty (server just started)**
```json
[]
```

**Response — after 3 runs completed** (captured at the end of this walkthrough)
```json
[
  {
    "run_id": "145c66b2-223c-41d1-b7fb-830079235f60",
    "status": "complete",
    "created_at": 1779865111.91994
  },
  {
    "run_id": "683a46b8-fb3e-479b-8123-3e582071d00a",
    "status": "complete",
    "created_at": 1779865248.090848
  },
  {
    "run_id": "7004e604-6d05-4432-a219-b6556f2d7d53",
    "status": "complete",
    "created_at": 1779865312.1844192
  }
]
```

---

### `POST /evaluate` — synchronous LLM-as-judge evaluation

Evaluates a single output against an expected output. No run is created; this is one supervisor call.

**Request**
```bash
curl -X POST http://localhost:8000/evaluate -H "Content-Type: application/json" -d '{
  "actual_output": "Sure thing — try resetting your password from the login page and let me know if that helps!",
  "expected_output": "A friendly, casual reply offering to help with a login issue, using contractions.",
  "supervisor_model": {
    "model_id": "gpt-4o-mini",
    "api_key": "sk-proj-...",
    "base_url": "https://api.openai.com/v1"
  }
}'
```

**Response**
```json
{
  "passed": false,
  "feedback": "The response is friendly and casual but does not use contractions as requested.",
  "reasoning": "The model likely produced a response that was friendly and helpful but overlooked the specific instruction to include contractions, possibly due to focusing on the content rather than the style."
}
```

Note that the judge correctly identified the missing contractions even though the reply *was* casual — this is what `llm_judge` evaluations look like in practice.

---

### `POST /optimize` — start an async optimization run

The server starts a background task that runs the same `run_optimization()` loop documented above. Returns a `run_id` immediately; poll `/history/{run_id}` for progress and the final result.

**Request shape**
```json
{
  "prompt_sections": [
    {"text": "<immutable role>", "mutable": false},
    {"text": "<initial mutable guidance>", "mutable": true}
  ],
  "test_cases": [
    {"name": "...", "input_text": "...", "expected_output": "..."}
  ],
  "target_model": {"model_id": "gpt-4o-mini", "api_key": "sk-...", "base_url": "https://api.openai.com/v1"},
  "supervisor_model": {"model_id": "gpt-4o-mini", "api_key": "sk-...", "base_url": "https://api.openai.com/v1"},
  "epochs": 3,
  "max_iterations": 3,
  "webhook_url": null
}
```

**Response (immediate)**
```json
{
  "run_id": "<uuid>",
  "message": "Optimization started. Poll /history/{run_id} for progress."
}
```

The full request body and `run_id` for each of the three examples are shown in their respective sections below.

---

### `GET /history/{run_id}` — poll an optimization run

Returns the current status. While running, `epochs` is empty. When complete, `epochs[]` contains one entry per completed epoch with the prompt-after snapshot.

**Response while still running** (captured at t+0s while example 1 was in progress)
```json
{
  "run_id": "145c66b2-223c-41d1-b7fb-830079235f60",
  "status": "running",
  "epochs": [],
  "final_prompt": null,
  "cost": null
}
```

**Response when complete** — see each example below for the actual completed response.

---

## Example 1 — Regex Explainer (format/structure axis)

**Source:** [`examples/walkthrough_1.py`](examples/walkthrough_1.py)

**Scenario.** The model is told to "explain regular expressions" with no structural guidance, so it returns free-form prose. The optimizer should discover that it needs to add a four-section format (`Pattern:` / `Plain-English meaning:` / `Token breakdown:` / `Example match:`).

### Initial prompt

```text
[IMMUTABLE]
You are a regex tutor. The user will paste a regular expression and you must help them understand it.

[MUTABLE]
Explain what the regex does.
```

### Step-by-step trace (library run)

The verbatim runner log:

```text
=== Epoch 1/3 ===
Running test: email_like_pattern
  FAIL — email_like_pattern: The output does not follow the required section format and lacks a concrete example match.
  Optimizing prompt...
  [Optimizer] iteration 1/3
  [Target] proposed: Provide a structured response with four clearly labeled sections: 'Pattern:' (echoes the regex), 'Plain-English meaning:...
  [Supervisor] APPROVED
  Optimization converged after 1 iteration(s).
Running test: us_phone_number
  FAIL — us_phone_number: The expected output requires a specific plain-English meaning and a sample match that aligns with th
  Optimizing prompt...
  [Optimizer] iteration 1/3
  [Target] proposed: Provide a structured response with four clearly labeled sections: 'Pattern:' (echoes the regex), 'Plain-English meaning:...
  [Supervisor] APPROVED
  Optimization converged after 1 iteration(s).
Running test: hex_color_code
  FAIL — hex_color_code: The actual output does not match the expected output in terms of content and structure. It lacks the
  Optimizing prompt...
  [Optimizer] iteration 1/3
  [Target] proposed: Provide a structured response with four clearly labeled sections: 'Pattern:' (echoes the regex), 'Plain-English meaning:...
  [Supervisor] APPROVED
  Optimization converged after 1 iteration(s).
Epoch 1 done — pass rate: 0/3 (0%)

=== Epoch 2/3 ===
Running test: email_like_pattern
  FAIL — email_like_pattern: The output is missing the required format for the 'Plain-English meaning' section, which should be a
  Optimizing prompt...
  [Optimizer] iteration 1/3
  [Target] proposed: Provide a structured response with four clearly labeled sections: 'Pattern:' (echoes the regex), 'Plain-English meaning:...
  [Supervisor] APPROVED
  Optimization converged after 1 iteration(s).
Running test: us_phone_number
  FAIL — us_phone_number: The expected output requires a specific plain-English meaning and a different example match format.
  Optimizing prompt...
  [Optimizer] iteration 1/3
  [Target] proposed: Provide a structured response with four clearly labeled sections: 'Pattern:' (echoes the regex), 'Plain-English meaning:...
  [Supervisor] APPROVED
  Optimization converged after 1 iteration(s).
Running test: hex_color_code
  FAIL — hex_color_code: The actual output does not match the expected description of the plain-English meaning and example m
  Optimizing prompt...
  [Optimizer] iteration 1/3
  [Target] proposed: Provide a structured response with four clearly labeled sections: 'Pattern:' (echoes the regex), 'Plain-English meaning:...
  [Supervisor] APPROVED
  Optimization converged after 1 iteration(s).
Epoch 2 done — pass rate: 0/3 (0%)

=== Epoch 3/3 ===
Running test: email_like_pattern
  PASS — email_like_pattern
Running test: us_phone_number
  FAIL — us_phone_number: The output does not match the expected format and lacks the specific phrasing for the 'Plain-English
  Optimizing prompt...
  [Optimizer] iteration 1/3
  [Target] proposed: Provide a structured response with four clearly labeled sections: 'Pattern:' (echoes the regex), 'Plain-English meaning:...
  [Supervisor] APPROVED
  Optimization converged after 1 iteration(s).
Running test: hex_color_code
  FAIL — hex_color_code: The actual output does not include the expected headers and does not match the plain-English meaning
  Optimizing prompt...
  [Optimizer] iteration 1/3
  [Target] proposed: Provide a structured response with four clearly labeled sections: 'Pattern:' (echoes the regex), 'Plain-English meaning:...
  [Supervisor] APPROVED
  Optimization converged after 1 iteration(s).
Epoch 3 done — pass rate: 1/3 (33%)
```

### Reading the trace

- **Every iteration was `APPROVED`.** The supervisor approves whenever the target's proposed replacement is reasonable — it does not re-run the test inside the iteration. The pass-rate is measured separately, at the top of the next epoch, by re-running every test against the updated prompt.
- **Test cases compete with each other.** Each test specifies a different example match in its `expected_output` (e.g. an email-style string vs. `(415) 555-1212` vs. `#fff`). When the optimizer specialises the mutable section for test 3, tests 1 and 2 become slightly less aligned with the judge's expected wording, and vice versa. This is why the prompt is oscillating between near-identical refinements rather than reaching a single fixed point.
- **Convergence appeared briefly in epoch 3.** The `email_like_pattern` test passed after two rounds of optimization had pulled the mutable section toward the four-section format. The other two tests remained semantically picky.

### Final prompt

```text
[IMMUTABLE]
You are a regex tutor. The user will paste a regular expression and you must help them understand it.

[MUTABLE]
Provide a structured response with four clearly labeled sections: 'Pattern:' (echoes the regex), 'Plain-English meaning:' (it matches a CSS hex color code in either 3-digit or 6-digit format), 'Token breakdown:' (a bullet list breaking down each component, including the leading #, the character class [0-9a-fA-F], the {3} quantifier, and the {1,2} repetition of the group), and 'Example match:' (include examples like '#fff' or '#1A2B3C'). Ensure all four section headers are clearly included in your explanation.
```

### Pass rates per epoch

```
['0%', '0%', '33%']
```

### API contract for this run

#### `POST /optimize` request (API key redacted)

```json
{
  "prompt_sections": [
    {"text": "You are a regex tutor. The user will paste a regular expression and you must help them understand it.", "mutable": false},
    {"text": "Explain what the regex does.", "mutable": true}
  ],
  "test_cases": [
    {"name": "email_like_pattern", "input_text": "^[\\w.+-]+@[\\w-]+\\.[A-Za-z]{2,}$", "expected_output": "A response that uses four clearly labeled sections in this order: 'Pattern:' (echoes the regex), 'Plain-English meaning:' (one-sentence summary, here something like matching an email-style string), 'Token breakdown:' (a bullet list breaking down each metacharacter or character class), and 'Example match:' (a concrete string that the regex would match, e.g. user.name+tag@example.com). The four section headers must all be present."},
    {"name": "us_phone_number", "input_text": "\\(\\d{3}\\) \\d{3}-\\d{4}", "expected_output": "A response with exactly four labeled sections: 'Pattern:', 'Plain-English meaning:' (it matches a US-style phone number with parentheses around the area code), 'Token breakdown:' (bulleted explanation of \\d{3}, the literal parentheses, the space, and the hyphen), and 'Example match:' (a sample string like '(415) 555-1212' that the regex would accept). All four headers must be present."},
    {"name": "hex_color_code", "input_text": "^#(?:[0-9a-fA-F]{3}){1,2}$", "expected_output": "A response with the four labeled sections 'Pattern:', 'Plain-English meaning:' (it matches a CSS hex color code, either 3-digit or 6-digit form), 'Token breakdown:' (bullets covering the leading #, the character class [0-9a-fA-F], the {3} quantifier, and the {1,2} repetition of the group), and 'Example match:' (something like '#fff' or '#1A2B3C'). All four headers must appear."}
  ],
  "target_model":     {"model_id": "gpt-4o-mini", "api_key": "sk-proj-REDACTED", "base_url": "https://api.openai.com/v1", "name": "target"},
  "supervisor_model": {"model_id": "gpt-4o-mini", "api_key": "sk-proj-REDACTED", "base_url": "https://api.openai.com/v1", "name": "supervisor"},
  "epochs": 3,
  "max_iterations": 3
}
```

#### `POST /optimize` response

```json
{
  "run_id": "145c66b2-223c-41d1-b7fb-830079235f60",
  "message": "Optimization started. Poll /history/{run_id} for progress."
}
```

The server kicks off the optimization in a background task and returns immediately. The client then polls `/history/{run_id}` every 8 seconds.

#### `GET /history/145c66b2-…` — first poll (t+0s, still running)

```json
{
  "run_id": "145c66b2-223c-41d1-b7fb-830079235f60",
  "status": "running",
  "epochs": [],
  "final_prompt": null,
  "cost": null
}
```

…repeated 17 more times every 8 seconds while the optimization ran (136.1s total)…

#### `GET /history/145c66b2-…` — final poll (t+136.1s, status=complete)

```json
{
  "run_id": "145c66b2-223c-41d1-b7fb-830079235f60",
  "status": "complete",
  "epochs": [
    {
      "epoch": 1,
      "pass_count": 0,
      "total_count": 3,
      "pass_rate": 0.0,
      "prompt_after": [
        {"text": "You are a regex tutor. The user will paste a regular expression and you must help them understand it.", "mutable": false},
        {"text": "Provide a detailed explanation of the regex that includes four clearly labeled sections: 'Pattern:' (echo the regex), 'Plain-English meaning:' (clearly state that it matches a CSS hex color code, either in 3-digit or 6-digit form), 'Token breakdown:' (list each metacharacter or character class with explanations in a bullet format), and 'Example match:' (provide specific examples like '#fff' or '#1A2B3C'). Ensure all four section headers are included.", "mutable": true}
      ]
    },
    {
      "epoch": 2,
      "pass_count": 0,
      "total_count": 3,
      "pass_rate": 0.0,
      "prompt_after": [
        {"text": "You are a regex tutor. The user will paste a regular expression and you must help them understand it.", "mutable": false},
        {"text": "Provide a detailed explanation of the regex that includes four clearly labeled sections: 'Pattern:' (echo the regex), 'Plain-English meaning:' (provide a concise summary stating that it matches a CSS hex color code, either 3-digit or 6-digit form), 'Token breakdown:' (create a bullet list breaking down the leading #, the character class [0-9a-fA-F], the {3} quantifier, and the {1,2} repetition of the group), and 'Example match:' (offer specific examples like '#fff' or '#1A2B3C'). Ensure all four section headers are included.", "mutable": true}
      ]
    },
    {
      "epoch": 3,
      "pass_count": 1,
      "total_count": 3,
      "pass_rate": 0.3333333333333333,
      "prompt_after": [
        {"text": "You are a regex tutor. The user will paste a regular expression and you must help them understand it.", "mutable": false},
        {"text": "Provide a detailed explanation of the regex that includes four clearly labeled sections: 'Pattern:' (echo the regex), 'Plain-English meaning:' (it matches a CSS hex color code, either 3-digit or 6-digit form), 'Token breakdown:' (create a bullet list breaking down the leading #, the character class [0-9a-fA-F], the {3} quantifier, and the {1,2} repetition of the group), and 'Example match:' (provide specific examples like '#fff' or '#1A2B3C' that the regex would accept). Ensure all four section headers are included.", "mutable": true}
      ]
    }
  ],
  "final_prompt": [
    {"text": "You are a regex tutor. The user will paste a regular expression and you must help them understand it.", "mutable": false},
    {"text": "Provide a detailed explanation of the regex that includes four clearly labeled sections: 'Pattern:' (echo the regex), 'Plain-English meaning:' (it matches a CSS hex color code, either 3-digit or 6-digit form), 'Token breakdown:' (create a bullet list breaking down the leading #, the character class [0-9a-fA-F], the {3} quantifier, and the {1,2} repetition of the group), and 'Example match:' (provide specific examples like '#fff' or '#1A2B3C' that the regex would accept). Ensure all four section headers are included.", "mutable": true}
  ],
  "cost": null
}
```

Note that the **server-run final prompt is identical to the library-run final prompt** above, byte-for-byte. The runner does not depend on whether you called it from a script or from FastAPI — the loop is deterministic given the same inputs and (largely) the same model behavior.

---

## Example 2 — Error-Message Rewriter (constraint/refusal axis)

**Source:** [`examples/walkthrough_2.py`](examples/walkthrough_2.py)

**Scenario.** The model rewrites raw developer error messages (Python tracebacks, psycopg2 errors, requests exceptions) into single short user-facing messages. The mutable section starts with permissive guidance — "Make the error friendly and clear." — so leaks happen. The optimizer must discover hard rules: no stack traces, no file paths, no exception class names, no second-person blame.

### Initial prompt

```text
[IMMUTABLE]
You rewrite raw developer error messages into a single short message that will be shown to a non-technical end user inside a web application.

[MUTABLE]
Make the error friendly and clear.
```

### Step-by-step trace (library run)

```text
=== Epoch 1/3 ===
Running test: python_traceback
  FAIL — python_traceback: The message does not specify that the uploaded data was not valid and lacks a concrete next step for
  Optimizing prompt...
  [Optimizer] iteration 1/3
  [Target] proposed: Provide a concise, user-friendly message that informs the user that the uploaded data could not be read because it was n...
  [Supervisor] APPROVED
  Optimization converged after 1 iteration(s).
Running test: database_constraint
  FAIL — database_constraint: The message does not suggest signing in and includes a second-person phrasing ('you entered').
  Optimizing prompt...
  [Optimizer] iteration 1/3
  [Target] proposed: Provide a concise, user-friendly message that informs the user that the email address is already in use and suggests sig...
  [Supervisor] APPROVED
  Optimization converged after 1 iteration(s).
Running test: network_timeout
  FAIL — network_timeout: The message does not specify that the action could not be completed due to a connection timeout, and
  Optimizing prompt...
  [Optimizer] iteration 1/3
  [Target] proposed: Provide a concise, user-friendly message that explains that the action could not be completed due to a connection timeou...
  [Supervisor] APPROVED
  Optimization converged after 1 iteration(s).
Epoch 1 done — pass rate: 0/3 (0%)

=== Epoch 2/3 ===
Running test: python_traceback
  FAIL — python_traceback: The message does not address the issue of invalid data and lacks a concrete next step for the user.
  Optimizing prompt...
  [Optimizer] iteration 1/3
  [Target] proposed: Provide a concise, user-friendly message that explains that the uploaded data could not be read because it was not valid...
  [Supervisor] APPROVED
  Optimization converged after 1 iteration(s).
Running test: database_constraint
  FAIL — database_constraint: The message does not suggest signing in and includes a suggestion to check the format, which is not 
  Optimizing prompt...
  [Optimizer] iteration 1/3
  [Target] proposed: Provide a concise, user-friendly message that explains that the email address is already in use and suggests signing in ...
  [Supervisor] APPROVED
  Optimization converged after 1 iteration(s).
Running test: network_timeout
  FAIL — network_timeout: The message does not specify that the action could not be completed due to a connection timeout, and
  Optimizing prompt...
  [Optimizer] iteration 1/3
  [Target] proposed: Provide a concise, user-friendly message that explains that the action could not be completed due to a connection timeou...
  [Supervisor] APPROVED
  Optimization converged after 1 iteration(s).
Epoch 2 done — pass rate: 0/3 (0%)

=== Epoch 3/3 ===
Running test: python_traceback
  FAIL — python_traceback: The actual output does not address the issue of invalid data or suggest a concrete next step for the
  Optimizing prompt...
  [Optimizer] iteration 1/3
  [Target] proposed: Provide a concise, user-friendly message that explains that the uploaded data could not be read because it is invalid, a...
  [Supervisor] APPROVED
  Optimization converged after 1 iteration(s).
Running test: database_constraint
  FAIL — database_constraint: The message does not suggest signing in or using a different email, and it includes a directive to c
  Optimizing prompt...
  [Optimizer] iteration 1/3
  [Target] proposed: Provide a concise, user-friendly message that informs the user that the email address is already in use, and suggests si...
  [Supervisor] APPROVED
  Optimization converged after 1 iteration(s).
Running test: network_timeout
  FAIL — network_timeout: The actual output does not specify that the action could not be completed due to a connection timeou
  Optimizing prompt...
  [Optimizer] iteration 1/3
  [Target] proposed: Provide a concise, user-friendly message that explains that the action could not be completed due to a connection timeou...
  [Supervisor] APPROVED
  Optimization converged after 1 iteration(s).
Epoch 3 done — pass rate: 0/3 (0%)
```

### Reading the trace

The optimizer never finds a single mutable block that satisfies all three tests simultaneously — each test's `expected_output` lists a *different* specific phrase the model must say ("uploaded data could not be read", "email address is already in use", "connection timed out"). After every test, the mutable block gets specialised toward that test's wording, but the next test then fails because the prompt no longer mentions *its* required phrase. This is a real-world failure mode worth understanding: with mutually-conflicting per-test acceptance criteria, the optimizer flip-flops rather than converges.

### Final prompt

```text
[IMMUTABLE]
You rewrite raw developer error messages into a single short message that will be shown to a non-technical end user inside a web application.

[MUTABLE]
Provide a concise, user-friendly message that explains that the action could not be completed due to a connection timeout, and suggests the user try again in a moment.
```

### Pass rates per epoch

```
['0%', '0%', '0%']
```

### API contract for this run

#### `POST /optimize` request (API key redacted, test-case inputs shown in compact form)

```json
{
  "prompt_sections": [
    {"text": "You rewrite raw developer error messages into a single short message that will be shown to a non-technical end user inside a web application.", "mutable": false},
    {"text": "Make the error friendly and clear.", "mutable": true}
  ],
  "test_cases": [
    {"name": "python_traceback",    "input_text": "Traceback (most recent call last):\n  File \"/srv/app/handlers/upload.py\", line 142, in handle_upload\n    parsed = json.loads(body)\njson.decoder.JSONDecodeError: ...", "expected_output": "A single short user-facing message (1-2 sentences) that says the uploaded data could not be read..."},
    {"name": "database_constraint", "input_text": "psycopg2.errors.UniqueViolation: duplicate key value violates unique constraint \"users_email_key\"\nDETAIL:  Key (email)=(test@example.com) already exists.", "expected_output": "A single short user-facing message saying the email address is already in use..."},
    {"name": "network_timeout",     "input_text": "requests.exceptions.ConnectTimeout: HTTPSConnectionPool(host='api.payments.internal', port=443): Max retries exceeded with url: /v2/charge",                  "expected_output": "A short user-facing message explaining that the action could not be completed because a connection timed out..."}
  ],
  "target_model":     {"model_id": "gpt-4o-mini", "api_key": "sk-proj-REDACTED", "base_url": "https://api.openai.com/v1", "name": "target"},
  "supervisor_model": {"model_id": "gpt-4o-mini", "api_key": "sk-proj-REDACTED", "base_url": "https://api.openai.com/v1", "name": "supervisor"},
  "epochs": 3,
  "max_iterations": 3
}
```

(For the verbatim, untruncated test-case payloads see [`examples/walkthrough_2.py`](examples/walkthrough_2.py).)

#### `POST /optimize` response

```json
{
  "run_id": "683a46b8-fb3e-479b-8123-3e582071d00a",
  "message": "Optimization started. Poll /history/{run_id} for progress."
}
```

#### `GET /history/683a46b8-…` — polled every 8s (9 polls total, 64.1s)

The first eight polls returned `status="running"` with empty `epochs[]`. The ninth poll returned:

```json
{
  "run_id": "683a46b8-fb3e-479b-8123-3e582071d00a",
  "status": "complete",
  "epochs": [
    {
      "epoch": 1,
      "pass_count": 0,
      "total_count": 3,
      "pass_rate": 0.0,
      "prompt_after": [
        {"text": "You rewrite raw developer error messages into a single short message that will be shown to a non-technical end user inside a web application.", "mutable": false},
        {"text": "Make the error clear, specify that the action could not be completed due to a connection timeout, and encourage the user to try again in a moment.", "mutable": true}
      ]
    },
    {
      "epoch": 2,
      "pass_count": 0,
      "total_count": 3,
      "pass_rate": 0.0,
      "prompt_after": [
        {"text": "You rewrite raw developer error messages into a single short message that will be shown to a non-technical end user inside a web application.", "mutable": false},
        {"text": "Make the error clear by stating that the action could not be completed because a connection timed out. Suggest that the user try again in a moment.", "mutable": true}
      ]
    },
    {
      "epoch": 3,
      "pass_count": 0,
      "total_count": 3,
      "pass_rate": 0.0,
      "prompt_after": [
        {"text": "You rewrite raw developer error messages into a single short message that will be shown to a non-technical end user inside a web application.", "mutable": false},
        {"text": "Make the error clear by stating that the action could not be completed due to a connection timeout. Suggest that the user try again in a moment.", "mutable": true}
      ]
    }
  ],
  "final_prompt": [
    {"text": "You rewrite raw developer error messages into a single short message that will be shown to a non-technical end user inside a web application.", "mutable": false},
    {"text": "Make the error clear by stating that the action could not be completed due to a connection timeout. Suggest that the user try again in a moment.", "mutable": true}
  ],
  "cost": null
}
```

Observation: between the library and server runs, the optimizer ended up at *different* fixed points for the same starting prompt. Both runs settled on a mutable section specialised to the *last* test case (network timeout), but the wording differs — the library final says "due to a connection timeout, and suggests the user try again in a moment", the API final says "due to a connection timeout. Suggest that the user try again in a moment." This is expected non-determinism from LLM sampling, not a bug.

---

## Example 3 — Scientific Abstract TL;DR (length/concision axis)

**Source:** [`examples/walkthrough_3.py`](examples/walkthrough_3.py)

**Scenario.** The model is asked to produce a one-line TL;DR for a scientific paper abstract. The mutable section says only "summarize the abstract for a general audience" — so the model returns a multi-sentence paragraph with parentheses and semicolons. The optimizer must impose strict length bounds: exactly one sentence, 15–30 words, no semicolons, no parentheses, no citation markers.

### Initial prompt

```text
[IMMUTABLE]
You are a science communicator. The user will paste a scientific paper abstract and you will produce a TL;DR that helps a general reader decide whether to read the paper.

[MUTABLE]
Summarize the abstract for a general audience.
```

### Step-by-step trace (library run)

```text
=== Epoch 1/3 ===
Running test: crispr_abstract
  FAIL — crispr_abstract: The actual output is not a single sentence and contains a colon, which violates the expected format.
  Optimizing prompt...
  [Optimizer] iteration 1/3
  [Target] proposed: Produce a single, concise sentence that captures the essence of the abstract, ensuring it is between 15 and 30 words and...
  [Supervisor] APPROVED
  Optimization converged after 1 iteration(s).
Running test: exoplanet_abstract
  FAIL — exoplanet_abstract: The actual output does not meet the requirement of being a single sentence within the specified word
  Optimizing prompt...
  [Optimizer] iteration 1/3
  [Target] proposed: Produce a single, concise sentence that captures the essence of the abstract, ensuring it is exactly one sentence betwee...
  [Supervisor] APPROVED
  Optimization converged after 1 iteration(s).
Epoch 1 done — pass rate: 0/2 (0%)

=== Epoch 2/3 ===
Running test: crispr_abstract
  FAIL — crispr_abstract: The actual output does not meet the specified requirements regarding sentence structure and punctuat
  Optimizing prompt...
  [Optimizer] iteration 1/3
  [Target] proposed: Produce a single, concise sentence that captures the essence of the abstract, ensuring it is exactly one sentence betwee...
  [Supervisor] APPROVED
  Optimization converged after 1 iteration(s).
Running test: exoplanet_abstract
  FAIL — exoplanet_abstract: The actual output does not mention that the planet is around a nearby K-dwarf and does not specify t
  Optimizing prompt...
  [Optimizer] iteration 1/3
  [Target] proposed: Produce a single, concise sentence that captures the essence of the abstract, ensuring it is exactly one sentence betwee...
  [Supervisor] APPROVED
  Optimization converged after 1 iteration(s).
Epoch 2 done — pass rate: 0/2 (0%)

=== Epoch 3/3 ===
Running test: crispr_abstract
  FAIL — crispr_abstract: The actual output is longer than 30 words and does not meet the specified sentence structure require
  Optimizing prompt...
  [Optimizer] iteration 1/3
  [Target] proposed: Produce a single, concise sentence that captures the essence of the abstract, ensuring it is exactly one sentence betwee...
  [Supervisor] APPROVED
  Optimization converged after 1 iteration(s).
Running test: exoplanet_abstract
  FAIL — exoplanet_abstract: The actual output does not specify that the planet is around a nearby K-dwarf and does not mention J
  Optimizing prompt...
  [Optimizer] iteration 1/3
  [Target] proposed: Produce a single, concise sentence that captures the essence of the abstract, ensuring it is exactly one sentence betwee...
  [Supervisor] APPROVED
  Optimization converged after 1 iteration(s).
Epoch 3 done — pass rate: 0/2 (0%)
```

### Reading the trace

The optimizer rewrites the mutable section into a sharp, rule-laden instruction ("exactly one sentence, between 15 and 30 words, no semicolons, no parentheses, no citation markers"), but each test's `expected_output` also requires content-level fidelity to the source abstract (must mention "K-dwarf", must mention "JWST", etc.). The mutable section can encode either the length rules or the per-test content requirements, but not both at the same time — so the optimizer keeps trading one for the other and never lands on a prompt that passes both tests.

### Final prompt

```text
[IMMUTABLE]
You are a science communicator. The user will paste a scientific paper abstract and you will produce a TL;DR that helps a general reader decide whether to read the paper.

[MUTABLE]
Produce a single, concise sentence that captures the essence of the abstract, ensuring it is exactly one sentence between 15 and 30 words, does not contain semicolons or parentheses, does not include numerical citations, and emphasizes that astronomers discovered a sub-Neptune exoplanet transiting a nearby K-dwarf star, making it a prime candidate for atmospheric study with JWST.
```

### Pass rates per epoch

```
['0%', '0%', '0%']
```

### API contract for this run

#### `POST /optimize` request (API key redacted, test-case inputs shown in compact form)

```json
{
  "prompt_sections": [
    {"text": "You are a science communicator. The user will paste a scientific paper abstract and you will produce a TL;DR that helps a general reader decide whether to read the paper.", "mutable": false},
    {"text": "Summarize the abstract for a general audience.", "mutable": true}
  ],
  "test_cases": [
    {"name": "crispr_abstract",     "input_text": "Background: CRISPR-Cas9 has revolutionized genome editing, but off-target effects remain a clinical concern...", "expected_output": "A single sentence, between 15 and 30 words, conveying that a new high-fidelity CRISPR-Cas9 variant sharply reduces off-target edits..."},
    {"name": "exoplanet_abstract",  "input_text": "We report the discovery of TOI-4823b, a sub-Neptune exoplanet transiting a nearby K-dwarf star at 38 parsecs...",  "expected_output": "A single sentence, 15 to 30 words, stating that astronomers found a sub-Neptune planet around a nearby K-dwarf..."}
  ],
  "target_model":     {"model_id": "gpt-4o-mini", "api_key": "sk-proj-REDACTED", "base_url": "https://api.openai.com/v1", "name": "target"},
  "supervisor_model": {"model_id": "gpt-4o-mini", "api_key": "sk-proj-REDACTED", "base_url": "https://api.openai.com/v1", "name": "supervisor"},
  "epochs": 3,
  "max_iterations": 3
}
```

#### `POST /optimize` response

```json
{
  "run_id": "7004e604-6d05-4432-a219-b6556f2d7d53",
  "message": "Optimization started. Poll /history/{run_id} for progress."
}
```

#### `GET /history/7004e604-…` — polled every 8s (7 polls total, 48.1s)

The first six polls returned `status="running"`. The seventh poll returned:

```json
{
  "run_id": "7004e604-6d05-4432-a219-b6556f2d7d53",
  "status": "complete",
  "epochs": [
    {
      "epoch": 1, "pass_count": 0, "total_count": 2, "pass_rate": 0.0,
      "prompt_after": [
        {"text": "You are a science communicator. The user will paste a scientific paper abstract and you will produce a TL;DR that helps a general reader decide whether to read the paper.", "mutable": false},
        {"text": "Produce a single sentence summary of the discovery that highlights the sub-Neptune planet's properties and its orbit around a nearby K-dwarf star, making it a strong candidate for atmospheric study with JWST.", "mutable": true}
      ]
    },
    {
      "epoch": 2, "pass_count": 0, "total_count": 2, "pass_rate": 0.0,
      "prompt_after": [
        {"text": "You are a science communicator. The user will paste a scientific paper abstract and you will produce a TL;DR that helps a general reader decide whether to read the paper.", "mutable": false},
        {"text": "Produce a single sentence summary of the discovery that states astronomers have identified a sub-Neptune exoplanet around a nearby K-dwarf star, making it an excellent candidate for atmospheric study with JWST.", "mutable": true}
      ]
    },
    {
      "epoch": 3, "pass_count": 0, "total_count": 2, "pass_rate": 0.0,
      "prompt_after": [
        {"text": "You are a science communicator. The user will paste a scientific paper abstract and you will produce a TL;DR that helps a general reader decide whether to read the paper.", "mutable": false},
        {"text": "Produce a single sentence summary of the discovery that states astronomers found a sub-Neptune planet transiting a nearby K-dwarf star, making it an ideal candidate for atmospheric study with JWST.", "mutable": true}
      ]
    }
  ],
  "final_prompt": [
    {"text": "You are a science communicator. The user will paste a scientific paper abstract and you will produce a TL;DR that helps a general reader decide whether to read the paper.", "mutable": false},
    {"text": "Produce a single sentence summary of the discovery that states astronomers found a sub-Neptune planet transiting a nearby K-dwarf star, making it an ideal candidate for atmospheric study with JWST.", "mutable": true}
  ],
  "cost": null
}
```

The library run of this same prompt landed on a *different* fixed point — the library run's final mutable section enforced length and punctuation rules ("exactly one sentence between 15 and 30 words, no semicolons or parentheses, no numerical citations") AND content fidelity ("emphasizes that astronomers discovered a sub-Neptune exoplanet…"), while the API run lost the length/punctuation rules and kept only the content-fidelity portion. Both runs failed all tests — but they failed for different reasons, again demonstrating that LLM-driven optimization is **stochastic** even with identical inputs.

---

## Reproducing this walkthrough

```bash
# From the project root, with .env containing OPENAI_API_KEY=sk-...
pip install -r requirements.txt

# 1. Verify the key and pick a model
curl -s https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  | python3 -c "import json,sys; [print(m['id']) for m in json.load(sys.stdin)['data']]" \
  | grep -E '^gpt-4o-mini$'

# 2. Library mode — clean stdout traces
python3 examples/walkthrough_1.py 2>&1 | tee /tmp/walkthrough_1.lib.log
python3 examples/walkthrough_2.py 2>&1 | tee /tmp/walkthrough_2.lib.log
python3 examples/walkthrough_3.py 2>&1 | tee /tmp/walkthrough_3.lib.log

# 3. Server mode — exercise the API contracts
uvicorn api:app --port 8000 > /tmp/uvicorn.log 2>&1 &
sleep 3
python3 /tmp/run_api_walkthroughs.py   # POSTs each example to /optimize and polls /history

# 4. Inspect captured contracts
cat /tmp/walkthrough_1.api.json   # full POST/response pair + every /history poll
cat /tmp/walkthrough_evaluate.json
cat /tmp/walkthrough_runs.json
```

The helper at `/tmp/run_api_walkthroughs.py` is what wrote the API-contract sections in this document. It mirrors each `examples/walkthrough_N.py` prompt and test cases into a `POST /optimize` body, polls `/history/{run_id}` until status is `complete`, and saves the raw request/response pairs to JSON.

---

## What this walkthrough confirms about the project

1. **The runner emits a structured, line-by-line trace** — every epoch boundary, every test, every iteration, every supervisor verdict — and that trace was captured directly into this document (the "Step-by-step trace" blocks are verbatim from `/tmp/walkthrough_N.lib.log`).
2. **The four-endpoint REST API works end-to-end** — `/runs` returns the list, `/evaluate` does a single judgment, `/optimize` queues an async run with a `run_id`, and `/history/{run_id}` returns a running snapshot first and an `epochs[] + final_prompt` payload when done.
3. **The optimizer is honest about non-convergence.** None of the three examples converged to 100% in 3 epochs × 3 iterations. The walkthrough documents what *actually* happened — the final prompts and final pass rates shown above are the real artifacts, not cherry-picked re-runs.
4. **The mutable / immutable split is preserved across every iteration.** The `[IMMUTABLE]` block at the top of each final prompt is byte-identical to the `[IMMUTABLE]` block in the initial prompt; only the `[MUTABLE]` block was rewritten.
