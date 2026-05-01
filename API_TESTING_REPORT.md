# Prompt Tuner — Full Feature Testing Report

**Date:** 2026-05-01
**Model under test:** `gpt-4o-mini` (OpenAI)
**Server:** FastAPI on `http://localhost:8000`

---

## REST API Endpoints

### `GET /runs`

Lists all optimization runs stored in the backend.

**curl:**
```bash
curl http://localhost:8000/runs
```

**Response (empty):**
```json
[]
```

**Response (after runs exist):**
```json
[
  {
    "run_id": "48a73fd2-23f8-4166-b6dc-a128312b68e9",
    "status": "complete",
    "created_at": 1777633411.095739
  }
]
```

**Status:** PASS

---

### `POST /evaluate`

Evaluates whether a model output satisfies an expected output using the LLM-as-judge strategy.

**curl (passing case):**
```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "actual_output": "Hey there! Sorry to hear you'\''re having trouble logging in. Let me help you get back in!",
    "expected_output": "A friendly, casual reply offering to help with the login issue",
    "supervisor_model": {
      "model_id": "gpt-4o-mini",
      "api_key": "sk-...",
      "base_url": "https://api.openai.com/v1"
    }
  }'
```

**Response (passing):**
```json
{
  "passed": true,
  "feedback": "",
  "reasoning": "The actual output conveys a friendly and casual tone while offering assistance with the login issue, which aligns with the expected output."
}
```

**curl (failing case):**
```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "actual_output": "We regret to inform you that your account access has been restricted. Please submit a formal support ticket.",
    "expected_output": "A friendly, casual reply offering to help with the login issue",
    "supervisor_model": {
      "model_id": "gpt-4o-mini",
      "api_key": "sk-...",
      "base_url": "https://api.openai.com/v1"
    }
  }'
```

**Response (failing):**
```json
{
  "passed": false,
  "feedback": "The response is formal and lacks a friendly tone, failing to offer assistance in a casual manner.",
  "reasoning": "The model likely interpreted the prompt as a need for a formal notification rather than a friendly offer to help."
}
```

**Status:** PASS

---

### `POST /optimize`

Starts an asynchronous prompt optimization run. Returns a `run_id` immediately; poll `/history/{run_id}` for results.

**curl:**
```bash
curl -X POST http://localhost:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "prompt_sections": [
      { "text": "You are a customer support agent for Acme Corp.", "mutable": false },
      { "text": "Respond formally and avoid contractions.", "mutable": true }
    ],
    "test_cases": [
      {
        "name": "greeting",
        "input_text": "Hey! I can'\''t log in.",
        "expected_output": "A casual, friendly reply offering to help with the login issue, using contractions."
      },
      {
        "name": "thanks",
        "input_text": "Thanks, got it working!",
        "expected_output": "A short cheerful human response, not stiff corporate speak."
      }
    ],
    "target_model": {
      "model_id": "gpt-4o-mini",
      "api_key": "sk-...",
      "base_url": "https://api.openai.com/v1"
    },
    "supervisor_model": {
      "model_id": "gpt-4o-mini",
      "api_key": "sk-...",
      "base_url": "https://api.openai.com/v1"
    },
    "epochs": 2,
    "max_iterations": 2
  }'
```

**Response (immediate):**
```json
{
  "run_id": "48a73fd2-23f8-4166-b6dc-a128312b68e9",
  "message": "Optimization started. Poll /history/{run_id} for progress."
}
```

**Status:** PASS
**Observed:** 0% pass rate (epoch 1) → 100% pass rate (epoch 2). Mutable section evolved from `"Respond formally and avoid contractions."` to `"Keep it light and friendly, using a casual tone with contractions."`

---

### `GET /history/{run_id}`

Polls the status and full results of an optimization run.

**curl:**
```bash
# Replace RUN_ID with the value returned by POST /optimize
curl http://localhost:8000/history/48a73fd2-23f8-4166-b6dc-a128312b68e9
```

**Response (while running):**
```json
{
  "run_id": "...",
  "status": "running",
  "epochs": []
}
```

**Response (complete):**
```json
{
  "run_id": "48a73fd2-23f8-4166-b6dc-a128312b68e9",
  "status": "complete",
  "epochs": [
    {
      "epoch": 1,
      "pass_count": 0,
      "total_count": 2,
      "pass_rate": 0.0,
      "prompt_after": [
        { "text": "You are a customer support agent for Acme Corp.", "mutable": false },
        { "text": "Keep it light and friendly, using a casual tone with contractions.", "mutable": true }
      ]
    },
    {
      "epoch": 2,
      "pass_count": 2,
      "total_count": 2,
      "pass_rate": 1.0,
      "prompt_after": [
        { "text": "You are a customer support agent for Acme Corp.", "mutable": false },
        { "text": "Keep it light and friendly, using a casual tone with contractions.", "mutable": true }
      ]
    }
  ],
  "final_prompt": [
    { "text": "You are a customer support agent for Acme Corp.", "mutable": false },
    { "text": "Keep it light and friendly, using a casual tone with contractions.", "mutable": true }
  ],
  "cost": null
}
```

**Status:** PASS

---

## Evaluator Strategies

| Strategy | Description | Test Result |
|----------|-------------|-------------|
| `llm_judge` | LLM-as-judge: semantic equivalence via supervisor model | PASS |
| `exact_match` | Case-insensitive string equality | PASS |
| `regex` | Python regex match against expected pattern | PASS |
| `custom` (`CallableEvaluator`) | Inject any Python function as evaluator | PASS |
| `composite` | All sub-evaluators must pass (AND logic) | PASS |
| `semantic` | Embedding-based similarity | Registered, not live-tested (requires sentence-transformers) |

### Exact Match
```python
ExactMatchEvaluator(case_sensitive=False)

evaluate("Hello World", "hello world")  → passed=True
evaluate("Hello World", "Goodbye World") → passed=False, feedback="Output does not match expected. Got: Hello World"
```

### Regex Match
```python
RegexMatchEvaluator()

evaluate("Your order #12345 is confirmed", r"order #\d+") → passed=True
evaluate("No order info here", r"order #\d+")             → passed=False, feedback="Output did not match pattern: order #\d+"
```

### Custom (CallableEvaluator)
```python
def word_count_check(actual, expected):
    wc = len(actual.split())
    ok = wc <= 20
    return EvalResult(passed=ok, feedback='' if ok else f'Too long: {wc} words', reasoning='')

CallableEvaluator(word_count_check).evaluate("Short reply.", "")     → passed=True
CallableEvaluator(word_count_check).evaluate("word " * 25, "")       → passed=False, feedback="Too long: 25 words"
```

### Composite
```python
CompositeEvaluator([ExactMatchEvaluator(), RegexMatchEvaluator()])

evaluate("hello world", "hello world")  → passed=True   (both pass)
evaluate("foo", "hello world")          → passed=False   (exact match fails)
```

---

## Optimizer Strategies

### Discussion Loop (default)
Target proposes → Supervisor approves/rejects → iterate. Converges on supervisor approval.

```
[Optimizer] iteration 1/2
[Target] proposed: Respond in a casual, friendly manner...
[Supervisor] APPROVED
Converged: True, iterations: 1
```

### Monte Carlo
Generates N candidate prompt rewrites in parallel at high temperature, evaluates each, picks the first passing one.

```python
MonteCarloOptimizer(n_samples=3)
# Output:
[MonteCarlo] generating 3 candidates...
[MonteCarlo] evaluating 3 candidates...
[MonteCarlo] candidate 1 failed
[MonteCarlo] candidate 2 failed
[MonteCarlo] candidate 3 PASSED
Converged: True
New mutable: "Respond in a casual, friendly manner."
```

### Beam Search
Maintains top-k candidate beams, expands each, evaluates, and keeps the best. Early-exits on first pass.

```python
BeamSearchOptimizer(beam_width=2)
# Output:
[BeamSearch] depth 1/2, beams=1
[BeamSearch] found passing candidate at depth 1
Converged: True
New mutable: "Respond in a friendly manner."
```

---

## Feature Flags

Flags are loaded in priority order: **env vars > config.yaml > hardcoded defaults**.

**Env var coercion test (`PO_FEATURE_*` prefix):**
```bash
PO_FEATURE_RETRY_ENABLED=true
PO_FEATURE_RETRY_MAX_ATTEMPTS=5
PO_FEATURE_CACHE_ENABLED=true
PO_FEATURE_CACHE_MAX_SIZE=512
```
```python
flags.retry_enabled       → True   ✓
flags.retry_max_attempts  → 5      ✓
flags.cache_enabled       → True   ✓
flags.cache_max_size      → 512    ✓
```

> **Bug fixed:** `from __future__ import annotations` caused `dataclasses.fields().type` to return string annotations instead of actual types, silently breaking all env var coercion. Fixed by switching to `typing.get_type_hints()`.

**Available flags (key ones):**

| Flag | Default | Effect |
|------|---------|--------|
| `evaluator_strategy` | `llm_judge` | Selects evaluator from registry |
| `optimizer_strategy` | `discussion_loop` | Selects optimizer from registry |
| `retry_enabled` | `false` | Wraps ModelClient with exponential backoff |
| `cache_enabled` | `false` | Wraps ModelClient with LRU response cache |
| `cost_tracking_enabled` | `false` | Tracks token usage and estimated cost |
| `convergence_detection_enabled` | `false` | Stops early when pass rate stagnates |
| `convergence_patience` | `3` | Epochs without improvement before stopping |
| `api_persistent_storage` | `false` | Use SQLite instead of in-memory storage |
| `ui_export_enabled` | `false` | Shows CSV/JSON export buttons in UI |

---

## Reliability Wrappers

### LRU Response Cache
```python
cache = ResponseCache(max_size=3, cache_all_temperatures=True)
cache.put("gpt-4o-mini", messages, 0.0, "Hi there!")
cache.get("gpt-4o-mini", messages, 0.0)  → "Hi there!"  (HIT)
cache.get("gpt-4o-mini", other_msgs, 0.0) → None         (MISS)
# LRU eviction: oldest entry dropped when size exceeded
cache.stats() → {"hits": 1, "misses": 2, "size": 3, "max_size": 3}
```
**Status:** PASS

### Retry with Exponential Backoff
```python
policy = RetryPolicy(max_attempts=3, base_delay=0.01)

# Flaky function that fails twice then succeeds:
result = flaky_fn()  → "success" (took 3 attempts)

# Function that always fails:
always_fails()  → raises after 3 attempts
```
**Status:** PASS

### Cost Tracker
```python
tracker = CostTracker()
tracker.record("gpt-4o-mini", {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})
tracker.record("gpt-4o-mini", {"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280})
tracker.record("gpt-4o",      {"prompt_tokens": 50,  "completion_tokens": 30, "total_tokens": 80})

tracker.summary() →
{
  "total_cost_usd": 0.000548,
  "total_tokens": 510,
  "total_prompt_tokens": 350,
  "total_completion_tokens": 160,
  "call_count": 3
}
```
**Status:** PASS

---

## Storage Backends

### InMemoryRunStorage
```python
mem = InMemoryRunStorage()
mem.save_run("run-1", "running")
mem.save_run("run-2", "complete", {"pass_rates": [0.5, 1.0]})
mem.list_runs()   → [{"run_id": "run-1", ...}, {"run_id": "run-2", ...}]
mem.get_run("run-2") → {"run_id": "run-2", "status": "complete", "data": {...}}
mem.get_run("nonexistent") → None
```
**Status:** PASS

### SQLiteRunStorage
```python
sql = SQLiteRunStorage("runs.db")
sql.save_run("run-a", "running")
sql.save_run("run-b", "complete", {"epoch_count": 3})
sql.list_runs()  → [{"run_id": "run-b", ...}, {"run_id": "run-a", ...}]  # DESC order
sql.delete_run("run-a")
sql.list_runs()  → [{"run_id": "run-b", ...}]
sql.cleanup_expired(ttl_seconds=0.05)  → 1  # deleted 1 expired run
```
**Status:** PASS

---

## Utilities

### Session Save/Load
```python
save_session("session.json", prompt, test_cases, target_cfg, supervisor_cfg, epochs=3, max_iterations=2)
loaded = load_session("session.json")
loaded["prompt"].render()    → "You are a support bot.\nBe friendly."
loaded["test_cases"][0].name → "t1"
loaded["epochs"]             → 3
```
**Status:** PASS

### Prompt Version History
```python
h = PromptVersionHistory()
h.record(p1, epoch=1, pass_rate=0.0)
h.record(p2, epoch=2, pass_rate=0.5)
h.record(p3, epoch=3, pass_rate=1.0)

h.get_version(1).prompt.render() → "Be formal."
h.get_version(3).prompt.render() → "Be casual, friendly, and use contractions."
h.diff(1, 3) →
  --- v1
  +++ v3
  @@ -1 +1 @@
  -Be formal.
  +Be casual, friendly, and use contractions.
h.rollback_to(1).render() → "Be formal."
```
**Status:** PASS

### Export (CSV / JSON / Prompt text)
```
CSV:
Epoch,Test Name,Passed,Feedback,Actual Output,Expected Output
1,t1,True,,Hi there!,hi there

JSON:
{
  "epochs": [{
    "epoch": 1, "pass_rate": 1.0, "pass_count": 1, "total_count": 1,
    "prompt_before": "Be helpful.", "prompt_after": "Be helpful.",
    "test_results": [{ "name": "t1", "passed": true, ... }]
  }]
}

Prompt text: "Be helpful."
```
**Status:** PASS

---

## A/B Comparison

Runs two prompts against the same test cases and compares pass rates.

```python
run_ab_comparison(
    prompt_a=Prompt([PromptSection("You are a formal assistant.", mutable=False)]),
    prompt_b=Prompt([PromptSection("You are a casual assistant.", mutable=False)]),
    test_cases=[TestCase("greet", "Hey!", "A friendly casual response.")],
    target=model,
    supervisor=supervisor,
)
→ ABComparisonResult(pass_rate_a=1.0, pass_rate_b=0.0, winner="A")
```
**Status:** PASS

---

## Multi-Model Evaluation

Runs a single prompt across multiple model configs and returns per-model results.

```python
run_multi_model_eval(
    prompt=Prompt([PromptSection("Answer with a single number only.", mutable=False)]),
    test_cases=[TestCase("math", "What is 2+2?", "4")],
    target_configs=[
        ModelConfig(model_id="gpt-4o-mini", name="model-a", ...),
        ModelConfig(model_id="gpt-4o-mini", name="model-b", ...),
    ],
    supervisor=supervisor,
)
→ [
    MultiModelResult(model_id="gpt-4o-mini", pass_rate=1.0, test_results=[...]),
    MultiModelResult(model_id="gpt-4o-mini", pass_rate=1.0, test_results=[...]),
  ]
```
**Status:** PASS

---

## Convergence Detection

When `convergence_detection_enabled=True` and `convergence_patience=N`, the runner stops early if pass rate does not improve for N consecutive epochs.

```
=== Epoch 1/5 === → pass rate: 0/2 (0%)
=== Epoch 2/5 === → pass rate: 0/2 (0%)
=== Epoch 3/5 === → pass rate: 0/2 (0%)
Pass rate stagnated for 2 epochs — stopping.
(Stopped after 3 of 5 epochs)
```
**Status:** PASS

---

## Strategy Registry

All strategies are auto-registered at import time via module-level `registry.register(...)`.

```python
evaluator_registry.list_keys() → ['composite', 'custom', 'exact_match', 'llm_judge', 'regex', 'semantic']
optimizer_registry.list_keys() → ['beam_search', 'discussion_loop', 'monte_carlo']
updater_registry.list_keys()   → ['merge_mutable', 'per_section']
```
**Status:** PASS

---

## Summary

| Component | Status | Notes |
|-----------|--------|-------|
| `GET /runs` | PASS | Lists all runs |
| `POST /evaluate` | PASS | Pass and fail cases confirmed |
| `POST /optimize` | PASS | 0%→100% in 2 epochs |
| `GET /history/{run_id}` | PASS | Running + complete states |
| Evaluator: llm_judge | PASS | Live API |
| Evaluator: exact_match | PASS | |
| Evaluator: regex | PASS | |
| Evaluator: custom | PASS | |
| Evaluator: composite | PASS | AND logic |
| Optimizer: discussion_loop | PASS | Live API |
| Optimizer: monte_carlo | PASS | Live API, n_samples=3 |
| Optimizer: beam_search | PASS | Live API, beam_width=2 |
| Feature flags (env vars) | PASS | After bug fix |
| Cache (LRU) | PASS | |
| Retry (backoff) | PASS | |
| Cost tracker | PASS | |
| InMemoryRunStorage | PASS | |
| SQLiteRunStorage | PASS | Including TTL cleanup |
| Session save/load | PASS | |
| Version history + diff + rollback | PASS | |
| Export (CSV/JSON/text) | PASS | |
| A/B comparison | PASS | |
| Multi-model eval | PASS | |
| Convergence detection | PASS | Early stop after stagnation |
| Strategy registry | PASS | All 11 strategies registered |
