# Prompt Optimizer

An iterative prompt improvement library — like ML training, but for prompts.

A **target model** runs your test cases. A **supervisor model** judges the outputs and guides improvements. Mutable sections of your prompt are refined sentence-by-sentence across epochs until all tests pass.

---

## How it works

```
for each epoch:
  for each test case:
    1. Run:      target_model(prompt + input)  ->  output
    2. Evaluate: supervisor judges output vs expected
    3. If fail:
         for up to N iterations:
           a. target proposes a sentence-level change to mutable sections
           b. supervisor reviews the proposal
           c. if approved -> apply and move on
              if rejected -> supervisor gives revision guidance, loop again
    4. Re-run remaining test cases with updated prompt
```

Only **mutable** sections of the prompt are ever changed. Immutable sections (role definitions, business rules, etc.) are locked.

---

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
OPENAI_API_KEY=sk-...
```

---

## Run the Gradio UI

```bash
python app.py
```

Opens at **http://localhost:7860** with 6+ tabs:

| Tab | Purpose |
|-----|---------|
| **1 - Prompt Editor** | Paste prompt, toggle mutable/immutable per line, mark lines immutable by number, color-coded preview |
| **2 - Test Cases** | Add test cases: name, input, expected output |
| **3 - Model Config** | Configure supervisor & target models (any OpenAI-compatible endpoint) |
| **4 - Run** | Set epochs & iterations, run optimization, view live log |
| **5 - Results** | Pass-rate chart, per-epoch diffs, per-test results, export CSV/JSON |
| **6 - Session** | Save/load full config as JSON |
| **7 - Cost Dashboard** | Token usage & estimated cost (when enabled) |

---

## Run the REST API

```bash
uvicorn api:app --reload --port 8000
```

API docs at **http://localhost:8000/docs**

### Endpoints

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| `POST` | `/optimize` | Yes (if enabled) | Start async optimization run |
| `GET` | `/history/{run_id}` | Yes | Poll run status and results |
| `POST` | `/evaluate` | Yes | Single synchronous evaluation |
| `GET` | `/runs` | Yes | List all runs |

---

## Feature Toggles

Every feature is controlled via **feature flags**. Configure them in three ways (later overrides earlier):

1. **Defaults** — hardcoded in `config/feature_flags.py`
2. **config.yaml** — file in project root
3. **Environment variables** — prefix `PO_FEATURE_`, e.g. `PO_FEATURE_RETRY_ENABLED=true`

### config.yaml example

```yaml
features:
  # Strategies
  evaluator_strategy: "llm_judge"
  optimizer_strategy: "discussion_loop"
  updater_strategy: "merge_mutable"

  # Reliability
  retry_enabled: true
  cache_enabled: true
  checkpoint_enabled: true
  cost_tracking_enabled: true

  # UI
  ui_thread_safe_state: true
  ui_auto_poll: true
  ui_input_validation: true
  ui_export_enabled: true
  ui_color_preview: true
  ui_cost_dashboard: true

  # API
  api_auth_enabled: true
  api_auth_key: "your-secret-key"
  api_persistent_storage: true
  api_storage_backend: "sqlite"

  # Advanced
  convergence_detection_enabled: true
  convergence_patience: 3
```

### Full Flag Reference

#### Core Strategy Selection

| Flag | Default | Options | Description |
|------|---------|---------|-------------|
| `evaluator_strategy` | `llm_judge` | `llm_judge`, `exact_match`, `regex`, `semantic`, `custom`, `composite` | How to evaluate model output vs expected |
| `optimizer_strategy` | `discussion_loop` | `discussion_loop`, `monte_carlo`, `beam_search` | How to optimize failing prompts |
| `updater_strategy` | `merge_mutable` | `merge_mutable`, `per_section` | How to apply changes to mutable sections |

**Evaluator strategies:**
- `llm_judge` — Supervisor LLM judges semantic equivalence (default)
- `exact_match` — Deterministic string comparison, no LLM needed
- `regex` — Expected output is treated as a regex pattern
- `semantic` — Embedding cosine similarity with configurable threshold
- `custom` — Wrap your own `Callable[[str, str], EvalResult]`
- `composite` — Majority vote across multiple evaluators

**Optimizer strategies:**
- `discussion_loop` — Target proposes changes, supervisor reviews (default)
- `monte_carlo` — Generate N proposals in parallel, pick the best
- `beam_search` — Maintain top-k candidates, expand and prune

**Updater strategies:**
- `merge_mutable` — All mutable sections treated as one block (default)
- `per_section` — Update each mutable section independently

#### Reliability

| Flag | Default | Type | Description |
|------|---------|------|-------------|
| `retry_enabled` | `false` | bool | Retry failed API calls with exponential backoff |
| `retry_max_attempts` | `3` | int | Max retry attempts per call |
| `retry_base_delay` | `1.0` | float | Base delay in seconds between retries |
| `cache_enabled` | `false` | bool | LRU cache for deterministic LLM calls (temperature=0.0) |
| `cache_max_size` | `256` | int | Max cached responses |
| `checkpoint_enabled` | `false` | bool | Save prompt state between epochs for crash recovery |
| `checkpoint_dir` | `.checkpoints` | str | Directory for checkpoint files |
| `cost_tracking_enabled` | `false` | bool | Track token usage and estimate USD cost |
| `timeout_seconds` | `120.0` | float | Timeout per LLM API call |

#### UI

| Flag | Default | Type | Description |
|------|---------|------|-------------|
| `ui_thread_safe_state` | `true` | bool | Thread-safe access to shared UI state |
| `ui_auto_poll` | `false` | bool | Auto-refresh log output (no manual "Refresh" needed) |
| `ui_auto_poll_interval` | `3.0` | float | Seconds between auto-polls |
| `ui_input_validation` | `false` | bool | Validate inputs before starting a run |
| `ui_export_enabled` | `false` | bool | Show "Export CSV/JSON" buttons on Results tab |
| `ui_cost_dashboard` | `false` | bool | Show Tab 7 (Cost Dashboard) |
| `ui_color_preview` | `false` | bool | Color-coded prompt preview (green=mutable, gray=immutable) |

#### API

| Flag | Default | Type | Description |
|------|---------|------|-------------|
| `api_auth_enabled` | `false` | bool | Require `Authorization: Bearer <key>` header |
| `api_auth_key` | `""` | str | The expected API key |
| `api_persistent_storage` | `false` | bool | Persist run history to disk |
| `api_storage_backend` | `memory` | str | `memory` or `sqlite` |
| `api_sqlite_path` | `runs.db` | str | SQLite database path |
| `api_webhook_enabled` | `false` | bool | POST to a webhook URL on run completion |
| `api_run_ttl_seconds` | `0` | int | Auto-delete runs older than N seconds (0=disabled) |

#### Advanced

| Flag | Default | Type | Description |
|------|---------|------|-------------|
| `ab_comparison_enabled` | `false` | bool | A/B prompt comparison |
| `version_history_enabled` | `false` | bool | Track prompt versions with diffs |
| `multi_model_eval_enabled` | `false` | bool | Run same prompt across multiple target models |
| `convergence_detection_enabled` | `false` | bool | Stop if pass rate stagnates |
| `convergence_patience` | `3` | int | Epochs of no improvement before stopping |
| `test_suite_tags_enabled` | `false` | bool | Tags, priority, weight on test cases |

---

## Model Compatibility

Any OpenAI-compatible endpoint works:

| Provider | Base URL |
|----------|----------|
| OpenAI | `https://api.openai.com/v1` |
| Anthropic (OpenAI compat) | `https://api.anthropic.com/v1` |
| OpenRouter | `https://openrouter.ai/api/v1` |
| Groq | `https://api.groq.com/openai/v1` |
| Together AI | `https://api.together.xyz/v1` |
| Ollama (local) | `http://localhost:11434/v1` |

---

## Examples

```bash
# Tone Fixer — makes a support bot friendlier
python examples/example_tone_fixer.py

# JSON Output Enforcer — teaches the model to return structured JSON
python examples/example_json_output.py

# API demo — submits a run to the live API server
# (requires: uvicorn api:app --reload --port 8000)
python examples/example_via_api.py

# A/B Comparison — compare two prompts head-to-head
python examples/example_ab_comparison.py
```

---

## Using as a Library

```python
from core.models import ModelConfig, ModelClient, build_client
from core.prompt import Prompt, PromptSection
from core.test_case import TestCase
from core.runner import run_optimization
from config import load_feature_flags

flags = load_feature_flags()

prompt = Prompt([
    PromptSection(text="You are a helpful assistant.", mutable=False),
    PromptSection(text="Keep replies under two sentences.", mutable=True),
])

test_cases = [
    TestCase(
        name="brevity",
        input_text="Explain quantum entanglement.",
        expected_output="A clear explanation in two sentences or fewer.",
    ),
]

cfg = ModelConfig(model_id="gpt-4o-mini", api_key="sk-...", base_url="https://api.openai.com/v1")
client = build_client(cfg, flags)  # applies retry, cache, cost tracking per flags

history = run_optimization(
    prompt=prompt,
    test_cases=test_cases,
    target=client,
    supervisor=client,
    epochs=5,
    max_iterations=3,
    log_callback=print,
    flags=flags,
)

print(history.final_prompt.render())
print("Pass rates:", history.pass_rates())
```

---

## Project Structure

```
model updater/
├── config/
│   ├── feature_flags.py              # FeatureFlags dataclass + loader
│   └── settings.py                   # AppSettings (defaults, pricing)
├── core/
│   ├── models.py                     # ModelConfig, ModelClient, build_client()
│   ├── prompt.py                     # Prompt + PromptSection
│   ├── test_case.py                  # TestCase (+ tags, priority, weight)
│   ├── evaluator.py                  # EvalResult + backward-compat evaluate()
│   ├── optimizer.py                  # OptimizationResult + backward-compat optimize()
│   ├── runner.py                     # run_optimization() + run_epoch()
│   ├── ab_comparison.py              # A/B prompt comparison
│   ├── multi_model.py                # Multi-model evaluation
│   ├── interfaces/
│   │   ├── evaluator.py              # EvaluatorProtocol
│   │   ├── optimizer.py              # OptimizerProtocol
│   │   └── prompt_updater.py         # PromptUpdaterProtocol
│   ├── evaluators/
│   │   ├── llm_judge.py              # LLM-as-judge
│   │   ├── exact_match.py            # String comparison
│   │   ├── regex_match.py            # Regex matching
│   │   ├── semantic_similarity.py    # Embedding similarity
│   │   ├── custom.py                 # User-provided callable
│   │   └── composite.py              # Majority-vote
│   ├── optimizers/
│   │   ├── discussion_loop.py        # Target↔supervisor loop
│   │   ├── monte_carlo.py            # Parallel proposals
│   │   └── beam_search.py            # Top-k candidate search
│   └── updaters/
│       ├── merge_mutable.py          # Merge all mutable into one
│       └── per_section.py            # Update sections independently
├── reliability/
│   ├── retry.py                      # Exponential backoff
│   ├── cache.py                      # LRU response cache
│   ├── checkpoint.py                 # Save/restore between epochs
│   ├── cost_tracker.py               # Token & cost tracking
│   └── timeout.py                    # API call timeout
├── storage/
│   ├── base.py                       # RunStorageProtocol
│   ├── memory.py                     # In-memory storage
│   └── sqlite.py                     # SQLite persistent storage
├── utils/
│   ├── serialization.py              # Session save/load
│   ├── export.py                     # CSV/JSON export
│   ├── registry.py                   # StrategyRegistry
│   └── version_history.py            # Prompt version tracking
├── app.py                            # Gradio UI
├── api.py                            # FastAPI REST API
├── config.yaml                       # Feature flag overrides
├── requirements.txt
├── .env                              # API keys
└── examples/
    ├── example_tone_fixer.py
    ├── example_json_output.py
    ├── example_via_api.py
    └── example_ab_comparison.py
```
