# Experiment Results — Prompt Optimizer

## Experiment: Tone Fixer

**Date:** 2026-04-29
**Script:** `examples/example_via_api.py`
**Model:** `gpt-4o-mini` (used as both target and supervisor)
**Epochs configured:** 3 (early-stopped at 2)
**Max optimizer iterations per failure:** 3

---

## Setup

### Prompt (before optimization)

```
[IMMUTABLE]
You are a customer support agent for a software company called Acme.

[MUTABLE]
Respond in a professional and formal manner. Use complete sentences and avoid contractions.
```

The immutable section defined the bot's role — it could not be changed.
The mutable section defined tone/style — this was the optimization target.

### Test Cases

| Name | Input | Expected character |
|------|-------|-------------------|
| `greeting` | "Hey! I can't log in to my account." | Casual, warm, uses contractions |
| `refund_request` | "I want a refund please" | Empathetic, conversational, not corporate |
| `thanks` | "Thanks, got it working!" | Short, cheerful, human |

---

## Results

### Pass rate per epoch

| Epoch | Passed | Total | Pass Rate |
|-------|--------|-------|-----------|
| 1     | 2      | 3     | **67%**   |
| 2     | 3      | 3     | **100%** ✓ |

The run **early-stopped** at epoch 2 — all tests were passing so there was no need to continue to epoch 3.

---

## What changed

### Mutable section — before
```
Respond in a professional and formal manner. Use complete sentences and avoid contractions.
```

### Mutable section — after (epoch 2)
```
Respond in a casual and friendly manner. Use complete sentences and include contractions to create a warm tone.
```

The optimizer made two targeted sentence-level changes:
1. **"professional and formal"** → **"casual and friendly"** — directly addresses the tone mismatch
2. **"avoid contractions"** → **"include contractions to create a warm tone"** — flipped the specific instruction that was causing stiffness

The immutable section ("You are a customer support agent...") was **not touched** — confirming that the mutable/immutable boundary works correctly.

---

## Observations

### What worked well

- **Precise changes**: The optimizer didn't rewrite the whole prompt. It made surgical, sentence-level edits — exactly as designed.
- **Fast convergence**: 2 epochs was enough to reach 100% pass rate for a 3-test suite.
- **Immutability respected**: The role definition was never modified, even though it was adjacent to the failing mutable section.
- **Supervisor feedback loop**: On the first epoch failure, the supervisor's feedback ("the tone is too formal, add contractions") directly shaped the proposal that succeeded in epoch 2.
- **Early stopping**: The run terminated as soon as all tests passed, saving unnecessary API calls.

### Limitations observed

- **Semantic expected outputs**: The test cases used natural-language descriptions as expected outputs (e.g., "A casual, friendly reply...") rather than exact strings. The LLM-as-judge handled this well, but it means pass/fail depends on the supervisor's interpretation, which may not be perfectly deterministic across runs.
- **Same model for target and supervisor**: Using `gpt-4o-mini` for both roles worked here, but in practice a more capable model as the supervisor (e.g., `gpt-4o`) would produce more reliable evaluations and richer optimization guidance.
- **Single mutable block**: This experiment had one mutable section. With multiple interleaved mutable/immutable sections, the merge logic consolidates them into one block — a known simplification in the current implementation.

---

## Takeaway

The core hypothesis holds: **prompts can be automatically improved through a target↔supervisor discussion loop**. In this experiment a single failing tone instruction ("avoid contractions") was identified, flipped, and validated within 2 epochs — with zero manual intervention.

This maps directly to the original motivation: when a model is updated (e.g., from GPT-4 to GPT-4o-mini), its default stylistic behaviour changes. Rather than manually re-tuning prompts, you can run the optimizer with a suite of expected-behaviour test cases and let it adapt automatically.

---

## API trace (summary)

```
POST /optimize  →  run_id: 33a99d40-1745-4542-9606-e66f7a9f0bb7
GET  /history/33a99d40-...  →  status: running  (×2 polls)
GET  /history/33a99d40-...  →  status: complete

Epoch 1: 2/3 passed (67%)
Epoch 2: 3/3 passed (100%) ← early stop
```
