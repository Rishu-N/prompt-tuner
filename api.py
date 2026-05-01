"""
Prompt Optimizer — REST API
Run with:  uvicorn api:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import json
import uuid
import os
from typing import Optional
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import load_feature_flags
from core.models import ModelConfig, ModelClient, build_client
from core.prompt import Prompt, PromptSection
from core.test_case import TestCase
from core.runner import run_optimization, OptimizationHistory
from core.evaluator import evaluate
from reliability.cost_tracker import CostTracker
from storage.memory import InMemoryRunStorage
from storage.sqlite import SQLiteRunStorage

FLAGS = load_feature_flags()

# -----------------------------------------------------------------------
# Storage backend (4B)
# -----------------------------------------------------------------------

def _build_storage():
    if FLAGS.api_persistent_storage and FLAGS.api_storage_backend == "sqlite":
        return SQLiteRunStorage(FLAGS.api_sqlite_path)
    return InMemoryRunStorage()


_storage = _build_storage()

# In-memory map for OptimizationHistory objects (not serializable to storage directly)
_history_cache: dict[str, OptimizationHistory] = {}


# -----------------------------------------------------------------------
# TTL cleanup (4D)
# -----------------------------------------------------------------------

async def _ttl_cleanup_loop():
    """Background coroutine that purges expired runs periodically."""
    while True:
        await asyncio.sleep(60)
        if FLAGS.api_run_ttl_seconds > 0:
            count = _storage.cleanup_expired(FLAGS.api_run_ttl_seconds)
            if count > 0:
                # Also clean history cache
                for run_id in list(_history_cache):
                    entry = _storage.get_run(run_id)
                    if entry is None:
                        _history_cache.pop(run_id, None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = None
    if FLAGS.api_run_ttl_seconds > 0:
        task = asyncio.create_task(_ttl_cleanup_loop())
    yield
    if task:
        task.cancel()


app = FastAPI(title="Prompt Optimizer API", version="0.2.0", lifespan=lifespan)


# -----------------------------------------------------------------------
# Auth middleware (4A)
# -----------------------------------------------------------------------

async def verify_api_key(authorization: Optional[str] = Header(None)):
    if not FLAGS.api_auth_enabled:
        return
    if not authorization or authorization != f"Bearer {FLAGS.api_auth_key}":
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# -----------------------------------------------------------------------
# Structured error handler (4E)
# -----------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": type(exc).__name__,
            "message": str(exc),
            "path": str(request.url.path),
        },
    )


# -----------------------------------------------------------------------
# Request / response schemas
# -----------------------------------------------------------------------

class PromptSectionSchema(BaseModel):
    text: str
    mutable: bool = True


class ModelConfigSchema(BaseModel):
    model_id: str
    api_key: str
    base_url: Optional[str] = None
    name: str = ""


class TestCaseSchema(BaseModel):
    name: str = ""
    input_text: str
    expected_output: str


class OptimizeRequest(BaseModel):
    prompt_sections: list[PromptSectionSchema]
    test_cases: list[TestCaseSchema]
    target_model: ModelConfigSchema
    supervisor_model: ModelConfigSchema
    epochs: int = 5
    max_iterations: int = 3
    webhook_url: Optional[str] = None  # 4C


class EvaluateRequest(BaseModel):
    actual_output: str
    expected_output: str
    supervisor_model: ModelConfigSchema


class EvaluateResponse(BaseModel):
    passed: bool
    feedback: str
    reasoning: str


class OptimizeResponse(BaseModel):
    run_id: str
    message: str


class EpochSummary(BaseModel):
    epoch: int
    pass_count: int
    total_count: int
    pass_rate: float
    prompt_after: list[PromptSectionSchema]


class HistoryResponse(BaseModel):
    run_id: str
    status: str
    epochs: list[EpochSummary]
    final_prompt: Optional[list[PromptSectionSchema]] = None
    cost: Optional[dict] = None


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _to_model_client(schema: ModelConfigSchema) -> ModelClient:
    cfg = ModelConfig(
        model_id=schema.model_id, api_key=schema.api_key,
        base_url=schema.base_url, name=schema.name or schema.model_id,
    )
    return build_client(cfg, FLAGS)


def _to_prompt(sections: list[PromptSectionSchema]) -> Prompt:
    return Prompt([PromptSection(text=s.text, mutable=s.mutable) for s in sections])


def _to_test_cases(schemas: list[TestCaseSchema]) -> list[TestCase]:
    return [TestCase(name=s.name, input_text=s.input_text, expected_output=s.expected_output) for s in schemas]


def _prompt_to_schema(prompt: Prompt) -> list[PromptSectionSchema]:
    return [PromptSectionSchema(text=s.text, mutable=s.mutable) for s in prompt.sections]


# -----------------------------------------------------------------------
# Webhook helper (4C)
# -----------------------------------------------------------------------

def _send_webhook(url: str, payload: dict) -> None:
    if not FLAGS.api_webhook_enabled or not url:
        return
    try:
        import httpx
        with httpx.Client(timeout=10.0) as client:
            client.post(url, json=payload)
    except Exception:
        pass  # best-effort


# -----------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------

@app.post("/optimize", response_model=OptimizeResponse, dependencies=[Depends(verify_api_key)])
def optimize_endpoint(request: OptimizeRequest, background_tasks: BackgroundTasks):
    run_id = str(uuid.uuid4())
    _storage.save_run(run_id, "running")

    prompt = _to_prompt(request.prompt_sections)
    test_cases = _to_test_cases(request.test_cases)
    target = _to_model_client(request.target_model)
    supervisor = _to_model_client(request.supervisor_model)
    cost_tracker = CostTracker() if FLAGS.cost_tracking_enabled else None

    def _run():
        try:
            history = run_optimization(
                prompt=prompt,
                test_cases=test_cases,
                target=target,
                supervisor=supervisor,
                epochs=request.epochs,
                max_iterations=request.max_iterations,
                flags=FLAGS,
            )
            _history_cache[run_id] = history
            result_data = {
                "pass_rates": history.pass_rates(),
                "epoch_count": len(history.epoch_results),
            }
            if cost_tracker:
                result_data["cost"] = cost_tracker.summary()
            _storage.save_run(run_id, "complete", result_data)

            # Webhook (4C)
            if request.webhook_url:
                _send_webhook(request.webhook_url, {
                    "run_id": run_id,
                    "status": "complete",
                    "pass_rates": history.pass_rates(),
                })
        except Exception as e:
            _storage.save_run(run_id, f"error: {e}")
            if request.webhook_url:
                _send_webhook(request.webhook_url, {
                    "run_id": run_id,
                    "status": f"error: {e}",
                })

    background_tasks.add_task(_run)
    return OptimizeResponse(run_id=run_id, message="Optimization started. Poll /history/{run_id} for progress.")


@app.post("/evaluate", response_model=EvaluateResponse, dependencies=[Depends(verify_api_key)])
def evaluate_endpoint(request: EvaluateRequest):
    supervisor = _to_model_client(request.supervisor_model)
    result = evaluate(
        target_output=request.actual_output,
        expected_output=request.expected_output,
        supervisor=supervisor,
    )
    return EvaluateResponse(passed=result.passed, feedback=result.feedback, reasoning=result.reasoning)


@app.get("/history/{run_id}", response_model=HistoryResponse, dependencies=[Depends(verify_api_key)])
def history_endpoint(run_id: str):
    entry = _storage.get_run(run_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Run not found")

    status = entry["status"]

    if status == "running":
        return HistoryResponse(run_id=run_id, status="running", epochs=[])

    if status.startswith("error"):
        return HistoryResponse(run_id=run_id, status=status, epochs=[])

    history = _history_cache.get(run_id)
    if not history:
        return HistoryResponse(
            run_id=run_id, status="complete",
            epochs=[], cost=entry.get("data", {}).get("cost"),
        )

    epochs = [
        EpochSummary(
            epoch=r.epoch,
            pass_count=r.pass_count,
            total_count=r.total_count,
            pass_rate=r.pass_rate,
            prompt_after=_prompt_to_schema(r.prompt_after),
        )
        for r in history.epoch_results
    ]
    final = _prompt_to_schema(history.final_prompt) if history.final_prompt else None
    cost = entry.get("data", {}).get("cost")
    return HistoryResponse(run_id=run_id, status="complete", epochs=epochs, final_prompt=final, cost=cost)


@app.get("/runs", dependencies=[Depends(verify_api_key)])
def list_runs():
    return _storage.list_runs()
