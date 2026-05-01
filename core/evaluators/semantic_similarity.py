"""Embedding-based semantic similarity evaluator."""
from __future__ import annotations

import math

from core.models import ModelClient
from core.evaluator import EvalResult
from utils.registry import evaluator_registry


class SemanticSimilarityEvaluator:
    def __init__(
        self,
        threshold: float = 0.85,
        embedding_model: str = "text-embedding-3-small",
    ):
        self.threshold = threshold
        self.embedding_model = embedding_model

    def evaluate(
        self,
        target_output: str,
        expected_output: str,
        *,
        supervisor: ModelClient | None = None,
    ) -> EvalResult:
        if supervisor is None:
            raise ValueError("SemanticSimilarityEvaluator requires a supervisor (for its API client)")

        client = supervisor._client  # access the underlying OpenAI client

        resp = client.embeddings.create(
            model=self.embedding_model,
            input=[target_output, expected_output],
        )
        vec_a = resp.data[0].embedding
        vec_b = resp.data[1].embedding
        similarity = _cosine_similarity(vec_a, vec_b)

        if similarity >= self.threshold:
            return EvalResult(
                passed=True,
                feedback="",
                reasoning=f"Cosine similarity {similarity:.3f} >= threshold {self.threshold}",
            )
        return EvalResult(
            passed=False,
            feedback=f"Semantic similarity too low ({similarity:.3f} < {self.threshold})",
            reasoning=f"Embeddings diverge — output may miss key concepts.",
        )


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


evaluator_registry.register("semantic", SemanticSimilarityEvaluator)
