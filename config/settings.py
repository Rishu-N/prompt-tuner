"""
Non-toggle application settings — default temperatures, pricing table, etc.
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class AppSettings:
    # Default temperatures used by core modules
    evaluator_temperature: float = 0.0
    optimizer_target_temperature: float = 0.7
    optimizer_supervisor_temperature: float = 0.0

    # Pricing (USD per 1M tokens) — used by CostTracker
    # Keys are model-id prefixes matched greedily
    pricing: dict[str, tuple[float, float]] = dataclasses.field(
        default_factory=lambda: {
            # (prompt_cost_per_1M, completion_cost_per_1M)
            "gpt-4o-mini": (0.15, 0.60),
            "gpt-4o": (2.50, 10.00),
            "gpt-4-turbo": (10.00, 30.00),
            "gpt-3.5-turbo": (0.50, 1.50),
            "claude-3-5-sonnet": (3.00, 15.00),
            "claude-3-5-haiku": (0.80, 4.00),
            "claude-sonnet-4": (3.00, 15.00),
            "claude-haiku-4": (0.80, 4.00),
        }
    )

    def get_pricing(self, model_id: str) -> tuple[float, float]:
        """Return (prompt_cost, completion_cost) per 1M tokens for a model."""
        for prefix, costs in sorted(self.pricing.items(), key=lambda x: -len(x[0])):
            if model_id.startswith(prefix):
                return costs
        return (0.0, 0.0)  # unknown model — free


DEFAULT_SETTINGS = AppSettings()
