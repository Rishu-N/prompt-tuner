"""
Central feature flag system. Every toggleable feature in the project is
controlled through this frozen dataclass.

Load order:  hardcoded defaults  →  config.yaml  →  env vars (PO_FEATURE_* prefix)
"""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any


@dataclasses.dataclass(frozen=True)
class FeatureFlags:
    # ── Core strategy selection ──────────────────────────────────────────
    evaluator_strategy: str = "llm_judge"
    optimizer_strategy: str = "discussion_loop"
    updater_strategy: str = "merge_mutable"

    # ── Reliability ──────────────────────────────────────────────────────
    retry_enabled: bool = False
    retry_max_attempts: int = 3
    retry_base_delay: float = 1.0
    cache_enabled: bool = False
    cache_max_size: int = 256
    checkpoint_enabled: bool = False
    checkpoint_dir: str = ".checkpoints"
    cost_tracking_enabled: bool = False
    timeout_seconds: float = 120.0

    # ── UI ───────────────────────────────────────────────────────────────
    ui_thread_safe_state: bool = True
    ui_auto_poll: bool = False
    ui_auto_poll_interval: float = 3.0
    ui_input_validation: bool = False
    ui_export_enabled: bool = False
    ui_cost_dashboard: bool = False
    ui_color_preview: bool = False

    # ── API ──────────────────────────────────────────────────────────────
    api_auth_enabled: bool = False
    api_auth_key: str = ""
    api_persistent_storage: bool = False
    api_storage_backend: str = "memory"
    api_sqlite_path: str = "runs.db"
    api_webhook_enabled: bool = False
    api_run_ttl_seconds: int = 0

    # ── Advanced ─────────────────────────────────────────────────────────
    ab_comparison_enabled: bool = False
    version_history_enabled: bool = False
    multi_model_eval_enabled: bool = False
    convergence_detection_enabled: bool = False
    convergence_patience: int = 3
    test_suite_tags_enabled: bool = False


_ENV_PREFIX = "PO_FEATURE_"


def _coerce(value: str, field_type: type) -> Any:
    """Coerce a string env-var value to the target field type."""
    if field_type is bool:
        return value.lower() in ("true", "1", "yes")
    if field_type is int:
        return int(value)
    if field_type is float:
        return float(value)
    return value


def load_feature_flags(config_path: str | Path | None = "config.yaml") -> FeatureFlags:
    """
    Build a FeatureFlags instance.
    Priority: env vars  >  config.yaml  >  hardcoded defaults.
    """
    overrides: dict[str, Any] = {}

    # 1. Load from config.yaml if it exists
    if config_path:
        p = Path(config_path)
        if p.exists():
            try:
                import yaml
                raw = yaml.safe_load(p.read_text()) or {}
                features = raw.get("features", raw)
                if isinstance(features, dict):
                    overrides.update(features)
            except ImportError:
                # PyYAML not installed — try JSON fallback for .yaml files
                import json
                try:
                    overrides.update(json.loads(p.read_text()))
                except Exception:
                    pass

    # 2. Load from env vars  (PO_FEATURE_RETRY_ENABLED=true  →  retry_enabled=True)
    field_types = {f.name: f.type for f in dataclasses.fields(FeatureFlags)}
    for key, value in os.environ.items():
        if key.startswith(_ENV_PREFIX):
            field_name = key[len(_ENV_PREFIX):].lower()
            if field_name in field_types:
                overrides[field_name] = _coerce(value, field_types[field_name])

    return FeatureFlags(**overrides)
