"""Merge-mutable updater — the original strategy. All mutable sections become one block."""
from __future__ import annotations

from core.prompt import Prompt
from utils.registry import updater_registry


class MergeMutableUpdater:
    def apply_update(self, prompt: Prompt, new_mutable_text: str) -> Prompt:
        return prompt.apply_mutable_update(new_mutable_text)


updater_registry.register("merge_mutable", MergeMutableUpdater)
