"""
Per-section updater — updates each mutable section independently.
Expects new_mutable_text to contain sections separated by '---SECTION N---' delimiters.
Falls back to merge strategy if delimiters are missing.
"""
from __future__ import annotations

import re
import copy

from core.prompt import Prompt, PromptSection
from utils.registry import updater_registry

_SECTION_DELIMITER = re.compile(r"---SECTION\s+\d+---\s*\n?")


class PerSectionUpdater:
    def apply_update(self, prompt: Prompt, new_mutable_text: str) -> Prompt:
        mutable_indices = [i for i, s in enumerate(prompt.sections) if s.mutable]
        if not mutable_indices:
            return prompt.clone()

        # Split on delimiters
        parts = _SECTION_DELIMITER.split(new_mutable_text)
        parts = [p.strip() for p in parts if p.strip()]

        # Fallback: if delimiter count doesn't match, use merge strategy
        if len(parts) != len(mutable_indices):
            return prompt.apply_mutable_update(new_mutable_text)

        new_sections = []
        mutable_idx = 0
        for i, s in enumerate(prompt.sections):
            if s.mutable:
                new_sections.append(PromptSection(text=parts[mutable_idx], mutable=True))
                mutable_idx += 1
            else:
                new_sections.append(copy.deepcopy(s))

        return Prompt(new_sections)


updater_registry.register("per_section", PerSectionUpdater)
