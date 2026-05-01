from __future__ import annotations

from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from core.prompt import Prompt


class PromptUpdaterProtocol(Protocol):
    def apply_update(self, prompt: Prompt, new_mutable_text: str) -> Prompt: ...
