"""Track prompt versions across optimization epochs with diffs."""
from __future__ import annotations

import difflib
import time
from dataclasses import dataclass, field

from core.prompt import Prompt


@dataclass
class PromptVersion:
    version: int
    prompt: Prompt
    epoch: int
    pass_rate: float
    timestamp: float
    diff_from_previous: str = ""


class PromptVersionHistory:
    def __init__(self):
        self._versions: list[PromptVersion] = []

    def record(self, prompt: Prompt, epoch: int, pass_rate: float) -> PromptVersion:
        version_num = len(self._versions) + 1
        diff = ""
        if self._versions:
            prev = self._versions[-1].prompt.render().splitlines(keepends=True)
            curr = prompt.render().splitlines(keepends=True)
            diff = "".join(difflib.unified_diff(prev, curr, fromfile=f"v{version_num-1}", tofile=f"v{version_num}"))
        v = PromptVersion(
            version=version_num,
            prompt=prompt.clone(),
            epoch=epoch,
            pass_rate=pass_rate,
            timestamp=time.time(),
            diff_from_previous=diff,
        )
        self._versions.append(v)
        return v

    def get_version(self, version: int) -> PromptVersion | None:
        if 1 <= version <= len(self._versions):
            return self._versions[version - 1]
        return None

    def diff(self, v1: int, v2: int) -> str:
        a = self.get_version(v1)
        b = self.get_version(v2)
        if not a or not b:
            return ""
        lines_a = a.prompt.render().splitlines(keepends=True)
        lines_b = b.prompt.render().splitlines(keepends=True)
        return "".join(difflib.unified_diff(lines_a, lines_b, fromfile=f"v{v1}", tofile=f"v{v2}"))

    def rollback_to(self, version: int) -> Prompt | None:
        v = self.get_version(version)
        return v.prompt.clone() if v else None

    def all_versions(self) -> list[PromptVersion]:
        return list(self._versions)
