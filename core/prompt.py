from dataclasses import dataclass, field
from typing import Iterator
import copy


@dataclass
class PromptSection:
    text: str
    mutable: bool = True

    def to_dict(self) -> dict:
        return {"text": self.text, "mutable": self.mutable}

    @classmethod
    def from_dict(cls, d: dict) -> "PromptSection":
        return cls(**d)


class Prompt:
    """
    A prompt made of ordered sections, each marked mutable or immutable.
    Only mutable sections are modified during optimization.
    """

    def __init__(self, sections: list[PromptSection]):
        self.sections = sections

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> str:
        return "\n".join(s.text for s in self.sections)

    def render_annotated(self) -> str:
        """Return the prompt with [MUTABLE] / [IMMUTABLE] markers for display."""
        parts = []
        for s in self.sections:
            tag = "[MUTABLE]" if s.mutable else "[IMMUTABLE]"
            parts.append(f"{tag}\n{s.text}")
        return "\n\n".join(parts)

    def render_mutable_only(self) -> str:
        """Return only the mutable sections joined together."""
        return "\n".join(s.text for s in self.sections if s.mutable)

    def mutable_sections(self) -> list[PromptSection]:
        return [s for s in self.sections if s.mutable]

    def immutable_sections(self) -> list[PromptSection]:
        return [s for s in self.sections if not s.mutable]

    # ------------------------------------------------------------------
    # Updating
    # ------------------------------------------------------------------

    def apply_mutable_update(self, new_mutable_text: str) -> "Prompt":
        """
        Return a new Prompt where the mutable sections are replaced by
        `new_mutable_text` (treated as a single block replacing all
        mutable content in order).
        """
        new_sections = []
        mutable_replaced = False
        for s in self.sections:
            if s.mutable and not mutable_replaced:
                new_sections.append(PromptSection(text=new_mutable_text, mutable=True))
                mutable_replaced = True
            elif s.mutable:
                # skip additional mutable sections; they are merged into one
                continue
            else:
                new_sections.append(copy.deepcopy(s))
        if not mutable_replaced:
            new_sections.append(PromptSection(text=new_mutable_text, mutable=True))
        return Prompt(new_sections)

    def clone(self) -> "Prompt":
        return Prompt([copy.deepcopy(s) for s in self.sections])

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {"sections": [s.to_dict() for s in self.sections]}

    @classmethod
    def from_dict(cls, d: dict) -> "Prompt":
        return cls([PromptSection.from_dict(s) for s in d["sections"]])

    @classmethod
    def from_text(cls, text: str, all_mutable: bool = True) -> "Prompt":
        """Create a single-section prompt from plain text."""
        return cls([PromptSection(text=text, mutable=all_mutable)])

    # ------------------------------------------------------------------
    # Line-level helpers (used by UI)
    # ------------------------------------------------------------------

    @classmethod
    def from_lines_with_mutability(cls, lines: list[str], mutable_flags: list[bool]) -> "Prompt":
        """
        Build a Prompt from per-line mutability flags by grouping consecutive
        lines that share the same flag into a single section.
        """
        if not lines:
            return cls([])
        sections: list[PromptSection] = []
        current_lines = [lines[0]]
        current_mutable = mutable_flags[0]
        for line, flag in zip(lines[1:], mutable_flags[1:]):
            if flag == current_mutable:
                current_lines.append(line)
            else:
                sections.append(PromptSection(text="\n".join(current_lines), mutable=current_mutable))
                current_lines = [line]
                current_mutable = flag
        sections.append(PromptSection(text="\n".join(current_lines), mutable=current_mutable))
        return cls(sections)

    def to_lines_with_mutability(self) -> tuple[list[str], list[bool]]:
        """Expand sections back to per-line lists."""
        lines: list[str] = []
        flags: list[bool] = []
        for s in self.sections:
            for line in s.text.split("\n"):
                lines.append(line)
                flags.append(s.mutable)
        return lines, flags

    def __repr__(self) -> str:
        return f"Prompt({len(self.sections)} sections)"
