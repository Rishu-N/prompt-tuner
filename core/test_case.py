from dataclasses import dataclass, field
import uuid


@dataclass
class TestCase:
    input_text: str
    expected_output: str
    name: str = ""
    tags: list[str] = field(default_factory=list)
    priority: int = 1      # 1=low, 2=medium, 3=high
    weight: float = 1.0    # for weighted pass rate

    def __post_init__(self):
        if not self.name:
            self.name = f"test_{uuid.uuid4().hex[:6]}"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "input_text": self.input_text,
            "expected_output": self.expected_output,
            "tags": self.tags,
            "priority": self.priority,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TestCase":
        return cls(
            name=d.get("name", ""),
            input_text=d["input_text"],
            expected_output=d["expected_output"],
            tags=d.get("tags", []),
            priority=d.get("priority", 1),
            weight=d.get("weight", 1.0),
        )
