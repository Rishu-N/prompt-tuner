"""Save and load full sessions (prompt, test cases, model configs) as JSON."""
import json
from pathlib import Path

from core.models import ModelConfig
from core.prompt import Prompt
from core.test_case import TestCase


def save_session(
    path: str | Path,
    prompt: Prompt,
    test_cases: list[TestCase],
    target_config: ModelConfig,
    supervisor_config: ModelConfig,
    epochs: int = 5,
    max_iterations: int = 3,
) -> None:
    data = {
        "prompt": prompt.to_dict(),
        "test_cases": [tc.to_dict() for tc in test_cases],
        "target_config": target_config.to_dict(),
        "supervisor_config": supervisor_config.to_dict(),
        "epochs": epochs,
        "max_iterations": max_iterations,
    }
    Path(path).write_text(json.dumps(data, indent=2))


def load_session(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text())
    return {
        "prompt": Prompt.from_dict(data["prompt"]),
        "test_cases": [TestCase.from_dict(tc) for tc in data["test_cases"]],
        "target_config": ModelConfig.from_dict(data["target_config"]),
        "supervisor_config": ModelConfig.from_dict(data["supervisor_config"]),
        "epochs": data.get("epochs", 5),
        "max_iterations": data.get("max_iterations", 3),
    }
