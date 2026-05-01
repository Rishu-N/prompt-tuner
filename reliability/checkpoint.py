"""Save and restore prompt state between epochs for crash recovery."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from core.prompt import Prompt

log = logging.getLogger(__name__)


class CheckpointManager:
    def __init__(self, checkpoint_dir: str = ".checkpoints"):
        self._dir = Path(checkpoint_dir)

    def save(self, run_id: str, epoch: int, prompt: Prompt) -> Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"{run_id}_epoch{epoch}.json"
        data = {"run_id": run_id, "epoch": epoch, "prompt": prompt.to_dict()}
        path.write_text(json.dumps(data, indent=2))
        log.info("Checkpoint saved: %s", path)
        return path

    def load_latest(self, run_id: str) -> tuple[int, Prompt] | None:
        if not self._dir.exists():
            return None
        files = sorted(self._dir.glob(f"{run_id}_epoch*.json"))
        if not files:
            return None
        data = json.loads(files[-1].read_text())
        prompt = Prompt.from_dict(data["prompt"])
        log.info("Checkpoint restored: epoch %d", data["epoch"])
        return data["epoch"], prompt

    def cleanup(self, run_id: str) -> None:
        if not self._dir.exists():
            return
        for f in self._dir.glob(f"{run_id}_epoch*.json"):
            f.unlink()
        log.info("Checkpoints cleaned up for run %s", run_id)
