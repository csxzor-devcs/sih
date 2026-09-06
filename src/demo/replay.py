"""Replay precomputed predictions at 5x speed. Polled at 60 s by the dashboard."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import artifacts_dir


class Replay:
    def __init__(self, speed: float = 5.0, path: Path | None = None):
        self.speed = speed
        self.path = path or (artifacts_dir() / "demo" / "predictions.parquet")
        if not self.path.exists():
            self.df = None
        else:
            self.df = pd.read_parquet(self.path)
        self.started_at = time.time()

    def current_row(self) -> dict[str, Any] | None:
        if self.df is None or len(self.df) == 0:
            return None
        elapsed = time.time() - self.started_at
        idx = int((elapsed * self.speed)) % len(self.df)
        return self.df.iloc[idx].to_dict()
