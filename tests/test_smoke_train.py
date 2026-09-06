"""Smoke training run produces a report with all 4 losses and F_entity."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_smoke_train_produces_report(tmp_path: Path):
    # PYTHONPATH must point at the project root (the parent of `src/`) so that
    # `from src.config import ...` resolves. The brief's `... / "src"` form
    # would point at the package directory itself and break the import.
    project_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        ["python", "scripts/smoke_train.py"],
        check=True,
        cwd=project_root,
        env={"PYTHONPATH": str(project_root), "PATH": __import__("os").environ["PATH"]},
    )
    report_path = Path(__file__).resolve().parents[1] / "artifacts" / "smoke_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    for k in ["L_onset", "L_class", "L_present", "L_transition", "L_total", "F_entity"]:
        assert k in report
    assert report["F_entity"] == 2 * (18 + 17 + 12 + 5)
