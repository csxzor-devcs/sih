"""/predict has no path to any ground-truth store. Static structural check."""
from __future__ import annotations

import re
from pathlib import Path

from src.config import forbidden_columns, load_schema


def test_predict_handler_no_label_import():
    """The /predict handler must not import any module whose name contains label/ground_truth/gt."""
    project = Path(__file__).resolve().parents[1]
    api_main = (project / "src" / "api" / "main.py").read_text()
    for tok in re.findall(r"^\s*(?:from|import)\s+([^\s]+)", api_main, re.MULTILINE):
        low = tok.lower()
        assert "label" not in low, f"/predict handler imports a label-related module: {tok}"
        assert "ground_truth" not in low
        assert ".gt" not in low


def test_window_payload_no_forbidden_field_names():
    from src.api.schemas import WindowPayload
    schema = load_schema()
    forbidden = forbidden_columns(schema)
    for fname in WindowPayload.model_fields:
        assert fname not in forbidden, f"WindowPayload field {fname!r} is in forbidden_columns"


def test_window_payload_no_label_like_field_names():
    from src.api.schemas import WindowPayload
    for fname in WindowPayload.model_fields:
        low = fname.lower()
        assert "label" not in low
        assert "ground_truth" not in low
        assert "gt" not in low
        assert "future" not in low
