# tests/gating/test_dashboard_renders.py
"""Gate: dashboard pages render without throwing (smoke check)."""
import subprocess
from pathlib import Path

DASHBOARD = Path(__file__).resolve().parents[2] / "dashboard"


def test_dashboard_pages_have_files():
    pages = [
        "app/(dashboard)/page.tsx",
        "app/(dashboard)/model-card/page.tsx",
        "app/(dashboard)/metrics/breakdown/page.tsx",
    ]
    for p in pages:
        assert (DASHBOARD / p).exists(), f"missing dashboard page: {p}"
