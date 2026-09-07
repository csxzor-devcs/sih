"""Tier 3 minimum gates.

These tests fail only if a Tier 3 component is REFERENCED in docs/README
but missing in code. We don't require Tier 3 to be complete to merge;
we only require that claimed features exist.
"""
from pathlib import Path
import re


REPO = Path(__file__).resolve().parents[2]


def _read(p: Path) -> str:
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8", errors="ignore")


def test_unsw_nb15_loader_exists_if_v3_claimed():
    readme = _read(REPO / "README.md")
    if "V3" in readme and "UNSW" in readme:
        assert (REPO / "src" / "data" / "unsw_nb15.py").exists()


def test_dkf_transition_exists_if_claimed():
    readme = _read(REPO / "README.md")
    if "DKF" in readme:
        assert (REPO / "src" / "model" / "dkf.py").exists()


def test_whatif_simulator_exists_if_claimed():
    readme = _read(REPO / "README.md")
    if "What-If" in readme or "simulator" in readme.lower():
        assert (REPO / "dashboard" / "app" / "(dashboard)" / "simulator" / "page.tsx").exists()
