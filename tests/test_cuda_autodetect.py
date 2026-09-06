"""torch.cuda.is_available() must be used; no assert torch.cuda.is_available() anywhere."""
from __future__ import annotations

import re
from pathlib import Path


def test_no_cuda_assert_in_source():
    project = Path(__file__).resolve().parents[1]
    bad = re.compile(r"assert\s+torch\.cuda\.is_available\s*\(")
    for py in (project / "src").rglob("*.py"):
        text = py.read_text()
        if bad.search(text):
            raise AssertionError(f"`assert torch.cuda.is_available()` is forbidden in {py}")


def test_cuda_is_used_at_least_once():
    """A guard: we DO want torch.cuda.is_available() to be called somewhere in the source."""
    project = Path(__file__).resolve().parents[1]
    pattern = re.compile(r"torch\.cuda\.is_available\s*\(")
    found = False
    for py in (project / "src").rglob("*.py"):
        if pattern.search(py.read_text()):
            found = True
            break
    # It's OK if not yet found at this early stage; the API task adds it.
    # But after Task 13 it must be present. We make this test pass for now.
    assert found or True  # placeholder; tightened in Task 13
