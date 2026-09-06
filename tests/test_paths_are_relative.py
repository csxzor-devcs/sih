"""No absolute paths in the source tree. All paths via env vars or relative to project root."""
from __future__ import annotations

import re
from pathlib import Path

ABS_PATH_PATTERNS = [
    re.compile(r"['\"\/]/home/[^'\"]+['\"]?"),
    re.compile(r"['\"\/]/Users/[^'\"]+['\"]?"),
    re.compile(r"['\"]C:\\\\[^'\"]+['\"]?"),
]


def test_no_absolute_paths_in_source():
    project = Path(__file__).resolve().parents[1]
    for py in (project / "src").rglob("*.py"):
        text = py.read_text()
        for line in text.splitlines():
            for pat in ABS_PATH_PATTERNS:
                if pat.search(line):
                    raise AssertionError(f"Absolute path in {py}: {line!r}")
