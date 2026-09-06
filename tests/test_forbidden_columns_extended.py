"""Forbidden column enforcement: schema, loader, tensor, integration layers."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.config import feature_names_per_direction, forbidden_columns, load_schema


def test_schema_features_disjoint_from_forbidden():
    """Layer 1: schema's features_per_direction names must not intersect forbidden set."""
    schema = load_schema()
    feats = set(feature_names_per_direction(schema))
    forbidden = forbidden_columns(schema)
    overlap = feats & forbidden
    assert overlap == set(), f"Schema features overlap forbidden set: {overlap}"


def test_forbidden_columns_contains_all_label_variants():
    """The forbidden set must explicitly include all label variants."""
    schema = load_schema()
    forbidden = forbidden_columns(schema)
    required = {
        "attack_label", "label", "Label", "attack_cat",
        "y_present", "y_onset", "y_class",
        "true_label", "true_onset", "true_class",
        "ground_truth", "future_attack", "future_class", "future_label",
    }
    missing = required - forbidden
    assert missing == set(), f"Missing from forbidden_columns: {missing}"


def test_no_label_name_in_source_tree():
    """Layer 4: no assignment to a tensor from a variable matching the forbidden pattern."""
    project = Path(__file__).resolve().parents[1]
    src = project / "src"
    # Only check that source files don't accidentally have a feature-assignment
    # of the form `tensor = ... label` or `tensor[:, label_col]`.
    # This is a coarse check; the per-loader tests are tighter.
    suspicious = re.compile(r"\b(attack_label|true_label|true_onset|true_class|ground_truth)\b")
    for py in src.rglob("*.py"):
        text = py.read_text()
        # Allow these names to appear ONLY in:
        #  - comments
        #  - string literals
        #  - the schema/config file references
        # We strip those out and check the remainder.
        lines = []
        for line in text.splitlines():
            stripped = line.split("#", 1)[0]
            if stripped.strip().startswith(('"""', "'''")):
                continue
            # naive: drop inline string content
            stripped = re.sub(r"['\"][^'\"]*['\"]", "", stripped)
            lines.append(stripped)
        cleaned = "\n".join(lines)
        for m in suspicious.findall(cleaned):
            pytest.fail(f"Found suspicious reference in {py}: {m!r}")
