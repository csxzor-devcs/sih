"""UNSW-NB15 loader and canonicalization tests (V3 cross-dataset prep)."""
from __future__ import annotations

from src.data.unsw_nb15 import UNSW_TO_CANONICAL, canonicalize_unsw_label, REQUIRED_COLS


def test_unsw_label_mapping_completeness():
    # All 9 UNSW-NB15 attack cats must map to either BENIGN or one of the 7 attack classes
    for unsw_cat in ["Normal", "Fuzzers", "Analysis", "Backdoors", "DoS",
                     "Exploits", "Generic", "Reconnaissance", "Shellcode", "Worms"]:
        mapped = canonicalize_unsw_label(unsw_cat)
        assert mapped in UNSW_TO_CANONICAL.values()


def test_unsw_required_columns_present():
    # Must have 18 scalar features per direction
    assert len(REQUIRED_COLS) == 18
