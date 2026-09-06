"""All 15 CIC-IDS-2017 raw labels map to one of 8 canonical labels."""
from __future__ import annotations

import pytest

from src.data.cic_ids import CIC_LABEL_CANONICAL
from src.data.preprocess import canonicalize_label, CANONICAL_LABELS, RAW_TO_CANONICAL


CANONICAL_NAMES = {
    0: "BENIGN",
    1: "BRUTE_FORCE",
    2: "DOS",
    3: "WEB_ATTACK",
    4: "INFILTRATION",
    5: "PORTSCAN",
    6: "BOTNET",
    7: "HEARTBLEED",
}


def test_benign_maps_to_zero():
    assert canonicalize_label("BENIGN") == 0


@pytest.mark.parametrize("raw,expected", [
    ("DoS Hulk", 2),
    ("DDoS", 2),
    ("DoS GoldenEye", 2),
    ("DoS Slowloris", 2),
    ("DoS Slowhttptest", 2),
    ("Heartbleed", 7),
    ("PortScan", 5),
    ("Bot", 6),
    ("Infiltration", 4),
    ("FTP-Patator", 1),
    ("SSH-Patator", 1),
    ("Web Attack - Brute Force", 3),
    ("Web Attack - XSS", 3),
    ("Web Attack - Sql Injection", 3),
])
def test_attack_labels_map_to_correct_class(raw: str, expected: int):
    assert canonicalize_label(raw) == expected


def test_unknown_label_raises():
    with pytest.raises(ValueError):
        canonicalize_label("NOT_A_REAL_LABEL")


def test_raw_to_canonical_has_15_entries():
    """The known CIC-IDS-2017 raw labels total 15 (BENIGN + 14 attack types)."""
    assert len(RAW_TO_CANONICAL) == 15


def test_canonical_labels_match_8_class_definitions():
    assert CANONICAL_LABELS == CANONICAL_NAMES


def test_cic_ids_label_mapping_matches_preprocess():
    """cic_ids.CIC_LABEL_CANONICAL and preprocess.RAW_TO_CANONICAL must agree."""
    assert CIC_LABEL_CANONICAL == RAW_TO_CANONICAL
