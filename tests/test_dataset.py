"""Dataset shape and contract test. Full implementation tested in Task 9."""
from __future__ import annotations

from src.data.dataset import DIRECTIONS, NUM_ATTACK_CLASSES, WindowDataset


def test_directions_constant():
    assert DIRECTIONS == ("IN", "OUT")


def test_num_attack_classes_constant():
    assert NUM_ATTACK_CLASSES == 7
