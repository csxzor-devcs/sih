"""Cleaning, label canonicalization, and shared schema constants."""
from __future__ import annotations

import numpy as np
import pandas as pd

# 15 CIC-IDS-2017 raw labels → 8 canonical (0=BENIGN, 1..7 attacks)
RAW_TO_CANONICAL: dict[str, int] = {
    "BENIGN": 0,
    "DoS Hulk": 2,
    "DDoS": 2,
    "DoS GoldenEye": 2,
    "DoS Slowloris": 2,
    "DoS Slowhttptest": 2,
    "Heartbleed": 7,
    "PortScan": 5,
    "Bot": 6,
    "Infiltration": 4,
    "FTP-Patator": 1,
    "SSH-Patator": 1,
    "Web Attack - Brute Force": 3,
    "Web Attack - XSS": 3,
    "Web Attack - Sql Injection": 3,
}

CANONICAL_LABELS: dict[int, str] = {
    0: "BENIGN",
    1: "BRUTE_FORCE",
    2: "DOS",
    3: "WEB_ATTACK",
    4: "INFILTRATION",
    5: "PORTSCAN",
    6: "BOTNET",
    7: "HEARTBLEED",
}

NUM_CLASSES: int = 8
NUM_ATTACK_CLASSES: int = 7  # p_class head output dimension
ATTACK_CLASS_INDICES: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)


def canonicalize_label(raw: str) -> int:
    """Map a raw CIC-IDS-2017 label string to its canonical class index 0..7."""
    if raw not in RAW_TO_CANONICAL:
        raise ValueError(f"Unknown raw label: {raw!r}. Known labels: {sorted(RAW_TO_CANONICAL)}")
    return RAW_TO_CANONICAL[raw]


def clean_floats(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Replace ±inf with NaN, then fill NaN with the column median. Returns a new DataFrame."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].replace([np.inf, -np.inf], np.nan)
            median = out[c].median()
            if pd.isna(median):
                median = 0.0
            out[c] = out[c].fillna(median)
    return out


def filter_canonical_floats(df: pd.DataFrame, cols: list[str], max_value: float = 1e12) -> pd.DataFrame:
    """Cap absurdly large float values that arise from division by zero in CIC CSV."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].clip(lower=-max_value, upper=max_value)
    return out
