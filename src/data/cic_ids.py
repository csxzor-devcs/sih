"""CIC-IDS-2017 loader.

Produces a per-flow DataFrame with the canonical 8-class label and the
columns needed to derive the 13 flow + 5 packet-level features.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.preprocess import (
    RAW_TO_CANONICAL,
    canonicalize_label,
    clean_floats,
    filter_canonical_floats,
)

# 15 CIC-IDS-2017 raw labels → 8 canonical (0=BENIGN, 1..7 attacks).
# Re-exported under this name for the loader's public interface. Single source
# of truth is `preprocess.RAW_TO_CANONICAL`.
CIC_LABEL_CANONICAL: dict[str, int] = RAW_TO_CANONICAL

# Columns we read from each per-day CSV. Field names match the official
# CIC-IDS-2017 naming (with spaces). The exact list is TO VERIFY against the
# actual downloaded CSVs; some 2017 CSVs have minor naming variations.
FLOAT_COLS = [
    "Flow Duration",
    "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max",
    "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max",
    "Fwd Pkt Len Max", "Fwd Pkt Len Min", "Fwd Pkt Len Mean", "Fwd Pkt Len Std",
    "Bwd Pkt Len Max", "Bwd Pkt Len Min", "Bwd Pkt Len Mean", "Bwd Pkt Len Std",
    "Pkt Len Max", "Pkt Len Min", "Pkt Len Mean", "Pkt Len Std",
    "Flow IAT Mean", "Flow IAT Max", "Flow IAT Std",
    "Flow Bytes/s", "Flow Pkts/s",
    "Fwd Pkts/s", "Bwd Pkts/s",
    "Down/Up Ratio",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
    "Active Mean", "Active Std", "Active Max", "Active Min",
]

INT_COLS = [
    "Total Fwd Packet", "Total Bwd packet",
    "Total Fwd Bytes", "Total Bwd Bytes",
    "Fwd Header Len", "Bwd Header Len",
    "Fwd Act Data Pkts", "Bwd Act Data Pkts",
    "Fwd Seg Size Min",
    "Subflow Fwd Pkts", "Subflow Fwd Byts", "Subflow Bwd Pkts", "Subflow Bwd Byts",
    "Init Fwd Win Byts", "Init Bwd Win Byts",
    "min_seg_size_forward",
]

CATEGORICAL_COLS = ["Protocol", "Service", "Flag"]  # last is the state/flag

REQUIRED_COLS = ["Source IP", "Destination IP", "Timestamp", "Label"] + FLOAT_COLS + INT_COLS + CATEGORICAL_COLS


def load_cic_ids_csv(path: Path) -> pd.DataFrame:
    """Load one per-day CIC-IDS-2017 CSV.

    The columns in REQUIRED_COLS are read; the per-packet statistics are
    preserved on the per-flow DataFrame so that aggregate.py can derive the
    5 packet-level features per (host, bin, direction).
    """
    df = pd.read_csv(
        path,
        usecols=REQUIRED_COLS,
        low_memory=False,
        encoding="latin1",  # CIC-IDS-2017 uses Latin-1 in some per-day files
    )
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="mixed", dayfirst=False, errors="coerce")
    df = df.dropna(subset=["Timestamp", "Source IP", "Destination IP", "Label"])
    df = clean_floats(df, FLOAT_COLS)
    df = filter_canonical_floats(df, FLOAT_COLS)
    df["attack_label"] = df["Label"].map(canonicalize_label)
    return df
