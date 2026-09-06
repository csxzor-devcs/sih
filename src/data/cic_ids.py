"""CIC-IDS-2017 CSV loader. Maps the 15 raw CSV labels to 8 canonical classes."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# 15 CIC-IDS-2017 raw labels -> 8 canonical (0=BENIGN, 1..7 attacks).
# Kept here (not in preprocess.py) so the loader is self-contained; preprocess.py
# re-exports the same mapping for canonicalize_label().
CIC_LABEL_CANONICAL: dict[str, int] = {
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


def load_cic_ids_csv(path: Path) -> pd.DataFrame:
    """Load a CIC-IDS-2017 CSV and return a canonical-schema DataFrame.

    The CSV's final column is conventionally named "Label" and contains one of the
    15 raw label strings. We rename it to "attack_label" and add a "y_present"
    binary target (1 if not BENIGN, 0 otherwise). The returned DataFrame is the
    raw per-flow rows; downstream code is responsible for binning into windows
    and assembling the per-entity features.
    """
    df = pd.read_csv(path)
    # CIC CSVs vary in capitalization and trailing whitespace of the Label column.
    label_col = None
    for cand in ("Label", "label", " Label", " Label\n"):
        if cand in df.columns:
            label_col = cand
            break
    if label_col is None:
        raise ValueError(
            f"No label column found in {path}. Expected one of: 'Label'."
        )
    df = df.rename(columns={label_col: "attack_label"})
    # Strip whitespace just in case.
    df["attack_label"] = df["attack_label"].astype(str).str.strip()
    df["y_present"] = (df["attack_label"] != "BENIGN").astype("int8")
    return df
