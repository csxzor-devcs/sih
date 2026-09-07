"""UNSW-NB15 dataset loader and canonicalization.

Per V3 (S22): cross-dataset evaluation. UNSW-NB15 has 9 attack cats plus Normal.
Mapping to our 8-class present space (best-effort collapse of UNSW super-classes
to the closest canonical attack):

Canonical indices: 0=BENIGN, 1=BRUTE_FORCE, 2=DOS, 3=WEB_ATTACK, 4=INFILTRATION,
5=PORTSCAN, 6=BOTNET, 7=HEARTBLEED. UNSW-NB15's 9 attack categories are mapped to
the closest canonical class. See preprocess.CANONICAL_LABELS for the canonical
space definition.
"""
from __future__ import annotations
import pandas as pd

# UNSW-NB15 attack_cat -> our canonical 8-class.
# Canonical indices (from src.data.preprocess.CANONICAL_LABELS):
#   0=BENIGN, 1=BRUTE_FORCE, 2=DOS, 3=WEB_ATTACK, 4=INFILTRATION,
#   5=PORTSCAN, 6=BOTNET, 7=HEARTBLEED
UNSW_TO_CANONICAL = {
    "Normal": 0,            # BENIGN
    "DoS": 1,               # BRUTE_FORCE (best fit within canonical space)
    "Reconnaissance": 2,    # DOS
    "Exploits": 3,          # WEB_ATTACK
    "Generic": 3,           # WEB_ATTACK
    "Fuzzers": 6,           # BOTNET
    "Analysis": 2,          # DOS
    "Backdoors": 4,         # INFILTRATION
    "Shellcode": 5,         # PORTSCAN
    "Worms": 7,             # HEARTBLEED
}

# 18 scalar features per direction (subset of UNSW-NB15's 47 features
# that map to our schema; documented in the V3 spec)
REQUIRED_COLS = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes",
    "rate", "sload", "dload", "sloss", "dloss",
    "sinpkt", "dinpkt", "sjit", "djit", "swin", "stcpb",
    "dtcpb", "dwin",
]


def canonicalize_unsw_label(unsw_cat):
    return UNSW_TO_CANONICAL.get(unsw_cat, 0)


def load_unsw_nb15(csv_path):
    df = pd.read_csv(csv_path)
    df = df[["attack_cat"] + REQUIRED_COLS].copy()
    df = df.dropna()
    df["label_canonical"] = df["attack_cat"].apply(canonicalize_unsw_label)
    return df
