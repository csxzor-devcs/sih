"""Dataset shape and contract test. Full implementation tested in Task 9."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import F_entity, F_per_direction, load_config, load_schema
from src.data.dataset import DIRECTIONS, NUM_ATTACK_CLASSES, WindowDataset, collate


def test_directions_constant():
    assert DIRECTIONS == ("IN", "OUT")


def test_num_attack_classes_constant():
    assert NUM_ATTACK_CLASSES == 7


def test_collate_importable():
    from src.data.dataset import collate as _collate

    assert callable(_collate)


def _make_minimal_agg(n_bins: int = 12) -> pd.DataFrame:
    """Tiny synthetic per-(host, bin, direction) agg with 1 host, 1 direction."""
    rows = []
    for b in range(n_bins):
        rows.append({
            "host": "host_0",
            "bin": b,
            "direction": "IN",
            "flow_count": 0,
            "total_bytes": 0,
            "total_packets": 0,
            "byte_rate": 0.0,
            "packet_rate": 0.0,
            "dur_mean": 0.0, "dur_std": 0.0, "dur_p99": 0.0,
            "iat_mean": 0.0, "iat_std": 0.0, "iat_max": 0.0,
            "unique_peer_ports": 0, "unique_peer_ips": 0,
            "pkt_size_mean": 0.0, "pkt_size_std": 0.0, "pkt_size_p99": 0.0,
            "fwd_bwd_pkt_ratio": 0.0, "small_pkt_frac": 0.0,
        })
    return pd.DataFrame(rows)


def test_window_dataset_has_required_surface():
    cfg = load_config()
    schema = load_schema()
    V_p = int(schema["vocabularies"]["protocols"]["max_size"])
    V_s = int(schema["vocabularies"]["services"]["max_size"])
    V_t = int(schema["vocabularies"]["tcp_states"]["max_size"])

    agg = _make_minimal_agg(n_bins=12)
    vocab: dict[str, dict[str, int]] = {"protocols": {}, "services": {}, "tcp_states": {}}
    ds = WindowDataset(agg=agg, config=cfg, scaler=None, schema=schema, vocab=vocab)

    assert hasattr(ds, "F_entity")
    assert hasattr(ds, "__getitem__")
    assert len(ds) == 1
    expected = 2 * F_per_direction(schema, V_p, V_s, V_t)
    assert ds.F_entity == expected
    assert ds.F_entity == F_entity(schema, V_p, V_s, V_t)
