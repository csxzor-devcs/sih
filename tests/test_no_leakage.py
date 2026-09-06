"""Label-derived columns never appear in any input tensor."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import forbidden_columns, load_schema


def test_input_df_has_no_label_columns():
    """A synthetic input DataFrame must have no forbidden column names."""
    schema = load_schema()
    forbidden = forbidden_columns(schema)
    df = pd.DataFrame({
        "flow_count": np.zeros(10, dtype=np.int32),
        "total_bytes": np.zeros(10, dtype=np.int64),
        "byte_rate": np.zeros(10, dtype=np.float32),
    })
    overlap = set(df.columns) & forbidden
    assert overlap == set(), f"Input DataFrame has forbidden columns: {overlap}"


def test_label_columns_are_labeled_targets():
    """The label columns listed in the schema must be target-only."""
    schema = load_schema()
    label_cols = schema["label_columns"]
    forbidden = forbidden_columns(schema)
    for col, role in label_cols.items():
        assert "TARGET" in role, f"{col} role must say 'TARGET': {role!r}"
        assert col in forbidden, f"Label column {col!r} must be in forbidden_columns"
