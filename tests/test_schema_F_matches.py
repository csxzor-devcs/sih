"""F_entity must come from the schema, not from a hard-coded number."""
from __future__ import annotations

from src.config import (
    F_per_direction,
    F_per_direction_fixed_scalars,
    F_entity,
    load_schema,
)


def test_F_per_direction_fixed_scalars_is_18():
    schema = load_schema()
    assert F_per_direction_fixed_scalars(schema) == 18


def test_F_per_direction_includes_histograms():
    schema = load_schema()
    V_p, V_s, V_t = 17, 12, 5
    # 18 scalars + 17 + 12 + 5 = 52 per direction
    assert F_per_direction(schema, V_p, V_s, V_t) == 52


def test_F_entity_is_2x_per_direction():
    schema = load_schema()
    V_p, V_s, V_t = 17, 12, 5
    assert F_entity(schema, V_p, V_s, V_t) == 2 * 52 == 104


def test_schema_example_F_matches():
    """The example in the schema (Vp=17, Vs=12, Vt=5) must match derivation."""
    schema = load_schema()
    V_p, V_s, V_t = 17, 12, 5
    assert F_entity(schema, V_p, V_s, V_t) == 2 * (18 + 17 + 12 + 5)
