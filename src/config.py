"""Configuration loading. Schema is the single source of truth for F."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    return Path(os.environ.get("SIH_DATA_DIR", str(project_root() / "data")))


def artifacts_dir() -> Path:
    return Path(os.environ.get("SIH_ARTIFACTS_DIR", str(project_root() / "artifacts")))


def load_config(path: str | None = None) -> dict[str, Any]:
    """Load YAML config. Relative paths resolved against project root."""
    p = Path(path) if path else project_root() / "configs" / "default.yaml"
    if not p.is_absolute():
        p = project_root() / p
    with open(p) as f:
        return yaml.safe_load(f)


def load_schema(path: str | None = None) -> dict[str, Any]:
    """Load JSON feature schema. The schema is the source of truth for F."""
    if path is None:
        cfg = load_config()
        path = cfg["data"]["schema_path"]
    p = Path(path)
    if not p.is_absolute():
        p = project_root() / p
    with open(p) as f:
        return json.load(f)


def feature_names_per_direction(schema: dict[str, Any]) -> list[str]:
    return [f["name"] for f in schema["features_per_direction"]]


def forbidden_columns(schema: dict[str, Any]) -> set[str]:
    return set(schema["forbidden_columns"])


def F_per_direction_fixed_scalars(schema: dict[str, Any]) -> int:
    """18 fixed scalars: 13 flow + 5 packet-derived. Histograms add V_p+V_s+V_t separately."""
    return sum(
        1
        for f in schema["features_per_direction"]
        if f["type"] not in ("int32_Vp", "int32_Vs", "int32_Vt")
    )


def F_per_direction(schema: dict[str, Any], V_p: int, V_s: int, V_t: int) -> int:
    """Total features per direction = scalars + histograms."""
    return F_per_direction_fixed_scalars(schema) + V_p + V_s + V_t


def F_entity(schema: dict[str, Any], V_p: int, V_s: int, V_t: int) -> int:
    """Total features per entity (IN+OUT concatenated) = 2 * F_per_direction."""
    return 2 * F_per_direction(schema, V_p, V_s, V_t)


def bin_size_seconds(config: dict[str, Any]) -> int:
    return int(config["data"]["bin_size_seconds"])


def window_size_seconds(config: dict[str, Any]) -> int:
    return int(config["data"]["window_size_seconds"])


def sequence_length(config: dict[str, Any]) -> int:
    """L = number of bins per window."""
    return int(config["data"]["sequence_length"])


def L_transition_weight(config: dict[str, Any]) -> float:
    return float(config["training"]["L_transition_weight"])
