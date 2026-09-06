"""Pydantic schemas for the API.

The /predict payload must NOT have any field whose name is in the
forbidden_columns set. This is enforced by test_demo_no_leakage.py and
test_forbidden_columns_extended.py.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class WindowPayload(BaseModel):
    """Payload for /predict.

    features: a list of L=12 bins, each of F_entity features. IN/OUT concatenated
    per bin: layout is [bin_1_in, bin_1_out, bin_2_in, bin_2_out, ...] or
    [bin_1_full, bin_2_full, ...] where each bin row has F_entity columns.
    """
    host: str = Field(default="anonymous")
    direction: str = Field(default="BOTH")
    features: list[list[float]] = Field(..., description="[L, F_entity]")


class ForecastResponse(BaseModel):
    p_onset: dict[str, float]            # keys: "1", "3", "5"
    p_class: dict[str, list[float]]      # keys: "1", "3", "5" -> [7]
    p_attack_present: list[float]        # [8]
    model_version: str
    device: str
