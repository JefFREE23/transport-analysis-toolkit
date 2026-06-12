"""Reusable result containers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

import numpy as np


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return result_to_dict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def result_to_dict(result: Any) -> dict[str, Any]:
    """Convert a dataclass result into plain Python values."""

    if not is_dataclass(result):
        raise TypeError("result_to_dict expects a dataclass instance")
    return {key: _json_value(value) for key, value in asdict(result).items()}


@dataclass(slots=True)
class FitResult:
    """Generic result for model fitting."""

    parameters: dict[str, float]
    covariance: np.ndarray | None = None
    uncertainties: dict[str, float] = field(default_factory=dict)
    residual_norm: float | None = None
    success: bool = True
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return result_to_dict(self)


@dataclass(slots=True)
class Peak:
    """Peak identified in a one-dimensional distribution."""

    position: float
    amplitude: float
    index: int
    kind: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return result_to_dict(self)
