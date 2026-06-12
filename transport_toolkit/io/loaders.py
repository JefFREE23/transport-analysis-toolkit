"""Data loading and column normalization."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


FIELD_ALIASES = {
    "b",
    "bt",
    "field",
    "fieldt",
    "fieldtesla",
    "magneticfield",
    "magneticfieldt",
    "mu0h",
}
RXX_ALIASES = {
    "rxx",
    "rxxohm",
    "rhoxx",
    "rhoxxohm",
    "rho_xx",
    "longitudinal",
    "longitudinalresistance",
    "resistancexx",
}
RXY_ALIASES = {
    "rxy",
    "rxyohm",
    "rhoxy",
    "rhoxyohm",
    "rho_xy",
    "hall",
    "hallresistance",
    "transverse",
}


@dataclass(slots=True)
class TransportData:
    """Loaded transport table plus import metadata."""

    frame: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)


def _normalize_column_name(name: str) -> str:
    return "".join(char for char in str(name).lower() if char.isalnum())


def detect_delimiter(path: str | Path, sample_size: int = 4096) -> str | None:
    """Detect a delimiter for CSV/TXT-like files."""

    text = Path(path).read_text(encoding="utf-8", errors="ignore")[:sample_size]
    if not text.strip():
        raise ValueError(f"{path} is empty")
    try:
        dialect = csv.Sniffer().sniff(text, delimiters=",\t; ")
    except csv.Error:
        suffix = Path(path).suffix.lower()
        if suffix == ".csv":
            return ","
        if "\t" in text:
            return "\t"
        return None
    return dialect.delimiter


def detect_columns(columns: list[str]) -> dict[str, str]:
    """Map source columns to canonical names B, Rxx, and Rxy when present."""

    mapping: dict[str, str] = {}
    for column in columns:
        normalized = _normalize_column_name(column)
        if normalized in FIELD_ALIASES and "B" not in mapping:
            mapping["B"] = column
        elif normalized in RXX_ALIASES and "Rxx" not in mapping:
            mapping["Rxx"] = column
        elif normalized in RXY_ALIASES and "Rxy" not in mapping:
            mapping["Rxy"] = column
    return mapping


def _read_table(path: Path, delimiter: str | None) -> pd.DataFrame:
    if delimiter is None or delimiter == " ":
        return pd.read_csv(path, sep=r"\s+", engine="python", comment="#")
    return pd.read_csv(path, sep=delimiter, engine="python", comment="#")


def load_data(
    path: str | Path,
    *,
    delimiter: str | None = None,
    required: tuple[str, ...] = ("B", "Rxx", "Rxy"),
    return_metadata: bool = False,
) -> pd.DataFrame | TransportData:
    """Load transport data from CSV or text and normalize common columns.

    The returned DataFrame stores import details in ``DataFrame.attrs`` under
    ``"transport_metadata"``.
    """

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)

    detected_delimiter = delimiter if delimiter is not None else detect_delimiter(source)
    frame = _read_table(source, detected_delimiter)
    frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if frame.empty:
        raise ValueError(f"{source} did not contain tabular data")

    original_columns = list(frame.columns)
    column_map = detect_columns(original_columns)
    missing = [name for name in required if name not in column_map]
    if missing:
        raise ValueError(
            "missing required columns "
            + ", ".join(missing)
            + f"; found columns: {', '.join(map(str, original_columns))}"
        )

    rename_map = {source_name: canonical for canonical, source_name in column_map.items()}
    frame = frame.rename(columns=rename_map)
    for column in column_map:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    ordered = [name for name in ("B", "Rxx", "Rxy") if name in frame.columns]
    ordered.extend(column for column in frame.columns if column not in ordered)
    frame = frame.loc[:, ordered].reset_index(drop=True)

    metadata = {
        "source": str(source),
        "delimiter": detected_delimiter,
        "column_map": column_map,
        "original_columns": [str(column) for column in original_columns],
    }
    frame.attrs["transport_metadata"] = metadata
    if return_metadata:
        return TransportData(frame=frame, metadata=metadata)
    return frame
