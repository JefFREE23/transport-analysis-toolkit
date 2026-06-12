"""Cleaning helpers for raw transport tables."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sort_by_field(frame: pd.DataFrame, field_column: str = "B") -> pd.DataFrame:
    """Return data sorted by magnetic field."""

    if field_column not in frame:
        raise KeyError(f"{field_column!r} column is required")
    return frame.sort_values(field_column).reset_index(drop=True)


def remove_nan_rows(frame: pd.DataFrame, columns: tuple[str, ...] = ("B", "Rxx", "Rxy")) -> pd.DataFrame:
    """Remove rows with missing values in required transport columns."""

    existing = [column for column in columns if column in frame.columns]
    return frame.dropna(subset=existing).reset_index(drop=True)


def remove_duplicate_fields(
    frame: pd.DataFrame,
    *,
    field_column: str = "B",
    method: str = "mean",
) -> pd.DataFrame:
    """Collapse duplicate magnetic-field rows."""

    if method not in {"mean", "first"}:
        raise ValueError("method must be 'mean' or 'first'")
    numeric_columns = frame.select_dtypes(include=[np.number]).columns.tolist()
    other_columns = [column for column in frame.columns if column not in numeric_columns]
    grouped = frame.groupby(field_column, as_index=False, sort=True)
    if method == "first":
        result = grouped.first()
    else:
        numeric = grouped[numeric_columns].mean()
        if other_columns:
            others = grouped[other_columns].first()
            result = pd.merge(numeric, others, on=field_column, how="left")
        else:
            result = numeric
    return result.reset_index(drop=True)


def remove_offset(
    values,
    field=None,
    *,
    field_range: tuple[float, float] | None = None,
    offset: float | None = None,
) -> np.ndarray:
    """Subtract an offset from a signal.

    If ``offset`` is not supplied, the offset is estimated from the selected
    field range, or from the value nearest zero field.
    """

    array = np.asarray(values, dtype=float)
    if offset is None:
        if field is None:
            offset = float(np.nanmean(array))
        else:
            field_array = np.asarray(field, dtype=float)
            if field_range is None:
                offset = float(array[np.argmin(np.abs(field_array))])
            else:
                low, high = sorted(field_range)
                mask = (field_array >= low) & (field_array <= high)
                if not np.any(mask):
                    raise ValueError("field_range did not select any points")
                offset = float(np.nanmean(array[mask]))
    return array - offset


def reject_outliers_mad(values, *, threshold: float = 5.0) -> np.ndarray:
    """Return a mask that rejects median-absolute-deviation outliers."""

    array = np.asarray(values, dtype=float)
    median = np.nanmedian(array)
    mad = np.nanmedian(np.abs(array - median))
    if mad == 0 or not np.isfinite(mad):
        return np.ones(array.shape, dtype=bool)
    modified_z = 0.6745 * (array - median) / mad
    return np.abs(modified_z) <= threshold


def clean_transport_data(
    frame: pd.DataFrame,
    *,
    field_column: str = "B",
    columns: tuple[str, ...] = ("B", "Rxx", "Rxy"),
    duplicate_method: str = "mean",
    outlier_column: str | None = None,
    outlier_threshold: float = 5.0,
) -> pd.DataFrame:
    """Apply common raw-data cleaning steps."""

    cleaned = frame.copy()
    for column in columns:
        if column in cleaned:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned = remove_nan_rows(cleaned, columns=columns)
    cleaned = remove_duplicate_fields(cleaned, field_column=field_column, method=duplicate_method)
    cleaned = sort_by_field(cleaned, field_column=field_column)
    if outlier_column is not None:
        mask = reject_outliers_mad(cleaned[outlier_column].to_numpy(), threshold=outlier_threshold)
        cleaned = cleaned.loc[mask].reset_index(drop=True)
    cleaned.attrs.update(frame.attrs)
    return cleaned
