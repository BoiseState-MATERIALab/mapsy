from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import linear_sum_assignment


def _validate_columns(frame: pd.DataFrame, columns: Sequence[str], *, frame_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}.")


def _numeric_column(
    frame: pd.DataFrame, column: str, *, frame_name: str
) -> npt.NDArray[np.float64]:
    values = frame[column].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{frame_name} column {column!r} must contain finite numeric values.")
    return values


def _normalize_periods(
    axes: tuple[str, ...],
    periods: Mapping[str, float | None] | Sequence[float | None] | None,
) -> tuple[float | None, ...]:
    if periods is None:
        return tuple(None for _ in axes)

    if isinstance(periods, Mapping):
        normalized = tuple(periods.get(axis) for axis in axes)
    else:
        if len(periods) != len(axes):
            raise ValueError("periods must have one entry per coordinate axis.")
        normalized = tuple(periods)

    for period in normalized:
        if period is not None and float(period) <= 0.0:
            raise ValueError("period values must be positive.")
    return tuple(None if period is None else float(period) for period in normalized)


def _coordinate_deltas(
    ref_coords: npt.NDArray[np.float64],
    pred_coords: npt.NDArray[np.float64],
    periods: tuple[float | None, ...],
) -> npt.NDArray[np.float64]:
    deltas = ref_coords[:, np.newaxis, :] - pred_coords[np.newaxis, :, :]
    for axis_index, period in enumerate(periods):
        if period is not None:
            deltas[:, :, axis_index] = (deltas[:, :, axis_index] + 0.5 * period) % period - (
                0.5 * period
            )
    return deltas


def _type_weight(point_type: Any, type_weights: Mapping[str, float] | None) -> float:
    if type_weights is None:
        return 1.0
    return float(type_weights.get(str(point_type), 1.0))


def _normalize_stationary_types(
    types: str | Sequence[str] | None,
    reference: pd.DataFrame,
    *,
    type_column: str,
) -> tuple[str, ...]:
    if types is None:
        return tuple(str(point_type) for point_type in pd.unique(reference[type_column]))
    if isinstance(types, str):
        return (types,)
    return tuple(types)


def _reference_weights(
    reference: pd.DataFrame,
    *,
    type_column: str,
    energy_column: str,
    energy_weight_scale: float | None,
    type_weights: Mapping[str, float] | None,
) -> npt.NDArray[np.float64]:
    energy = reference[energy_column].to_numpy(dtype=np.float64)
    if energy_weight_scale is None:
        weights = np.ones(len(reference), dtype=np.float64)
    else:
        if float(energy_weight_scale) <= 0.0:
            raise ValueError("energy_weight_scale must be positive or None.")
        weights = np.exp(-(energy - float(np.min(energy))) / float(energy_weight_scale))

    type_multipliers = np.array(
        [_type_weight(point_type, type_weights) for point_type in reference[type_column]],
        dtype=np.float64,
    )
    if np.any(type_multipliers < 0.0):
        raise ValueError("type_weights must be non-negative.")

    weights *= type_multipliers
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("Stationary-point comparison weights sum to zero.")
    return weights / total


def compare_stationary_points(
    reference: pd.DataFrame,
    predicted: pd.DataFrame,
    *,
    types: str | Sequence[str] | None = ("minimum", "saddle_1"),
    axes: Sequence[str] = ("x", "y"),
    type_column: str = "type",
    energy_column: str = "value",
    periods: Mapping[str, float | None] | Sequence[float | None] | None = None,
    xy_scale: float = 1.0,
    energy_scale: float = 0.05,
    energy_weight_scale: float | None = 0.10,
    max_xy_error: float | None = None,
    max_energy_error: float | None = 0.25,
    beta: float = 1.0,
    type_weights: Mapping[str, float] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compare stationary points from a predicted map against a reference map.

    The comparison matches stationary points type-by-type with a Hungarian assignment using a
    combined coordinate/energy cost. Reference points are weighted by their energy, so low-energy
    minima and lower saddle points can dominate the benchmark metric.
    """
    axes_tuple = tuple(axes)
    if len(axes_tuple) == 0:
        raise ValueError("axes must contain at least one coordinate column.")
    if len(set(axes_tuple)) != len(axes_tuple):
        raise ValueError(f"axes must be distinct, got {axes_tuple!r}.")
    if float(xy_scale) <= 0.0:
        raise ValueError("xy_scale must be positive.")
    if float(energy_scale) <= 0.0:
        raise ValueError("energy_scale must be positive.")
    if float(beta) < 0.0:
        raise ValueError("beta must be non-negative.")
    if max_xy_error is None:
        max_xy_error = 2.0 * float(xy_scale)
    elif float(max_xy_error) < 0.0:
        raise ValueError("max_xy_error must be non-negative or None.")
    if max_energy_error is not None and float(max_energy_error) < 0.0:
        raise ValueError("max_energy_error must be non-negative or None.")

    required_columns = [type_column, energy_column, *axes_tuple]
    _validate_columns(reference, required_columns, frame_name="reference")
    if not predicted.empty:
        _validate_columns(predicted, required_columns, frame_name="predicted")

    selected_types = _normalize_stationary_types(types, reference, type_column=type_column)
    if len(selected_types) == 0:
        raise ValueError("types must contain at least one stationary-point type.")

    reference_use = reference.loc[reference[type_column].isin(selected_types)].copy()
    if predicted.empty:
        predicted_use = pd.DataFrame(columns=required_columns)
    else:
        predicted_use = predicted.loc[predicted[type_column].isin(selected_types)].copy()

    if reference_use.empty:
        raise ValueError("reference has no stationary points for the requested types.")

    _numeric_column(reference_use, energy_column, frame_name="reference")
    for axis in axes_tuple:
        _numeric_column(reference_use, axis, frame_name="reference")
    if not predicted_use.empty:
        _numeric_column(predicted_use, energy_column, frame_name="predicted")
        for axis in axes_tuple:
            _numeric_column(predicted_use, axis, frame_name="predicted")

    reference_use.loc[:, "_comparison_weight"] = _reference_weights(
        reference_use,
        type_column=type_column,
        energy_column=energy_column,
        energy_weight_scale=energy_weight_scale,
        type_weights=type_weights,
    )
    normalized_periods = _normalize_periods(axes_tuple, periods)

    rows: list[dict[str, Any]] = []
    assigned_predicted_indexes: set[Any] = set()
    large_cost = 1.0e12

    def add_row(
        *,
        point_type: Any,
        reference_row: pd.Series | None,
        predicted_row: pd.Series | None,
        matched: bool,
        status: str,
        dxy: float = np.nan,
        d_energy: float = np.nan,
        match_cost: float = np.nan,
        deltas: npt.NDArray[np.float64] | None = None,
    ) -> None:
        row = {
            "type": point_type,
            "matched": matched,
            "status": status,
            "reference_index": np.nan if reference_row is None else reference_row.name,
            "predicted_index": np.nan if predicted_row is None else predicted_row.name,
            "ref_energy": np.nan if reference_row is None else float(reference_row[energy_column]),
            "pred_energy": np.nan if predicted_row is None else float(predicted_row[energy_column]),
            "dxy": dxy,
            "dE": d_energy,
            "abs_dE": abs(d_energy) if np.isfinite(d_energy) else np.nan,
            "weight": 0.0 if reference_row is None else float(reference_row["_comparison_weight"]),
            "match_cost": match_cost,
        }
        for axis_index, axis in enumerate(axes_tuple):
            row[f"ref_{axis}"] = np.nan if reference_row is None else float(reference_row[axis])
            row[f"pred_{axis}"] = np.nan if predicted_row is None else float(predicted_row[axis])
            row[f"delta_{axis}"] = np.nan if deltas is None else float(deltas[axis_index])
        rows.append(row)

    for point_type in selected_types:
        ref_type = reference_use.loc[reference_use[type_column] == point_type]
        pred_type = predicted_use.loc[predicted_use[type_column] == point_type]

        if ref_type.empty:
            for _, pred_row in pred_type.iterrows():
                add_row(
                    point_type=point_type,
                    reference_row=None,
                    predicted_row=pred_row,
                    matched=False,
                    status="extra",
                )
            continue

        if pred_type.empty:
            for _, ref_row in ref_type.iterrows():
                add_row(
                    point_type=point_type,
                    reference_row=ref_row,
                    predicted_row=None,
                    matched=False,
                    status="missed",
                )
            continue

        ref_coords = ref_type.loc[:, axes_tuple].to_numpy(dtype=np.float64)
        pred_coords = pred_type.loc[:, axes_tuple].to_numpy(dtype=np.float64)
        deltas = _coordinate_deltas(ref_coords, pred_coords, normalized_periods)
        distances = np.sqrt(np.sum(deltas**2, axis=2))
        d_energy = (
            pred_type[energy_column].to_numpy(dtype=np.float64)[np.newaxis, :]
            - ref_type[energy_column].to_numpy(dtype=np.float64)[:, np.newaxis]
        )
        costs = np.sqrt(
            (distances / float(xy_scale)) ** 2 + float(beta) * (d_energy / float(energy_scale)) ** 2
        )

        allowed = distances <= float(max_xy_error)
        if max_energy_error is not None:
            allowed &= np.abs(d_energy) <= float(max_energy_error)

        assignment_costs = np.where(allowed, costs, large_cost + costs)
        ref_assignment, pred_assignment = linear_sum_assignment(assignment_costs)
        assigned_by_ref = {
            int(ref_idx): int(pred_idx)
            for ref_idx, pred_idx in zip(ref_assignment, pred_assignment, strict=True)
        }

        for ref_position, (_, ref_row) in enumerate(ref_type.iterrows()):
            pred_position = assigned_by_ref.get(ref_position)
            if pred_position is None:
                add_row(
                    point_type=point_type,
                    reference_row=ref_row,
                    predicted_row=None,
                    matched=False,
                    status="missed",
                )
                continue

            pred_row = pred_type.iloc[pred_position]
            assigned_predicted_indexes.add(pred_row.name)
            matched = bool(allowed[ref_position, pred_position])
            status = "matched" if matched else "outside_tolerance"

            add_row(
                point_type=point_type,
                reference_row=ref_row,
                predicted_row=pred_row,
                matched=matched,
                status=status,
                dxy=float(distances[ref_position, pred_position]),
                d_energy=float(d_energy[ref_position, pred_position]),
                match_cost=float(costs[ref_position, pred_position]),
                deltas=deltas[ref_position, pred_position, :],
            )

        for _, pred_row in pred_type.iterrows():
            if pred_row.name in assigned_predicted_indexes:
                continue
            add_row(
                point_type=point_type,
                reference_row=None,
                predicted_row=pred_row,
                matched=False,
                status="extra",
            )

    matches = pd.DataFrame(rows)
    matched_rows = matches["matched"].to_numpy(dtype=bool)
    matched_weight = float(matches.loc[matched_rows, "weight"].sum())
    weighted_position_error = float(
        np.sum(
            matches.loc[matched_rows, "weight"].to_numpy(dtype=np.float64)
            * matches.loc[matched_rows, "dxy"].to_numpy(dtype=np.float64)
        )
    )
    matched_d_energy = matches.loc[matched_rows, "dE"].to_numpy(dtype=np.float64)
    matched_weights = matches.loc[matched_rows, "weight"].to_numpy(dtype=np.float64)
    weighted_energy_mae = float(np.sum(matched_weights * np.abs(matched_d_energy)))
    weighted_energy_rmse = float(np.sqrt(np.sum(matched_weights * matched_d_energy**2)))

    if matched_weight > 0.0:
        matched_position_mae = weighted_position_error / matched_weight
        matched_energy_mae = weighted_energy_mae / matched_weight
        matched_energy_rmse = float(
            np.sqrt(np.sum(matched_weights * matched_d_energy**2) / matched_weight)
        )
    else:
        matched_position_mae = np.nan
        matched_energy_mae = np.nan
        matched_energy_rmse = np.nan

    n_reference = int(len(reference_use))
    n_predicted = int(len(predicted_use))
    n_matched = int(np.count_nonzero(matched_rows))
    summary: dict[str, Any] = {
        "types": selected_types,
        "n_reference": n_reference,
        "n_predicted": n_predicted,
        "n_matched": n_matched,
        "n_missed": n_reference - n_matched,
        "n_extra": n_predicted - n_matched,
        "weighted_recall": matched_weight,
        "missed_weight": max(0.0, 1.0 - matched_weight),
        "weighted_position_mae": weighted_position_error,
        "weighted_energy_mae": weighted_energy_mae,
        "weighted_energy_rmse": weighted_energy_rmse,
        "matched_position_mae": matched_position_mae,
        "matched_energy_mae": matched_energy_mae,
        "matched_energy_rmse": matched_energy_rmse,
        "xy_scale": float(xy_scale),
        "energy_scale": float(energy_scale),
        "energy_weight_scale": energy_weight_scale,
        "max_xy_error": float(max_xy_error),
        "max_energy_error": max_energy_error,
        "beta": float(beta),
    }
    return summary, matches


__all__ = ["compare_stationary_points"]
