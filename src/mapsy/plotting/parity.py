from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _as_numeric_1d(values: npt.ArrayLike, *, name: str) -> npt.NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value.")
    return array


def _axis_label(label: str, units: str | None) -> str:
    return label if units is None else f"{label} ({units})"


def _format_error(value: float, units: str | None, precision: int) -> str:
    formatted = f"{value:.{precision}g}"
    return formatted if units is None else f"{formatted} {units}"


def _resolve_reference_predictions(
    dataset: Any | None,
    loo: Mapping[str, Any] | None,
    *,
    reference: npt.ArrayLike | None,
    predicted: npt.ArrayLike | None,
    residuals: npt.ArrayLike | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if reference is None:
        if dataset is None:
            raise ValueError("reference must be provided when dataset is None.")
        if hasattr(dataset, "y"):
            reference = dataset.y()
        else:
            frame = getattr(dataset, "frame", None)
            target_column = getattr(dataset, "target_column", None)
            if frame is None or target_column is None:
                raise ValueError("dataset must provide y() or frame and target_column.")
            reference = frame.loc[:, target_column].to_numpy(dtype=np.float64)

    if predicted is None:
        if loo is None or "predictions" not in loo:
            raise ValueError("predicted must be provided when loo['predictions'] is unavailable.")
        predicted = loo["predictions"]

    ref = _as_numeric_1d(reference, name="reference")
    pred = _as_numeric_1d(predicted, name="predicted")
    if ref.shape != pred.shape:
        raise ValueError(
            f"reference and predicted must have the same shape, got {ref.shape} and {pred.shape}."
        )

    if residuals is None:
        residuals = loo["residuals"] if loo is not None and "residuals" in loo else pred - ref
    err = _as_numeric_1d(residuals, name="residuals")
    if err.shape != ref.shape:
        raise ValueError(
            f"residuals must have the same shape as reference, got {err.shape} and {ref.shape}."
        )
    return ref, pred, err


def _dataset_index(dataset: Any | None, nrows: int) -> npt.NDArray[np.object_]:
    frame = getattr(dataset, "frame", None)
    if frame is not None and len(frame) == nrows:
        return frame.index.to_numpy(dtype=object)
    return np.arange(nrows, dtype=np.int64).astype(object)


def _default_metadata_columns(dataset: Any | None) -> tuple[str, ...]:
    frame = getattr(dataset, "frame", None)
    if frame is None:
        return ()
    common = ("point_index", "bfgs_step", "kind", "iteration", "label_status")
    return tuple(column for column in common if column in frame.columns)


def parity_metrics(
    reference: npt.ArrayLike,
    predicted: npt.ArrayLike,
    *,
    drop_nonfinite: bool = True,
) -> dict[str, Any]:
    """Compute parity metrics with error defined as predicted - reference."""
    ref = _as_numeric_1d(reference, name="reference")
    pred = _as_numeric_1d(predicted, name="predicted")
    if ref.shape != pred.shape:
        raise ValueError(
            f"reference and predicted must have the same shape, got {ref.shape} and {pred.shape}."
        )

    if drop_nonfinite:
        mask = np.isfinite(ref) & np.isfinite(pred)
        ref = ref[mask]
        pred = pred[mask]
    elif not (np.all(np.isfinite(ref)) and np.all(np.isfinite(pred))):
        raise ValueError("reference and predicted must contain finite values.")

    if ref.size == 0:
        raise ValueError("No finite reference/predicted pairs are available.")

    error = pred - ref
    abs_error = np.abs(error)
    return {
        "n": int(ref.size),
        "r2": float(r2_score(ref, pred)),
        "mae": float(mean_absolute_error(ref, pred)),
        "rmse": float(np.sqrt(mean_squared_error(ref, pred))),
        "median_abs_error": float(np.median(abs_error)),
        "max_abs_error": float(np.max(abs_error)),
        "mean_error": float(np.mean(error)),
    }


def loo_error_table(
    dataset: Any | None = None,
    loo: Mapping[str, Any] | None = None,
    *,
    reference: npt.ArrayLike | None = None,
    predicted: npt.ArrayLike | None = None,
    residuals: npt.ArrayLike | None = None,
    top_n: int | None = None,
    metadata_columns: Sequence[str] | None = None,
    sort: bool = True,
    drop_nonfinite: bool = True,
) -> pd.DataFrame:
    """Build a ranked table of leave-one-out prediction residuals.

    Residuals are defined as ``predicted - reference``. If ``dataset`` is provided, the table keeps
    the dataset row index and common metadata columns such as ``point_index`` and ``bfgs_step``.
    """
    ref, pred, err = _resolve_reference_predictions(
        dataset,
        loo,
        reference=reference,
        predicted=predicted,
        residuals=residuals,
    )
    if top_n is not None and int(top_n) < 0:
        raise ValueError("top_n must be non-negative or None.")

    mask = np.isfinite(ref) & np.isfinite(pred) & np.isfinite(err)
    if drop_nonfinite:
        selected = mask
    elif not np.all(mask):
        raise ValueError("reference, predicted, and residuals must contain finite values.")
    else:
        selected = np.ones_like(mask, dtype=bool)

    sample_index = np.arange(ref.size, dtype=np.int64)
    table = pd.DataFrame(
        {
            "sample_index": sample_index[selected],
            "frame_index": _dataset_index(dataset, ref.size)[selected],
            "reference": ref[selected],
            "prediction": pred[selected],
            "residual": err[selected],
            "abs_residual": np.abs(err[selected]),
        }
    )

    frame = getattr(dataset, "frame", None)
    if metadata_columns is None:
        metadata_columns = _default_metadata_columns(dataset)
    if frame is not None and metadata_columns:
        for column in metadata_columns:
            if column not in frame.columns:
                raise ValueError(f"metadata column {column!r} is not present in dataset.frame.")
            table.loc[:, column] = frame[column].to_numpy()[selected]

    if sort:
        table = table.sort_values("abs_residual", ascending=False)
    table = table.reset_index(drop=True)
    table.insert(0, "error_rank", np.arange(1, len(table) + 1, dtype=np.int64))
    if top_n is not None:
        table = table.head(int(top_n)).reset_index(drop=True)
        table.loc[:, "error_rank"] = np.arange(1, len(table) + 1, dtype=np.int64)
    return table


def _summary_text(metrics: Mapping[str, Any], *, units: str | None, precision: int) -> str:
    return "\n".join(
        [
            f"N = {metrics['n']}",
            f"R2 = {metrics['r2']:.4f}",
            f"MAE = {_format_error(float(metrics['mae']), units, precision)}",
            f"RMSE = {_format_error(float(metrics['rmse']), units, precision)}",
            "Median |err| = "
            f"{_format_error(float(metrics['median_abs_error']), units, precision)}",
            f"Max |err| = {_format_error(float(metrics['max_abs_error']), units, precision)}",
        ]
    )


def plot_parity(
    reference: npt.ArrayLike,
    predicted: npt.ArrayLike,
    *,
    title: str = "Parity Plot",
    reference_label: str = "Reference",
    predicted_label: str = "Predicted",
    units: str | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (5.5, 5.5),
    point_kwargs: Mapping[str, Any] | None = None,
    ideal_kwargs: Mapping[str, Any] | None = None,
    summary_box: bool = True,
    summary_loc: tuple[float, float] = (0.05, 0.95),
    summary_precision: int = 4,
    grid: bool = True,
    drop_nonfinite: bool = True,
) -> tuple[Figure, Axes, dict[str, Any]]:
    """Plot predicted values against reference values with parity metrics.

    The x-axis is the reference value, the y-axis is the predicted value, and errors are computed
    as ``predicted - reference``.
    """
    ref = _as_numeric_1d(reference, name="reference")
    pred = _as_numeric_1d(predicted, name="predicted")
    if ref.shape != pred.shape:
        raise ValueError(
            f"reference and predicted must have the same shape, got {ref.shape} and {pred.shape}."
        )

    if drop_nonfinite:
        mask = np.isfinite(ref) & np.isfinite(pred)
        ref = ref[mask]
        pred = pred[mask]
    elif not (np.all(np.isfinite(ref)) and np.all(np.isfinite(pred))):
        raise ValueError("reference and predicted must contain finite values.")

    metrics = parity_metrics(ref, pred, drop_nonfinite=False)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    scatter_kwargs = {
        "s": 18,
        "alpha": 0.65,
        "edgecolor": "none",
    }
    if point_kwargs is not None:
        scatter_kwargs.update(point_kwargs)
    ax.scatter(ref, pred, **scatter_kwargs)

    lo = float(min(np.min(ref), np.min(pred)))
    hi = float(max(np.max(ref), np.max(pred)))
    pad = 0.04 * (hi - lo) if hi > lo else 1.0
    lo -= pad
    hi += pad

    line_kwargs = {
        "color": "black",
        "linestyle": "--",
        "linewidth": 1.2,
        "label": "Ideal",
    }
    if ideal_kwargs is not None:
        line_kwargs.update(ideal_kwargs)
    ax.plot([lo, hi], [lo, hi], **line_kwargs)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel(_axis_label(reference_label, units))
    ax.set_ylabel(_axis_label(predicted_label, units))

    if summary_box:
        ax.text(
            summary_loc[0],
            summary_loc[1],
            _summary_text(metrics, units=units, precision=summary_precision),
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "0.35",
                "alpha": 0.9,
            },
        )

    if grid:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig, ax, metrics


def plot_loo_parity(
    dataset: Any | None = None,
    loo: Mapping[str, Any] | None = None,
    *,
    reference: npt.ArrayLike | None = None,
    predicted: npt.ArrayLike | None = None,
    residuals: npt.ArrayLike | None = None,
    top_n: int = 3,
    title: str = "LOO Parity Plot",
    reference_label: str | None = None,
    predicted_label: str = "LOO prediction",
    units: str | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (5.5, 5.5),
    point_kwargs: Mapping[str, Any] | None = None,
    ideal_kwargs: Mapping[str, Any] | None = None,
    highlight_kwargs: Mapping[str, Any] | None = None,
    annotate: bool = True,
    label_column: str = "sample_index",
    metadata_columns: Sequence[str] | None = None,
    summary_box: bool = True,
    summary_loc: tuple[float, float] = (0.05, 0.95),
    summary_precision: int = 4,
    grid: bool = True,
    drop_nonfinite: bool = True,
) -> tuple[Figure, Axes, dict[str, Any], pd.DataFrame]:
    """Plot LOO predicted values against reference values and highlight largest residuals."""
    ranked = loo_error_table(
        dataset,
        loo,
        reference=reference,
        predicted=predicted,
        residuals=residuals,
        top_n=None,
        metadata_columns=metadata_columns,
        sort=True,
        drop_nonfinite=drop_nonfinite,
    )
    if ranked.empty:
        raise ValueError("No finite LOO prediction rows are available.")
    if int(top_n) < 0:
        raise ValueError("top_n must be non-negative.")

    target_column = getattr(dataset, "target_column", None)
    resolved_reference_label = reference_label or str(target_column or "True label")
    fig, ax, metrics = plot_parity(
        ranked["reference"].to_numpy(dtype=np.float64),
        ranked["prediction"].to_numpy(dtype=np.float64),
        title=title,
        reference_label=resolved_reference_label,
        predicted_label=predicted_label,
        units=units,
        ax=ax,
        figsize=figsize,
        point_kwargs=point_kwargs,
        ideal_kwargs=ideal_kwargs,
        summary_box=summary_box,
        summary_loc=summary_loc,
        summary_precision=summary_precision,
        grid=grid,
        drop_nonfinite=False,
    )

    highlighted = ranked.head(int(top_n))
    if not highlighted.empty:
        marker_kwargs = {
            "s": 60,
            "facecolors": "none",
            "edgecolors": "red",
            "linewidths": 1.4,
        }
        if highlight_kwargs is not None:
            marker_kwargs.update(highlight_kwargs)
        ax.scatter(
            highlighted["reference"].to_numpy(dtype=np.float64),
            highlighted["prediction"].to_numpy(dtype=np.float64),
            **marker_kwargs,
        )

        if annotate:
            if label_column not in highlighted.columns:
                raise ValueError(
                    f"label_column {label_column!r} is not present in the error table."
                )
            for _, row in highlighted.iterrows():
                ax.annotate(
                    str(row[label_column]),
                    (float(row["reference"]), float(row["prediction"])),
                    xytext=(5.0, 5.0),
                    textcoords="offset points",
                    fontsize=8,
                    ha="left",
                    va="bottom",
                )

    return fig, ax, metrics, ranked


__all__ = ["loo_error_table", "parity_metrics", "plot_loo_parity", "plot_parity"]
