import numpy as np
import pandas as pd
from ase import Atoms
from matplotlib import pyplot as plt

from mapsy import Maps, loo_error_table, parity_metrics, plot_loo_parity, plot_parity
from mapsy.data import Grid, System
from mapsy.learning import SupervisedDataset


def _build_maps(frame: pd.DataFrame) -> Maps:
    cell = np.diag([10.0, 10.0, 10.0])
    grid = Grid(scalars=[2, 2, 2], cell=cell)
    atoms = Atoms("H", positions=[[5.0, 5.0, 5.0]], cell=cell, pbc=True)
    maps = Maps(System(grid=grid, atoms=atoms), [])
    maps.data = frame.copy()
    maps.features = [column for column in frame.columns if column not in {"x", "y", "z"}]
    return maps


def test_parity_metrics_uses_predicted_minus_reference_error() -> None:
    reference = np.array([0.0, 1.0, 2.0])
    predicted = np.array([0.1, 0.8, 2.3])

    metrics = parity_metrics(reference, predicted)

    assert metrics["n"] == 3
    np.testing.assert_allclose(metrics["mean_error"], (0.1 - 0.2 + 0.3) / 3.0)
    np.testing.assert_allclose(metrics["mae"], 0.2)
    np.testing.assert_allclose(metrics["rmse"], np.sqrt((0.01 + 0.04 + 0.09) / 3.0))
    np.testing.assert_allclose(metrics["median_abs_error"], 0.2)
    np.testing.assert_allclose(metrics["max_abs_error"], 0.3)


def test_plot_parity_draws_square_plot_and_summary_box() -> None:
    reference = np.array([0.0, 1.0, 2.0])
    predicted = np.array([0.1, 0.8, 2.3])

    fig, ax, metrics = plot_parity(
        reference,
        predicted,
        title="PES Model Parity Plot",
        reference_label="Reference energy",
        predicted_label="Predicted energy",
        units="eV",
    )

    assert metrics["n"] == 3
    assert ax.get_title() == "PES Model Parity Plot"
    assert ax.get_xlabel() == "Reference energy (eV)"
    assert ax.get_ylabel() == "Predicted energy (eV)"
    assert ax.get_xlim() == ax.get_ylim()
    assert len(ax.lines) == 1
    assert len(ax.texts) == 1
    assert "RMSE" in ax.texts[0].get_text()
    plt.close(fig)


def test_plot_parity_drops_nonfinite_pairs() -> None:
    reference = np.array([0.0, np.nan, 2.0, 3.0])
    predicted = np.array([0.1, 1.0, np.inf, 2.7])

    fig, _, metrics = plot_parity(reference, predicted)

    assert metrics["n"] == 2
    plt.close(fig)


def test_maps_plot_parity_compares_data_columns() -> None:
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0],
            "reference_energy": [0.0, 1.0, 2.0],
            "predicted_energy": [0.1, 0.8, 2.3],
        }
    )
    maps = _build_maps(frame)

    fig, ax, metrics = maps.plot_parity(
        "reference_energy",
        "predicted_energy",
        title="Predicted vs Reference",
        units="eV",
    )

    assert metrics["n"] == 3
    assert ax.get_title() == "Predicted vs Reference"
    assert ax.get_xlabel() == "reference_energy (eV)"
    assert ax.get_ylabel() == "predicted_energy (eV)"
    plt.close(fig)


def test_loo_error_table_ranks_largest_absolute_residuals() -> None:
    frame = pd.DataFrame(
        {
            "point_index": [10, 11, 12, 13],
            "f1": [0.0, 1.0, 2.0, 3.0],
            "target": [0.0, 1.0, 2.0, 3.0],
        },
        index=[100, 101, 102, 103],
    )
    dataset = SupervisedDataset(frame=frame, feature_columns=["f1"], target_column="target")
    loo = {
        "predictions": np.array([0.1, 0.7, 2.05, 3.5]),
        "residuals": np.array([0.1, -0.3, 0.05, 0.5]),
    }

    table = loo_error_table(dataset, loo, top_n=2)

    assert table["sample_index"].tolist() == [3, 1]
    assert table["frame_index"].tolist() == [103, 101]
    assert table["point_index"].tolist() == [13, 11]
    np.testing.assert_allclose(table["abs_residual"], [0.5, 0.3])


def test_loo_error_table_can_use_raw_arrays() -> None:
    table = loo_error_table(
        reference=np.array([0.0, 1.0, 2.0]),
        predicted=np.array([0.1, 0.7, 2.2]),
    )

    assert table["sample_index"].tolist() == [1, 2, 0]
    np.testing.assert_allclose(table["residual"], [-0.3, 0.2, 0.1])


def test_plot_loo_parity_highlights_top_residuals() -> None:
    frame = pd.DataFrame(
        {
            "point_index": [10, 11, 12, 13],
            "f1": [0.0, 1.0, 2.0, 3.0],
            "target": [0.0, 1.0, 2.0, 3.0],
        }
    )
    dataset = SupervisedDataset(frame=frame, feature_columns=["f1"], target_column="target")
    loo = {
        "predictions": np.array([0.1, 0.7, 2.05, 3.5]),
        "residuals": np.array([0.1, -0.3, 0.05, 0.5]),
    }

    fig, ax, metrics, table = plot_loo_parity(
        dataset,
        loo,
        top_n=2,
        units="eV",
        label_column="point_index",
    )

    assert metrics["n"] == 4
    assert table["sample_index"].head(2).tolist() == [3, 1]
    assert ax.get_xlabel() == "target (eV)"
    assert ax.get_ylabel() == "LOO prediction (eV)"
    assert len(ax.collections) == 2
    assert [text.get_text() for text in ax.texts[1:]] == ["13", "11"]
    plt.close(fig)
