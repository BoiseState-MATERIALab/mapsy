import numpy as np
import pandas as pd
from ase import Atoms

from mapsy import (
    AdaptiveWorkflow,
    CalculationWorkflow,
    ModelSuite,
    ModelTrainingSpec,
    PointPropertyDatasetBuilder,
    RelaxStepDatasetBuilder,
    RobustGaussianProcessSurrogate,
)
from mapsy.data import Grid, System
from mapsy.maps import Maps


def _build_maps(frame: pd.DataFrame) -> Maps:
    cell = np.diag([10.0, 10.0, 10.0])
    grid = Grid(scalars=[2, 2, 2], cell=cell)
    atoms = Atoms("H", positions=[[5.0, 5.0, 5.0]], cell=cell, pbc=True)
    maps = Maps(System(grid=grid, atoms=atoms), [])
    maps.data = frame.copy()
    maps.features = [column for column in frame.columns if column not in {"x", "y", "z"}]
    return maps


def _add_relax_training_point(
    maps: Maps,
    point_index: int,
    *,
    kind: str = "centroid",
) -> None:
    row = maps.data.loc[point_index]
    maps.add_special_points(
        [point_index],
        kind=kind,
        iteration=0,
        label_status="completed",
        E_bfgs_steps_Ry=[np.array([-1.0 - row["pca0"], -1.2 - row["pca0"]])],
        z_H_bfgs_steps_A=[np.array([1.6 + row["pca1"], 1.4 + row["pca1"]])],
        E_bfgs_final_Ry=[-1.2 - row["pca0"]],
        z_H_final_A=[1.4 + row["pca1"]],
    )


def test_relax_step_dataset_builder_expands_bfgs_histories() -> None:
    frame = pd.DataFrame(
        {
            "point_index": [7],
            "pca0": [0.1],
            "pca1": [0.2],
            "E_bfgs_steps_Ry": [np.array([-1.0, -1.2, -1.3])],
            "z_H_bfgs_steps_A": [np.array([1.5, 1.4, 1.3])],
        }
    )

    dataset = RelaxStepDatasetBuilder(
        adsorbate_label="H",
        coordinate_feature="distance",
    ).build(frame, dataset_name="pes")

    assert dataset.name == "pes"
    assert dataset.feature_columns == ["pca0", "pca1", "distance"]
    assert dataset.target_column == "E_bfgs_steps_Ry"
    assert dataset.frame["point_index"].tolist() == [7, 7, 7]
    assert dataset.frame["bfgs_step"].tolist() == [0, 1, 2]
    assert dataset.frame["distance"].tolist() == [1.5, 1.4, 1.3]


def test_robust_gaussian_process_surrogate_matches_notebook_style_fit() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(18, 2))
    y = 0.8 * X[:, 0] ** 2 + 0.3 * X[:, 1] - 0.2 * X[:, 0] * X[:, 1]

    model = RobustGaussianProcessSurrogate(
        name="pes",
        role="pes",
        feature_names=["pc1", "distance"],
        target_name="energy",
        n_random_starts=4,
        n_cv_splits=3,
        seed=0,
    )
    model.fit(X, y)

    summary = model.summary()
    prediction = model.predict(X[:3])
    validation = model.validate_loo(X, y)

    assert model.model_ is not None
    assert len(model.fit_records_) >= 1
    assert summary["name"] == "pes"
    assert summary["role"] == "pes"
    assert prediction.shape == (3,)
    assert validation["predictions"].shape == y.shape
    assert np.isfinite(validation["r2"])

    dataset = PointPropertyDatasetBuilder(
        feature_columns=["pc1", "distance"],
        target_column="energy",
    ).build(
        pd.DataFrame(
            {
                "pc1": X[:, 0],
                "distance": X[:, 1],
                "energy": y,
            }
        )
    )
    dataset_validation = model.validate_loo_dataset(dataset)
    assert dataset_validation["predictions"].shape == y.shape


def test_robust_gaussian_process_surrogate_parallel_starts_match_serial_fit() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(16, 2))
    y = 0.7 * X[:, 0] ** 2 - 0.4 * X[:, 1] + 0.1 * X[:, 0] * X[:, 1]
    kwargs = {
        "name": "pes",
        "role": "pes",
        "feature_names": ["pc1", "distance"],
        "target_name": "energy",
        "n_random_starts": 2,
        "n_cv_splits": 2,
        "seed": 1,
    }

    serial = RobustGaussianProcessSurrogate(**kwargs)
    parallel = RobustGaussianProcessSurrogate(**kwargs, n_jobs=2)

    serial.fit(X, y)
    parallel.fit(X, y)

    assert len(parallel.fit_records_) == len(serial.fit_records_)
    np.testing.assert_allclose(
        parallel.best_record_.cv_rmse_mean,
        serial.best_record_.cv_rmse_mean,
    )
    np.testing.assert_allclose(parallel.predict(X[:4]), serial.predict(X[:4]))


def test_model_training_spec_validates_builder_dataset() -> None:
    frame = pd.DataFrame(
        {
            "point_index": [0, 1, 2],
            "pca0": [0.0, 0.5, 1.0],
            "pca1": [0.2, 0.1, 0.4],
            "E_bfgs_steps_Ry": [
                np.array([-1.00, -1.10]),
                np.array([-1.30, -1.35]),
                np.array([-1.55, -1.70]),
            ],
            "z_H_bfgs_steps_A": [
                np.array([1.6, 1.4]),
                np.array([1.7, 1.5]),
                np.array([1.8, 1.6]),
            ],
        }
    )
    model = RobustGaussianProcessSurrogate(
        name="pes",
        role="pes",
        feature_names=["pca0", "pca1", "distance"],
        target_name="E_bfgs_steps_Ry",
        n_random_starts=1,
        n_cv_splits=2,
        seed=0,
    )
    spec = ModelTrainingSpec(
        model=model,
        builder=RelaxStepDatasetBuilder(adsorbate_label="H", coordinate_feature="distance"),
        role="pes",
    )

    spec.fit(frame)
    validation = spec.validate_loo()

    assert spec.last_dataset is not None
    assert len(spec.last_dataset.frame) == 6
    assert spec.last_dataset.feature_columns == ["pca0", "pca1", "distance"]
    assert validation["predictions"].shape == (6,)
    assert np.isfinite(validation["rmse"])


class _LinearModel:
    def __init__(
        self, name: str, role: str | None, columns: list[str], weights: list[float]
    ) -> None:
        self.name = name
        self.role = role
        self.columns = columns
        self.weights = np.array(weights, dtype=float)
        self.last_dataset = None

    def fit_dataset(self, dataset):
        self.last_dataset = dataset
        return self

    def predict(self, frame: pd.DataFrame, return_std: bool = False):
        values = frame.loc[:, self.columns].to_numpy(dtype=float) @ self.weights
        if return_std:
            return values, np.full(len(frame), 0.05, dtype=float)
        return values


def test_adaptive_workflow_fits_models_from_special_points() -> None:
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0],
            "pca0": [0.1, 0.2, 0.5],
            "pca1": [0.2, 0.3, 0.4],
        }
    )
    maps = _build_maps(frame)
    _add_relax_training_point(maps, 0)
    _add_relax_training_point(maps, 1)

    pes_model = _LinearModel("pes_model", "pes", ["pca0", "distance"], [1.0, -1.0])
    relaxed_energy_model = _LinearModel(
        "relaxed_energy_model",
        "relaxed_energy",
        ["pca0", "pca1"],
        [1.0, 0.5],
    )

    adaptive = AdaptiveWorkflow(
        calculation_workflow=CalculationWorkflow(
            calculation_name="qe_relax",
            calculation_description={},
        ),
        training_specs=[
            ModelTrainingSpec(
                model=pes_model,
                builder=RelaxStepDatasetBuilder(adsorbate_label="H", coordinate_feature="distance"),
                role="pes",
            ),
            ModelTrainingSpec(
                model=relaxed_energy_model,
                builder=PointPropertyDatasetBuilder(target_column="E_bfgs_final_Ry"),
                role="relaxed_energy",
            ),
        ],
    )

    suite = adaptive.fit_models(maps, source_kind="centroid")

    assert isinstance(suite, ModelSuite)
    assert suite.get_by_role("pes") is pes_model
    assert suite.get_by_role("relaxed_energy") is relaxed_energy_model
    assert pes_model.last_dataset.feature_columns == ["pca0", "pca1", "distance"]
    assert relaxed_energy_model.last_dataset.target_column == "E_bfgs_final_Ry"


def test_adaptive_workflow_proposes_points_and_stores_initialization_metadata() -> None:
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0],
            "pca0": [0.1, 0.2, 0.7],
            "pca1": [0.2, 0.4, 0.1],
        }
    )
    maps = _build_maps(frame)
    maps.add_special_points([0], kind="centroid", iteration=0, label_status="completed")

    suite = ModelSuite()
    suite.add(_LinearModel("pes_model", "pes", ["pca0", "distance"], [1.0, -1.0]))
    suite.add(
        _LinearModel(
            "relaxed_coordinate_model",
            "relaxed_coordinate",
            ["pca0", "pca1"],
            [0.2, -0.1],
        )
    )

    adaptive = AdaptiveWorkflow(
        calculation_workflow=CalculationWorkflow(
            calculation_name="qe_relax",
            calculation_description={},
        ),
        model_suite=suite,
        acquisition_role="pes",
        acquisition_strategy="minimum_over_coordinate",
        acquisition_coordinate_values=np.array([1.0, 2.0, 3.0]),
        initialization_role="relaxed_coordinate",
        initialization_prediction_column="predicted_initial_distance",
        initialization_uncertainty_column="predicted_initial_distance_std",
    )

    proposed = adaptive.propose_points(
        maps,
        npoints=1,
        kind="adaptive",
        iteration=1,
        feature_columns=["pca0", "pca1"],
        real_space_weight=0.0,
        feature_space_weight=0.0,
        energy_weight=1.0,
        uncertainty_weight=0.0,
    )

    assert proposed["kind"].tolist() == ["adaptive"]
    assert proposed["iteration"].tolist() == [1]
    assert "adaptive_predicted_energy" in proposed.columns
    assert "adaptive_predicted_coordinate" in proposed.columns
    assert "predicted_initial_distance" in proposed.columns


def test_adaptive_workflow_runs_proposed_points_through_calculation_workflow() -> None:
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
            "pca0": [0.1, 0.3],
            "pca1": [0.2, 0.1],
        }
    )
    maps = _build_maps(frame)
    maps.add_special_points([1], kind="adaptive", iteration=2, label_status="unlabeled")

    calc_workflow = CalculationWorkflow(
        calculation_name="qe_relax",
        calculation_description={"code": "mock"},
        runner=lambda maps, workflow, special_point: f"calc_{int(special_point['point_index'])}.out",
    )
    adaptive = AdaptiveWorkflow(calculation_workflow=calc_workflow)

    result = adaptive.run_proposed(maps, kind="adaptive")

    assert result["label_file"].tolist() == ["calc_1.out"]
