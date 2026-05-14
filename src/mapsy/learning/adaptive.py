from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd

from .datasets import SupervisedDataset
from .models import ModelSuite


@dataclass
class ModelTrainingSpec:
    """One model plus the dataset builder used to fit it."""

    model: Any
    builder: Any
    role: str | None = None
    last_dataset: SupervisedDataset | None = field(default=None, init=False)

    @property
    def name(self) -> str:
        return str(self.model.name)

    def build_dataset(self, frame: pd.DataFrame) -> SupervisedDataset:
        self.last_dataset = self.builder.build(frame, dataset_name=self.name)
        return self.last_dataset

    def fit(self, frame: pd.DataFrame) -> Any:
        dataset = self.build_dataset(frame)
        self.model.fit_dataset(dataset)
        return self.model

    def validate_loo(self, frame: pd.DataFrame | None = None) -> dict[str, Any]:
        dataset = self.build_dataset(frame) if frame is not None else self.last_dataset
        if dataset is None:
            raise RuntimeError("No dataset has been built. Pass a frame or call fit first.")
        if hasattr(self.model, "validate_loo_dataset"):
            return cast(dict[str, Any], self.model.validate_loo_dataset(dataset))
        if hasattr(self.model, "validate_loo_frame"):
            return cast(
                dict[str, Any],
                self.model.validate_loo_frame(
                    dataset.frame,
                    feature_columns=dataset.feature_columns,
                    target_column=dataset.target_column,
                ),
            )
        if hasattr(self.model, "validate_loo"):
            return cast(dict[str, Any], self.model.validate_loo(dataset.X(), dataset.y()))
        raise AttributeError("Model does not provide validate_loo_frame or validate_loo.")


@dataclass
class AdaptiveWorkflow:
    """Adaptive loop that connects fitted surrogates to Maps and CalculationWorkflow."""

    calculation_workflow: Any
    training_specs: list[ModelTrainingSpec] = field(default_factory=list)
    model_suite: ModelSuite = field(default_factory=ModelSuite)
    acquisition_role: str = "pes"
    acquisition_strategy: str = "minimum_over_coordinate"
    acquisition_coordinate_values: np.ndarray | None = None
    acquisition_coordinate_feature: str = "distance"
    acquisition_energy_column: str = "adaptive_predicted_energy"
    acquisition_uncertainty_column: str = "adaptive_predicted_energy_std"
    acquisition_coordinate_column: str = "adaptive_predicted_coordinate"
    acquisition_score_column: str = "adaptive_acquisition_score"
    initialization_role: str | None = "relaxed_coordinate"
    initialization_prediction_column: str = "adaptive_initial_coordinate"
    initialization_uncertainty_column: str = "adaptive_initial_coordinate_std"
    use_acquisition_coordinate_as_initialization: bool = True

    def fit_models(
        self,
        maps: Any,
        *,
        source_kind: str | None = None,
        label_status: str | list[str] | tuple[str, ...] | None = "completed",
    ) -> ModelSuite:
        training_frame = maps.get_special_points(kind=source_kind, label_status=label_status)
        if training_frame.empty:
            raise RuntimeError("No training special points available for adaptive model fitting.")

        self.model_suite = ModelSuite()
        for spec in self.training_specs:
            model = spec.fit(training_frame)
            self.model_suite.add(model, role=spec.role or getattr(model, "role", None))
        return self.model_suite

    def collect_and_fit(
        self,
        maps: Any,
        *,
        collect_kind: str | None = None,
        collect_label_status: str | list[str] | tuple[str, ...] | None = "unlabeled",
        fit_kind: str | None = None,
        fit_label_status: str | list[str] | tuple[str, ...] | None = "completed",
        **collect_kwargs: Any,
    ) -> ModelSuite:
        self.calculation_workflow.collect(
            maps,
            kind=collect_kind,
            label_status=collect_label_status,
            **collect_kwargs,
        )
        return self.fit_models(
            maps,
            source_kind=fit_kind,
            label_status=fit_label_status,
        )

    def score_candidates(
        self,
        maps: Any,
        *,
        region: int | None = None,
    ) -> pd.DataFrame:
        if maps.data is None:
            raise RuntimeError("Maps has no data to score.")

        candidates = maps.data.copy()
        if region is not None:
            if maps.contactspace is None:
                raise RuntimeError(
                    "Cannot filter adaptive candidates by region without contact space."
                )
            mask = maps.contactspace.data["region"] == region
            candidates = candidates.loc[mask].copy()

        scored = self._score_with_acquisition_model(candidates)
        scored.loc[:, self.acquisition_score_column] = -scored[self.acquisition_energy_column]
        if self.acquisition_uncertainty_column in scored.columns:
            scored.loc[:, self.acquisition_score_column] += scored[
                self.acquisition_uncertainty_column
            ]

        if self.initialization_role is not None:
            initialization_model = self.model_suite.get_by_role(self.initialization_role)
            prediction, std = initialization_model.predict(scored, return_std=True)
            scored.loc[:, self.initialization_prediction_column] = np.asarray(
                prediction, dtype=float
            )
            scored.loc[:, self.initialization_uncertainty_column] = np.asarray(std, dtype=float)
        elif (
            self.use_acquisition_coordinate_as_initialization
            and self.acquisition_coordinate_column in scored.columns
        ):
            scored.loc[:, self.initialization_prediction_column] = scored[
                self.acquisition_coordinate_column
            ].to_numpy(dtype=float)

        return scored

    def propose_points(
        self,
        maps: Any,
        *,
        npoints: int,
        kind: str = "adaptive",
        iteration: int | None = None,
        region: int | None = None,
        feature_columns: list[str] | None = None,
        special_point_indexes: Any = None,
        centroid_indexes: Any = None,
        real_space_weight: float = 0.0,
        feature_space_weight: float = 1.0,
        energy_weight: float = 1.0,
        uncertainty_weight: float = 0.0,
        scale_features: bool = True,
    ) -> pd.DataFrame:
        scored = self.score_candidates(maps, region=region)
        self._sync_scored_columns(maps, scored)

        selection = maps.select_points(
            npoints=npoints,
            feature_columns=feature_columns,
            energy_column=self.acquisition_energy_column,
            uncertainty_column=self.acquisition_uncertainty_column,
            special_point_indexes=special_point_indexes,
            centroid_indexes=centroid_indexes,
            region=region,
            real_space_weight=real_space_weight,
            feature_space_weight=feature_space_weight,
            energy_weight=energy_weight,
            uncertainty_weight=uncertainty_weight,
            scale_features=scale_features,
        )

        selected_indexes = selection.index.to_numpy(dtype=np.int64)
        selected_scored = scored.loc[selected_indexes]
        maps.add_special_points(
            selected_indexes,
            kind=kind,
            iteration=iteration,
            label_status="unlabeled",
            selection_rank=selection["selection_rank"].to_numpy(),
            selection_score=selection["selection_score"].to_numpy(),
            real_space_score=selection["real_space_score"].to_numpy(),
            feature_space_score=selection["feature_space_score"].to_numpy(),
            energy_score=selection["energy_score"].to_numpy(),
            uncertainty_score=selection["uncertainty_score"].to_numpy(),
            **self._selected_metadata(selected_scored),
        )

        proposed = maps.get_special_points(kind=kind)
        proposed = proposed.loc[proposed["point_index"].isin(selected_indexes)].copy()
        if "selection_rank" in proposed.columns:
            proposed = proposed.sort_values("selection_rank")
        return proposed.reset_index(drop=True)

    def run_proposed(
        self,
        maps: Any,
        *,
        kind: str = "adaptive",
        point_indexes: Any = None,
        parallel: bool = False,
        max_workers: int | None = None,
        **runner_kwargs: Any,
    ) -> pd.DataFrame:
        return self.calculation_workflow.run(
            maps,
            kind=kind,
            point_indexes=point_indexes,
            parallel=parallel,
            max_workers=max_workers,
            **runner_kwargs,
        )

    def collect(
        self,
        maps: Any,
        *,
        kind: str | None = None,
        label_status: str | list[str] | tuple[str, ...] | None = "unlabeled",
        **collect_kwargs: Any,
    ) -> pd.DataFrame:
        return self.calculation_workflow.collect(
            maps,
            kind=kind,
            label_status=label_status,
            **collect_kwargs,
        )

    def _score_with_acquisition_model(self, candidates: pd.DataFrame) -> pd.DataFrame:
        model = self.model_suite.get_by_role(self.acquisition_role)
        if self.acquisition_strategy == "direct_prediction":
            prediction, std = model.predict(candidates, return_std=True)
            scored = candidates.copy()
            scored.loc[:, self.acquisition_energy_column] = np.asarray(prediction, dtype=float)
            scored.loc[:, self.acquisition_uncertainty_column] = np.asarray(std, dtype=float)
            return scored

        if self.acquisition_strategy != "minimum_over_coordinate":
            raise ValueError(
                "acquisition_strategy must be 'direct_prediction' or 'minimum_over_coordinate', "
                f"got {self.acquisition_strategy!r}."
            )
        coordinate_values = np.asarray(self.acquisition_coordinate_values, dtype=float).reshape(-1)
        if coordinate_values.size == 0:
            raise ValueError(
                "acquisition_coordinate_values must be provided for minimum_over_coordinate."
            )

        rows = []
        for _, row in candidates.iterrows():
            expanded = pd.DataFrame(
                [
                    {**row.to_dict(), self.acquisition_coordinate_feature: float(value)}
                    for value in coordinate_values
                ]
            )
            prediction, std = model.predict(expanded, return_std=True)
            prediction = np.asarray(prediction, dtype=float)
            std = np.asarray(std, dtype=float)
            best = int(np.argmin(prediction))
            rows.append(
                {
                    **row.to_dict(),
                    self.acquisition_energy_column: float(prediction[best]),
                    self.acquisition_uncertainty_column: float(std[best]),
                    self.acquisition_coordinate_column: float(coordinate_values[best]),
                }
            )
        return pd.DataFrame(rows, index=candidates.index)

    def _selected_metadata(self, selected_scored: pd.DataFrame) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        if self.initialization_prediction_column in selected_scored.columns:
            metadata[self.initialization_prediction_column] = selected_scored[
                self.initialization_prediction_column
            ].to_numpy()
        if self.initialization_uncertainty_column in selected_scored.columns:
            metadata[self.initialization_uncertainty_column] = selected_scored[
                self.initialization_uncertainty_column
            ].to_numpy()
        return metadata

    def _sync_scored_columns(self, maps: Any, scored: pd.DataFrame) -> None:
        if maps.data is None:
            return
        for column in [
            self.acquisition_energy_column,
            self.acquisition_uncertainty_column,
            self.acquisition_coordinate_column,
            self.acquisition_score_column,
        ]:
            if column in scored.columns:
                maps.data.loc[scored.index, column] = scored[column].to_numpy()
