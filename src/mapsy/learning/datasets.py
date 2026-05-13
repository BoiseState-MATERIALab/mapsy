from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SupervisedDataset:
    """Tabular supervised dataset with explicit feature and target columns."""

    frame: pd.DataFrame
    feature_columns: list[str]
    target_column: str
    name: str | None = None

    def X(self) -> np.ndarray:
        return self.frame.loc[:, self.feature_columns].to_numpy(dtype=float)

    def y(self) -> np.ndarray:
        return self.frame.loc[:, self.target_column].to_numpy(dtype=float)


@dataclass
class PointPropertyDatasetBuilder:
    """Build one-row-per-point supervised datasets from special-point tables."""

    feature_columns: list[str] | None = None
    target_column: str | None = None

    def build(
        self,
        frame: pd.DataFrame,
        *,
        target_column: str | None = None,
        dataset_name: str | None = None,
    ) -> SupervisedDataset:
        resolved_target = target_column or self.target_column
        if resolved_target is None:
            raise ValueError(
                "target_column must be provided either on the builder or at build time."
            )
        feature_columns = (
            list(self.feature_columns)
            if self.feature_columns is not None
            else self._infer_pca_columns(frame)
        )
        required_columns = [*feature_columns, resolved_target]
        data = frame.loc[:, required_columns].dropna().copy()
        return SupervisedDataset(
            frame=data,
            feature_columns=feature_columns,
            target_column=resolved_target,
            name=dataset_name,
        )

    @staticmethod
    def _infer_pca_columns(frame: pd.DataFrame) -> list[str]:
        columns = [column for column in frame.columns if column.startswith("pca")]
        if not columns:
            raise ValueError("Could not infer PCA feature columns from the dataframe.")
        return sorted(columns, key=lambda name: int(name[3:]) if name[3:].isdigit() else name)


@dataclass
class RelaxStepDatasetBuilder:
    """Build one-row-per-BFGS-step datasets from relax parser outputs."""

    adsorbate_label: str = "H"
    feature_columns: list[str] | None = None
    coordinate_feature: str = "distance"
    energy_column: str = "E_bfgs_steps_Ry"
    step_index_column: str = "bfgs_step"

    def build(
        self,
        frame: pd.DataFrame,
        *,
        dataset_name: str | None = None,
    ) -> SupervisedDataset:
        feature_columns = (
            list(self.feature_columns)
            if self.feature_columns is not None
            else PointPropertyDatasetBuilder._infer_pca_columns(frame)
        )
        coordinate_column = f"z_{self.adsorbate_label}_bfgs_steps_A"
        if coordinate_column not in frame.columns:
            raise ValueError(f"Missing relax coordinate history column {coordinate_column!r}.")
        if self.energy_column not in frame.columns:
            raise ValueError(f"Missing relax energy history column {self.energy_column!r}.")

        rows: list[dict[str, Any]] = []
        for _, row in frame.iterrows():
            energies = np.asarray(row[self.energy_column], dtype=float).reshape(-1)
            coordinates = np.asarray(row[coordinate_column], dtype=float).reshape(-1)
            if energies.size == 0 or coordinates.size == 0:
                continue
            if energies.size != coordinates.size:
                raise ValueError(
                    f"Point {row.get('point_index')} has {energies.size} energies and "
                    f"{coordinates.size} coordinates."
                )

            base = {column: float(row[column]) for column in feature_columns}
            if "point_index" in row.index:
                base["point_index"] = int(row["point_index"])
            if "kind" in row.index:
                base["kind"] = row["kind"]
            if "iteration" in row.index:
                base["iteration"] = row["iteration"]

            for step_index, (energy, coordinate) in enumerate(
                zip(energies, coordinates, strict=True)
            ):
                rows.append(
                    {
                        **base,
                        self.step_index_column: step_index,
                        self.coordinate_feature: float(coordinate),
                        self.energy_column: float(energy),
                    }
                )

        if not rows:
            empty_columns = [*feature_columns, self.coordinate_feature, self.energy_column]
            return SupervisedDataset(
                frame=pd.DataFrame(columns=empty_columns),
                feature_columns=[*feature_columns, self.coordinate_feature],
                target_column=self.energy_column,
                name=dataset_name,
            )

        data = pd.DataFrame(rows)
        return SupervisedDataset(
            frame=data,
            feature_columns=[*feature_columns, self.coordinate_feature],
            target_column=self.energy_column,
            name=dataset_name,
        )
