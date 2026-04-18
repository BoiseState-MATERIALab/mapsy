from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .maps import Maps


RunnerResult = str | Path | dict[str, Any] | None
ParserResult = Any | dict[str, Any]


@dataclass(init=False)
class CalculationWorkflow:
    """External workflow that attaches calculation artifacts and parsed outputs to special points."""

    calculation_name: str
    calculation_description: Any
    runner: Callable[..., RunnerResult] | None
    parser: Callable[..., ParserResult] | None
    file_column: str
    scalar_output_column: str

    def __init__(
        self,
        calculation_name: str | None = None,
        calculation_description: Any = None,
        *,
        runner: Callable[..., RunnerResult] | None = None,
        parser: Callable[..., ParserResult] | None = None,
        file_column: str = "label_file",
        scalar_output_column: str = "observed_label",
    ) -> None:

        if calculation_name is None:
            raise ValueError("calculation_name is required")

        self.calculation_name = calculation_name
        self.calculation_description = calculation_description
        self.runner = runner
        self.parser = parser
        self.file_column = file_column
        self.scalar_output_column = scalar_output_column

    def targets(
        self,
        maps: Maps,
        *,
        kind: str | None = None,
        label_status: str | list[str] | tuple[str, ...] | None = "unlabeled",
        point_indexes: Any = None,
    ) -> pd.DataFrame:
        frame = maps.get_special_points(kind=kind, label_status=label_status)
        if point_indexes is None or frame.empty:
            return frame

        indexes = np.asarray(point_indexes, dtype=np.int64).reshape(-1)
        return frame.loc[frame["point_index"].isin(indexes)].copy()

    def record_outputs(
        self,
        maps: Maps,
        point_indexes: Any,
        output_files: Any,
        *,
        kind: str | None = None,
        reset_label_status: bool = True,
    ) -> pd.DataFrame:
        indexes = np.asarray(point_indexes, dtype=np.int64).reshape(-1)
        files = np.asarray(output_files, dtype=object).reshape(-1)
        if indexes.size != files.size:
            raise ValueError(
                f"output_files has length {files.size}, expected {indexes.size} entries."
            )

        rows = []
        for point_index, output_file in zip(indexes, files, strict=False):
            metadata: dict[str, Any] = {
                **self._workflow_metadata(),
                self.file_column: str(output_file) if output_file is not None else None,
            }
            if reset_label_status and output_file is not None:
                metadata["label_status"] = "unlabeled"
                metadata["label_error"] = None
            rows.append(
                maps.update_special_points(
                    kind=kind,
                    point_indexes=[int(point_index)],
                    **metadata,
                )
            )

        if rows:
            return pd.concat(rows, ignore_index=True)
        return maps.get_special_points(kind=kind)

    def run(
        self,
        maps: Maps,
        *,
        kind: str | None = None,
        label_status: str | list[str] | tuple[str, ...] | None = "unlabeled",
        point_indexes: Any = None,
        parallel: bool = False,
        max_workers: int | None = None,
        **runner_kwargs: Any,
    ) -> pd.DataFrame:
        if self.runner is None:
            raise RuntimeError("CalculationWorkflow.runner is not configured.")

        targets = self.targets(
            maps,
            kind=kind,
            label_status=label_status,
            point_indexes=point_indexes,
        )
        if targets.empty:
            return targets

        rows = [row.copy() for _, row in targets.iterrows()]
        if parallel and len(rows) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        self.runner,
                        maps=maps,
                        workflow=self,
                        special_point=row,
                        **runner_kwargs,
                    )
                    for row in rows
                ]
                results = [future.result() for future in futures]
        else:
            results = [
                self.runner(maps=maps, workflow=self, special_point=row, **runner_kwargs)
                for row in rows
            ]

        for row, result in zip(rows, results, strict=False):
            metadata = self._coerce_runner_result(result)
            for key, value in self._workflow_metadata().items():
                metadata.setdefault(key, value)
            maps.update_special_points(
                kind=row["kind"] if kind is None else kind,
                point_indexes=[int(row["point_index"])],
                **metadata,
            )

        return maps.get_special_points(kind=kind, label_status=None)

    def discover_outputs(
        self,
        maps: Maps,
        *,
        root: str | Path = ".",
        directory_template: str = "qe_job_{point_index}",
        kind: str | None = None,
        label_status: str | list[str] | tuple[str, ...] | None = None,
        point_indexes: Any = None,
        only_existing: bool = True,
        reset_label_status: bool = True,
    ) -> pd.DataFrame:
        """Infer output folders from point indexes and attach them to special points."""
        targets = self.targets(
            maps,
            kind=kind,
            label_status=label_status,
            point_indexes=point_indexes,
        )
        if targets.empty:
            return targets

        root_path = Path(root).expanduser()
        discovered_indexes: list[int] = []
        discovered_files: list[str] = []
        for _, row in targets.iterrows():
            point_index = int(row["point_index"])
            output_path = root_path / directory_template.format(point_index=point_index)
            if only_existing and not output_path.exists():
                continue
            discovered_indexes.append(point_index)
            discovered_files.append(str(output_path))

        if not discovered_indexes:
            return targets.iloc[0:0].copy()

        return self.record_outputs(
            maps,
            discovered_indexes,
            discovered_files,
            kind=kind,
            reset_label_status=reset_label_status,
        )

    def retry_failed(
        self,
        maps: Maps,
        *,
        kind: str | None = None,
        point_indexes: Any = None,
        parallel: bool = False,
        max_workers: int | None = None,
        **runner_kwargs: Any,
    ) -> pd.DataFrame:
        """Rerun the workflow runner for failed special points."""
        return self.run(
            maps,
            kind=kind,
            label_status="failed",
            point_indexes=point_indexes,
            parallel=parallel,
            max_workers=max_workers,
            **runner_kwargs,
        )

    def collect_discovered(
        self,
        maps: Maps,
        *,
        root: str | Path = ".",
        directory_template: str = "qe_job_{point_index}",
        kind: str | None = None,
        label_status: str | list[str] | tuple[str, ...] | None = None,
        point_indexes: Any = None,
        only_existing: bool = True,
        reset_label_status: bool = True,
        **parser_kwargs: Any,
    ) -> pd.DataFrame:
        """Discover existing output folders from point indexes, attach them, and collect results."""
        attached = self.discover_outputs(
            maps,
            root=root,
            directory_template=directory_template,
            kind=kind,
            label_status=label_status,
            point_indexes=point_indexes,
            only_existing=only_existing,
            reset_label_status=reset_label_status,
        )
        if attached.empty:
            return attached
        return self.collect(
            maps,
            kind=kind,
            point_indexes=attached["point_index"].to_numpy(dtype=np.int64),
            **parser_kwargs,
        )

    def collect(
        self,
        maps: Maps,
        *,
        kind: str | None = None,
        label_status: str | list[str] | tuple[str, ...] | None = "unlabeled",
        point_indexes: Any = None,
        root: str | Path = ".",
        directory_template: str = "qe_job_{point_index}",
        discover_outputs: bool = True,
        only_existing: bool = True,
        reset_label_status: bool = True,
        **parser_kwargs: Any,
    ) -> pd.DataFrame:
        targets = self.targets(
            maps,
            kind=kind,
            label_status=label_status,
            point_indexes=point_indexes,
        )

        if self.parser is None:
            raise RuntimeError("CalculationWorkflow.parser is not configured.")

        if discover_outputs:
            if targets.empty:
                discovery_status = None if label_status == "unlabeled" else label_status
                attached = self.discover_outputs(
                    maps,
                    root=root,
                    directory_template=directory_template,
                    kind=kind,
                    label_status=discovery_status,
                    point_indexes=point_indexes,
                    only_existing=only_existing,
                    reset_label_status=reset_label_status,
                )
                if not attached.empty:
                    targets = self.targets(
                        maps,
                        kind=kind,
                        label_status="unlabeled",
                        point_indexes=attached["point_index"].to_numpy(dtype=np.int64),
                    )
            else:
                missing_file = (
                    self.file_column not in targets.columns
                    or targets[self.file_column].isna().any()
                )
                if missing_file:
                    missing_indexes = (
                        targets["point_index"].to_numpy(dtype=np.int64)
                        if self.file_column not in targets.columns
                        else targets.loc[targets[self.file_column].isna(), "point_index"].to_numpy(
                            dtype=np.int64
                        )
                    )
                    self.discover_outputs(
                        maps,
                        root=root,
                        directory_template=directory_template,
                        kind=kind,
                        label_status=None,
                        point_indexes=missing_indexes,
                        only_existing=only_existing,
                        reset_label_status=reset_label_status,
                    )
                    targets = self.targets(
                        maps,
                        kind=kind,
                        label_status=label_status,
                        point_indexes=targets["point_index"].to_numpy(dtype=np.int64),
                    )

        if targets.empty:
            return targets

        for _, row in targets.iterrows():
            output_file = row.get(self.file_column)
            point_index = int(row["point_index"])
            point_kind = str(row["kind"])
            try:
                if output_file is None or pd.isna(output_file):
                    raise ValueError(f"No {self.file_column} recorded for point {point_index}.")
                parsed = self.parser(
                    output_file=output_file,
                    maps=maps,
                    workflow=self,
                    special_point=row,
                    **parser_kwargs,
                )
                metadata = self._coerce_parser_result(parsed)
                for key, value in self._workflow_metadata().items():
                    metadata.setdefault(key, value)
                metadata.setdefault("label_status", "completed")
            except Exception as exc:
                metadata = {
                    **self._workflow_metadata(),
                    "label_status": "failed",
                    "label_error": str(exc),
                }

            maps.update_special_points(
                kind=point_kind if kind is None else kind,
                point_indexes=[point_index],
                **metadata,
            )

        return maps.get_special_points(kind=kind, label_status=None)

    def _workflow_metadata(self) -> dict[str, Any]:
        return {
            "calculation_name": self.calculation_name,
            "calculation_description": self.calculation_description,
        }

    def _coerce_runner_result(self, result: RunnerResult) -> dict[str, Any]:
        if isinstance(result, dict):
            return dict(result)
        return {self.file_column: str(result) if result is not None else None}

    def _coerce_parser_result(self, result: ParserResult) -> dict[str, Any]:
        if isinstance(result, dict):
            return dict(result)
        return {self.scalar_output_column: result}


__all__ = ["CalculationWorkflow"]
