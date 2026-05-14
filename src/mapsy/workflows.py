from __future__ import annotations

from collections.abc import Callable, Sequence
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


@dataclass(slots=True)
class WorkflowReferenceSpec:
    name: str
    scope: str
    description: Any
    runner: Callable[..., RunnerResult] | None
    parser: Callable[..., ParserResult] | None
    file_column: str
    scalar_output_column: str
    postprocess: Callable[..., dict[str, Any] | None] | None = None


@dataclass(init=False)
class CalculationWorkflow:
    """External workflow that attaches calculation artifacts and parsed outputs to special points."""

    calculation_name: str
    calculation_description: Any
    runner: Callable[..., RunnerResult] | None
    parser: Callable[..., ParserResult] | None
    file_column: str
    scalar_output_column: str
    reference_specs: dict[str, WorkflowReferenceSpec]
    reference_records: dict[tuple[str, int | None], dict[str, Any]]

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
        self.reference_specs = {}
        self.reference_records = {}

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
            output_file = metadata.get(self.file_column)
            if output_file is not None and not pd.isna(output_file):
                metadata.setdefault("label_status", "unlabeled")
                metadata.setdefault("label_error", None)
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

    def add_reference(
        self,
        name: str,
        *,
        scope: str = "global",
        description: Any = None,
        runner: Callable[..., RunnerResult] | None = None,
        parser: Callable[..., ParserResult] | None = None,
        file_column: str = "reference_file",
        scalar_output_column: str = "reference_value",
        postprocess: Callable[..., dict[str, Any] | None] | None = None,
    ) -> WorkflowReferenceSpec:
        if scope not in {"global", "per_map"}:
            raise ValueError(f"scope must be 'global' or 'per_map', got {scope!r}.")
        if not name:
            raise ValueError("Reference name must be non-empty.")

        spec = WorkflowReferenceSpec(
            name=name,
            scope=scope,
            description=description,
            runner=runner,
            parser=parser,
            file_column=file_column,
            scalar_output_column=scalar_output_column,
            postprocess=postprocess,
        )
        self.reference_specs[name] = spec
        return spec

    def reference_frame(
        self,
        *,
        name: str | None = None,
        scope: str | None = None,
        label_status: str | Sequence[str] | None = None,
    ) -> pd.DataFrame:
        if not self.reference_records:
            return pd.DataFrame()

        rows = []
        for record in self.reference_records.values():
            if name is not None and record.get("name") != name:
                continue
            if scope is not None and record.get("scope") != scope:
                continue
            if label_status is not None:
                statuses = [label_status] if isinstance(label_status, str) else list(label_status)
                if record.get("label_status") not in statuses:
                    continue
            rows.append(dict(record))
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def get_reference(
        self,
        name: str,
        *,
        map_index: int | None = None,
        require_completed: bool = True,
    ) -> dict[str, Any]:
        spec = self._get_reference_spec(name)
        key = (name, None) if spec.scope == "global" else (name, map_index)
        if key not in self.reference_records:
            raise KeyError(f"Reference {name!r} with key {key!r} is not available.")
        record = dict(self.reference_records[key])
        if require_completed and record.get("label_status") != "completed":
            raise RuntimeError(
                f"Reference {name!r} with key {key!r} has status {record.get('label_status')!r}."
            )
        return record

    def record_reference_outputs(
        self,
        maps: Any,
        name: str,
        output_files: Any,
        *,
        map_indexes: Any = None,
        reset_label_status: bool = True,
    ) -> pd.DataFrame:
        spec = self._get_reference_spec(name)
        targets = self._reference_targets(maps, spec, map_indexes=map_indexes)

        if spec.scope == "global":
            output_file = output_files.get(name) if isinstance(output_files, dict) else output_files
            record = self._upsert_reference_record(
                spec=spec,
                map_index=None,
                system_name=None,
                maps_target=maps,
                metadata={
                    spec.file_column: str(output_file) if output_file is not None else None,
                    **(
                        {"label_status": "unlabeled", "label_error": None}
                        if reset_label_status and output_file is not None
                        else {}
                    ),
                },
            )
            return pd.DataFrame([record])

        files_by_index = self._normalize_reference_outputs_by_map_index(output_files, targets)
        rows = []
        for target in targets:
            output_file = files_by_index.get(int(target["map_index"]))
            rows.append(
                self._upsert_reference_record(
                    spec=spec,
                    map_index=int(target["map_index"]),
                    system_name=target["system_name"],
                    maps_target=target["maps_target"],
                    metadata={
                        spec.file_column: str(output_file) if output_file is not None else None,
                        **(
                            {"label_status": "unlabeled", "label_error": None}
                            if reset_label_status and output_file is not None
                            else {}
                        ),
                    },
                )
            )
        return pd.DataFrame(rows)

    def discover_reference_outputs(
        self,
        maps: Any,
        *,
        names: Sequence[str] | None = None,
        map_indexes: Any = None,
        root: str | Path = "references",
        global_directory_template: str = "global/{reference_name}",
        per_map_directory_template: str = "map_{map_index}/{reference_name}",
        only_existing: bool = True,
        reset_label_status: bool = True,
    ) -> pd.DataFrame:
        """Infer reference output folders from the expected workflow layout and attach them."""
        specs = self._selected_reference_specs(names)
        rows = []
        root_path = Path(root).expanduser()
        for spec in specs:
            targets = self._reference_targets(maps, spec, map_indexes=map_indexes)
            discovered_by_index: dict[int, str] = {}
            discovered_global: str | None = None
            for target in targets:
                output_path = self._expected_reference_output_path(
                    spec=spec,
                    target=target,
                    root_path=root_path,
                    global_directory_template=global_directory_template,
                    per_map_directory_template=per_map_directory_template,
                )
                if only_existing and not output_path.exists():
                    continue
                if spec.scope == "global":
                    discovered_global = str(output_path)
                else:
                    discovered_by_index[int(target["map_index"])] = str(output_path)

            if spec.scope == "global":
                if discovered_global is None:
                    continue
                rows.append(
                    self.record_reference_outputs(
                        maps,
                        spec.name,
                        discovered_global,
                        reset_label_status=reset_label_status,
                    )
                )
            else:
                if not discovered_by_index:
                    continue
                rows.append(
                    self.record_reference_outputs(
                        maps,
                        spec.name,
                        discovered_by_index,
                        map_indexes=map_indexes,
                        reset_label_status=reset_label_status,
                    )
                )

        if rows:
            return pd.concat(rows, ignore_index=True)
        return pd.DataFrame()

    def run_references(
        self,
        maps: Any,
        *,
        names: Sequence[str] | None = None,
        map_indexes: Any = None,
        parallel: bool = False,
        max_workers: int | None = None,
        **runner_kwargs: Any,
    ) -> pd.DataFrame:
        specs = self._selected_reference_specs(names)
        tasks = []
        for spec in specs:
            if spec.runner is None:
                raise RuntimeError(f"Reference {spec.name!r} runner is not configured.")
            tasks.extend(
                [
                    (spec, target)
                    for target in self._reference_targets(maps, spec, map_indexes=map_indexes)
                ]
            )
        if not tasks:
            return self.reference_frame()

        if parallel and len(tasks) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        self._checked_reference_runner(spec),
                        maps=target["maps_target"],
                        workflow=self,
                        reference=spec,
                        subject=maps,
                        map_index=target["map_index"],
                        system_name=target["system_name"],
                        **runner_kwargs,
                    )
                    for spec, target in tasks
                ]
                results = [future.result() for future in futures]
        else:
            results = [
                self._checked_reference_runner(spec)(
                    maps=target["maps_target"],
                    workflow=self,
                    reference=spec,
                    subject=maps,
                    map_index=target["map_index"],
                    system_name=target["system_name"],
                    **runner_kwargs,
                )
                for spec, target in tasks
            ]

        rows = []
        for (spec, target), result in zip(tasks, results, strict=False):
            metadata = self._coerce_runner_result_for_spec(spec, result)
            rows.append(
                self._upsert_reference_record(
                    spec=spec,
                    map_index=target["map_index"],
                    system_name=target["system_name"],
                    maps_target=target["maps_target"],
                    metadata=metadata,
                )
            )
        return pd.DataFrame(rows)

    def collect_references(
        self,
        maps: Any,
        *,
        names: Sequence[str] | None = None,
        map_indexes: Any = None,
        root: str | Path = "references",
        global_directory_template: str = "global/{reference_name}",
        per_map_directory_template: str = "map_{map_index}/{reference_name}",
        discover_outputs: bool = True,
        only_existing: bool = True,
        reset_label_status: bool = True,
        **parser_kwargs: Any,
    ) -> pd.DataFrame:
        specs = self._selected_reference_specs(names)
        rows = []
        for spec in specs:
            if spec.parser is None:
                raise RuntimeError(f"Reference {spec.name!r} parser is not configured.")
            for target in self._reference_targets(maps, spec, map_indexes=map_indexes):
                key = (spec.name, None if spec.scope == "global" else int(target["map_index"]))
                record = self.reference_records.get(key)
                if discover_outputs and (
                    record is None
                    or spec.file_column not in record
                    or record.get(spec.file_column) is None
                    or pd.isna(record.get(spec.file_column))
                ):
                    self.discover_reference_outputs(
                        maps,
                        names=[spec.name],
                        map_indexes=(
                            None if target["map_index"] is None else [int(target["map_index"])]
                        ),
                        root=root,
                        global_directory_template=global_directory_template,
                        per_map_directory_template=per_map_directory_template,
                        only_existing=only_existing,
                        reset_label_status=reset_label_status,
                    )
                    record = self.reference_records.get(key)
                if record is None:
                    raise ValueError(
                        f"No record found for reference {spec.name!r} and key {key!r}."
                    )
                output_file = record.get(spec.file_column)
                try:
                    if output_file is None or pd.isna(output_file):
                        raise ValueError(
                            f"No {spec.file_column} recorded for reference {spec.name!r}."
                        )
                    parsed = spec.parser(
                        output_file=output_file,
                        maps=target["maps_target"],
                        workflow=self,
                        reference=spec,
                        subject=maps,
                        map_index=target["map_index"],
                        system_name=target["system_name"],
                        **parser_kwargs,
                    )
                    metadata = self._coerce_parser_result_for_spec(spec, parsed)
                    if spec.postprocess is not None:
                        extra = spec.postprocess(
                            parsed=metadata.copy(),
                            maps=target["maps_target"],
                            workflow=self,
                            reference=spec,
                            subject=maps,
                            map_index=target["map_index"],
                            system_name=target["system_name"],
                            record=dict(record),
                            **parser_kwargs,
                        )
                        if extra:
                            metadata.update(dict(extra))
                    metadata.setdefault("label_status", "completed")
                    metadata["label_error"] = None
                except Exception as exc:
                    metadata = {
                        "label_status": "failed",
                        "label_error": str(exc),
                    }
                rows.append(
                    self._upsert_reference_record(
                        spec=spec,
                        map_index=target["map_index"],
                        system_name=target["system_name"],
                        maps_target=target["maps_target"],
                        metadata=metadata,
                    )
                )
        return pd.DataFrame(rows)

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

    def _get_reference_spec(self, name: str) -> WorkflowReferenceSpec:
        if name not in self.reference_specs:
            raise KeyError(f"Reference {name!r} is not registered on this workflow.")
        return self.reference_specs[name]

    def _selected_reference_specs(self, names: Sequence[str] | None) -> list[WorkflowReferenceSpec]:
        if names is None:
            return list(self.reference_specs.values())
        return [self._get_reference_spec(name) for name in names]

    def _reference_targets(
        self,
        maps: Any,
        spec: WorkflowReferenceSpec,
        *,
        map_indexes: Any = None,
    ) -> list[dict[str, Any]]:
        if spec.scope == "global":
            return [
                {
                    "map_index": None,
                    "system_name": None,
                    "maps_target": maps,
                }
            ]

        if hasattr(maps, "maps") and hasattr(maps, "names"):
            selected_indexes = (
                None
                if map_indexes is None
                else set(np.asarray(map_indexes, dtype=np.int64).reshape(-1).tolist())
            )
            targets = []
            for map_index, (name, child_maps) in enumerate(
                zip(maps.names, maps.maps, strict=False)
            ):
                if selected_indexes is not None and map_index not in selected_indexes:
                    continue
                targets.append(
                    {
                        "map_index": map_index,
                        "system_name": name,
                        "maps_target": child_maps,
                    }
                )
            return targets

        return [
            {
                "map_index": 0,
                "system_name": None,
                "maps_target": maps,
            }
        ]

    def _normalize_reference_outputs_by_map_index(
        self,
        output_files: Any,
        targets: list[dict[str, Any]],
    ) -> dict[int, Any]:
        if isinstance(output_files, dict):
            by_index: dict[int, Any] = {}
            name_to_index = {
                str(target["system_name"]): int(target["map_index"])
                for target in targets
                if target["map_index"] is not None
            }
            for key, value in output_files.items():
                if isinstance(key, str):
                    if key not in name_to_index:
                        raise KeyError(f"Unknown system name {key!r} in output_files.")
                    by_index[name_to_index[key]] = value
                else:
                    by_index[int(key)] = value
            return by_index

        values = np.asarray(output_files, dtype=object).reshape(-1)
        if values.size != len(targets):
            raise ValueError(
                f"output_files has length {values.size}, expected {len(targets)} entries."
            )
        return {
            int(target["map_index"]): value
            for target, value in zip(targets, values, strict=False)
            if target["map_index"] is not None
        }

    def _coerce_runner_result_for_spec(
        self,
        spec: WorkflowReferenceSpec,
        result: RunnerResult,
    ) -> dict[str, Any]:
        if isinstance(result, dict):
            return dict(result)
        return {spec.file_column: str(result) if result is not None else None}

    def _coerce_parser_result_for_spec(
        self,
        spec: WorkflowReferenceSpec,
        result: ParserResult,
    ) -> dict[str, Any]:
        if isinstance(result, dict):
            return dict(result)
        return {spec.scalar_output_column: result}

    def _checked_reference_runner(
        self,
        spec: WorkflowReferenceSpec,
    ) -> Callable[..., RunnerResult]:
        if spec.runner is None:
            raise RuntimeError(f"Reference {spec.name!r} runner is not configured.")
        return spec.runner

    def _upsert_reference_record(
        self,
        *,
        spec: WorkflowReferenceSpec,
        map_index: int | None,
        system_name: str | None,
        maps_target: Any,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        key = (spec.name, None if spec.scope == "global" else map_index)
        record = dict(self.reference_records.get(key, {}))
        record.update(
            {
                "name": spec.name,
                "scope": spec.scope,
                "map_index": map_index,
                "system_name": system_name,
                "reference_description": spec.description,
                **self._workflow_metadata(),
            }
        )
        record.update(dict(metadata))
        self.reference_records[key] = record
        return dict(record)

    def _expected_reference_output_path(
        self,
        *,
        spec: WorkflowReferenceSpec,
        target: dict[str, Any],
        root_path: Path,
        global_directory_template: str,
        per_map_directory_template: str,
    ) -> Path:
        if spec.scope == "global":
            relative = global_directory_template.format(reference_name=spec.name)
        else:
            relative = per_map_directory_template.format(
                reference_name=spec.name,
                map_index=int(target["map_index"]),
                system_name="" if target["system_name"] is None else str(target["system_name"]),
            )
        return root_path / relative


__all__ = ["CalculationWorkflow", "WorkflowReferenceSpec"]
