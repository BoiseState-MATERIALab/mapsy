import json
import logging
import os
import pickle
import time
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from copy import deepcopy
from inspect import Parameter, signature
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import dill
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.spatial.distance as distance
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from pathos.multiprocessing import ProcessingPool
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from yaml import SafeLoader, load

from mapsy.analysis import (
    ArchetypePropagationMode,
    ArchetypeSelectionMode,
    FeatureConnectivityMode,
    GraphMode,
    aggregate_cluster_graph,
    build_point_graph,
    fit_clusters,
    fit_pca_analysis,
    project_pca,
    propagate_cluster_labels,
    screen_clusters,
)
from mapsy.analysis import (
    propagate_archetypes as propagate_archetypes_on_graph,
)
from mapsy.analysis import (
    select_archetypes as select_feature_archetypes,
)
from mapsy.clustering import (
    clustering_uses_random_state,
    normalize_cluster_method,
)
from mapsy.results import (
    ArchetypePropagationResult,
    ArchetypeSelectionResult,
    ClusterResult,
    ClusterScreeningResult,
    GraphResult,
    PCAAnalysisResult,
    PCAResult,
)
from mapsy.selection import (
    normalize_point_selection_method,
    normalize_site_selection_method,
    pivoted_cholesky_rbf_selection,
    pivoted_qr_selection,
    selection_metadata_columns,
)

if TYPE_CHECKING:
    from .maps import Maps


logger = logging.getLogger(__name__)


def _coerce_layer_values(layer: int | Sequence[int]) -> list[int]:
    if isinstance(layer, (int, np.integer)):
        return [int(layer)]
    return [int(value) for value in layer]


def _nearest_reference_indexes(
    points: npt.NDArray[np.float64],
    references: npt.NDArray[np.float64],
) -> npt.NDArray[np.int64]:
    if references.shape[0] == 0:
        raise ValueError("references must contain at least one row.")
    nearest = NearestNeighbors(n_neighbors=1)
    nearest.fit(references)
    return nearest.kneighbors(points, return_distance=False).reshape(-1).astype(np.int64)


def _build_maps_from_file_worker(
    task: tuple[str, str, dict[str, Any], dict[str, Any], dict[str, Any]]
) -> Any:
    path, basepath, system_params, symmetryfunctions, contactspace = task

    from mapsy.io.parser import ContactSpaceGenerator, SystemParser
    from mapsy.maps import (
        Maps,
        _namespace_contactspace,
        _namespace_symmetryfunctions,
        _namespace_system,
    )
    from mapsy.symfunc.parser import SymmetryFunctionsParser

    system_parser = SystemParser(_namespace_system(system_params), basepath=basepath)
    system = system_parser.parse_file(path)
    parsed_symmetryfunctions = SymmetryFunctionsParser(
        _namespace_symmetryfunctions(symmetryfunctions)
    ).parse()
    parsed_contactspace = ContactSpaceGenerator(_namespace_contactspace(contactspace)).generate(
        system
    )
    return Maps(system, parsed_symmetryfunctions, parsed_contactspace)


def _resolve_multimaps_workers(control: Mapping[str, Any], nitems: int) -> int:
    if nitems <= 1:
        return 1

    raw_workers = None
    for key in ("workers", "nworkers", "n_workers", "n_jobs", "njobs", "max_workers"):
        if key in control:
            raw_workers = control[key]
            break

    if raw_workers is None:
        if not bool(control.get("parallel", False)):
            return 1
        available = os.cpu_count() or 1
        return max(1, min(4, available, nitems))

    workers = int(raw_workers)
    if workers <= 0:
        available = os.cpu_count() or 1
        return max(1, min(4, available, nitems))
    return max(1, min(workers, nitems))


def _write_map_feature_status(
    status_directory: str | None,
    map_index: int,
    status: str,
    **metadata: Any,
) -> None:
    if status_directory is None:
        return
    directory = Path(status_directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"map_{map_index:05d}.status.json"
    tmp_path = path.with_name(f".{path.name}.tmp")
    payload = {
        "map_index": map_index,
        "status": status,
        "pid": os.getpid(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        **metadata,
    }
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    tmp_path.replace(path)


def _threadpool_context(threadpool_threads: int | None) -> AbstractContextManager[Any]:
    if threadpool_threads is None or threadpool_threads <= 0:
        return nullcontext()
    return cast(
        AbstractContextManager[Any],
        threadpool_limits(limits=int(threadpool_threads)),
    )


def _call_maps_atcontactspace(
    maps: Any,
    *,
    workers: int | None,
    release_contactspace_cache: bool,
) -> pd.DataFrame:
    method = maps.atcontactspace
    parameters = signature(method).parameters
    accepts_kwargs = any(
        parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    kwargs: dict[str, Any] = {}
    if accepts_kwargs or "workers" in parameters:
        kwargs["workers"] = workers
    if accepts_kwargs or "release_contactspace_cache" in parameters:
        kwargs["release_contactspace_cache"] = release_contactspace_cache
    return cast(pd.DataFrame, method(**kwargs))


def _compute_map_features_worker(
    task: tuple[int, Any, int | None, bool, str | None, int | None],
) -> tuple[int, Any, float, int]:
    (
        map_index,
        maps,
        point_workers,
        release_contactspace_cache,
        status_directory,
        threadpool_threads,
    ) = task
    start = time.perf_counter()
    npoints = 0 if maps.contactspace is None else len(maps.contactspace.data)
    _write_map_feature_status(
        status_directory,
        map_index,
        "running",
        point_workers=point_workers,
        threadpool_threads=threadpool_threads,
        npoints=npoints,
    )
    try:
        with _threadpool_context(threadpool_threads):
            _call_maps_atcontactspace(
                maps,
                workers=point_workers,
                release_contactspace_cache=release_contactspace_cache,
            )
    except Exception as exc:
        _write_map_feature_status(
            status_directory,
            map_index,
            "failed",
            elapsed_seconds=time.perf_counter() - start,
            error=repr(exc),
        )
        raise
    elapsed = time.perf_counter() - start
    npoints = 0 if maps.data is None else len(maps.data)
    _write_map_feature_status(
        status_directory,
        map_index,
        "done",
        elapsed_seconds=elapsed,
        npoints=npoints,
    )
    return map_index, maps, elapsed, npoints


class MultiMaps:
    _METADATA_COLUMNS = (
        "region",
        "signed_distance",
        "core_distance",
        "patch",
        "layer",
        "patch_size",
        "layer_size",
        "patch_mean_distance",
        "layer_mean_distance",
        "source_file",
        "source_file_name",
        "source_file_stem",
        "source_folder",
        "source_folder_name",
        "source_folder_number",
    )

    def __init__(
        self,
        maps: Sequence["Maps"],
        names: Sequence[str] | None = None,
        metadata: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        if not maps:
            raise ValueError("maps must contain at least one Maps instance")

        self.maps = list(maps)
        self.names = (
            list(names)
            if names is not None
            else [f"system_{index}" for index in range(len(self.maps))]
        )
        if len(self.names) != len(self.maps):
            raise ValueError("names and maps must have the same length")
        self.map_metadata = (
            [dict(item) for item in metadata] if metadata is not None else [{} for _ in self.maps]
        )
        if len(self.map_metadata) != len(self.maps):
            raise ValueError("metadata and maps must have the same length")

        self.data: pd.DataFrame | None = None
        self.features: list[str] = []
        self.npca: int | None = None
        self.pca_analysis_result: PCAAnalysisResult | None = None
        self.pca_result: PCAResult | None = None
        self.graph_result: GraphResult | None = None
        self.archetype_selection_result: ArchetypeSelectionResult | None = None
        self.archetype_propagation_result: ArchetypePropagationResult | None = None

        self.nclusters: int = 0
        self.cluster_method: str = "spectral"
        self.cluster_screening_method: str | None = None
        self.cluster_result: ClusterResult | None = None
        self.cluster_features: list[str] = []
        self.cluster_centers: np.ndarray[Any, np.dtype[np.float64]] | None = None
        self.cluster_graph: np.ndarray[Any, np.dtype[np.int64]] | None = None
        self.cluster_edges: np.ndarray[Any, np.dtype[np.int64]] | None = None
        self.best_clusters: pd.DataFrame | None = None
        self.cluster_screening: pd.DataFrame | None = None

        self._slices: list[slice] = []
        self._special_points = pd.DataFrame(
            columns=["global_point_index", "kind", "iteration", "label_status"]
        )

    def atcontactspace(
        self,
        *,
        recompute: bool = True,
        parallel: str | bool = "points",
        workers: int | None = None,
        point_workers: int | None = None,
        checkpoint: str | Path | None = None,
        checkpoint_every: int = 1,
        map_indexes: npt.ArrayLike | None = None,
        release_contactspace_cache: bool = False,
        checkpoint_mode: str = "full",
        progress: bool = False,
        threadpool_threads: int | None = None,
    ) -> pd.DataFrame:
        """Compute child-map features with optional map-level parallelism and checkpoints."""
        mode = self._normalize_feature_parallelism(parallel)
        checkpoint_mode = self._normalize_checkpoint_mode(checkpoint_mode)
        if checkpoint is not None and checkpoint_mode == "maps" and not recompute:
            self._load_map_feature_checkpoints(checkpoint, map_indexes=map_indexes)

        targets = self._feature_target_indexes(
            recompute=recompute,
            map_indexes=map_indexes,
        )
        self._feature_progress(
            progress,
            (
                f"feature run: mode={mode}, targets={len(targets)}/{len(self.maps)}, "
                f"checkpoint_mode={checkpoint_mode}"
            ),
        )
        if release_contactspace_cache:
            self.release_contactspace_cache(
                map_indexes=self._feature_target_indexes(
                    recompute=True,
                    map_indexes=map_indexes,
                )
            )

        if mode == "maps":
            map_workers = self._resolve_feature_workers(workers, len(targets))
            if map_workers > 1 and targets:
                child_point_workers = 1 if point_workers is None else point_workers
                child_threadpool_threads = 1 if threadpool_threads is None else threadpool_threads
                status_directory = (
                    str(self._map_checkpoint_directory(checkpoint))
                    if checkpoint is not None and checkpoint_mode == "maps"
                    else None
                )
                self._feature_progress(
                    progress,
                    (
                        f"launching {len(targets)} map jobs with map_workers={map_workers}, "
                        f"point_workers={child_point_workers}, "
                        f"threadpool_threads={child_threadpool_threads}"
                    ),
                )
                tasks = (
                    (
                        index,
                        self.maps[index],
                        child_point_workers,
                        release_contactspace_cache,
                        status_directory,
                        child_threadpool_threads,
                    )
                    for index in targets
                )
                with ProcessingPool(nodes=map_workers) as pool:
                    for completed, result in enumerate(
                        pool.uimap(_compute_map_features_worker, tasks),
                        start=1,
                    ):
                        index, maps, elapsed, npoints = result
                        self.maps[index] = maps
                        self._feature_progress(
                            progress,
                            (
                                f"finished map {index} ({self._map_name(index)}) "
                                f"{completed}/{len(targets)} in {elapsed:.1f}s "
                                f"({npoints} points)"
                            ),
                        )
                        if checkpoint is not None and checkpoint_mode == "maps":
                            checkpoint_start = time.perf_counter()
                            path = self._save_map_feature_checkpoint(checkpoint, index)
                            self._feature_progress(
                                progress,
                                (
                                    f"checkpointed map {index} in "
                                    f"{time.perf_counter() - checkpoint_start:.1f}s -> {path}"
                                ),
                            )
            else:
                serial_point_workers = 1 if point_workers is None else point_workers
                self._compute_feature_targets_serial(
                    targets,
                    point_workers=serial_point_workers,
                    checkpoint=checkpoint,
                    checkpoint_every=checkpoint_every,
                    release_contactspace_cache=release_contactspace_cache,
                    checkpoint_mode=checkpoint_mode,
                    progress=progress,
                    threadpool_threads=threadpool_threads,
                )
        else:
            selected_point_workers = (
                1 if mode == "none" else workers if point_workers is None else point_workers
            )
            self._compute_feature_targets_serial(
                targets,
                point_workers=selected_point_workers,
                checkpoint=checkpoint,
                checkpoint_every=checkpoint_every,
                release_contactspace_cache=release_contactspace_cache,
                checkpoint_mode=checkpoint_mode,
                progress=progress,
                threadpool_threads=threadpool_threads,
            )

        if not self._all_maps_have_data():
            self.data = None
            if checkpoint is not None and checkpoint_mode == "full":
                checkpoint_start = time.perf_counter()
                path = self._save_feature_checkpoint(checkpoint)
                self._feature_progress(
                    progress,
                    (
                        f"wrote partial checkpoint in "
                        f"{time.perf_counter() - checkpoint_start:.1f}s -> {path}"
                    ),
                )
            self._feature_progress(progress, "feature run incomplete; returning partial dataset")
            return self._partial_feature_dataset()

        data = self._build_dataset(recompute=False)
        if checkpoint is not None:
            checkpoint_start = time.perf_counter()
            path = self._save_feature_checkpoint(checkpoint)
            self._feature_progress(
                progress,
                (
                    f"wrote final checkpoint in "
                    f"{time.perf_counter() - checkpoint_start:.1f}s -> {path}"
                ),
            )
        self._feature_progress(progress, f"feature run complete: {len(data)} total points")
        return data

    def release_contactspace_cache(self, map_indexes: npt.ArrayLike | None = None) -> None:
        """Release dense contact-space fields from all child maps."""
        indexes = self._feature_target_indexes(
            recompute=True,
            map_indexes=map_indexes,
        )
        for index in indexes:
            maps = self.maps[index]
            release = getattr(maps, "release_contactspace_cache", None)
            if release is not None:
                release()

    def _map_name(self, index: int) -> str:
        if index < len(self.names):
            return self.names[index]
        return f"map_{index}"

    @staticmethod
    def _feature_progress(enabled: bool, message: str) -> None:
        if enabled:
            logger.info("[mapsy] %s", message)

    def save(
        self,
        filename: str | Path,
        *,
        protocol: int = pickle.HIGHEST_PROTOCOL,
    ) -> Path:
        path = Path(filename).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.tmp")
        with tmp_path.open("wb") as handle:
            dill.dump(self, handle, protocol=protocol)
        tmp_path.replace(path)
        return path

    @classmethod
    def load(cls, filename: str | Path) -> "MultiMaps":
        path = Path(filename).expanduser().resolve()
        with path.open("rb") as handle:
            try:
                loaded = dill.load(handle)
            except Exception:
                handle.seek(0)
                loaded = pickle.load(handle)
        if not isinstance(loaded, cls):
            raise TypeError(
                f"Pickle at {path} contains {type(loaded).__name__}, expected {cls.__name__}."
            )
        return loaded

    def scatter(
        self,
        feature: str | None = None,
        index: int | None = None,
        axes: Sequence[str] | None = None,
        region: int | None = 0,
        *,
        layer: int | Sequence[int] | None = None,
        map_indexes: npt.ArrayLike | None = None,
        categorical: bool = False,
        centroids: bool = False,
        categories: Sequence[Any] | None = None,
        ncols: int | None = None,
        figsize: tuple[float, float] | None = None,
        panel_size: tuple[float, float] = (4.5, 4.0),
        cmap: str | mpl.colors.Colormap | None = None,
        colorbar: bool = True,
        legend: bool = True,
        sharex: bool = False,
        sharey: bool = False,
        set_aspect: str = "scaled",
        **kwargs: Any,
    ) -> tuple[Figure, npt.NDArray[np.object_]]:
        """Scatter a feature for multiple maps with one shared color scale or legend."""
        data = self._ensure_data()
        frame = data.copy()
        if feature is not None:
            if feature not in frame.columns:
                frame.loc[:, feature] = self._collect_contactspace_column(feature)
        elif index is not None:
            if index >= len(self.features) or index < 0:
                raise ValueError(f"Index {index} out of bounds.")
            feature = self.features[index]
            logger.info("Plotting feature %s", feature)
        else:
            raise ValueError("Either feature or index must be provided.")
        if feature not in frame.columns:
            raise ValueError(f"Feature {feature} not found in multimaps data.")

        resolved_axes = list(axes) if axes is not None else ["x", "y"]
        if len(resolved_axes) != 2:
            raise ValueError("axes must contain exactly two columns.")
        for axis in resolved_axes:
            if axis not in frame.columns:
                frame.loc[:, axis] = self._collect_contactspace_column(axis)

        if set_aspect not in ["on", "off", "equal", "scaled"]:
            raise ValueError("set_aspect must be one of ['on','off','equal','scaled']")

        if map_indexes is None:
            selected_maps = np.arange(len(self.maps), dtype=np.int64)
        else:
            selected_maps = np.asarray(map_indexes, dtype=np.int64).reshape(-1)
        if selected_maps.size == 0:
            raise ValueError("map_indexes must select at least one map.")
        invalid_maps = selected_maps[(selected_maps < 0) | (selected_maps >= len(self.maps))]
        if invalid_maps.size:
            raise ValueError(f"map_indexes contains invalid indexes: {invalid_maps.tolist()}.")

        if "map_index" not in frame.columns:
            raise RuntimeError("MultiMaps data is missing the 'map_index' column.")
        mask = np.isin(frame["map_index"].to_numpy(dtype=np.int64), selected_maps)

        if region is not None:
            if "region" not in frame.columns:
                frame.loc[:, "region"] = self._collect_contactspace_column("region").astype(
                    np.int64
                )
            mask &= frame["region"].to_numpy(dtype=np.int64) == int(region)

        if layer is not None:
            if "layer" not in frame.columns:
                frame.loc[:, "layer"] = self._collect_contactspace_column("layer").astype(np.int64)
            mask &= np.isin(frame["layer"].to_numpy(dtype=np.int64), _coerce_layer_values(layer))

        plotdata = frame.loc[mask].copy()
        if plotdata.empty:
            raise ValueError("No points available after applying the map/region/layer filters.")

        nmaps = int(selected_maps.size)
        if ncols is None:
            ncols = min(4, int(np.ceil(np.sqrt(nmaps))))
        ncols = max(1, min(int(ncols), nmaps))
        nrows = int(np.ceil(nmaps / ncols))
        if figsize is None:
            extra_width = 1.7 if categorical and legend else 0.8 if colorbar else 0.0
            figsize = (panel_size[0] * ncols + extra_width, panel_size[1] * nrows)

        fig, axs = plt.subplots(
            nrows,
            ncols,
            figsize=figsize,
            sharex=sharex,
            sharey=sharey,
            squeeze=False,
        )
        axslist = axs.reshape(-1)

        scatter_kwargs = dict(kwargs)
        scatter_artist = None
        legend_handles: list[Line2D] = []
        centroid_frame = pd.DataFrame()
        if centroids:
            centroid_frame = self.get_special_points(kind="centroid", label_status=None)
            if centroid_frame.empty:
                raise RuntimeError("No centroids available.")
            if region is not None and "region" in centroid_frame.columns:
                centroid_frame = centroid_frame.loc[
                    centroid_frame["region"].to_numpy(dtype=np.int64) == int(region)
                ]
            if layer is not None and "layer" in centroid_frame.columns:
                centroid_frame = centroid_frame.loc[
                    np.isin(
                        centroid_frame["layer"].to_numpy(dtype=np.int64),
                        _coerce_layer_values(layer),
                    )
                ]

        if categorical:
            scatter_kwargs.pop("c", None)
            scatter_kwargs.pop("color", None)
            observed_categories = pd.unique(plotdata[feature].dropna())
            if categories is not None:
                category_values = np.array(list(categories), dtype=object)
            elif feature == "Cluster" and self.nclusters > 0:
                category_values = np.arange(self.nclusters, dtype=np.int64)
                observed_numeric = np.asarray(observed_categories, dtype=np.float64)
                if observed_numeric.size and np.any(observed_numeric < 0):
                    category_values = np.concatenate(
                        [np.array([-1], dtype=np.int64), category_values]
                    )
            else:
                category_values = observed_categories
            if category_values.size == 0:
                raise ValueError(f"Feature {feature!r} does not contain plottable values.")
            if np.issubdtype(category_values.dtype, np.number):
                category_values = np.sort(category_values.astype(np.float64, copy=False))
            color_map = (
                cmap if isinstance(cmap, mpl.colors.Colormap) else mpl.colormaps[cmap or "tab20"]
            )
            if feature == "Cluster" and self.nclusters > 0:
                cluster_colors = color_map(np.linspace(0, 1, self.nclusters, endpoint=False))
                fallback_colors = color_map(np.linspace(0, 1, len(category_values), endpoint=False))
                category_colors = {}
                for fallback_index, category in enumerate(category_values):
                    category_color = fallback_colors[fallback_index]
                    if isinstance(category, (int, float, np.integer, np.floating)):
                        category_number = float(category)
                        if category_number.is_integer():
                            cluster_index = int(category_number)
                            if cluster_index == -1:
                                category_color = (0.6, 0.6, 0.6, 1.0)
                            elif 0 <= cluster_index < self.nclusters:
                                category_color = cluster_colors[cluster_index]
                    category_colors[category] = category_color
            else:
                colors = color_map(np.linspace(0, 1, len(category_values), endpoint=False))
                category_colors = dict(zip(category_values.tolist(), colors, strict=False))
            for category in category_values:
                label = (
                    str(int(category))
                    if isinstance(category, float) and category.is_integer()
                    else str(category)
                )
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        linestyle="",
                        markerfacecolor=category_colors[category],
                        markeredgecolor="none",
                        label=f"{feature} = {label}",
                    )
                )
        else:
            values = plotdata[feature].to_numpy(dtype=np.float64)
            if "norm" in scatter_kwargs:
                norm = scatter_kwargs.pop("norm")
            else:
                vmin = scatter_kwargs.pop("vmin", float(np.nanmin(values)))
                vmax = scatter_kwargs.pop("vmax", float(np.nanmax(values)))
                norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            color_map = (
                cmap
                if isinstance(cmap, mpl.colors.Colormap)
                else mpl.colormaps[cmap or scatter_kwargs.pop("cmap", "viridis")]
            )

        for ax, map_index in zip(axslist, selected_maps, strict=False):
            map_data = plotdata.loc[plotdata["map_index"].to_numpy(dtype=np.int64) == map_index]
            ax.set_title(self._map_name(int(map_index)))
            ax.set_xlabel(resolved_axes[0])
            ax.set_ylabel(resolved_axes[1])
            if map_data.empty:
                ax.axis(set_aspect)
                continue

            x1 = map_data[resolved_axes[0]].to_numpy(dtype=np.float64)
            x2 = map_data[resolved_axes[1]].to_numpy(dtype=np.float64)
            if categorical:
                map_values = map_data[feature].to_numpy()
                for category in category_values:
                    category_mask = map_values == category
                    if not np.any(category_mask):
                        continue
                    scatter_artist = ax.scatter(
                        x1[category_mask],
                        x2[category_mask],
                        color=category_colors[category],
                        **scatter_kwargs,
                    )
            else:
                scatter_artist = ax.scatter(
                    x1,
                    x2,
                    c=map_data[feature].to_numpy(dtype=np.float64),
                    cmap=color_map,
                    norm=norm,
                    **scatter_kwargs,
                )

            if centroids:
                map_centroids = centroid_frame.loc[
                    centroid_frame["map_index"].to_numpy(dtype=np.int64) == map_index
                ]
                if not map_centroids.empty:
                    ax.scatter(
                        map_centroids[resolved_axes[0]].to_numpy(dtype=np.float64),
                        map_centroids[resolved_axes[1]].to_numpy(dtype=np.float64),
                        c="black",
                        marker="x",
                        s=max(float(scatter_kwargs.get("s", 20)) * 2.0, 30.0),
                        label="Centroids",
                    )
            ax.axis(set_aspect)

        for ax in axslist[nmaps:]:
            ax.axis("off")

        visible_axes = axslist[:nmaps].tolist()
        if categorical:
            if centroids:
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="x",
                        linestyle="",
                        color="black",
                        label="Centroids",
                    )
                )
            if legend:
                fig.legend(
                    handles=legend_handles,
                    bbox_to_anchor=(0.995, 0.5),
                    loc="center right",
                    borderaxespad=0.0,
                )
                fig.tight_layout(rect=(0.0, 0.0, 0.86, 1.0))
            else:
                fig.tight_layout()
        else:
            fig.tight_layout()
            if colorbar and scatter_artist is not None:
                cbar = fig.colorbar(scatter_artist, ax=visible_axes)
                cbar.set_label(feature)
                cbar.solids.set_alpha(1.0)
        return fig, axs

    def min_projection(
        self,
        feature: str | None = None,
        index: int | None = None,
        *,
        minimize: str | None = None,
        plane: tuple[str, str] = ("x", "y"),
        region: int | None = 0,
        layer: int | Sequence[int] | None = None,
        map_indexes: npt.ArrayLike | None = None,
        coordinate_decimals: int | None = None,
    ) -> pd.DataFrame:
        """Return per-map plane projections selected by the minimum value along the normal."""
        self._ensure_data()
        if map_indexes is None:
            selected_maps = np.arange(len(self.maps), dtype=np.int64)
        else:
            selected_maps = np.asarray(map_indexes, dtype=np.int64).reshape(-1)
        if selected_maps.size == 0:
            raise ValueError("map_indexes must select at least one map.")

        frames: list[pd.DataFrame] = []
        for map_index in selected_maps:
            if map_index < 0 or map_index >= len(self.maps):
                raise ValueError(
                    f"map index {int(map_index)} out of range for {len(self.maps)} maps"
                )
            maps = self.maps[int(map_index)]
            try:
                projected = maps.min_projection(
                    feature=feature,
                    index=index,
                    minimize=minimize,
                    plane=plane,
                    region=region,
                    layer=layer,
                    coordinate_decimals=coordinate_decimals,
                ).copy()
            except ValueError as exc:
                if "No points available after applying the projection filters" in str(exc):
                    continue
                raise

            projected.insert(0, "system", self._map_name(int(map_index)))
            projected.insert(1, "map_index", int(map_index))
            if "point_index" in projected.columns:
                global_indexes = (
                    projected["point_index"].to_numpy(dtype=np.int64)
                    + self._slices[int(map_index)].start
                )
                projected.insert(2, "global_point_index", global_indexes)
            frames.append(projected)

        return self._combine_frames(frames)

    def core_projection(
        self,
        feature: str | None = None,
        index: int | None = None,
        *,
        plane: tuple[str, str] = ("x", "y"),
        selector: str = "core",
        distance_column: str = "core_distance",
        region: int | None = 0,
        layer: int | Sequence[int] | None = 0,
        map_indexes: npt.ArrayLike | None = None,
        categorical: bool = False,
    ) -> pd.DataFrame:
        """Return per-map plane projections selected by distance, height, or weighted mean."""
        frame = self._ensure_data().copy()
        if feature is not None:
            resolved_feature = feature
        elif index is not None:
            if index >= len(self.features) or index < 0:
                raise ValueError(f"Index {index} out of bounds.")
            resolved_feature = self.features[index]
            logger.info("Projecting feature %s", resolved_feature)
        else:
            raise ValueError("Either feature or index must be provided.")
        if len(plane) != 2 or plane[0] == plane[1]:
            raise ValueError(f"plane must contain two distinct axes, got {plane!r}.")

        center_selectors = {"core", "center", "distance"}
        valid_selectors = center_selectors | {"top", "bottom", "weighted_mean"}
        if selector not in valid_selectors:
            raise ValueError(
                "selector must be one of "
                "{'core', 'center', 'distance', 'top', 'bottom', 'weighted_mean'}."
            )
        if categorical and selector == "weighted_mean":
            raise ValueError("selector='weighted_mean' is only valid for numeric features.")

        remaining_axes = [axis for axis in ["x", "y", "z"] if axis not in plane]
        if len(remaining_axes) != 1:
            raise ValueError(f"plane must be a subset of ('x', 'y', 'z'), got {plane!r}.")
        normal_axis = remaining_axes[0]

        if map_indexes is None:
            selected_maps = np.arange(len(self.maps), dtype=np.int64)
        else:
            selected_maps = np.asarray(map_indexes, dtype=np.int64).reshape(-1)
        if selected_maps.size == 0:
            raise ValueError("map_indexes must select at least one map.")
        invalid_maps = selected_maps[(selected_maps < 0) | (selected_maps >= len(self.maps))]
        if invalid_maps.size:
            raise ValueError(f"map_indexes contains invalid indexes: {invalid_maps.tolist()}.")

        required_columns = {
            "system",
            "map_index",
            "point_index",
            resolved_feature,
            *plane,
            normal_axis,
            distance_column,
            "probability",
        }
        if region is not None:
            required_columns.add("region")
        if layer is not None:
            required_columns.add("layer")
        for column in required_columns:
            if column in frame.columns:
                continue
            frame.loc[:, column] = self._collect_contactspace_column(column)

        mask = np.isin(frame["map_index"].to_numpy(dtype=np.int64), selected_maps)
        if region is not None:
            mask &= frame["region"].to_numpy(dtype=np.int64) == int(region)
        if layer is not None:
            mask &= np.isin(frame["layer"].to_numpy(dtype=np.int64), _coerce_layer_values(layer))

        columns = [
            "system",
            "map_index",
            "point_index",
            plane[0],
            plane[1],
            normal_axis,
            resolved_feature,
            distance_column,
            "probability",
        ]
        projected_input = frame.loc[mask, list(dict.fromkeys(columns))].copy()
        if projected_input.empty:
            raise ValueError("No points available after applying the projection filters.")

        group_columns = ["map_index", plane[0], plane[1]]
        if selector == "weighted_mean":
            feature_values = pd.to_numeric(projected_input[resolved_feature], errors="raise")
            projected_input.loc[:, resolved_feature] = feature_values.to_numpy(dtype=np.float64)
            groups: list[dict[str, Any]] = []
            for coords, group in projected_input.groupby(group_columns, sort=False, dropna=False):
                weights = group["probability"].to_numpy(dtype=np.float64)
                if np.allclose(weights.sum(), 0.0):
                    weights = np.ones(len(group), dtype=np.float64)
                first = group.iloc[0]
                row = {
                    "system": first["system"],
                    "map_index": int(coords[0]),
                    "point_index": int(first["point_index"]),
                    plane[0]: coords[1],
                    plane[1]: coords[2],
                    normal_axis: np.average(
                        group[normal_axis].to_numpy(dtype=np.float64),
                        weights=weights,
                    ),
                    resolved_feature: np.average(
                        group[resolved_feature].to_numpy(dtype=np.float64),
                        weights=weights,
                    ),
                    distance_column: float(
                        np.min(group[distance_column].to_numpy(dtype=np.float64))
                    ),
                    "probability": float(np.max(weights)),
                    "multiplicity": int(len(group)),
                }
                groups.append(row)
            projected = pd.DataFrame(groups)
        else:
            ascending = {
                "core": [True, False, False],
                "center": [True, False, False],
                "distance": [True, False, False],
                "top": [False, True, False],
                "bottom": [True, True, False],
            }[selector]
            sort_columns = {
                "core": [distance_column, "probability", normal_axis],
                "center": [distance_column, "probability", normal_axis],
                "distance": [distance_column, "probability", normal_axis],
                "top": [normal_axis, distance_column, "probability"],
                "bottom": [normal_axis, distance_column, "probability"],
            }[selector]
            ordered = projected_input.sort_values(
                sort_columns, ascending=ascending, kind="mergesort"
            )
            projected = ordered.groupby(group_columns, sort=False, as_index=False).first()
            counts = (
                projected_input.groupby(group_columns, sort=False)
                .size()
                .rename("multiplicity")
                .reset_index()
            )
            projected = projected.merge(counts, on=group_columns, how="left")

        projected.loc[:, "global_point_index"] = projected["point_index"].to_numpy(
            dtype=np.int64
        ) + np.array(
            [self._slices[int(map_index)].start for map_index in projected["map_index"]],
            dtype=np.int64,
        )
        projected.loc[:, "projection_selector"] = selector
        projected.loc[:, "projection_distance_column"] = distance_column
        return projected

    def scatter_core_projection(
        self,
        feature: str | None = None,
        index: int | None = None,
        *,
        plane: tuple[str, str] = ("x", "y"),
        selector: str = "core",
        distance_column: str = "core_distance",
        region: int | None = 0,
        layer: int | Sequence[int] | None = 0,
        map_indexes: npt.ArrayLike | None = None,
        categorical: bool = False,
        categories: Sequence[Any] | None = None,
        ncols: int | None = None,
        figsize: tuple[float, float] | None = None,
        panel_size: tuple[float, float] = (4.5, 4.0),
        cmap: str | mpl.colors.Colormap | None = None,
        colorbar: bool = True,
        legend: bool = True,
        sharex: bool = False,
        sharey: bool = False,
        set_aspect: str = "scaled",
        return_projection: bool = False,
        **kwargs: Any,
    ) -> (
        tuple[Figure, npt.NDArray[np.object_]]
        | tuple[Figure, npt.NDArray[np.object_], pd.DataFrame]
    ):
        """Scatter core-distance plane projections for multiple maps."""
        if set_aspect not in ["on", "off", "equal", "scaled"]:
            raise ValueError("set_aspect must be one of ['on','off','equal','scaled']")
        if feature is not None:
            resolved_feature = feature
        elif index is not None:
            if index >= len(self.features) or index < 0:
                raise ValueError(f"Index {index} out of bounds.")
            resolved_feature = self.features[index]
        else:
            raise ValueError("Either feature or index must be provided.")

        projection = self.core_projection(
            feature=feature,
            index=index,
            plane=plane,
            selector=selector,
            distance_column=distance_column,
            region=region,
            layer=layer,
            map_indexes=map_indexes,
            categorical=categorical,
        )
        if projection.empty:
            raise ValueError("No points available after applying the projection filters.")

        if map_indexes is None:
            selected_maps = np.arange(len(self.maps), dtype=np.int64)
        else:
            selected_maps = np.asarray(map_indexes, dtype=np.int64).reshape(-1)
        nmaps = int(selected_maps.size)
        if ncols is None:
            ncols = min(4, int(np.ceil(np.sqrt(nmaps))))
        ncols = max(1, min(int(ncols), nmaps))
        nrows = int(np.ceil(nmaps / ncols))
        if figsize is None:
            extra_width = 1.7 if categorical and legend else 0.8 if colorbar else 0.0
            figsize = (panel_size[0] * ncols + extra_width, panel_size[1] * nrows)

        fig, axs = plt.subplots(
            nrows,
            ncols,
            figsize=figsize,
            sharex=sharex,
            sharey=sharey,
            squeeze=False,
        )
        axslist = axs.reshape(-1)

        scatter_kwargs = dict(kwargs)
        scatter_artist = None
        legend_handles: list[Line2D] = []
        if categorical:
            scatter_kwargs.pop("c", None)
            scatter_kwargs.pop("color", None)
            observed_categories = pd.unique(projection[resolved_feature].dropna())
            if categories is not None:
                category_values = np.array(list(categories), dtype=object)
            elif resolved_feature == "Cluster" and self.nclusters > 0:
                category_values = np.arange(self.nclusters, dtype=np.int64)
                observed_numeric = np.asarray(observed_categories, dtype=np.float64)
                if observed_numeric.size and np.any(observed_numeric < 0):
                    category_values = np.concatenate(
                        [np.array([-1], dtype=np.int64), category_values]
                    )
            else:
                category_values = observed_categories
            if category_values.size == 0:
                raise ValueError(f"Feature {resolved_feature!r} does not contain plottable values.")
            if np.issubdtype(category_values.dtype, np.number):
                category_values = np.sort(category_values.astype(np.float64, copy=False))
            color_map = (
                cmap if isinstance(cmap, mpl.colors.Colormap) else mpl.colormaps[cmap or "tab20"]
            )
            if resolved_feature == "Cluster" and self.nclusters > 0:
                cluster_colors = color_map(np.linspace(0, 1, self.nclusters, endpoint=False))
                fallback_colors = color_map(np.linspace(0, 1, len(category_values), endpoint=False))
                category_colors = {}
                for fallback_index, category in enumerate(category_values):
                    category_color = fallback_colors[fallback_index]
                    if isinstance(category, (int, float, np.integer, np.floating)):
                        category_number = float(category)
                        if category_number.is_integer():
                            cluster_index = int(category_number)
                            if cluster_index == -1:
                                category_color = (0.6, 0.6, 0.6, 1.0)
                            elif 0 <= cluster_index < self.nclusters:
                                category_color = cluster_colors[cluster_index]
                    category_colors[category] = category_color
            else:
                colors = color_map(np.linspace(0, 1, len(category_values), endpoint=False))
                category_colors = dict(zip(category_values.tolist(), colors, strict=False))
            for category in category_values:
                label = (
                    str(int(category))
                    if isinstance(category, float) and category.is_integer()
                    else str(category)
                )
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        linestyle="",
                        markerfacecolor=category_colors[category],
                        markeredgecolor="none",
                        label=f"{resolved_feature} = {label}",
                    )
                )
        else:
            values = projection[resolved_feature].to_numpy(dtype=np.float64)
            if "norm" in scatter_kwargs:
                norm = scatter_kwargs.pop("norm")
            else:
                vmin = scatter_kwargs.pop("vmin", float(np.nanmin(values)))
                vmax = scatter_kwargs.pop("vmax", float(np.nanmax(values)))
                norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            color_map = (
                cmap
                if isinstance(cmap, mpl.colors.Colormap)
                else mpl.colormaps[cmap or scatter_kwargs.pop("cmap", "viridis")]
            )

        for ax, map_index in zip(axslist, selected_maps, strict=False):
            map_projection = projection.loc[
                projection["map_index"].to_numpy(dtype=np.int64) == int(map_index)
            ]
            ax.set_title(self._map_name(int(map_index)))
            ax.set_xlabel(plane[0])
            ax.set_ylabel(plane[1])
            if map_projection.empty:
                ax.axis(set_aspect)
                continue

            x = map_projection[plane[0]].to_numpy(dtype=np.float64)
            y = map_projection[plane[1]].to_numpy(dtype=np.float64)
            if categorical:
                map_values = map_projection[resolved_feature].to_numpy()
                for category in category_values:
                    category_mask = map_values == category
                    if not np.any(category_mask):
                        continue
                    scatter_artist = ax.scatter(
                        x[category_mask],
                        y[category_mask],
                        color=category_colors[category],
                        **scatter_kwargs,
                    )
            else:
                scatter_artist = ax.scatter(
                    x,
                    y,
                    c=map_projection[resolved_feature].to_numpy(dtype=np.float64),
                    cmap=color_map,
                    norm=norm,
                    **scatter_kwargs,
                )
            ax.axis(set_aspect)

        for ax in axslist[nmaps:]:
            ax.axis("off")

        center_selectors = {"core", "center", "distance"}
        title_prefix = (
            f"Closest {distance_column}"
            if selector in center_selectors
            else selector.replace("_", " ").title()
        )
        fig.suptitle(f"{title_prefix} projection of {resolved_feature} on {plane[0]}-{plane[1]}")
        visible_axes = axslist[:nmaps].tolist()
        if categorical:
            if legend:
                fig.legend(
                    handles=legend_handles,
                    bbox_to_anchor=(0.995, 0.5),
                    loc="center right",
                    borderaxespad=0.0,
                )
                fig.tight_layout(rect=(0.0, 0.0, 0.86, 1.0))
            else:
                fig.tight_layout()
        else:
            fig.tight_layout()
            if colorbar and scatter_artist is not None:
                cbar = fig.colorbar(scatter_artist, ax=visible_axes)
                cbar.set_label(resolved_feature)
                cbar.solids.set_alpha(1.0)
        if return_projection:
            return fig, axs, projection
        return fig, axs

    def scatter_min_projection(
        self,
        feature: str | None = None,
        index: int | None = None,
        *,
        minimize: str | None = None,
        plane: tuple[str, str] = ("x", "y"),
        region: int | None = 0,
        layer: int | Sequence[int] | None = None,
        map_indexes: npt.ArrayLike | None = None,
        coordinate_decimals: int | None = None,
        ncols: int | None = None,
        figsize: tuple[float, float] | None = None,
        panel_size: tuple[float, float] = (4.5, 4.0),
        cmap: str | mpl.colors.Colormap | None = None,
        colorbar: bool = True,
        sharex: bool = False,
        sharey: bool = False,
        set_aspect: str = "scaled",
        return_projection: bool = False,
        **kwargs: Any,
    ) -> (
        tuple[Figure, npt.NDArray[np.object_]]
        | tuple[Figure, npt.NDArray[np.object_], pd.DataFrame]
    ):
        """Scatter minimum-value plane projections for multiple maps with one colorbar."""
        if set_aspect not in ["on", "off", "equal", "scaled"]:
            raise ValueError("set_aspect must be one of ['on','off','equal','scaled']")
        if feature is not None:
            resolved_feature = feature
        elif index is not None:
            if index >= len(self.features) or index < 0:
                raise ValueError(f"Index {index} out of bounds.")
            resolved_feature = self.features[index]
        else:
            raise ValueError("Either feature or index must be provided.")

        projection = self.min_projection(
            feature=feature,
            index=index,
            minimize=minimize,
            plane=plane,
            region=region,
            layer=layer,
            map_indexes=map_indexes,
            coordinate_decimals=coordinate_decimals,
        )
        if projection.empty:
            raise ValueError("No points available after applying the projection filters.")

        if map_indexes is None:
            selected_maps = np.arange(len(self.maps), dtype=np.int64)
        else:
            selected_maps = np.asarray(map_indexes, dtype=np.int64).reshape(-1)
        nmaps = int(selected_maps.size)
        if ncols is None:
            ncols = min(4, int(np.ceil(np.sqrt(nmaps))))
        ncols = max(1, min(int(ncols), nmaps))
        nrows = int(np.ceil(nmaps / ncols))
        if figsize is None:
            extra_width = 0.8 if colorbar else 0.0
            figsize = (panel_size[0] * ncols + extra_width, panel_size[1] * nrows)

        fig, axs = plt.subplots(
            nrows,
            ncols,
            figsize=figsize,
            sharex=sharex,
            sharey=sharey,
            squeeze=False,
        )
        axslist = axs.reshape(-1)

        values = projection[resolved_feature].to_numpy(dtype=np.float64)
        scatter_kwargs = dict(kwargs)
        if "norm" in scatter_kwargs:
            norm = scatter_kwargs.pop("norm")
        else:
            vmin = scatter_kwargs.pop("vmin", float(np.nanmin(values)))
            vmax = scatter_kwargs.pop("vmax", float(np.nanmax(values)))
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        color_map = (
            cmap
            if isinstance(cmap, mpl.colors.Colormap)
            else mpl.colormaps[cmap or scatter_kwargs.pop("cmap", "viridis")]
        )

        scatter_artist = None
        minimized_column = minimize or resolved_feature
        for ax, map_index in zip(axslist, selected_maps, strict=False):
            map_projection = projection.loc[
                projection["map_index"].to_numpy(dtype=np.int64) == int(map_index)
            ]
            ax.set_title(self._map_name(int(map_index)))
            ax.set_xlabel(plane[0])
            ax.set_ylabel(plane[1])
            if not map_projection.empty:
                scatter_artist = ax.scatter(
                    map_projection[plane[0]].to_numpy(dtype=np.float64),
                    map_projection[plane[1]].to_numpy(dtype=np.float64),
                    c=map_projection[resolved_feature].to_numpy(dtype=np.float64),
                    cmap=color_map,
                    norm=norm,
                    **scatter_kwargs,
                )
            ax.axis(set_aspect)

        for ax in axslist[nmaps:]:
            ax.axis("off")

        fig.suptitle(f"Minimum {minimized_column} projection on {plane[0]}-{plane[1]}")
        fig.tight_layout()
        if colorbar and scatter_artist is not None:
            cbar = fig.colorbar(scatter_artist, ax=axslist[:nmaps].tolist())
            cbar.set_label(resolved_feature)
            cbar.solids.set_alpha(1.0)
        if return_projection:
            return fig, axs, projection
        return fig, axs

    def analyze_pca(
        self,
        scale: bool = False,
        *,
        layer: int | Sequence[int] | None = None,
    ) -> PCAAnalysisResult:
        data = self._ensure_data()
        mask = np.ones(len(data), dtype=bool)
        if layer is not None:
            mask &= np.isin(
                self._collect_contactspace_column("layer").astype(np.int64),
                _coerce_layer_values(layer),
            )
        X = data.loc[mask, self.features].values.astype(np.float64)
        if len(X) == 0:
            raise ValueError("No points available after applying the layer filter for PCA.")
        result = fit_pca_analysis(X, feature_columns=list(self.features), scale=scale)
        self.pca_analysis_result = result
        return result

    def reduce(
        self,
        npca: int | None = None,
        scale: bool = False,
        *,
        layer: int | Sequence[int] | None = None,
    ) -> PCAAnalysisResult | PCAResult:
        if npca is None:
            return self.analyze_pca(
                scale=scale,
                layer=layer,
            )
        data = self._ensure_data()
        analysis = self.analyze_pca(
            scale=scale,
            layer=layer,
        )
        result = project_pca(analysis, data[self.features].values.astype(np.float64), npca=npca)
        self.npca = npca
        self.pca_result = result
        data.loc[:, result.transformed_columns] = result.transformed_values
        self._propagate_columns(result.transformed_columns)
        return result

    def cluster(
        self,
        nclusters: int | None = None,
        features: list[str] | None = None,
        method: str = "spectral",
        maxclusters: int = 20,
        ntries: int = 1,
        random_state: int | None = None,
        scale: bool = False,
        graph: GraphResult | None = None,
        layer: int | Sequence[int] | None = None,
        propagate: bool = False,
        propagation_mode: ArchetypePropagationMode = "diffusion",
        propagation_alpha: float = 0.9,
        propagation_max_iter: int = 500,
        propagation_tol: float = 1.0e-8,
        propagation_confidence_threshold: float = 0.0,
        propagation_margin_threshold: float = 0.0,
        propagation_realspace_scale: float = 1.0,
        propagation_feature_scale: float = 1.0,
        propagation_use_node_weights: bool = False,
    ) -> ClusterResult | ClusterScreeningResult:
        data = self._ensure_data()
        self.cluster_features = list(features) if features is not None else list(self.features)
        self.cluster_method = normalize_cluster_method(method)
        selected_mask = np.ones(len(data), dtype=bool)
        if layer is not None:
            selected_mask &= np.isin(
                self._collect_contactspace_column("layer").astype(np.int64),
                _coerce_layer_values(layer),
            )
        X = data.loc[selected_mask, self.cluster_features].values.astype(np.float64)
        if len(X) == 0:
            raise ValueError("No points available after applying the layer filter for clustering.")
        selected_graph = (
            self._subset_graph_result(graph, selected_mask)
            if graph is not None and not np.all(selected_mask)
            else graph
        )
        full_graph = graph if graph is not None else self.graph_result
        if graph is not None and graph.matrix.shape[0] != len(data):
            raise ValueError(
                f"graph has {graph.matrix.shape[0]} nodes, expected {len(data)} to match multimaps.data."
            )
        if selected_graph is not None and selected_graph.matrix.shape[0] != len(X):
            raise ValueError(
                f"graph has {selected_graph.matrix.shape[0]} nodes, expected {len(X)} to match the selected clustering points."
            )

        if nclusters is not None:
            chosen_random_state = self._select_random_state(nclusters, random_state)
            screening_result = (
                ClusterScreeningResult(
                    method=self.cluster_method,
                    feature_columns=list(self.cluster_features),
                    scale=scale,
                    table=self.cluster_screening.copy(),
                    best_by_db=self.best_clusters.copy(),
                    best_by_silhouette=self.cluster_screening.loc[
                        self.cluster_screening.groupby("nclusters")["silhouette_score"].idxmax()
                    ].copy(),
                )
                if self.cluster_screening is not None
                and self.best_clusters is not None
                and self.cluster_screening_method == self.cluster_method
                else None
            )
            result = fit_clusters(
                X,
                feature_columns=list(self.cluster_features),
                nclusters=nclusters,
                method=self.cluster_method,
                random_state=chosen_random_state,
                scale=scale,
                screening=screening_result,
                graph=selected_graph,
            )
            if propagate:
                if full_graph is None:
                    raise RuntimeError(
                        "No graph available for cluster propagation. "
                        "Call multimaps.build_graph(...) first or pass graph explicitly."
                    )
                if full_graph.matrix.shape[0] != len(data):
                    raise ValueError(
                        f"graph has {full_graph.matrix.shape[0]} nodes, expected {len(data)} to match multimaps.data."
                    )
                seed_labels = np.full(len(data), -1, dtype=np.int64)
                seed_labels[selected_mask] = result.labels
                (
                    full_labels,
                    cluster_confidence,
                    cluster_margin,
                    cluster_ambiguous,
                    cluster_scores,
                ) = propagate_cluster_labels(
                    full_graph,
                    seed_mask=selected_mask,
                    seed_labels=seed_labels,
                    propagation_mode=propagation_mode,
                    alpha=propagation_alpha,
                    max_iter=propagation_max_iter,
                    tol=propagation_tol,
                    confidence_threshold=propagation_confidence_threshold,
                    margin_threshold=propagation_margin_threshold,
                    propagation_realspace_scale=propagation_realspace_scale,
                    propagation_feature_scale=propagation_feature_scale,
                    propagation_use_node_weights=propagation_use_node_weights,
                )
                unassigned_mask = full_labels < 0
                if np.any(unassigned_mask):
                    seed_indexes = np.flatnonzero(selected_mask)
                    for map_index in np.unique(data.loc[unassigned_mask, "map_index"]):
                        map_unassigned = np.flatnonzero(
                            unassigned_mask
                            & (data["map_index"].to_numpy(dtype=np.int64) == int(map_index))
                        )
                        map_seed_indexes = seed_indexes[
                            data.loc[seed_indexes, "map_index"].to_numpy(dtype=np.int64)
                            == int(map_index)
                        ]
                        if map_seed_indexes.size == 0:
                            continue
                        nearest_seed_positions = _nearest_reference_indexes(
                            data.loc[map_unassigned, self.cluster_features].to_numpy(
                                dtype=np.float64
                            ),
                            data.loc[map_seed_indexes, self.cluster_features].to_numpy(
                                dtype=np.float64
                            ),
                        )
                        full_labels[map_unassigned] = seed_labels[
                            map_seed_indexes[nearest_seed_positions]
                        ].astype(np.int64, copy=False)

                    remaining_mask = full_labels < 0
                    if np.any(remaining_mask):
                        nearest_labels = np.argmin(
                            distance.cdist(
                                data.loc[remaining_mask, self.cluster_features].to_numpy(
                                    dtype=np.float64
                                ),
                                result.centers,
                            ),
                            axis=1,
                        )
                        full_labels[remaining_mask] = nearest_labels.astype(
                            np.int64,
                            copy=False,
                        )
            else:
                full_labels = np.full(len(data), -1, dtype=np.int64)
                full_labels[selected_mask] = result.labels
                cluster_confidence = np.zeros(len(data), dtype=np.float64)
                cluster_margin = np.zeros(len(data), dtype=np.float64)
                cluster_ambiguous = np.ones(len(data), dtype=bool)
                cluster_confidence[selected_mask] = 1.0
                cluster_margin[selected_mask] = 1.0
                cluster_ambiguous[selected_mask] = False
                cluster_scores = None
            data.loc[:, "Cluster"] = full_labels
            data.loc[:, "cluster_confidence"] = cluster_confidence
            data.loc[:, "cluster_margin"] = cluster_margin
            data.loc[:, "cluster_is_ambiguous"] = cluster_ambiguous
            if cluster_scores is not None:
                for cluster_id in range(cluster_scores.shape[1]):
                    data.loc[:, f"cluster_score_{cluster_id}"] = cluster_scores[:, cluster_id]
            self.nclusters = nclusters
            self.cluster_result = result
            self.cluster_result.labels = full_labels
            self.cluster_centers = result.centers
            self.cluster_graph = self._aggregate_graph(full_labels.astype(np.int64))
            self.cluster_edges = self.cluster_graph.copy()
            for i in range(nclusters):
                self.cluster_edges[i, i] = 0
            result.graph = self.cluster_graph
            result.edges = self.cluster_edges
            result.metadata = {
                **(result.metadata or {}),
                "layer": list(_coerce_layer_values(layer)) if layer is not None else None,
                "propagate": propagate,
                "propagation_mode": propagation_mode if propagate else None,
                "n_selected_points": int(np.count_nonzero(selected_mask)),
            }
            propagated_columns = [
                "Cluster",
                "cluster_confidence",
                "cluster_margin",
                "cluster_is_ambiguous",
            ]
            if cluster_scores is not None:
                propagated_columns.extend(
                    [f"cluster_score_{cluster_id}" for cluster_id in range(cluster_scores.shape[1])]
                )
            self._propagate_columns(propagated_columns)
            return result

        screening = screen_clusters(
            X,
            feature_columns=list(self.cluster_features),
            method=self.cluster_method,
            maxclusters=maxclusters,
            ntries=ntries,
            scale=scale,
            graph=selected_graph,
        )
        self.cluster_screening = screening.table.copy()
        self.cluster_screening_method = screening.method
        self.best_clusters = screening.best_by_db.copy()
        return screening

    def sites(
        self,
        region: int = 0,
        *,
        scope: str = "global",
        per_layer: bool = False,
        method: str = "cluster_centroid",
        nsites: int | None = None,
        kind: str = "centroid",
        iteration: int | None = 0,
        label_status: str = "unlabeled",
        replace_kind: bool = True,
        feature_columns: list[str] | None = None,
        layer: int | Sequence[int] | None = None,
        scale_features: bool = True,
        kernel: str = "rbf",
        gamma: float | str | None = "median",
    ) -> pd.DataFrame:
        """Register cluster-centroid sites globally or independently within each map."""
        scope = self._validate_scope(scope)
        method = normalize_site_selection_method(method)
        if method != "cluster_centroid":
            if nsites is None:
                raise ValueError("nsites is required for non-cluster site selection methods.")
            return self.select_special_points(
                npoints=nsites,
                scope=scope,
                kind=kind,
                iteration=iteration,
                label_status=label_status,
                replace_kind=replace_kind,
                feature_columns=feature_columns,
                region=region,
                layer=layer,
                scale_features=scale_features,
                method=method,
                kernel=kernel,
                gamma=gamma,
            )

        data = self._ensure_data()
        if (
            self.cluster_centers is None
            or self.cluster_edges is None
            or self.cluster_result is None
            or self.cluster_result.sizes is None
        ):
            raise RuntimeError("No clusters have been generated.")
        if not self.cluster_features:
            raise RuntimeError("No cluster feature columns are available.")
        if "Cluster" not in data.columns:
            raise RuntimeError(
                "No cluster labels are available. Call multimaps.cluster(...) first."
            )

        if "region" not in data.columns:
            data.loc[:, "region"] = self._collect_contactspace_column("region")
        if per_layer and "layer" not in data.columns:
            data.loc[:, "layer"] = self._collect_contactspace_column("layer").astype(np.int64)

        if scope == "global":
            selected = self._select_cluster_centroids(data, region=region, per_layer=per_layer)
        else:
            per_map_frames: list[pd.DataFrame] = []
            for data_slice in self._slices:
                selected_map = self._select_cluster_centroids(
                    data.iloc[data_slice],
                    region=region,
                    per_layer=per_layer,
                )
                if not selected_map.empty:
                    per_map_frames.append(selected_map)
            selected = (
                pd.concat(per_map_frames, ignore_index=False)
                if per_map_frames
                else data.iloc[0:0].copy()
            )

        self._register_global_special_points(
            selected.index.to_numpy(dtype=np.int64),
            kind=kind,
            iteration=iteration,
            label_status=label_status,
            replace_kind=replace_kind,
            cluster=selected["Cluster"].to_numpy(dtype=np.int64),
            layer=(
                selected["layer"].to_numpy(dtype=np.int64) if "layer" in selected.columns else None
            ),
        )
        special = self.get_special_points(kind=kind, label_status=None)
        if special.empty:
            return special
        return special.loc[
            special["global_point_index"].isin(selected.index.to_numpy(dtype=np.int64))
        ].reset_index(drop=True)

    def build_graph(
        self,
        *,
        mode: GraphMode = "hybrid",
        feature_columns: list[str] | None = None,
        node_weight_column: str = "probability",
        feature_k: int = 8,
        feature_connectivity: FeatureConnectivityMode = "knn",
        sigma_feature: float | None = None,
        realspace_weight: float = 1.0,
        feature_weight: float = 1.0,
        connect_systems: bool = False,
        normalize_node_weights: bool = True,
        use_node_weights_in_edges: bool = True,
        direction_columns: tuple[str, str, str] | None = None,
        directional_weight: float = 0.0,
        directional_power: float = 1.0,
    ) -> GraphResult:
        data = self._ensure_data()
        selected_features = (
            list(feature_columns) if feature_columns is not None else list(self.features)
        )
        node_table = data.loc[
            :, ["system", "map_index", "point_index", "x", "y", "z", *selected_features]
        ].copy()
        node_table.insert(0, "global_point_index", data.index.to_numpy(dtype=np.int64))
        required_columns = [node_weight_column]
        if directional_weight > 0.0:
            required_columns.extend(
                list(
                    direction_columns
                    or (
                        "boundary_gradient_x",
                        "boundary_gradient_y",
                        "boundary_gradient_z",
                    )
                )
            )
        for column in required_columns:
            if column in node_table.columns:
                continue
            if column in data.columns:
                node_table.loc[:, column] = data[column].to_numpy()
            else:
                node_table.loc[:, column] = self._collect_contactspace_column(column)

        neighbors = self._combine_contactspace_neighbors()
        result = build_point_graph(
            node_table,
            mode=mode,
            feature_columns=selected_features,
            neighbors=neighbors,
            node_weight_column=node_weight_column,
            feature_k=feature_k,
            feature_connectivity=feature_connectivity,
            sigma_feature=sigma_feature,
            realspace_weight=realspace_weight,
            feature_weight=feature_weight,
            connect_systems=connect_systems,
            system_ids=node_table["map_index"].to_numpy(dtype=np.int64),
            normalize_node_weights=normalize_node_weights,
            use_node_weights_in_edges=use_node_weights_in_edges,
            direction_columns=direction_columns,
            directional_weight=directional_weight,
            directional_power=directional_power,
        )
        self.graph_result = result
        return result

    def select_archetypes(
        self,
        n_archetypes: int,
        *,
        scope: str = "global",
        feature_columns: list[str] | None = None,
        graph: GraphResult | None = None,
        probability_column: str = "probability",
        region: int | None = None,
        layer: int | Sequence[int] | None = None,
        min_probability: float | None = None,
        min_probability_quantile: float | None = 0.75,
        scale_features: bool = True,
        selection_mode: ArchetypeSelectionMode = "feature_extreme",
        probability_weight: float = 1.0,
        extremeness_weight: float = 1.0,
        diversity_weight: float = 1.0,
        endpointness_weight: float = 1.0,
        geodesic_weight: float = 1.0,
        branching_weight: float = 0.5,
        register: bool = True,
        kind: str = "archetype",
        iteration: int | None = 0,
        label_status: str = "unlabeled",
        replace_kind: bool = True,
    ) -> ArchetypeSelectionResult:
        scope = self._validate_scope(scope)
        if scope == "per_map":
            if graph is not None:
                raise ValueError(
                    "scope='per_map' does not accept a global graph. "
                    "Build graphs on the child maps or omit graph."
                )
            return self._select_archetypes_per_map(
                n_archetypes=n_archetypes,
                feature_columns=feature_columns,
                probability_column=probability_column,
                region=region,
                layer=layer,
                min_probability=min_probability,
                min_probability_quantile=min_probability_quantile,
                scale_features=scale_features,
                selection_mode=selection_mode,
                probability_weight=probability_weight,
                extremeness_weight=extremeness_weight,
                diversity_weight=diversity_weight,
                endpointness_weight=endpointness_weight,
                geodesic_weight=geodesic_weight,
                branching_weight=branching_weight,
                register=register,
                kind=kind,
                iteration=iteration,
                label_status=label_status,
                replace_kind=replace_kind,
            )

        data = self._ensure_data()
        resolved_feature_columns = (
            list(feature_columns) if feature_columns is not None else list(self.features)
        )
        point_table = data.copy()
        point_table.loc[:, "global_point_index"] = data.index.to_numpy(dtype=np.int64)
        if probability_column not in point_table.columns:
            point_table.loc[:, probability_column] = self._collect_contactspace_column(
                probability_column
            )

        candidate_mask: np.ndarray[Any, np.dtype[np.bool_]] | None = None
        if region is not None or layer is not None:
            candidate_mask = np.ones(len(point_table), dtype=bool)
            if region is not None:
                candidate_mask &= (
                    self._collect_contactspace_column("region").astype(np.int64) == region
                )
            if layer is not None:
                candidate_mask &= np.isin(
                    self._collect_contactspace_column("layer").astype(np.int64),
                    _coerce_layer_values(layer),
                )
            if layer is not None and min_probability is None and min_probability_quantile == 0.75:
                min_probability_quantile = None

        selected_graph = graph if graph is not None else self.graph_result
        if selection_mode == "graph_endpoint" and selected_graph is None:
            raise RuntimeError(
                "selection_mode='graph_endpoint' requires a graph. "
                "Call multimaps.build_graph(...) first or pass graph explicitly."
            )

        result = select_feature_archetypes(
            point_table,
            n_archetypes=n_archetypes,
            feature_columns=resolved_feature_columns,
            probability_column=probability_column,
            point_index_column="global_point_index",
            candidate_mask=candidate_mask,
            min_probability=min_probability,
            min_probability_quantile=min_probability_quantile,
            scale_features=scale_features,
            selection_mode=selection_mode,
            graph=selected_graph,
            probability_weight=probability_weight,
            extremeness_weight=extremeness_weight,
            diversity_weight=diversity_weight,
            endpointness_weight=endpointness_weight,
            geodesic_weight=geodesic_weight,
            branching_weight=branching_weight,
        )
        self.archetype_selection_result = result

        if register:
            if replace_kind:
                for maps in self.maps:
                    maps.special_points.remove(kind=kind)
            self._register_archetypes_on_children(
                result.archetype_table,
                kind=kind,
                iteration=iteration,
                label_status=label_status,
            )

        return result

    def select_special_points(
        self,
        npoints: int,
        *,
        scope: str = "global",
        kind: str,
        iteration: int | None = None,
        label_status: str = "unlabeled",
        replace_kind: bool = False,
        store_selection_metadata: bool = True,
        feature_columns: list[str] | None = None,
        energy_column: str | None = None,
        uncertainty_column: str | None = None,
        special_point_indexes: npt.ArrayLike | None = None,
        centroid_indexes: npt.ArrayLike | None = None,
        region: int | None = None,
        layer: int | Sequence[int] | None = None,
        real_space_weight: float = 0.0,
        feature_space_weight: float = 1.0,
        energy_weight: float = 0.0,
        uncertainty_weight: float = 0.0,
        scale_features: bool = True,
        method: str = "greedy",
        kernel: str = "rbf",
        gamma: float | str | None = "median",
        energy_selection_mode: str = "global_minimum",
        gradient_columns: list[str] | None = None,
        gradient_norm_column: str | None = None,
        curvature_columns: list[str] | None = None,
        stationary_orders: int | Sequence[int] = 0,
        gradient_tolerance: float = 1.0e-6,
        curvature_tolerance: float = 1.0e-8,
        **metadata: Any,
    ) -> pd.DataFrame:
        """Select and register special points either globally or independently per map."""
        scope = self._validate_scope(scope)
        if scope == "per_map":
            frames: list[pd.DataFrame] = []
            self._ensure_data()
            for map_index, (name, maps) in enumerate(zip(self.names, self.maps, strict=False)):
                selected = maps.select_special_points(
                    npoints=npoints,
                    kind=kind,
                    iteration=iteration,
                    label_status=label_status,
                    replace_kind=replace_kind,
                    store_selection_metadata=store_selection_metadata,
                    feature_columns=feature_columns,
                    energy_column=energy_column,
                    uncertainty_column=uncertainty_column,
                    special_point_indexes=special_point_indexes,
                    centroid_indexes=centroid_indexes,
                    region=region,
                    layer=layer,
                    real_space_weight=real_space_weight,
                    feature_space_weight=feature_space_weight,
                    energy_weight=energy_weight,
                    uncertainty_weight=uncertainty_weight,
                    scale_features=scale_features,
                    method=method,
                    kernel=kernel,
                    gamma=gamma,
                    energy_selection_mode=energy_selection_mode,
                    gradient_columns=gradient_columns,
                    gradient_norm_column=gradient_norm_column,
                    curvature_columns=curvature_columns,
                    stationary_orders=stationary_orders,
                    gradient_tolerance=gradient_tolerance,
                    curvature_tolerance=curvature_tolerance,
                    **metadata,
                )
                frames.append(
                    self._with_map_identity(
                        selected,
                        map_index=map_index,
                        system_name=name,
                        global_index_column=True,
                    )
                )
            return self._combine_frames(frames)

        selected = self._select_points_global(
            npoints=npoints,
            feature_columns=feature_columns,
            energy_column=energy_column,
            uncertainty_column=uncertainty_column,
            special_point_indexes=special_point_indexes,
            centroid_indexes=centroid_indexes,
            region=region,
            layer=layer,
            real_space_weight=real_space_weight,
            feature_space_weight=feature_space_weight,
            energy_weight=energy_weight,
            uncertainty_weight=uncertainty_weight,
            scale_features=scale_features,
            method=method,
            kernel=kernel,
            gamma=gamma,
            energy_selection_mode=energy_selection_mode,
            gradient_columns=gradient_columns,
            gradient_norm_column=gradient_norm_column,
            curvature_columns=curvature_columns,
            stationary_orders=stationary_orders,
            gradient_tolerance=gradient_tolerance,
            curvature_tolerance=curvature_tolerance,
        )

        selected_global_indexes = selected.index.to_numpy(dtype=np.int64)
        add_metadata = dict(metadata)
        if store_selection_metadata:
            for column in selection_metadata_columns(method):
                if column in selected.columns and column not in add_metadata:
                    add_metadata[column] = selected[column].to_numpy()

        if replace_kind:
            for maps in self.maps:
                maps.special_points.remove(kind=kind)

        for map_index, group in selected.groupby("map_index", sort=False):
            group_positions = selected.index.get_indexer(group.index)
            local_metadata: dict[str, Any] = {}
            for column, value in add_metadata.items():
                if np.isscalar(value) or value is None:
                    local_metadata[column] = value
                    continue
                values = np.asarray(value, dtype=object).reshape(-1)
                if values.size == len(selected):
                    local_metadata[column] = values[group_positions]
                else:
                    local_metadata[column] = value

            self.maps[int(map_index)].add_special_points(
                group["point_index"].to_numpy(dtype=np.int64),
                kind=kind,
                iteration=iteration,
                label_status=label_status,
                replace_kind=False,
                **local_metadata,
            )

        special = self.get_special_points(kind=kind, label_status=None)
        if special.empty:
            return special
        special = special.loc[special["global_point_index"].isin(selected_global_indexes)].copy()
        if "selection_rank" in special.columns:
            return special.sort_values("selection_rank").reset_index(drop=True)
        return special.reset_index(drop=True)

    def get_special_points(
        self,
        *,
        kind: str | None = None,
        label_status: str | Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Return special points from all child maps with map and global indexes attached."""
        self._ensure_data()
        if not all(hasattr(maps, "get_special_points") for maps in self.maps):
            return self._global_special_points_frame(kind=kind, label_status=label_status)

        frames = []
        for map_index, (name, maps) in enumerate(zip(self.names, self.maps, strict=False)):
            frame = maps.get_special_points(kind=kind, label_status=label_status)
            frames.append(
                self._with_map_identity(
                    frame,
                    map_index=map_index,
                    system_name=name,
                    global_index_column=True,
                )
            )
        return self._combine_frames(frames)

    def propagate_archetypes(
        self,
        *,
        graph: GraphResult | None = None,
        selected_indexes: npt.ArrayLike | None = None,
        propagation_mode: ArchetypePropagationMode = "diffusion",
        alpha: float = 0.9,
        max_iter: int = 500,
        tol: float = 1.0e-8,
        confidence_threshold: float = 0.5,
        margin_threshold: float = 0.0,
        propagation_realspace_scale: float = 1.0,
        propagation_feature_scale: float = 1.0,
        propagation_use_node_weights: bool = False,
        kind: str = "archetype",
        update_data: bool = True,
    ) -> ArchetypePropagationResult:
        data = self._ensure_data()
        if selected_indexes is None:
            if self.archetype_selection_result is None:
                raise RuntimeError(
                    "No archetype seeds available. Call multimaps.select_archetypes(...) first "
                    "or pass selected_indexes explicitly."
                )
            selected = self.archetype_selection_result.selected_indexes
        else:
            selected = np.asarray(selected_indexes, dtype=np.int64).reshape(-1)
        if selected.size == 0:
            raise RuntimeError("No archetype seeds available for propagation.")

        selected_graph = graph if graph is not None else self.graph_result
        if selected_graph is None:
            raise RuntimeError(
                "No graph available for propagation. Call multimaps.build_graph(...) first "
                "or pass graph explicitly."
            )
        if "global_point_index" not in selected_graph.node_table.columns:
            selected_graph.node_table.loc[:, "global_point_index"] = np.arange(
                len(selected_graph.node_table), dtype=np.int64
            )

        result = propagate_archetypes_on_graph(
            selected_graph,
            selected_indexes=selected,
            point_index_column="global_point_index",
            propagation_mode=propagation_mode,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            confidence_threshold=confidence_threshold,
            margin_threshold=margin_threshold,
            propagation_realspace_scale=propagation_realspace_scale,
            propagation_feature_scale=propagation_feature_scale,
            propagation_use_node_weights=propagation_use_node_weights,
        )
        self.archetype_propagation_result = result

        if update_data:
            assignment = result.assignment_table.set_index("global_point_index")
            columns = [
                "assigned_archetype_rank",
                "assigned_archetype_index",
                "archetype_confidence",
                "archetype_margin",
                "is_ambiguous",
            ]
            data.loc[assignment.index, columns] = assignment.loc[:, columns]
            self._propagate_columns(columns)

        valid_assignments = result.assigned_archetype_indexes[
            result.assigned_archetype_indexes >= 0
        ]
        assignment_table = result.assignment_table
        seed_rows = data.loc[selected, ["map_index", "point_index"]].copy()
        for map_index, group in seed_rows.groupby("map_index"):
            local_point_indexes = group["point_index"].to_numpy(dtype=np.int64)
            global_ids = group.index.to_numpy(dtype=np.int64)
            assigned_counts = np.array(
                [int(np.count_nonzero(valid_assignments == global_id)) for global_id in global_ids],
                dtype=np.int64,
            )
            mean_confidences = []
            for global_id in global_ids:
                point_mask = assignment_table["assigned_archetype_index"].to_numpy(
                    dtype=np.int64
                ) == int(global_id)
                mean_confidences.append(
                    float(assignment_table.loc[point_mask, "archetype_confidence"].mean())
                    if np.any(point_mask)
                    else 0.0
                )
            self.maps[int(map_index)].update_special_points(
                kind=kind,
                point_indexes=local_point_indexes,
                assigned_point_count=assigned_counts,
                mean_assignment_confidence=np.asarray(mean_confidences, dtype=np.float64),
            )

        return result

    def _compute_feature_targets_serial(
        self,
        targets: Sequence[int],
        *,
        point_workers: int | None,
        checkpoint: str | Path | None,
        checkpoint_every: int,
        release_contactspace_cache: bool,
        checkpoint_mode: str,
        progress: bool,
        threadpool_threads: int | None,
    ) -> None:
        if checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive")

        for completed, index in enumerate(targets, start=1):
            maps = self.maps[index]
            if maps.contactspace is None:
                raise RuntimeError("Each Maps instance must define a contact space")
            self._feature_progress(
                progress,
                (
                    f"starting map {index} ({self._map_name(index)}) "
                    f"{completed}/{len(targets)} with point_workers={point_workers}"
                ),
            )
            start = time.perf_counter()
            with _threadpool_context(threadpool_threads):
                _call_maps_atcontactspace(
                    maps,
                    workers=point_workers,
                    release_contactspace_cache=release_contactspace_cache,
                )
            elapsed = time.perf_counter() - start
            npoints = 0 if maps.data is None else len(maps.data)
            self._feature_progress(
                progress,
                (
                    f"finished map {index} ({self._map_name(index)}) "
                    f"{completed}/{len(targets)} in {elapsed:.1f}s ({npoints} points)"
                ),
            )
            self.data = None
            if checkpoint is not None and completed % checkpoint_every == 0:
                checkpoint_start = time.perf_counter()
                if checkpoint_mode == "maps":
                    path = self._save_map_feature_checkpoint(checkpoint, index)
                else:
                    path = self._save_feature_checkpoint(checkpoint)
                self._feature_progress(
                    progress,
                    (
                        f"checkpointed map {index} in "
                        f"{time.perf_counter() - checkpoint_start:.1f}s -> {path}"
                    ),
                )

    def _feature_target_indexes(
        self,
        *,
        recompute: bool,
        map_indexes: npt.ArrayLike | None,
    ) -> list[int]:
        if map_indexes is None:
            selected = list(range(len(self.maps)))
        else:
            selected = [int(index) for index in np.asarray(map_indexes, dtype=np.int64).reshape(-1)]

        targets = []
        for index in selected:
            if index < 0 or index >= len(self.maps):
                raise IndexError(f"map index {index} out of range for {len(self.maps)} maps")
            if recompute or self.maps[index].data is None:
                targets.append(index)
        return targets

    def _all_maps_have_data(self) -> bool:
        return all(maps.data is not None for maps in self.maps)

    def _partial_feature_dataset(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        reference_features: list[str] | None = None
        for index, (name, maps) in enumerate(zip(self.names, self.maps, strict=False)):
            if maps.data is None:
                continue
            frame = maps.data.copy()
            self._attach_map_metadata(index, maps, frame)
            current_features = self._get_map_features(maps)
            if reference_features is None:
                reference_features = current_features
            elif set(current_features) != set(reference_features):
                raise ValueError("All completed maps must expose the same feature columns")
            metadata_columns = [
                column
                for column in self._METADATA_COLUMNS
                if column in frame.columns and column not in current_features
            ]
            ordered = frame.loc[:, ["x", "y", "z", *metadata_columns, *current_features]].copy()
            ordered.insert(0, "point_index", np.arange(len(ordered), dtype=np.int64))
            ordered.insert(0, "map_index", index)
            ordered.insert(0, "system", name)
            frames.append(ordered)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _save_feature_checkpoint(self, checkpoint: str | Path) -> Path:
        current_data = self.data
        self.data = None if not self._all_maps_have_data() else self.data
        try:
            return self.save(checkpoint)
        finally:
            self.data = current_data

    def _save_map_feature_checkpoint(self, checkpoint: str | Path, index: int) -> Path:
        directory = self._map_checkpoint_directory(checkpoint)
        directory.mkdir(parents=True, exist_ok=True)
        return self.maps[index].save(self._map_checkpoint_path(checkpoint, index))

    def _load_map_feature_checkpoints(
        self,
        checkpoint: str | Path,
        *,
        map_indexes: npt.ArrayLike | None,
    ) -> None:
        indexes = self._feature_target_indexes(
            recompute=True,
            map_indexes=map_indexes,
        )
        for index in indexes:
            path = self._map_checkpoint_path(checkpoint, index)
            if not path.exists():
                continue
            loaded = self.maps[index].__class__.load(path)
            if loaded.data is not None:
                self.maps[index] = loaded

    @staticmethod
    def _map_checkpoint_directory(checkpoint: str | Path) -> Path:
        path = Path(checkpoint).expanduser().resolve()
        return path.with_name(f"{path.name}.maps")

    @classmethod
    def _map_checkpoint_path(cls, checkpoint: str | Path, index: int) -> Path:
        return cls._map_checkpoint_directory(checkpoint) / f"map_{index:05d}.pkl"

    @staticmethod
    def _normalize_feature_parallelism(parallel: str | bool) -> str:
        if isinstance(parallel, bool):
            return "maps" if parallel else "none"
        normalized = str(parallel).lower()
        aliases = {
            "false": "none",
            "no": "none",
            "off": "none",
            "serial": "none",
            "map": "maps",
            "per_map": "maps",
            "structures": "maps",
            "structure": "maps",
            "point": "points",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"points", "maps", "none"}:
            raise ValueError(
                "parallel must be one of 'points', 'maps', or 'none', " f"got {parallel!r}."
            )
        return normalized

    @staticmethod
    def _normalize_checkpoint_mode(checkpoint_mode: str) -> str:
        normalized = str(checkpoint_mode).lower()
        aliases = {
            "object": "full",
            "global": "full",
            "whole": "full",
            "map": "maps",
            "per_map": "maps",
            "sidecar": "maps",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"full", "maps"}:
            raise ValueError(
                "checkpoint_mode must be either 'full' or 'maps', " f"got {checkpoint_mode!r}."
            )
        return normalized

    @staticmethod
    def _resolve_feature_workers(workers: int | None, ntargets: int) -> int:
        if ntargets <= 1:
            return 1
        if workers is None:
            return max(1, min(4, os.cpu_count() or 1, ntargets))
        if workers <= 0:
            return max(1, min(4, os.cpu_count() or 1, ntargets))
        return max(1, min(int(workers), ntargets))

    def _select_archetypes_per_map(
        self,
        *,
        n_archetypes: int,
        feature_columns: list[str] | None,
        probability_column: str,
        region: int | None,
        layer: int | Sequence[int] | None,
        min_probability: float | None,
        min_probability_quantile: float | None,
        scale_features: bool,
        selection_mode: ArchetypeSelectionMode,
        probability_weight: float,
        extremeness_weight: float,
        diversity_weight: float,
        endpointness_weight: float,
        geodesic_weight: float,
        branching_weight: float,
        register: bool,
        kind: str,
        iteration: int | None,
        label_status: str,
        replace_kind: bool,
    ) -> ArchetypeSelectionResult:
        self._ensure_data()
        candidate_frames: list[pd.DataFrame] = []
        archetype_frames: list[pd.DataFrame] = []
        candidate_indexes: list[np.ndarray[Any, np.dtype[np.int64]]] = []
        selected_indexes: list[np.ndarray[Any, np.dtype[np.int64]]] = []
        feature_result: list[str] | None = None
        metadata: dict[str, Any] | None = None

        for map_index, (name, maps) in enumerate(zip(self.names, self.maps, strict=False)):
            result = maps.select_archetypes(
                n_archetypes=n_archetypes,
                feature_columns=feature_columns,
                probability_column=probability_column,
                region=region,
                layer=layer,
                min_probability=min_probability,
                min_probability_quantile=min_probability_quantile,
                scale_features=scale_features,
                selection_mode=selection_mode,
                probability_weight=probability_weight,
                extremeness_weight=extremeness_weight,
                diversity_weight=diversity_weight,
                endpointness_weight=endpointness_weight,
                geodesic_weight=geodesic_weight,
                branching_weight=branching_weight,
                register=register,
                kind=kind,
                iteration=iteration,
                label_status=label_status,
                replace_kind=replace_kind,
            )

            candidate_frame = self._with_map_identity(
                result.candidate_table,
                map_index=map_index,
                system_name=name,
                global_index_column=True,
            )
            archetype_frame = self._with_map_identity(
                result.archetype_table,
                map_index=map_index,
                system_name=name,
                global_index_column=True,
            )
            candidate_frames.append(candidate_frame)
            archetype_frames.append(archetype_frame)
            candidate_indexes.append(candidate_frame["global_point_index"].to_numpy(dtype=np.int64))
            selected_indexes.append(archetype_frame["global_point_index"].to_numpy(dtype=np.int64))
            if feature_result is None:
                feature_result = list(result.feature_columns)
            if metadata is None:
                metadata = dict(result.metadata or {})

        combined_candidates = self._combine_frames(candidate_frames)
        combined_archetypes = self._combine_frames(archetype_frames)
        if "selection_rank" in combined_archetypes.columns:
            combined_archetypes = combined_archetypes.sort_values(
                ["map_index", "selection_rank"]
            ).reset_index(drop=True)

        combined_result = ArchetypeSelectionResult(
            feature_columns=feature_result or list(feature_columns or self.features),
            probability_column=probability_column,
            candidate_indexes=(
                np.concatenate(candidate_indexes).astype(np.int64, copy=False)
                if candidate_indexes
                else np.array([], dtype=np.int64)
            ),
            selected_indexes=(
                np.concatenate(selected_indexes).astype(np.int64, copy=False)
                if selected_indexes
                else np.array([], dtype=np.int64)
            ),
            candidate_table=combined_candidates,
            archetype_table=combined_archetypes,
            metadata={
                **(metadata or {}),
                "scope": "per_map",
                "n_archetypes_per_map": n_archetypes,
                "point_index_column": "global_point_index",
            },
        )
        self.archetype_selection_result = combined_result
        return combined_result

    def _select_points_global(
        self,
        *,
        npoints: int,
        feature_columns: list[str] | None,
        energy_column: str | None,
        uncertainty_column: str | None,
        special_point_indexes: npt.ArrayLike | None,
        centroid_indexes: npt.ArrayLike | None,
        region: int | None,
        layer: int | Sequence[int] | None,
        real_space_weight: float,
        feature_space_weight: float,
        energy_weight: float,
        uncertainty_weight: float,
        scale_features: bool,
        method: str,
        kernel: str,
        gamma: float | str | None,
        energy_selection_mode: str,
        gradient_columns: list[str] | None,
        gradient_norm_column: str | None,
        curvature_columns: list[str] | None,
        stationary_orders: int | Sequence[int],
        gradient_tolerance: float,
        curvature_tolerance: float,
    ) -> pd.DataFrame:
        if npoints <= 0:
            raise ValueError("npoints must be positive")

        data = self._ensure_data()
        selection_method = normalize_point_selection_method(method)
        if selection_method == "greedy":
            weights = np.array(
                [real_space_weight, feature_space_weight, energy_weight, uncertainty_weight],
                dtype=np.float64,
            )
            if np.any(weights < 0.0):
                raise ValueError("Selection weights must be non-negative")
            if not np.any(weights > 0.0):
                raise ValueError("At least one selection weight must be positive")
        elif selection_method == "pivoted_cholesky" and str(kernel).lower() != "rbf":
            raise ValueError("Only kernel='rbf' is supported for pivoted selection methods.")

        frame = data.copy()
        if region is not None:
            if "region" not in frame.columns:
                frame.loc[:, "region"] = self._collect_contactspace_column("region")
            frame = frame.loc[frame["region"].to_numpy(dtype=np.int64) == region].copy()
        if layer is not None:
            if "layer" not in frame.columns:
                frame.loc[:, "layer"] = self._collect_contactspace_column("layer").astype(np.int64)
            frame = frame.loc[
                np.isin(frame["layer"].to_numpy(dtype=np.int64), _coerce_layer_values(layer))
            ].copy()

        if frame.empty:
            raise RuntimeError("No candidate points available for selection")

        seed_source = (
            special_point_indexes if special_point_indexes is not None else centroid_indexes
        )
        if seed_source is None:
            seed_indexes = self._global_special_point_indexes()
        else:
            seed_indexes = np.asarray(seed_source, dtype=np.int64).reshape(-1)
        seed_indexes = seed_indexes[np.isin(seed_indexes, frame.index.to_numpy())]

        candidate_frame = frame.drop(index=seed_indexes, errors="ignore").copy()
        if candidate_frame.empty:
            raise RuntimeError("No candidate points remain after excluding seed points")
        if npoints > len(candidate_frame):
            raise ValueError(
                f"Requested {npoints} points, but only {len(candidate_frame)} candidates are available."
            )

        resolved_feature_columns = self._resolve_selection_feature_columns(
            feature_columns,
            energy_column=energy_column,
            uncertainty_column=uncertainty_column,
            feature_space_weight=feature_space_weight if selection_method == "greedy" else 1.0,
        )

        real_candidates = candidate_frame[["x", "y", "z"]].to_numpy(dtype=np.float64)
        real_seeds = frame.loc[seed_indexes, ["x", "y", "z"]].to_numpy(dtype=np.float64)

        if resolved_feature_columns:
            full_features = frame.loc[:, resolved_feature_columns].to_numpy(dtype=np.float64)
            if scale_features:
                full_features = StandardScaler().fit_transform(full_features)
            feature_frame = pd.DataFrame(
                full_features,
                index=frame.index,
                columns=resolved_feature_columns,
            )
            feature_candidates = feature_frame.loc[candidate_frame.index].to_numpy(dtype=np.float64)
            feature_seeds = feature_frame.loc[seed_indexes].to_numpy(dtype=np.float64)
        else:
            feature_candidates = np.zeros((len(candidate_frame), 0), dtype=np.float64)
            feature_seeds = np.zeros((len(seed_indexes), 0), dtype=np.float64)

        if selection_method != "greedy":
            if selection_method == "pivoted_qr":
                selected_local_array, pivot_scores = pivoted_qr_selection(
                    feature_candidates,
                    npoints,
                    seed_features=feature_seeds,
                )
                kernel_gamma: float | None = None
            else:
                selected_local_array, pivot_scores, kernel_gamma = pivoted_cholesky_rbf_selection(
                    feature_candidates,
                    npoints,
                    seed_features=feature_seeds,
                    gamma=gamma,
                )
            global_indexes = candidate_frame.index[selected_local_array].to_numpy(dtype=np.int64)
            selection = candidate_frame.loc[global_indexes].copy()
            selection.loc[:, "global_point_index"] = selection.index.to_numpy(dtype=np.int64)
            selection.loc[:, "selection_rank"] = np.arange(npoints, dtype=np.int64)
            selection.loc[:, "selection_score"] = pivot_scores
            selection.loc[:, "selection_method"] = selection_method
            selection.loc[:, "real_space_score"] = 0.0
            selection.loc[:, "feature_space_score"] = pivot_scores
            selection.loc[:, "energy_score"] = 0.0
            selection.loc[:, "uncertainty_score"] = 0.0
            selection.loc[:, "pivot_score"] = pivot_scores
            if kernel_gamma is not None:
                selection.loc[:, "kernel_gamma"] = kernel_gamma
            return selection.sort_values("selection_rank")

        energy_values = (
            candidate_frame[energy_column].to_numpy(dtype=np.float64)
            if energy_column is not None
            else np.zeros(len(candidate_frame), dtype=np.float64)
        )
        uncertainty_values = (
            candidate_frame[uncertainty_column].to_numpy(dtype=np.float64)
            if uncertainty_column is not None
            else np.zeros(len(candidate_frame), dtype=np.float64)
        )

        remaining = np.arange(len(candidate_frame), dtype=np.int64)
        selected_local: list[int] = []
        selected_real: list[npt.NDArray[np.float64]] = []
        selected_feature: list[npt.NDArray[np.float64]] = []
        records: list[dict[str, Any]] = []

        for rank in range(npoints):
            current_real = real_candidates[remaining]
            current_feature = feature_candidates[remaining]
            current_energy = energy_values[remaining]
            current_uncertainty = uncertainty_values[remaining]

            real_refs = self._stack_references(real_seeds, selected_real, width=3)
            feature_refs = self._stack_references(
                feature_seeds,
                selected_feature,
                width=feature_candidates.shape[1],
            )

            real_component = (
                self._normalize_component(self._min_distance_to_references(current_real, real_refs))
                if real_space_weight > 0.0
                else np.zeros(len(remaining), dtype=np.float64)
            )
            feature_component = (
                self._normalize_component(
                    self._min_distance_to_references(current_feature, feature_refs)
                )
                if feature_space_weight > 0.0
                else np.zeros(len(remaining), dtype=np.float64)
            )
            energy_component = (
                self._energy_component(
                    candidate_frame.iloc[remaining],
                    energy_values=current_energy,
                    energy_column=energy_column,
                    energy_selection_mode=energy_selection_mode,
                    gradient_columns=gradient_columns,
                    gradient_norm_column=gradient_norm_column,
                    curvature_columns=curvature_columns,
                    stationary_orders=stationary_orders,
                    gradient_tolerance=gradient_tolerance,
                    curvature_tolerance=curvature_tolerance,
                )
                if energy_weight > 0.0
                else np.zeros(len(remaining), dtype=np.float64)
            )
            uncertainty_component = (
                self._normalize_component(current_uncertainty)
                if uncertainty_weight > 0.0
                else np.zeros(len(remaining), dtype=np.float64)
            )

            total_score = (
                real_space_weight * real_component
                + feature_space_weight * feature_component
                + energy_weight * energy_component
                + uncertainty_weight * uncertainty_component
            )

            best_position = int(np.argmax(total_score))
            best_local = int(remaining[best_position])
            best_index = candidate_frame.index[best_local]

            selected_local.append(best_local)
            selected_real.append(real_candidates[best_local])
            if feature_candidates.shape[1] > 0:
                selected_feature.append(feature_candidates[best_local])

            records.append(
                {
                    "selection_rank": rank,
                    "selection_score": float(total_score[best_position]),
                    "real_space_score": float(real_component[best_position]),
                    "feature_space_score": float(feature_component[best_position]),
                    "energy_score": float(energy_component[best_position]),
                    "uncertainty_score": float(uncertainty_component[best_position]),
                    "selection_method": selection_method,
                    "global_point_index": int(best_index),
                }
            )

            remaining = np.delete(remaining, best_position)

        selection = candidate_frame.loc[[record["global_point_index"] for record in records]].copy()
        selection.loc[:, "global_point_index"] = selection.index.to_numpy(dtype=np.int64)
        selection.loc[:, "selection_rank"] = [record["selection_rank"] for record in records]
        selection.loc[:, "selection_score"] = [record["selection_score"] for record in records]
        selection.loc[:, "real_space_score"] = [record["real_space_score"] for record in records]
        selection.loc[:, "feature_space_score"] = [
            record["feature_space_score"] for record in records
        ]
        selection.loc[:, "energy_score"] = [record["energy_score"] for record in records]
        selection.loc[:, "uncertainty_score"] = [record["uncertainty_score"] for record in records]
        selection.loc[:, "selection_method"] = [record["selection_method"] for record in records]
        return selection.sort_values("selection_rank")

    def _resolve_selection_feature_columns(
        self,
        feature_columns: list[str] | None,
        *,
        energy_column: str | None,
        uncertainty_column: str | None,
        feature_space_weight: float,
    ) -> list[str]:
        if feature_columns is not None:
            return list(feature_columns)
        if feature_space_weight <= 0.0:
            return []

        if self.cluster_features:
            return list(self.cluster_features)

        excluded = {"Cluster"}
        if energy_column is not None:
            excluded.add(energy_column)
        if uncertainty_column is not None:
            excluded.add(uncertainty_column)
        return [column for column in self.features if column not in excluded]

    def _global_special_point_indexes(self) -> np.ndarray[Any, np.dtype[np.int64]]:
        indexes: list[np.ndarray[Any, np.dtype[np.int64]]] = []
        self._ensure_data()
        for maps, data_slice in zip(self.maps, self._slices, strict=False):
            local_indexes = maps.special_points.reference_indexes()
            if local_indexes.size == 0:
                continue
            indexes.append(local_indexes.astype(np.int64, copy=False) + data_slice.start)
        if not indexes:
            return np.array([], dtype=np.int64)
        return np.unique(np.concatenate(indexes).astype(np.int64, copy=False))

    def _select_cluster_centroids(
        self,
        frame: pd.DataFrame,
        *,
        region: int,
        per_layer: bool = False,
    ) -> pd.DataFrame:
        if (
            self.cluster_centers is None
            or self.cluster_edges is None
            or self.cluster_result is None
        ):
            raise RuntimeError("No clusters have been generated.")
        cluster_sizes = self.cluster_result.sizes
        if cluster_sizes is None:
            raise RuntimeError("No cluster sizes are available.")

        filterdata = frame.loc[frame["region"].to_numpy(dtype=np.int64) == region].copy()
        if filterdata.empty:
            return filterdata.iloc[0:0].copy()
        if per_layer and "layer" not in filterdata.columns:
            raise RuntimeError("per_layer=True requires a 'layer' column in the MultiMaps data.")

        selected_rows: list[pd.Series[Any]] = []
        cluster_values = np.unique(filterdata["Cluster"].to_numpy(dtype=np.int64))
        for cluster_index in cluster_values:
            if cluster_index < 0:
                continue
            cluster_data = filterdata.loc[filterdata["Cluster"] == cluster_index]
            if cluster_data.empty:
                continue
            if per_layer:
                subgroups = [subgroup for _, subgroup in cluster_data.groupby("layer", sort=True)]
            else:
                subgroups = [cluster_data]

            for subgroup in subgroups:
                if subgroup.empty:
                    continue
                cluster_points = subgroup[self.cluster_features].to_numpy(dtype=np.float64)
                dist = distance.cdist(cluster_points, self.cluster_centers)
                connected_indexes = np.where(self.cluster_edges[cluster_index, :] != 0)[0]
                if connected_indexes.size == 0:
                    cluster_centroid = int(np.argmin(dist[:, cluster_index]))
                else:
                    full_dist = np.sum(
                        cluster_sizes[connected_indexes] / dist[:, connected_indexes] ** 2,
                        axis=1,
                    )
                    cluster_centroid = int(np.argmin(full_dist))
                selected_rows.append(subgroup.iloc[cluster_centroid].copy())

        if not selected_rows:
            return filterdata.iloc[0:0].copy()
        return pd.DataFrame(selected_rows)

    def _register_global_special_points(
        self,
        global_point_indexes: npt.ArrayLike,
        *,
        kind: str,
        iteration: int | None,
        label_status: str,
        replace_kind: bool,
        **metadata: Any,
    ) -> None:
        indexes = np.asarray(global_point_indexes, dtype=np.int64).reshape(-1)
        if replace_kind:
            self._special_points = self._special_points.loc[
                self._special_points["kind"] != kind
            ].reset_index(drop=True)
            for maps in self.maps:
                special_points = getattr(maps, "special_points", None)
                if special_points is not None:
                    special_points.remove(kind=kind)
        if indexes.size == 0:
            return

        data = self._ensure_data()
        rows = pd.DataFrame(
            {
                "global_point_index": indexes,
                "kind": kind,
                "iteration": iteration,
                "label_status": label_status,
            }
        )
        for key, values in metadata.items():
            if values is None:
                continue
            array = np.asarray(values, dtype=object).reshape(-1)
            if array.size != indexes.size:
                raise ValueError(
                    f"Metadata column {key!r} has length {array.size}, expected {indexes.size}."
                )
            rows.loc[:, key] = array
        self._special_points = pd.concat([self._special_points, rows], ignore_index=True)
        self._special_points = self._special_points.drop_duplicates(
            subset=["global_point_index", "kind"],
            keep="last",
        )

        selected = data.loc[indexes, ["map_index", "point_index"]].copy()
        for map_index, group in selected.groupby("map_index", sort=False):
            add_special_points = getattr(self.maps[int(map_index)], "add_special_points", None)
            if add_special_points is None:
                continue
            group_metadata: dict[str, Any] = {}
            for key, values in metadata.items():
                if values is None:
                    continue
                array = np.asarray(values, dtype=object).reshape(-1)
                if array.size != indexes.size:
                    raise ValueError(
                        f"Metadata column {key!r} has length {array.size}, expected {indexes.size}."
                    )
                local_values = array[
                    selected["map_index"].to_numpy(dtype=np.int64) == int(map_index)
                ]
                group_metadata[key] = local_values
            add_special_points(
                group["point_index"].to_numpy(dtype=np.int64),
                kind=kind,
                iteration=iteration,
                label_status=label_status,
                replace_kind=False,
                **group_metadata,
            )

    def _global_special_points_frame(
        self,
        *,
        kind: str | None,
        label_status: str | Sequence[str] | None,
    ) -> pd.DataFrame:
        frame = self._special_points
        if kind is not None:
            frame = frame.loc[frame["kind"] == kind]
        if label_status is not None:
            statuses = [label_status] if isinstance(label_status, str) else list(label_status)
            frame = frame.loc[frame["label_status"].isin(statuses)]
        frame = frame.copy()
        if frame.empty:
            return frame

        data = self._ensure_data()
        duplicate_columns = [
            column
            for column in frame.columns
            if column != "global_point_index" and column in data.columns
        ]
        point_data = data.drop(columns=duplicate_columns).copy()
        point_data.index.name = "global_point_index"
        return frame.merge(point_data, on="global_point_index", how="left")

    def _with_map_identity(
        self,
        frame: pd.DataFrame,
        *,
        map_index: int,
        system_name: str,
        global_index_column: bool = False,
    ) -> pd.DataFrame:
        result = frame.copy()
        if "system" in result.columns:
            result.loc[:, "system"] = system_name
        else:
            result.insert(0, "system", system_name)
        if "map_index" in result.columns:
            result.loc[:, "map_index"] = map_index
        else:
            result.insert(1, "map_index", map_index)

        if global_index_column and "point_index" in result.columns:
            local_indexes = result["point_index"].to_numpy(dtype=np.int64)
            global_indexes = local_indexes + self._slices[map_index].start
            if "global_point_index" in result.columns:
                result.loc[:, "global_point_index"] = global_indexes
            else:
                result.insert(2, "global_point_index", global_indexes)
        return result

    @staticmethod
    def _combine_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
        nonempty = [frame for frame in frames if not frame.empty]
        if nonempty:
            return pd.concat(nonempty, ignore_index=True)
        return pd.DataFrame()

    @staticmethod
    def _validate_scope(scope: str) -> str:
        normalized = scope.lower()
        if normalized not in {"global", "per_map"}:
            raise ValueError(f"scope must be 'global' or 'per_map', got {scope!r}.")
        return normalized

    def _energy_component(
        self,
        frame: pd.DataFrame,
        *,
        energy_values: npt.NDArray[np.float64],
        energy_column: str | None,
        energy_selection_mode: str,
        gradient_columns: list[str] | None,
        gradient_norm_column: str | None,
        curvature_columns: list[str] | None,
        stationary_orders: int | Sequence[int],
        gradient_tolerance: float,
        curvature_tolerance: float,
    ) -> npt.NDArray[np.float64]:
        if energy_column is None:
            raise ValueError("energy_column is required when energy_weight > 0")

        mode = energy_selection_mode.lower()
        if mode == "global_minimum":
            return self._normalize_component(-energy_values)
        if mode != "stationary":
            raise ValueError(
                "energy_selection_mode must be 'global_minimum' or 'stationary', "
                f"got {energy_selection_mode!r}."
            )

        stationary_mask = self._stationary_mask(
            frame,
            gradient_columns=gradient_columns,
            gradient_norm_column=gradient_norm_column,
            curvature_columns=curvature_columns,
            stationary_orders=stationary_orders,
            gradient_tolerance=gradient_tolerance,
            curvature_tolerance=curvature_tolerance,
        )
        component = np.zeros(len(frame), dtype=np.float64)
        if not np.any(stationary_mask):
            return component

        stationary_energies = energy_values[stationary_mask]
        if stationary_energies.size == 1:
            component[stationary_mask] = 1.0
            return component

        component[stationary_mask] = 0.5 + 0.5 * self._normalize_component(-stationary_energies)
        return component

    def _stationary_mask(
        self,
        frame: pd.DataFrame,
        *,
        gradient_columns: list[str] | None,
        gradient_norm_column: str | None,
        curvature_columns: list[str] | None,
        stationary_orders: int | Sequence[int],
        gradient_tolerance: float,
        curvature_tolerance: float,
    ) -> npt.NDArray[np.bool_]:
        gradient_norm = self._gradient_norm(
            frame,
            gradient_columns=gradient_columns,
            gradient_norm_column=gradient_norm_column,
        )
        curvature_values = self._curvature_values(frame, curvature_columns=curvature_columns)
        negative_counts = np.sum(curvature_values < -curvature_tolerance, axis=1)
        positive_counts = np.sum(curvature_values > curvature_tolerance, axis=1)
        fully_classified = (negative_counts + positive_counts) == curvature_values.shape[1]

        if isinstance(stationary_orders, int):
            orders = {stationary_orders}
        else:
            orders = {int(order) for order in stationary_orders}
        if any(order < 0 for order in orders):
            raise ValueError("stationary_orders must be non-negative")

        return (
            (gradient_norm <= gradient_tolerance)
            & fully_classified
            & np.isin(negative_counts, list(orders))
        )

    @staticmethod
    def _gradient_norm(
        frame: pd.DataFrame,
        *,
        gradient_columns: list[str] | None,
        gradient_norm_column: str | None,
    ) -> npt.NDArray[np.float64]:
        if gradient_norm_column is not None:
            return np.abs(frame[gradient_norm_column].to_numpy(dtype=np.float64))
        if gradient_columns is not None:
            gradient_values = frame.loc[:, gradient_columns].to_numpy(dtype=np.float64)
            return np.linalg.norm(gradient_values, axis=1)
        raise ValueError(
            "Stationary-point energy selection requires gradient_norm_column or gradient_columns."
        )

    @staticmethod
    def _curvature_values(
        frame: pd.DataFrame,
        *,
        curvature_columns: list[str] | None,
    ) -> npt.NDArray[np.float64]:
        if not curvature_columns:
            raise ValueError("Stationary-point energy selection requires curvature_columns.")
        return frame.loc[:, curvature_columns].to_numpy(dtype=np.float64)

    @staticmethod
    def _normalize_component(
        values: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        if values.size == 0:
            return values
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if np.isclose(vmax, vmin):
            return np.zeros_like(values)
        return (values - vmin) / (vmax - vmin)

    @staticmethod
    def _min_distance_to_references(
        points: npt.NDArray[np.float64],
        references: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        if points.size == 0:
            return np.zeros(0, dtype=np.float64)
        if references.size == 0:
            return np.zeros(points.shape[0], dtype=np.float64)
        return np.min(distance.cdist(points, references), axis=1)

    @staticmethod
    def _stack_references(
        seeds: npt.NDArray[np.float64],
        selected: list[npt.NDArray[np.float64]],
        *,
        width: int,
    ) -> npt.NDArray[np.float64]:
        selected_array = np.vstack(selected) if selected else np.zeros((0, width), dtype=np.float64)
        if seeds.size == 0:
            return selected_array
        if selected_array.size == 0:
            return seeds
        return np.vstack([seeds, selected_array])

    def _build_dataset(self, recompute: bool) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        reference_features: list[str] | None = None
        self._slices = []

        start = 0
        for index, (name, maps) in enumerate(zip(self.names, self.maps, strict=False)):
            frame = self._get_map_frame(maps, recompute).copy()
            self._attach_map_metadata(index, maps, frame)
            current_features = self._get_map_features(maps)
            if reference_features is None:
                reference_features = current_features
            elif set(current_features) != set(reference_features):
                raise ValueError("All maps in MultiMaps must expose the same feature columns")

            metadata_columns = [
                column
                for column in self._METADATA_COLUMNS
                if column in frame.columns and column not in reference_features
            ]
            ordered = frame.loc[:, ["x", "y", "z", *metadata_columns, *reference_features]].copy()
            ordered.insert(0, "point_index", np.arange(len(ordered), dtype=np.int64))
            ordered.insert(0, "map_index", index)
            ordered.insert(0, "system", name)
            frames.append(ordered)

            stop = start + len(ordered)
            self._slices.append(slice(start, stop))
            start = stop

        if reference_features is None:
            raise RuntimeError("No maps data available.")

        self.features = reference_features
        self.data = pd.concat(frames, ignore_index=True)
        return self.data

    def _ensure_data(self) -> pd.DataFrame:
        if self.data is None:
            return self._build_dataset(recompute=False)
        return self.data

    @staticmethod
    def _subset_graph_result(
        graph: GraphResult,
        mask: np.ndarray[Any, np.dtype[np.bool_]],
    ) -> GraphResult:
        indexes = np.flatnonzero(mask).astype(np.int64, copy=False)
        remap = -np.ones(len(mask), dtype=np.int64)
        remap[indexes] = np.arange(len(indexes), dtype=np.int64)

        edge_table = graph.edge_table.copy()
        if not edge_table.empty:
            sources = edge_table["source"].to_numpy(dtype=np.int64)
            targets = edge_table["target"].to_numpy(dtype=np.int64)
            edge_mask = np.isin(sources, indexes) & np.isin(targets, indexes)
            edge_table = edge_table.loc[edge_mask].copy()
            edge_table.loc[:, "source"] = remap[edge_table["source"].to_numpy(dtype=np.int64)]
            edge_table.loc[:, "target"] = remap[edge_table["target"].to_numpy(dtype=np.int64)]
            edge_table = edge_table.reset_index(drop=True)

        return GraphResult(
            mode=graph.mode,
            feature_columns=list(graph.feature_columns),
            node_weight_column=graph.node_weight_column,
            node_table=graph.node_table.iloc[indexes].reset_index(drop=True).copy(),
            node_weights=graph.node_weights[indexes].astype(np.float64, copy=True),
            edge_table=edge_table,
            matrix=graph.matrix[indexes][:, indexes].tocsr(),
            metadata=deepcopy(graph.metadata),
        )

    def _get_map_frame(self, maps: "Maps", recompute: bool) -> pd.DataFrame:
        if recompute or maps.data is None:
            if maps.contactspace is None:
                raise RuntimeError("Each Maps instance must define a contact space")
            return maps.atcontactspace()
        return maps.data

    def _get_map_features(self, maps: "Maps") -> list[str]:
        features = list(maps.features)
        if features:
            return features
        if maps.data is None:
            raise RuntimeError("No maps data available.")
        return [
            column
            for column in maps.data.columns
            if column
            not in {
                "x",
                "y",
                "z",
                *self._METADATA_COLUMNS,
                *getattr(maps, "_METADATA_COLUMNS", set()),
            }
        ]

    def _attach_map_metadata(self, index: int, maps: "Maps", frame: pd.DataFrame) -> None:
        metadata = self.map_metadata[index]
        if not metadata:
            return
        for column, value in metadata.items():
            frame.loc[:, column] = value
            if maps.data is not None:
                maps.data.loc[:, column] = value
        if maps.data is not None and hasattr(maps, "_refresh_features"):
            maps._refresh_features()

    def _propagate_columns(self, columns: list[str]) -> None:
        if self.data is None:
            raise RuntimeError("No MultiMaps data available.")

        for maps, data_slice in zip(self.maps, self._slices, strict=False):
            if maps.data is None:
                raise RuntimeError("Child Maps data is not initialized.")
            for column in columns:
                maps.data.loc[:, column] = self.data.iloc[data_slice][column].to_numpy()

    def _aggregate_graph(
        self, labels: np.ndarray[Any, np.dtype[np.int64]]
    ) -> np.ndarray[Any, np.dtype[np.int64]]:
        graph = np.zeros((self.nclusters, self.nclusters), dtype=np.int64)
        for maps, data_slice in zip(self.maps, self._slices, strict=False):
            local_labels = labels[data_slice]
            if maps.contactspace is None:
                raise RuntimeError("Each Maps instance must define a contact space")
            local_graph = aggregate_cluster_graph(local_labels, maps.contactspace.neighbors)
            nrows, ncols = local_graph.shape
            graph[:nrows, :ncols] += local_graph
        return graph

    def _collect_contactspace_column(self, column: str) -> np.ndarray[Any, np.dtype[np.float64]]:
        values: list[np.ndarray[Any, np.dtype[np.float64]]] = []
        for maps in self.maps:
            if maps.contactspace is None:
                raise RuntimeError("Each Maps instance must define a contact space")
            contactspace_data = getattr(maps.contactspace, "data", None)
            if contactspace_data is not None and column in contactspace_data.columns:
                values.append(contactspace_data[column].to_numpy(dtype=np.float64))
                continue
            maps_data = getattr(maps, "data", None)
            if maps_data is not None and column in maps_data.columns:
                values.append(maps_data[column].to_numpy(dtype=np.float64))
                continue
            raise ValueError(
                f"node_weight_column {column!r} not present in child contactspace.data."
            )
        return np.concatenate(values).astype(np.float64, copy=False)

    def _register_archetypes_on_children(
        self,
        archetype_table: pd.DataFrame,
        *,
        kind: str,
        iteration: int | None,
        label_status: str,
    ) -> None:
        for map_index, group in archetype_table.groupby("map_index", sort=False):
            ordered = group.sort_values("selection_rank").reset_index(drop=True)
            metadata: dict[str, np.ndarray[Any, Any]] = {
                "selection_rank": ordered["selection_rank"].to_numpy(dtype=np.int64),
                "selection_score": ordered["selection_score"].to_numpy(dtype=np.float64),
                "probability_score": ordered["probability_score"].to_numpy(dtype=np.float64),
                "extremeness_score": ordered["extremeness_score"].to_numpy(dtype=np.float64),
                "diversity_score": ordered["diversity_score"].to_numpy(dtype=np.float64),
            }
            for column in [
                "euclidean_extremeness_score",
                "endpoint_score",
                "endpointness_score",
                "geodesic_score",
                "branching_score",
            ]:
                if column in ordered.columns:
                    metadata[column] = ordered[column].to_numpy(dtype=np.float64)
            self.maps[int(map_index)].add_special_points(
                ordered["point_index"].to_numpy(dtype=np.int64),
                kind=kind,
                iteration=iteration,
                label_status=label_status,
                replace_kind=False,
                **metadata,
            )

    def _combine_contactspace_neighbors(self) -> list[np.ndarray[Any, np.dtype[np.int64]]]:
        combined: list[np.ndarray[Any, np.dtype[np.int64]]] = []
        offset = 0
        for maps in self.maps:
            if maps.contactspace is None:
                raise RuntimeError("Each Maps instance must define a contact space")
            for row in maps.contactspace.neighbors:
                local_neighbors = np.asarray(row, dtype=np.int64).reshape(-1).copy()
                mask = local_neighbors >= 0
                local_neighbors[mask] += offset
                combined.append(local_neighbors)
            offset += len(maps.contactspace.neighbors)
        return combined

    def _select_random_state(self, nclusters: int, random_state: int | None) -> int:
        if not clustering_uses_random_state(self.cluster_method):
            return 0
        if random_state is not None:
            return random_state
        if (
            self.best_clusters is not None
            and self.cluster_screening_method == self.cluster_method
            and nclusters in self.best_clusters["nclusters"].values
        ):
            return int(
                self.best_clusters[self.best_clusters["nclusters"] == nclusters][
                    "random_state"
                ].values[0]
            )
        return int(np.random.randint(0, 1000))


class MultiMapsFromFile(MultiMaps):
    def __init__(self, filename: str) -> None:
        from mapsy.io.parser import SystemParser
        from mapsy.maps import (
            _namespace_system,
        )

        with open(Path(filename).expanduser().resolve()) as handle:
            params = load(handle, SafeLoader) or {}

        control = params.get("control") or {}
        system = params.get("system")
        symmetryfunctions = params.get("symmetryfunctions")
        contactspace = params.get("contactspace")

        if system is None:
            raise RuntimeError("System section missing in input file")
        if symmetryfunctions is None:
            raise RuntimeError("Symmetry functions section missing in input file")
        if contactspace is None:
            raise RuntimeError("Contact space section missing in input file")

        basepath = Path(filename).expanduser().resolve().parent
        systemmodel = _namespace_system(system)
        system_parser = SystemParser(systemmodel, basepath=basepath)
        file_records = system_parser.file_records()
        system_files = [record.path for record in file_records]

        if len(system_files) > 1 and systemmodel.properties:
            raise NotImplementedError(
                "MultiMapsFromFile does not support per-system properties yet."
            )

        workers = _resolve_multimaps_workers(control, len(system_files))
        tasks = [
            (path, str(basepath), system, symmetryfunctions, contactspace) for path in system_files
        ]

        if workers > 1:
            with ProcessingPool(nodes=workers) as pool:
                maps_list = list(pool.map(_build_maps_from_file_worker, tasks))
        else:
            maps_list = [_build_maps_from_file_worker(task) for task in tasks]

        super().__init__(
            maps_list,
            names=[Path(path).stem for path in system_files],
            metadata=[record.metadata() for record in file_records],
        )
        self.debug = bool(control.get("debug", False))
        self.verbosity = int(control.get("verbosity", 0))
        self.workers = workers
