import logging
import pickle
from collections.abc import Callable, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, TypeAlias

import dill
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.spatial.distance as distance
from ase.geometry import get_distances
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
from pathos.multiprocessing import ProcessingPool
from sklearn.preprocessing import StandardScaler
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
    screen_clusters,
)
from mapsy.analysis import (
    propagate_archetypes as propagate_archetypes_on_graph,
)
from mapsy.analysis import (
    select_archetypes as select_feature_archetypes,
)
from mapsy.boundaries import ContactSpace
from mapsy.boundaries.ionic import IonicGeometry
from mapsy.clustering import clustering_uses_random_state, normalize_cluster_method
from mapsy.data import ScalarField, System
from mapsy.results import (
    ArchetypePropagationResult,
    ArchetypeSelectionResult,
    ClusterResult,
    ClusterScreeningResult,
    GraphResult,
    PCAAnalysisResult,
    PCAResult,
)
from mapsy.symfunc.symmetryfunction import SymmetryFunction
from mapsy.utils import full2chunk, multiproc

logger = logging.getLogger(__name__)

AxesLike: TypeAlias = Axes | npt.NDArray[np.object_]  # array of Axes objects
NDArrayF: TypeAlias = npt.NDArray[np.float64]
NDArrayI: TypeAlias = npt.NDArray[np.int64]


# Tell mypy we are calling multiproc with a worker that returns a list of ndarrays.
# At runtime this just forwards to mapsy.utils.multiproc.
def _run_multiproc_lists(
    func: Callable[[NDArrayF], list[NDArrayF]],
    args: Sequence[NDArrayF],
    workers: int | None = None,
) -> list[list[NDArrayF]]:
    if len(args) == 0:
        return []
    if workers is not None and workers <= 1:
        return [func(arg) for arg in args]
    if workers is not None:
        with ProcessingPool(nodes=max(1, int(workers))) as pool:
            return list(pool.map(func, list(args)))
    return multiproc(func, list(args))


def _chunk_positions(
    positions: npt.NDArray,
    workers: int | None,
) -> list[NDArrayF]:
    if workers is None:
        return [np.asarray(a, dtype=np.float64) for a in full2chunk(positions)]
    n_chunks = min(max(1, int(workers)), len(positions))
    if n_chunks <= 1:
        return [np.asarray(positions, dtype=np.float64)]
    return [np.asarray(a, dtype=np.float64) for a in np.array_split(positions, n_chunks, axis=0)]


class SpecialPointRegistry:
    """Registry of contact-space points selected for downstream simulations."""

    _BASE_COLUMNS = ["point_index", "kind", "iteration", "label_status"]
    _ALLOWED_LABEL_STATUS = {"unlabeled", "completed", "failed"}

    def __init__(self) -> None:
        self._data = pd.DataFrame(columns=self._BASE_COLUMNS)

    @staticmethod
    def _normalize_single_metadata_value(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return value.item()
            if value.ndim == 1 and value.size == 1:
                return value.reshape(-1)[0]
            return value
        if isinstance(value, pd.Series):
            if len(value) == 1:
                return value.iloc[0]
            return value.to_numpy()
        if isinstance(value, (list, tuple)):
            if len(value) == 1:
                return value[0]
            return value
        return value

    def add(
        self,
        point_indexes: npt.ArrayLike,
        *,
        kind: str,
        iteration: int | None = None,
        label_status: str = "unlabeled",
        replace_kind: bool = False,
        **metadata: Any,
    ) -> pd.DataFrame:
        indexes = np.asarray(point_indexes, dtype=np.int64).reshape(-1)
        if indexes.size == 0:
            return self.frame(kind=kind)

        self._validate_label_status(label_status)

        if replace_kind:
            self.remove(kind=kind)

        rows = pd.DataFrame(
            {
                "point_index": indexes,
                "kind": kind,
                "iteration": iteration,
                "label_status": label_status,
            }
        )

        for key, value in metadata.items():
            if np.isscalar(value) or value is None:
                rows.loc[:, key] = value
                continue

            if indexes.size == 1:
                if key not in rows.columns:
                    rows[key] = pd.Series([None] * len(rows), dtype=object)
                rows.at[rows.index[0], key] = self._normalize_single_metadata_value(value)
                continue

            values = np.asarray(value)
            if values.reshape(-1).size != indexes.size:
                raise ValueError(
                    f"Metadata column {key!r} has length {values.reshape(-1).size}, expected {indexes.size}."
                )
            rows.loc[:, key] = values.reshape(-1)

        self._data = pd.concat([self._data, rows], ignore_index=True)
        self._data = self._data.drop_duplicates(subset=["point_index", "kind"], keep="last")
        return self.frame(kind=kind)

    def remove(
        self,
        *,
        kind: str | None = None,
        point_indexes: npt.ArrayLike | None = None,
    ) -> None:
        if self._data.empty:
            return

        mask = np.ones(len(self._data), dtype=bool)
        if kind is not None:
            mask &= self._data["kind"].to_numpy() == kind
        if point_indexes is not None:
            indexes = np.asarray(point_indexes, dtype=np.int64).reshape(-1)
            mask &= self._data["point_index"].isin(indexes).to_numpy()

        self._data = self._data.loc[~mask].reset_index(drop=True)

    def update(
        self,
        *,
        kind: str | None = None,
        point_indexes: npt.ArrayLike | None = None,
        **metadata: Any,
    ) -> pd.DataFrame:
        if self._data.empty:
            return self.frame(kind=kind)

        mask = np.ones(len(self._data), dtype=bool)
        if kind is not None:
            mask &= self._data["kind"].to_numpy() == kind
        if point_indexes is not None:
            indexes = np.asarray(point_indexes, dtype=np.int64).reshape(-1)
            mask &= self._data["point_index"].isin(indexes).to_numpy()

        if not np.any(mask):
            return self.frame(kind=kind)

        count = int(np.count_nonzero(mask))
        for key, value in metadata.items():
            if key == "label_status":
                if np.isscalar(value) or value is None:
                    self._validate_label_status(str(value))
                else:
                    statuses = np.asarray(value, dtype=object).reshape(-1)
                    if statuses.size != count:
                        raise ValueError(
                            f"Metadata column {key!r} has length {statuses.size}, expected {count}."
                        )
                    for status in statuses:
                        self._validate_label_status(str(status))

            if np.isscalar(value) or value is None:
                self._data.loc[mask, key] = value
                continue

            if count == 1:
                if key not in self._data.columns:
                    self._data[key] = pd.Series([None] * len(self._data), dtype=object)
                self._data.at[self._data.index[mask][0], key] = (
                    self._normalize_single_metadata_value(value)
                )
                continue

            values = np.asarray(value, dtype=object).reshape(-1)
            if values.size != count:
                raise ValueError(
                    f"Metadata column {key!r} has length {values.size}, expected {count}."
                )
            self._data.loc[mask, key] = values

        return self.frame(kind=kind)

    def frame(
        self,
        *,
        kind: str | None = None,
        label_status: str | Sequence[str] | None = None,
    ) -> pd.DataFrame:
        data = self._data
        if kind is not None:
            data = data.loc[data["kind"] == kind]
        if label_status is not None:
            statuses = [label_status] if isinstance(label_status, str) else list(label_status)
            for status in statuses:
                self._validate_label_status(status)
            data = data.loc[data["label_status"].isin(statuses)]
        return data.copy()

    def indexes(
        self,
        *,
        kind: str | None = None,
        label_status: str | Sequence[str] | None = None,
    ) -> NDArrayI:
        frame = self.frame(kind=kind, label_status=label_status)
        if frame.empty:
            return np.array([], dtype=np.int64)
        return frame["point_index"].to_numpy(dtype=np.int64)

    def reference_indexes(self) -> NDArrayI:
        """Unique point indexes that should act as reference special points."""
        indexes = self.indexes()
        if indexes.size == 0:
            return indexes
        return np.unique(indexes.astype(np.int64))

    def _validate_label_status(self, label_status: str) -> None:
        if label_status not in self._ALLOWED_LABEL_STATUS:
            raise ValueError(
                f"label_status must be one of {sorted(self._ALLOWED_LABEL_STATUS)}, got {label_status!r}."
            )


# Class for Maps, which includes functionality for processing symmetry functions
class Maps:
    _METADATA_COLUMNS = {
        "region",
        "core_distance",
        "is_core",
        "source_file",
        "source_file_name",
        "source_file_stem",
        "source_folder",
        "source_folder_name",
        "source_folder_number",
    }
    bedug: bool = False
    verbosity: int = 0  # Verbosity level (how much logging to show)

    hascs: bool = False  # Indicates whether contact space has been defined
    hasatomicsf: bool = False  # Indicates if atomic symmetry functions are used
    nsfs: int = 0  # Number of symmetry functions

    system: System | None = None
    symmetryfunctions: list[SymmetryFunction] = []
    contactspace: ContactSpace | None = None

    data: pd.DataFrame | None = None
    features: list[str] = []

    npca: int | None = None
    pca_analysis_result: PCAAnalysisResult | None = None
    pca_result: PCAResult | None = None
    graph_result: GraphResult | None = None
    archetype_selection_result: ArchetypeSelectionResult | None = None
    archetype_propagation_result: ArchetypePropagationResult | None = None

    nclusters: int = 0
    cluster_method: str = "spectral"
    cluster_screening_method: str | None = None
    cluster_result: ClusterResult | None = None
    cluster_centers: npt.NDArray[np.float64] | None = None
    cluster_graph: npt.NDArray[np.int64] | None = None
    cluster_edges: npt.NDArray[np.int64] | None = None
    best_clusters: pd.DataFrame | None = None
    cluster_screening: pd.DataFrame | None = None
    special_points: SpecialPointRegistry

    def __init__(
        self,
        system: System,
        symmetryfunctions: list[SymmetryFunction],
        contactspace: ContactSpace | None = None,
    ) -> None:
        # Set the contact space if provided
        if contactspace is not None:
            self.contactspace: ContactSpace = contactspace
        # Initialize system and symmetry functions
        self.system: System = system
        self.symmetryfunctions: list[SymmetryFunction] = deepcopy(symmetryfunctions)
        self.hasatomicsf = any(sf.atomic for sf in self.symmetryfunctions)
        self.nsfs = self.len()
        self.special_points = SpecialPointRegistry()
        # Call setup for each symmetry function to initialize with the system
        for sf in self.symmetryfunctions:
            sf.setup(self.system.atoms)

    @property
    def centroids(self) -> NDArrayI:
        """Backward-compatible view over centroid special points."""
        return self.special_points.indexes(kind="centroid")

    @centroids.setter
    def centroids(self, point_indexes: npt.ArrayLike) -> None:
        indexes = np.asarray(point_indexes, dtype=np.int64).reshape(-1)
        self.special_points.add(
            indexes,
            kind="centroid",
            label_status="unlabeled",
            replace_kind=True,
        )

    # Method to return the number of symmetry functions
    def len(self) -> int:
        return len(self.symmetryfunctions)

    def atcontactspace(
        self,
        *,
        workers: int | None = None,
        release_contactspace_cache: bool = False,
    ) -> pd.DataFrame:
        """
        Compute symmetry functions for points in the contact space
        """
        # Calculation symmetry functions on contact points
        if self.contactspace is None:
            raise RuntimeError("Trying to use maps.compute() without contact space")
        contactspace = self.contactspace
        positions: npt.NDArray = self.contactspace.data[["x", "y", "z"]].values
        if release_contactspace_cache:
            self.release_contactspace_cache()

        if release_contactspace_cache:
            self.contactspace = None
        try:
            self.data = self.atpoints(positions, workers=workers)
        finally:
            self.contactspace = contactspace
        self._sync_contactspace_metadata()
        self._sync_contactspace_features()
        self._refresh_features()

        return self.data

    def release_contactspace_cache(self) -> None:
        """Release dense contact-space fields not needed for tabular feature analysis."""
        if self.contactspace is None:
            return
        release = getattr(self.contactspace, "release_dense_fields", None)
        if release is not None:
            release()

    def annotate_contactspace(
        self,
        name: str,
        values: npt.ArrayLike,
        *,
        as_feature: bool = True,
    ) -> npt.NDArray[np.float64]:
        """Attach pointwise values to the current contact space."""
        if self.contactspace is None:
            raise RuntimeError("Trying to annotate contact space without contact space")

        annotation = self.contactspace.annotate(name, values, as_feature=as_feature)
        if self.data is not None:
            self.data.loc[:, name] = annotation
            self._refresh_features()

        return annotation

    def annotate_ionic_distance(
        self,
        column: str = "ionic_distance",
        radiusmode: str = "muff",
        alpha: float = 1.0,
        radius_table_file: str | None = None,
        *,
        as_feature: bool = True,
    ) -> npt.NDArray[np.float64]:
        """Annotate contact-space points with their distance to the ionic interface."""
        if self.contactspace is None:
            raise RuntimeError("Trying to annotate contact space without contact space")
        if self.system is None:
            raise RuntimeError("Trying to annotate ionic distance without a system")

        system = self.system
        metric = IonicGeometry(
            mode=radiusmode,
            alpha=alpha,
            system=system,
            radius_table_file=radius_table_file,
        )
        positions = self.contactspace.data[["x", "y", "z"]].to_numpy(dtype=np.float64)
        distances = metric.signed_distance(positions)
        return self.annotate_contactspace(column, distances, as_feature=as_feature)

    def add_special_points(
        self,
        point_indexes: npt.ArrayLike,
        *,
        kind: str,
        iteration: int | None = None,
        label_status: str = "unlabeled",
        replace_kind: bool = False,
        **metadata: Any,
    ) -> pd.DataFrame:
        """Register contact-space points chosen for downstream simulations."""
        return self.special_points.add(
            point_indexes,
            kind=kind,
            iteration=iteration,
            label_status=label_status,
            replace_kind=replace_kind,
            **metadata,
        )

    def select_special_points(
        self,
        npoints: int,
        *,
        kind: str = "adaptive",
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
        real_space_weight: float = 0.0,
        feature_space_weight: float = 1.0,
        energy_weight: float = 0.0,
        uncertainty_weight: float = 0.0,
        scale_features: bool = True,
        energy_selection_mode: str = "global_minimum",
        gradient_columns: list[str] | None = None,
        gradient_norm_column: str | None = None,
        curvature_columns: list[str] | None = None,
        stationary_orders: int | Sequence[int] = 0,
        gradient_tolerance: float = 1.0e-6,
        curvature_tolerance: float = 1.0e-8,
        **metadata: Any,
    ) -> pd.DataFrame:
        """Select and register new special points in one step."""
        selected = self.select_points(
            npoints=npoints,
            feature_columns=feature_columns,
            energy_column=energy_column,
            uncertainty_column=uncertainty_column,
            special_point_indexes=special_point_indexes,
            centroid_indexes=centroid_indexes,
            region=region,
            real_space_weight=real_space_weight,
            feature_space_weight=feature_space_weight,
            energy_weight=energy_weight,
            uncertainty_weight=uncertainty_weight,
            scale_features=scale_features,
            energy_selection_mode=energy_selection_mode,
            gradient_columns=gradient_columns,
            gradient_norm_column=gradient_norm_column,
            curvature_columns=curvature_columns,
            stationary_orders=stationary_orders,
            gradient_tolerance=gradient_tolerance,
            curvature_tolerance=curvature_tolerance,
        )

        point_indexes = selected.index.to_numpy(dtype=np.int64)
        add_metadata = dict(metadata)
        if store_selection_metadata:
            for column in [
                "selection_rank",
                "selection_score",
                "real_space_score",
                "feature_space_score",
                "energy_score",
                "uncertainty_score",
            ]:
                if column in selected.columns and column not in add_metadata:
                    add_metadata[column] = selected[column].to_numpy()

        self.add_special_points(
            point_indexes,
            kind=kind,
            iteration=iteration,
            label_status=label_status,
            replace_kind=replace_kind,
            **add_metadata,
        )

        special = self.get_special_points(kind=kind)
        special = special.loc[special["point_index"].isin(point_indexes)].copy()
        if "selection_rank" in special.columns:
            return special.sort_values("selection_rank").reset_index(drop=True)
        return special.reset_index(drop=True)

    def get_special_points(
        self,
        *,
        kind: str | None = None,
        label_status: str | Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Return the special-point registry joined with point data when available."""
        frame = self.special_points.frame(kind=kind, label_status=label_status)
        if self.data is None or frame.empty:
            return frame

        point_data = self.data.copy()
        point_data.index.name = "point_index"
        return frame.merge(point_data, on="point_index", how="left")

    def update_special_points(
        self,
        *,
        kind: str | None = None,
        point_indexes: npt.ArrayLike | None = None,
        **metadata: Any,
    ) -> pd.DataFrame:
        """Update metadata for existing special points."""
        return self.special_points.update(
            kind=kind,
            point_indexes=point_indexes,
            **metadata,
        )

    def atpoints(self, positions: npt.NDArray, *, workers: int | None = None) -> pd.DataFrame:
        """
        Compute symmetry functions for the given positions in parallel
        """

        # Parallelize over the postitions
        args: list[NDArrayF] = _chunk_positions(positions, workers)
        # Map chunks -> per-chunk list of arrays (one array per symmetry function)
        chunk_results: list[list[NDArrayF]] = _run_multiproc_lists(
            self._process_chunk,
            args,
            workers=workers,
        )
        # Combine across chunks: for each symmetry function, vstack chunk results
        combined_per_sf: list[list[NDArrayF]] = [[] for _ in range(self.nsfs)]
        for chunk_list in chunk_results:
            for i in range(self.nsfs):
                combined_per_sf[i].append(chunk_list[i])
        results: list[NDArrayF] = [
            (
                np.vstack(lst)
                if lst
                else np.zeros((0, len(self.symmetryfunctions[i].keys)), dtype=np.float64)
            )
            for i, lst in enumerate(combined_per_sf)
        ]
        return self._results2df(positions, results)

    def tofile(self) -> None:
        """TODO"""
        return None

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
    def load(cls, filename: str | Path) -> "Maps":
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

    def plot(
        self,
        feature: str | None = None,
        index: int | None = None,
        axes: list[str] | None = None,
        region: int | None = 0,
        splitby: str = "z",
        set_aspect: str = "on",
        levels: int = 10,
        **kwargs: dict[str, Any],
    ) -> tuple[Figure, AxesLike]:
        """ """
        # Check that contact space exists
        if self.contactspace is None:
            raise RuntimeError("Trying to use maps.plot() without contact space")
        # Check that contact space maps have been generated
        if self.data is None:
            raise RuntimeError("No contact space data available.")
        # Filter data by region
        filterdata: pd.DataFrame = self.data[self.contactspace.data["region"] == region]
        # Check if feature or index is provided and if it is valid
        if feature is not None:
            if feature not in self.data.columns:
                raise ValueError(f"Feature {feature} not found in maps data.")
        elif index is not None:
            if index >= len(self.features) or index < 0:
                raise ValueError(f"Index {index} out of bounds.")
            feature = self.features[index]
        else:
            raise ValueError("Either feature or index must be provided.")
        if axes is None:
            axes = ["x", "y"]
        for axis in axes:
            if axis not in self.data.columns:
                raise ValueError(f"Axis {axis} not found in maps data.")
        if set_aspect not in ["on", "off", "equal", "scaled"]:
            raise ValueError("set_aspect must be one of ['on','off','equal','scaled']")
        # Select the axes for the plot
        x1 = filterdata[axes[0]].values.astype(np.float64)
        x2 = filterdata[axes[1]].values.astype(np.float64)
        # Select the axis to plot
        f = filterdata[feature].values.astype(np.float64)
        # Select the axis for splitting
        if splitby not in self.data.columns:
            raise ValueError(f"Split axis {splitby} not found in maps data.")
        s = filterdata[splitby].values
        nsplit = np.unique(s).size
        if nsplit == 1:
            fig, axs = plt.subplots(1, 1, figsize=(8, 4))
            axslist = [axs]
        else:
            fig, axs = plt.subplots(nsplit, 1, figsize=(8, 4 * nsplit))
            axslist = axs.flat
        fig.subplots_adjust(hspace=0.3)
        # Define levels for contour plot
        fmin = np.min(f)
        fmax = np.max(f)
        flevels = np.linspace(fmin, fmax, levels)
        # Generate 2D plots for each unique value of the split variable
        for i, ax in enumerate(axslist):
            sval = np.unique(s)[i]
            ax.set_title(f"Map of {feature} for {splitby} = {sval:6.2f}")
            mask = filterdata[splitby] == sval
            im = ax.tricontourf(x1[mask], x2[mask], f[mask], levels=flevels, **kwargs)
            ax.axis(set_aspect)
            ax.set_xlabel(f"{axes[0]}")
            ax.set_ylabel(f"{axes[1]}")
        # Add colorbar
        if nsplit == 1:
            fig.colorbar(im, ax=axs)
        else:
            fig.colorbar(im, ax=axs.ravel().tolist())
        return fig, axs

    def scatter(
        self,
        feature: str | None = None,
        index: int | None = None,
        axes: list[str] | None = None,
        region: int | None = 0,
        categorical: bool = False,
        centroids: bool = False,
        splitby: str | None = None,
        set_aspect: str = "on",
        **kwargs: dict[str, Any],
    ) -> tuple[Figure, AxesLike]:
        """ """
        # Check that contact space exists
        if self.contactspace is None:
            raise RuntimeError("Trying to use maps.scatter() without contact space")
        # Check that contact space maps have been generated
        if self.data is None:
            raise RuntimeError("No contact space data available.")
        frame = self.data.copy()
        contactspace_columns = self.contactspace.data
        for column in {feature, splitby}:
            if column is None or column in frame.columns:
                continue
            if column in contactspace_columns.columns:
                frame.loc[:, column] = contactspace_columns[column].to_numpy()
            else:
                raise ValueError(f"Column {column!r} not found in maps data or contactspace data.")

        mask = np.ones(len(frame), dtype=bool)
        if region is not None:
            mask &= contactspace_columns["region"].to_numpy(dtype=np.int64) == region
        filterdata = frame.loc[mask]
        # Check if feature or index is provided and if it is valid
        if feature is not None:
            if feature not in filterdata.columns:
                raise ValueError(f"Feature {feature} not found in maps data.")
        elif index is not None:
            if index >= len(self.features) or index < 0:
                raise ValueError(f"Index {index} out of bounds.")
            feature = self.features[index]
            logger.info(f"Plotting feature {feature}")
        else:
            raise ValueError("Either feature or index must be provided.")
        if axes is None:
            axes = ["x", "y"]
        for axis in axes:
            if axis not in self.data.columns:
                raise ValueError(f"Axis {axis} not found in maps data.")
        if set_aspect not in ["on", "off", "equal", "scaled"]:
            raise ValueError("set_aspect must be one of ['on','off','equal','scaled']")
        # Select the axes for the plot
        x1 = filterdata[axes[0]].values.astype(np.float64)
        x2 = filterdata[axes[1]].values.astype(np.float64)
        # Select the axis to plot
        f = filterdata[feature].values.astype(np.float64)
        # Select the axis for splitting
        if splitby is not None:
            if splitby not in filterdata.columns:
                raise ValueError(f"Split axis {splitby[0]} not found in maps data.")
            s = filterdata[splitby].values
            nsplit = np.unique(s).size
            fig, axs = plt.subplots(nsplit, 1, figsize=(8, 4 * nsplit))
            axslist = axs.flat
        else:
            nsplit = 1
            fig, axs = plt.subplots(nsplit, 1, figsize=(8, 4 * nsplit))
            axslist = [axs]
        if categorical:
            category_values = np.unique(f)
            nf = category_values.size
            colors = plt.cm.tab20(np.linspace(0, 1, nf, endpoint=False)).tolist()
        else:
            fmin = np.min(f)
            fmax = np.max(f)
        fig.subplots_adjust(hspace=0.3)
        # Plot the data
        # Generate 2D plots for each unique value of the split variable
        for i, ax in enumerate(axslist):
            if splitby is not None:
                sval = np.unique(s)[i]
                ax.set_title(f"Map of {feature} for {splitby} = {sval:6.2f}")
                mask = filterdata[splitby] == sval
                x1m = x1[mask]
                x2m = x2[mask]
                fm = f[mask]
            else:
                ax.set_title(f"Map of {feature}")
                x1m = x1
                x2m = x2
                fm = f
            ax.set_xlabel(f"{axes[0]}")
            ax.set_ylabel(f"{axes[1]}")
            if not categorical:
                if f is None:
                    scatter = ax.scatter(x1m, x2m, **kwargs)
                else:
                    scatter = ax.scatter(x1m, x2m, c=fm, vmin=fmin, vmax=fmax, **kwargs)
            else:
                for i, fvalue in enumerate(category_values):
                    label_value = (
                        str(int(fvalue)) if np.isclose(fvalue, int(fvalue)) else str(fvalue)
                    )
                    scatter = ax.scatter(
                        x1m[fm == fvalue],
                        x2m[fm == fvalue],
                        color=colors[i],
                        label=f"{feature} = {label_value}",
                        **kwargs,
                    )
            if centroids:
                centroid_indexes = self.centroids
                if centroid_indexes.size == 0:
                    raise RuntimeError("No centroids available.")
                pos = self.data.loc[centroid_indexes, axes].values
                if splitby is not None:
                    posmask = self.data.loc[centroid_indexes, splitby].values == sval
                else:
                    posmask = np.ones(pos.shape[0], dtype=bool)
                ax.scatter(
                    pos[posmask, 0],
                    pos[posmask, 1],
                    c="black",
                    marker="x",
                    label="Centroids",
                    **kwargs,
                )
            if categorical:
                leg = ax.legend(bbox_to_anchor=(1.01, 0.5), loc="center left")
                for lh in leg.legend_handles:
                    lh.set_alpha(1)
            ax.axis(set_aspect)
        if not categorical and f is not None:
            if nsplit > 1:
                colorbar = fig.colorbar(scatter, ax=axs.ravel().tolist())
            else:
                colorbar = fig.colorbar(scatter, ax=ax)
            colorbar.solids.set_alpha(1.0)
        return fig, axs

    def scatter_core_projection(
        self,
        feature: str | None = None,
        index: int | None = None,
        *,
        plane: tuple[str, str] = ("x", "y"),
        selector: str = "core",
        region: int | None = 0,
        core_only: bool = True,
        max_core_distance: float | None = None,
        categorical: bool = False,
        set_aspect: str = "on",
        **kwargs: dict[str, Any],
    ) -> tuple[Figure, Axes]:
        if self.contactspace is None:
            raise RuntimeError("Trying to use maps.scatter_core_projection() without contact space")
        if self.data is None:
            raise RuntimeError("No contact space data available.")
        if len(plane) != 2 or plane[0] == plane[1]:
            raise ValueError(f"plane must contain two distinct axes, got {plane!r}.")
        if selector not in {"core", "top", "bottom", "weighted_mean"}:
            raise ValueError("selector must be one of {'core', 'top', 'bottom', 'weighted_mean'}.")
        if categorical and selector == "weighted_mean":
            raise ValueError("selector='weighted_mean' is only valid for numeric features.")
        if set_aspect not in ["on", "off", "equal", "scaled"]:
            raise ValueError("set_aspect must be one of ['on','off','equal','scaled']")

        frame = self.data.copy()
        required_columns = {feature, *plane, "core_distance", "probability"}
        for column in required_columns:
            if column is None or column in frame.columns:
                continue
            if column in self.contactspace.data.columns:
                frame.loc[:, column] = self.contactspace.data[column].to_numpy()
            else:
                raise ValueError(f"Column {column!r} not found in maps data or contactspace data.")

        if feature is not None:
            if feature not in frame.columns:
                raise ValueError(f"Feature {feature} not found in maps data.")
        elif index is not None:
            if index >= len(self.features) or index < 0:
                raise ValueError(f"Index {index} out of bounds.")
            feature = self.features[index]
            logger.info(f"Plotting feature {feature}")
        else:
            raise ValueError("Either feature or index must be provided.")

        if feature is None:
            raise RuntimeError("Feature resolution failed unexpectedly.")

        remaining_axes = [axis for axis in ["x", "y", "z"] if axis not in plane]
        if len(remaining_axes) != 1:
            raise ValueError(f"plane must be a subset of ('x', 'y', 'z'), got {plane!r}.")
        normal_axis = remaining_axes[0]

        mask = self._core_selection_mask(
            core_only=core_only,
            max_core_distance=max_core_distance,
        )
        if region is not None:
            mask &= self.contactspace.data["region"].to_numpy(dtype=np.int64) == region

        columns = [plane[0], plane[1], normal_axis, feature, "core_distance", "probability"]
        projected_input = frame.loc[mask, columns].copy()
        if projected_input.empty:
            raise ValueError("No points available after applying the projection filters.")

        if selector == "weighted_mean":
            feature_values = pd.to_numeric(projected_input[feature], errors="raise")
            projected_input.loc[:, feature] = feature_values.to_numpy(dtype=np.float64)
            groups: list[dict[str, Any]] = []
            for coords, group in projected_input.groupby(list(plane), sort=False, dropna=False):
                weights = group["probability"].to_numpy(dtype=np.float64)
                if np.allclose(weights.sum(), 0.0):
                    weights = np.ones(len(group), dtype=np.float64)
                row = {
                    plane[0]: coords[0],
                    plane[1]: coords[1],
                    normal_axis: np.average(
                        group[normal_axis].to_numpy(dtype=np.float64),
                        weights=weights,
                    ),
                    feature: np.average(group[feature].to_numpy(dtype=np.float64), weights=weights),
                    "core_distance": float(
                        np.min(group["core_distance"].to_numpy(dtype=np.float64))
                    ),
                    "probability": float(np.max(weights)),
                    "multiplicity": int(len(group)),
                }
                groups.append(row)
            projected = pd.DataFrame(groups)
        else:
            ascending = {
                "core": [True, False, False],
                "top": [False, True, False],
                "bottom": [True, True, False],
            }[selector]
            sort_columns = {
                "core": ["core_distance", "probability", normal_axis],
                "top": [normal_axis, "core_distance", "probability"],
                "bottom": [normal_axis, "core_distance", "probability"],
            }[selector]
            ordered = projected_input.sort_values(
                sort_columns, ascending=ascending, kind="mergesort"
            )
            projected = ordered.groupby(list(plane), sort=False, as_index=False).first()
            counts = (
                projected_input.groupby(list(plane), sort=False)
                .size()
                .rename("multiplicity")
                .reset_index()
            )
            projected = projected.merge(counts, on=list(plane), how="left")

        x = projected[plane[0]].to_numpy(dtype=np.float64)
        y = projected[plane[1]].to_numpy(dtype=np.float64)

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.set_title(f"Core projection of {feature} on {plane[0]}-{plane[1]}")
        ax.set_xlabel(plane[0])
        ax.set_ylabel(plane[1])

        if categorical:
            values = projected[feature].to_numpy()
            category_values = pd.unique(values)
            colors = plt.cm.tab20(np.linspace(0, 1, len(category_values), endpoint=False)).tolist()
            for color, category in zip(colors, category_values, strict=False):
                scatter = ax.scatter(
                    x[values == category],
                    y[values == category],
                    color=color,
                    label=f"{feature} = {category}",
                    **kwargs,
                )
            leg = ax.legend(bbox_to_anchor=(1.01, 0.5), loc="center left")
            for lh in leg.legend_handles:
                lh.set_alpha(1)
        else:
            values = projected[feature].to_numpy(dtype=np.float64)
            scatter = ax.scatter(x, y, c=values, **kwargs)
            colorbar = fig.colorbar(scatter, ax=ax)
            colorbar.solids.set_alpha(1.0)

        ax.axis(set_aspect)
        return fig, ax

    def scatter_pca_grid(
        self,
        feature: str | None = None,
        index: int | None = None,
        set_aspect: str = "on",
        **kwargs: dict[str, Any],
    ) -> tuple[Figure, GridSpec]:
        # Check that contact space exists
        if self.contactspace is None:
            raise RuntimeError("Trying to use maps.scatter_pca_grid() without contact space")
        # Check that contact space maps have been generated
        if self.data is None:
            raise RuntimeError("No contact space data available.")
        # Check if feature or index is provided and if it is valid
        if feature is not None:
            if feature not in self.data.columns:
                raise ValueError(f"Feature {feature} not found in maps data.")
        elif index is not None:
            if index >= len(self.features) or index < 0:
                raise ValueError(f"Index {index} out of bounds.")
            feature = self.features[index]
            logger.info(f"Plotting feature {self.features[index]}")
        else:
            raise ValueError("Either feature or index must be provided.")
        if self.npca is None:
            raise ValueError("Missing principal components")
        if set_aspect not in ["on", "off", "equal", "scaled"]:
            raise ValueError("set_aspect must be one of ['on','off','equal','scaled']")
        # Step 1: Generate all combinations of PCA components
        pcalabels = [f"pca{i}" for i in range(self.npca)]
        pcamaxs = [self.data[pcalabels[i]].max() for i in range(self.npca)]
        pcamins = [self.data[pcalabels[i]].min() for i in range(self.npca)]
        pcaranges = [pcamaxs[i] - pcamins[i] for i in range(self.npca)]
        maxrange = max(pcaranges)
        pcaproportions = [pcaranges[i] / maxrange for i in range(self.npca)]
        xratios = [pcaproportions[i + 1] for i in range(self.npca - 1)]
        yratios = [pcaproportions[i] for i in range(self.npca - 1)]
        y = self.data[feature].values.astype(np.float64)
        # Step 2: Create subplots
        n_cols = len(xratios)  # Set the number of columns in the subplot grid
        n_rows = len(yratios)
        fig = plt.figure(figsize=(12, 12))
        gs = GridSpec(n_cols, n_rows, figure=fig, height_ratios=yratios, width_ratios=xratios)
        # Step 3: Plot each upper triangle component pair
        for i in range(self.npca - 1):
            for j in range(i + 1, self.npca):
                ax = fig.add_subplot(gs[i, j - 1])
                X1 = self.data[pcalabels[i]].values.astype(np.float64)
                X2 = self.data[pcalabels[j]].values.astype(np.float64)
                scatter = ax.scatter(X2, X1, c=y, edgecolor="k", **kwargs)
                ax.set_title(f"PC{j + 1} vs PC{i + 1}")
                ax.set_xlim(np.min(X2) - 0.5, np.max(X2) + 0.5)
                ax.set_ylim(np.min(X1) - 0.5, np.max(X1) + 0.5)
                ax.axis(set_aspect)
                ax.xaxis.set_major_locator(MultipleLocator(2.5))
                ax.yaxis.set_major_locator(MultipleLocator(2.5))
        # Step 4: Add color bar at the bottom left
        cbar_ax = fig.add_subplot(gs[n_rows - 1, 0 : n_rows - 1])
        colorbar = fig.colorbar(scatter, cax=cbar_ax, orientation="horizontal")
        colorbar.solids.set_alpha(1.0)
        # Step 5: Add Title in the remaining space
        if n_rows == 2:  # This fixes the case when npca == 3
            fig.suptitle(f"Maps of {feature}\n\n in PC Space", fontsize=22)
        else:
            title_ax = fig.add_subplot(gs[n_rows - 2, 0 : n_rows - 2])
            title_ax.text(
                0.5,
                0.5,
                f"Maps of {feature}\n\n in PC Space",
                ha="center",
                va="center",
                fontsize=22,
            )
            title_ax.axis("off")
        return fig, gs

    def tovolumetric(
        self,
        feature: str | None = None,
        index: int | None = None,
        reference: float = 0.0,
    ) -> ScalarField:
        """
        Converts contact space data to a volumetric (grid-based) scalar field.

        Args:
            feature:
            index:
            reference: the value assigned to grid points that do not belong to the contact space.

        Returns:
            ScalarField: A volumetric representation of the contact space data mapped on a structured
            grid in the system cell. Data on points different from the contact space is set to reference.
        """
        # Check that contact space exists
        if self.contactspace is None:
            raise RuntimeError("Trying to use maps.tovolumetric() without contact space")
        if self.contactspace.grid is None or self.contactspace.mask is None:
            raise RuntimeError(
                "Cannot convert to volumetric data after contact-space dense fields were "
                "released. Recreate the contact space, or compute features with "
                "release_contactspace_cache=False."
            )
        # Check that contact space maps have been generated
        if self.data is None:
            raise RuntimeError("No contact space data available.")
        # Check if feature or index is provided and if it is valid
        if feature is not None:
            if feature not in self.data.columns:
                raise ValueError(f"Feature {feature} not found in maps data.")
            f = self.data[feature].values.astype(np.float64)
        elif index is not None:
            if index >= len(self.features) or index < 0:
                raise ValueError(f"Index {index} out of bounds.")
            logger.info(f"Converting feature {self.features[index]}")
            f = self.data.iloc[:, index + 3].values.astype(np.float64)
        else:
            raise ValueError("Either feature or index must be provided.")
        if reference == 0.0:
            data = np.zeros(self.contactspace.grid.coordinates.shape[1:])
        else:
            data = reference * np.ones(self.contactspace.grid.coordinates.shape[1:])
        data[self.contactspace.mask] = f
        return ScalarField(self.contactspace.grid, data=data)

    def analyze_pca(
        self,
        scale: bool = False,
        *,
        core_only: bool = False,
        max_core_distance: float | None = None,
    ) -> PCAAnalysisResult:
        # Check that contact space exists
        if self.contactspace is None:
            raise RuntimeError("Trying to use maps.analyze_pca() without contact space")
        # Check that contact space maps have been generated
        if self.data is None:
            raise RuntimeError("No contact space data available.")
        mask = self._core_selection_mask(
            core_only=core_only,
            max_core_distance=max_core_distance,
        )
        X = self.data.loc[mask, self.features].values.astype(np.float64)
        if len(X) == 0:
            raise ValueError("No points available after applying the core filter for PCA.")
        result = fit_pca_analysis(X, feature_columns=list(self.features), scale=scale)
        self.pca_analysis_result = result
        return result

    def reduce(
        self,
        npca: int | None = None,
        scale: bool = False,
        *,
        core_only: bool = False,
        max_core_distance: float | None = None,
    ) -> PCAAnalysisResult | PCAResult:
        if npca is None:
            return self.analyze_pca(
                scale=scale,
                core_only=core_only,
                max_core_distance=max_core_distance,
            )
        if self.data is None:
            raise RuntimeError("No contact space data available.")
        analysis = self.analyze_pca(
            scale=scale,
            core_only=core_only,
            max_core_distance=max_core_distance,
        )
        data = self.data
        result = project_pca(analysis, data[self.features].values.astype(np.float64), npca=npca)
        self.npca = npca
        self.pca_result = result
        data.loc[:, result.transformed_columns] = result.transformed_values
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
        core_only: bool = False,
        max_core_distance: float | None = None,
    ) -> ClusterResult | ClusterScreeningResult:
        """ """
        # Check that contact space exists
        if self.contactspace is None:
            raise RuntimeError("Trying to use maps.cluster() without contact space")
        # Check that contact space maps have been generated
        if self.data is None:
            raise RuntimeError("No contact space data available.")
        # Select the features for clustering
        if features is None:
            self.cluster_features = self.features
        else:
            self.cluster_features = features
        self.cluster_method = normalize_cluster_method(method)
        core_mask = self._core_selection_mask(
            core_only=core_only,
            max_core_distance=max_core_distance,
        )
        X = self.data.loc[core_mask, self.cluster_features].values.astype(np.float64)
        if len(X) == 0:
            raise ValueError("No points available after applying the core filter for clustering.")
        selected_graph = (
            self._subset_graph_result(graph, core_mask)
            if graph is not None and not np.all(core_mask)
            else graph
        )
        if graph is not None and graph.matrix.shape[0] != len(self.data):
            raise ValueError(
                f"graph has {graph.matrix.shape[0]} nodes, expected {len(self.data)} to match maps.data."
            )
        if selected_graph is not None and selected_graph.matrix.shape[0] != len(X):
            raise ValueError(
                f"graph has {selected_graph.matrix.shape[0]} nodes, expected {len(X)} to match the selected clustering points."
            )
        if nclusters is not None:
            if not clustering_uses_random_state(self.cluster_method):
                self.random_state = 0
            elif random_state is None:
                # If we performed a screening, use the best random state
                if (
                    self.best_clusters is not None
                    and self.cluster_screening_method == self.cluster_method
                    and clustering_uses_random_state(self.cluster_method)
                ):
                    if nclusters in self.best_clusters["nclusters"].values:
                        self.random_state = self.best_clusters[
                            self.best_clusters["nclusters"] == nclusters
                        ]["random_state"].values[0]
                        logger.info(f"Use best random state = {self.random_state} from screening")
                    else:
                        self.random_state = np.random.randint(0, 1000)
                        logger.info(f"Use new random state = {self.random_state}")
                # Otherwise, pick a random number
                else:
                    self.random_state = np.random.randint(0, 1000)
                    logger.info(f"Use new random state = {self.random_state}")
            else:
                self.random_state = random_state
                logger.info(f"Use given random state = {self.random_state}")
            effective_random_state = (
                int(self.random_state)
                if clustering_uses_random_state(self.cluster_method)
                else None
            )
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
                random_state=effective_random_state,
                scale=scale,
                screening=screening_result,
                graph=selected_graph,
            )
            full_labels = np.full(len(self.data), -1, dtype=np.int64)
            full_labels[core_mask] = result.labels
            self.data["Cluster"] = full_labels
            # Store number of clusters
            self.nclusters = nclusters
            self.cluster_result = result
            self.cluster_result.labels = full_labels
            # Compute the cluster centers in features space
            self.cluster_centers = result.centers
            # Compute the number of points in each cluster
            self.cluster_sizes = result.sizes
            # Generate clusters connectivity matrix
            if self.contactspace is None:
                raise RuntimeError("Trying to use maps.cluster() without contact space")
            self.cluster_graph = aggregate_cluster_graph(full_labels, self.contactspace.neighbors)
            self.cluster_edges = self.cluster_graph.copy()
            for i in range(nclusters):
                self.cluster_edges[i, i] = 0
            result.graph = self.cluster_graph
            result.edges = self.cluster_edges
            result.metadata = {
                **(result.metadata or {}),
                "core_only": core_only,
                "max_core_distance": max_core_distance,
                "n_selected_points": int(np.count_nonzero(core_mask)),
            }
            return result
        else:
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
        normalize_node_weights: bool = True,
        use_node_weights_in_edges: bool = True,
        direction_columns: tuple[str, str, str] | None = None,
        directional_weight: float = 0.0,
        directional_power: float = 1.0,
    ) -> GraphResult:
        if self.contactspace is None:
            raise RuntimeError("Trying to use maps.build_graph() without contact space")
        if self.data is None:
            raise RuntimeError("No contact space data available.")

        selected_features = (
            list(feature_columns) if feature_columns is not None else list(self.features)
        )
        node_table = self.data.loc[:, ["x", "y", "z", *selected_features]].copy()
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
            if column in self.data.columns:
                node_table.loc[:, column] = self.data[column].to_numpy()
            elif column in self.contactspace.data.columns:
                node_table.loc[:, column] = self.contactspace.data[column].to_numpy()
            else:
                raise ValueError(
                    f"required graph column {column!r} not present in maps.data or contactspace.data."
                )

        node_table.insert(0, "point_index", self.data.index.to_numpy(dtype=np.int64))
        result = build_point_graph(
            node_table,
            mode=mode,
            feature_columns=selected_features,
            neighbors=self.contactspace.neighbors,
            node_weight_column=node_weight_column,
            feature_k=feature_k,
            feature_connectivity=feature_connectivity,
            sigma_feature=sigma_feature,
            realspace_weight=realspace_weight,
            feature_weight=feature_weight,
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
        feature_columns: list[str] | None = None,
        graph: GraphResult | None = None,
        probability_column: str = "probability",
        region: int | None = None,
        core_only: bool = False,
        max_core_distance: float | None = None,
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
        if self.data is None:
            raise RuntimeError("No contact space data available.")

        resolved_feature_columns = (
            list(feature_columns) if feature_columns is not None else list(self.features)
        )
        point_table = self.data.copy()
        point_table.loc[:, "point_index"] = point_table.index.to_numpy(dtype=np.int64)
        if probability_column not in point_table.columns:
            if (
                self.contactspace is None
                or probability_column not in self.contactspace.data.columns
            ):
                raise ValueError(
                    f"probability_column {probability_column!r} not present in maps.data "
                    "or contactspace.data."
                )
            point_table.loc[:, probability_column] = self.contactspace.data[
                probability_column
            ].to_numpy()

        candidate_mask: npt.NDArray[np.bool_] | None = None
        if region is not None or core_only or max_core_distance is not None:
            if self.contactspace is None:
                raise RuntimeError(
                    "Trying to filter archetypes by region or core_distance without contact space"
                )
            candidate_mask = np.ones(len(point_table), dtype=bool)
            if region is not None:
                region_values = self.contactspace.data["region"].to_numpy(dtype=np.int64)
                candidate_mask &= region_values == region
            candidate_mask &= self._core_selection_mask(
                core_only=core_only,
                max_core_distance=max_core_distance,
            )
            if (
                (core_only or max_core_distance is not None)
                and min_probability is None
                and min_probability_quantile == 0.75
            ):
                min_probability_quantile = None

        selected_graph = graph if graph is not None else self.graph_result
        if selection_mode == "graph_endpoint" and selected_graph is None:
            raise RuntimeError(
                "selection_mode='graph_endpoint' requires a graph. "
                "Call maps.build_graph(...) first or pass graph explicitly."
            )

        result = select_feature_archetypes(
            point_table,
            n_archetypes=n_archetypes,
            feature_columns=resolved_feature_columns,
            probability_column=probability_column,
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
            archetype_table = result.archetype_table.sort_values("selection_rank").reset_index(
                drop=True
            )
            metadata: dict[str, Any] = {
                "selection_rank": archetype_table["selection_rank"].to_numpy(dtype=np.int64),
                "selection_score": archetype_table["selection_score"].to_numpy(dtype=np.float64),
                "probability_score": archetype_table["probability_score"].to_numpy(
                    dtype=np.float64
                ),
                "extremeness_score": archetype_table["extremeness_score"].to_numpy(
                    dtype=np.float64
                ),
                "diversity_score": archetype_table["diversity_score"].to_numpy(dtype=np.float64),
            }
            for column in [
                "euclidean_extremeness_score",
                "endpoint_score",
                "endpointness_score",
                "geodesic_score",
                "branching_score",
            ]:
                if column in archetype_table.columns:
                    metadata[column] = archetype_table[column].to_numpy(dtype=np.float64)
            self.add_special_points(
                result.selected_indexes,
                kind=kind,
                iteration=iteration,
                label_status=label_status,
                replace_kind=replace_kind,
                **metadata,
            )

        return result

    def propagate_archetypes(
        self,
        *,
        graph: GraphResult | None = None,
        selected_indexes: npt.ArrayLike | None = None,
        kind: str = "archetype",
        propagation_mode: ArchetypePropagationMode = "diffusion",
        alpha: float = 0.9,
        max_iter: int = 500,
        tol: float = 1.0e-8,
        confidence_threshold: float = 0.5,
        margin_threshold: float = 0.0,
        propagation_realspace_scale: float = 1.0,
        propagation_feature_scale: float = 1.0,
        propagation_use_node_weights: bool = False,
        update_data: bool = True,
    ) -> ArchetypePropagationResult:
        if self.data is None:
            raise RuntimeError("No contact space data available.")

        if selected_indexes is None:
            selected = self.special_points.indexes(kind=kind)
        else:
            selected = np.asarray(selected_indexes, dtype=np.int64).reshape(-1)
        if selected.size == 0:
            raise RuntimeError(
                "No archetype seeds available. Call maps.select_archetypes(...) first "
                "or pass selected_indexes explicitly."
            )

        selected_graph = graph if graph is not None else self.graph_result
        if selected_graph is None:
            raise RuntimeError(
                "No graph available for propagation. Call maps.build_graph(...) first "
                "or pass graph explicitly."
            )

        result = propagate_archetypes_on_graph(
            selected_graph,
            selected_indexes=selected,
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
            assignment = result.assignment_table.set_index("point_index")
            columns = [
                "assigned_archetype_rank",
                "assigned_archetype_index",
                "archetype_confidence",
                "archetype_margin",
                "is_ambiguous",
            ]
            self.data.loc[assignment.index, columns] = assignment.loc[:, columns]

        valid_assignments = result.assigned_archetype_indexes[
            result.assigned_archetype_indexes >= 0
        ]
        assignment_counts: dict[int, int] = {
            int(point_index): int(np.count_nonzero(valid_assignments == point_index))
            for point_index in selected
        }
        mean_confidences: dict[int, float] = {}
        assignment_table = result.assignment_table
        for point_index in selected:
            point_mask = assignment_table["assigned_archetype_index"].to_numpy(
                dtype=np.int64
            ) == int(point_index)
            if np.any(point_mask):
                mean_confidences[int(point_index)] = float(
                    assignment_table.loc[point_mask, "archetype_confidence"].mean()
                )
            else:
                mean_confidences[int(point_index)] = 0.0

        self.update_special_points(
            kind=kind,
            point_indexes=selected,
            assigned_point_count=np.array(
                [assignment_counts[int(point_index)] for point_index in selected],
                dtype=np.int64,
            ),
            mean_assignment_confidence=np.array(
                [mean_confidences[int(point_index)] for point_index in selected],
                dtype=np.float64,
            ),
        )

        return result

    def sites(self, region: int = 0) -> None:
        # Check that contact space exists
        if self.contactspace is None:
            raise RuntimeError("Trying to use maps.cluster() without contact space")
        # Check that contact space maps have been generated
        if self.data is None:
            raise RuntimeError("No contact space data available.")
        # Filter data by region
        filterdata = self.data[self.contactspace.data["region"] == region]
        # Check that the contact space maps have been generated
        if self.data is None:
            raise RuntimeError("No contact space data available.")
        # Check that the clusters have been generated
        if (
            self.cluster_centers is None
            or self.cluster_graph is None
            or self.cluster_edges is None
            or self.cluster_sizes is None
        ):
            raise RuntimeError("No clusters have been generated.")
        # Compute the cluster centroids
        centroid_indexes = []
        cluster_sizes = np.unique(filterdata["Cluster"].values)
        for cluster_index in cluster_sizes:
            # Select points that belong to the cluster and save their global indexes
            cluster_points = filterdata[filterdata["Cluster"] == cluster_index][
                self.cluster_features
            ].values.astype(np.float64)
            cluster_indexes = filterdata[filterdata["Cluster"] == cluster_index].index.values
            # Compute the distance between the cluster points and the cluster centers
            dist = distance.cdist(cluster_points, self.cluster_centers)
            # Only consider the clusters that are connected to the current one
            indexes = np.where(self.cluster_edges[cluster_index, :] != 0)[0]
            # Minimize the sum of the inverse distances to the connected clusters
            full_dist = np.sum(self.cluster_sizes[indexes] / dist[:, indexes] ** 2, axis=1)
            cluster_centroid = np.argmin(full_dist)
            # Save the global index of the cluster centroid
            centroid_indexes.append(cluster_indexes[cluster_centroid])
        # Convert the list of indexes to a numpy array
        self.add_special_points(
            np.array(centroid_indexes, dtype=np.int64),
            kind="centroid",
            iteration=0,
            label_status="unlabeled",
            replace_kind=True,
        )
        return None

    def select_points(
        self,
        npoints: int,
        feature_columns: list[str] | None = None,
        energy_column: str | None = None,
        uncertainty_column: str | None = None,
        special_point_indexes: npt.ArrayLike | None = None,
        centroid_indexes: npt.ArrayLike | None = None,
        region: int | None = None,
        real_space_weight: float = 0.0,
        feature_space_weight: float = 1.0,
        energy_weight: float = 0.0,
        uncertainty_weight: float = 0.0,
        scale_features: bool = True,
        energy_selection_mode: str = "global_minimum",
        gradient_columns: list[str] | None = None,
        gradient_norm_column: str | None = None,
        curvature_columns: list[str] | None = None,
        stationary_orders: int | Sequence[int] = 0,
        gradient_tolerance: float = 1.0e-6,
        curvature_tolerance: float = 1.0e-8,
    ) -> pd.DataFrame:
        """
        Greedily select contact-space points with a weighted balance of diversity and utility.

        The acquisition score combines:
            - real-space distance from existing centroids / already selected points
            - feature-space distance from existing centroids / already selected points
            - low energy or stationary-point preference
            - high uncertainty
        """
        if npoints <= 0:
            raise ValueError("npoints must be positive")
        if self.data is None:
            raise RuntimeError("No contact space data available.")

        weights = np.array(
            [real_space_weight, feature_space_weight, energy_weight, uncertainty_weight],
            dtype=np.float64,
        )
        if np.any(weights < 0.0):
            raise ValueError("Selection weights must be non-negative")
        if not np.any(weights > 0.0):
            raise ValueError("At least one selection weight must be positive")

        frame = self.data
        if region is not None:
            if self.contactspace is None:
                raise RuntimeError("Trying to filter by region without contact space")
            mask = self.contactspace.data["region"] == region
            frame = self.data.loc[mask].copy()
        else:
            frame = self.data.copy()

        if frame.empty:
            raise RuntimeError("No candidate points available for selection")

        seed_source = (
            special_point_indexes if special_point_indexes is not None else centroid_indexes
        )
        if seed_source is None:
            seed_source = self.special_points.reference_indexes()

        if seed_source is None:
            seed_indexes = np.array([], dtype=np.int64)
        else:
            seed_indexes = np.asarray(seed_source, dtype=np.int64).reshape(-1)
            seed_indexes = seed_indexes[np.isin(seed_indexes, frame.index.to_numpy())]

        candidate_frame = frame.drop(index=seed_indexes, errors="ignore").copy()
        if candidate_frame.empty:
            raise RuntimeError("No candidate points remain after excluding seed centroids")
        if npoints > len(candidate_frame):
            raise ValueError(
                f"Requested {npoints} points, but only {len(candidate_frame)} candidates are available."
            )

        feature_columns = self._resolve_selection_feature_columns(
            feature_columns,
            energy_column=energy_column,
            uncertainty_column=uncertainty_column,
            feature_space_weight=feature_space_weight,
        )

        real_candidates = candidate_frame[["x", "y", "z"]].to_numpy(dtype=np.float64)
        real_seeds = frame.loc[seed_indexes, ["x", "y", "z"]].to_numpy(dtype=np.float64)

        if feature_columns:
            full_features = frame.loc[:, feature_columns].to_numpy(dtype=np.float64)
            if scale_features:
                full_features = StandardScaler().fit_transform(full_features)
            feature_frame = pd.DataFrame(full_features, index=frame.index, columns=feature_columns)
            feature_candidates = feature_frame.loc[candidate_frame.index].to_numpy(dtype=np.float64)
            feature_seeds = feature_frame.loc[seed_indexes].to_numpy(dtype=np.float64)
        else:
            feature_candidates = np.zeros((len(candidate_frame), 0), dtype=np.float64)
            feature_seeds = np.zeros((len(seed_indexes), 0), dtype=np.float64)

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
                    "point_index": int(best_index),
                }
            )

            remaining = np.delete(remaining, best_position)

        selection = candidate_frame.loc[[record["point_index"] for record in records]].copy()
        selection.loc[:, "selection_rank"] = [record["selection_rank"] for record in records]
        selection.loc[:, "selection_score"] = [record["selection_score"] for record in records]
        selection.loc[:, "real_space_score"] = [record["real_space_score"] for record in records]
        selection.loc[:, "feature_space_score"] = [
            record["feature_space_score"] for record in records
        ]
        selection.loc[:, "energy_score"] = [record["energy_score"] for record in records]
        selection.loc[:, "uncertainty_score"] = [record["uncertainty_score"] for record in records]
        return selection.sort_values("selection_rank")

    def _process_chunk(self, chunk: NDArrayF) -> list[NDArrayF]:
        """
        Main calculation of symmetry functions over a chunk of points
        Arguments:
            chunk: array containing the coordinates of points on which to compute the symmetry functions

        Returns:
            results: list containing the values of the different symmetry functions on the given points
        """
        # collect per-sf rows for this chunk
        per_sf_rows: list[list[NDArrayF]] = [[] for _ in range(self.nsfs)]
        for position in chunk:
            # If the system has atomic symmetry functions, calculate atomic distances and vectors
            if self.hasatomicsf:
                if self.system is None:
                    raise RuntimeError("System not defined for atomic symmetry functions")
                if self.system.atoms is None:
                    raise RuntimeError(
                        "Atoms not defined in the system for atomic symmetry functions"
                    )
                atvect, atdist = get_distances(
                    position,
                    self.system.atoms.positions,
                    cell=self.system.atoms.cell,
                    pbc=self.system.atoms.pbc,
                )
            # For each symmetry function, calculate values based on distances
            for i, sf in enumerate(self.symmetryfunctions):
                if sf.atomic:
                    if sf.angular:
                        values = np.asarray(sf.values(atdist[0], atvect[0]), dtype=np.float64)
                    else:
                        values = np.asarray(sf.values(atdist[0]), dtype=np.float64)
                    # ensure row shape (1, n_features_for_sf)
                    row = values.reshape(1, -1)
                    per_sf_rows[i].append(row)
                else:
                    raise RuntimeError("Non-atomic symmetry functions not implemented yet")
        # Stack per-sf rows into arrays (#points_in_chunk, #features_for_sf)
        out: list[NDArrayF] = []
        for i, rows in enumerate(per_sf_rows):
            if rows:
                out.append(np.vstack(rows))
            else:
                ncols = len(self.symmetryfunctions[i].keys)
                out.append(np.zeros((0, ncols), dtype=np.float64))
        return out

    def _results2df(self, positions: npt.NDArray, results: list[npt.NDArray]) -> pd.DataFrame:
        """
        Convert computed symmetry functions from list to DataFrame

        Arguments:
            positions: list of positions of points used to compute symmetry functions
            results: list of values of symmetry functions

        Returns:
            data: a combined DataFrame with positions and symmetry functions appropriately labelled
        """
        data = pd.DataFrame(data=positions, columns=["x", "y", "z"])
        for i, sf in enumerate(self.symmetryfunctions):
            values = np.array(results[i])
            if values.ndim == 1:  # If values are 1D, reshape to handle single positions
                values = values.reshape(1, -1)
            for j, key in enumerate(sf.keys):
                y: pd.DataFrame = pd.DataFrame({key: values[:, j]}, dtype="Float64")
                data.loc[:, key] = y
        return data

    def _sync_contactspace_features(self) -> None:
        """Propagate scalar contact-space annotations into Maps data."""
        if self.contactspace is None or self.data is None:
            return

        for column in self.contactspace.feature_columns:
            self.data.loc[:, column] = self.contactspace.data[column].to_numpy()

    def _sync_contactspace_metadata(self) -> None:
        """Propagate canonical contact-space metadata columns into Maps data."""
        if self.contactspace is None or self.data is None:
            return

        for column in self._METADATA_COLUMNS:
            if column in self.contactspace.data.columns:
                self.data.loc[:, column] = self.contactspace.data[column].to_numpy()

    def _refresh_features(self) -> None:
        """Update the list of feature columns exposed by the current dataset."""
        if self.data is None:
            self.features = []
            return

        self.features = [
            column
            for column in self.data.columns
            if column not in {"x", "y", "z", *self._METADATA_COLUMNS}
        ]

    def _core_selection_mask(
        self,
        *,
        core_only: bool = False,
        max_core_distance: float | None = None,
    ) -> npt.NDArray[np.bool_]:
        if self.data is None:
            raise RuntimeError("No contact space data available.")

        mask = np.ones(len(self.data), dtype=bool)
        if not core_only and max_core_distance is None:
            return mask
        if self.contactspace is None:
            raise RuntimeError("Trying to filter by core without contact space")

        if core_only:
            if "is_core" in self.data.columns:
                mask &= self.data["is_core"].to_numpy(dtype=bool)
            elif "is_core" in self.contactspace.data.columns:
                mask &= self.contactspace.data["is_core"].to_numpy(dtype=bool)
            else:
                raise ValueError(
                    "core_only=True requires the is_core annotation. "
                    "Set core_tolerance during contact-space generation or use max_core_distance."
                )

        if max_core_distance is not None:
            core_distance = self.contactspace.data["core_distance"].to_numpy(dtype=np.float64)
            mask &= core_distance <= float(max_core_distance)

        return mask

    @staticmethod
    def _subset_graph_result(
        graph: GraphResult,
        mask: npt.NDArray[np.bool_],
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

        cluster_features = getattr(self, "cluster_features", [])
        if cluster_features:
            return list(cluster_features)

        excluded = {"Cluster"}
        if energy_column is not None:
            excluded.add(energy_column)
        if uncertainty_column is not None:
            excluded.add(uncertainty_column)
        return [column for column in self.features if column not in excluded]

    def _energy_component(
        self,
        frame: pd.DataFrame,
        *,
        energy_values: NDArrayF,
        energy_column: str | None,
        energy_selection_mode: str,
        gradient_columns: list[str] | None,
        gradient_norm_column: str | None,
        curvature_columns: list[str] | None,
        stationary_orders: int | Sequence[int],
        gradient_tolerance: float,
        curvature_tolerance: float,
    ) -> NDArrayF:
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
    ) -> NDArrayF:
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
    ) -> NDArrayF:
        if not curvature_columns:
            raise ValueError("Stationary-point energy selection requires curvature_columns.")
        return frame.loc[:, curvature_columns].to_numpy(dtype=np.float64)

    @staticmethod
    def _normalize_component(values: NDArrayF) -> NDArrayF:
        if values.size == 0:
            return values
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if np.isclose(vmax, vmin):
            return np.zeros_like(values)
        return (values - vmin) / (vmax - vmin)

    @staticmethod
    def _min_distance_to_references(points: NDArrayF, references: NDArrayF) -> NDArrayF:
        if points.size == 0:
            return np.zeros(0, dtype=np.float64)
        if references.size == 0:
            return np.zeros(points.shape[0], dtype=np.float64)
        return np.min(distance.cdist(points, references), axis=1)

    @staticmethod
    def _stack_references(
        seeds: NDArrayF,
        selected: list[NDArrayF],
        *,
        width: int,
    ) -> NDArrayF:
        selected_array = np.vstack(selected) if selected else np.zeros((0, width), dtype=np.float64)
        if seeds.size == 0:
            return selected_array
        if selected_array.size == 0:
            return seeds
        return np.vstack([seeds, selected_array])


# Class that extends the Maps class to load data from a file
class MapsFromFile(Maps):
    def __init__(self, filename: str) -> None:
        from pathlib import Path

        from mapsy.io.parser import ContactSpaceGenerator, SystemParser
        from mapsy.symfunc.parser import SymmetryFunctionsParser

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
        if control is None:
            raise RuntimeError("Control section missing in input file")

        basepath = Path(filename).expanduser().resolve().parent
        # Set debug and verbosity levels from the input file
        self.debug = bool(control.get("debug", False))
        self.verbosity = int(control.get("verbosity", 0))
        # Parse the system from the input file and assign it to the system attribute
        systemmodel = _namespace_system(system)
        self.system = SystemParser(systemmodel, basepath=basepath).parse()
        # Parse the symmetry functions from the input file
        self.symmetryfunctions = SymmetryFunctionsParser(
            _namespace_symmetryfunctions(symmetryfunctions)
        ).parse()
        # Generate the contact space for the system
        self.contactspace = ContactSpaceGenerator(_namespace_contactspace(contactspace)).generate(
            self.system
        )
        # Call the parent class (Maps) constructor to complete initialization
        super().__init__(self.system, self.symmetryfunctions, self.contactspace)


def _namespace_system(system: dict[str, Any]) -> Any:
    from types import SimpleNamespace

    file = system.get("file") or {}
    properties = system.get("properties") or []
    return SimpleNamespace(
        systemtype=system.get("systemtype", system.get("type", "ions")),
        file=SimpleNamespace(
            fileformat=file.get("fileformat", "xyz+"),
            name=file.get("name", ""),
            names=file.get("names", []),
            folder=file.get("folder", ""),
            folders=file.get("folders", []),
            root=file.get("root", ""),
            pattern=file.get("pattern", ""),
            recursive=file.get("recursive", False),
            units=file.get("units", "bohr"),
        ),
        dimension=system.get("dimension", 2),
        axis=system.get("axis", 2),
        properties=[
            SimpleNamespace(
                name=prop.get("name", ""),
                label=prop.get("label", ""),
                file=(
                    SimpleNamespace(
                        fileformat=(prop.get("file") or {}).get("fileformat", "cube"),
                        name=(prop.get("file") or {}).get("name", ""),
                        names=(prop.get("file") or {}).get("names", []),
                        folder=(prop.get("file") or {}).get("folder", ""),
                        folders=(prop.get("file") or {}).get("folders", []),
                        root=(prop.get("file") or {}).get("root", ""),
                        pattern=(prop.get("file") or {}).get("pattern", ""),
                        recursive=(prop.get("file") or {}).get("recursive", False),
                        units=(prop.get("file") or {}).get("units", "bohr"),
                    )
                    if prop.get("file") is not None
                    else None
                ),
            )
            for prop in properties
        ],
    )


def _namespace_contactspace(contactspace: dict[str, Any]) -> Any:
    from types import SimpleNamespace

    allowed_keys = {
        "mode",
        "radiusmode",
        "radii",
        "radiusfile",
        "alpha",
        "spread",
        "distance",
        "cutoff",
        "threshold",
        "side",
        "core_epsilon",
        "core_tolerance",
    }
    unexpected = sorted(set(contactspace) - allowed_keys)
    if unexpected:
        names = ", ".join(repr(name) for name in unexpected)
        raise ValueError(f"Unknown contactspace keys: {names}.")

    return SimpleNamespace(
        mode=contactspace.get("mode", "system"),
        radiusmode=contactspace.get("radiusmode", contactspace.get("radii", "muff")),
        radiusfile=contactspace.get("radiusfile"),
        alpha=contactspace.get("alpha", 1.0),
        spread=contactspace.get("spread", 0.5),
        distance=contactspace.get("distance", 0.0),
        cutoff=contactspace.get("cutoff", 300),
        threshold=contactspace.get("threshold", 0.1),
        side=contactspace.get("side", 1.0),
        core_epsilon=contactspace.get("core_epsilon", 1.0e-12),
        core_tolerance=contactspace.get("core_tolerance"),
    )


def _namespace_symmetryfunctions(symmetryfunctions: dict[str, Any]) -> Any:
    from types import SimpleNamespace

    def normalize_order(value: Any) -> list[int]:
        if isinstance(value, int):
            return list(range(value))
        if isinstance(value, np.integer):
            return list(range(int(value)))
        if isinstance(value, (list, tuple)):
            return [int(item) for item in value]
        array = np.asarray(value, dtype=np.int64)
        if array.ndim == 0:
            return list(range(int(array)))
        return [int(item) for item in array.reshape(-1)]

    functions = []
    for function in symmetryfunctions.get("functions") or []:
        functions.append(
            SimpleNamespace(
                type=function.get("type", "bp"),
                cutoff=function.get("cutoff", "cos"),
                radius=function.get("radius", 5.0),
                order=normalize_order(function.get("order", 1)),
                etas=function.get("etas", [1.0]),
                rss=function.get("rss", [0.0]),
                lambdas=function.get("lambdas", [-1.0, 1.0]),
                kappas=function.get("kappas", [1.0]),
                zetas=function.get("zetas", [1]),
                radial=function.get("radial", True),
                compositional=function.get("compositional", False),
                structural=function.get("structural", False),
            )
        )
    return SimpleNamespace(functions=functions)
