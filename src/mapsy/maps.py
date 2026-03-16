import logging
from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import Any, TypeAlias

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
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from yaml import SafeLoader, load

from mapsy.boundaries import ContactSpace
from mapsy.boundaries.ionic import IonicGeometry
from mapsy.data import ScalarField, System
from mapsy.symfunc.symmetryfunction import SymmetryFunction
from mapsy.utils import full2chunk, multiproc

logger = logging.getLogger(__name__)

AxesLike: TypeAlias = Axes | npt.NDArray[np.object_]  # array of Axes objects
NDArrayF: TypeAlias = npt.NDArray[np.float64]


# Tell mypy we are calling multiproc with a worker that returns a list of ndarrays.
# At runtime this just forwards to mapsy.utils.multiproc.
def _run_multiproc_lists(
    func: Callable[[NDArrayF], list[NDArrayF]],
    args: Sequence[NDArrayF],
) -> list[list[NDArrayF]]:
    return multiproc(func, list(args))


# Class for Maps, which includes functionality for processing symmetry functions
class Maps:
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

    nclusters: int = 0
    cluster_centers: npt.NDArray[np.float64] | None = None
    cluster_graph: npt.NDArray[np.int64] | None = None
    cluster_edges: npt.NDArray[np.int64] | None = None
    best_clusters: pd.DataFrame | None = None
    cluster_screening: pd.DataFrame | None = None

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
        # Call setup for each symmetry function to initialize with the system
        for sf in self.symmetryfunctions:
            sf.setup(self.system.atoms)

    # Method to return the number of symmetry functions
    def len(self) -> int:
        return len(self.symmetryfunctions)

    def atcontactspace(self) -> pd.DataFrame:
        """
        Compute symmetry functions for points in the contact space
        """
        # Calculation symmetry functions on contact points
        if self.contactspace is None:
            raise RuntimeError("Trying to use maps.compute() without contact space")
        positions: npt.NDArray = self.contactspace.data[["x", "y", "z"]].values

        self.data = self.atpoints(positions)
        self._sync_contactspace_features()
        self._refresh_features()

        return self.data

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

    def atpoints(self, positions: npt.NDArray) -> pd.DataFrame:
        """
        Compute symmetry functions for the given positions in parallel
        """

        # Parallelize over the postitions
        args: list[NDArrayF] = [np.asarray(a, dtype=np.float64) for a in full2chunk(positions)]
        # Map chunks -> per-chunk list of arrays (one array per symmetry function)
        chunk_results: list[list[NDArrayF]] = _run_multiproc_lists(self._process_chunk, args)
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
        # Filter data by region
        filterdata = self.data[self.contactspace.data["region"] == region]
        # Check if feature or index is provided and if it is valid
        if feature is not None:
            if feature not in self.data.columns:
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
            if splitby not in self.data.columns:
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
            nf = np.unique(f).size
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
                for i, fvalue in enumerate(np.unique(f)):
                    scatter = ax.scatter(
                        x1m[fm == fvalue],
                        x2m[fm == fvalue],
                        color=colors[i],
                        label=f"{feature} = {i:2}",
                        **kwargs,
                    )
            if centroids:
                if self.centroids is None:
                    raise RuntimeError("No centroids available.")
                pos = self.data.loc[self.centroids, axes].values
                if splitby is not None:
                    posmask = self.data.loc[self.centroids, splitby].values == sval
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

    def reduce(
        self, npca: int | None = None, scale: bool = False
    ) -> tuple[Figure, AxesLike, AxesLike] | None:
        # Check that contact space exists
        if self.contactspace is None:
            raise RuntimeError("Trying to use maps.reduce() without contact space")
        # Check that contact space maps have been generated
        if self.data is None:
            raise RuntimeError("No contact space data available.")
        features = self.data[self.features].values.astype(np.float64)
        if scale:
            features = StandardScaler().fit_transform(features)
        # Initialize PCA
        if npca is not None:
            self.npca = npca
            pca = PCA(npca)
            # Generate labels for the PCA features
            pca_features = [f"pca{i:1d}" for i in range(npca)]
            # Perform PCA on the data
            self.data[pca_features] = pca.fit_transform(features)
            return None
        else:
            pca = PCA()
            pca.fit(features)
            explained_variance = np.cumsum(pca.explained_variance_ratio_)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.plot(range(1, len(explained_variance) + 1), explained_variance, marker="o")
            ax1.set_xlabel("Number of Components")
            ax1.set_ylabel("Cumulative Explained Variance")
            ax1.grid(True)
            ax1.set_title("Optimal Number of Components in PCA")
            ax1.axhline(y=0.98, color="r", linestyle="--")  # Optional: line at 90% variance
            ax2.plot(
                range(1, len(pca.explained_variance_ratio_) + 1),
                pca.explained_variance_ratio_,
                marker="o",
            )
            ax2.set_xlabel("Number of Components")
            ax2.set_ylabel("Explained Variance Ratio")
            ax2.set_title("Scree Plot")
            ax2.grid(True)
            return fig, ax1, ax2

    def graph(self, clusters: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        # Check that contact space exists
        if self.contactspace is None:
            raise RuntimeError("Trying to use maps.graph() without contact space")
        nclusters: np.int64 = np.max(clusters) + 1
        graph: npt.NDArray[np.int64] = np.zeros((nclusters, nclusters), dtype=np.int64)
        for i in range(self.contactspace.nm):
            ci: np.int64 = clusters[i]
            for j in np.delete(
                self.contactspace.neighbors[i],
                np.where(self.contactspace.neighbors[i] < 0),
            ):
                cj: np.int64 = clusters[j]
                graph[ci, cj] += 1
                graph[cj, ci] += 1
        return graph // 2

    def cluster(
        self,
        nclusters: int | None = None,
        features: list[str] | None = None,
        maxclusters: int = 20,
        ntries: int = 1,
        random_state: int | None = None,
        scale: bool = False,
    ) -> tuple[Figure, AxesLike, AxesLike] | None:
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
        X = self.data[self.cluster_features].values.astype(np.float64)
        if scale:
            X = StandardScaler().fit_transform(X)
        if nclusters is not None:
            if random_state is None:
                # If we performed a screening, use the best random state
                if self.best_clusters is not None:
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
            labels = SpectralClustering(
                n_clusters=nclusters, random_state=self.random_state
            ).fit_predict(X)
            self.data["Cluster"] = labels
            # Store number of clusters
            self.nclusters = nclusters
            # Compute the cluster centers in features space
            self.cluster_centers = (
                self.data.groupby("Cluster")[self.cluster_features].mean().values.astype(np.float64)
            )
            # Compute the number of points in each cluster
            self.cluster_sizes = self.data.groupby("Cluster").size().values
            # Generate clusters connectivity matrix
            self.cluster_graph = self.graph(self.data["Cluster"])
            self.cluster_edges = self.cluster_graph.copy()
            for i in range(nclusters):
                self.cluster_edges[i, i] = 0
            return None
        else:
            cluster_range = range(2, maxclusters)
            cluster_random_states = []
            cluster_sizes = []
            silhouette_scores = []
            db_indexes = []
            # Loop over different numbers of clusters
            for nclusters in cluster_range:
                random_states = np.random.randint(0, 1000, ntries)
                for random_state in random_states:
                    cluster_random_states.append(random_state)
                    labels = SpectralClustering(
                        n_clusters=nclusters, random_state=random_state
                    ).fit_predict(X)
                    actual_nclusters = len(np.unique(labels))
                    cluster_sizes.append(actual_nclusters)
                    score = silhouette_score(X, labels)
                    silhouette_scores.append(score)
                    index = davies_bouldin_score(X, labels)
                    db_indexes.append(index)
            # Store all clusters data
            self.cluster_screening = pd.DataFrame(
                {
                    "nclusters": cluster_sizes,
                    "random_state": cluster_random_states,
                    "silhouette_score": silhouette_scores,
                    "db_index": db_indexes,
                }
            )
            # Find the best clusters according to Silhouette Score
            best_db = self.cluster_screening.loc[
                self.cluster_screening.groupby("nclusters")["db_index"].idxmin()
            ]
            best_sil = self.cluster_screening.loc[
                self.cluster_screening.groupby("nclusters")["silhouette_score"].idxmax()
            ]
            self.best_clusters = best_db
            # Plot Silhouette Scores
            fig, ax1 = plt.subplots()
            # Plot Silhouette Scores on the left y-axis
            ax1.scatter(
                cluster_sizes,
                silhouette_scores,
                color="b",
                marker="o",
                label="Silhouette Score",
            )
            ax1.plot(best_db["nclusters"], best_db["silhouette_score"], "-", color="b")
            ax1.plot(best_sil["nclusters"], best_sil["silhouette_score"], ":", color="b")
            ax1.set_xlabel("Number of Clusters")
            ax1.set_ylabel("Silhouette Score", color="b")
            ax1.tick_params(axis="y", labelcolor="b")
            # Create a second y-axis to the right for Davies-Bouldin Index
            ax2 = ax1.twinx()
            ax2.scatter(
                cluster_sizes,
                db_indexes,
                color="r",
                marker="s",
                label="Davies-Bouldin Index",
            )
            ax2.plot(best_db["nclusters"], best_db["db_index"], "-", color="r")
            ax2.plot(best_sil["nclusters"], best_sil["db_index"], ":", color="r")
            ax2.set_ylabel("Davies-Bouldin Index", color="r")
            ax2.tick_params(axis="y", labelcolor="r")
            # Title and grid
            ax1.set_title("Silhouette Score and Davies-Bouldin Index vs. Number of Clusters")
            ax1.grid(True)
            return fig, ax1, ax2

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
        self.centroids = np.array(centroid_indexes, dtype=int)
        return None

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

    def _refresh_features(self) -> None:
        """Update the list of feature columns exposed by the current dataset."""
        if self.data is None:
            self.features = []
            return

        self.features = [column for column in self.data.columns if column not in {"x", "y", "z"}]


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
            folder=file.get("folder", ""),
            root=file.get("root", ""),
            units=file.get("units", "bohr"),
        ),
        dimension=system.get("dimension", 2),
        axis=system.get("axis", 2),
        properties=[
            SimpleNamespace(
                name=prop.get("name", ""),
                label=prop.get("label", ""),
                file=SimpleNamespace(
                    fileformat=(prop.get("file") or {}).get("fileformat", "cube"),
                    name=(prop.get("file") or {}).get("name", ""),
                    folder=(prop.get("file") or {}).get("folder", ""),
                    root=(prop.get("file") or {}).get("root", ""),
                    units=(prop.get("file") or {}).get("units", "bohr"),
                )
                if prop.get("file") is not None
                else None,
            )
            for prop in properties
        ],
    )


def _namespace_contactspace(contactspace: dict[str, Any]) -> Any:
    from types import SimpleNamespace

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
    )


def _namespace_symmetryfunctions(symmetryfunctions: dict[str, Any]) -> Any:
    from types import SimpleNamespace

    functions = []
    for function in symmetryfunctions.get("functions") or []:
        functions.append(
            SimpleNamespace(
                type=function.get("type", "bp"),
                cutoff=function.get("cutoff", "cos"),
                radius=function.get("radius", 5.0),
                order=function.get("order", 1),
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
