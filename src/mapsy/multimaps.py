from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from .maps import Maps

AxesLike: TypeAlias = Axes | npt.NDArray[np.object_]


class MultiMaps:
    def __init__(
        self,
        maps: Sequence["Maps"],
        names: Sequence[str] | None = None,
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

        self.data: pd.DataFrame | None = None
        self.features: list[str] = []
        self.npca: int | None = None

        self.nclusters: int = 0
        self.cluster_features: list[str] = []
        self.cluster_centers: np.ndarray[Any, np.dtype[np.float64]] | None = None
        self.cluster_graph: np.ndarray[Any, np.dtype[np.int64]] | None = None
        self.cluster_edges: np.ndarray[Any, np.dtype[np.int64]] | None = None
        self.best_clusters: pd.DataFrame | None = None
        self.cluster_screening: pd.DataFrame | None = None

        self._slices: list[slice] = []

    def atcontactspace(self) -> pd.DataFrame:
        return self._build_dataset(recompute=True)

    def reduce(
        self, npca: int | None = None, scale: bool = False
    ) -> tuple[Figure, AxesLike, AxesLike] | None:
        data = self._ensure_data()
        features = data[self.features].values.astype(np.float64)
        if scale:
            features = StandardScaler().fit_transform(features)

        if npca is not None:
            self.npca = npca
            pca = PCA(npca)
            pca_features = [f"pca{i:1d}" for i in range(npca)]
            data.loc[:, pca_features] = pca.fit_transform(features)
            self._propagate_columns(pca_features)
            return None

        pca = PCA()
        pca.fit(features)
        explained_variance = np.cumsum(pca.explained_variance_ratio_)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(range(1, len(explained_variance) + 1), explained_variance, marker="o")
        ax1.set_xlabel("Number of Components")
        ax1.set_ylabel("Cumulative Explained Variance")
        ax1.grid(True)
        ax1.set_title("Optimal Number of Components in PCA")
        ax1.axhline(y=0.98, color="r", linestyle="--")
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

    def cluster(
        self,
        nclusters: int | None = None,
        features: list[str] | None = None,
        maxclusters: int = 20,
        ntries: int = 1,
        random_state: int | None = None,
        scale: bool = False,
    ) -> tuple[Figure, AxesLike, AxesLike] | None:
        data = self._ensure_data()
        self.cluster_features = list(features) if features is not None else list(self.features)
        X = data[self.cluster_features].values.astype(np.float64)
        if scale:
            X = StandardScaler().fit_transform(X)

        if nclusters is not None:
            chosen_random_state = self._select_random_state(nclusters, random_state)
            labels = SpectralClustering(
                n_clusters=nclusters, random_state=chosen_random_state
            ).fit_predict(X)
            data.loc[:, "Cluster"] = labels
            self.nclusters = nclusters
            self.cluster_centers = (
                data.groupby("Cluster")[self.cluster_features].mean().values.astype(np.float64)
            )
            self.cluster_graph = self._aggregate_graph(labels.astype(np.int64))
            self.cluster_edges = self.cluster_graph.copy()
            for i in range(nclusters):
                self.cluster_edges[i, i] = 0
            self._propagate_columns(["Cluster"])
            return None

        cluster_range = range(2, maxclusters)
        cluster_random_states: list[int] = []
        cluster_sizes: list[int] = []
        silhouette_scores: list[float] = []
        db_indexes: list[float] = []

        for candidate_nclusters in cluster_range:
            random_states = np.random.randint(0, 1000, ntries)
            for candidate_random_state in random_states:
                cluster_random_states.append(int(candidate_random_state))
                labels = SpectralClustering(
                    n_clusters=candidate_nclusters, random_state=int(candidate_random_state)
                ).fit_predict(X)
                actual_nclusters = len(np.unique(labels))
                cluster_sizes.append(actual_nclusters)
                silhouette_scores.append(silhouette_score(X, labels))
                db_indexes.append(davies_bouldin_score(X, labels))

        self.cluster_screening = pd.DataFrame(
            {
                "nclusters": cluster_sizes,
                "random_state": cluster_random_states,
                "silhouette_score": silhouette_scores,
                "db_index": db_indexes,
            }
        )
        best_db = self.cluster_screening.loc[
            self.cluster_screening.groupby("nclusters")["db_index"].idxmin()
        ]
        best_sil = self.cluster_screening.loc[
            self.cluster_screening.groupby("nclusters")["silhouette_score"].idxmax()
        ]
        self.best_clusters = best_db

        fig, ax1 = plt.subplots()
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
        ax1.set_title("Silhouette Score and Davies-Bouldin Index vs. Number of Clusters")
        ax1.grid(True)
        return fig, ax1, ax2

    def _build_dataset(self, recompute: bool) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        reference_features: list[str] | None = None
        self._slices = []

        start = 0
        for index, (name, maps) in enumerate(zip(self.names, self.maps, strict=False)):
            frame = self._get_map_frame(maps, recompute).copy()
            current_features = self._get_map_features(maps)
            if reference_features is None:
                reference_features = current_features
            elif set(current_features) != set(reference_features):
                raise ValueError("All maps in MultiMaps must expose the same feature columns")

            ordered = frame.loc[:, ["x", "y", "z", *reference_features]].copy()
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

    def _get_map_frame(self, maps: "Maps", recompute: bool) -> pd.DataFrame:
        if recompute or maps.data is None:
            if maps.contactspace is None:
                raise RuntimeError("Each Maps instance must define a contact space")
            return maps.atcontactspace()
        return maps.data

    def _get_map_features(self, maps: "Maps") -> list[str]:
        if maps.features:
            return list(maps.features)
        if maps.data is None:
            raise RuntimeError("No maps data available.")
        return [column for column in maps.data.columns if column not in {"x", "y", "z"}]

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
            local_graph = maps.graph(local_labels)
            nrows, ncols = local_graph.shape
            graph[:nrows, :ncols] += local_graph
        return graph

    def _select_random_state(self, nclusters: int, random_state: int | None) -> int:
        if random_state is not None:
            return random_state
        if self.best_clusters is not None and nclusters in self.best_clusters["nclusters"].values:
            return int(
                self.best_clusters[self.best_clusters["nclusters"] == nclusters][
                    "random_state"
                ].values[0]
            )
        return int(np.random.randint(0, 1000))
