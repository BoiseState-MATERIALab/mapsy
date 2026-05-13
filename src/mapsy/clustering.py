from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture

ClusterMethod: TypeAlias = Literal[
    "spectral",
    "gaussian_mixture",
    "kmeans",
    "agglomerative",
]

NDArrayI: TypeAlias = npt.NDArray[np.int64]

_METHOD_ALIASES: dict[str, ClusterMethod] = {
    "spectral": "spectral",
    "gmm": "gaussian_mixture",
    "gaussianmixture": "gaussian_mixture",
    "gaussian_mixture": "gaussian_mixture",
    "kmeans": "kmeans",
    "agglomerative": "agglomerative",
    "hierarchical": "agglomerative",
}


def normalize_cluster_method(method: str) -> ClusterMethod:
    normalized = _METHOD_ALIASES.get(method.lower())
    if normalized is None:
        supported = ", ".join(sorted(_METHOD_ALIASES))
        raise ValueError(
            f"Unsupported clustering method {method!r}. Supported values: {supported}."
        )
    return normalized


def clustering_uses_random_state(method: str) -> bool:
    normalized = normalize_cluster_method(method)
    return normalized in {"spectral", "gaussian_mixture", "kmeans"}


def fit_predict_clusters(
    X: npt.NDArray[np.float64],
    *,
    method: str,
    nclusters: int,
    random_state: int | None = None,
    graph_matrix: csr_matrix | None = None,
) -> NDArrayI:
    normalized = normalize_cluster_method(method)

    if normalized == "spectral":
        if graph_matrix is None:
            labels = SpectralClustering(
                n_clusters=nclusters,
                random_state=random_state,
            ).fit_predict(X)
        else:
            labels = SpectralClustering(
                n_clusters=nclusters,
                random_state=random_state,
                affinity="precomputed",
            ).fit_predict(graph_matrix)
    elif normalized == "gaussian_mixture":
        if graph_matrix is not None:
            raise ValueError("gaussian_mixture clustering does not accept a graph_matrix input.")
        labels = GaussianMixture(
            n_components=nclusters,
            random_state=random_state,
        ).fit_predict(X)
    elif normalized == "kmeans":
        if graph_matrix is not None:
            raise ValueError("kmeans clustering does not accept a graph_matrix input.")
        labels = KMeans(
            n_clusters=nclusters,
            random_state=random_state,
            n_init=10,
        ).fit_predict(X)
    else:
        if graph_matrix is None:
            labels = AgglomerativeClustering(n_clusters=nclusters).fit_predict(X)
        else:
            connectivity = graph_matrix.copy()
            connectivity.data = np.ones_like(connectivity.data, dtype=np.float64)
            labels = AgglomerativeClustering(
                n_clusters=nclusters,
                connectivity=connectivity,
            ).fit_predict(X)

    return labels.astype(np.int64, copy=False)
