from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass(slots=True)
class PCAAnalysisResult:
    feature_columns: list[str]
    scale: bool
    estimator: PCA
    scaler: StandardScaler | None
    explained_variance_ratio: npt.NDArray[np.float64]
    cumulative_explained_variance: npt.NDArray[np.float64]
    components: npt.NDArray[np.float64]


@dataclass(slots=True)
class PCAResult:
    feature_columns: list[str]
    transformed_columns: list[str]
    scale: bool
    npca: int
    analysis: PCAAnalysisResult
    transformed_values: npt.NDArray[np.float64]


@dataclass(slots=True)
class ClusterScreeningResult:
    method: str
    feature_columns: list[str]
    scale: bool
    table: pd.DataFrame
    best_by_db: pd.DataFrame
    best_by_silhouette: pd.DataFrame


@dataclass(slots=True)
class ClusterResult:
    method: str
    feature_columns: list[str]
    scale: bool
    nclusters: int
    labels: npt.NDArray[np.int64]
    centers: npt.NDArray[np.float64]
    sizes: npt.NDArray[np.int64]
    random_state: int | None
    graph: npt.NDArray[np.int64] | None = None
    edges: npt.NDArray[np.int64] | None = None
    screening: ClusterScreeningResult | None = None
    metadata: dict[str, Any] | None = None
