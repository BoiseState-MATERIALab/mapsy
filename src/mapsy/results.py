from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse import csr_matrix
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


@dataclass(slots=True)
class GraphResult:
    mode: str
    feature_columns: list[str]
    node_weight_column: str
    node_table: pd.DataFrame
    node_weights: npt.NDArray[np.float64]
    edge_table: pd.DataFrame
    matrix: csr_matrix
    metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class ArchetypeSelectionResult:
    feature_columns: list[str]
    probability_column: str
    candidate_indexes: npt.NDArray[np.int64]
    selected_indexes: npt.NDArray[np.int64]
    candidate_table: pd.DataFrame
    archetype_table: pd.DataFrame
    metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class ArchetypePropagationResult:
    selected_indexes: npt.NDArray[np.int64]
    assigned_archetype_ranks: npt.NDArray[np.int64]
    assigned_archetype_indexes: npt.NDArray[np.int64]
    confidence: npt.NDArray[np.float64]
    margin: npt.NDArray[np.float64]
    soft_assignments: npt.NDArray[np.float64]
    assignment_table: pd.DataFrame
    metadata: dict[str, Any] | None = None
