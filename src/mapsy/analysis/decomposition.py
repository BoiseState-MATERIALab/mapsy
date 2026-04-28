import numpy as np
import numpy.typing as npt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from mapsy.results import PCAAnalysisResult, PCAResult


def fit_pca_analysis(
    X: npt.NDArray[np.float64],
    *,
    feature_columns: list[str],
    scale: bool = False,
) -> PCAAnalysisResult:
    scaler = StandardScaler() if scale else None
    X_scaled = scaler.fit_transform(X) if scaler is not None else X
    estimator = PCA()
    estimator.fit(X_scaled)
    explained_variance_ratio = estimator.explained_variance_ratio_.astype(np.float64, copy=False)
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    return PCAAnalysisResult(
        feature_columns=list(feature_columns),
        scale=scale,
        estimator=estimator,
        scaler=scaler,
        explained_variance_ratio=explained_variance_ratio,
        cumulative_explained_variance=cumulative_explained_variance,
        components=estimator.components_.astype(np.float64, copy=False),
    )


def project_pca(
    analysis: PCAAnalysisResult,
    X: npt.NDArray[np.float64],
    *,
    npca: int,
) -> PCAResult:
    if npca <= 0:
        raise ValueError(f"npca must be positive, got {npca}.")
    ncomponents = int(analysis.explained_variance_ratio.shape[0])
    if npca > ncomponents:
        raise ValueError(f"npca={npca} exceeds the available number of components ({ncomponents}).")

    X_scaled = analysis.scaler.transform(X) if analysis.scaler is not None else X
    transformed_full = analysis.estimator.transform(X_scaled)
    transformed_values = transformed_full[:, :npca].astype(np.float64, copy=False)
    transformed_columns = [f"pca{i:1d}" for i in range(npca)]
    return PCAResult(
        feature_columns=list(analysis.feature_columns),
        transformed_columns=transformed_columns,
        scale=analysis.scale,
        npca=npca,
        analysis=analysis,
        transformed_values=transformed_values,
    )
