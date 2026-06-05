from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from itertools import repeat
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .datasets import SupervisedDataset


@dataclass
class WarmStartProfile:
    """Semantic multi-start profile for order-stable ARD GPR fits."""

    coordinate_names: tuple[str, ...] = ("distance",)
    coordinate_base_scale: float = 0.5
    pc_base_scale: float = 5.0

    def base_scale(self, name: str) -> float:
        if name.startswith("pc"):
            return self.pc_base_scale
        if name in self.coordinate_names:
            return self.coordinate_base_scale
        return 1.0

    def canonicalize_feature_names(self, feature_names: list[str]) -> list[str]:
        pcs = sorted(
            [name for name in feature_names if name.startswith("pc")],
            key=lambda name: int(name[2:]) if name[2:].isdigit() else name,
        )
        coordinates = [name for name in self.coordinate_names if name in feature_names]
        others = [name for name in feature_names if name not in {*pcs, *coordinates}]
        return [*coordinates, *pcs, *sorted(others)]

    def semantic_starts(
        self,
        feature_names: list[str],
        *,
        n_random: int = 12,
        seed: int = 0,
    ) -> list[np.ndarray]:
        rng = np.random.default_rng(seed)
        canonical = self.canonicalize_feature_names(feature_names)

        def vector_from_dict(mapping: dict[str, float]) -> np.ndarray:
            return np.array([mapping[name] for name in feature_names], dtype=float)

        starts: list[np.ndarray] = []
        for coordinate_scale, pca_scale in [
            (0.3, 3.0),
            (0.5, 5.0),
            (1.0, 10.0),
            (0.2, 10.0),
        ]:
            init: dict[str, float] = {}
            for name in canonical:
                if name.startswith("pc"):
                    init[name] = pca_scale
                elif name in self.coordinate_names:
                    init[name] = coordinate_scale
                else:
                    init[name] = 1.0
            starts.append(vector_from_dict(init))

        starts.append(np.ones(len(feature_names), dtype=float))

        for _ in range(n_random):
            init = {
                name: self.base_scale(name) * np.exp(rng.normal(0.0, 0.8)) for name in canonical
            }
            starts.append(vector_from_dict(init))

        return starts


@dataclass
class GaussianProcessFitRecord:
    init_lengthscales: np.ndarray
    cv_rmse_mean: float
    cv_rmse_std: float
    model: Pipeline
    kernel: Any
    log_marginal_likelihood: float


@dataclass
class RobustGaussianProcessSurrogate:
    """Scalar-output GPR surrogate with semantic multi-start fitting."""

    name: str
    feature_names: list[str]
    target_name: str
    role: str | None = None
    warm_start_profile: WarmStartProfile = field(default_factory=WarmStartProfile)
    n_random_starts: int = 12
    seed: int = 0
    n_cv_splits: int = 5
    constant_bounds: tuple[float, float] = (1.0e-3, 1.0e3)
    length_scale_bounds: tuple[float, float] = (1.0e-2, 1.0e2)
    noise_level_bounds: tuple[float, float] = (1.0e-3, 1.0e0)
    initial_noise_level: float = 1.0e-3
    normalize_y: bool = True
    n_jobs: int | None = None
    threadpool_threads: int | None = None

    model_: Pipeline | None = None
    fit_records_: list[GaussianProcessFitRecord] = field(default_factory=list)
    best_record_: GaussianProcessFitRecord | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> RobustGaussianProcessSurrogate:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        starts = self.warm_start_profile.semantic_starts(
            self.feature_names,
            n_random=self.n_random_starts,
            seed=self.seed,
        )

        n_jobs = self._effective_n_jobs(len(starts))
        with self._threadpool_context(n_jobs):
            if n_jobs == 1:
                records = [self._fit_start(init_lengthscales, X, y) for init_lengthscales in starts]
            else:
                with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                    records = list(executor.map(self._fit_start, starts, repeat(X), repeat(y)))

        records.sort(key=lambda record: (record.cv_rmse_mean, record.cv_rmse_std))
        self.fit_records_ = records
        self.best_record_ = records[0]
        self.model_ = self.best_record_.model
        return self

    def fit_dataset(self, dataset: SupervisedDataset) -> RobustGaussianProcessSurrogate:
        self.feature_names = list(dataset.feature_columns)
        self.target_name = dataset.target_column
        return self.fit(dataset.X(), dataset.y())

    def validate_loo_dataset(self, dataset: SupervisedDataset) -> dict[str, Any]:
        return self.validate_loo(dataset.X(), dataset.y())

    def fit_frame(
        self,
        frame: pd.DataFrame,
        *,
        feature_columns: list[str] | None = None,
        target_column: str | None = None,
    ) -> RobustGaussianProcessSurrogate:
        features = list(feature_columns) if feature_columns is not None else self.feature_names
        target = target_column or self.target_name
        return self.fit(
            frame.loc[:, features].to_numpy(dtype=float),
            frame.loc[:, target].to_numpy(dtype=float),
        )

    def predict(
        self,
        X: np.ndarray | pd.DataFrame,
        *,
        return_std: bool = False,
    ) -> Any:
        if self.model_ is None:
            raise RuntimeError("Model has not been fitted.")
        if isinstance(X, pd.DataFrame):
            data = X.loc[:, self.feature_names].to_numpy(dtype=float)
        else:
            data = np.asarray(X, dtype=float)
        return self.model_.predict(data, return_std=return_std)

    def predict_frame(
        self,
        frame: pd.DataFrame,
        *,
        prediction_column: str = "prediction",
        uncertainty_column: str = "uncertainty",
    ) -> pd.DataFrame:
        prediction, std = self.predict(frame, return_std=True)
        out = frame.copy()
        out.loc[:, prediction_column] = np.asarray(prediction, dtype=float)
        out.loc[:, uncertainty_column] = np.asarray(std, dtype=float)
        return out

    def leave_one_out_predictions(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        kernel = self.fitted_kernel()
        predictions = np.zeros_like(y, dtype=float)
        loo = LeaveOneOut()
        for train, test in loo.split(X):
            gp = GaussianProcessRegressor(
                kernel=kernel,
                optimizer=None,
                normalize_y=self.normalize_y,
                random_state=self.seed,
            )
            pipeline = make_pipeline(StandardScaler(), gp)
            pipeline.fit(X[train], y[train])
            predictions[test] = pipeline.predict(X[test])
        return predictions

    def validate_loo(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        predictions = self.leave_one_out_predictions(X, y)
        residuals = predictions - np.asarray(y, dtype=float).reshape(-1)
        return {
            "r2": float(r2_score(y, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y, predictions))),
            "predictions": predictions,
            "residuals": residuals,
        }

    def validate_loo_frame(
        self,
        frame: pd.DataFrame,
        *,
        feature_columns: list[str] | None = None,
        target_column: str | None = None,
    ) -> dict[str, Any]:
        features = list(feature_columns) if feature_columns is not None else self.feature_names
        target = target_column or self.target_name
        return self.validate_loo(
            frame.loc[:, features].to_numpy(dtype=float),
            frame.loc[:, target].to_numpy(dtype=float),
        )

    def fitted_kernel(self) -> Any:
        if self.best_record_ is None:
            raise RuntimeError("Model has not been fitted.")
        return self.best_record_.kernel

    def summary(self) -> dict[str, Any]:
        if self.best_record_ is None:
            raise RuntimeError("Model has not been fitted.")
        return {
            "name": self.name,
            "role": self.role,
            "feature_names": list(self.feature_names),
            "target_name": self.target_name,
            "cv_rmse_mean": self.best_record_.cv_rmse_mean,
            "cv_rmse_std": self.best_record_.cv_rmse_std,
            "kernel": self.best_record_.kernel,
            "log_marginal_likelihood": self.best_record_.log_marginal_likelihood,
        }

    def _make_pipeline(self, init_lengthscales: np.ndarray) -> Pipeline:
        kernel = ConstantKernel(1.0, self.constant_bounds) * RBF(
            length_scale=np.array(init_lengthscales, dtype=float),
            length_scale_bounds=self.length_scale_bounds,
        ) + WhiteKernel(
            noise_level=self.initial_noise_level,
            noise_level_bounds=self.noise_level_bounds,
        )
        return make_pipeline(
            StandardScaler(),
            GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=self.normalize_y,
                optimizer="fmin_l_bfgs_b",
                n_restarts_optimizer=0,
                random_state=self.seed,
            ),
        )

    def _fit_start(
        self,
        init_lengthscales: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
    ) -> GaussianProcessFitRecord:
        model = self._make_pipeline(init_lengthscales)
        mean_rmse, std_rmse = self._cv_rmse(model, X, y)
        model.fit(X, y)
        gpr = model.named_steps["gaussianprocessregressor"]
        return GaussianProcessFitRecord(
            init_lengthscales=np.array(init_lengthscales, dtype=float),
            cv_rmse_mean=mean_rmse,
            cv_rmse_std=std_rmse,
            model=model,
            kernel=gpr.kernel_,
            log_marginal_likelihood=gpr.log_marginal_likelihood(gpr.kernel_.theta),
        )

    def _cv_rmse(self, model: Pipeline, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        cv = KFold(n_splits=self.n_cv_splits, shuffle=True, random_state=self.seed)
        rmses = []
        for train, valid in cv.split(X):
            fitted = clone(model)
            fitted.fit(X[train], y[train])
            prediction = fitted.predict(X[valid])
            rmses.append(float(np.sqrt(mean_squared_error(y[valid], prediction))))
        return float(np.mean(rmses)), float(np.std(rmses))

    def _effective_n_jobs(self, n_tasks: int) -> int:
        if n_tasks <= 0:
            return 1
        if self.n_jobs is None:
            return 1
        requested = int(self.n_jobs)
        if requested == 0:
            raise ValueError("n_jobs must be non-zero.")
        if requested < 0:
            requested = max(1, (os.cpu_count() or 1) + 1 + requested)
        return max(1, min(requested, int(n_tasks)))

    def _threadpool_context(self, n_jobs: int) -> Any:
        threadpool_threads = self.threadpool_threads
        if threadpool_threads is None and n_jobs > 1:
            threadpool_threads = 1
        if threadpool_threads is None or threadpool_threads <= 0:
            return nullcontext()

        from threadpoolctl import threadpool_limits

        return threadpool_limits(limits=int(threadpool_threads))


@dataclass
class ModelSuite:
    """Named registry of fitted surrogate models."""

    models: dict[str, Any] = field(default_factory=dict)
    roles: dict[str, str] = field(default_factory=dict)

    def add(self, model: Any, *, role: str | None = None) -> None:
        self.models[model.name] = model
        assigned_role = role or model.role
        if assigned_role is not None:
            self.roles[assigned_role] = model.name

    def get(self, name: str) -> Any:
        return self.models[name]

    def get_by_role(self, role: str) -> Any:
        if role not in self.roles:
            raise KeyError(f"No model registered for role {role!r}.")
        return self.models[self.roles[role]]
