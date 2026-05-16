import numpy as np
import numpy.typing as npt
import pandas as pd

from mapsy.boundaries import Boundary
from mapsy.data import GradientField, Grid, ScalarField


class ContactSpace:
    nn = np.zeros((6, 3), dtype=int)
    nn[0], nn[2], nn[4] = np.eye(3, dtype=int)
    nn[1], nn[3], nn[5] = -np.eye(3, dtype=int)

    def __init__(
        self,
        boundary: Boundary,
        tol: float = 0.1,
        epsilon: float = 0.0001,
        *,
        core_epsilon: float = 1.0e-12,
        core_tolerance: float | None = None,
    ) -> None:
        """"""
        self.boundary: Boundary | None = boundary
        self.grid: Grid | None = boundary.grid
        self.mask: npt.NDArray[np.bool_] | None = None
        self.m2i: npt.NDArray[np.int64] | None = None
        self.i2m: npt.NDArray[np.int64] | None = None

        if tol < 0.0:
            # only select the points with the highest modulus
            tol = np.max(boundary.gradient.modulus) - epsilon

        if core_epsilon <= 0.0:
            raise ValueError(f"core_epsilon must be positive, got {core_epsilon}.")
        if core_tolerance is not None and core_tolerance < 0.0:
            raise ValueError(f"core_tolerance must be non-negative, got {core_tolerance}.")

        self.mask = boundary.gradient.modulus > tol
        self.norm = np.sum(boundary.gradient.modulus[self.mask])
        self._annotation_columns: list[str] = []
        self._feature_columns: list[str] = []

        self._get_indexes()

        self._get_neighbors()

        self._get_regions()
        self._get_core_annotations(core_epsilon=core_epsilon, core_tolerance=core_tolerance)

        data = {
            "probability": boundary.gradient.modulus[self.mask],
            "x": self.grid.coordinates[0, self.mask],
            "y": self.grid.coordinates[1, self.mask],
            "z": self.grid.coordinates[2, self.mask],
            "nn": self.neighbors,
            "region": self.regions,
            "core_distance": self.core_distance,
        }
        if self.is_core is not None:
            data["is_core"] = self.is_core
        boundary_columns = self._extract_boundary_columns()
        data.update(boundary_columns)
        self.data = pd.DataFrame(data)
        self._annotation_columns.append("core_distance")
        if self.is_core is not None:
            self._annotation_columns.append("is_core")
        self._annotation_columns.extend(boundary_columns)

    @property
    def annotation_columns(self) -> list[str]:
        return list(self._annotation_columns)

    @property
    def feature_columns(self) -> list[str]:
        return list(self._feature_columns)

    def annotate(
        self,
        name: str,
        values: npt.ArrayLike,
        *,
        as_feature: bool = True,
    ) -> npt.NDArray[np.float64]:
        """Attach pointwise values to the sampled contact-space points."""
        reserved = {"x", "y", "z", "nn", "region", "probability", "core_distance", "is_core"}
        if name in reserved and name not in self._annotation_columns:
            raise ValueError(f"{name!r} is a reserved contact-space column")

        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.size != self.nm:
            raise ValueError(
                f"Annotation {name!r} has length {array.size}, expected {self.nm} contact-space points."
            )

        self.data.loc[:, name] = array
        if name not in self._annotation_columns:
            self._annotation_columns.append(name)

        if as_feature:
            if name not in self._feature_columns:
                self._feature_columns.append(name)
        elif name in self._feature_columns:
            self._feature_columns.remove(name)

        return array

    def release_dense_fields(self) -> None:
        """Drop dense grid fields after their sampled contact-space data has been copied."""
        self.boundary = None
        self.grid = None
        self.mask = None
        self.m2i = None
        self.i2m = None

    def _get_indexes(self) -> None:
        """docstring"""
        if self.grid is None or self.mask is None:
            raise RuntimeError("ContactSpace grid metadata is not available.")
        # -- Number of grid points and number of contact space points
        self.ni = self.grid.ndata
        self.nm = np.count_nonzero(self.mask)
        # -- Find indexes
        i2m = np.zeros(self.nm, dtype=np.int64)
        m2i = -np.ones(self.ni, dtype=np.int64)
        count = 0
        for i in range(self.ni):
            if self.mask.reshape(-1, 1)[i]:
                m2i[i] = count
                i2m[count] = i
                count += 1
        # -- Given position on grid matrix find index of contact space point
        self.m2i = m2i.reshape(self.grid.scalars)
        # -- Given index of contact space point find position on grid matrix
        self.i2m = np.array(np.unravel_index(i2m, self.grid.scalars)).T

    def _get_neighbors(self) -> None:
        """docstring"""
        if self.grid is None or self.m2i is None or self.i2m is None:
            raise RuntimeError("ContactSpace neighbor metadata is not available.")
        self.neighbors = []
        # -- For each contact space point find the indexes of its six neighbors
        for m in self.i2m:
            nearest = m[np.newaxis, :] + self.nn
            # -- Apply periodic boundary conditions
            nearest = nearest - self.grid.scalars * (nearest // self.grid.scalars)
            self.neighbors.append(self.m2i[tuple(nearest.T)])

    def _get_regions(self) -> None:
        """docstring"""
        # -- Depth First Traversal of the graph
        visited = np.zeros(self.nm, dtype=int)
        count = 0
        while True:
            # -- If all nodes have been visited, exit
            if visited.all():
                break
            # -- Update region number
            count += 1  # NOTE: count from 1 to allow boolean operations on visited
            # -- Add first non-visited point to the stack
            stack = [np.where(visited == 0)[0][0]]
            while stack:
                # -- Pop last entry from the stack
                i = stack.pop()
                # -- Set its group to the current region number
                visited[i] = count
                # -- Add its non-visited neighbors to the stack
                for nn in np.delete(self.neighbors[i], np.where(self.neighbors[i] < 0)):
                    if not visited[nn]:
                        stack.append(nn)
        # -- Save regions for each contact space point (counting from 0)
        self.regions = visited - 1
        self.nregions = np.max(visited)

    def _get_core_annotations(
        self,
        *,
        core_epsilon: float,
        core_tolerance: float | None,
    ) -> None:
        if self.boundary is None or self.mask is None:
            raise RuntimeError("ContactSpace boundary data is not available.")
        switch_values = np.asarray(self.boundary.switch[self.mask], dtype=np.float64).reshape(-1)
        probability = np.asarray(
            self.boundary.gradient.modulus[self.mask], dtype=np.float64
        ).reshape(-1)
        self.core_distance = np.abs(switch_values - 0.5) / np.maximum(probability, core_epsilon)
        self.is_core = self.core_distance <= core_tolerance if core_tolerance is not None else None

    def _extract_boundary_columns(self) -> dict[str, npt.NDArray[np.float64]]:
        """Copy pointwise boundary-derived fields onto the sampled contact-space points."""
        if self.boundary is None or self.mask is None:
            raise RuntimeError("ContactSpace boundary data is not available.")
        columns: dict[str, npt.NDArray[np.float64]] = {}
        for name, value in vars(self.boundary).items():
            if isinstance(value, GradientField):
                columns[f"boundary_{name}_x"] = np.asarray(value[0, self.mask], dtype=np.float64)
                columns[f"boundary_{name}_y"] = np.asarray(value[1, self.mask], dtype=np.float64)
                columns[f"boundary_{name}_z"] = np.asarray(value[2, self.mask], dtype=np.float64)
            elif isinstance(value, ScalarField):
                columns[f"boundary_{name}"] = np.asarray(value[self.mask], dtype=np.float64)
        return columns
