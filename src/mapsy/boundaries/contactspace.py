from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from mapsy.boundaries import Boundary
from mapsy.data import GradientField, ScalarField


class ContactSpace:
    nn = np.zeros((6, 3), dtype=int)
    nn[0], nn[2], nn[4] = np.eye(3, dtype=int)
    nn[1], nn[3], nn[5] = -np.eye(3, dtype=int)
    layer_nn = np.array(
        [
            [i, j, k]
            for i in (-1, 0, 1)
            for j in (-1, 0, 1)
            for k in (-1, 0, 1)
            if not (i == 0 and j == 0 and k == 0)
        ],
        dtype=int,
    )

    def __init__(
        self,
        boundary: Boundary,
        tol: float = 0.1,
        epsilon: float = 0.0001,
        *,
        assign_layers: bool = False,
        layer_switch_tolerance: float = 0.25,
        layer_gradient_cosine_min: float = 0.9,
        layer_orthogonality_tolerance: float = 0.25,
    ) -> None:
        """"""
        self.boundary = boundary
        self.grid = boundary.grid

        if tol < 0.0:
            # only select the points with the highest modulus
            tol = np.max(boundary.gradient.modulus) - epsilon

        if not -1.0 <= layer_gradient_cosine_min <= 1.0:
            raise ValueError(
                "layer_gradient_cosine_min must lie in [-1, 1], "
                f"got {layer_gradient_cosine_min}."
            )
        if layer_switch_tolerance < 0.0:
            raise ValueError(
                f"layer_switch_tolerance must be non-negative, got {layer_switch_tolerance}."
            )
        if layer_orthogonality_tolerance < 0.0:
            raise ValueError(
                "layer_orthogonality_tolerance must be non-negative, "
                f"got {layer_orthogonality_tolerance}."
            )

        self.mask: npt.NDArray[np.bool_] = boundary.gradient.modulus > tol
        self.norm = np.sum(boundary.gradient.modulus[self.mask])
        self._annotation_columns: list[str] = []
        self._feature_columns: list[str] = []

        self._get_indexes()

        self._get_neighbors()

        self._get_regions()

        if assign_layers:
            self._get_layers(
                switch_tolerance=layer_switch_tolerance,
                gradient_cosine_min=layer_gradient_cosine_min,
                orthogonality_tolerance=layer_orthogonality_tolerance,
            )
        else:
            self.layers = np.zeros(self.nm, dtype=np.int64)
            self.nlayers = 1 if self.nm > 0 else 0

        data = {
            "probability": boundary.gradient.modulus[self.mask],
            "x": self.grid.coordinates[0, self.mask],
            "y": self.grid.coordinates[1, self.mask],
            "z": self.grid.coordinates[2, self.mask],
            "nn": self.neighbors,
            "region": self.regions,
            "layer": self.layers,
        }
        boundary_columns = self._extract_boundary_columns()
        data.update(boundary_columns)
        self.data = pd.DataFrame(data)
        self._annotation_columns.append("layer")
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
        reserved = {"x", "y", "z", "nn", "region", "layer", "probability"}
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

    def _get_indexes(self) -> None:
        """docstring"""
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

    def _get_layers(
        self,
        *,
        switch_tolerance: float,
        gradient_cosine_min: float,
        orthogonality_tolerance: float,
    ) -> None:
        switch_values = np.asarray(self.boundary.switch[self.mask], dtype=np.float64).reshape(-1)
        gradients = np.stack(
            [
                np.asarray(self.boundary.gradient[0, self.mask], dtype=np.float64),
                np.asarray(self.boundary.gradient[1, self.mask], dtype=np.float64),
                np.asarray(self.boundary.gradient[2, self.mask], dtype=np.float64),
            ],
            axis=1,
        )

        layer_neighbors: list[np.ndarray[np.int64]] = []
        for row, neighbor_rows in enumerate(self._get_layer_candidate_neighbors()):
            valid: list[int] = []
            for neighbor in np.asarray(neighbor_rows, dtype=np.int64).reshape(-1):
                if neighbor < 0 or neighbor == row:
                    continue
                if self._points_share_layer(
                    row,
                    int(neighbor),
                    switch_values=switch_values,
                    gradients=gradients,
                    switch_tolerance=switch_tolerance,
                    gradient_cosine_min=gradient_cosine_min,
                    orthogonality_tolerance=orthogonality_tolerance,
                ):
                    valid.append(int(neighbor))
            layer_neighbors.append(np.asarray(valid, dtype=np.int64))

        self.layers = self._connected_components(layer_neighbors)
        self.nlayers = int(np.max(self.layers) + 1) if self.layers.size else 0

    def _get_layer_candidate_neighbors(self) -> list[np.ndarray[np.int64]]:
        layer_candidates: list[np.ndarray[np.int64]] = []
        for m in self.i2m:
            nearest = m[np.newaxis, :] + self.layer_nn
            nearest = nearest - self.grid.scalars * (nearest // self.grid.scalars)
            mapped = self.m2i[tuple(nearest.T)]
            valid = np.asarray(mapped[mapped >= 0], dtype=np.int64)
            if valid.size == 0:
                layer_candidates.append(valid)
                continue
            layer_candidates.append(np.unique(valid))
        return layer_candidates

    def _points_share_layer(
        self,
        row: int,
        neighbor: int,
        *,
        switch_values: npt.NDArray[np.float64],
        gradients: npt.NDArray[np.float64],
        switch_tolerance: float,
        gradient_cosine_min: float,
        orthogonality_tolerance: float,
    ) -> bool:
        if abs(float(switch_values[row] - switch_values[neighbor])) > switch_tolerance:
            return False

        gradient_a = gradients[row]
        gradient_b = gradients[neighbor]
        norm_a = float(np.linalg.norm(gradient_a))
        norm_b = float(np.linalg.norm(gradient_b))
        if norm_a <= 1.0e-12 or norm_b <= 1.0e-12:
            return False

        unit_a = gradient_a / norm_a
        unit_b = gradient_b / norm_b
        if float(np.dot(unit_a, unit_b)) < gradient_cosine_min:
            return False

        mean_normal = unit_a + unit_b
        mean_norm = float(np.linalg.norm(mean_normal))
        mean_normal = unit_a if mean_norm <= 1.0e-12 else mean_normal / mean_norm

        displacement = self._neighbor_displacement(row, neighbor)
        displacement_norm = float(np.linalg.norm(displacement))
        if displacement_norm <= 1.0e-12:
            return False

        orthogonality = abs(float(np.dot(displacement / displacement_norm, mean_normal)))
        return orthogonality <= orthogonality_tolerance

    def _neighbor_displacement(self, row: int, neighbor: int) -> npt.NDArray[np.float64]:
        delta_index = self.i2m[neighbor] - self.i2m[row]
        half_scalars = self.grid.scalars // 2
        delta_index = np.where(
            delta_index > half_scalars, delta_index - self.grid.scalars, delta_index
        )
        delta_index = np.where(
            delta_index < -half_scalars, delta_index + self.grid.scalars, delta_index
        )
        return np.einsum("ij,j->i", self.grid.basis.T, delta_index.astype(np.float64))

    def _connected_components(
        self, neighbors: Sequence[npt.NDArray[np.int64]]
    ) -> npt.NDArray[np.int64]:
        labels = -np.ones(self.nm, dtype=np.int64)
        component = 0
        for start in range(self.nm):
            if labels[start] >= 0:
                continue
            stack = [start]
            labels[start] = component
            while stack:
                row = stack.pop()
                for neighbor in neighbors[row]:
                    neighbor_index = int(neighbor)
                    if labels[neighbor_index] >= 0:
                        continue
                    labels[neighbor_index] = component
                    stack.append(neighbor_index)
            component += 1
        return labels

    def _extract_boundary_columns(self) -> dict[str, npt.NDArray[np.float64]]:
        """Copy pointwise boundary-derived fields onto the sampled contact-space points."""
        columns: dict[str, npt.NDArray[np.float64]] = {}
        for name, value in vars(self.boundary).items():
            if isinstance(value, GradientField):
                columns[f"boundary_{name}_x"] = np.asarray(value[0, self.mask], dtype=np.float64)
                columns[f"boundary_{name}_y"] = np.asarray(value[1, self.mask], dtype=np.float64)
                columns[f"boundary_{name}_z"] = np.asarray(value[2, self.mask], dtype=np.float64)
            elif isinstance(value, ScalarField):
                columns[f"boundary_{name}"] = np.asarray(value[self.mask], dtype=np.float64)
        return columns
