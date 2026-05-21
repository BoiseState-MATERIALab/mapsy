import numpy as np
import numpy.typing as npt
import pandas as pd

from mapsy.boundaries import Boundary
from mapsy.data import GradientField, Grid, ScalarField


class ContactSpace:
    nn = np.zeros((6, 3), dtype=int)
    nn[0], nn[2], nn[4] = np.eye(3, dtype=int)
    nn[1], nn[3], nn[5] = -np.eye(3, dtype=int)
    # Stop auto-splitting when the next layer explains less than this fraction
    # of the original weighted distance variance.
    _AUTO_LAYER_MIN_EXPLAINED_VARIANCE_GAIN = 0.05

    def __init__(
        self,
        boundary: Boundary,
        tol: float = 0.1,
        selection_epsilon: float = 0.0001,
        *,
        core_epsilon: float = 1.0e-12,
        layer_distance_tolerance: float | None = None,
        layer_gradient_cosine_min: float = 0.94,
        layer_tangent_tolerance: float = 0.35,
        n_layers: int | str = "auto",
        layer_min_patch_size: int = 2,
    ) -> None:
        """"""
        self.boundary: Boundary | None = boundary
        self.grid: Grid | None = boundary.grid
        self.mask: npt.NDArray[np.bool_] | None = None
        self.m2i: npt.NDArray[np.int64] | None = None
        self.i2m: npt.NDArray[np.int64] | None = None

        if selection_epsilon < 0.0:
            raise ValueError(f"selection_epsilon must be non-negative, got {selection_epsilon}.")
        if core_epsilon <= 0.0:
            raise ValueError(f"core_epsilon must be positive, got {core_epsilon}.")

        distance_metric = self._distance_to_interface(
            boundary,
            denominator_epsilon=core_epsilon,
        )

        if tol < 0.0:
            # Negative tolerances keep only the points closest to the interface center.
            tol = float(np.min(distance_metric) + selection_epsilon)

        if layer_distance_tolerance is not None and layer_distance_tolerance <= 0.0:
            raise ValueError(
                "layer_distance_tolerance must be positive when provided, "
                f"got {layer_distance_tolerance}."
            )
        if not -1.0 <= layer_gradient_cosine_min <= 1.0:
            raise ValueError(
                "layer_gradient_cosine_min must be between -1 and 1, "
                f"got {layer_gradient_cosine_min}."
            )
        if not 0.0 <= layer_tangent_tolerance <= 1.0:
            raise ValueError(
                "layer_tangent_tolerance must be between 0 and 1, "
                f"got {layer_tangent_tolerance}."
            )
        if isinstance(n_layers, str):
            if n_layers != "auto":
                raise ValueError("n_layers must be a positive integer or 'auto'.")
        elif not isinstance(n_layers, (int, np.integer)) or int(n_layers) < 1:
            raise ValueError(f"n_layers must be a positive integer or 'auto', got {n_layers}.")
        if layer_min_patch_size < 1:
            raise ValueError(
                f"layer_min_patch_size must be at least 1, got {layer_min_patch_size}."
            )

        self.mask = distance_metric <= tol
        if not np.any(self.mask):
            min_distance = float(np.min(distance_metric))
            self.mask = distance_metric <= (min_distance + selection_epsilon)
        self.norm: float = float(np.sum(boundary.gradient.modulus[self.mask]))
        self._annotation_columns: list[str] = []
        self._feature_columns: list[str] = []

        self._get_indexes()

        self._get_neighbors()

        self._get_regions()
        self._get_distance_annotations(core_epsilon=core_epsilon)
        self._get_layer_annotations(
            layer_distance_tolerance=layer_distance_tolerance,
            layer_gradient_cosine_min=layer_gradient_cosine_min,
            layer_tangent_tolerance=layer_tangent_tolerance,
            n_layers=n_layers,
            layer_min_patch_size=layer_min_patch_size,
        )

        data = {
            "probability": boundary.gradient.modulus[self.mask],
            "x": self.grid.coordinates[0, self.mask],
            "y": self.grid.coordinates[1, self.mask],
            "z": self.grid.coordinates[2, self.mask],
            "nn": self.neighbors,
            "region": self.regions,
            "signed_distance": self.signed_distance,
            "core_distance": self.core_distance,
            "patch": self.patches,
            "layer": self.layers,
            "patch_size": self.patch_sizes,
            "layer_size": self.layer_sizes,
            "patch_mean_distance": self.patch_mean_distances,
            "layer_mean_distance": self.layer_mean_distances,
        }
        boundary_columns = self._extract_boundary_columns()
        data.update(boundary_columns)
        self.data = pd.DataFrame(data)
        self._annotation_columns.extend(
            [
                "signed_distance",
                "core_distance",
                "patch",
                "layer",
                "patch_size",
                "layer_size",
                "patch_mean_distance",
                "layer_mean_distance",
            ]
        )
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
        reserved = {
            "x",
            "y",
            "z",
            "nn",
            "region",
            "probability",
            "signed_distance",
            "core_distance",
            "patch",
            "layer",
            "patch_size",
            "layer_size",
            "patch_mean_distance",
            "layer_mean_distance",
        }
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
        if self.nm == 0:
            self.regions = np.zeros(0, dtype=np.int64)
            self.nregions = 0
            return
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

    def _get_distance_annotations(
        self,
        *,
        core_epsilon: float,
    ) -> None:
        if self.boundary is None or self.mask is None:
            raise RuntimeError("ContactSpace boundary data is not available.")
        switch_values = np.asarray(self.boundary.switch[self.mask], dtype=np.float64).reshape(-1)
        probability = np.asarray(
            self.boundary.gradient.modulus[self.mask], dtype=np.float64
        ).reshape(-1)
        self.signed_distance = (switch_values - 0.5) / np.maximum(probability, core_epsilon)
        self.core_distance = np.abs(switch_values - 0.5) / np.maximum(probability, core_epsilon)

    def _get_layer_annotations(
        self,
        *,
        layer_distance_tolerance: float | None,
        layer_gradient_cosine_min: float,
        layer_tangent_tolerance: float,
        n_layers: int | str,
        layer_min_patch_size: int,
    ) -> None:
        if (
            self.boundary is None
            or self.grid is None
            or self.mask is None
            or self.m2i is None
            or self.i2m is None
        ):
            raise RuntimeError("ContactSpace layer metadata is not available.")

        if self.nm == 0:
            self.patches = np.zeros(0, dtype=np.int64)
            self.layers = np.zeros(0, dtype=np.int64)
            self.patch_sizes = np.zeros(0, dtype=np.int64)
            self.layer_sizes = np.zeros(0, dtype=np.int64)
            self.patch_mean_distances = np.zeros(0, dtype=np.float64)
            self.layer_mean_distances = np.zeros(0, dtype=np.float64)
            return

        if layer_distance_tolerance is None:
            layer_distance_tolerance = self._default_layer_distance_tolerance()

        gradients = np.stack(
            [
                np.asarray(self.boundary.gradient[0, self.mask], dtype=np.float64).reshape(-1),
                np.asarray(self.boundary.gradient[1, self.mask], dtype=np.float64).reshape(-1),
                np.asarray(self.boundary.gradient[2, self.mask], dtype=np.float64).reshape(-1),
            ],
            axis=1,
        )
        probability = np.asarray(
            self.boundary.gradient.modulus[self.mask], dtype=np.float64
        ).reshape(-1)
        normals = np.zeros_like(gradients)
        valid_normals = probability > 1.0e-12
        normals[valid_normals] = gradients[valid_normals] / probability[valid_normals, None]

        adjacency: list[list[int]] = [[] for _ in range(self.nm)]
        for point_index in range(self.nm):
            if not valid_normals[point_index]:
                continue
            for neighbor_index in self.neighbors[point_index]:
                if neighbor_index < 0 or neighbor_index <= point_index:
                    continue
                if int(self.regions[neighbor_index]) != int(self.regions[point_index]):
                    continue
                if not valid_normals[neighbor_index]:
                    continue
                if (
                    abs(self.signed_distance[point_index] - self.signed_distance[neighbor_index])
                    > layer_distance_tolerance
                ):
                    continue
                if (
                    np.dot(normals[point_index], normals[neighbor_index])
                    < layer_gradient_cosine_min
                ):
                    continue
                displacement = self._neighbor_displacement(point_index, int(neighbor_index))
                displacement_norm = float(np.linalg.norm(displacement))
                if displacement_norm <= 1.0e-12:
                    continue
                average_normal = normals[point_index] + normals[neighbor_index]
                average_normal_norm = float(np.linalg.norm(average_normal))
                if average_normal_norm <= 1.0e-12:
                    average_normal = normals[point_index]
                    average_normal_norm = float(np.linalg.norm(average_normal))
                    if average_normal_norm <= 1.0e-12:
                        continue
                average_normal = average_normal / average_normal_norm
                normal_fraction = (
                    abs(float(np.dot(displacement, average_normal))) / displacement_norm
                )
                if normal_fraction > layer_tangent_tolerance:
                    continue
                adjacency[point_index].append(int(neighbor_index))
                adjacency[int(neighbor_index)].append(point_index)

        self.patches = self._connected_components(adjacency)
        patch_count = int(self.patches.max()) + 1 if self.patches.size else 0
        patch_sizes_by_id = np.bincount(self.patches, minlength=patch_count).astype(np.int64)
        patch_mean_distances_by_id = np.zeros(patch_count, dtype=np.float64)
        patch_mean_core_distances_by_id = np.zeros(patch_count, dtype=np.float64)
        for patch_id in range(patch_count):
            patch_mask = self.patches == patch_id
            patch_mean_distances_by_id[patch_id] = float(np.mean(self.signed_distance[patch_mask]))
            patch_mean_core_distances_by_id[patch_id] = float(
                np.mean(self.core_distance[patch_mask])
            )

        major_patch_ids = np.flatnonzero(patch_sizes_by_id >= layer_min_patch_size)
        if major_patch_ids.size == 0:
            major_patch_ids = np.arange(patch_count, dtype=np.int64)

        layer_by_patch, layer_mean_distances_by_patch = self._assign_layers_from_patch_clusters(
            patch_sizes_by_id=patch_sizes_by_id,
            patch_mean_distances_by_id=patch_mean_distances_by_id,
            major_patch_ids=major_patch_ids,
            n_layers=n_layers,
        )

        layer_count = (
            int(np.max(layer_by_patch[major_patch_ids])) + 1 if major_patch_ids.size else 0
        )
        layer_sizes_by_id = np.zeros(layer_count, dtype=np.int64)
        layer_mean_core_distances_by_id = np.full(layer_count, np.nan, dtype=np.float64)
        layer_mean_signed_distances_by_id = np.full(layer_count, np.nan, dtype=np.float64)
        for layer_id in range(layer_count):
            layer_patch_ids = np.where(layer_by_patch == layer_id)[0]
            layer_sizes_by_id[layer_id] = int(np.sum(patch_sizes_by_id[layer_patch_ids]))
            layer_mean_core_distances_by_id[layer_id] = float(
                np.average(
                    patch_mean_core_distances_by_id[layer_patch_ids],
                    weights=patch_sizes_by_id[layer_patch_ids],
                )
            )
            layer_mean_signed_distances_by_id[layer_id] = float(
                np.average(
                    patch_mean_distances_by_id[layer_patch_ids],
                    weights=patch_sizes_by_id[layer_patch_ids],
                )
            )

        if layer_count > 0:
            layer_order = np.lexsort(
                (
                    layer_mean_signed_distances_by_id,
                    np.round(np.abs(layer_mean_signed_distances_by_id), decimals=12),
                )
            )
            remapped_layer_ids = -np.ones(layer_count, dtype=np.int64)
            remapped_layer_ids[layer_order] = np.arange(layer_count, dtype=np.int64)
            for patch_id in major_patch_ids:
                old_layer_id = layer_by_patch[patch_id]
                if old_layer_id >= 0:
                    layer_by_patch[patch_id] = remapped_layer_ids[old_layer_id]
                    layer_mean_distances_by_patch[patch_id] = layer_mean_signed_distances_by_id[
                        old_layer_id
                    ]
            layer_sizes_by_id = layer_sizes_by_id[layer_order]

        self.layers = np.array(
            [layer_by_patch[patch_id] for patch_id in self.patches], dtype=np.int64
        )
        self.patch_sizes = patch_sizes_by_id[self.patches]
        self.patch_mean_distances = patch_mean_distances_by_id[self.patches]
        self.layer_sizes = np.where(
            self.layers >= 0,
            np.array([layer_sizes_by_id[layer_id] for layer_id in self.layers], dtype=np.int64),
            0,
        )
        self.layer_mean_distances = np.where(
            self.layers >= 0,
            np.array(
                [layer_mean_distances_by_patch[patch_id] for patch_id in self.patches],
                dtype=np.float64,
            ),
            np.nan,
        )

    def _default_layer_distance_tolerance(self) -> float:
        if self.grid is None:
            raise RuntimeError("ContactSpace grid metadata is not available.")
        step_lengths = np.linalg.norm(self.grid.basis, axis=1)
        positive_lengths = step_lengths[step_lengths > 1.0e-12]
        if positive_lengths.size == 0:
            return 0.1
        return 0.75 * float(np.min(positive_lengths))

    @staticmethod
    def _assign_layers_from_patch_clusters(
        *,
        patch_sizes_by_id: npt.NDArray[np.int64],
        patch_mean_distances_by_id: npt.NDArray[np.float64],
        major_patch_ids: npt.NDArray[np.int64],
        n_layers: int | str,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        patch_count = int(patch_sizes_by_id.size)
        layer_by_patch = -np.ones(patch_count, dtype=np.int64)
        layer_mean_distances_by_patch = np.full(patch_count, np.nan, dtype=np.float64)
        if major_patch_ids.size == 0:
            return layer_by_patch, layer_mean_distances_by_patch

        values = patch_mean_distances_by_id[major_patch_ids].astype(np.float64, copy=False)
        weights = patch_sizes_by_id[major_patch_ids].astype(np.float64, copy=False)
        weights = np.where(weights > 0.0, weights, 1.0)

        if isinstance(n_layers, str):
            if n_layers != "auto":
                raise ValueError("n_layers must be a positive integer or 'auto'.")
            labels = ContactSpace._auto_layer_labels(
                values=values,
                weights=weights,
            )
        else:
            count = int(n_layers)
            if count < 1:
                raise ValueError(f"n_layers must be positive, got {n_layers}.")
            if count > values.size:
                raise ValueError(
                    f"n_layers={count} exceeds the number of assignable patches ({values.size})."
                )
            labels, _centers, _sse = ContactSpace._weighted_1d_kmeans(values, weights, count)

        labels = ContactSpace._renumber_labels_by_abs_weighted_mean(labels, values, weights)
        for local_index, patch_id in enumerate(major_patch_ids):
            layer_by_patch[int(patch_id)] = int(labels[local_index])

        for layer_id in np.unique(labels):
            local_mask = labels == layer_id
            mean_distance = float(np.average(values[local_mask], weights=weights[local_mask]))
            for patch_id in major_patch_ids[local_mask]:
                layer_mean_distances_by_patch[int(patch_id)] = mean_distance

        return layer_by_patch, layer_mean_distances_by_patch

    @staticmethod
    def _auto_layer_labels(
        *,
        values: npt.NDArray[np.float64],
        weights: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.int64]:
        candidate_count = min(12, values.size)
        candidate_labels: list[npt.NDArray[np.int64]] = []
        candidate_sse = np.zeros(candidate_count, dtype=np.float64)
        for count in range(1, candidate_count + 1):
            labels, _centers, sse = ContactSpace._weighted_1d_kmeans(
                values,
                weights,
                count,
            )
            candidate_labels.append(labels)
            candidate_sse[count - 1] = sse

        baseline_sse = float(candidate_sse[0])
        if baseline_sse <= np.finfo(np.float64).eps:
            return candidate_labels[0]

        selected_index = 0
        previous_sse = baseline_sse
        for index in range(1, candidate_count):
            explained_variance_gain = (previous_sse - candidate_sse[index]) / baseline_sse
            if explained_variance_gain < ContactSpace._AUTO_LAYER_MIN_EXPLAINED_VARIANCE_GAIN:
                break
            selected_index = index
            previous_sse = float(candidate_sse[index])

        return candidate_labels[selected_index]

    @staticmethod
    def _weighted_1d_kmeans(
        values: npt.NDArray[np.float64],
        weights: npt.NDArray[np.float64],
        nclusters: int,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], float]:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if values.size != weights.size:
            raise ValueError("values and weights must have the same length.")
        if values.size == 0:
            raise ValueError("Cannot cluster an empty array.")
        nclusters = min(max(1, int(nclusters)), values.size)

        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        sorted_weights = np.where(weights[order] > 0.0, weights[order], 1.0)
        npoints = sorted_values.size

        prefix_w = np.concatenate([[0.0], np.cumsum(sorted_weights)])
        prefix_x = np.concatenate([[0.0], np.cumsum(sorted_weights * sorted_values)])
        prefix_x2 = np.concatenate([[0.0], np.cumsum(sorted_weights * sorted_values**2)])

        dp = np.full((nclusters + 1, npoints + 1), np.inf, dtype=np.float64)
        previous = np.full((nclusters + 1, npoints + 1), -1, dtype=np.int64)
        dp[0, 0] = 0.0

        for cluster_count in range(1, nclusters + 1):
            for end in range(cluster_count, npoints + 1):
                starts = np.arange(cluster_count - 1, end, dtype=np.int64)
                costs = dp[cluster_count - 1, starts] + ContactSpace._segment_sse(
                    starts,
                    end,
                    prefix_w=prefix_w,
                    prefix_x=prefix_x,
                    prefix_x2=prefix_x2,
                )
                best = int(np.argmin(costs))
                dp[cluster_count, end] = float(costs[best])
                previous[cluster_count, end] = int(starts[best])

        sorted_labels = np.zeros(npoints, dtype=np.int64)
        centers = np.zeros(nclusters, dtype=np.float64)
        end = npoints
        for cluster_index in range(nclusters - 1, -1, -1):
            start = int(previous[cluster_index + 1, end])
            if start < 0:
                raise RuntimeError("Weighted 1D clustering traceback failed.")
            sorted_labels[start:end] = cluster_index
            weight_sum = prefix_w[end] - prefix_w[start]
            centers[cluster_index] = (prefix_x[end] - prefix_x[start]) / weight_sum
            end = start

        labels = np.zeros(npoints, dtype=np.int64)
        labels[order] = sorted_labels
        return labels, centers, float(dp[nclusters, npoints])

    @staticmethod
    def _segment_sse(
        starts: npt.NDArray[np.int64],
        end: int,
        *,
        prefix_w: npt.NDArray[np.float64],
        prefix_x: npt.NDArray[np.float64],
        prefix_x2: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        weight_sum = prefix_w[end] - prefix_w[starts]
        value_sum = prefix_x[end] - prefix_x[starts]
        value2_sum = prefix_x2[end] - prefix_x2[starts]
        with np.errstate(divide="ignore", invalid="ignore"):
            sse = value2_sum - (value_sum * value_sum) / weight_sum
        return np.maximum(np.where(weight_sum > 0.0, sse, 0.0), 0.0)

    @staticmethod
    def _renumber_labels_by_abs_weighted_mean(
        labels: npt.NDArray[np.int64],
        values: npt.NDArray[np.float64],
        weights: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.int64]:
        order = []
        for label in np.unique(labels):
            mask = labels == label
            mean = float(np.average(values[mask], weights=weights[mask]))
            order.append((round(abs(mean), 12), mean, int(label)))
        order.sort()
        remap = {
            old_label: new_label for new_label, (_abs_mean, _mean, old_label) in enumerate(order)
        }
        return np.array([remap[int(label)] for label in labels], dtype=np.int64)

    def _neighbor_displacement(
        self, point_index: int, neighbor_index: int
    ) -> npt.NDArray[np.float64]:
        if self.grid is None or self.i2m is None:
            raise RuntimeError("ContactSpace neighbor metadata is not available.")
        delta = np.asarray(self.i2m[neighbor_index] - self.i2m[point_index], dtype=np.int64)
        half_scalars = self.grid.scalars // 2
        delta = np.where(delta > half_scalars, delta - self.grid.scalars, delta)
        delta = np.where(delta < -half_scalars, delta + self.grid.scalars, delta)
        return self.grid.basis.T @ delta.astype(np.float64)

    @staticmethod
    def _connected_components(adjacency: list[list[int]]) -> npt.NDArray[np.int64]:
        count = len(adjacency)
        if count == 0:
            return np.zeros(0, dtype=np.int64)
        labels = -np.ones(count, dtype=np.int64)
        component = 0
        for start in range(count):
            if labels[start] >= 0:
                continue
            stack = [start]
            labels[start] = component
            while stack:
                point_index = stack.pop()
                for neighbor_index in adjacency[point_index]:
                    if labels[neighbor_index] >= 0:
                        continue
                    labels[neighbor_index] = component
                    stack.append(neighbor_index)
            component += 1
        return labels

    @staticmethod
    def _distance_to_interface(
        boundary: Boundary,
        *,
        denominator_epsilon: float,
    ) -> npt.NDArray[np.float64]:
        switch_values = np.asarray(boundary.switch[:], dtype=np.float64)
        probability = np.asarray(boundary.gradient.modulus[:], dtype=np.float64)
        distance: npt.NDArray[np.float64] = np.asarray(
            np.abs(switch_values - 0.5) / np.maximum(probability, denominator_epsilon),
            dtype=np.float64,
        )
        return distance

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
