import numpy as np
import pandas as pd

from mapsy.boundaries import Boundary


class ContactSpace:

    nn = np.zeros((6, 3), dtype=int)
    nn[0], nn[2], nn[4] = np.eye(3, dtype=int)
    nn[1], nn[3], nn[5] = -np.eye(3, dtype=int)

    def __init__(
        self,
        boundary: Boundary,
        tol: float = 0.1,
        epsilon: float = 0.0001,
    ) -> None:
        """"""
        self.boundary = boundary
        self.grid = boundary.grid

        if tol < 0.0:
            # only select the points with the highest modulus
            tol = np.max(boundary.gradient.modulus) - epsilon

        self.mask = boundary.gradient.modulus > tol
        self.norm = np.sum(boundary.gradient.modulus[self.mask])

        self._get_indexes()

        self._get_neighbors()

        self._get_regions()

        self.data = pd.DataFrame(
            {
                "probability": boundary.gradient.modulus[self.mask],
                "x": self.grid.coordinates[0, self.mask],
                "y": self.grid.coordinates[1, self.mask],
                "z": self.grid.coordinates[2, self.mask],
                "nn": self.neighbors,
                "region": self.regions,
            }
        )

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
        for i, m in enumerate(self.i2m):
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
            stack = [np.where(visited == False)[0][0]]
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
