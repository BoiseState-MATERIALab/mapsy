#
import numpy.typing as npt
from typing import Tuple, Optional
from itertools import product
import numpy as np


class Grid:

    def __init__(self, scalars=None, basis=None, cell=None, origin=None):
        """"""
        if scalars is not None:
            self.scalars = np.array(scalars, dtype=np.int64)
        else:
            self.scalars = np.ones(3, dtype=np.int64)
        if basis is not None:
            self.basis = np.array(basis, dtype=np.float64).T
            self.cell = self.basis * self.scalars  # broadcasting
        elif cell is not None:
            self.cell = np.array(cell, dtype=np.float64)
            self.basis = self.cell / self.scalars
        else:
            raise ValueError("Grid needs either basis or cell")
        if origin is not None:
            self.origin = origin
        else:
            self.origin = np.zeros(3, dtype=np.float64)

        self.ndata = np.prod(self.scalars)

        # -- Construct the grid
        mesh: npt.NDArray = np.mgrid[
            0 : self.scalars[0], 0 : self.scalars[1], 0 : self.scalars[2]
        ]
        self.coordinates = (
            np.einsum("ij,jklm->iklm", self.basis.T, mesh)
            + self.origin[:, None, None, None]
        )

        self.corners = -np.array(list(product(range(2), repeat=3))).dot(self.cell)

        self._volume: Optional[np.float64] = None

    def get_min_distance(
        self,
        origin: npt.NDArray,
        dim: int = 0,
        axis: int = 0,
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """docstring"""
        r = self.coordinates - origin[:, np.newaxis, np.newaxis, np.newaxis]
        r, r2 = self._apply_minimum_image_convension(r, dim, axis)
        return r, r2

    def _get_direction(
        self,
        dim: int = 0,
        axis: int = 0,
    ):
        """docstring"""
        if dim == 0:
            n = np.zeros(3)
        elif dim == 1:
            n = self.cell[axis, :]
        elif dim == 2:
            n1, n2 = self.cell[np.arange(3) != axis, :]
            n = np.cross(n2, n1)
        else:
            raise ValueError("dimensions out of range")

        norm = np.linalg.norm(n)
        if norm > 1.0e-16:
            n = n / norm
        return n

    def _reduce_dimension(
        self,
        r: npt.NDArray,
        n: npt.NDArray,
        dim: int = 0,
    ):
        """docstring"""
        if dim == 0:
            pass
        elif dim == 1:
            r = r - np.einsum("jkl,i->ijkl", np.einsum("ijkl,i->jkl", r, n), n)
        elif dim == 2:
            r = np.einsum("jkl,i->ijkl", np.einsum("ijkl,i->jkl", r, n), n)
        else:
            raise ValueError("dimensions out of range")
        return r

    def _apply_minimum_image_convension(
        self,
        r: npt.NDArray,
        dim: int = 0,
        axis: int = 0,
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """docstring"""

        n = self._get_direction(dim, axis)

        # apply minimum image convension
        reciprocal_lattice = self.reciprocal()
        s = np.einsum("lijk,ml->mijk", r, reciprocal_lattice)
        s -= np.floor(s)
        r = np.einsum("lm,lijk->mijk", self.cell, s)
        r = self._reduce_dimension(r, n, dim)

        # pre-corner-check results
        rmin = r
        r2min = np.einsum("i...,i...", r, r)

        t = r
        # check against corner shifts
        for corner in self.corners[1:]:
            r = t + corner[:, np.newaxis, np.newaxis, np.newaxis]
            r = self._reduce_dimension(r, n, dim)
            r2 = np.einsum("i...,i...", r, r)
            mask = r2 < r2min
            rmin = np.where(mask[np.newaxis, :, :, :], r, rmin)
            r2min = np.where(mask, r2, r2min)

        return rmin, r2min

    @property
    def volume(self) -> np.float64:
        if self._volume is None:
            self._compute_volume()
        return self._volume

    def _compute_volume(self):
        a1 = self.cell[:, 0]
        a2 = self.cell[:, 1]
        a3 = self.cell[:, 2]
        a2crossa3 = np.cross(a2, a3)
        self._volume = np.dot(a1, a2crossa3)

    def reciprocal(self):

        a1 = self.cell[:, 0]
        a2 = self.cell[:, 1]
        a3 = self.cell[:, 2]
        a2crossa3 = np.cross(a2, a3)
        volume = np.dot(a1, a2crossa3)

        b1 = a2crossa3 / volume
        b2 = np.cross(a3, a1) / volume
        b3 = np.cross(a1, a2) / volume

        return np.stack([b1, b2, b3], axis=0).T
