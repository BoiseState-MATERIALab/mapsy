from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .grid import Grid
from .scalar import ScalarField
from .volumetric import VolumetricField


class GradientField(VolumetricField):
    """ """

    _modulus: ScalarField

    def __new__(
        cls: type[GradientField],
        grid: Grid,
        rank: int = 3,
        label: str | None = None,
        name: str | None = None,
        data: npt.NDArray | None = None,
    ) -> GradientField:
        if label is None:
            label = "GRA"
        if name is None:
            name = "gradient"

        # VolumetricField.__new__ signature: (grid, rank, label, name, data)
        obj = super().__new__(cls, grid, rank, label, name, data)
        obj._modulus = ScalarField(grid, name=name + "modulus", label=label + "M")
        return obj

    def __array_finalize__(self, obj: np.ndarray | None) -> None:
        # Restore attributes when we are taking a slice
        super().__array_finalize__(obj)
        if obj is None:
            return
        if isinstance(obj, GradientField):
            self._modulus = obj._modulus

    @property
    def modulus(self) -> ScalarField:
        """docstring"""
        self._compute_modulus()
        return self._modulus

    def _compute_modulus(self) -> None:
        """"""
        self._modulus[:] = np.sqrt(np.einsum("ijkl,ijkl->jkl", self, self))
