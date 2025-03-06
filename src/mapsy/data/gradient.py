# Refactored from Stephen Weitzner cube_vizkit
from typing import Optional
import numpy.typing as npt
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from mapsy.data import Grid, VolumetricField, ScalarField


class GradientField(VolumetricField):
    """ """

    def __new__(
        cls,
        grid: Grid,
        rank: Optional[int] = 3,
        label: Optional[str] = None,
        name: Optional[str] = None,
        data: Optional[npt.NDArray] = None,
    ):

        if label is None:
            label = "GRA"
        if name is None:
            name = "gradient"

        obj = super().__new__(cls, grid, rank, name, label, data)

        obj._modulus = ScalarField(grid, name=name + "modulus", label=label + "M")
        return obj

    def __array_finalize__(self, obj) -> None:
        # Restore attributes when we are taking a slice
        super().__array_finalize__(obj)
        if obj is None:
            return
        if isinstance(obj, (GradientField)):
            self._modulus = getattr(obj, "_modulus", None)

    @property
    def modulus(self) -> ScalarField:
        """docstring"""
        self._compute_modulus()
        return self._modulus

    def _compute_modulus(self) -> None:
        """"""
        self._modulus[:] = np.sqrt(np.einsum("ijkl,ijkl->jkl", self, self))
