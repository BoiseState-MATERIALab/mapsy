# Refactored from Stephen Weitzner cube_vizkit
from typing import Optional
import numpy.typing as npt
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from mapsy.data import Grid, VolumetricField


class HessianField(VolumetricField):
    """ """

    def __new__(
        cls,
        grid: Grid,
        rank: Optional[int] = 9,
        label: Optional[str] = None,
        name: Optional[str] = None,
        data: Optional[npt.NDArray] = None,
    ):

        if label is None:
            label = "HES"
        if name is None:
            name = "hessian"

        obj = super().__new__(cls, grid, rank, name, label, data)
        return obj

    def __array_finalize__(self, obj) -> None:
        # Restore attributes when we are taking a slice
        super().__array_finalize__(obj)
        if obj is None:
            return
