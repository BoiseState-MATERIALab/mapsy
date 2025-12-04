from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .grid import Grid
from .volumetric import VolumetricField


class HessianField(VolumetricField):
    """ """

    def __new__(
        cls: type[HessianField],
        grid: Grid,
        rank: int = 9,
        label: str | None = None,
        name: str | None = None,
        data: npt.NDArray | None = None,
    ) -> HessianField:
        if label is None:
            label = "HES"
        if name is None:
            name = "hessian"

        # VolumetricField.__new__ signature: (grid, rank, rank_axis_first, label, name, data)
        obj = super().__new__(cls, grid, rank, True, label, name, data)
        return obj

    def __array_finalize__(self, obj: np.ndarray | None) -> None:
        # Restore attributes when we are taking a slice
        super().__array_finalize__(obj)
        if obj is None:
            return
