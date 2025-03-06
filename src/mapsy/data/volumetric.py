# Refactored from Stephen Weitzner cube_vizkit
from typing import Optional, Any
import numpy.typing as npt
import numpy as np
from numpy import ndarray

from mapsy.data import Grid


class VolumetricField(np.ndarray):
    """docstring"""

    def __new__(
        cls,
        grid: Grid,
        rank: int = 1,
        label: Optional[str] = None,
        name: Optional[str] = None,
        data: Optional[Any] = None,
    ):

        if rank == 1:
            scalars: tuple = tuple(grid.scalars)
        else:
            scalars: tuple = rank, *grid.scalars

        if data is not None:
            inputdata = np.asarray(data).reshape(scalars)
        else:
            inputdata = np.zeros(scalars, dtype=np.float64)

        obj = inputdata

        obj = obj.view(cls)

        obj.grid = grid  # type: ignore
        obj.rank = rank  # type: ignore
        if label is not None:
            obj.label = label  # type: ignore
        if name is not None:
            obj.name = name  # type: ignore

        return obj

    def __array_finalize__(self, obj) -> None:
        # Restore attributes when we are taking a slice
        if obj is None:
            return
        self.grid = getattr(obj, "grid", None)
        self.rank = getattr(obj, "rank", None)
        self.lable = getattr(obj, "label", None)
        self.name = getattr(obj, "name", None)
