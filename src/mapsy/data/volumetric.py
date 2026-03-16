# Refactored from Stephen Weitzner cube_vizkit
from typing import Any, TypeAlias, TypeVar, cast

import numpy as np
import numpy.typing as npt

from .grid import Grid

Vec3i: TypeAlias = npt.NDArray[np.int64]
T = TypeVar("T", bound="VolumetricField")


def _shape_from_scalars(scalars: Vec3i) -> tuple[int, int, int]:
    arr = np.asarray(scalars, dtype=np.int64).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"grid.scalars must be 3 integers, got {arr.tolist()}")
    nx, ny, nz = int(arr[0]), int(arr[1]), int(arr[2])
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError(f"grid.scalars must be positive, got {(nx, ny, nz)}")
    return nx, ny, nz


class VolumetricField(np.ndarray):
    """docstring"""

    def __new__(
        cls: type[T],
        grid: Grid,
        rank: int = 1,
        rank_axis_first: bool = False,
        label: str | None = None,
        name: str | None = None,
        data: npt.ArrayLike | None = None,
    ) -> T:
        base_shape = _shape_from_scalars(grid.scalars)  # (nx, ny, nz)
        expected: tuple[int, ...]
        if rank == 1:
            expected = base_shape
        elif rank_axis_first:
            expected = (rank, *base_shape)  # (rank, nx, ny, nz)
        else:
            expected = (*base_shape, rank)  # (nx, ny, nz, rank)

        if data is None:
            arr = np.zeros(expected, dtype=np.float64)
        else:
            arr = np.asarray(data, dtype=np.float64)
            if arr.shape != expected:
                # Try a common alternative layout (swap leading/trailing rank axis) before broadcasting.
                if rank != 1:
                    alt_expected = (*base_shape, rank) if rank_axis_first else (rank, *base_shape)
                    if arr.shape == alt_expected:
                        moved_axis = -1 if rank_axis_first else 0
                        arr = np.moveaxis(arr, moved_axis, 0 if rank_axis_first else -1)
                if arr.shape != expected:
                    try:
                        arr = np.broadcast_to(arr, expected).copy()
                    except ValueError as e:
                        raise ValueError(
                            f"data.shape {arr.shape} incompatible with expected {expected}"
                        ) from e

        out = cast(T, np.asarray(arr, dtype=np.float64).view(cls))
        out.grid = grid
        out.rank = int(rank)
        out.rank_axis_first = bool(rank_axis_first)
        if label is not None:
            out.label = label
        if name is not None:
            out.name = name
        return out

    def __array_finalize__(self, obj: Any) -> None:
        # Restore attributes when we are taking a slice
        if obj is None:
            return
        self.grid = getattr(obj, "grid", None)
        self.rank = getattr(obj, "rank", None)
        self.rank_axis_first = getattr(obj, "rank_axis_first", False)
        self.label = getattr(obj, "label", None)
        self.name = getattr(obj, "name", None)
