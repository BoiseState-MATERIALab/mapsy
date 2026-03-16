from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt


def cutoff(
    type: str,
    rcut: float,
) -> Callable[..., npt.NDArray[np.float64]]:
    if type == "cos":
        fc = cosfc
    elif type == "tanh":
        fc = tanhfc
    else:
        raise ValueError("Unknown cutoff type")
    return wraprcut(fc, rcut)


def wraprcut(
    f: Callable[..., npt.NDArray[np.float64]],
    rcut: float,
) -> Callable[..., npt.NDArray[np.float64]]:
    def wrapped(*args: Any, **kwargs: Any) -> npt.NDArray[np.float64]:
        # Preserve ndarray outputs (needed for vectorized distance evaluations)
        return np.asarray(f(rcut, *args, **kwargs), dtype=np.float64)

    return wrapped


def cosfc(
    rcut: float,
    rij: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return np.where(rij <= rcut, 0.5 * (np.cos(np.pi * rij / rcut) + 1.0), 0.0).astype(np.float64)


def tanhfc(
    rcut: float,
    rij: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return np.where(rij <= rcut, np.tanh(1 - rij / rcut) ** 3.0, 0.0).astype(np.float64)
