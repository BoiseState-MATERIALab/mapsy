from collections.abc import Callable
from typing import Any, SupportsFloat

import numpy as np
import numpy.typing as npt


def cutoff(
    type: str,
    rcut: float,
) -> Callable:
    if type == "cos":
        fc = cosfc
    elif type == "tanh":
        fc = tanhfc
    else:
        raise ValueError("Unknown cutoff type")
    return wraprcut(fc, rcut)


def wraprcut(
    f: Callable[..., SupportsFloat],
    rcut: float,
) -> Callable[..., float]:
    def wrapped(*args: Any, **kwargs: Any) -> float:
        return float(f(rcut, *args, **kwargs))

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
