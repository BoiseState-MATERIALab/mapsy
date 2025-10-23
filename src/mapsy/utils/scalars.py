import numpy as np
import numpy.typing as npt

from mapsy.utils.constants import BOHR_RADIUS_ANGS


def _is_235_smooth(n: int) -> bool:
    """Return True if n's prime factors are only 2, 3, and/or 5."""
    if n < 1:
        return False
    for p in (2, 3, 5):
        while n % p == 0:
            n //= p
    return n == 1


def goodscalars(guesses: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """
    Given an array of integers, return the smallest integers >= each guess
    that are composed only of prime factors 2, 3, and 5.
    """
    out: list[int] = []
    # Convert to Python ints so mypy doesn't complain about np.int64 vs int.
    for g in guesses.tolist():
        n: int = int(g)  # start at guess (>=, not strictly >)
        while not _is_235_smooth(n):
            n += 1
        out.append(n)
    return np.asarray(out, dtype=np.int64)


def setscalars(cell: npt.NDArray[np.float64], cutoff: float) -> npt.NDArray[np.int64]:
    """
    Compute grid scalars following QE convention, then snap each to the nearest
    2/3/5-smooth integer >= estimate.
    """
    cell_vector_lengths: npt.NDArray[np.float64] = np.sqrt(np.einsum("ij,ij->i", cell, cell))
    # Compute float estimates, then cast to int64 array of initial guesses
    estimates_f = (np.sqrt(float(cutoff)) / BOHR_RADIUS_ANGS) * (cell_vector_lengths / np.pi)
    estimates_i: npt.NDArray[np.int64] = np.asarray(estimates_f, dtype=np.int64)
    return goodscalars(estimates_i)
