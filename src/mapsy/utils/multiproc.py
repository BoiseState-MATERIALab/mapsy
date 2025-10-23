from collections.abc import Callable, Sequence
from typing import Any, TypeVar, cast

import numpy as np
import numpy.typing as npt
from pathos.multiprocessing import ProcessingPool, cpu_count

T = TypeVar("T")


def full2chunk(inputs: npt.NDArray) -> list[npt.NDArray]:
    """Split along axis 0 and return a list of ndarray chunks."""
    n_cpus = min(cpu_count(), len(inputs))
    return list(np.array_split(inputs, n_cpus, axis=0))


def chunk2full(outputs: Sequence[Sequence[Sequence[Any]]], n: int) -> list[list[np.float64]]:
    """Combine per-chunk results into a list of length n."""
    results: list[list[np.float64]] = [[] for _ in range(n)]
    for i in range(n):
        for chunk in outputs:
            results[i].extend(chunk[i])
    return results


def multiproc(function: Callable[[npt.NDArray], T], args: Sequence[npt.NDArray]) -> list[T]:
    """Map `function` over array chunks using a process pool."""
    with ProcessingPool() as pool:
        # `pool.map` is untyped; tell mypy what it returns.
        return cast(list[T], pool.map(function, args))
