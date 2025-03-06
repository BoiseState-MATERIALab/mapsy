from pathos.multiprocessing import cpu_count, ProcessingPool
from typing import Callable
import numpy.typing as npt
import numpy as np

def full2chunk(inputs: npt.NDArray) -> list[tuple[npt.NDArray]]:
    """ 
    Split a set of input values into chunks according to the first dimension of the array 
    """
    n_cpus = min(cpu_count(), len(inputs))  # Ensure we don't use more CPUs than input values
    chunks: list[npt.NDArray] = np.array_split(inputs,n_cpus,0)# Split positions into chunks along first axis
    args: list[tuple[npt.NDArray]] = [(chunk) for chunk in chunks] # do we need to put the chunks into a tuple?
    return args

def chunk2full(outputs: list, n: int) -> list:
    """ 
    Combine results from different processors into a sigle list. 
    Assume results for each processor are lists of size n
    Return a single list of size n with all the results
    """
    results: list[list] = [[] for i in range(n)]
    for i in range(n):
        for chunk in outputs:
            results[i].extend(chunk[i])
    return results

def multiproc(function: Callable, args: list[tuple])-> list:
    """
    Split processing of function over multiple processors
    """
    with ProcessingPool() as pool:
        chunk_results: list = pool.map(function, args)
    return chunk_results
