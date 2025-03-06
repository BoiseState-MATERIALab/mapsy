import numpy.typing as npt
import numpy as np

from mapsy.utils.constants import BOHR_RADIUS_ANGS

def goodscalars(guesses: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """
    Given a list of integer numbers, returns a corresponding list
    with larger or equal integers that are only multiples of 2, 3, and 5
    """
    scalars : list = []
    for guess in guesses:
        while True:
            guess += 1
            temp: np.int64 = guess
            powers: list[np.int64] = []
            factors: list[np.int64] = [2, 3, 5]  # can add factors, if needed
            for factor in factors:
                maxpower: np.int64 = np.int64(np.log(temp) / np.log(factor))
                power: np.int64 = 0
                for i in range(maxpower):
                    if temp % factor == 0:
                        power += 1
                        temp: np.int64 = temp // factor
                        if temp == 1:
                            break
                powers.append(power)
            if temp == 1:
                break
        scalars.append(guess)
    return np.array(scalars, dtype=np.int64)

def setscalars(cell: npt.NDArray[np.float64], cutoff: np.float64) -> npt.NDArray[np.int64]:
    """
       follows QE convention (realspace_grid_init in FFTXLib)
    ! ... first, an estimate of nr1,nr2,nr3, based on the max values
    ! ... of n_i indices in:   G = i*b_1 + j*b_2 + k*b_3
    ! ... We use G*a_i = n_i => n_i .le. |Gmax||a_i|
       cutoff is ecutrho
       this is the energy associated with the highest G vector plane wave
       the kinetic energy of a plane wave is hbar**2 * Gmax**2 / 2 / me
       if ecutrho is in Ry units (hbar**2/bohr**2/2/me = 1 Ry)
       means Gmax (in 1/bohr) = np.sqrt(ecutrho)
       in QE G vectors are in units of 2*pi/alat, while cell vectors are in units of alat
       the number of gridpoints in one direction 2*pi*n_i <= |Gmax||cell_i|
       assuming Gmax and cell are expressed in terms of the same units
       n_i <= np.sqrt(ecutrho)/bohr2ang*|cell_i|/pi NOTE:there is a missing factor of 2
       for FFTs we want to choose numbers that are only multiple of 2, 3 and 5
    """
    cell_vector_lengths: npt.NDArray = np.sqrt(
        np.einsum("ij,ij->i", cell, cell)
    )
    scalars: npt.NDArray = goodscalars(
        np.int64(
            np.sqrt(cutoff) / BOHR_RADIUS_ANGS * cell_vector_lengths / np.pi
        )
    )
    return scalars
