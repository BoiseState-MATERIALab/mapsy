# Refactored from Stephen Weitzner cube_vizkit
import numpy as np
import numpy.typing as npt
import sys

#
from ase import Atoms
from ase.units import Bohr

from mapsy.io.base import BaseParser
from mapsy.data import Grid, ScalarField


class CubeParser(BaseParser):
    """
    A data class for storing and manipulating cube files. Contains cell basis,
    atomic basis, and scalar field data.

    atoms: ase.Atoms
        Atoms object constructed from cell and atomic positions in cube file.

    grid: np.ndarray, shape [3, Nx, Ny, Nz]
        Rank 4 array that contains the cartesian coordinates of the numerical
        grid in units of Bohr.

    data3D: np.ndarray, shape [Nx, Ny, Nz]
        Rank 3 array that contains scalar field data on the corresponding grid.

        If charge data, in units of 'e'.
        If potential data, in units of 'Ry/e'.

    cell: np.ndarray, shape [3, 3]
        Each column contains a basis vector of the supercell in units of Bohr.

    origin: np.ndarray, shape[1, 3]
        origin of the supercell / atoms.
    """

    def _read_header(
        self,
    ) -> Grid:
        """
        _read_header()

        Extracts cell and grid information from Gaussian *.cube files.
        Atomic units are assumed

        Parameters
        ----------
        None

        Returns
        -------
        grid: Grid = information on cell and grid

        References
        ----------
        [1] http://www.gaussian.com/g_tech/g_ur/u_cubegen.htm
        """
        try:
            # -- Parse the header of the cube file
            self.natoms = int(self.contents[2].split()[0])
            origin: npt.NDArray[np.float64] = np.array(
                list(map(float, self.contents[2].split()[1:])), dtype=np.float64
            )
            header: list[str] = self.contents[3:6]
            N1: int = int(header[0].split()[0])
            N2: int = int(header[1].split()[0])
            N3: int = int(header[2].split()[0])
            R1: list[float] = list(map(float, header[0].split()[1:4]))
            R2: list[float] = list(map(float, header[1].split()[1:4]))
            R3: list[float] = list(map(float, header[2].split()[1:4]))
        except Exception as e:
            print("Error parsing header:")
            print(e)
            print("Exiting with code -2.")
            sys.exit(-2)

        # -- Get supercell dimensions
        basis: npt.NDArray[np.float64] = np.array(
            [R1, R2, R3], dtype=np.float64
        ).T  # store vectors as columns
        scalars = np.array([N1, N2, N3], dtype=np.int64)

        return Grid(basis=basis, scalars=scalars)

    def _read_atoms(
        self,
        grid: Grid,
    ) -> Atoms:
        """
        _read_atoms()

        Extracts atoms information from Gaussian *.cube files.
        Atomic units are assumed

        Parameters
        ----------
        grid: Grid = information on cell

        Returns
        -------
        atoms: Atoms = ASE Atoms read from file

        References
        ----------
        [1] http://www.gaussian.com/g_tech/g_ur/u_cubegen.htm
        """
        # -- Create an ASE Atoms object
        atoms: list[str] = self.contents[6 : 6 + self.natoms]
        tmp: npt.NDArray[np.float64] = np.array(
            [line.split() for line in atoms], dtype=np.float64
        )
        numbers: npt.NDArray[np.int64] = tmp[:, 0].astype(np.int64)
        charges: npt.NDArray[np.float64] = tmp[:, 1]
        positions: npt.NDArray[np.float64] = tmp[:, 2:]
        if self.units == "bohr":
            positions = positions * Bohr
        return Atoms(
            numbers=numbers, positions=positions, charges=charges, cell=grid.cell.T
        )

    def _read_data(
        self,
        grid: Grid,
        name: str = "data",
        label: str = "DAT",
    ) -> ScalarField:
        """
        _read_atoms()

        Extracts volumetric data from Gaussian *.cube files.
        Atomic units are assumed

        Parameters
        ----------
        grid: Grid = information on grid dimensions

        Returns
        -------
        data3D: ScalarField = volumetric data from file

        References
        ----------
        [1] http://www.gaussian.com/g_tech/g_ur/u_cubegen.htm
        """
        # -- Isolate scalar field data
        data1D = np.array(
            [
                float(val)
                for line in self.contents[6 + self.natoms :]
                for val in line.split()
            ]
        )
        data3D = data1D.reshape(
            (grid.scalars[2], grid.scalars[1], grid.scalars[0]),
            order="F",
        ).T
        return ScalarField(grid=grid, data=data3D, name=name, label=label)
