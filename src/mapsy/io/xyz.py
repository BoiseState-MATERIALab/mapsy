# Read XYZ+ files (+: cell information in the second line)
import numpy as np
import numpy.typing as npt
import sys

#
from ase import Atoms
from ase.units import Bohr

from mapsy.io.base import BaseParser
from mapsy.data import Grid


class XYZParser(BaseParser):
    """ """

    def _read_header(
        self,
    ) -> Grid:
        """
        _read_header()

        Extracts number of atoms and cell information from xyz files.
        Cell information is expected in the second line of the file.
        Atomic units are assumed

        Parameters
        ----------
        units: string, optional (default='bohr')

        Returns
        -------
        grid: Grid = information on cell and grid read from file
        """
        try:
            # -- Parse the header of the cube file
            self.natoms = int(self.contents[0].split()[0])
            origin = None
            if len(self.contents[0].split()) == 4:
                # assume the remaining numbers in the first line to be
                # the origin of the cell
                print("Reading cell origin information from first line")
                origin: npt.NDArray[np.float64] = np.array(
                    list(map(float, self.contents[0].split()[1:])), dtype=np.float64
                )
            if len(self.contents[1].split()) == 1:
                print("Assuming a cubic cell of size alat")
                a: float = float(self.contents[1].split()[0])
                cell = np.eye(3) * a
            elif len(self.contents[1].split()) == 3:
                print("Assuming an orthorombic cell with sides a, b, c")
                a, b, c = map(float, self.contents[1].split()[:])
                cell = np.diag([a, b, c])
                if self.units == "bohr":
                    cell *= Bohr
                elif self.units == "alat":
                    cell[1, 1] *= a
                    cell[2, 2] *= a
            elif len(self.contents[1].split()) == 9:
                print("Reading a full 3x3 cell matrix")
                cell = np.array(self.contents[1].split(), dtype=np.float64).reshape(
                    3, 3
                )
                if self.units == "bohr":
                    cell *= Bohr
                elif self.units == "alat":
                    a: float = cell[0, 0]
                    cell *= a
                    cell[0, 0] = a

        except Exception as e:
            print("Error parsing xyz+ header:")
            print(e)
            print("Exiting with code -2.")
            sys.exit(-2)

        return Grid(cell=cell, origin=origin)

    def _read_atoms(self, grid: Grid) -> Atoms:
        """
        _read_atoms(cube_file)

        Extracts atoms information from xyz files.
        Atomic units are assumed

        Parameters
        ----------
        grid: Grid = information on cell

        Returns
        -------
        atoms: Atoms = ASE atoms read from file
        """
        print(f"Reading {self.natoms} atoms")
        # -- Create an ASE Atoms object
        atoms: list[str] = self.contents[2 : 2 + self.natoms]
        elements: list[str] = [line.split()[0] for line in atoms]
        positions: npt.NDArray[np.float64] = np.array(
            [line.split()[1:4] for line in atoms], dtype=np.float64
        )
        if self.units == "bohr":
            positions = positions * Bohr
        elif self.units == "alat":
            positions = positions * grid.cell[0, 0]
        return Atoms(symbols=elements, positions=positions, cell=grid.cell, pbc=True)

    def _read_data(self, grid: Grid, name: str = "data", label: str = "DAT") -> None:
        """
        _read_data(xyz_file)

        Extracts volumetric data from xyz files.
        Atomic units are assumed

        Parameters
        ----------
        grid: Grid = information on cell

        Returns
        -------
        None
        """
        return None
