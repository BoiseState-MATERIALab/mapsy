# Read XYZ+ files (+: cell information in the second line)
import logging

import numpy as np
import numpy.typing as npt

#
from ase import Atoms
from ase.units import Bohr

from mapsy.data import Grid, ScalarField
from mapsy.io.base import BaseParser

logger = logging.getLogger(__name__)


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
            origin: None | npt.NDArray[np.float64] = None
            if len(self.contents[0].split()) == 4:
                # assume the remaining numbers in the first line to be
                # the origin of the cell
                logger.info("Reading cell origin information from first line")
                origin = np.array(list(map(float, self.contents[0].split()[1:])), dtype=np.float64)
            a: float = 1.0
            if len(self.contents[1].split()) == 1:
                logger.info("Assuming a cubic cell of size alat")
                a = float(self.contents[1].split()[0])
                cell = np.eye(3) * a
            elif len(self.contents[1].split()) == 3:
                logger.info("Assuming an orthorombic cell with sides a, b, c")
                a, b, c = map(float, self.contents[1].split()[:])
                cell = np.diag([a, b, c])
                if self.units == "bohr":
                    cell *= Bohr
                elif self.units == "alat":
                    cell[1, 1] *= a
                    cell[2, 2] *= a
            elif len(self.contents[1].split()) == 9:
                logger.info("Reading a full 3x3 cell matrix")
                cell = np.array(self.contents[1].split(), dtype=np.float64).reshape(3, 3)
                if self.units == "bohr":
                    cell *= Bohr
                elif self.units == "alat":
                    a = cell[0, 0]
                    cell *= a
                    cell[0, 0] = a

        except Exception as e:
            logger.exception(f"Error parsing xyz+ header for {self.fname}")
            raise ValueError("Invalid xyz+ header") from e

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
        logger.info(f"Reading {self.natoms} atoms")
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

    def _read_data(self, grid: Grid, name: str = "data", label: str = "DAT") -> ScalarField | None:
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
