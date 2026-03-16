import logging
from abc import ABC, abstractmethod

from ase import Atoms

from mapsy.data import Grid, ScalarField, System

logger = logging.getLogger(__name__)


class BaseParser(ABC):
    fname: str = ""
    units: str = "bohr"
    natoms: int = 0
    hasdata: bool = False
    prefix: str = ""
    contents: list[str] = []

    def __init__(
        self,
        fname: str = "",
        units: str = "bohr",
        natoms: int = 0,
        hasdata: bool = False,
    ) -> None:
        """ """
        # Check that file exists
        self._check_file(fname)
        # Check that units are allowed
        self._check_units(units)
        # Number of atoms
        self.natoms = natoms
        # Contains volumetric data
        self.hasdata = hasdata

    def _check_file(self, fname: str) -> None:
        if fname:
            self.fname = fname
        if not self.fname:
            raise ValueError("File name is required for file parser")

        logger.info(f"Loading {self.fname} ...")
        self.prefix = self.fname.rsplit(".", 1)[0]
        try:
            with open(self.fname) as f:
                self.contents: list[str] = f.readlines()
        except Exception as e:
            logger.exception(f"Unable to open file:{self.fname}")
            raise OSError(f"Unable to open file {self.fname}") from e

    def _check_units(self, units: str = "bohr") -> None:
        allowed_units = ["bohr", "angstrom", "alat"]
        if units not in allowed_units:
            raise ValueError("Wrong units option for file parser")
        self.units = units

    def systemparse(self) -> System:
        # Read the header and generate the grid
        grid: Grid = self._read_header()

        # Read atoms
        atoms: Atoms | None = None
        if self.natoms:
            atoms = self._read_atoms(grid)

        # Read data
        electrons: ScalarField | None = None
        if self.hasdata:
            electrons = self._read_data(grid, name="electrons", label="ELE")

        return System(grid, atoms, electrons)

    def dataparse(self, name: str = "data", label: str = "DAT") -> ScalarField | None:
        # Read the header and generate the grid
        grid: Grid = self._read_header()
        # Read data
        data: ScalarField | None = None
        if self.hasdata:
            data = self._read_data(grid, name, label)
        return data

    @abstractmethod
    def _read_header(self) -> Grid:
        """"""
        ...

    @abstractmethod
    def _read_atoms(self, grid: Grid) -> Atoms:
        """"""
        ...

    @abstractmethod
    def _read_data(self, grid: Grid, name: str, label: str) -> ScalarField | None:
        """"""
        ...
