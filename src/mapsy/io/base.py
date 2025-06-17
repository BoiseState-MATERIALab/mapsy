from sys import exit
from abc import ABC, abstractmethod
from mapsy.data import System, ScalarField
from ase import Atoms

from mapsy.data import Grid


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

    def _check_file(self, fname) -> None:
        try:
            if fname:
                self.fname = fname
            assert self.fname, "No filename provided."

            print(f"Loading {self.fname} ...")

            self.prefix = self.fname.split(".")[0]

            with open(self.fname, "r") as f:
                self.contents: list[str] = f.readlines()

        except Exception as e:
            print("Unable to open file:")
            print(e)
            print("Exiting with code -1.")
            exit(-1)

    def _check_units(self, units: str = "bohr") -> None:
        allowed_units = ["bohr", "angstrom","alat"]
        if units not in allowed_units:
            raise ValueError("Wrong units option for file parser")
        self.units = units

    def systemparse(self) -> System:
        # Read the header and generate the grid
        grid: Grid = self._read_header()
        # Read atoms
        atoms = None
        if self.natoms:
            atoms: Atoms = self._read_atoms(grid)
        # Read data
        electrons = None
        if self.hasdata:
            electrons: ScalarField = self._read_data(
                grid, name="electrons", label="ELE"
            )
        return System(grid, atoms, electrons)

    def dataparse(self, name="data", label="DAT") -> ScalarField:
        # Read the header and generate the grid
        grid: Grid = self._read_header()
        # Read data
        data = None
        if self.hasdata:
            data: ScalarField = self._read_data(grid, name, label)
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
    def _read_data(self, grid: Grid, name: str, label: str) -> ScalarField:
        """"""
        ...
