#
import numpy as np
import numpy.typing as npt

from dataclasses import dataclass, field


from ase import Atoms
from mapsy.data import Grid, ScalarField


@dataclass
class System:

    dimension: int = 0
    axis: int = 0
    atoms: Atoms = None
    grid: Grid = None
    electrons: ScalarField = None
    data: list = field(default_factory=list, repr=False)
    center: npt.NDArray[np.float64] = None

    def __init__(
        self,
        grid: Grid,
        atoms: Atoms = None,
        electrons: ScalarField = None,
        dimension: int = 0,
        axis: int = 0,
    ) -> None:
        """"""
        self.grid = grid
        self.center = grid.origin
        if atoms is not None:
            self.atoms = atoms
            self.center = self.atoms.get_center_of_mass()
        if electrons is not None:
            self.electrons = electrons
        self.data = []
        self.dimension = dimension
        self.axis = axis

    def addproperty(self, data: ScalarField) -> None:
        self.data.append(data)
