#
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from ase import Atoms

from mapsy.data import Grid, ScalarField


@dataclass
class System:
    # Required (no default): must be provided by the caller
    grid: Grid

    # Optional payload
    atoms: Atoms | None = None
    electrons: ScalarField | None = None

    # Simple defaults
    dimension: int = 0
    axis: int = 0

    # Initialized with default_factory to avoid mutable default argument
    center: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    # Collected scalar fields/properties
    data: list[ScalarField] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Ensure center reflects atoms if present, otherwise the grid origin
        self.center = np.asarray(
            (self.atoms.get_center_of_mass() if self.atoms is not None else self.grid.origin),
            dtype=np.float64,
        )

    def addproperty(self, data: ScalarField | None) -> None:
        if data is not None:
            self.data.append(data)
