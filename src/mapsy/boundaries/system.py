from ase.atoms import Atoms

from mapsy.boundaries import Boundary
from mapsy.data import Grid, System
from mapsy.utils.functions import ERFC


class SystemBoundary(Boundary):
    """docstring"""

    def __init__(
        self,
        mode: str,
        grid: Grid,
        distance: float,
        spread: float,
        system: System,
        label: str = "",
    ) -> None:
        super().__init__(
            mode,
            grid,
            label,
        )

        self.system = system

        atoms = self.system.atoms
        if not isinstance(atoms, Atoms) or len(atoms) == 0:
            raise ValueError("System has no atoms defined.")
        com = atoms.get_center_of_mass()

        self.simple = ERFC(
            grid=self.grid,
            kind=3,
            dim=system.dimension,
            axis=system.axis,
            width=distance,
            spread=spread,
            volume=1.0,
            pos=com,
        )

    def update(self) -> None:
        """docstring"""

        self._build()
        self._update_solvent_aware_boundary()

    def _build(self) -> None:
        """docstring"""

        self.switch[:] = self.simple.density
        self.gradient[:] = self.simple.gradient

    def _build_solvent_aware_boundary(self) -> None:
        # TODO: real implementation
        return None
