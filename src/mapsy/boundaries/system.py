from mapsy.data import Grid, System, ScalarField, GradientField
from mapsy.utils.functions import ERFC
from mapsy.boundaries import Boundary


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

        com = self.system.atoms.get_center_of_mass()

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
