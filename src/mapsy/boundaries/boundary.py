from abc import ABC, abstractmethod

from mapsy.data import Grid, ScalarField, GradientField
from mapsy.utils.functions import ERFC


class Boundary(ABC):
    """docstring"""

    def __init__(
        self,
        mode: str,
        grid: Grid,
        label: str = "",
    ) -> None:
        self.mode = mode
        self.label = label
        self.volume = 0.0
        self.surface = 0.0
        self.solvent_aware = False

        self.grid = grid

        boundary_label = f"{label}_boundary"
        self.switch = ScalarField(grid, label=boundary_label)
        gradient_label = f"{label}_boundary_gradient"
        self.gradient = GradientField(grid, label=gradient_label)

    @abstractmethod
    def update(self) -> None:
        """docstring"""

    @abstractmethod
    def _build(self) -> None:
        """docstring"""

    def _update_solvent_aware_boundary(self) -> None:
        """docstring"""
        if self.solvent_aware:
            self._build_solvent_aware_boundary()

    def _build_solvent_aware_boundary(self) -> None:
        """docstring"""
        pass
