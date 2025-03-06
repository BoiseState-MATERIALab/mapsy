from copy import deepcopy
from typing import Optional

import numpy as np
import numpy.typing as npt

from mapsy.data import Grid, System, ScalarField, GradientField
from mapsy.utils import get_vdw_radii
from mapsy.utils.functions import FunctionContainer, ERFC
from mapsy.boundaries import Boundary


class IonicBoundary(Boundary):
    """docstring"""

    def __init__(
        self,
        mode: str,
        grid: Grid,
        alpha: float,
        softness: float,
        system: System,
        label: str = "",
    ) -> None:

        super().__init__(
            mode,
            grid,
            label,
        )

        self.alpha = alpha
        self.softness = softness

        self.ions = system.atoms

        self._set_soft_spheres()

    def update(self) -> None:
        """docstring"""

        self._build()
        self._update_solvent_aware_boundary()

    def _build(self) -> None:
        """docstring"""

        self.soft_spheres.reset_derivatives()

        self.switch[:] = 1.0

        for sphere in self.soft_spheres:
            self.switch[:] *= sphere.density

        self._compute_gradient()

        self.switch[:] = 1.0 - self.switch
        self.gradient[:] *= -1

    def _compute_gradient(self) -> None:
        """docstring"""
        for sphere in self.soft_spheres:
            mask = np.abs(sphere.density) > 1e-60
            if not np.any(mask):
                continue

            self.gradient[:, mask] += (
                sphere.gradient[:, mask] * self.switch[mask] / sphere.density[mask]
            )

    def _set_soft_spheres(self) -> None:
        """docstring"""

        self.soft_spheres = FunctionContainer(self.grid)

        atomic_numbers: npt.NDArray[np.int64] = self.ions.get_atomic_numbers()
        for i, atomic_number in enumerate(atomic_numbers):
            sphere = ERFC(
                grid=self.grid,
                kind=4,
                dim=0,
                axis=0,
                width=get_vdw_radii(atomic_number, self.mode) * self.alpha,
                spread=self.softness,
                volume=1.0,
                pos=self.ions.positions[i],
                label=f"{atomic_number}_soft_sphere",
            )
            self.soft_spheres.append(sphere)

    def _update_soft_spheres(self) -> None:
        """docstring"""
        pass
