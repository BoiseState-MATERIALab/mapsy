from pathlib import Path

import numpy as np
import numpy.typing as npt
from ase.atoms import Atoms

from mapsy.boundaries import Boundary
from mapsy.data import Grid, System
from mapsy.utils import get_vdw_radii
from mapsy.utils.functions import ERFC, FunctionContainer


class IonicBoundary(Boundary):
    """docstring"""

    def __init__(
        self,
        mode: str,
        grid: Grid,
        alpha: float,
        softness: float,
        system: System,
        radius_table_file: str | None = None,
        label: str = "",
    ) -> None:
        super().__init__(
            mode,
            grid,
            label,
        )

        self.alpha = alpha
        self.softness = softness
        self.radius_table_file = Path(radius_table_file) if radius_table_file else None
        self._user_radius_table: dict[int, float] | None = None

        if self.radius_table_file and self.mode != "user":
            self.mode = "user"
        if self.mode == "user" and self.radius_table_file is None:
            raise ValueError("User radius mode requires a radius table file.")

        atoms = system.atoms
        if not isinstance(atoms, Atoms) or len(atoms) == 0:
            raise ValueError("System has no atoms defined.")
        self.ions = atoms.copy()

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
            radius = self._get_radius(atomic_number)
            sphere = ERFC(
                grid=self.grid,
                kind=4,
                dim=0,
                axis=0,
                width=radius * self.alpha,
                spread=self.softness,
                volume=1.0,
                pos=self.ions.positions[i],
                label=f"{atomic_number}_soft_sphere",
            )
            self.soft_spheres.append(sphere)

    def _get_radius(self, atomic_number: int) -> float:
        """Return the radius for an atom, using a user recipe when provided."""
        if self.mode == "user":
            radius = self._get_user_radius(atomic_number)
        else:
            radius = get_vdw_radii(atomic_number, self.mode)

        if radius <= 0:
            raise ValueError("Radius recipe must return a positive value.")

        return radius

    def _get_user_radius(self, atomic_number: int) -> float:
        """Fetch a user-specified radius either from a table file or stored mapping."""
        if self._user_radius_table is None:
            self._user_radius_table = self._load_radius_table()

        try:
            return self._user_radius_table[int(atomic_number)]
        except KeyError as exc:
            raise ValueError(
                f"User radius table does not contain an entry for atomic number {atomic_number}."
            ) from exc

    def _load_radius_table(self) -> dict[int, float]:
        """Load a mapping of atomic number -> radius from a two-column file."""
        if self.radius_table_file is None:
            raise ValueError("User radius mode requires a radius table file.")

        data = np.loadtxt(self.radius_table_file)
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)

        if data.shape[1] < 2:
            raise ValueError(
                f"Radius table file {self.radius_table_file} must have at least two columns (Z, radius)."
            )

        radii: dict[int, float] = {}
        for row in data:
            z = int(row[0])
            radius = float(row[1])
            if radius <= 0:
                raise ValueError(
                    f"Invalid radius {radius} for atomic number {z} in user table; values must be positive."
                )
            radii[z] = radius

        if not radii:
            raise ValueError(f"Radius table file {self.radius_table_file} contains no data.")

        return radii

    def _update_soft_spheres(self) -> None:
        """docstring"""
        pass

    def _build_solvent_aware_boundary(self) -> None:
        # TODO: real implementation
        return None
