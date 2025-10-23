import warnings

import numpy as np
import scipy.special as sp

from mapsy.data import GradientField, HessianField, ScalarField
from mapsy.utils.constants import FPI, SQRTPI
from mapsy.utils.functions.functions import FUNC_TOL, FieldFunction


class ERFC(FieldFunction):
    """docstring"""

    def _compute_density(self) -> None:
        """docstring"""

        _, r2 = self.grid.get_min_distance(self.pos, self.dim, self.axis)
        dist = np.sqrt(r2)
        arg = (dist - self.width) / self.spread

        self._density = ScalarField(self.grid, label=self.label)

        if self.kind == 4:
            self._density[:] = self.volume

        density = np.zeros(self._density.shape)
        density[:] = sp.erfc(arg)

        integral = np.sum(density) * self.grid.volume / self.grid.ndata * 0.5

        charge = self._charge()
        analytic = self._erfc_volume()

        rel_err = abs((integral - analytic) / analytic)
        if rel_err > 1e-4:
            warnings.warn(
                f"ERFC integral check failed (rel err={rel_err:.2e}).",
                category=RuntimeWarning,
                stacklevel=2,
            )

        scale = charge / analytic * 0.5

        self._density[:] += density * scale

    def _compute_gradient(self) -> None:
        """docstring"""

        r, r2 = self.grid.get_min_distance(self.pos, self.dim, self.axis)
        dist = np.sqrt(r2)
        arg = (dist - self.width) / self.spread

        mask = dist > FUNC_TOL

        r = r[:, mask]
        dist = dist[mask]
        arg = arg[mask]

        self._gradient = GradientField(self.grid, label=self.label)

        gradient = np.zeros(self._gradient.shape)
        gradient[:, mask] = -np.exp(-(arg**2)) * r / dist

        charge = self._charge()
        analytic = self._erfc_volume()
        scale = charge / analytic / SQRTPI / self.spread

        self._gradient[:, mask] += gradient[:, mask] * scale

    def _compute_laplacian(self) -> None:
        """docstring"""

        _, r2 = self.grid.get_min_distance(self.pos, self.dim, self.axis)
        dist = np.sqrt(r2)
        arg = (dist - self.width) / self.spread

        mask = dist > FUNC_TOL

        dist = dist[mask]
        arg = arg[mask]

        self._laplacian = ScalarField(self.grid, label=self.label)

        laplacian = np.zeros(self._laplacian.shape)

        exp = np.exp(-(arg**2))

        if self.dim == 0:
            laplacian[mask] = -exp * (1 / dist - arg / self.spread) * 2
        elif self.dim == 1:
            laplacian[mask] = -exp * (1 / dist - 2 * arg / self.spread)
        elif self.dim == 2:
            laplacian[mask] = exp * arg / self.spread * 2
        else:
            raise ValueError("unexpected system dimensions")

        charge = self._charge()
        analytic = self._erfc_volume()
        scale = charge / analytic / SQRTPI / self.spread

        self._laplacian[mask] += laplacian[mask] * scale

    def _compute_hessian(self) -> None:
        """docstring"""

        r, r2 = self.grid.get_min_distance(self.pos, self.dim, self.axis)
        dist = np.sqrt(r2)
        arg = (dist - self.width) / self.spread

        mask = dist > FUNC_TOL

        r = r[:, mask]
        dist = dist[mask]
        arg = arg[mask]

        self._hessian = HessianField(self.grid, label=self.label)

        hessian = np.zeros(self._hessian.shape)

        shape = np.count_nonzero(mask)
        outer = np.reshape(np.einsum("i...,j...->ij...", -r, r), (9, shape))
        outer *= 1 / dist + 2 * arg / self.spread
        outer += dist * np.identity(3).flatten()[:, None]

        hessian[:, mask] = -np.exp(-(arg**2)) * outer / dist**2

        charge = self._charge()
        analytic = self._erfc_volume()
        scale = charge / analytic / SQRTPI / self.spread

        self._hessian[:, mask] += hessian[:, mask] * scale

    def _compute_derivative(self) -> None:
        """docstring"""

        _, r2 = self.grid.get_min_distance(self.pos, self.dim, self.axis)
        dist = np.sqrt(r2)
        arg = (dist - self.width) / self.spread

        mask = dist > FUNC_TOL

        arg = arg[mask]

        self._derivative = ScalarField(self.grid, label=self.label)

        derivative = np.zeros(self._derivative.shape)
        derivative[mask] = -np.exp(-(arg**2))

        integral = np.sum(self._derivative) * self.grid.volume / self.grid.ndata * 0.5

        charge = self._charge()
        analytic = self._erfc_volume()

        rel_err = abs((integral - analytic) / analytic)
        if rel_err > 1e-4:
            warnings.warn(
                f"ERFC integral check failed (rel err={rel_err:.2e}).",
                category=RuntimeWarning,
                stacklevel=2,
            )

        scale = charge / analytic / SQRTPI / self.spread

        self._derivative[mask] += derivative[mask] * scale

    def _charge(self) -> float:
        """docstring"""
        charge: float = float(self.volume)
        if self.kind == 1:
            raise ValueError("wrongly set as a gaussian")
        elif self.kind == 2:
            pass
        elif self.kind == 3:
            charge *= self._erfc_volume()
        elif self.kind == 4:
            charge *= -self._erfc_volume()
        else:
            raise ValueError("unexpected function type")
        return charge

    def _erfc_volume(self) -> float:
        """docstring"""

        spread: float = float(self.spread)
        width: float = float(self.width)

        if any(attr < FUNC_TOL for attr in (spread, width)):
            raise ValueError("wrong parameters for erfc function")

        t: float = spread / width
        invt: float = width / spread
        f1: float = (1 + sp.erf(invt)) * 0.5
        f2: float = np.exp(-(invt**2)) * 0.5 / SQRTPI

        volume: float = 0.0
        if self.dim == 0:
            volume = FPI / 3 * width**3 * ((1.0 + 1.5 * t**2) * f1 + (1.0 + t**2) * t * f2)

        elif self.dim == 1:
            volume = (
                np.pi
                * width**2
                * self.grid.cell[self.axis, self.axis]
                * ((1.0 + 0.5 * t**2) * f1 + t * f2)
            )

        elif self.dim == 2:
            volume = 2.0 * width * self.grid.volume / self.grid.cell[self.axis, self.axis]

        else:
            raise ValueError("unexpected system dimensions")

        return volume
