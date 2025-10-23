# Atom-Centered Symmetry Functions
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from ase import Atoms
from numpy.polynomial.chebyshev import Chebyshev as cheb

from mapsy.utils import cutoff

from ..input import SymFuncModel
from ..symmetryfunction import SymmetryFunction


def wrapcheby(
    f: Callable[[int, float, npt.ArrayLike], npt.NDArray[np.float64]],
    order: int,
    rcut: float,
) -> Callable[[npt.ArrayLike], npt.NDArray[np.float64]]:
    def wrapped(x: npt.ArrayLike) -> npt.NDArray[np.float64]:
        return np.asarray(f(order, rcut, x), dtype=np.float64)

    return wrapped


def dual_basis_function(order: int, rcut: float, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
    return np.asarray(cheb.basis(order)(2 * np.asarray(x) / rcut - 1), dtype=np.float64)


def basis_function(order: int, rcut: float, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
    xa = np.asarray(x, dtype=np.float64)
    k = 2 if order == 0 else 1
    return np.asarray(
        (k / (2 * np.pi * np.sqrt(xa / rcut - xa**2 / rcut**2)) / 4)
        * rcut
        * cheb.basis(order)(2 * xa / rcut - 1),
        dtype=np.float64,
    )


class ACSFParser:
    def __init__(self, symfuncmodel: SymFuncModel) -> None:
        self.order = np.array(symfuncmodel.order, dtype=np.int64)
        self.cutoff = symfuncmodel.cutoff
        self.radius = symfuncmodel.radius
        self.radial = symfuncmodel.radial
        self.compositional = symfuncmodel.compositional
        self.structural = symfuncmodel.structural

    def parse(self) -> list[SymmetryFunction]:
        symmetryfunctions: list[SymmetryFunction] = []
        symmetryfunctions.append(
            ACSymmetryFunction(
                self.order,
                self.radius,
                self.cutoff,
                self.compositional,
                self.structural,
                self.radial,
            )
        )
        return symmetryfunctions


class ACSymmetryFunction(SymmetryFunction):
    def __init__(
        self,
        order: npt.NDArray[np.int64],
        radius: float,
        cutofftype: str,
        compositional: bool = False,
        structural: bool = False,
        radial: bool = True,
    ) -> None:
        super().__init__(kind=2, label="ACSF")
        self.order = order

        self.fc: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] = cutoff(
            cutofftype, radius
        )
        self.rcut: float = radius

        self.compositional = bool(compositional)
        self.structural = bool(structural)
        self.radial = bool(radial)
        # If neither structural nor compositional was requested, default to structural to avoid empty outputs.
        if not (self.structural or self.compositional):
            self.structural = True
        if self.radial:
            if radius == 0:
                raise ValueError()
            self.cutoff = float(radius)
        else:
            self.cutoff = float(np.pi)
        # polynomial basis functions: arraylike -> ndarray
        self.polynomia: list[Callable[[npt.ArrayLike], npt.NDArray[np.float64]]] = []
        self._generate_functions()

    def _generate_functions(self) -> None:
        function = basis_function if self.radial else dual_basis_function
        for order in self.order:
            pol_order = wrapcheby(function, order, self.cutoff)
            self.polynomia.append(pol_order)

    def _init_weights(self) -> None:
        # This will have to be implemented later with a correct weight function
        n = int(self.ntypes) // 2
        weights = list(range(-n, n + 1))
        if self.ntypes % 2 == 0:
            weights.remove(0)
        self.weights = np.array([weights[int(i)] for i in self.itypes], dtype=np.float64)

    def _check_angular(self) -> bool:
        return not self.radial

    def _check_atomic(self) -> bool:
        return True

    def setup(self, atoms: Atoms | None = None) -> None:
        """"""
        if atoms is None:
            raise ValueError("atoms must be provided to setup()")
        elements = atoms.get_chemical_symbols()
        self.types = sorted(dict.fromkeys(elements))
        self.itypes = np.array([self.types.index(e) for e in elements], dtype=np.int64)
        self.ntypes = int(np.max(self.itypes)) + 1
        if self.compositional:
            self._init_weights()

    @property
    def order(self) -> npt.NDArray[np.int64]:
        return self.__order

    @order.setter
    def order(self, order: npt.NDArray[np.int64]) -> None:
        """docstring"""
        for i in order:
            if not 0 <= i < 1000:
                raise ValueError("order out of range")
        self.__order = order

    def _generate_keys(self) -> list[str]:
        """docstring"""
        self.__keys: list[str] = []
        typelabel = "R" if self.radial else "A"
        if self.structural:
            locallabel = "S"
            for order in self.order:
                self.__keys.append(f"{self.label}_{typelabel}{locallabel}_r{self.rcut}_{order:03d}")
        if self.compositional:
            locallabel = "C"
            for order in self.order:
                self.__keys.append(f"{self.label}_{typelabel}{locallabel}_r{self.rcut}_{order:03d}")
        return self.__keys

    def _compute_values(
        self,
        distances: npt.NDArray[np.float64],
        vectors: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """"""
        if self.radial:
            return self._calculate_radial(distances)
        else:
            return self._calculate_angular(distances, vectors)

    def _calculate_radial(
        self,
        distances: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """"""
        fci = self.fc(distances)
        mask = fci > 1.0e-6
        structural_coefficients = np.zeros(len(self.order), dtype=np.float64)
        composition_coefficients = np.zeros(len(self.order), dtype=np.float64)

        for i in range(len(self.order)):
            if self.structural:
                structural_coefficients[i] = np.sum(self.polynomia[i](distances[mask]) * fci[mask])
            if self.compositional:
                composition_coefficients[i] = np.sum(
                    self.polynomia[i](distances[mask]) * fci[mask] * self.weights[mask]
                )

        if self.structural and self.compositional:
            return np.concatenate((structural_coefficients, composition_coefficients))
        if self.structural:
            return structural_coefficients
        return composition_coefficients

    def _calculate_angular(
        self,
        distances: npt.NDArray[np.float64],
        vectors: npt.NDArray[np.float64] | None,
    ) -> npt.NDArray[np.float64]:
        if vectors is None:
            raise ValueError("vectors are required for angular ACSF")
        fci = self.fc(distances)
        structural_coefficients = np.zeros(len(self.order), dtype=np.float64)
        composition_coefficients = np.zeros(len(self.order), dtype=np.float64)

        # iterate through distance pairs and their corresponding vector pairs
        for j in fci.nonzero()[0]:
            rij = distances[j]
            rij_vec = vectors[j]
            for k in fci.nonzero()[0]:
                if k <= j:
                    continue
                rik = distances[k]
                rik_vec = vectors[k]
                angle = np.arccos(np.clip(np.dot(rij_vec, rik_vec) / (rij * rik), -1, 1))
                for i in range(len(self.order)):
                    if self.structural:
                        structural_coefficients[i] += self.polynomia[i](angle) * fci[j] * fci[k]
                    if self.compositional:
                        composition_coefficients[i] += (
                            self.polynomia[i](angle)
                            * fci[j]
                            * fci[k]
                            * self.weights[j]
                            * self.weights[k]
                        )

        if self.structural and self.compositional:
            return np.concatenate((structural_coefficients, composition_coefficients))
        if self.structural:
            return structural_coefficients
        return composition_coefficients
