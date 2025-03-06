# Atom-Centered Symmetry Functions
from abc import ABC
from typing import Callable, List, Union, Tuple, Optional

from ase import Atoms
import numpy as np
import numpy.typing as npt
from numpy.polynomial.chebyshev import Chebyshev as cheb

from mapsy.symfunc.input import SymFuncModel
from mapsy.symfunc import SymmetryFunction
from mapsy.utils import cutoff


def wrapcheby(
    f: Callable,
    order: int,
    rcut: float,
) -> Callable:
    def wrapped(*args) -> float:
        return f(order, rcut, *args)

    return wrapped


def dual_basis_function(order: int, rcut: float, x: float):
    return cheb.basis(order)(2 * x / rcut - 1)


def basis_function(order: int, rcut: float, x: float):
    k = 1
    if order == 0: k = 2
    return (
        k
        / (2 * np.pi * np.sqrt(x / rcut - x**2 / rcut**2))
        / 4
        * rcut
        * cheb.basis(order)(2 * x / rcut - 1)
    )


class ACSFParser:

    def __init__(self, symfuncmodel: SymFuncModel) -> None:
        self.order = np.array(symfuncmodel.order,dtype=np.int64)
        self.cutoff = symfuncmodel.cutoff
        self.radius = symfuncmodel.radius
        self.radial = symfuncmodel.radial
        self.compositional = symfuncmodel.compositional
        self.structural = symfuncmodel.structural

    def parse(self) -> list[SymmetryFunction]:
        symmetryfunctions: list = []
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

    order: npt.NDArray[np.int64] = None
    radial: bool = True  # angular if False
    cutoff: float = 0.0
    fc: Callable = None
    rcut: float = 0.0

    compositional: bool = False
    weights: npt.NDArray[np.float64] = None
    structural: bool = False

    polynomia: List[Callable] = []

    types: list[str] = None
    itypes: npt.NDArray[np.int64] = None

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

        self.fc: Callable = cutoff(cutofftype, radius)
        self.rcut: float = radius

        self.compositional: bool = compositional
        self.structural: bool = structural
        self.radial: bool = radial
        if self.radial:
            if radius == 0:
                raise ValueError()
            self.cutoff: float = radius
        else:
            self.cutoff: float = np.pi
        self.polynomia: List[Callable] = []
        self._generate_functions()

    def _generate_functions(self) -> None:
        if self.radial : 
            function = basis_function
        else:
            function = dual_basis_function
        for order in self.order:
            pol_order = wrapcheby(function, order, self.cutoff)
            self.polynomia.append(pol_order)

    def _init_weights(self):
        # This will have to be implemented later with a correct weight function
        n = self.ntypes//2
        weights = list(range(-n,n+1))
        if self.ntypes%2 == 0:
            weights.remove(0)
        self.weights = np.array([ weights[i] for i in self.itypes ])

    def _check_angular(self) -> bool:
        return not self.radial

    def _check_atomic(self) -> bool:
        return True

    def setup(self, atoms: Optional[Atoms] = None) -> None:
        """"""
        elements = atoms.get_chemical_symbols()
        self.types = sorted(list(dict.fromkeys(elements)))
        self.itypes = np.array([self.types.index(e) for e in elements]).astype("int")
        self.ntypes = np.max(self.itypes).astype("int")+1
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
        self.__keys = []
        if self.radial:
            typelabel = "R"
        else:
            typelabel = "A"
        if self.structural:
            locallabel = "S"
            for order in self.order:
                self.__keys.append(
                    f"{self.label}_{typelabel}{locallabel}_r{self.rcut}_{order:03d}"
                )
        if self.compositional:
            locallabel = "C"
            for order in self.order:
                self.__keys.append(
                    f"{self.label}_{typelabel}{locallabel}_r{self.rcut}_{order:03d}"
                )
        return self.__keys

    def _compute_values(
        self,
        distances: npt.NDArray,
        vectors: Optional[npt.NDArray] = None,
    ) -> npt.NDArray:
        """"""
        if self.radial:
            return self._calculate_radial(distances)
        else:
            return self._calculate_angular(distances, vectors)

    def _calculate_radial(
        self,
        distances: npt.NDArray,
    ):
        """"""
        fci = self.fc(distances)
        mask = fci > 1.0e-6
        if self.structural:
            structural_coefficients = np.zeros(len(self.order))
        if self.compositional:
            composition_coefficients = np.zeros(len(self.order))

        for i,order in enumerate(self.order):
            if self.structural:
                structural_coefficients[i] = np.sum(
                    self.polynomia[i](distances[mask]) * fci[mask]
                )
            if self.compositional:
                composition_coefficients[i] = np.sum(
                    self.polynomia[i](distances[mask])
                    * fci[mask]
                    * self.weights[mask]
                )

        if self.structural:
            if self.compositional:
                return np.concatenate((structural_coefficients, composition_coefficients))
            return structural_coefficients
        if self.compositional:
            return composition_coefficients

    def _calculate_angular(self, distances, vectors):
        fci = self.fc(distances)
        if self.structural:
            structural_coefficients = np.zeros(len(self.order))
        if self.compositional:
            composition_coefficients = np.zeros(len(self.order))

        # iterate through distance pairs and their corresponding vector pairs
        for j in fci.nonzero()[0]:
            rij = distances[j]
            rij_vec = vectors[j]
            for k in fci.nonzero()[0]:
                if k <= j:
                    continue
                rik = distances[k]
                rik_vec = vectors[k]
                angle = np.arccos(
                    np.clip(np.dot(rij_vec, rik_vec) / (rij * rik), -1, 1)
                )
                for i,order in enumerate(self.order):
                    if self.structural:
                        structural_coefficients[i] += (
                            self.polynomia[i](angle)*fci[j]*fci[k]
                        )
                    if self.compositional:
                        composition_coefficients[i] += (
                            self.polynomia[i](angle)*fci[j]*fci[k]
                            * self.weights[j] * self.weights[k]
                        )

        if self.structural:
            if self.compositional:
                return np.concatenate((structural_coefficients, composition_coefficients))
            return structural_coefficients
        if self.compositional:
            return composition_coefficients
