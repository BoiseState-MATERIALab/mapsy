# Behler-Parrinello Symmetry Functions
import numpy as np
import numpy.typing as npt
from typing import Callable, Optional
import itertools

from mapsy.symfunc.input import SymFuncModel
from mapsy.symfunc import SymmetryFunction
from mapsy.utils import cutoff


class BPSFParser:

    def __init__(self, symfuncmodel: SymFuncModel) -> None:
        self.cutoff = symfuncmodel.cutoff
        self.radius = symfuncmodel.radius

        self.order = np.array(symfuncmodel.order,dtype=np.int64)
        self.etas = np.array(symfuncmodel.etas)
        self.rss = np.array(symfuncmodel.rss)
        self.zetas = np.array(symfuncmodel.zetas)
        self.lambdas = np.array(symfuncmodel.lambdas)
        self.kappas = np.array(symfuncmodel.kappas)

    def parse(self) -> list[SymmetryFunction]:
        symmetryfunctions: list = []
        for i in self.order:
            symmetryfunctions.append(
                BPSymmetryFunction(
                    i+1,
                    self.radius,
                    self.cutoff,
                    self.etas,
                    self.rss,
                    self.lambdas,
                    self.kappas,
                    self.zetas,
                )
            )
        return symmetryfunctions


class BPSymmetryFunction(SymmetryFunction):

    order: int = 0
    rcut: float = 0.0
    fc: Callable = None

    etas: npt.NDArray[np.float64] = None
    rss: npt.NDArray[np.float64] = None
    lambdas: npt.NDArray[np.float64] = None
    kappas: npt.NDArray[np.float64] = None
    zetas: npt.NDArray[np.float64] = None

    types: list[str] = None
    itypes: npt.NDArray[np.int64] = None
    icombos: list = None

    def __init__(
        self,
        order: int,
        radius: float,
        cutofftype: str,
        etas: Optional[npt.NDArray[np.float64]] = None,
        rss: Optional[npt.NDArray[np.float64]] = None,
        lambdas: Optional[npt.NDArray[np.float64]] = None,
        kappas: Optional[npt.NDArray[np.float64]] = None,
        zetas: Optional[npt.NDArray[np.float64]] = None,
    ) -> None:
        super().__init__(kind=1, label="BP-G")
        self.order = order

        self.fc = cutoff(cutofftype, radius)
        self.rcut = radius

        if self.order == 2:
            self.etas = etas
            self.rss = rss
        elif self.order == 3:
            self.kappas = kappas
        elif self.order >= 4:
            self.etas = etas
            self.zetas = zetas
            self.lambdas = lambdas

    def setup(self, atoms):
        elements = atoms.get_chemical_symbols()
        self.types = sorted(list(dict.fromkeys(elements)))
        self.itypes = np.array([self.types.index(e) for e in elements]).astype("int")
        self.icombos = list(
            itertools.combinations_with_replacement(np.unique(self.itypes), 2)
        )

    @property
    def order(self) -> int:
        return self.__order

    @order.setter
    def order(self, order: int) -> None:
        """docstring"""
        if not 1 <= order <= 5:
            raise ValueError("order out of range")
        self.__order = order

    def _generate_keys(self) -> list[str]:
        """docstring"""
        self.__keys = []
        if self.order == 1:
            for elem in self.types:
                self.__keys.append(f"{self.label}1_{elem}_r{self.rcut}")
        elif self.order == 2:
            for eta, rs in zip(self.etas, self.rss):
                for elem in self.types:
                    self.__keys.append(
                        f"{self.label}2_{elem}_r{self.rcut}_eta{eta:3.2f}_rs{rs:3.2f}"
                    )
        elif self.order == 3:
            for kappa in self.kappas:
                for elem in self.types:
                    self.__keys.append(
                        f"{self.label}3_{elem}_r{self.rcut}_kappa{kappa}"
                    )
        elif self.order == 4:
            for zeta in self.zetas:
                for eta in self.etas:
                    for lambd in self.lambdas:
                        for combo in self.icombos:
                            self.__keys.append(
                                f"{self.label}4_{self.types[combo[0]]}_{self.types[combo[1]]}_r{self.rcut}_l{lambd}_z{zeta}_eta{eta:.2f}"
                            )
        elif self.order == 5:
            for zeta in self.zetas:
                for eta in self.etas:
                    for lambd in self.lambdas:
                        for combo in self.icombos:
                            self.__keys.append(
                                f"{self.label}5_{self.types[combo[0]]}_{self.types[combo[1]]}_r{self.rcut}_l{lambd}_z{zeta}_eta{eta:.2f}"
                            )
        return self.__keys

    def _check_angular(self) -> bool:
        return self.order >= 4

    def _check_atomic(self) -> bool:
        return True

    def _compute_values(
        self,
        distances: npt.NDArray,
        vectors: Optional[npt.NDArray],
    ) -> npt.NDArray:
        """docstring"""
        if self.order == 1:
            values = G1(self.fc, distances, self.itypes)
        elif self.order == 2:
            values = G2(self.fc, distances, self.etas, self.rss, self.itypes)
        elif self.order == 3:
            values = G3(self.fc, distances, self.kappas, self.itypes)
        elif self.order == 4:
            values = G4(
                self.fc,
                distances,
                vectors,
                self.zetas,
                self.etas,
                self.lambdas,
                self.itypes,
                self.icombos,
            )
        elif self.order == 5:
            values = G5(
                self.fc,
                distances,
                vectors,
                self.zetas,
                self.etas,
                self.lambdas,
                self.itypes,
                self.icombos,
            )
        return values.reshape(-1)


def G1(
    fc: Callable,
    distances: npt.NDArray[np.float64],
    types: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    ntypes: np.int64 = np.max(types) + 1
    g1: npt.NDArray[np.float64] = np.zeros(ntypes, dtype=np.float64)
    for itype in range(ntypes):
        rij: npt.NDArray[np.float64] = distances[types == itype]
        g1[itype] = np.sum(fc(rij))
    return g1


def G2(
    fc: Callable,
    distances: npt.NDArray[np.float64],
    etas: npt.NDArray[np.float64],
    rss: npt.NDArray[np.float64],
    types: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    ntypes: np.int64 = np.max(types) + 1
    g2: npt.NDArray[np.float64] = np.zeros((len(etas), ntypes), dtype=np.float64)
    for itype in range(ntypes):
        rij: npt.NDArray[np.float64] = distances[types == itype]
        for ieta, (eta, rs) in enumerate(zip(etas, rss)):
            g2[ieta, itype] = np.sum(np.exp(-eta * (rij - rs) ** 2) * fc(rij))
    return g2


def G3(
    fc: Callable,
    distances: npt.NDArray[np.float64],
    kappas: npt.NDArray[np.float64],
    types: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    ntypes: np.int64 = np.max(types) + 1
    g3: npt.NDArray[np.float64] = np.zeros((len(kappas), ntypes), dtype=float)
    for itype in range(ntypes):
        rij: npt.NDArray[np.float64] = distances[types == itype]
        for ikappa, kappa in enumerate(kappas):
            g3[ikappa, itype] = np.sum(np.cos(kappa * rij) * fc(rij))
    return g3


def G4(
    fc: Callable,
    distances: npt.NDArray[np.float64],
    vectors: npt.NDArray[np.float64],
    zetas: npt.NDArray[np.float64],
    etas: npt.NDArray[np.float64],
    lambdas: npt.NDArray[np.float64],
    types: npt.NDArray[np.int64],
    combos: list[tuple],
) -> npt.NDArray[np.float64]:
    ncombos: np.int64 = (np.unique(combos)).shape[0] + 1
    g4: npt.NDArray[np.float64] = np.zeros(
        (len(zetas), len(etas), len(lambdas), ncombos), dtype=np.float64
    )

    fci = fc(distances)
    nonzeros = fci.nonzero()[0]
    for j in nonzeros:
        rij = distances[j]
        rij_vec = vectors[j]
        for k in nonzeros:
            if k < j:
                continue
            rik = distances[k]
            rik_vec = vectors[k]

            rjk_vec = rij_vec - rik_vec
            rjk = np.linalg.norm(rjk_vec)

            costheta = np.dot(rij_vec, rik_vec) / (rij * rik)
            fc_fac = fci[j] * fci[k] * fc(rjk)
            exp_arg = rij**2 + rik**2 + rjk**2

            combo: tuple = tuple(sorted([types[j], types[k]]))
            if combo in combos:
                icombo: int = combos.index(combo)

            exp_fac = np.exp(-etas * exp_arg)
            for ilambda, lambd in enumerate(lambdas):
                ang_lin = (2.0 ** (1 - zetas)) * (1 + lambd * costheta) ** zetas
                g4[:, :, ilambda, icombo] += np.outer(ang_lin, exp_fac) * fc_fac

    return g4


def G5(
    fc: Callable,
    distances: npt.NDArray[np.float64],
    vectors: npt.NDArray[np.float64],
    zetas: npt.NDArray[np.float64],
    etas: npt.NDArray[np.float64],
    lambdas: npt.NDArray[np.float64],
    types: npt.NDArray[np.int64],
    combos: list[tuple],
) -> npt.NDArray[np.float64]:
    ncombos: np.int64 = (np.unique(combos)).shape[0] + 1
    g5: npt.NDArray[np.float64] = np.zeros(
        (len(zetas), len(etas), len(lambdas), ncombos), dtype=np.float64
    )

    fci = fc(distances)
    nonzeros = fci.nonzero()[0]
    for j in nonzeros:
        rij = distances[j]
        rij_vec = vectors[j]
        for k in nonzeros:
            if k < j:
                continue
            rik = distances[k]
            rik_vec = vectors[k]

            costheta = np.dot(rij_vec, rik_vec) / (rij * rik)
            g5_fc_fac = fci[j] * fci[k]

            combo: tuple = tuple(sorted([types[j], types[k]]))
            if combo in combos:
                icombo: int = combos.index(combo)

            g5_exp_fac = np.exp(-etas * (rij**2 + rik**2))
            for ilambda, lambd in enumerate(lambdas):
                ang_lin = (2.0 ** (1 - zetas)) * (1 + lambd * costheta) ** zetas
                g5[:, :, ilambda, icombo] += np.outer(ang_lin, g5_exp_fac) * g5_fc_fac

    return g5
