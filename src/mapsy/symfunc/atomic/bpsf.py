# Behler-Parrinello Symmetry Functions
import itertools
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from ase import Atoms

from mapsy.symfunc import SymmetryFunction
from mapsy.symfunc.input import SymFuncModel
from mapsy.utils import cutoff


class BPSFParser:
    def __init__(self, symfuncmodel: SymFuncModel) -> None:
        self.cutoff = symfuncmodel.cutoff
        self.radius = symfuncmodel.radius

        self.order = np.array(symfuncmodel.order, dtype=np.int64)
        self.etas = np.array(symfuncmodel.etas)
        self.rss = np.array(symfuncmodel.rss)
        self.zetas = np.array(symfuncmodel.zetas)
        self.lambdas = np.array(symfuncmodel.lambdas)
        self.kappas = np.array(symfuncmodel.kappas)

    def parse(self) -> list[SymmetryFunction]:
        symmetryfunctions: list[SymmetryFunction] = []
        for i in self.order:
            symmetryfunctions.append(
                BPSymmetryFunction(
                    i + 1,
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
    def __init__(
        self,
        order: int,
        radius: float,
        cutofftype: str,
        etas: npt.NDArray[np.float64] | None = None,
        rss: npt.NDArray[np.float64] | None = None,
        lambdas: npt.NDArray[np.float64] | None = None,
        kappas: npt.NDArray[np.float64] | None = None,
        zetas: npt.NDArray[np.float64] | None = None,
    ) -> None:
        super().__init__(kind=1, label="BP-G")
        self.order = order
        self.rcut: float = radius
        # fc(r) returns an array-like; annotate as callable over ndarray -> ndarray
        self.fc: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] = cutoff(
            cutofftype, radius
        )

        # Normalize optional arrays: keep attributes as ndarrays (possibly empty) to avoid Optional flow
        self.etas: npt.NDArray[np.float64] = (
            np.asarray(etas, dtype=np.float64)
            if etas is not None
            else np.empty(0, dtype=np.float64)
        )
        self.rss: npt.NDArray[np.float64] = (
            np.asarray(rss, dtype=np.float64) if rss is not None else np.empty(0, dtype=np.float64)
        )
        self.lambdas: npt.NDArray[np.float64] = (
            np.asarray(lambdas, dtype=np.float64)
            if lambdas is not None
            else np.empty(0, dtype=np.float64)
        )
        self.kappas: npt.NDArray[np.float64] = (
            np.asarray(kappas, dtype=np.float64)
            if kappas is not None
            else np.empty(0, dtype=np.float64)
        )
        self.zetas: npt.NDArray[np.float64] = (
            np.asarray(zetas, dtype=np.float64)
            if zetas is not None
            else np.empty(0, dtype=np.float64)
        )

        # Filled in during setup()
        self.types: list[str] = []
        self.itypes: npt.NDArray[np.int64] = np.empty(0, dtype=np.int64)
        self.icombos: list[tuple[int, int]] = []

    def setup(self, atoms: Atoms | None = None) -> None:
        if atoms is None:
            raise ValueError("atoms must be provided to setup()")
        elements = atoms.get_chemical_symbols()
        self.types = sorted(dict.fromkeys(elements))
        self.itypes = np.array([self.types.index(e) for e in elements], dtype=np.int64)
        self.icombos = list(itertools.combinations_with_replacement(np.unique(self.itypes), 2))

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
        if not self.types:
            raise RuntimeError("setup() must be called before generating keys")
        self.__keys: list[str] = []
        if self.order == 1:
            for elem in self.types:
                self.__keys.append(f"{self.label}1_{elem}_r{self.rcut}")
        elif self.order == 2:
            for eta, rs in zip(self.etas, self.rss, strict=False):
                for elem in self.types:
                    self.__keys.append(
                        f"{self.label}2_{elem}_r{self.rcut}_eta{eta:3.2f}_rs{rs:3.2f}"
                    )
        elif self.order == 3:
            for kappa in self.kappas:
                for elem in self.types:
                    self.__keys.append(f"{self.label}3_{elem}_r{self.rcut}_kappa{kappa}")
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
        distances: npt.NDArray[np.float64],
        vectors: npt.NDArray[np.float64] | None,
    ) -> npt.NDArray[np.float64]:
        """docstring"""
        if self.order == 1:
            values = G1(self.fc, distances, self.itypes)
        elif self.order == 2:
            values = G2(self.fc, distances, self.etas, self.rss, self.itypes)
        elif self.order == 3:
            values = G3(self.fc, distances, self.kappas, self.itypes)
        elif self.order == 4:
            if vectors is None:
                raise ValueError("vectors are required for order-4 symmetry functions")
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
            if vectors is None:
                raise ValueError("vectors are required for order-5 symmetry functions")
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
    fc: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    distances: npt.NDArray[np.float64],
    types: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    ntypes: int = int(np.max(types) + 1)
    g1: npt.NDArray[np.float64] = np.zeros(ntypes, dtype=np.float64)
    for itype in range(ntypes):
        rij: npt.NDArray[np.float64] = distances[types == itype]
        g1[itype] = np.sum(fc(rij))
    return g1


def G2(
    fc: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    distances: npt.NDArray[np.float64],
    etas: npt.NDArray[np.float64],
    rss: npt.NDArray[np.float64],
    types: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    ntypes: int = int(np.max(types) + 1)
    g2: npt.NDArray[np.float64] = np.zeros((len(etas), ntypes), dtype=np.float64)
    for itype in range(ntypes):
        rij: npt.NDArray[np.float64] = distances[types == itype]
        for ieta, (eta, rs) in enumerate(zip(etas, rss, strict=False)):
            g2[ieta, itype] = np.sum(np.exp(-eta * (rij - rs) ** 2) * fc(rij))
    return g2


def G3(
    fc: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    distances: npt.NDArray[np.float64],
    kappas: npt.NDArray[np.float64],
    types: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    ntypes: int = int(np.max(types) + 1)
    g3: npt.NDArray[np.float64] = np.zeros((len(kappas), ntypes), dtype=float)
    for itype in range(ntypes):
        rij: npt.NDArray[np.float64] = distances[types == itype]
        for ikappa, kappa in enumerate(kappas):
            g3[ikappa, itype] = np.sum(np.cos(kappa * rij) * fc(rij))
    return g3


def G4(
    fc: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    distances: npt.NDArray[np.float64],
    vectors: npt.NDArray[np.float64],
    zetas: npt.NDArray[np.float64],
    etas: npt.NDArray[np.float64],
    lambdas: npt.NDArray[np.float64],
    types: npt.NDArray[np.int64],
    combos: list[tuple[int, int]],
) -> npt.NDArray[np.float64]:
    ncombos: int = int(np.unique(combos).shape[0] + 1)
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
            fc_fac = fci[j] * fci[k] * fc(np.asarray([rjk], dtype=np.float64))
            exp_arg = rij**2 + rik**2 + rjk**2

            tj = int(types[j])
            tk = int(types[k])
            combo: tuple[int, int] = (tj, tk) if tj <= tk else (tk, tj)
            if combo not in combos:
                continue
            icombo: int = combos.index(combo)

            exp_fac = np.exp(-etas * exp_arg)
            for ilambda, lambd in enumerate(lambdas):
                ang_lin = (2.0 ** (1 - zetas)) * (1 + lambd * costheta) ** zetas
                g4[:, :, ilambda, icombo] += np.outer(ang_lin, exp_fac) * fc_fac

    return g4


def G5(
    fc: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    distances: npt.NDArray[np.float64],
    vectors: npt.NDArray[np.float64],
    zetas: npt.NDArray[np.float64],
    etas: npt.NDArray[np.float64],
    lambdas: npt.NDArray[np.float64],
    types: npt.NDArray[np.int64],
    combos: list[tuple[int, int]],
) -> npt.NDArray[np.float64]:
    ncombos: int = int(np.unique(combos).shape[0] + 1)
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

            tj = int(types[j])
            tk = int(types[k])
            combo: tuple[int, int] = (tj, tk) if tj <= tk else (tk, tj)
            if combo not in combos:
                continue
            icombo: int = combos.index(combo)

            g5_exp_fac = np.exp(-etas * (rij**2 + rik**2))
            for ilambda, lambd in enumerate(lambdas):
                ang_lin = (2.0 ** (1 - zetas)) * (1 + lambd * costheta) ** zetas
                g5[:, :, ilambda, icombo] += np.outer(ang_lin, g5_exp_fac) * g5_fc_fac

    return g5
