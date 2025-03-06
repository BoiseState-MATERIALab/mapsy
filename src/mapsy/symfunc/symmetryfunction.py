from abc import ABC, abstractmethod
from typing import Optional
import numpy.typing as npt
from ase import Atoms

KINDS = {
    1: "BP",
    2: "AC",
    3: "Fukui",
    4: "Environ",
}


class SymmetryFunction(ABC):

    def __init__(
        self,
        kind: int,
        label: Optional[str] = None,
    ) -> None:
        self.kind = kind
        self.label = label

        self.__atomic: bool = False

    @abstractmethod
    def setup(self, atoms: Optional[Atoms] = None) -> None:
        """docstring"""

    @property
    def kind(self) -> int:
        """docstring"""
        return self.__kind

    @kind.setter
    def kind(self, kind: int) -> None:
        """docstring"""
        if not 0 <= kind <= 4:
            raise ValueError("kind out of range")
        self.__kind = kind

    @property
    def atomic(self) -> bool:
        """docstring"""
        return self._check_atomic()

    @property
    def angular(self) -> bool:
        """docstring"""
        return self._check_angular()

    @property
    def keys(self) -> list[str]:
        """docstring"""
        return self._generate_keys()

    def values(
        self,
        distances: npt.NDArray,
        vectors: Optional[npt.NDArray] = None,
    ) -> npt.NDArray:
        """docstring"""
        return self._compute_values(distances, vectors)

    @abstractmethod
    def _compute_values(
        self,
        distances: npt.NDArray,
        vectors: Optional[npt.NDArray],
    ) -> npt.NDArray:
        """docstring"""

    @abstractmethod
    def _generate_keys(self) -> list[str]:
        """docstring"""

    @abstractmethod
    def _check_angular(self) -> bool:
        """docstring"""

    @abstractmethod
    def _check_atomic(self) -> bool:
        """docstring"""
