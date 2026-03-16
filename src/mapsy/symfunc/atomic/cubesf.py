from typing import TYPE_CHECKING

from ..symmetryfunction import SymmetryFunction

if TYPE_CHECKING:
    from ..input import SymFuncModel


class CubeSFParser:
    def __init__(self, symfuncmodel: "SymFuncModel") -> None:
        self.model = symfuncmodel

    def parse(self) -> list[SymmetryFunction]:
        # TODO: implement
        return []
