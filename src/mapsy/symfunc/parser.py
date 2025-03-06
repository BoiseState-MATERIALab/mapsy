from mapsy.symfunc.atomic import BPSFParser, ACSFParser, CubeSFParser
from mapsy.symfunc.input import SymmetryFunctionsModel
from mapsy.symfunc.symmetryfunction import SymmetryFunction

class SymmetryFunctionsParser:

    def __init__(self, sfsmodel: SymmetryFunctionsModel) -> None:

        self.symmfuncs: list = []
        for sf in sfsmodel.functions:
            if sf.type == "bp":
                self.symmfuncs.append(BPSFParser(sf))
            elif sf.type == "ac":
                self.symmfuncs.append(ACSFParser(sf))
            elif sf.type == "cube":
                self.symmfuncs.append(CubeSFParser(sf))
            else:
                raise ValueError("Unkonwn symmetry function type in input")

    def parse(self) -> list[SymmetryFunction]:
        symmetryfunctions: list[SymmetryFunction] = []
        for sf in self.symmfuncs:
            symmetryfunctions.extend(sf.parse())
        return symmetryfunctions