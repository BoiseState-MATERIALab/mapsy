from typing import Literal, Any, List

from pydantic import (
    NonNegativeFloat,
    BaseModel as PydanticBaseModel,
)
from mapsy.utils.iotypes import (NonNegativeFloatList, NonNegativeIntList, int_list, list_ge_zero,)

CutoffType = Literal[
    'cos',
    'tanh',
]

SFType = Literal[
    'bp',
    'ac',
    'cube',
]

class Order():

    @classmethod
    def __get_validators__(cls):
        yield cls.orderize
        yield int_list
        yield list_ge_zero

    def orderize(cls, value: Any) -> List[Any]:
        """"""
        if type(value) == int:
            value = list(range(value))
        elif type(value) == list:
            pass
        return value

class SymFuncBaseModel(PydanticBaseModel):
    """Global configurations of validation mechanics."""

    class Config:
        validate_assignment: bool = True

class SymFuncModel(SymFuncBaseModel):
    """Symmetry function input model."""

    type: SFType = "bp"
    cutoff: CutoffType = "cos"
    radius: NonNegativeFloat = 5.0  # type: ignore
    order: Order = 1
    # BP Parameters
    etas: NonNegativeFloatList = [1.0]  # type: ignore
    rss: NonNegativeFloatList = [0.0]  # type: ignore
    lambdas: list[float] = [-1.0, 1.0]
    kappas: NonNegativeFloatList = [1.0]  # type: ignore
    zetas: NonNegativeIntList = [1]  # type: ignore
    # AC Parameters
    radial: bool = True
    compositional: bool = False
    structural: bool = False

class SymmetryFunctionsModel(SymFuncBaseModel):
    """Symmetry functions input model."""

    functions: List[SymFuncModel] = []

