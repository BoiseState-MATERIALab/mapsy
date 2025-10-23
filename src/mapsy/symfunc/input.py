from collections.abc import Callable, Iterator
from typing import Any, Literal, cast

from pydantic import (
    BaseModel as PydanticBaseModel,
)
from pydantic import (
    NonNegativeFloat,
)

from mapsy.utils.iotypes import (
    NonNegativeFloatList,
    NonNegativeIntList,
    int_list,
    list_ge_zero,
)

CutoffType = Literal[
    "cos",
    "tanh",
]

SFType = Literal[
    "bp",
    "ac",
    "cube",
]


class Order:
    @classmethod
    def __get_validators__(cls) -> Iterator[Callable[[Any], Any]]:
        yield cls.orderize
        yield int_list
        yield list_ge_zero

    @staticmethod
    def orderize(value: Any) -> list[int]:
        """Coerce an int or sequence into a list[int] of orders."""
        if isinstance(value, int):
            return list(range(value))
        if isinstance(value, list | tuple):
            return [int(v) for v in value]
        # Let pydantic raise on bad types
        raise TypeError(f"order must be int or sequence of ints, got {type(value)!r}")


class SymFuncBaseModel(PydanticBaseModel):
    """Global configurations of validation mechanics."""

    class Config:
        validate_assignment: bool = True


class SymFuncModel(SymFuncBaseModel):
    """Symmetry function input model."""

    type: SFType = "bp"
    cutoff: CutoffType = "cos"
    radius: NonNegativeFloat = 5.0
    # Allow int/list defaults at type level; validators will coerce to list[int]
    order: Order | int | list[int] = 1
    # BP Parameters
    # mypy needs these defaults to be of the annotated type; cast the literals.
    etas: NonNegativeFloatList = cast(NonNegativeFloatList, [1.0])
    rss: NonNegativeFloatList = cast(NonNegativeFloatList, [0.0])
    lambdas: list[float] = [-1.0, 1.0]
    kappas: NonNegativeFloatList = cast(NonNegativeFloatList, [1.0])
    zetas: NonNegativeIntList = cast(NonNegativeIntList, [1])
    # AC Parameters
    radial: bool = True
    compositional: bool = False
    structural: bool = False


class SymmetryFunctionsModel(SymFuncBaseModel):
    """Symmetry functions input model."""

    functions: list[SymFuncModel] = []
