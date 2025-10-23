from collections.abc import Callable, Iterator
from typing import (
    Annotated,
    Any,
)

from pydantic import (
    confloat,
    conint,
)
from pydantic.errors import (
    NumberNotGeError,
    NumberNotGtError,
)
from pydantic.validators import (
    float_validator,
    int_validator,
)

Validator = Callable[[Any], Any]

# numerical type aliases
IntFloat = int | float
IntGT1 = Annotated[int, conint(gt=1)]
FloatGE1 = Annotated[float, confloat(ge=1)]
NonZeroFloat = Annotated[float, confloat(le=-1.0e-6, ge=1.0e-6)]
Dimensions = Annotated[int, conint(ge=0, le=3)]
Axis = Annotated[int, conint(ge=0, le=2)]

# yapf: enable


def int_list(value: list[Any]) -> list[int]:
    """Convert items to integers.

    Parameters
    ----------
    value : List[Any]
        A list of items

    Returns
    -------
    List[int]
        A validated list of integers
    """
    return [int_validator(v) for v in value]


def float_list(value: list[Any]) -> list[float]:
    """Convert items to floats.

    Parameters
    ----------
    value : List[Any]
        A list of items

    Returns
    -------
    List[float]
        A validated list of floats
    """
    return [float_validator(v) for v in value]


def list_ge_zero(value: list[IntFloat]) -> list[IntFloat]:
    """Check that all array values are greater than or equal to zero.

    Parameters
    ----------
    value : List[IntFloat]
        A list of integers or floats

    Returns
    -------
    List[IntFloat]
        A list of integers or floats greater than or equal to zero
    """
    for v in value:
        if v < 0:
            raise NumberNotGeError(limit_value=0)
    return value


def list_gt_zero(value: list[IntFloat]) -> list[IntFloat]:
    """Check that all array values are positive.

    Parameters
    ----------
    value : List[IntFloat]
        A list of integers or floats

    Returns
    -------
    List[IntFloat]
        A list of integers or floats greater than zero
    """
    for v in value:
        if v <= 0:
            raise NumberNotGtError(limit_value=0)
    return value


def ne_zero(value: IntFloat) -> IntFloat:
    """Check that value is non-zero.

    Parameters
    ----------
    value : IntFloat
        An integer or float

    Returns
    -------
    IntFloat
        A non-zero integer or float
    """
    if value == 0:
        raise ValueError("ensure this value is not zero")
    return value


class NonNegativeFloatList(list):
    @classmethod
    def __get_validators__(cls) -> Iterator[Validator]:
        yield float_list
        yield list_ge_zero


class PositiveFloatList(list):
    @classmethod
    def __get_validators__(cls) -> Iterator[Validator]:
        yield float_list
        yield list_gt_zero


class NonNegativeIntList(list):
    @classmethod
    def __get_validators__(cls) -> Iterator[Validator]:
        yield int_list
        yield list_ge_zero


class PositiveIntList(list):
    @classmethod
    def __get_validators__(cls) -> Iterator[Validator]:
        yield int_list
        yield list_gt_zero


class Vector(list):
    @classmethod
    def __get_validators__(cls) -> Iterator[Validator]:
        yield cls.vectorize

    @classmethod
    def vectorize(cls, value: list[Any]) -> list[Any]:
        """Scale vector input to 3D.

        Parameters
        ----------
        value : List[Any]
            A list of items

        Returns
        -------
        List[Any]
            A 3D vector of items
        """
        if len(value) == 1:
            value = value * 3
        assert len(value) == 3, "array size should be 3"
        return value


class FloatVector(Vector):
    @classmethod
    def __get_validators__(cls) -> Iterator[Validator]:
        yield cls.vectorize
        yield float_list


class NonNegativeIntVector(Vector):
    @classmethod
    def __get_validators__(cls) -> Iterator[Validator]:
        yield cls.vectorize
        yield int_list
        yield list_ge_zero
