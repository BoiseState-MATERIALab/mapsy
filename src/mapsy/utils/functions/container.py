from __future__ import annotations

from typing import Iterator, List, Union

from mapsy.data import Grid
from mapsy.data import ScalarField, GradientField, HessianField
from mapsy.utils.functions import FieldFunction


class FunctionContainer:
    """docstring"""

    def __init__(self, grid: Grid) -> None:
        self.grid = grid
        self.functions: List[FieldFunction] = []
        self.count = 0

    def __getitem__(
        self,
        slice: Union[int, List[int], slice],
    ) -> Union[FieldFunction, FunctionContainer]:

        if isinstance(slice, int):
            return self.functions[slice]
        else:
            subset = FunctionContainer(self.grid)

            if isinstance(slice, list):
                for i in slice:
                    subset.functions.append(self.functions[i])
            else:
                for function in self.functions[slice]:
                    subset.functions.append(function)

            return subset

    def __iter__(self) -> Iterator[FieldFunction]:
        return iter(self.functions)

    def __len__(self) -> int:
        return self.count

    def append(self, function: FieldFunction) -> None:
        """docstring"""
        self.functions.append(function)
        self.count += 1

    def reset_derivatives(self) -> None:
        """docstring"""
        for function in self:
            function.reset_derivatives()

    def density(self) -> ScalarField:
        """docstring"""
        density = ScalarField(self.grid)

        for function in self:
            density[:] += function.density

        return density

    def gradient(self) -> GradientField:
        """docstring"""
        gradient = GradientField(self.grid)

        for function in self:
            gradient[:] += function.gradient

        return gradient

    def laplacian(self) -> ScalarField:
        """docstring"""
        laplacian = ScalarField(self.grid)

        for function in self:
            laplacian[:] += function.laplacian

        return laplacian

    def hessian(self) -> HessianField:
        """docstring"""
        hessian = HessianField(self.grid)

        for function in self:
            hessian[:] += function.hessian

        return hessian

    def derivative(self) -> ScalarField:
        """docstring"""
        derivative = ScalarField(self.grid)

        for function in self:
            derivative[:] += function.derivative

        return derivative
