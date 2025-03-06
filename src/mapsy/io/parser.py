from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import numpy.typing as npt

from mapsy.io.input.base import (
    SystemModel,
    FileModel,
    ContactSpaceModel,
)
from mapsy.io import BaseParser, XYZParser, CubeParser
from mapsy.data import System, ScalarField, Grid
from mapsy.boundaries import ContactSpace
from mapsy.boundaries import Boundary, IonicBoundary, SystemBoundary

from mapsy.utils.constants import BOHR_RADIUS_ANGS
from mapsy.utils import setscalars


class DataParser:

    file: BaseParser = None
    name: str = ""
    label: str = ""

    def __init__(self, filemodel: FileModel, name="data", label="DAT") -> None:

        if filemodel.fileformat == "cube":
            self.file = CubeParser(
                filemodel.name,
                units=filemodel.units,
                hasdata=True,
            )
        self.name = name
        self.label = label

    def parse(self) -> ScalarField:
        return self.file.dataparse(self.name, self.label)


class SystemParser:

    file: BaseParser = None
    propfiles: list = []

    def __init__(self, systemmodel: SystemModel) -> None:

        if systemmodel.systemtype == "ions":
            readdata = False
        else:
            readdata = True

        if systemmodel.file.fileformat == "xyz+":
            self.file = XYZParser(systemmodel.file.name, units=systemmodel.file.units)
        elif systemmodel.file.fileformat == "cube":
            self.file = CubeParser(
                systemmodel.file.name, units=systemmodel.file.units, hasdata=readdata
            )

        self.dimension = systemmodel.dimension
        self.axis = systemmodel.axis

        for prop in systemmodel.properties:
            self.propfiles.append(DataParser(prop.file, prop.name, prop.label))

    def parse(self) -> System:
        system = self.file.systemparse()
        system.dimension = self.dimension
        system.axis = self.axis
        for propfile in self.propfiles:
            system.addproperty(propfile.parse())
        return system



class ContactSpaceGenerator:

    def __init__(self, csmodel: ContactSpaceModel) -> None:

        self.mode = csmodel.mode
        self.cutoff = csmodel.cutoff
        self.threshold = csmodel.threshold
        self.side = csmodel.side

        self.spread = csmodel.spread
        if csmodel.mode == "system":
            self.distance = csmodel.distance
        elif csmodel.mode == "ionic":
            self.radiusmode = csmodel.radiusmode
            self.alpha = csmodel.alpha
        else:
            raise ValueError("Unkonwn contact space mode in input")

    def generate(self, system: System) -> ContactSpace:

        self.cell = system.grid.cell
        self.scalars = setscalars(self.cell,self.cutoff)
        self.grid = Grid(scalars=self.scalars, cell=self.cell)
        self.system = system

        if self.mode == "system":
            self.boundary: Boundary = SystemBoundary(
                mode="system",
                grid=self.grid,
                distance=self.distance,
                spread=self.spread,
                system=system,
            )
        elif self.mode == "ionic":
            self.boundary: Boundary = IonicBoundary(
                mode=self.radiusmode,
                grid=self.grid,
                alpha=self.alpha,
                softness=self.spread,
                system=system,
            )
        elif self.mode == "electronic":
            raise NotImplementedError("Electronic boundary not implemented yet")
        else:
            raise ValueError("Unkonwn contact space mode in input")

        self.boundary.update()

        # If the system is 2D, we need to select the points that are on the correct side
        if self.system.dimension == 2:
            switch = ScalarField(self.grid) 
            # Get the vector distance from the center of the system
            r, _ = self.grid.get_min_distance(self.system.center, self.system.dimension, self.system.axis)
            switch[:] = r[self.system.axis,:]*self.side > 0
            # Multiply boundary and gradient by the switch 
            self.boundary.switch[:] *= switch
            self.boundary.gradient[:] *= switch

        contactspace: ContactSpace = ContactSpace(self.boundary, self.threshold)
        return contactspace
