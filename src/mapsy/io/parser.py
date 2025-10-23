from mapsy.boundaries import Boundary, ContactSpace, IonicBoundary, SystemBoundary
from mapsy.data import Grid, ScalarField, System
from mapsy.io import BaseParser, CubeParser, XYZParser
from mapsy.io.input.base import (
    ContactSpaceModel,
    FileModel,
    SystemModel,
)
from mapsy.utils import setscalars


class DataParser:
    file: BaseParser | None = None
    name: str = ""
    label: str = ""

    def __init__(self, filemodel: FileModel, name: str = "data", label: str = "DAT") -> None:
        if filemodel.fileformat == "cube":
            self.file = CubeParser(
                filemodel.name,
                units=filemodel.units,
                hasdata=True,
            )
        else:
            raise ValueError(f"Unsupported data file format: {filemodel.fileformat!r}")
        self.name = name
        self.label = label

    def parse(self) -> ScalarField | None:
        if self.file is None:
            raise RuntimeError("DataParser is not configured with a file.")
        return self.file.dataparse(self.name, self.label)


class SystemParser:
    file: BaseParser | None = None
    propfiles: list["DataParser"]

    def __init__(self, systemmodel: SystemModel) -> None:
        readdata = systemmodel.systemtype != "ions"

        f = systemmodel.file
        if f is None:
            raise ValueError("SystemModel.file is required.")

        if f.fileformat == "xyz+":
            self.file = XYZParser(f.name, units=f.units)
        elif f.fileformat == "cube":
            self.file = CubeParser(f.name, units=f.units, hasdata=readdata)
        else:
            raise ValueError(f"Unsupported system file format: {f.fileformat!r}")

        self.dimension = systemmodel.dimension
        self.axis = systemmodel.axis

        self.propfiles = []
        for prop in systemmodel.properties or []:
            if prop.file is None:
                raise ValueError(f"Property {prop.name!r} has no file attached.")
            self.propfiles.append(DataParser(prop.file, prop.name, prop.label))

    def parse(self) -> System:
        if self.file is None:
            raise RuntimeError("SystemParser is not configured with a file.")
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
        self.scalars = setscalars(self.cell, self.cutoff)
        self.grid = Grid(scalars=self.scalars, cell=self.cell)
        self.system = system

        boundary: Boundary
        if self.mode == "system":
            boundary = SystemBoundary(
                mode="system",
                grid=self.grid,
                distance=self.distance,
                spread=self.spread,
                system=system,
            )
        elif self.mode == "ionic":
            boundary = IonicBoundary(
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

        self.boundary = boundary
        self.boundary.update()

        # If the system is 2D, we need to select the points that are on the correct side
        if self.system.dimension == 2:
            switch = ScalarField(self.grid)
            # Get the vector distance from the center of the system
            r, _ = self.grid.get_min_distance(
                self.system.center, self.system.dimension, self.system.axis
            )
            switch[:] = r[self.system.axis, :] * self.side > 0
            # Multiply boundary and gradient by the switch
            self.boundary.switch[:] *= switch
            self.boundary.gradient[:] *= switch

        contactspace: ContactSpace = ContactSpace(self.boundary, self.threshold)
        return contactspace
