import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from mapsy.boundaries import Boundary, ContactSpace, IonicBoundary, SystemBoundary
from mapsy.data import Grid, ScalarField, System
from mapsy.utils import setscalars

from .base import BaseParser
from .cube import CubeParser
from .xyz import XYZParser

if TYPE_CHECKING:
    from mapsy.io.input.base import ContactSpaceModel, FileModel, SystemModel


def _resolve_path(pathname: str, basepath: str | Path | None = None) -> Path:
    path = Path(pathname).expanduser()
    if not path.is_absolute() and basepath is not None:
        path = Path(basepath).expanduser() / path
    return path.resolve()


def _file_pattern(filemodel: "FileModel") -> str:
    pattern = getattr(filemodel, "pattern", "")
    if pattern:
        return str(pattern)
    suffixes = {
        "xyz+": ".xyz",
        "cube": ".cube",
        "ase": "",
    }
    suffix = suffixes[filemodel.fileformat]
    return f"{filemodel.root}*{suffix}" if suffix else f"{filemodel.root}*"


@dataclass(frozen=True)
class ResolvedFileRecord:
    path: str
    source_file: str
    source_file_name: str
    source_file_stem: str
    source_folder: str
    source_folder_name: str
    source_folder_number: int | float | None

    def metadata(self) -> dict[str, str | int | float | None]:
        return {
            "source_file": self.source_file,
            "source_file_name": self.source_file_name,
            "source_file_stem": self.source_file_stem,
            "source_folder": self.source_folder,
            "source_folder_name": self.source_folder_name,
            "source_folder_number": self.source_folder_number,
        }


def _folder_number(folder_name: str) -> int | float | None:
    matches = re.findall(r"\d+(?:\.\d+)?", folder_name)
    if not matches:
        return None
    value = matches[-1]
    return float(value) if "." in value else int(value)


def _record_for_path(path: Path) -> ResolvedFileRecord:
    resolved = path.resolve()
    folder = resolved.parent
    return ResolvedFileRecord(
        path=str(resolved),
        source_file=str(resolved),
        source_file_name=resolved.name,
        source_file_stem=resolved.stem,
        source_folder=str(folder),
        source_folder_name=folder.name,
        source_folder_number=_folder_number(folder.name),
    )


def resolve_file_records(
    filemodel: "FileModel",
    basepath: str | Path | None = None,
) -> list[ResolvedFileRecord]:
    if filemodel.name:
        return [_record_for_path(_resolve_path(filemodel.name, basepath))]

    names = list(getattr(filemodel, "names", []) or [])
    if names:
        return [_record_for_path(_resolve_path(name, basepath)) for name in names]

    folders = list(getattr(filemodel, "folders", []) or [])
    folder = getattr(filemodel, "folder", None)
    if not folders and folder:
        folders = [folder]
    if not folders:
        raise ValueError("File input requires name, names, folder, or folders.")

    pattern = _file_pattern(filemodel)
    recursive = bool(getattr(filemodel, "recursive", False))
    records: list[ResolvedFileRecord] = []
    for folder_name in folders:
        folder = _resolve_path(folder_name, basepath)
        if not folder.exists():
            raise OSError(f"Input folder does not exist: {folder}")
        if not folder.is_dir():
            raise OSError(f"Input folder is not a directory: {folder}")

        iterator = folder.rglob(pattern) if recursive else folder.glob(pattern)
        matches = sorted(path for path in iterator if path.is_file())
        if not matches:
            raise OSError(
                f"No files matched pattern {pattern!r} in {folder} for format {filemodel.fileformat!r}"
            )
        records.extend(_record_for_path(path) for path in matches)
    return records


def resolve_file_model(filemodel: "FileModel", basepath: str | Path | None = None) -> list[str]:
    return [record.path for record in resolve_file_records(filemodel, basepath)]


class DataParser:
    file: BaseParser | None = None
    name: str = ""
    label: str = ""

    def __init__(
        self,
        filemodel: "FileModel",
        name: str = "data",
        label: str = "DAT",
        basepath: str | Path | None = None,
    ) -> None:
        filenames = resolve_file_model(filemodel, basepath)
        if len(filenames) != 1:
            raise ValueError("Data input must resolve to exactly one file.")
        if filemodel.fileformat == "cube":
            self.file = CubeParser(
                filenames[0],
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
    propfiles: list["DataParser"]

    def __init__(self, systemmodel: "SystemModel", basepath: str | Path | None = None) -> None:
        f = systemmodel.file
        if f is None:
            raise ValueError("SystemModel.file is required.")
        self.filemodel = f
        self.basepath = basepath
        self.readdata = systemmodel.systemtype != "ions"

        self.dimension = systemmodel.dimension
        self.axis = systemmodel.axis

        self.propfiles = []
        for prop in systemmodel.properties or []:
            if prop.file is None:
                raise ValueError(f"Property {prop.name!r} has no file attached.")
            self.propfiles.append(DataParser(prop.file, prop.name, prop.label, basepath))

    def parse(self) -> System:
        filenames = self.filenames()
        if len(filenames) != 1:
            raise ValueError("System input resolved to multiple files; use MultiMapsFromFile.")
        return self._parse_file(filenames[0])

    def parse_many(self) -> list[System]:
        return [self._parse_file(filename) for filename in self.filenames()]

    def filenames(self) -> list[str]:
        return resolve_file_model(self.filemodel, self.basepath)

    def file_records(self) -> list[ResolvedFileRecord]:
        return resolve_file_records(self.filemodel, self.basepath)

    def parse_file(self, filename: str) -> System:
        return self._parse_file(filename)

    def _parse_file(self, filename: str) -> System:
        file = self._build_file_parser(filename)
        system = file.systemparse()
        system.dimension = self.dimension
        system.axis = self.axis
        for propfile in self.propfiles:
            system.addproperty(propfile.parse())
        return system

    def _build_file_parser(self, filename: str) -> BaseParser:
        if self.filemodel.fileformat == "xyz+":
            return XYZParser(filename, units=self.filemodel.units)
        if self.filemodel.fileformat == "cube":
            return CubeParser(filename, units=self.filemodel.units, hasdata=self.readdata)
        raise ValueError(f"Unsupported system file format: {self.filemodel.fileformat!r}")


class ContactSpaceGenerator:
    def __init__(self, csmodel: "ContactSpaceModel") -> None:
        self.mode = csmodel.mode
        self.cutoff = csmodel.cutoff
        self.threshold = csmodel.threshold
        self.side = csmodel.side
        self.assign_layers = getattr(csmodel, "assign_layers", False)
        self.layer_switch_tolerance = getattr(csmodel, "layer_switch_tolerance", 0.25)
        self.layer_gradient_cosine_min = getattr(csmodel, "layer_gradient_cosine_min", 0.9)
        self.layer_orthogonality_tolerance = getattr(
            csmodel,
            "layer_orthogonality_tolerance",
            0.25,
        )

        self.spread = csmodel.spread
        if csmodel.mode == "system":
            self.distance = csmodel.distance
        elif csmodel.mode == "ionic":
            self.radiusmode = csmodel.radiusmode
            self.radiusfile = csmodel.radiusfile
            self.alpha = csmodel.alpha
        else:
            raise ValueError("Unkonwn contact space mode in input")

    def generate(
        self,
        system: System,
        radius_table_file: str | None = None,
    ) -> ContactSpace:
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
            user_radius_table = radius_table_file or self.radiusfile
            boundary = IonicBoundary(
                mode=self.radiusmode,
                grid=self.grid,
                alpha=self.alpha,
                softness=self.spread,
                system=system,
                radius_table_file=user_radius_table,
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

        contactspace: ContactSpace = ContactSpace(
            self.boundary,
            self.threshold,
            assign_layers=self.assign_layers,
            layer_switch_tolerance=self.layer_switch_tolerance,
            layer_gradient_cosine_min=self.layer_gradient_cosine_min,
            layer_orthogonality_tolerance=self.layer_orthogonality_tolerance,
        )
        return contactspace
