from pydantic import (
    BaseModel as PydanticBaseModel,
)
from pydantic import NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt

from mapsy.io.input.keytypes import (
    ContactSpaceMode,
    FileFormat,
    RadiusMode,
    SystemType,
    Units,
)
from mapsy.utils.iotypes import (
    Axis,
    Dimensions,
    NonZeroFloat,
)


class BaseModel(PydanticBaseModel):
    """Global configurations of validation mechanics."""

    class Config:
        validate_assignment: bool = True


class FileModel(BaseModel):
    """File input model."""

    fileformat: FileFormat = "xyz+"
    name: str = ""
    names: list[str] = []
    folder: str = ""
    folders: list[str] = []
    root: str = ""
    pattern: str = ""
    recursive: bool = False
    units: Units = "bohr"


class PropertyModel(BaseModel):
    """System property model"""

    name: str = ""
    label: str = ""
    file: FileModel | None = None


class ControlModel(BaseModel):
    """Control input model."""

    debug: bool = False
    verbosity: NonNegativeInt = 0
    output: str = ""


class SystemModel(BaseModel):
    """System input model."""

    systemtype: SystemType = "ions"
    file: FileModel | None = None
    dimension: Dimensions = 2
    axis: Axis = 2
    properties: list[PropertyModel] = []


class ContactSpaceModel(BaseModel):
    """Contact space input model"""

    class Config(BaseModel.Config):
        extra = "forbid"

    mode: ContactSpaceMode = "system"
    radiusmode: RadiusMode = "muff"
    radiusfile: str | None = None
    alpha: PositiveFloat = 1.0
    spread: PositiveFloat = 0.5
    distance: PositiveFloat = 0
    cutoff: PositiveInt = 300
    threshold: NonZeroFloat = 0.1
    side: NonZeroFloat = 1.0
    core_epsilon: PositiveFloat = 1.0e-12
    core_tolerance: NonNegativeFloat | None = None
