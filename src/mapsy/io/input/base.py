from typing import (
    List,
)

from pydantic import (
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    BaseModel as PydanticBaseModel,
)

from mapsy.io.input.keytypes import (
    SystemType,
    FileFormat,
    Units,
    ContactSpaceMode,
    RadiusMode,
)

from mapsy.utils.iotypes import (
    NonZeroFloat,
    Dimensions,
    Axis,
)


class BaseModel(PydanticBaseModel):
    """Global configurations of validation mechanics."""

    class Config:
        validate_assignment: bool = True


class FileModel(BaseModel):
    """File input model."""

    fileformat: FileFormat = "xyz+"
    name: str = ""
    units: Units = "bohr"


class PropertyModel(BaseModel):
    """System property model"""

    name: str = ""
    label: str = ""
    file: FileModel = None


class ControlModel(BaseModel):
    """Control input model."""

    debug: bool = False
    verbosity: NonNegativeInt = 0
    output: str = ""


class SystemModel(BaseModel):
    """System input model."""

    systemtype: SystemType = "ions"
    file: FileModel = None
    dimension: Dimensions = 2
    axis: Axis = 2
    properties: List[PropertyModel] = []


class ContactSpaceModel(BaseModel):
    """Contact space input model"""

    mode: ContactSpaceMode = "system"
    radiusmode: RadiusMode = "muff"
    alpha: PositiveFloat = 1.0
    spread: PositiveFloat = 0.5
    distance: PositiveFloat = 0
    cutoff: PositiveInt = 300
    threshold: NonZeroFloat = 0.1
    side: NonZeroFloat = 1.0

