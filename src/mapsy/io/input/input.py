from typing import (
    Any,
    Dict,
    Optional,
)

from pathlib import Path
from yaml import load, SafeLoader

from .base import (
    BaseModel,
    ControlModel,
    SystemModel,
    ContactSpaceModel,
)
from mapsy.symfunc.parser import SymmetryFunctionsModel


class Input(BaseModel):
    """
    Model for MapSy input.
    """

    control: Optional[ControlModel] = None
    system: Optional[SystemModel] = None
    contactspace: Optional[ContactSpaceModel] = None
    symmetryfunctions: Optional[SymmetryFunctionsModel] = None

    def __init__(
        self,
        filename: Optional[str] = None,
        **params: Dict[str, Any],
    ) -> None:

        # default parameter dictionary
        param_dict: Dict[str, Dict[str, Any]] = {
            "control": {},
            "system": {},
            "contactspace": {},
            "symmetryfunctions": {},
        }

        input_dict = {}

        if params:
            input_dict = params
        elif filename is not None:
            input_dict = self.read(filename)

        param_dict.update(input_dict)

        super().__init__(**param_dict)

        #        self.adjust_ionic_arrays(natoms)
        #
        if input_dict:
            self.sanity_check()

    def read(self, filename: str) -> Dict[str, Any]:
        """Read parameter dictionary from a YAML input file.

        Parameters
        ----------
        filename : str
            The name of the YAML input file

        Returns
        -------
        Dict[str, Any]
            A parameter dictionary
        """
        try:
            with open(Path(filename).absolute()) as f:
                return load(f, SafeLoader)
        except Exception:
            raise

    def sanity_check(self) -> None:
        """Check for bad input values."""
        self._validate_system()

    def _validate_system(self) -> None:
        """Validate system input"""
        if self.system.systemtype != "ions" and self.system.file.fileformat != "cube":
            raise ValueError("System mode electronic or full requires a cubefile")
