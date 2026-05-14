from pathlib import Path
from typing import Any, cast

from yaml import SafeLoader, load

from mapsy.symfunc.input import SymmetryFunctionsModel

from .base import (
    BaseModel,
    ContactSpaceModel,
    ControlModel,
    FileModel,
    SystemModel,
)


class Input(BaseModel):
    """
    Model for MapSy input.
    """

    control: ControlModel | None = None
    system: SystemModel | None = None
    contactspace: ContactSpaceModel | None = None
    symmetryfunctions: SymmetryFunctionsModel | None = None

    def __init__(
        self,
        filename: str | None = None,
        **params: dict[str, Any],
    ) -> None:
        # default parameter dictionary
        param_dict: dict[str, dict[str, Any]] = {
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

    def read(self, filename: str) -> dict[str, Any]:
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
                data = load(f, SafeLoader)  # returns Any
            if data is None:
                return {}
            return cast(dict[str, Any], data)
        except Exception:
            raise

    def sanity_check(self) -> None:
        """Check for bad input values."""
        self._validate_system()

    def _validate_system(self) -> None:
        """Validate system input"""
        sys = self.system
        if sys is None:
            return  # nothing to validate if no system provided
        self._validate_file(sys.file, "system")
        for prop in sys.properties or []:
            self._validate_file(prop.file, f"property {prop.name!r}")
        if sys.systemtype != "ions":
            file = sys.file
            if file is None or file.fileformat != "cube":
                raise ValueError("System mode 'electronic' or 'full' requires a cube file.")

    def _validate_file(self, file: FileModel | None, context: str) -> None:
        if file is None:
            return

        has_name = bool(file.name)
        has_names = bool(file.names)
        has_folder = bool(file.folder)
        has_folders = bool(file.folders)
        has_root = bool(file.root)
        has_pattern = bool(file.pattern)
        has_folder_input = has_folder or has_folders

        if has_name and has_names:
            raise ValueError(f"{context} file input must use either 'name' or 'names', not both.")
        if (has_name or has_names) and (has_folder_input or has_root or has_pattern):
            raise ValueError(
                f"{context} file input must use explicit file names or folder discovery, not both."
            )
        if has_folder and has_folders:
            raise ValueError(
                f"{context} file input must use either 'folder' or 'folders', not both."
            )
        if has_root and has_pattern:
            raise ValueError(f"{context} file input must use either 'root' or 'pattern', not both.")
        if has_folder_input and not (has_root or has_pattern):
            raise ValueError(f"{context} file input requires 'root' or 'pattern'.")
        if (has_root or has_pattern) and not has_folder_input:
            raise ValueError(f"{context} file input requires 'folder' or 'folders'.")
        if not (has_name or has_names or has_folder_input):
            raise ValueError(
                f"{context} file input requires 'name', 'names', 'folder', or 'folders'."
            )
