"""A Python Tool to Compute Local Symmetry Maps"""

from contextlib import suppress
from typing import Any

__author__ = "MATERIALab"
__contact__ = "olivieroandreuss@boisestate.edu"
__license__ = "MIT"
__version__ = "0.0.1"  # fallback if package metadata isn't available
__date__ = "2024-05-24"

try:
    import importlib.metadata as _stdlib_metadata

    _md: Any = _stdlib_metadata
except ImportError:  # pragma: no cover
    import importlib_metadata as _backport_metadata

    _md = _backport_metadata

# Optional: define a default so __version__ always exists
# __version__ = "0.0.1"

with suppress(_md.PackageNotFoundError):
    __version__ = _md.version("mapsy")
