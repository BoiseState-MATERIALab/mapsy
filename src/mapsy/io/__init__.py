from typing import Any

__all__ = [
    "BaseParser",
    "CubeParser",
    "XYZParser",
    "Input",
    "DataParser",
    "SystemParser",
    "ContactSpaceGenerator",
    "resolve_file_model",
]


def __getattr__(name: str) -> Any:
    if name in {"BaseParser"}:
        from .base import BaseParser

        return BaseParser
    if name in {"CubeParser"}:
        from .cube import CubeParser

        return CubeParser
    if name in {"XYZParser"}:
        from .xyz import XYZParser

        return XYZParser
    if name in {"Input"}:
        from .input import Input

        return Input
    if name in {"DataParser", "SystemParser", "ContactSpaceGenerator", "resolve_file_model"}:
        from .parser import ContactSpaceGenerator, DataParser, SystemParser, resolve_file_model

        exports = {
            "DataParser": DataParser,
            "SystemParser": SystemParser,
            "ContactSpaceGenerator": ContactSpaceGenerator,
            "resolve_file_model": resolve_file_model,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
