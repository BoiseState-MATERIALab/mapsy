from typing import Any

__all__ = [
    "BaseParser",
    "CubeParser",
    "XYZParser",
    "Input",
    "DataParser",
    "SystemParser",
    "ContactSpaceGenerator",
    "ResolvedFileRecord",
    "resolve_file_model",
    "resolve_file_records",
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
    if name in {
        "DataParser",
        "SystemParser",
        "ContactSpaceGenerator",
        "ResolvedFileRecord",
        "resolve_file_model",
        "resolve_file_records",
    }:
        from .parser import (
            ContactSpaceGenerator,
            DataParser,
            ResolvedFileRecord,
            SystemParser,
            resolve_file_model,
            resolve_file_records,
        )

        exports = {
            "DataParser": DataParser,
            "SystemParser": SystemParser,
            "ContactSpaceGenerator": ContactSpaceGenerator,
            "ResolvedFileRecord": ResolvedFileRecord,
            "resolve_file_model": resolve_file_model,
            "resolve_file_records": resolve_file_records,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
