"""A Python Tool to Compute Local Symmetry Maps"""
import os

__author__ = "MATERIALab"
__contact__ = "olivieroandreuss@boisestate.edu"
__license__ = "MIT"
__version__ = "0.0.1"
__date__ = "2024-05-24"

try:
    from importlib.metadata import version # python >= 3.8
except Exception :
    try:
        from importlib_metadata import version
    except Exception :
        pass

try:
    __version__ = version("mapsy")
except Exception:
    pass

