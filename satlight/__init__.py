"""
Satlight - Satellite lightcurve modelling using Blender raytracing
"""

# single source of truth: the version lives in pyproject.toml and is read back
# from the installed metadata, so the two can never drift apart
from importlib.metadata import version as _version, PackageNotFoundError as _PNF
try:
    __version__ = _version("satlight")
except _PNF:                      # running from a source tree, not installed
    __version__ = "0.0.0.dev0"

from .raytracer import *
from .brdf import *
from .geometry import *
from .tools import *
