# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup --------------------------------------------------------------
# Make satlight importable for autodoc (points at the repo root, two levels
# up from docs/source/).
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information ------------------------------------------------------

project = "satlight"
copyright = "2026, L. Beesley"
author = "L. Beesley"
release = "0.1.0a1"

# -- General configuration ----------------------------------------------------

extensions = [
    "myst_parser",              # MyST Markdown support
    "sphinx.ext.autodoc",       # Pull docstrings from satlight source
    "sphinx.ext.napoleon",      # NumPy/Google-style docstring support
    "sphinx.ext.viewcode",      # Link to highlighted source
    "sphinx.ext.intersphinx",   # Link to external docs (numpy, astropy, etc.)
]

myst_enable_extensions = [
    "colon_fence",   # Allows ::: fenced directives as an alternative to ```{}
    "deflist",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# bpy (Blender's Python API) isn't installable on Read the Docs' build image.
# Mock it so autodoc can still import satlight.raytracer without it.
autodoc_mock_imports = ["bpy", "mathutils"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Options for HTML output --------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Intersphinx mapping -------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
}
