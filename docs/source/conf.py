# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import tomllib
from pathlib import Path

# -- Path setup ---------------------------------------------------------------
# satlight is imported for autodoc straight from the source tree rather than
# being pip-installed, because a real install would pull bpy==4.5.3 - several
# hundred MB of Blender - which Read the Docs has neither the time nor the disk
# for. bpy and mathutils are mocked below instead; see autodoc_mock_imports.
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

# -- Project information ------------------------------------------------------

project = "satlight"
author = "L. Beesley"
copyright = "2026, L. Beesley"

# Single source of truth. The package reads its version from the installed
# metadata, which is absent here (see above), so read the same value from
# pyproject.toml directly - the two can then never drift.
with open(_ROOT / "pyproject.toml", "rb") as _f:
    release = tomllib.load(_f)["project"]["version"]
version = ".".join(release.split(".")[:2])

# -- General configuration ----------------------------------------------------

extensions = [
    "myst_parser",                 # MyST Markdown
    "sphinx.ext.autodoc",          # docstrings from satlight source
    "sphinx.ext.napoleon",         # NumPy/Google docstring styles
    "sphinx.ext.viewcode",         # link to highlighted source
    "sphinx.ext.intersphinx",      # link out to numpy, astropy, python
    "sphinx.ext.mathjax",          # the radiometry needs real equations
    "sphinx_copybutton",           # copy button on every code block
    "sphinx_design",               # grids/cards on the landing page
]

myst_enable_extensions = [
    "colon_fence",      # ::: fences as an alternative to ```{}
    "deflist",
    "dollarmath",       # $...$ and $$...$$ inline maths
    "amsmath",          # align/gather environments
    "attrs_inline",
    "substitution",
    "linkify",          # bare URLs become links
]
myst_heading_anchors = 3       # auto anchors for ##/### so cross-page links work

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# -- autodoc ------------------------------------------------------------------

# Blender's Python API cannot be installed on the docs builder. Everything in
# satlight.raytracer is still documented: autodoc reads the source, and only
# the bpy/mathutils names resolve to mocks. hide_warnings is NOT mocked - it
# decorates a method, and a mocked decorator would replace that method with a
# Mock and lose its signature - so it is pip-installed instead (it is tiny).
autodoc_mock_imports = ["bpy", "mathutils"]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,       # an undocumented member is noise, not docs
    "show-inheritance": True,
    "member-order": "bysource",   # source order tells a story; alphabetical does not
}
autodoc_typehints = "description"
autoclass_content = "both"        # class docstring + __init__ docstring

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_rtype = False

# -- intersphinx --------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- HTML output --------------------------------------------------------------

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = f"satlight {release}"

# Dark pink, after the Pink Cat Boo VS Code theme. The same palette is given to
# Furo's light AND dark modes, so the site looks the same whatever the
# visitor's system setting; custom.css hides the now-pointless mode toggle.
_PINK_CAT_BOO = {
    "color-background-primary": "#202330",
    "color-background-secondary": "#2d2f42",
    "color-background-hover": "#3a3d55",
    "color-background-hover--transparent": "#3a3d5500",
    "color-background-border": "#3a3d55",
    "color-background-item": "#3a3d55",
    "color-foreground-primary": "#FFF0F5",
    "color-foreground-secondary": "#C5C8C6",
    "color-foreground-muted": "#9a9cb0",
    "color-foreground-border": "#6b6e88",
    "color-brand-primary": "#fe7c8e",
    "color-brand-content": "#fe7c8e",
    "color-brand-visited": "#DCBFF2",
    "color-highlighted-background": "#472541",
    "color-highlight-on-target": "#472541",
    "color-inline-code-background": "#2d2f42",
    "color-admonition-background": "#2d2f42",
    "color-table-header-background": "#2d2f42",
    "color-table-border": "#3a3d55",
    "color-card-background": "#2d2f42",
    "color-card-border": "#3a3d55",
    "color-api-name": "#fe7c8e",
    "color-api-pre-name": "#f3b4bf",
    "color-problematic": "#FA508C",
}
html_theme_options = {
    "source_repository": "https://github.com/lfbeesley/satlight/",
    "source_branch": "main",
    "source_directory": "docs/source/",
    "navigation_with_keys": True,
    "light_css_variables": _PINK_CAT_BOO,
    "dark_css_variables": _PINK_CAT_BOO,
}
pygments_style = "dracula"          # dark code blocks in both modes
pygments_dark_style = "dracula"
html_last_updated_fmt = "%Y-%m-%d"

# Nitpicky mode would flag every unresolved cross-reference. Worth having on,
# with the known-unresolvable third-party names listed rather than suppressed
# wholesale.
nitpick_ignore = [
    ("py:class", "numpy.ndarray"),
    ("py:class", "optional"),
]
