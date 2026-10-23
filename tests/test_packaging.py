"""Gates on what the package pulls in and what it touches on import.

These are cheap, they need no Blender render, and each one exists because
something was actually wrong:

  * four dependencies were declared and never imported (netCDF4 drags in binary
    HDF5 wheels for nothing);
  * `import matplotlib.pyplot as plt` sat at module scope and `plt` was never
    used anywhere in the file;
  * `pio.renderers.default = 'browser'` ran AT IMPORT TIME, silently
    reconfiguring plotly for the whole process - anyone plotting in a notebook
    found their figures opening in a browser instead of inline;
  * the Earth's radius was hardcoded as 6371.0 in three places in raytracer.py
    while geometry.py used two other values.
"""
import ast
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
PKG = ROOT / 'satlight'


def _declared_dependencies():
    try:
        import tomllib
    except ModuleNotFoundError:                     # pragma: no cover
        pytest.skip('tomllib needs Python 3.11+')
    data = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    return [re.split(r'[=<>!~\[]', d)[0].strip()
            for d in data['project']['dependencies']]


def _imported_by_package():
    found = set()
    for f in sorted(PKG.glob('*.py')):
        for n in ast.walk(ast.parse(f.read_text())):
            if isinstance(n, ast.Import):
                found |= {a.name.split('.')[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                found.add(n.module.split('.')[0])
    return found


def test_every_declared_dependency_is_actually_imported():
    """A dependency nobody imports is a download every user pays for."""
    imported = _imported_by_package()
    unused = [d for d in _declared_dependencies() if d not in imported]
    assert unused == [], f'declared but never imported: {unused}'


def test_no_undeclared_third_party_imports():
    """The converse: importing something that is not declared is an ImportError
    waiting for the first user with a clean environment."""
    declared = set(_declared_dependencies())
    stdlib = set(sys.stdlib_module_names)
    local = {'satlight'}
    # lazily-imported extras are allowed, but must be declared as such
    try:
        import tomllib
        extras = tomllib.loads((ROOT / 'pyproject.toml').read_text())
        allowed = {re.split(r'[=<>!~\[]', d)[0].strip()
                   for v in extras['project'].get('optional-dependencies', {}).values()
                   for d in v}
    except Exception:                                # pragma: no cover
        allowed = set()
    undeclared = sorted(
        m for m in _imported_by_package()
        if m not in declared | stdlib | local | allowed and not m.startswith('_'))
    assert undeclared == [], f'imported but not declared anywhere: {undeclared}'


def test_importing_satlight_does_not_import_plotly_or_matplotlib():
    """Visualisation is optional, so it must not load just because someone
    imported the package. Run in a SUBPROCESS: this test session has almost
    certainly imported both already."""
    code = ('import satlight.raytracer, sys; '
            'print(int("plotly" in sys.modules), int("matplotlib" in sys.modules))')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr[-2000:]
    plotly_loaded, mpl_loaded = out.stdout.strip().split()[-2:]
    assert plotly_loaded == '0', 'importing satlight.raytracer pulled in plotly'
    assert mpl_loaded == '0', 'importing satlight.raytracer pulled in matplotlib'


def test_import_has_no_global_side_effects_on_plotly():
    """`pio.renderers.default = 'browser'` used to run at import time."""
    code = (
        'import plotly.io as pio; before = pio.renderers.default; '
        'import satlight.raytracer; '
        'print(before == pio.renderers.default)')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, cwd=ROOT)
    if out.returncode != 0:
        pytest.skip('plotly not installed')
    assert out.stdout.strip().endswith('True'), \
        'importing satlight.raytracer changed plotly\'s default renderer'


def test_the_earth_radius_is_single_sourced():
    """Three hardcoded 6371.0 in raytracer.py, plus two more values in
    geometry.py, is four Earth models in one package."""
    from satlight.geometry import R_EARTH_MEAN_KM
    assert R_EARTH_MEAN_KM == pytest.approx(6371.0)
    src = (PKG / 'raytracer.py').read_text()
    assert '6371' not in src, 'raytracer.py still hardcodes an Earth radius'


def test_no_unused_imports_in_the_package():
    """`plt` was imported and never used; nothing else should be."""
    offenders = []
    for f in sorted(PKG.glob('*.py')):
        src = f.read_text()
        tree = ast.parse(src)
        bound = {}
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    bound[a.asname or a.name.split('.')[0]] = n.lineno
            elif isinstance(n, ast.ImportFrom):
                for a in n.names:
                    if a.name != '*':
                        bound[a.asname or a.name] = n.lineno
        for name, lineno in bound.items():
            # count real uses: the binding itself contributes one occurrence
            if len(re.findall(rf'\b{re.escape(name)}\b', src)) <= 1:
                offenders.append(f'{f.name}:{lineno} {name}')
    assert offenders == [], f'imported but unused: {offenders}'
