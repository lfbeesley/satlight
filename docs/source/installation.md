# Installation

```{code-block} bash
pip install --pre satlight
```

satlight is published as a **pre-release**, so `pip` has to be told to accept
one. A plain `pip install satlight` reports *"no matching distribution found"*,
which looks like the package does not exist. Either `--pre` or an exact pin
works:

```bash
pip install "satlight==0.1.0a3"
```

## Python 3.11, and only 3.11

```{important}
satlight requires `>=3.11,<3.12`. This is not conservatism — it is `bpy`.
```

The ray casting is done by Blender, imported as a Python module
(`bpy==4.5.3`). `bpy` wheels are built against one specific CPython ABI, so
the interpreter version is fixed by whichever Blender release you pin. On 3.12
or 3.13 the install fails at dependency resolution rather than at runtime.

A clean environment is the path of least resistance:

```bash
conda create -n satlight python=3.11
conda activate satlight
pip install --pre satlight
```

## The download is large

Expect several hundred megabytes, almost all of it `bpy`. That is a whole
Blender, including its renderer. The other dependencies are small:

| Package | Why |
|---|---|
| `bpy`, `mathutils` | ray casting, mesh handling, the scene graph |
| `numpy` | everything numerical |
| `skyfield`, `sgp4` | orbit propagation and observer geometry |
| `astropy` | physical constants and units |
| `hide_warnings` | silences Blender's mesh-import chatter |

```{note}
Earlier versions declared `trimesh`, `tqdm`, `xarray` and `netCDF4`. None were
ever imported — they were left over from an earthshine path that was never
finished, and `netCDF4` alone dragged in binary HDF5 wheels for nothing. They
are gone as of `0.1.0a3`, which removes about 177 MB from a fresh install.
```

### Optional extras

The 3-D preview (`Renderer.preview_lvlh_views`) needs plotly and matplotlib.
They are imported lazily, inside that one method, so the core package neither
needs nor loads them:

```bash
pip install --pre "satlight[preview]"
```

## First run downloads an ephemeris

`Geometry()` needs a JPL planetary ephemeris for Sun and Moon positions. On
first use, Skyfield fetches `de421.bsp` (about 16 MB) into `~/.satlight/` and
reuses it from then on. The first `Geometry()` of your life therefore needs
network access; nothing after it does.

## Bundled data

One example model ships with the package, so the {doc}`tutorial` runs as-is
with nothing to download:

The package itself is the code only: no models and no data files. Example
meshes, including the `spacex_ai1.obj` used in the {doc}`tutorial`, are in the
repository's [`models/`](https://github.com/lfbeesley/satlight/tree/main/models)
directory.

```{warning}
satlight does **not** ship a BRDF measurement database. Surface properties are
supplied as analytic BRDF parameters — see {doc}`guide/materials`. Older
documentation referred to a bundled MERL database; it was removed, and no part
of the package reads it.
```

## Verifying the install

```bash
python -c "import satlight; print(satlight.__version__)"
python -c "from satlight import Renderer, Geometry; print('ok')"
```

The second import is the real test: it is the one that loads `bpy`.

## Development install

```bash
git clone https://github.com/lfbeesley/satlight.git
cd satlight
pip install -e .
pip install pytest
pytest                     # fast offline tier, no Blender, no network
```

See {doc}`contributing` for the test tiers — the default `pytest` run
deliberately skips anything needing Blender or an ephemeris.

## Licence

GPL-3.0-or-later, and this is also not a free choice: satlight imports `bpy`
at module scope, and `bpy` is GPL-3.0, which makes a distributed satlight a
derivative work. Any redistribution, modified or not, must stay GPL-3.0.
