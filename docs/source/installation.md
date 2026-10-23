# Installation

```{code-block} bash
pip install --pre satlight
```

satlight is a pre-release, so `pip` needs `--pre`, or an exact version:

```bash
pip install "satlight==0.1.0a3"
```

## Python 3.11

satlight requires Python 3.11, because `bpy==4.5.3` (Blender as a Python
module) is only built for 3.11. In a new environment:

```bash
conda create -n satlight python=3.11
conda activate satlight
pip install --pre satlight
```

## The download is large

Several hundred megabytes, almost all of it `bpy`. The other dependencies are
small:

| Package | Why |
|---|---|
| `bpy`, `mathutils` | ray casting, mesh handling, the scene graph |
| `numpy` | everything numerical |
| `skyfield`, `sgp4` | orbit propagation and observer geometry |
| `astropy` | physical constants and units |
| `plotly` | the 3-D preview, `preview_lvlh_views()` |
| `hide_warnings` | silences Blender's mesh-import chatter - it gets angry when you don't want to use an .obj's associated .mtl files |

## First run downloads an ephemeris

`Geometry()` needs a JPL planetary ephemeris for Sun and Moon positions. On
first use, Skyfield fetches `de421.bsp` (about 16 MB) into `~/.satlight/` and
reuses it from then on, so the first run needs network access.

## Example code and models

Example code and `.obj` files can be found in the
[satlight repository](https://github.com/lfbeesley/satlight): scripts in
[`examples/`](https://github.com/lfbeesley/satlight/tree/main/examples), and
meshes, including the `spacex_ai1.obj` used in the {doc}`tutorial`, in
[`models/`](https://github.com/lfbeesley/satlight/tree/main/models). The pip
package contains only the code.


## Licence

GPL-3.0-or-later, because satlight imports `bpy`, which is GPL-3.0.
