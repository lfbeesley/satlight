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

## Python 3.11

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

Several hundred megabytes due to `bpy`. That is a whole Blender, including its renderer. 
The other dependencies are small:

| Package | Why |
|---|---|
| `bpy`, `mathutils` | ray casting, mesh handling, the scene graph |
| `numpy` | everything numerical |
| `skyfield`, `sgp4` | orbit propagation and observer geometry |
| `astropy` | physical constants and units |
| `plotly` | the 3-D preview, `preview_lvlh_views()` |
| `hide_warnings` | silences Blender's mesh-import chatter - it get's angry when you don't want to use an .obj's associated .mtl's|

## First run downloads an ephemeris

`Geometry()` needs a JPL planetary ephemeris for Sun and Moon positions. On
first use, Skyfield fetches `de421.bsp` (about 16 MB) into `~/.satlight/` and
reuses it from then on. The first `Geometry()` of your life therefore needs
network access; nothing after it does.

## Example code and models

Example code and `.obj` files can be found in the
[satlight repository](https://github.com/lfbeesley/satlight): scripts in
[`examples/`](https://github.com/lfbeesley/satlight/tree/main/examples), and
meshes, including the `spacex_ai1.obj` used in the {doc}`tutorial`, in
[`models/`](https://github.com/lfbeesley/satlight/tree/main/models). The pip
package contains only the code.


## Licence

GPL-3.0-or-later, and this is also not a free choice: satlight imports `bpy`
at module scope, and `bpy` is GPL-3.0, which makes a distributed satlight a
derivative work. Any redistribution, modified or not, must stay GPL-3.0.
