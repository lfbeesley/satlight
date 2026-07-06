# Installation

Install `satlight` from PyPI:

```bash
pip install satlight
```

This pulls in the required dependencies automatically: `numpy`, `matplotlib`,
`astropy`, `skyfield`, `sgp4`, `trimesh`, `tqdm`, `plotly`, `xarray`, and
`bpy` (Blender's Python API, used for ray tracing).

```{note}
`bpy` is a large package. If you only need to build the docs (not run
renders), you don't need it installed in your docs build environment.
```

## Data files

Some larger data files are not bundled with the pip package:

- The BRDF material reflectance database (`satlight/data/BRDFDatabase/`)
- Example 3D satellite models (`satlight/data/models/`)

Contact the maintainer for access, and place them under `satlight/data/`
after installing the package.

## Development install

To work on `satlight` itself:

```bash
git clone https://github.com/lfbeesley/satlight.git
cd satlight
pip install -e .
```
