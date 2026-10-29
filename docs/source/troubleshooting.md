# Troubleshooting

## Installation

### `pip install satlight` says no matching distribution

satlight is a pre-release. Use `pip install --pre satlight`. See
{doc}`installation`.

### The install fails resolving `bpy==4.5.3`

`bpy` needs Python 3.11:

```bash
conda create -n satlight python=3.11 && conda activate satlight
```

### `ModuleNotFoundError: No module named '_bpy'`

If you are using multiprocessing, import satlight inside the worker function,
not in the parent process (see {doc}`guide/performance`). Otherwise, reinstall:
`pip install --force-reinstall bpy==4.5.3`.

## Renders

### The flux is zero

Check:

1. Is the satellite observable?
   `print(geom.eclipse_type(), geom.is_visible())`. An eclipsed or occulted
   satellite gives exactly zero.
2. Did you call `r.update(geom)` before `r.render()`?
3. Did any rays hit? `print(r.hit_count)` after rendering. Zero means the mesh
   didn't load or is at the wrong scale.
4. Are the mesh normals pointing inwards? Check in Blender with the face
   orientation overlay.

### The flux is wrong by about 10⁶

`ViewGeometry(distance_km=...)` and `Geometry.prop_distance` are in
kilometres, not metres.

### `RuntimeError: add_earthshine is True but no earthshine has been computed`

Pass `add_earthshine=True` when creating the `Renderer`, or call
`r.update(geom)` after setting it.

### A warning that the body touches the frame edge

Part of the spacecraft is outside the image, so the flux is too low. Increase
`ortho_margin`, or call `update()` after moving parts.

## Geometry

### `next_visibility` returns a pass that is in eclipse

It checks elevation only. Use `eclipse_type()` to find the sunlit part.

### Times are off by some hours

`Geometry` works in UTC, and a `datetime` without a time zone is read as UTC.

## Getting help

Open an issue at <https://github.com/lfbeesley/satlight/issues> with your
satlight version (`satlight.__version__`), Python version, and a short script
that reproduces the problem.
