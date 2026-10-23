# Troubleshooting

## Installation

### `pip install satlight` says no matching distribution

satlight is a pre-release. Use `pip install --pre satlight`, or pin the exact
version. See {doc}`installation`.

### The install fails resolving `bpy==4.5.3`

Your Python is not 3.11. `bpy` wheels are built against one CPython ABI, so
satlight requires `>=3.11,<3.12`:

```bash
python --version
conda create -n satlight python=3.11 && conda activate satlight
```

### `ModuleNotFoundError: No module named '_bpy'`

Two distinct causes:

1. **In a child process** — you are using multiprocessing and the parent
   already imported `bpy`. `bpy` cannot be re-initialised in a spawned
   interpreter. Import satlight *inside* the worker function; see
   {doc}`guide/performance`.
2. **A broken install** — `pip install --force-reinstall bpy==4.5.3`.

## Renders

### Zero flux everywhere

Work down this list, in order — the first three are by far the most common.

**Is it observable at all?**
: ```python
  print(geom.eclipse_type(), geom.is_visible(), geom.observable_fraction())
  ```
  An eclipsed or occulted satellite returns **exactly** zero by design. This
  is a result, not a bug.

**Did you call `update()` before `render()`?**
: The renderer caches the camera placement and the earthshine term. Without an
  `update(geom)` it renders the geometry it was constructed with.

**Did you call `add_camera()`?**
: `initialise_scene()` imports the mesh; `add_camera()` creates and aims the
  camera. Without the second, there is nothing to cast rays from.

**Is `hit_count` zero?**
: ```python
  r.render(); print(r.hit_count, r.shadow_count)
  ```
  Zero hits with a non-zero `observable_fraction` means the camera sees no
  geometry — a failed mesh import, or a mesh at the wrong scale.

**Are the mesh normals inward-facing?**
: A BRDF returns exactly 0.0 when `dot(normal, light_dir) <= 0`, so an
  inverted mesh is uniformly black. `initialise_scene()` runs a normal repair,
  but it cannot fix a mesh that is consistently wound inside-out. Check in
  Blender with face orientation overlay on.

### Flux is wrong by a factor of ~10⁶

A kilometres/metres mix-up. `ViewGeometry(distance_km=...)` and
`Geometry.prop_distance` are both **kilometres**; the renderer converts once,
internally. A factor of 1000 in range is $10^6$ in flux — 15 magnitudes — and
the image looks entirely normal.

### Everything is far too bright

Almost always non-conserving BRDF parameters. The specular lobe **adds** to the
diffuse floor rather than being budgeted within a total, so `albedo=0.6,
specular=0.8` describes a surface reflecting more light than reaches it.

Integrate the hemisphere and check $\rho \le 1$. See the energy-conservation
section in {doc}`guide/materials`.

### Magnitudes are ~2.2 too bright

You used `tools.AB_mag` on broadband output. Use
`-2.5*np.log10(flux) - 18.915`. See {doc}`guide/photometry`.

### The body is rendered from nadir, not from the observer

Fixed in current versions — `add_camera` used to hardcode the $-Z$ axis. If you
see this, you are on an old version: upgrade, or make sure you are not calling
an internal placement method directly.

### Light curve structure that moves with attitude

Suspect frame fitting. With `auto_ortho_scale=False` the frame is sized for the
worst-case attitude, so an end-on view lands on far fewer pixels and reads
faint — an error that moves *with* attitude and can masquerade as real
structure. Leave `auto_ortho_scale=True` (the default).

### `RuntimeError: add_earthshine is True but no earthshine has been computed`

You set `add_earthshine = True` on an existing `Renderer`. The term is derived
from the geometry, so pass it at construction, or call
`compute_earthshine(geometry)` / `update(geometry)` afterwards. See
{doc}`guide/earthshine`.

### A warning about the body touching the frame edge

The fitted frame has clipped the body, so the flux is an underestimate. Raise
`ortho_margin` above its default 1.06.

### Monte-Carlo noise in a light curve

There should not be any: `bounce_seed` makes the hemisphere sampling
reproducible and the RNG is reset at the start of every `render()`. If you see
epoch-to-epoch jitter with `n_bounce > 1`, raise `bounce_samples` — noise falls
as $1/\sqrt{N}$ — and check you are not reseeding per frame yourself.

## Geometry

### `next_visibility` returns a pass that is entirely dark

It tests elevation only. Check `eclipse_type()` across each candidate.

### A `datetime` gave an unexpected epoch

Everything in `Geometry` is UTC, and a naive `datetime` is read as UTC — not as
local time. If your times are local, convert them first.

### `set_time` rejects my argument

Three forms are accepted: separate scalars, one tuple, or a `datetime`. For
many epochs, put a range in **one** component — not several.

## Performance

### Parallel rendering is slower than serial

Pin the BLAS threads before importing numpy. Unpinned, Skyfield's numpy calls
spread across every core and the workers fight each other. This is the
difference between 2.8× and 0.6×. See {doc}`guide/performance`.

### Each epoch takes as long as the first

You are rebuilding the `Renderer` per epoch and re-importing the mesh.
Construct once, then call `update(geom)` per epoch.

## Memory

- Reduce `resolution`.
- Simplify the mesh.
- Keep **one** `Renderer` alive at a time — `bpy` has a single global
  application state, and two renderers share one Blender scene.

## Still stuck?

Open an issue at
<https://github.com/lfbeesley/satlight/issues> with your satlight version
(`satlight.__version__`), your Python version, and the smallest script that
reproduces it.
