# The renderer

`Renderer` is an orthographic ray caster wrapped around Blender's BVH. It
casts one ray per pixel toward the spacecraft, evaluates a BRDF at each hit,
and sums the result into an irradiance at the observer.

This page is the order of operations, and what each step assumes.

## The radiometry in one equation

Each lit pixel contributes

$$
\Delta E = \underbrace{f_r \cos\theta_i}_{\text{BRDF}} \;\cdot\;
           \underbrace{E_\odot}_{\substack{\text{solar}\\ \text{constant}}} \;\cdot\;
           \frac{A_{\text{pix}}}{d^2}
$$

and `render()` returns the per-pixel array, so `np.sum(r.render())` is the
total irradiance at the observer in W/m².

The whole of the geometric scaling is the single factor

```python
k_geom = self.pixel_area / self.distance_to_observer ** 2
```

```{important}
`pixel_area` **is** the radiometry. Per-pixel flux scales with it directly, so
it can never be left stale behind a change of frame width — which is why
`_set_pixel_scale` is called every time `ortho_scale` moves, rather than once
at construction.
```

`distance_to_observer` is in **metres**. `Geometry.prop_distance` is in
kilometres, and the conversion happens in exactly one place
(`_frame_from_geometry`) so that it cannot be done twice or not at all.

## The call sequence

```python
r = Renderer(obj_path, geometry=geom, brdf=brdf)
r.initialise_scene()          # import the mesh, centre it, repair normals
r.add_camera()                # create the ortho camera, aim it, fit the frame
r.set_attitude(euler_xyz=...) # optional; before update()
r.update(geom)                # re-read the geometry; re-fit camera & earthshine
image = r.render()            # the ray cast
```

`initialise_scene` and `add_camera` are called once. `update` is called **once
per epoch**, before every render.

```{warning}
Call `set_attitude` *before* `update()`, so the camera is fitted around the new
orientation rather than the old one.
```

## Why the camera is orthographic

The spacecraft subtends microradians from the ground. A perspective camera at
the true range would give every ray essentially the same direction while
costing numerical precision, so the renderer uses a parallel projection and
carries the $1/d^2$ falloff analytically instead. This is exact in the limit
that matters and avoids the whole problem.

### Frame fitting

By default (`auto_ortho_scale=True`) the orthographic frame is re-fitted to
the body's **projected** extent on every `update`. The alternative —
`auto_ortho_scale=False`, which fixes the frame to the bounding-box diagonal —
sizes for the worst-case attitude and wastes resolution on every other one.

```{note}
Concretely: viewed end-on, Skynet 5D lands on 69 pixels instead of 469 and
reads 0.070 mag too faint. That error moves *with* attitude, so in a light
curve it can masquerade as real structure. This is why auto-fitting is the
default.
```

Frame width and camera standoff are two unrelated quantities: width comes from
the projected extent, standoff from the along-view extent. `ortho_margin=1.06`
puts a 6% border around the fitted frame.

```{note}
`add_camera` once hardcoded the $-Z$ axis, so
`initialise_scene → add_camera → render` silently rendered from **nadir**
whatever geometry it was given — and for an oblique observer that meant looking
into the back of the model and returning zero flux. Both paths now go through
one `_place_camera`.
```

## Attitude

```python
import numpy as np
r.set_attitude(euler_xyz=(0.0, 0.0, np.radians(30.0)))
```

Angles are in **radians** about the [LVLH axes](geometry.md#the-lvlh-frame) — X
along-track, Y orbit normal, Z zenith — so `(0, 0, 0)` is the mesh exactly as
imported.

```{important}
The rotation is always applied to the **imported** pose, never to the current
one, so calling `set_attitude` twice does not compound. It is safe to call once
per frame with an absolute orientation, which is what an attitude profile
gives you.
```

## Shadows

`add_shadows=True` (the default) lets the body shadow itself: one shadow ray
per lit pixel, toward the Sun. Without it, a pixel on a surface facing the Sun
is lit even if an antenna stands between it and the Sun.

The cost is roughly one extra ray cast per lit pixel — close to a doubling of
the direct term, and usually worth it. Concave geometry (louvres, box
cavities, antenna stand-offs) is where it changes the answer most.

## Multiple scattering

`n_bounce` is the number of scattering events per path.

| `n_bounce` | Meaning | Extra rays per lit pixel |
|---|---|---|
| 1 (default) | single bounce | none |
| 2 | one inter-reflection | `bounce_samples` |
| 3 | two | `bounce_samples²` (64× at the default 8) |
| 4 | — | not worth attempting |

`n_bounce=2` matters for concave geometry that sees itself, and captures most
of a series converging like $(\rho F)^k$. Directions are cosine-weighted
hemisphere samples; noise falls as $1/\sqrt{N}$, and `bounce_samples=8` is
enough unless the indirect term is a large fraction of the total.

```{note}
`bounce_seed` makes the sampling reproducible, and the RNG is reset at the
start of every `render()`, so a given geometry always renders identically.
Monte-Carlo noise never appears as spurious temporal structure in a light
curve.
```

## Shading normals

`smooth_normals=True` interpolates shading normals across edges meeting at
less than `smooth_angle_deg` (default 30°).

```{warning}
Smoothing **redistributes** light over the disk; it does not change the total.
It is a presentation choice, not an accuracy improvement — on a coarsely
tessellated sphere it measurably makes the body *darker*, not more accurate.
If you need accuracy on a curved surface, refine the mesh.
```

## Observability

`apply_observability=True` (the default) scales the rendered flux by the
geometry's `observable_fraction()` — that is `eclipse_type() × is_visible()`.

Without it, a satellite that has **set** still renders at full brightness,
because the renderer sees only three direction vectors and a range, and none
of them knows where the Earth is.

Two efficiencies fall out of this. `render()` returns a zero array immediately
when `observable_fraction == 0.0`, without spending a frame's ray casts
discovering that — which for a LEO time series skips most epochs outright.

```{important}
A flux of exactly zero is not a bug. It means eclipsed or occulted, and
returning zero is deliberately preferred over a plausible wrong number.
```

A `ViewGeometry` cannot answer the observability question, so for one the
factor is 1.0 and the caller keeps responsibility.

## Per-component attribution

```python
boxes = r.get_object_pixel_bboxes()
```

The renderer builds its pixel mask from these per-object bounding boxes, which
means it only casts rays where the body actually projects — and it means you
can attribute a peak to a named component after the fact.

## Diagnostics after a render

```python
r.hit_count       # pixels that hit the body
r.shadow_count    # of those, how many were shadowed
r.image           # the (resolution_y, resolution_x) array
```

`_check_frame_edges` warns when the body touches the frame edge, i.e. when the
fit has clipped it and the flux is an underestimate.

```{note}
A `hit_count` of 0 with a non-zero `observable_fraction` means the camera is
looking at the body from a direction where it projects to nothing, or the mesh
failed to import. See {doc}`../troubleshooting`.
```

## Resolution

`resolution=(500, 500)` by default. Cost is roughly linear in the pixel count,
and so is the time, since the ray cast dominates.

Convergence is a property of the geometry, not a fixed number: a flat plate
converges almost immediately, a narrow specular lobe may not converge at 500².
{doc}`validation` has the measured behaviour. The practical test is to render
the same epoch at two resolutions and see whether the answer moves.
