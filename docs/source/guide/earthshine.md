# Earthshine

A satellite in eclipse is not necessarily dark: the Earth beneath it is a
bright, enormous reflector. At LEO the Earth subtends about 68° — it is not a
point source, and treating it as one gives the wrong answer in a way that is
easy to miss.

```python
r = Renderer(obj_path, geometry=geom,
             add_earthshine=True,
             earth_albedo=0.3,
             earthshine_n_rings=4)
```

That is all that is required. The term is computed from the geometry in the
constructor and recomputed on every `update()`.

## Why it is an extended source

A patch of Earth subtending solid angle $\omega$ with radiance $L$ is, to any
surface element, exactly a directional source of irradiance

$$ E = L\,\omega $$

arriving from that patch's direction. So the visible Earth becomes a **set** of
directional sources, which `render()` sums over, each one evaluated through the
surface's BRDF like any other light.

```{important}
A lumped single direction cannot represent this. A facet perpendicular to the
mean Earth direction collects exactly zero from it, while physically still
seeing half the disc. No single direction can represent a source 68° wide.
```

An earlier version of satlight did collapse the Earth into one effective
direction and one total irradiance. That is the behaviour this replaced.

## The discretisation

The Earth cap is discretised into patches of **equal solid angle**, not equal
surface area.

```{note}
This distinction is the whole accuracy of the method. The earthshine integral
weights each patch by the solid angle it subtends at the satellite. Equal-area
discretisation piles samples into the few patches directly beneath the
satellite, under-resolves the limb, and reads low at LEO. Equal solid angle
samples exactly what the integral weights by.
```

`earthshine_n_rings` sets the resolution, and the total is gated in the test
suite against the analytic spherical cap $2\pi(1 - \cos\alpha)$.

| `n_rings` | Patches | Agreement with a 960-patch grid |
|---|---|---|
| 4 (default) | 88 | 0.001 mag |

Cost is linear in the patch count, and each patch costs one BRDF evaluation
*per lit pixel* — so earthshine with 88 patches is roughly 88× the direct
term's BRDF work. It is the most expensive option in the package. `n_rings=4`
already converges, so there is no reason to raise it.

## Albedo

`earth_albedo=0.3`
: A uniform Bond albedo per patch. 0.3 is the global mean, and is the right
  choice when you want a clean, reproducible number.

## Self-shadowing of earthshine

```python
r = Renderer(..., add_earthshine=True, earthshine_shadows=True)
```

Whether the spacecraft occludes its own earthshine — a nadir-facing panel
shadowed by its own stand-offs. `None` (the default) follows `add_shadows`.

```{warning}
This is **expensive**: one shadow ray per patch per pixel. With 88 patches
that is 88 shadow rays per lit pixel on top of everything else. Turn it on
deliberately, for geometry where it plausibly matters, not by default.
```

## Using it with `ViewGeometry`

The Earth's angular size depends on **altitude**, which a slant range alone
does not give. A `ViewGeometry` must therefore be told:

```python
vg = ViewGeometry(sun_lvlh, observer_lvlh, distance_km=1200.0,
                  altitude_km=550.0)       # required for earthshine
```

Without `altitude_km`, `compute_earthshine` cannot proceed.

## Frames

Patch directions are produced in the same frame as `sun_direction`, with nadir
taken from `nadir_direction` (which is $-\hat{Z}$ in LVLH).

```{warning}
If you render in a **rolled body frame** rather than LVLH, pass `altitude_km`
and `sun_direction` in *that* frame, or the Earth will sit at the wrong
attitude relative to the spacecraft — producing a plausible image lit from the
wrong side.
```

## Gotchas

```{important}
Setting `add_earthshine = True` on an **existing** `Renderer` does not work.
`render()` raises a `RuntimeError` telling you so, rather than silently
dropping the term.
```

Earthshine used to be a flag that set an attribute and nothing else: the term
was silently dropped unless the caller also remembered to call
`compute_earthshine()`. It is derived entirely from the geometry, so it is now
computed in the constructor and again on every `update()` — which also stops it
going stale as the satellite moves. If you must set the flag late, call
`compute_earthshine(geometry)` or `update(geometry)` after it.

### Backwards-compatible attributes

For callers that read them, `compute_earthshine` still sets:

`earthshine_irradiance`
: the total irradiance on a plate facing the Earth

`earthshine_direction`
: the flux-weighted mean direction

Neither is what `render()` uses. The per-patch arrays
(`earthshine_dirs`, `earthshine_patch_irradiance`) are the extended-source
form, and they are the physics.
