# satlight

**Physically-based light curves for resident space objects.**

satlight renders a spacecraft the way a telescope sees it. You give it an
orbit, an observing site, a mesh and a set of surface reflectance models; it
ray-traces the illuminated geometry and returns the irradiance arriving at the
observer, in W/m². Nothing in the chain is fitted or empirical — the output is
a radiometric prediction, which is what makes it useful for asking *why* a
light curve looks the way it does.

```{code-block} python
:caption: The whole package, in a few lines

from satlight import Geometry, Renderer
import numpy as np, urllib.request

model, _ = urllib.request.urlretrieve(
    'https://raw.githubusercontent.com/lfbeesley/satlight/main/models/reference/spacex_ai1.obj', 'spacex_ai1.obj')

geom = Geometry()
geom.create_observer(observer_lat=28.7622, observer_lon=-17.8897, observer_alt_m=2396)
geom.create_satellite(line1, line2)                     # a TLE
geom.set_time(2024, 1, 15, 19, 53, 0)

r = Renderer(model, geometry=geom, brdf={'default': ('lambertian', {'albedo': 0.25})})
r.initialise_scene(); r.add_camera()
flux = float(np.sum(r.render()))                        # W/m²
print(-2.5 * np.log10(flux) - 18.915)                   # apparent magnitude
```

## How it works

```text
  TLE / Keplerian elements ─┐
  observer site ────────────┼──▶  Geometry  ──┐
  epoch ────────────────────┘                 │  sun & observer directions
                                              │  in LVLH, + slant range
                   mesh + BRDFs ──▶ Renderer ◀┘
                                       │
                                       ▼
                            irradiance (W/m²) ──▶ apparent magnitude
```

{doc}`Geometry <guide/geometry>`
: The orbital mechanics. Where are the Sun and the observer *as seen from the
  spacecraft*; is the spacecraft sunlit; is the Earth in the way.

{doc}`Renderer <guide/renderer>`
: The optics. Ray-cast the mesh under that illumination, evaluate a BRDF per
  lit pixel, and integrate to a flux.

You never pass those vectors by hand. You hand the `Geometry` to the
`Renderer` and it reads what it needs — which is also the reason the
kilometres-to-metres conversion lives in exactly one place.

## What it does, concretely

- **Self-shadowing**, one shadow ray per lit pixel.
- **Multiple scattering** between components, for concave geometry that sees
  itself.
- **Earthshine** as an extended source — a solid-angle-discretised Earth cap,
  not a single ray.
- **Eclipse and occultation** handled as independent conditions, so a
  satellite that has set returns zero rather than a plausible wrong number.
- **Per-component attribution**: which part of the spacecraft produced the
  glint, by object name.
- Seven BRDF models, or any callable you supply.

## What it does not do

- **It is broadband.** `render()` integrates the full solar constant with no
  spectral response. See {doc}`guide/photometry` before you compare to a
  filtered measurement.
- **It is not an attitude propagator.** You supply the orientation per epoch;
  satlight will not tumble the body for you.
- **It has no atmosphere.** No extinction, no seeing, no sky background.
- **It is single-process.** See {doc}`guide/performance` for the cost model and
  how to parallelise over epochs.

```{toctree}
:hidden:
:caption: Getting started

installation
tutorial
```

```{toctree}
:hidden:
:caption: User guide

guide/index
guide/geometry
guide/materials
guide/renderer
guide/earthshine
guide/photometry
guide/performance
guide/validation
```

```{toctree}
:hidden:
:caption: Reference

api/index
troubleshooting
contributing
changelog
```
