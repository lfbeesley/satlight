# satlight

**Physically-based light curves for resident space objects.**

satlight renders a spacecraft from the perspective of an observer. You give it an
orbit, an observing site, a mesh and a set of surface reflectance models; it
ray-traces the illuminated geometry and returns the irradiance arriving at the
observer, in W/m².

```{code-block} python
:caption: Example block

from satlight import Geometry, Renderer
import numpy as np, urllib.request

model, _ = urllib.request.urlretrieve(
    'https://raw.githubusercontent.com/lfbeesley/satlight/main/models/reference/spacex_ai1.obj', 'spacex_ai1.obj')

geom = Geometry()
geom.create_observer(observer_lat=28.7622, observer_lon=-17.8897, observer_alt_m=2396)
geom.create_satellite(line1, line2)                     # a TLE
geom.set_time(2024, 1, 15, 19, 53, 0)

r = Renderer(model, geometry=geom, brdf={'default': ('lambertian', {'albedo': 0.25})})
r.initialise_scene()
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
: The brightness modelling. Ray-cast the mesh under that illumination, evaluate a BRDF per
  lit pixel, and integrate to a flux.

Hand `Geometry` into the `Renderer` to derive the radiance conditions.

## What it does

- **Self-shadowing**, one shadow ray per lit pixel.
- **Multiple scattering** between components, useful for concave geometry such as rocket nozzles.
- **Earthshine**, reflected from a uniform-albedo Earth.
- **Eclipse and Earth occultation**
- **Per-component attribution**: which part of the spacecraft produced the
  glint, by object name.
- Seven standard BRDF models, or any callable you supply.

## What it does not do
- **It is broadband.** `render()` integrates the full solar constant with no
  spectral response. See {doc}`guide/photometry` before you compare to a
  filtered measurement.
- **It is not an attitude propagator.** You supply the orientation per epoch;
  satlight will not tumble the body for you.
- **Not tested for space-based observers**  -- I'm sure it would work... but I haven't
  tried to configure this scenario.

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
guide/articulation
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
changelog
```
