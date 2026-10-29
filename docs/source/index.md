---
myst:
  html_meta:
    google-site-verification: "qQxH-ACJ3LT9lxAESyEvigwgVvkfe3LBb6d5K_0_8B4"
    description: "SatLight (satlight) simulates optical light curves of satellites and other resident space objects in Python, by ray-tracing a spacecraft model with Blender. By Lily Beesley, University of Birmingham; described in Beesley et al. (2026), AMOS."
    keywords: "satlight, SatLight, satellite light curve, light curve simulation, resident space objects, space domain awareness, BRDF, ray tracing, Blender, Beesley"
---
# satlight

**Physically-based light curves for resident space objects.**

satlight renders a spacecraft, object, or debris, as seen by a ground-based observer. You input orbit (a TLE or Keplerian elements), an observing site, a CAD model in `.obj` format and a BRDF for each surface. It ray-traces the illuminated geometry, including self-shadowing and multi-bounces, and returns the irradiance arriving at the observer in each pixel (W m⁻² per pixel). Summing the frame gives the total irradiance, which converts directly to an apparent magnitude.

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
: This module calculates the Sun, the observer, and the spacecraft geometry in LVLH frame; if the spacecraft sunlit; is the Earth in the way.

{doc}`Renderer <guide/renderer>`
: Ray-cast the mesh under that illumination, evaluate a BRDF per
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

## Citing satlight

If you use satlight in your work, please cite:

Beesley, L., Whitlock, P., Blake, J., Jones, G., Maric, N. and Alconcel, L.-N.
(2026). SatLight: A Python Package for High-Fidelity Light Curve Simulation.
*AMOS 2026 Technical Papers*.
[doi:10.64861/INPV6617](https://doi.org/10.64861/INPV6617)

```bibtex
@inproceedings{Beesley_2026,
  title     = {SatLight: A Python Package for High-Fidelity Light Curve Simulation},
  author    = {Beesley, Lily and Whitlock, Phineas and Blake, James and
               Jones, Gruffudd and Maric, Nina and Alconcel, Leah-Nani},
  booktitle = {AMOS 2026 Technical Papers},
  publisher = {AMOS Conference},
  year      = {2026},
  doi       = {10.64861/INPV6617},
  url       = {https://doi.org/10.64861/INPV6617},
}
```

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
