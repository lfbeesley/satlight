# satlight

Physically-based light curves for satellites and debris. satlight uses Blender to ray-trace a
spacecraft model (.obj) with representative BRDFs and returns the flux an observer on the
ground would measure.

Features:
- Per-component BRDFs (Lambertian, Phong, Cook-Torrance, Oren-Nayar, ABg)
- Self-shadowing and multi bounce scattering between components
- Earthshine as an extended source (currently constant albedo of 0.3, to be improved)
- Orbit propagation and observer geometry via skyfield and SGP4
- Output is radiometrically accurate rendered frame: W/m²/pixel at the observer
- Integrate over frame to give W/m² and apparent magnitude 

## Install

```bash
pip install satlight
```

Python 3.11 only, satlight uses `bpy==4.5.3` (Blender Python
API). The install is large — `bpy` is most of it.

## Quick start

```python
import numpy as np
from satlight.geometry import ViewGeometry
from satlight.raytracer import Renderer

# a 2 m x 2 m flat plate, replace with your own .obj
with open("plate.obj", "w") as f:
    f.write("o plate\nv -1 0 -1\nv 1 0 -1\nv 1 0 1\nv -1 0 1\nf 4 3 2 1\n")

for phase_deg in range(0, 90, 10):
    a = np.radians(phase_deg)
    geom = ViewGeometry(
        sun_lvlh=[np.sin(a), 0.0, np.cos(a)],   # direction to Sun
        observer_lvlh=[0.0, 0.0, 1.0],          # direction to observer
        distance_km=550.0,                      # slant range, in kilometres
    )
    r = Renderer("plate.obj", geometry=geom, resolution=(256, 256),
                 brdf=("lambertian", {"albedo": 0.3}))
    r.initialise_scene()
    r.add_camera()
    r.update(geom)                              # call before every render
    flux = float(r.render().sum())              # W/m² at the observer
    print(f"phase {phase_deg:3d} deg   flux {flux:.3e} W/m²")
```

A flat Lambertian plate must follow the cosine law, and this does: the flux at
80° is `cos(80°) = 0.174` times the flux at 0°. That makes the snippet a usable
check that your install is working.

For an orbit rather than a fixed viewing geometry, build a
`satlight.geometry.Geometry` from a TLE or orbital elements and pass that as
`geometry=` instead.

## Magnitudes

`render()` is **broadband** — it integrates the full solar constant with no
spectral response. To convert to a V-band-like magnitude use the solar
calibration:

```python
m = -2.5 * np.log10(flux) - 18.915      # = m_sun + 2.5*log10(1361), m_sun = -26.75
```

`satlight.tools.AB_mag` is a true AB magnitude and is correct only for in-band
(500–600 nm) flux. Applying it to `render()` output reads about 2.2 mag too
bright.

## Status

Alpha. The API still moves between releases. The forward model is gated against
closed-form truth in `tests/` — a Lambertian plate to 0.08% and a Lambertian
sphere's analytic phase function to 0.1–0.25% across phase angles 0–120°.

```bash
pytest                  # fast tier, no Blender, seconds
pytest -m ""            # everything, including renders
```

## Licence

GPL-3.0-or-later

## Contact

L Beesley — l.beesley@bham.ac.uk
