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
    r.update(geom)                              # call before every render
    flux = float(r.render().sum())              # W/m² at the observer
    print(f"phase {phase_deg:3d} deg   flux {flux:.3e} W/m²")
```

A flat Lambertian plate must follow the cosine law, so that the flux at
80° is `cos(80°) = 0.174` times the flux at 0°. Use this snippet to check that
your install is correct.

For an orbit rather than a fixed viewing geometry, build a
`satlight.geometry.Geometry` from a TLE or orbital elements and pass that as
`geometry=` instead.

## Real geometry: from a TLE

`ViewGeometry` above is the explicit form using ingoing vector, outgoing vector and a range, for
BRDF studies and phase sweeps with no orbit behind them. Currently cannot account for eclipsed 
satellite. For real observations build a `Geometry` instead, and hand that to the `Renderer`.

```python
from satlight.geometry import Geometry

geom = Geometry()
geom.create_observer(
    observer_lat=28.7622,        # degrees north
    observer_lon=-17.8897,       # degrees east
    observer_alt_m=2396,         # metres
)
geom.create_satellite(
    "1 25544U 98067A   24015.50000000  .00016717  00000-0  10270-3 0  9990",
    "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.49815174 10000",
)
geom.set_time(2024, 1, 15, 19, 53, 0)       # UTC, as separate arguments

print(geom.prop_distance)        # 1229.4   slant range, km
print(geom.is_visible())         # True     False if the Earth is in the way
print(geom.eclipse_type())       # 1.0      sunlit; 0.0 in shadow
```

To render it the `Renderer` reads the Sun and observer directions off the
`Geometry`:

```python
brdf = {
    'solar':    ('lambertian', {'albedo': 0.08}),   # cells are dark
    'radiator': ('lambertian', {'albedo': 0.60}),   # white thermal surfaces
    'default':  ('lambertian', {'albedo': 0.25}),   # everything else
}

r = Renderer("your_model.obj", geometry=geom, resolution=(300, 300), brdf=brdf)
r.initialise_scene()
r.update(geom)                              # before every render
flux = float(np.sum(r.render()))            # W/m² at the observer
```

**Materials are matched by object name.** A `brdf` key matches if it appears
anywhere in the object's name, compared case-insensitively, so `'solar'` catches
every `solar_*` component of your model. `'default'` catches the rest. See
`satlight.brdf` for the full list of models and for how to write your own.

A satellite is only worth rendering when it is above the horizon **and**
sunlit, so check both `is_visible()` and `eclipse_type()` before you trust a
frame.

## Real geometry: from Keplerian elements

For a notional target, such as a GEO slot:

```python
import datetime

geo = Geometry()
geo.create_observer(observer_lat=28.7622, observer_lon=-17.8897, observer_alt_m=2396)
geo.create_satellite_from_elements(
    a_km=6371.0 + 35786.0,       # semi-major axis
    e=0.0, i_deg=0.0,
    raan_deg=0.0, argp_deg=0.0,
    nu_deg=100.0,                # true anomaly: sets the GEO longitude
    epoch=datetime.datetime(2026, 8, 15, 22, 30,
                            tzinfo=datetime.timezone.utc),   # must be tz-aware
)
geo.set_time(2026, 8, 15, 22, 30, 0)
print(geo.prop_distance)         # 47839.7  km
```

## When is it observable?

```python
passes = geom.next_visibility((2024, 1, 15, 18, 0, 0), N=24, i=20.0)
for year, month, day, hour, minute, seconds in passes:
    print(f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d} UTC")
```

```
2024-01-15 19:53 UTC
2024-01-16 10:55 UTC
```

That tests elevation only, so check `eclipse_type()` across the pass to find
the part that is actually sunlit.

## Magnitudes

`render()` is **broadband** — it integrates the full solar constant with no
spectral response. To convert to a V-band-like magnitude use the solar
calibration:

```python
m = -2.5 * np.log10(flux) - 18.915      # = m_sun + 2.5*log10(1361), m_sun = -26.75
```

In-band spectral magnitudes can only be calculated by changing the `solar_constant` from 1361 W/m^2
to the in-band solar irradiance (174 W/m² for V).

## Citing satlight

If you use satlight in your work, please cite:

Beesley, L., Whitlock, P., Blake, J., Jones, G., Maric, N. and Alconcel, L.-N.
(2026). SatLight: A Python Package for High-Fidelity Light Curve Simulation.
*AMOS 2026 Technical Papers*. https://doi.org/10.64861/INPV6617

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

## Licence

GPL-3.0-or-later

## Contact

L Beesley — l.beesley@bham.ac.uk
