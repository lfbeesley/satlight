# Tutorial

This page computes the brightness of a satellite over part of a pass.

## 1. Install

```bash
pip install --pre satlight
```

satlight needs Python 3.11, because it depends on `bpy==4.5.3` (Blender as a
Python module), which does the ray tracing.

## 2. Using satlight

There are two main classes:

- **`Geometry`** works out where the Sun and the observer are as seen from
  the spacecraft (the incident and outgoing vectors), whether the satellite is
  sunlit, and whether the Earth is in the way.
- **`Renderer`** simulates the brightness. It ray-traces the model under that
  illumination and returns the irradiance at the observer, in W/m².
  Self-shadowing is on by default; you can also add earthshine
  (`add_earthshine=True`) and multiple bounces (`n_bounce=2`).

For more on these options and what they cost to run, see {doc}`guide/renderer`
and {doc}`guide/performance`.

```
TLE → Geometry → Sun and observer directions → Renderer → flux → magnitude
```

You pass the `Geometry` to the `Renderer`, which reads the directions from it.

## 3. The observer

```python
from satlight.geometry import Geometry

geom = Geometry()
geom.create_observer(
    observer_lat=28.7622,       # degrees north
    observer_lon=-17.8897,      # degrees east
    observer_alt_m=2396,        # metres above the ellipsoid
)
```

This is Roque de los Muchachos on La Palma.

## 4. The satellite

From a TLE:

```python
line1 = "1 25544U 98067A   24015.50000000  .00016717  00000-0  10270-3 0  9990"
line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.49815174 10000"
geom.create_satellite(line1, line2)
```

Or from Keplerian elements, for example a GEO slot:

```python
import datetime

geom.create_satellite_from_elements(
    a_km=6371.0 + 35786.0,
    e=0.0, i_deg=0.0, raan_deg=0.0, argp_deg=0.0,
    nu_deg=100.0,
    epoch=datetime.datetime(2026, 8, 15, 22, 30,
                            tzinfo=datetime.timezone.utc),
)
```

All times are UTC.

## 5. The time

```python
geom.set_time(2024, 1, 15, 19, 53, 0)                    # separate arguments
geom.set_time((2024, 1, 15, 19, 53, 0))                  # one tuple
geom.set_time(datetime.datetime(2024, 1, 15, 19, 53))    # a datetime
```

These are equivalent. A range in one component gives many epochs:

```python
geom.set_time(2024, 1, 15, 19, 53, range(0, 60))   # 60 consecutive seconds
```

```python
print(geom.prop_distance)        # range, km
print(geom.is_visible())         # False if the Earth is in the way
print(geom.eclipse_type())       # 1.0 sunlit, 0.0 in shadow, in between in penumbra
```

To find passes, use `next_visibility`. It checks elevation only, so use
`eclipse_type()` to find the sunlit part:

```python
passes = geom.next_visibility((2024, 1, 15, 18, 0, 0), N=24, i=20.0)
for year, month, day, hour, minute, seconds in passes:
    print(f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d} UTC")
```

## 6. The model and its materials

The model used here is in the repository's
[`models/`](https://github.com/lfbeesley/satlight/tree/main/models) directory:

```python
import urllib.request
model, _ = urllib.request.urlretrieve(
    'https://raw.githubusercontent.com/lfbeesley/satlight/main/models/reference/spacex_ai1.obj', 'spacex_ai1.obj')
```

It has twelve named components: `bus`, `boom_left`, `boom_right`, three
`radiator_*` and six `solar_*`.

Materials are assigned by object name. A key matches if it appears anywhere in
the name, case-insensitively, so `'solar'` matches all six `solar_*` parts.
`'default'` is used for anything unmatched.

```python
brdf = {
    'solar':    ('lambertian', {'albedo': 0.08}),
    'radiator': ('lambertian', {'albedo': 0.60}),
    'boom':     ('lambertian', {'albedo': 0.20}),
    'bus':      ('lambertian', {'albedo': 0.30}),
    'default':  ('lambertian', {'albedo': 0.25}),
}
```

See {doc}`guide/materials` for the other BRDF models.

```python
from satlight.raytracer import Renderer

r = Renderer(model, geometry=geom, resolution=(300, 300), brdf=brdf)
r.initialise_scene()
```

To use your own mesh, pass its path instead. The mesh should be in metres.

## 7. Orienting the spacecraft

```python
import numpy as np

r.set_attitude(euler_xyz=(0.0, 0.0, np.radians(30.0)))
```

Angles are in radians about the LVLH axes: X along-track, Y orbit normal, Z
zenith. `(0, 0, 0)` is the model as imported. Call it before `update()`.

To check the orientation:

```python
r.preview_lvlh_views()        # interactive 3-D view in your browser
```

## 8. Rendering the pass

Build a `Geometry` for each epoch, pass it to `update()`, and render:

```python
def geometry_at(when):
    g = Geometry()
    g.create_observer(observer_lat=28.7622, observer_lon=-17.8897,
                      observer_alt_m=2396)
    g.create_satellite(line1, line2)
    g.set_time(when.year, when.month, when.day,
               when.hour, when.minute, when.second)
    return g

t0 = datetime.datetime(2024, 1, 15, 19, 53, 0)

for step in range(7):
    when = t0 + datetime.timedelta(seconds=30 * step)
    g = geometry_at(when)
    r.update(g)
    flux = float(np.sum(r.render()))              # W/m² at the observer
    mag = -2.5 * np.log10(flux) - 18.915 if flux > 0 else None
    print(f"{when:%H:%M:%S}  {g.prop_distance:7,.0f} km  "
          f"{np.degrees(g.phase_angle):5.1f} deg  "
          + (f"m_V {mag:6.2f}" if mag else "not observable"))
```

```
19:53:00    1,229 km   99.2 deg  m_V   2.15
19:53:30    1,040 km   94.5 deg  m_V   1.75
19:54:00      861 km   87.8 deg  m_V   1.30
19:54:30      702 km   78.1 deg  m_V   0.83
19:55:00      579 km   63.9 deg  m_V   0.44
19:55:30      519 km   45.7 deg  m_V  -0.30
19:56:00      542 km   30.6 deg  not observable
```

At 19:56 the satellite enters the Earth's shadow and the flux is zero.

## 9. Flux to magnitude

```python
m = -2.5 * np.log10(flux) - 18.915     # = m_sun + 2.5*log10(1361), m_sun = -26.75
```

Don't use `satlight.tools.AB_mag` on `render()` output; see
{doc}`guide/photometry`.
