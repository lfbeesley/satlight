# Tutorial: a satellite light curve with satlight

Everything on this page is runnable. It builds one real light curve of a real
pass, in six steps:

**observer → satellite → time → materials → attitude → propagate**

## 1. Install

```bash
pip install satlight
```

Python 3.11, because satlight pins `bpy==4.5.3` (Blender as a Python module),
which does the ray casting. The download is large; most of it is `bpy`.

```bash
pip install "satlight[preview]"     # adds plotly for the 3-D preview
```

## 2. How it fits together

Two halves, and they meet at three vectors.

- **`Geometry`** — the orbital mechanics. Given an observer and a satellite, it
  works out where the Sun and the observer are *as seen from the spacecraft*,
  whether the satellite is sunlit, and whether the Earth is in the way.
- **`Renderer`** — the optics. It ray-traces the model under that illumination
  and returns the irradiance at the observer, in W/m².

```
TLE → Geometry → sun & observer directions (LVLH) → Renderer → flux → magnitude
```

You never pass those vectors by hand: you hand the `Geometry` to the
`Renderer`, and it reads what it needs.

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

That is Roque de los Muchachos on La Palma.

## 4. The satellite

From a TLE, which is what you will usually have:

```python
line1 = "1 25544U 98067A   24015.50000000  .00016717  00000-0  10270-3 0  9990"
line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.49815174 10000"
geom.create_satellite(line1, line2)
```

Or from Keplerian elements, for a notional target such as a GEO slot:

```python
import datetime

geom.create_satellite_from_elements(
    a_km=6371.0 + 35786.0,
    e=0.0, i_deg=0.0, raan_deg=0.0, argp_deg=0.0,
    nu_deg=100.0,                                     # sets the GEO longitude
    epoch=datetime.datetime(2026, 8, 15, 22, 30,
                            tzinfo=datetime.timezone.utc),   # MUST be tz-aware
)
```

```{note}
A naive `datetime` is read as UTC, so `tzinfo` is optional. Everything in
`Geometry` is UTC.
```

## 5. The time

```python
geom.set_time(2024, 1, 15, 19, 53, 0)              # separate arguments
geom.set_time((2024, 1, 15, 19, 53, 0))            # or one tuple
geom.set_time(datetime.datetime(2024, 1, 15, 19, 53))   # or a datetime
```

All three are equivalent. For many epochs at once, put a range in any one
component:

```python
geom.set_time(2024, 1, 15, 19, 53, range(0, 60))   # 60 consecutive seconds
```

Now the geometry can answer the questions that decide whether there is
anything to see at all:

```python
print(geom.prop_distance)        # slant range, km
print(geom.is_visible())         # False if the Earth is in the way
print(geom.eclipse_type())       # 1.0 sunlit, 0.0 in shadow, between in penumbra
```

### Finding a time worth rendering

A satellite is only worth rendering when it is **above the horizon and sunlit**.
`next_visibility` finds the passes:

```python
passes = geom.next_visibility((2024, 1, 15, 18, 0, 0), N=24, i=20.0)
for year, month, day, hour, minute, seconds in passes:
    print(f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d} UTC")
```

It tests elevation only, so check `eclipse_type()` across the pass to find the
sunlit part.

## 6. The model and its materials

satlight ships no models. The one used here is in the repository's
[`models/`](https://github.com/lfbeesley/satlight/tree/main/models) directory:

```python
import urllib.request
model, _ = urllib.request.urlretrieve(
    'https://raw.githubusercontent.com/lfbeesley/satlight/main/models/reference/spacex_ai1.obj', 'spacex_ai1.obj')
```

`spacex_ai1` is a 70 m orbital data-centre concept with twelve named
components: `bus`, `boom_left`, `boom_right`, three `radiator_*` and six
`solar_*`.

Materials are assigned **by object name**, and this is the one rule worth
learning:

```{important}
A `brdf` key matches if it appears **anywhere in the object's name**, compared
case-insensitively. `'solar'` matches all six `solar_*` parts; `'radiator'`
matches all three. `'default'` catches anything unmatched.
```

```python
brdf = {
    'solar':    ('lambertian', {'albedo': 0.08}),   # cells are dark
    'radiator': ('lambertian', {'albedo': 0.60}),   # white thermal surfaces
    'boom':     ('lambertian', {'albedo': 0.20}),
    'bus':      ('lambertian', {'albedo': 0.30}),
    'default':  ('lambertian', {'albedo': 0.25}),
}
```

Available BRDFs: `lambertian`, `phong`, `blinn_phong`, `cook_torrance`,
`oren_nayar`. You can also pass your own callable.

```python
from satlight.raytracer import Renderer

r = Renderer(model, geometry=geom, resolution=(300, 300),
             brdf=brdf, add_shadows=True)
r.initialise_scene()
r.add_camera()
```

To use your own mesh, pass its path instead. Use **metres**, and give each
component a name you can match on.

## 7. Orienting the spacecraft

```python
import numpy as np

r.set_attitude(euler_xyz=(0.0, 0.0, np.radians(30.0)))
```

Angles are in radians about the LVLH axes — X along-track, Y orbit normal, Z
zenith — so `(0, 0, 0)` is the model as imported. The rotation is always
applied to the imported pose, never compounded, so it is safe to call once per
frame. Call it **before** `update()`, so the camera is placed around the new
orientation.

With `satlight[preview]` installed you can check the orientation first:

```python
r.preview_lvlh_views()        # opens an interactive 3-D view in your browser
```

## 8. Propagating the pass

Build a fresh `Geometry` per epoch, hand it to `update()`, and render:

```python
import datetime

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
    r.update(g)                                   # before every render
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

The satellite brightens as the range closes and the phase angle falls, then
drops out entirely when it enters eclipse. **A flux of exactly zero is not a
bug**: `render()` multiplies by `eclipse_type() × is_visible()`, so an eclipsed
or occulted satellite returns zero rather than a plausible wrong number.

## 9. Flux to magnitude

`render()` is **broadband** — it integrates the full solar constant with no
spectral response. Convert with the solar calibration:

```python
m = -2.5 * np.log10(flux) - 18.915     # = m_sun + 2.5*log10(1361), m_sun = -26.75
```

```{warning}
`satlight.tools.AB_mag` is a true AB magnitude, valid only for **in-band**
(500–600 nm) flux. Applied to `render()` output it reads about **2.2 mag too
bright**.
```

## 10. Where to go next

- `add_earthshine=True` adds light reflected from the Earth, as an extended
  source. Computed automatically from the geometry.
- `n_bounce=2` adds one inter-reflection between components — it matters for
  concave geometry that sees itself.
- `smooth_normals=True` interpolates shading normals across curved surfaces.
- Swap `lambertian` for a specular BRDF to get glints. Note that the specular
  lobe **adds** to the diffuse floor rather than being budgeted within a total,
  so raising it raises the surface's overall reflectance too.
