# Tutorial: Generating a Satellite Lightcurve with satlight

This tutorial walks through the core `satlight` workflow end to end: setting
up an observer and a satellite, computing the observation geometry for a
given time, rendering the satellite with Blender's ray tracer, and converting
the result into an astronomical magnitude. By the end you'll have a single
synthetic lightcurve point, and the tools to step that over time into a full
lightcurve.

```{note}
`satlight` uses Blender's Python API (`bpy`) for ray tracing. Make sure `bpy`
is installed in your environment before continuing — see {ref}`installation`.
```

(installation)=
## 1. Installation

Install `satlight` and its dependencies:

```bash
pip install satlight
```

`satlight` depends on `bpy` for rendering, plus `numpy`, `astropy`, `skyfield`,
`sgp4`, and `trimesh` for the orbital mechanics and geometry side. These are
installed automatically via `pyproject.toml`.

```{warning}
Some larger data files — the BRDF material database and example satellite
models — are not bundled with the pip package. Contact the maintainer for
access, and place them under `satlight/data/` once you have them.
```

## 2. The two halves of satlight

`satlight` is built around two objects that hand off to one another:

- **`Geometry`** — orbital mechanics. Given an observer location and a
  satellite (from a TLE or orbital elements), it computes where the sun and
  observer are relative to the satellite at a given time, in the satellite's
  own body frame (LVLH).
- **`Renderer`** — the Blender ray tracer. Given a 3D model and the sun/observer
  directions from `Geometry`, it renders the satellite and returns the
  reflected flux.

The general flow is:

```text
TLE / elements → Geometry → sun & observer vectors (LVLH) → Renderer → flux → magnitude
```

## 3. Setting up the geometry

Start by creating a `Geometry` object, defining where you're observing from,
and defining the satellite.

### Observer

```python
from satlight.geometry import Geometry

geom = Geometry()
geom.create_observer(
    lat=28.29156,      # degrees
    lon=-16.62,        # degrees
    alt_m=2070,        # metres above sea level
)
```

### Satellite from a TLE

If you have a two-line element set for your target:

```python
line1 = "1 25544U 98067A   ..."
line2 = "2 25544  51.6416 ..."

geom.create_satellite(line1, line2)
```

### Satellite from orbital elements

For synthetic scenarios (e.g. a notional GEO satellite at a given
longitude), you can instead build the satellite directly from Keplerian
elements:

```python
R_EARTH_KM = 6371.0
geo_altitude_km = 35786.0

geom.create_satellite_from_elements(
    a_km=R_EARTH_KM + geo_altitude_km,
    e=0.0,
    i_deg=0.0,
    raan_deg=0.0,
    argp_deg=0.0,
    nu_deg=100.0,      # sets the satellite's longitude for a circular GEO
    epoch=epoch_datetime,
)
```

### Setting the observation time

```python
geom.set_time((2026, 8, 15, 22, 30, 0))  # year, month, day, hour, min, sec (UTC)
```

This propagates the orbit and updates `Geometry`'s internal vectors for that
instant.

### Don't know a good observation time? Search for one

If you don't already have a target time in mind, use `next_visibility()` to
search forward from a starting time and find windows when the satellite is
above a given elevation from your observer:

```python
geom.next_visibility(
    time_utc=(2026, 8, 15, 0, 0, 0),  # search start (UTC)
    N=24,                             # search window, in hours
    i=30.0,                           # minimum elevation, in degrees
)
```

This prints each rise/set window found in the next `N` hours where the
satellite is above `i` degrees elevation, e.g.:

```text
Observable above 30° from 15/08/2026 20:14:03 to 15/08/2026 20:21:47 for 464 seconds
```

Pick a time inside one of the printed windows and pass it to `set_time()` as
above.

```{note}
`next_visibility` only checks geometric elevation, not illumination — it
doesn't tell you whether the satellite is sunlit or the sky is dark enough to
observe against. Combine it with `geom.eclipse_type()` at your chosen time to
confirm the satellite isn't in Earth's shadow.
```

## 4. Reading off the geometry vectors

After `set_time()`, the sun and observer directions are available in the
satellite's body (LVLH) frame:

```python
sun_lvlh = -geom.incident_vector_lvlh   # incident_vector points sat→sun's *source*, i.e. away from sun
obs_lvlh = geom.outgoing_vector_lvlh    # direction from satellite toward the observer
```

```{note}
`incident_vector_lvlh` is defined sun→satellite, so the direction *toward*
the sun (what the `Renderer` expects) is its negative.
```

You can also check whether the satellite is eclipsed at this time before
bothering to render it:

```python
illumination_fraction = geom.eclipse_type()
if illumination_fraction == 0:
    print("Satellite is in Earth's shadow — skip render")
```

## 5. Rendering with satlight's Renderer

```{important}
Your `.obj` model needs to be oriented in the **LVLH frame** before rendering:
+X along-track (velocity), +Y cross-track (orbit normal), +Z zenith/anti-nadir
(away from Earth). If your model was exported from Blender or CAD with a
different convention (a common mismatch is panels running along the wrong
axis, or the bus not facing nadir), the render will be geometrically wrong
even though nothing errors out.
```

With sun and observer directions in hand, set up the `Renderer` and render a
frame:

```python
from satlight.raytracer import Renderer

renderer = Renderer(
    obj_path="models/satellite.obj",
    sun_direction=sun_lvlh,
    observer_direction=obs_lvlh,
    distance_to_observer=40_000_000,   # metres
    resolution=(300, 300),
    solar_constant=1361,               # W/m^2 at 1 AU
)

renderer.initialise_scene()
```

### Checking model orientation before you render

Before trusting a render, call `preview_lvlh_views()` to open an interactive
3D plot of the model with the LVLH axes overlaid, so you can see at a glance
whether the model needs correcting:

```python
renderer.preview_lvlh_views()   # opens an interactive plot — click and drag to rotate
```

If the model is misaligned (e.g. solar panels running along X instead of Y),
pass a `rotation` to the `Renderer` to correct it before rendering:

```python
renderer = Renderer(
    obj_path="models/satellite.obj",
    sun_direction=sun_lvlh,
    observer_direction=obs_lvlh,
    distance_to_observer=40_000_000,
    resolution=(300, 300),
    solar_constant=1361,
    rotation=(0, 0, 90),   # degrees about (X, Y, Z) to align model with LVLH
)
renderer.initialise_scene()
renderer.preview_lvlh_views()   # re-check alignment after the correction
```

Now render:

```python
image = renderer.render()
```

`image` is a 2D array of per-pixel flux values. Summing it gives the total
reflected flux collected by the observer for that instant:

```python
total_flux = image.sum()
```

## 6. Converting flux to magnitude

Use the `tools` module to convert flux into an AB magnitude:

```python
from satlight.tools import AB_mag

magnitude = AB_mag(total_flux)
print(f"Apparent magnitude: {magnitude:.2f}")
```

## 7. Putting it together: a lightcurve over time

To generate a full lightcurve, step `set_time()` across a range of
timestamps, re-deriving the geometry and re-rendering at each step:

```python
import numpy as np
from datetime import datetime, timedelta

times = [datetime(2026, 8, 15, 20, 0, 0) + timedelta(minutes=5*i) for i in range(60)]

records = []
for t in times:
    geom.set_time((t.year, t.month, t.day, t.hour, t.minute, t.second))

    if geom.eclipse_type() == 0:
        continue  # satellite not illuminated

    sun_lvlh = -geom.incident_vector_lvlh
    obs_lvlh = geom.outgoing_vector_lvlh

    renderer.update_geometry(sun_lvlh, obs_lvlh)  # or re-instantiate the Renderer
    image = renderer.render()

    records.append({
        "time": t,
        "flux": image.sum(),
        "magnitude": AB_mag(image.sum()),
    })
```

`records` now holds a time-series of magnitude estimates — your synthetic
lightcurve.

## 8. Next steps

- Explore different BRDF materials in `satlight.brdf` for solar panels vs.
  bus surfaces.
- Enable earthshine contributions via the CERES-based albedo model for more
  realistic illumination near eclipse boundaries.
- See the {doc}`api` reference for the full parameter list on `Geometry` and
  `Renderer`.

```{seealso}
For installation issues, model-loading errors, or dark/zero-flux renders,
see the {doc}`troubleshooting` page.
```
