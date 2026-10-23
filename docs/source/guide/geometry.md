# Observing geometry

`Geometry` works out, for a given time, where the Sun and the observer are as
seen from the spacecraft, and how far away the observer is. These are what the
renderer needs.

Times are UTC. Angles go in as degrees and come out as radians.

## The LVLH frame

Directions are given in the spacecraft's Local-Vertical/Local-Horizontal
frame:

| Axis | Direction |
|---|---|
| X | along-track (velocity) |
| Y | orbit normal |
| Z | zenith (away from the Earth) |

Nadir is $-\hat{Z}$. `Renderer.set_attitude` rotates about these axes.

## Observer

```python
from satlight import Geometry

geom = Geometry()
geom.create_observer(
    observer_lat=28.7622,       # degrees north
    observer_lon=-17.8897,      # degrees east
    observer_alt_m=2396,        # metres above the WGS84 ellipsoid
)
```

## Satellite

From a TLE:

```python
line1 = "1 25544U 98067A   24015.50000000  .00016717  00000-0  10270-3 0  9990"
line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.49815174 10000"
geom.create_satellite(line1, line2, name='ISS')
```

From Keplerian elements:

```python
import datetime

geom.create_satellite_from_elements(
    a_km=6371.0 + 35786.0,      # semi-major axis
    e=0.0,
    i_deg=0.0,
    raan_deg=0.0,
    argp_deg=0.0,
    nu_deg=100.0,               # true anomaly
    epoch=datetime.datetime(2026, 8, 15, 22, 30,
                            tzinfo=datetime.timezone.utc),
)
```

Both are propagated with SGP4 (via Skyfield). A `datetime` without a time zone
is read as UTC.

## Time

```python
geom.set_time(2024, 1, 15, 19, 53, 0)                    # separate arguments
geom.set_time((2024, 1, 15, 19, 53, 0))                  # one tuple
geom.set_time(datetime.datetime(2024, 1, 15, 19, 53))    # a datetime
```

A range in any one component gives many epochs at once:

```python
geom.set_time(2024, 1, 15, 19, 53, range(0, 60))         # 60 consecutive seconds
```

This is useful for checking visibility and eclipse over a pass. The renderer
takes one epoch at a time, so build a single-epoch `Geometry` for each frame
you render.

## Outputs

```python
geom.prop_distance          # range to the observer, km
geom.phase_angle            # solar phase angle, radians
geom.incident_vector_lvlh   # direction to the Sun, LVLH
geom.outgoing_vector_lvlh   # direction to the observer, LVLH
geom.eclipse_type()         # 1.0 sunlit, 0.0 in shadow, in between in penumbra
geom.is_visible()           # False if the Earth blocks the line of sight
geom.observable_fraction()  # eclipse_type() * is_visible()
```

`eclipse_type()` says whether sunlight reaches the satellite. `is_visible()`
says whether the observer can see it. They are independent: a satellite can be
sunlit but below the horizon, or overhead but in the Earth's shadow.

`is_visible(atmosphere_km=20.0)` raises the Earth's limb by that height.

## Finding passes

```python
passes = geom.next_visibility((2024, 1, 15, 18, 0, 0), N=24, i=20.0)
for year, month, day, hour, minute, seconds in passes:
    print(f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d} UTC")
```

`N` is the number of hours to search and `i` the minimum elevation in degrees.
This checks elevation only, so use `eclipse_type()` to find the sunlit part.

## ViewGeometry

For work without an orbit, such as phase-angle sweeps or BRDF studies, give
the directions directly:

```python
from satlight import ViewGeometry

vg = ViewGeometry(
    sun_lvlh=[0.0, 0.0, 1.0],
    observer_lvlh=[0.5, 0.0, 0.866],
    distance_km=550.0,          # kilometres
    altitude_km=550.0,          # only needed for earthshine
)
r = Renderer(obj_path, geometry=vg)
```

A `ViewGeometry` doesn't know where the Earth is, so it can't check eclipse or
visibility; `render()` won't zero the flux for you.
