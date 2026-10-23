# Observing geometry

`Geometry` answers the orbital-mechanics questions, and hands the renderer
three things: where the Sun is, where the observer is — both as unit vectors in
the spacecraft's own frame — and how far away the observer is.

Everything is UTC. Everything angular is radians on the way out, degrees on the
way in, which is the convention the constructors use.

## The LVLH frame

Directions reaching the renderer are expressed in the spacecraft's
Local-Vertical/Local-Horizontal frame:

| Axis | Direction |
|---|---|
| **X** | along-track (velocity) |
| **Y** | orbit normal |
| **Z** | zenith (away from the Earth) |

So nadir is $-\hat{Z}$, and a mesh imported with its solar panels along X has
them along-track. This matters when you call
{meth}`~satlight.raytracer.Renderer.set_attitude`, whose Euler angles are about
these axes.

## Building one

### The observer

```python
from satlight import Geometry

geom = Geometry()
geom.create_observer(
    observer_lat=28.7622,       # degrees north
    observer_lon=-17.8897,      # degrees east
    observer_alt_m=2396,        # metres above the WGS84 ellipsoid
)
```

That is Roque de los Muchachos, La Palma.

### The satellite, from a TLE

The usual case:

```python
line1 = "1 25544U 98067A   24015.50000000  .00016717  00000-0  10270-3 0  9990"
line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.49815174 10000"
geom.create_satellite(line1, line2, name='ISS')
```

Propagation is SGP4, via Skyfield. TLE accuracy degrades over days, so use an
element set close to the epoch you care about.

### The satellite, from Keplerian elements

For a notional target — a GEO slot, a proposed constellation shell — there is
no TLE to have:

```python
import datetime

geom.create_satellite_from_elements(
    a_km=6371.0 + 35786.0,      # semi-major axis
    e=0.0,
    i_deg=0.0,
    raan_deg=0.0,
    argp_deg=0.0,
    nu_deg=100.0,               # true anomaly; for GEO this sets the longitude
    epoch=datetime.datetime(2026, 8, 15, 22, 30,
                            tzinfo=datetime.timezone.utc),
)
```

The elements are converted to a `Satrec` and propagated with the same SGP4
code, so a Keplerian target and a TLE target behave identically downstream.

```{note}
A naive `datetime` is read as UTC rather than rejected or silently treated as
local time, so `tzinfo` is optional. It is still worth passing.
```

## Setting the time

Three spellings, all equivalent:

```python
geom.set_time(2024, 1, 15, 19, 53, 0)                    # separate arguments
geom.set_time((2024, 1, 15, 19, 53, 0))                  # one tuple
geom.set_time(datetime.datetime(2024, 1, 15, 19, 53))    # a datetime
```

For many epochs at once, put a range or array in any one component. Skyfield
vectorises over it, which is far faster than a Python loop over `set_time`:

```python
geom.set_time(2024, 1, 15, 19, 53, range(0, 60))         # 60 consecutive seconds
geom.set_time(2024, 1, 15, range(18, 24), 0, 0)          # six hourly samples
```

```{important}
`Renderer` renders **one epoch**. A vectorised `Geometry` is for cheap
geometric screening — finding the sunlit part of a pass, computing phase
angles — not for handing to the renderer. Build a fresh single-epoch
`Geometry` per frame; see {doc}`performance`.
```

## What it tells you

```python
geom.prop_distance        # slant range, KILOMETRES
geom.phase_angle          # solar phase angle, radians
geom.is_visible()         # False if the Earth is in the way
geom.eclipse_type()       # 1.0 sunlit, 0.0 umbra, between = penumbra
geom.observable_fraction()  # eclipse_type() * is_visible()
```

### Eclipse and occultation are different questions

They are independent, and conflating them is the easiest way to produce a
light curve that is confidently wrong:

**Eclipse** (`eclipse_type`)
: Is sunlight reaching the *spacecraft*? Geometric, with a penumbral
  transition computed from the angular sizes of Sun and Earth, so it returns a
  fraction rather than a flag.

**Occultation** (`is_visible`)
: Is the *observer's* line of sight to the spacecraft clear of the Earth?

A satellite can be fully sunlit and completely invisible (below the horizon),
or high in your sky and entirely dark (eclipsed). `render()` multiplies by the
product of the two, so an unobservable epoch returns exactly zero.

```{note}
`line_of_sight` uses the **WGS84 ellipsoid**, not a sphere. The 21 km
equatorial-to-polar difference matters for a grazing line of sight, which is
precisely the geometry at the start and end of every pass. The earthshine cap
uses a mean sphere instead, where the same difference changes the subtended
solid angle by 0.02% and is not worth the complication.
```

An `atmosphere_km` argument raises the effective limb, if you want to exclude
sight lines through the troposphere:

```python
geom.is_visible(atmosphere_km=20.0)
```

## Finding a pass worth rendering

A satellite is only interesting when it is **above the horizon and sunlit**.
`next_visibility` finds the horizon part:

```python
passes = geom.next_visibility((2024, 1, 15, 18, 0, 0), N=24, i=20.0)
for year, month, day, hour, minute, seconds in passes:
    print(f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d} UTC")
```

`N` is the number of hours to search; `i` the minimum elevation in degrees.

```{warning}
`next_visibility` tests elevation only. It will happily return a pass that is
entirely in eclipse. Check `eclipse_type()` across each candidate before
rendering it.
```

## ViewGeometry: geometry without an orbit

Plenty of legitimate work has no orbit behind it — BRDF studies, phase-angle
sweeps, attitude scans. `ViewGeometry` is the minimal interface the renderer
needs:

```python
from satlight import ViewGeometry

vg = ViewGeometry(
    sun_lvlh=[0.0, 0.0, 1.0],
    observer_lvlh=[0.5, 0.0, 0.866],
    distance_km=550.0,
    altitude_km=550.0,       # only earthshine needs this
)
r = Renderer(obj_path, geometry=vg)
```

```{important}
`distance_km` is in **kilometres**, matching `Geometry.prop_distance`. The
renderer converts once, internally. Getting this wrong by a factor of 1000 is
a factor of $10^6$ in flux — 15 magnitudes — with a perfectly normal-looking
image to show for it. That conversion living in exactly one place is why the
old explicit three-vector constructor was removed.
```

Two consequences of there being no orbit:

- **Observability cannot be evaluated.** `ViewGeometry` has no idea where the
  Earth is, so `observable_fraction` is 1.0 and eclipse is your
  responsibility.
- **Earthshine needs `altitude_km`.** Without it the Earth's angular size is
  unknown — the slant range alone does not give it — and
  {meth}`~satlight.raytracer.Renderer.compute_earthshine` cannot proceed.
