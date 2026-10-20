# tests/

`validation/` and `tests/` do different jobs and should not be merged.

**`validation/`** is research: long renders, real MMT tracks, inversions, figures
for the paper. Slow, exploratory, judged by eye. Nothing there is pass/fail.

**`tests/`** is pass/fail, fast, deterministic, and offline. It answers "does
the code still compute what it computed yesterday, and does it agree with an
independent reference".

## Tiers

Most of the physics here depends only on `Geometry.positions`, so it can be
tested with CONSTRUCTED vectors - no ephemeris download, no `bpy`, no orbit
propagation. That is where the bulk of the suite lives and why it runs in
seconds.

| marker | needs | typical runtime |
|---|---|---|
| (none) | numpy only | milliseconds |
| `needs_ephemeris` | skyfield + de421.bsp (downloads on first use) | seconds |
| `needs_bpy` / `slow` | Blender + a render | seconds to minutes |

    pytest                          # fast tier only (default, see pyproject)
    pytest -m needs_ephemeris       # add the ephemeris tier
    pytest -m "slow or needs_bpy"   # the render tier
    pytest -m ""                    # everything

## Why these particular tests

Each one exists because something real went wrong, not to pad coverage:

- `test_geometry_frames.py` - `along_track` points ANTI-velocity and the LVLH
  triad is left-handed. Currently `xfail`: the test states the standard
  convention and fails until a decision is made, rather than silently blessing
  the present behaviour.
- `test_brdf.py` - backfacing geometry must return zero, and energy must not be
  created. Cheap, and catches sign errors that are invisible in a rendered frame.
- `test_render.py` - `render()` against CLOSED-FORM truth: a flat Lambertian
  plate, and a Lambertian sphere's analytic phase function (geometric albedo
  2*rho/3, `Phi = [sin a + (pi-a) cos a]/pi`) at six phase angles. Plus the
  invariants - 1/r^2, area, albedo, resolution convergence, bit-for-bit
  reproducibility - and an analytic self-shadowing case where the shadowed
  fraction is exactly 0.75. This is the file to run first when a lightcurve
  looks wrong: if it passes, the forward model is sound and the fault is
  upstream in the geometry or downstream in the magnitude conversion.

  Two things it documents by measurement rather than assumption. A sharp-edged
  plate's error OSCILLATES inside a +/-1-pixel-row envelope instead of
  converging, so the tight gate averages over sub-pixel phase. And smooth
  shading normals make a coarse sphere slightly DARKER, not brighter, so the
  test asserts convergence between smooth and flat, not an improvement.

### Not yet written

- `test_time.py` - the tuple form of `set_time()` is documented but makes
  skyfield read the tuple as an array of years, yielding six epochs including
  year 0. Should fail loudly instead of silently giving wrong answers.
- `test_occultation.py` - nothing checks whether the Earth is between observer
  and satellite, so a target below the horizon renders at full brightness.
  Should also pin the ellipsoid: a spherical Earth puts every station outside
  the tropics underground.
