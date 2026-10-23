# Contributing

## Development install

```bash
git clone https://github.com/lfbeesley/satlight.git
cd satlight
conda create -n satlight python=3.11 && conda activate satlight
pip install -e .
pip install pytest
```

## The test tiers

`validation/` and `tests/` do different jobs and should not be merged.

**`tests/`**
: Pass/fail, fast, deterministic, offline. It answers *"does the code still
  compute what it computed yesterday, and does it agree with an independent
  reference?"*

**`validation/`**
: Research — long renders, real MMT tracks, inversions, paper figures. Slow,
  exploratory, judged by eye. Nothing there is pass/fail.

Most of satlight's physics depends only on `Geometry.positions`, so it can be
tested with **constructed** vectors — no ephemeris download, no `bpy`, no orbit
propagation. That is where the bulk of the suite lives, and why it runs in
seconds.

| Marker | Needs | Typical runtime |
|---|---|---|
| (none) | numpy only | milliseconds |
| `needs_ephemeris` | skyfield + `de421.bsp` (downloads on first use) | seconds |
| `needs_bpy` / `slow` | Blender and a real render | seconds to minutes |

```bash
pytest                          # fast tier only (the default, set in pyproject)
pytest -m needs_ephemeris       # add the ephemeris tier
pytest -m "slow or needs_bpy"   # the render tier
pytest -m ""                    # everything
```

```{important}
The default `pytest` run deliberately skips Blender and the network. CI and a
pre-commit check can run it in seconds; run `-m ""` before a release.
```

## What the tests are for

Each test file exists because something real went wrong, not to pad coverage.

`test_render.py`
: `render()` against **closed-form truth** — a flat Lambertian plate, and a
  Lambertian sphere's analytic phase function at six phase angles — plus the
  invariants: $1/r^2$, area, albedo, resolution convergence, bit-for-bit
  reproducibility, and an analytic self-shadowing case where the shadowed
  fraction is exactly 0.75. **This is the file to run first when a light curve
  looks wrong.** If it passes, the forward model is sound and the fault is
  upstream in the geometry or downstream in the magnitude conversion.

`test_geometry_frames.py`
: Pins `along_track` to **+velocity** and the (along, cross, zenith) triad to
  **right-handed**. Both were wrong in a published release, and nothing
  errored — a left-handed basis silently mirrors every rendered scene while
  leaving the total flux unchanged.

`test_brdf.py`
: Back-facing geometry must return zero, and energy must not be created. Cheap,
  and catches sign errors that are invisible in a rendered frame.

`test_occultation.py`
: Whether the Earth is between observer and satellite, and that the
  **ellipsoid** is used — a spherical Earth puts every station outside the
  tropics underground.

`test_earthshine_extended.py`
: The discretised cap against the analytic spherical cap.

`test_packaging.py`
: That the declared dependencies match what is actually imported, and that
  bundled data files are really bundled.

`test_tutorial.py`
: That the code in {doc}`tutorial` runs. This has caught a published README
  with an undefined variable in it.

## Writing a test

Two conventions make the suite worth having:

1. **Gate against an independent reference**, not against the current output.
   A regression test that enshrines today's number cannot tell you the number
   was wrong.
2. **If the behaviour is surprising, assert the property that holds** rather
   than the number you measured. The plate's error oscillates with resolution,
   so the test averages over sub-pixel phase; smooth normals make a coarse
   sphere darker, so the test asserts convergence rather than improvement. See
   {doc}`guide/validation`.

## Documentation

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs/source docs/_build/html -W
```

`-W` turns warnings into errors, which is what Read the Docs does
(`fail_on_warning: true`), so a broken cross-reference fails the build rather
than publishing quietly.

```{note}
satlight is **not** pip-installed on the docs builder — that would pull
`bpy`, several hundred MB of Blender. `conf.py` puts the source tree on
`sys.path` and mocks `bpy`/`mathutils` instead, so autodoc still reads every
docstring. If you add a module that imports something new at module scope, add
it to `docs/requirements.txt` or to `autodoc_mock_imports`.
```

## Things known to need doing

- `AB_mag` is in-band only but is not named that way; a broadband form should
  sit beside it. See {doc}`guide/photometry`.
- A number of scripts under `validation/`, `multiaspect/`, `large_spacecraft/`
  and `reflect_orbital/` still use the pre-`geometry=` constructor signature.
- `abg_from_peak` can be driven to a non-conserving parameter set.
- Narrow specular lobes are not guaranteed converged at the default
  resolution.

## Licence

GPL-3.0-or-later. satlight imports `bpy` at module scope, and `bpy` is
GPL-3.0, which makes a distributed satlight a derivative work — so
contributions are under the same licence, and this is not a free choice.
