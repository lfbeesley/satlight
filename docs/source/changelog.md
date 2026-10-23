# Changelog

satlight is pre-1.0 and the API may change between releases.

## 0.1.0a3 (2026-10-23)

### Breaking

- `Renderer` requires `geometry=`, a {class}`~satlight.geometry.Geometry` or
  {class}`~satlight.geometry.ViewGeometry`. The old three-vector constructor
  is gone.
- All `Renderer` options after `geometry` are keyword-only.
- `compute_earthshine` no longer accepts `altitude_km` or `sun_direction`;
  both come from the geometry.
- `geometryBuilder` and `workbench` are no longer part of the package.
- `earth_albedo=None` (the CERES albedo map) is removed.

### Fixed

- Satellites below the horizon rendered at full brightness. `render()` now
  multiplies by `eclipse_type() × is_visible()`.
- The camera always looked along $-Z$ instead of from the observer.
- `add_earthshine=True` had no effect unless `compute_earthshine()` was also
  called. Earthshine is now computed on creation and on every `update()`.
- Eclipse was applied twice in one code path.
- `set_time` with a single tuple produced the wrong epochs.
- A `datetime` without a time zone is read as UTC.
- One Earth radius constant, {data}`~satlight.geometry.R_EARTH_MEAN_KM`,
  replaces three hard-coded values.

### Added

- {meth}`~satlight.raytracer.Renderer.set_attitude`.
- `apply_observability=` option.
- {class}`~satlight.geometry.ViewGeometry`.
- Earthshine from a grid of Earth patches instead of a single direction.
- Tests against analytic results for a plate and a sphere.

### Changed

- The pip package contains only the code; example models are in the
  repository's `models/` directory.
- Dependencies reduced to numpy, astropy, skyfield, sgp4, bpy, mathutils,
  hide_warnings and plotly.
- Licence declared as GPL-3.0-or-later.
- New documentation.

## 0.1.0a1

First pre-release.
