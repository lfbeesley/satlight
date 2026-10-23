# Changelog

satlight is pre-1.0 and the API is not yet stable. Breaking changes are listed
first in each release.

## 0.1.0a3 — unreleased

### Breaking

- **`Renderer` requires `geometry=`.** The old explicit three-vector
  constructor is gone, and passing its arguments raises `TypeError` rather
  than being silently ignored. Pass a {class}`~satlight.geometry.Geometry`, or
  a {class}`~satlight.geometry.ViewGeometry` for work with no orbit behind it.
  The reason is the kilometres-to-metres conversion: it used to sit in every
  caller, and getting it wrong is a factor of $10^6$ in flux with a perfectly
  normal-looking image to show for it. It now lives in one place.
- All `Renderer` options after `geometry` are **keyword-only**.
- `compute_earthshine` no longer accepts `altitude_km` or `sun_direction`;
  both come off the geometry.
- `geometryBuilder` and `workbench` are no longer part of the package. They
  are development tools and live under `dev/`.

### Fixed

- **Occultation was not applied.** A satellite below the horizon rendered at
  full brightness, because the renderer sees only direction vectors and a
  range, and none of them knows where the Earth is. `render()` now scales by
  `observable_fraction()` — `eclipse_type() × is_visible()` — and
  short-circuits to zeros when that is 0.0. See {doc}`guide/renderer`.
- **`add_camera` rendered from nadir.** It hardcoded the $-Z$ axis, so
  `initialise_scene → add_camera → render` ignored the observer direction it
  was given; for an oblique observer that meant looking into the back of the
  model and returning zero flux. Both placement paths now share one
  `_place_camera`.
- **Earthshine was silently dropped.** `add_earthshine=True` set a flag and
  nothing else — the term was lost unless the caller also remembered to call
  `compute_earthshine()`. It is now computed in the constructor and again on
  every `update()`, which also stops it going stale as the satellite moves.
  Setting the flag on an existing `Renderer` raises rather than failing quietly.
- **Eclipse was applied twice** in one path.
- `set_time` now accepts a single tuple or a `datetime`. The tuple form was
  documented but made Skyfield read the tuple as an array of *years*, yielding
  six epochs including year 0.
- A naive `datetime` is read as UTC rather than being misinterpreted.
- Three separate hardcoded Earth radii replaced by one
  {data}`~satlight.geometry.R_EARTH_MEAN_KM`.

### Added

- {meth}`~satlight.raytracer.Renderer.set_attitude` — orient the body in LVLH.
  The rotation is applied to the **imported** pose, never the current one, so
  repeated calls do not compound.
- `apply_observability=` to opt out of the observability factor.
- {class}`~satlight.geometry.ViewGeometry` — the minimal viewing geometry, for
  BRDF studies, phase sweeps and attitude scans.
- A 67-line module docstring on `satlight.brdf` documenting the calling
  contract — in particular that a BRDF returns $f_r\cos\theta_i$, **not**
  $f_r$. See {doc}`guide/materials`.
- Earthshine as a true extended source: equal-**solid-angle** cap
  discretisation, replacing a single lumped direction. Equal-area
  discretisation under-resolves the limb and reads low at LEO.
- Test suite grown to 172 tests across tiered markers, including closed-form
  gates for a plate and a sphere. See {doc}`guide/validation`.

### Changed

- **The wheel is code only**: `raytracer`, `brdf`, `geometry` and `tools`. No
  models and no data files are shipped. Example meshes are in the repository's
  `models/` directory.
- **The CERES albedo map is removed**, along with `earth_albedo=None`. The
  Earth is a uniform albedo, `earth_albedo=0.3` by default; `None` now raises
  `ValueError`.
- **Dependencies cut from 13 to 7.** `trimesh`, `tqdm`, `xarray` and `netCDF4`
  were declared but never imported — left over from an unfinished earthshine
  path, and `netCDF4` alone pulled in binary HDF5 wheels for nothing. About
  177 MB off a fresh install. `matplotlib` and `plotly` moved to the
  `[preview]` extra, since only one method uses them and it imports them
  lazily.
- Licence declared as `GPL-3.0-or-later` in PEP 639 SPDX form. satlight
  imports `bpy` at module scope and `bpy` is GPL-3.0, which makes a
  distributed satlight a derivative work.
- `__version__` is read from installed metadata, so it cannot drift from
  `pyproject.toml`.
- Documentation rebuilt: a user guide covering the geometry, materials, the
  renderer, earthshine, photometry, performance and validation.

### Known issues

- `tools.AB_mag` is an in-band magnitude but is not named that way, and is
  ~2.2 mag too bright on broadband `render()` output. See
  {doc}`guide/photometry`.
- `abg_from_peak` can produce a non-conserving parameter set.
- Narrow specular lobes are not guaranteed converged at the default
  resolution.
- No built-in parallelism. See {doc}`guide/performance` for what to do from
  outside the package.

## 0.1.0a1

First published pre-release.
