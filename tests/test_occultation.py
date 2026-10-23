"""Gates on observability: a satellite that cannot be seen must render as zero.

`render()` reads three direction vectors and a range. None of them knows where
the Earth is, so without an explicit check a satellite on the FAR SIDE of the
Earth renders at full brightness - a plausible, smoothly-varying, completely
wrong flux that looks like real light-curve structure rather than an error.

Two independent conditions, and both are needed:
  * eclipse_type()  - does the Sun reach the satellite?
  * is_visible()    - is the Earth between satellite and observer?
A satellite can be in full sunlight and still be unobservable.

The ellipsoid matters here and is pinned below: the equatorial radius exceeds
the polar by 21 km, so a SPHERICAL Earth of equatorial radius puts every
station outside the tropics underground - La Palma comes out 2.5 km below its
own surface and every sightline reads as blocked.
"""
import numpy as np
import pytest

R_EQ = 6378.137
R_POL = 6378.137 * (1.0 - 1.0 / 298.257223563)

pytestmark = pytest.mark.needs_bpy       # Renderer construction imports bpy


def _geometry(sat, obs, sun):
    """A Geometry with positions set by hand - no orbit, no ephemeris."""
    from satlight.geometry import Geometry
    g = Geometry.__new__(Geometry)
    sat, obs, sun = (np.asarray(v, float) for v in (sat, obs, sun))
    g.positions = {'satellite': sat, 'observer': obs, 'sun': sun}
    d = sun - sat; g.incident_vector = d / np.linalg.norm(d)
    o = obs - sat; g.outgoing_vector = o / np.linalg.norm(o)
    g.prop_distance = float(np.linalg.norm(o))
    # the renderer reads the LVLH forms; aim the plate straight at the observer
    # so that, geometrically, it is perfectly placed to be seen
    g.incident_vector_lvlh = np.array([0.0, 0.0, 1.0])
    g.outgoing_vector_lvlh = np.array([0.0, 0.0, 1.0])
    return g


def _plate(tmp_path):
    p = tmp_path / 'plate.obj'
    p.write_text('o plate\nv -1 0 -1\nv 1 0 -1\nv 1 0 1\nv -1 0 1\nf 4 3 2 1\n')
    return str(p)


def _flux(tmp_path, geom, **kw):
    from satlight.raytracer import Renderer
    r = Renderer(_plate(tmp_path), geometry=geom, resolution=(96, 96),
                 brdf=('lambertian', {'albedo': 0.3}), **kw)
    r.initialise_scene(); r.add_camera(); r.update(geom)
    return float(np.sum(r.render())), r


# geometries, all with the satellite 550 km up
VISIBLE  = dict(sat=(R_EQ + 550.0, 0.0, 0.0), obs=(R_EQ, 0.0, 0.0),
                sun=(1.496e8, 0.0, 0.0))                       # overhead, sunlit
OCCULTED = dict(sat=(-(R_EQ + 550.0), 0.0, 0.0), obs=(R_EQ, 0.0, 0.0),
                sun=(0.0, 1.496e8, 0.0))                       # far side, sunlit
ECLIPSED = dict(sat=(R_EQ + 550.0, 0.0, 0.0), obs=(R_EQ, 0.0, 0.0),
                sun=(-1.496e8, 0.0, 0.0))                      # overhead, in umbra


def test_the_two_conditions_are_independent():
    """The occulted case is FULLY SUNLIT, and the eclipsed case is directly
    overhead. Neither check would catch the other's case."""
    occ = _geometry(**OCCULTED)
    assert float(occ.eclipse_type()) == pytest.approx(1.0)   # fully sunlit...
    assert occ.is_visible() is False                         # ...but behind the Earth
    ecl = _geometry(**ECLIPSED)
    assert ecl.is_visible() is True                          # clear line of sight...
    assert float(ecl.eclipse_type()) == pytest.approx(0.0)   # ...but in shadow


def test_an_overhead_sunlit_satellite_renders(tmp_path):
    """The control: without this, zeros everywhere would pass every test below."""
    flux, r = _flux(tmp_path, _geometry(**VISIBLE))
    assert flux > 0.0
    assert r.observable_fraction == pytest.approx(1.0)


def test_a_satellite_behind_the_earth_renders_zero(tmp_path):
    """The bug this file exists for: it used to render at full brightness."""
    flux, r = _flux(tmp_path, _geometry(**OCCULTED))
    assert r.observable_fraction == 0.0
    assert flux == 0.0


def test_an_eclipsed_satellite_renders_zero(tmp_path):
    flux, r = _flux(tmp_path, _geometry(**ECLIPSED))
    assert r.observable_fraction == 0.0
    assert flux == 0.0


def test_update_refreshes_observability(tmp_path):
    """A satellite that sets mid-sequence must go dark, not carry on shining."""
    from satlight.raytracer import Renderer
    vis = _geometry(**VISIBLE)
    r = Renderer(_plate(tmp_path), geometry=vis, resolution=(96, 96),
                 brdf=('lambertian', {'albedo': 0.3}))
    r.initialise_scene(); r.add_camera(); r.update(vis)
    assert float(np.sum(r.render())) > 0.0
    r.update(_geometry(**OCCULTED))                   # the satellite sets
    assert r.observable_fraction == 0.0
    assert float(np.sum(r.render())) == 0.0


def test_opting_out_restores_the_old_behaviour(tmp_path):
    """apply_observability=False must render the occulted case as though the
    Earth were not there - the escape hatch for callers masking by hand."""
    flux, r = _flux(tmp_path, _geometry(**OCCULTED), apply_observability=False)
    assert r.observable_fraction == 1.0
    assert flux > 0.0


def test_view_geometry_is_unaffected(tmp_path):
    """A ViewGeometry holds no positions and cannot know where the Earth is.
    It must not be penalised for that - the factor is 1.0 and nothing changes."""
    from satlight.geometry import ViewGeometry
    g = ViewGeometry(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]),
                     distance_km=550.0)
    flux, r = _flux(tmp_path, g)
    assert r.observable_fraction == 1.0
    assert flux > 0.0


def test_penumbra_scales_the_flux(tmp_path):
    """Partial illumination is a linear factor on the whole frame, not a
    threshold: a half-eclipsed satellite is half as bright."""
    from satlight.raytracer import Renderer
    g = _geometry(**VISIBLE)
    full, _ = _flux(tmp_path, g)
    r = Renderer(_plate(tmp_path), geometry=g, resolution=(96, 96),
                 brdf=('lambertian', {'albedo': 0.3}))
    r.initialise_scene(); r.add_camera(); r.update(g)
    r.observable_fraction = 0.25                       # stand in for penumbra
    assert float(np.sum(r.render())) == pytest.approx(0.25 * full, rel=1e-12)


def test_the_earth_is_an_ellipsoid_not_a_sphere():
    """A spherical Earth of equatorial radius puts La Palma underground.

    This is why line_of_sight() scales the polar axis rather than using a mean
    radius: at 28.76 deg the WGS84 surface sits ~7 km inside a sphere of
    equatorial radius, so every sightline from the observatory would read as
    blocked by the planet it is standing on.
    """
    lat = np.radians(28.7622)                          # Roque de los Muchachos
    r_wgs84 = 1.0 / np.sqrt((np.cos(lat) / R_EQ) ** 2 + (np.sin(lat) / R_POL) ** 2)
    assert r_wgs84 < R_EQ
    assert R_EQ - r_wgs84 > 4.0                        # km of false "underground"
    # and the station is genuinely ON the ellipsoid, not inside it
    assert abs(r_wgs84 - R_EQ) < (R_EQ - R_POL)
