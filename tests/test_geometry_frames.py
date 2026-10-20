"""LVLH frame conventions.

These exist because the frame was wrong in a published release: along_track was
built as cross(orbit_normal, radial) with `radial` the NADIR vector, which
returns -velocity and makes the triad left-handed. Nothing errored. Renders
succeeded, magnitudes looked plausible, and a body held at fixed Euler angles
came out ~0.8 mag off - about 16x the photometric noise floor.

A flipped frame produces no exception, so only an assertion against an
INDEPENDENT reference catches it. Velocity comes from skyfield here, not from
anything satlight computes.

Keep these green across any refactor of the vector plumbing.
"""
import datetime

import numpy as np
import pytest

pytestmark = pytest.mark.needs_ephemeris

A_KM = 6371.0 + 10000.0
EPOCH = datetime.datetime(2026, 11, 15, 18, 0, 0, tzinfo=datetime.timezone.utc)
EPOCHS = [EPOCH + datetime.timedelta(minutes=m) for m in (0, 37, 115, 240)]


@pytest.fixture(params=EPOCHS, ids=lambda t: t.strftime('%H%M'))
def geom(request, ephemeris):
    from satlight.geometry import Geometry
    t = request.param
    g = Geometry()
    g.create_observer(28.7622, -17.8897, 2396.0)
    g.create_satellite_from_elements(a_km=A_KM, e=0.0, i_deg=56.0, raan_deg=300.0,
                                     argp_deg=0.0, nu_deg=0.0, epoch=EPOCH)
    g.set_time(t.year, t.month, t.day, t.hour, t.minute, t.second)
    return g


def _state(g):
    r = np.asarray(g.positions['satellite'], float)
    v = np.asarray(g.satellite.at(g.time).velocity.km_per_s, float)
    return r / np.linalg.norm(r), v / np.linalg.norm(v)


def test_along_track_points_along_velocity(geom):
    """The bug: this was -1. Velocity is taken from skyfield, independently."""
    _, vhat = _state(geom)
    assert float(geom.along_track @ vhat) == pytest.approx(1.0, abs=1e-3)


def test_zenith_points_away_from_earth(geom):
    rhat, _ = _state(geom)
    assert float(geom.zenith @ rhat) == pytest.approx(1.0, abs=1e-9)
    assert float(geom.nadir @ rhat) == pytest.approx(-1.0, abs=1e-9)


def test_cross_track_is_the_orbit_normal(geom):
    rhat, vhat = _state(geom)
    n = np.cross(rhat, vhat)
    n /= np.linalg.norm(n)
    assert float(geom.cross_track @ n) == pytest.approx(1.0, abs=1e-6)


def test_triad_is_right_handed(geom):
    """along x cross = zenith, i.e. det(+1).

    A left-handed basis silently mirrors every rendered scene. Total flux
    survives a mirror, so this never shows up as an error - only as an attitude
    that means something other than what it says.
    """
    assert float(np.cross(geom.along_track, geom.cross_track) @ geom.zenith) == \
        pytest.approx(1.0, abs=1e-9)
    R = np.stack([geom.along_track, geom.cross_track, geom.zenith], axis=1)
    assert float(np.linalg.det(R)) == pytest.approx(1.0, abs=1e-9)


def test_triad_is_orthonormal(geom):
    R = np.stack([geom.along_track, geom.cross_track, geom.zenith], axis=1)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)


def test_incident_vector_points_at_the_sun(geom):
    """Documented as 'unit vector from satellite toward sun'; BRDFs need n.l>0."""
    to_sun = np.asarray(geom.positions['sun'], float) - np.asarray(
        geom.positions['satellite'], float)
    to_sun /= np.linalg.norm(to_sun)
    assert float(np.asarray(geom.incident_vector, float) @ to_sun) == \
        pytest.approx(1.0, abs=1e-9)


def test_outgoing_vector_points_at_the_observer(geom):
    to_obs = np.asarray(geom.positions['observer'], float) - np.asarray(
        geom.positions['satellite'], float)
    to_obs /= np.linalg.norm(to_obs)
    assert float(np.asarray(geom.outgoing_vector, float) @ to_obs) == \
        pytest.approx(1.0, abs=1e-9)


def test_lvlh_components_are_consistent_with_the_basis(geom):
    """*_lvlh must be the GCRS vector resolved on (along, cross, zenith)."""
    R = np.stack([geom.along_track, geom.cross_track, geom.zenith], axis=1)
    for gcrs, lvlh in ((geom.incident_vector, geom.incident_vector_lvlh),
                       (geom.outgoing_vector, geom.outgoing_vector_lvlh)):
        assert np.allclose(R.T @ np.asarray(gcrs, float).ravel(),
                           np.asarray(lvlh, float).ravel(), atol=1e-9)


def test_lvlh_transform_preserves_angles(geom):
    """A rotation preserves the sun-observer angle; a reflection would too, so
    this is a weaker check than handedness - it guards against scaling or a
    non-orthogonal basis sneaking in."""
    a = float(np.asarray(geom.incident_vector, float).ravel()
              @ np.asarray(geom.outgoing_vector, float).ravel())
    b = float(np.asarray(geom.incident_vector_lvlh, float).ravel()
              @ np.asarray(geom.outgoing_vector_lvlh, float).ravel())
    assert a == pytest.approx(b, abs=1e-9)
