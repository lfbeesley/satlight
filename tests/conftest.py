"""Shared fixtures. Heavy dependencies are optional and skipped, not required."""
import datetime
import os

import numpy as np
import pytest


def pytest_configure(config):
    for m in ('slow: takes seconds or more',
              'needs_bpy: requires Blender',
              'needs_ephemeris: requires skyfield and a JPL ephemeris'):
        config.addinivalue_line('markers', m)


@pytest.fixture(scope='session')
def ephemeris():
    """Skip rather than fail when the ephemeris is unavailable (e.g. offline CI)."""
    skyfield = pytest.importorskip('skyfield.api')
    try:
        loader = skyfield.Loader(os.path.join(os.path.expanduser('~'), '.satlight'))
        return loader.timescale(), loader('de421.bsp')
    except Exception as exc:                       # network down, file missing
        pytest.skip(f'ephemeris unavailable: {exc}')


@pytest.fixture(scope='session')
def bpy_available():
    pytest.importorskip('bpy')
    return True


@pytest.fixture
def bare_geometry():
    """A Geometry with positions set by hand and no orbit propagation.

    The geometric methods - eclipse_type, line_of_sight, is_visible - read only
    `positions` and `incident_vector`, so they can be exercised exactly and
    instantly without skyfield. Anything needing a real orbit belongs in the
    needs_ephemeris tier.
    """
    from satlight.geometry import Geometry

    def build(sat_km, obs_km, sun_km=(1.496e8, 0.0, 0.0)):
        g = Geometry.__new__(Geometry)
        sat = np.asarray(sat_km, float)
        obs = np.asarray(obs_km, float)
        sun = np.asarray(sun_km, float)
        g.positions = {'satellite': sat, 'observer': obs, 'sun': sun}
        d = sun - sat
        g.incident_vector = d / np.linalg.norm(d)
        g.outgoing_vector = (obs - sat) / np.linalg.norm(obs - sat)
        g.prop_distance = np.linalg.norm(obs - sat)
        return g

    return build


R_EQ = 6378.137
R_POL = 6378.137 * (1.0 - 1.0 / 298.257223563)
