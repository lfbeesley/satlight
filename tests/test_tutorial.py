"""The tutorial must keep working.

It is the only onramp a new user has, and it was previously broken in two
places at once: its elements example passed a NAIVE datetime
(`TypeError: can't subtract offset-naive and offset-aware datetimes`) and it
documented `set_time((y, m, d, ...))` as a TUPLE, which skyfield reads as an
array of YEARS - 2024, then 1, then 15 - and which fails with
`EphemerisRangeError`. Someone following the page could not get past step one.

These tests run the page's own code so it cannot rot again.
"""
import datetime
import pathlib
import re

import numpy as np
import pytest

# The tutorial model, kept in the repo rather than shipped in the wheel.
MODEL = pathlib.Path(__file__).resolve().parents[1] / 'models' / 'reference' / 'spacex_ai1.obj'

TUTORIAL = pathlib.Path(__file__).resolve().parent.parent / 'docs' / 'source' / 'tutorial.md'
LAT, LON, ALT_M = 28.7622, -17.8897, 2396.0
LINE1 = "1 25544U 98067A   24015.50000000  .00016717  00000-0  10270-3 0  9990"
LINE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.49815174 10000"


def _geometry_at(when):
    from satlight.geometry import Geometry
    g = Geometry()
    g.create_observer(observer_lat=LAT, observer_lon=LON, observer_alt_m=ALT_M)
    g.create_satellite(LINE1, LINE2)
    g.set_time(when.year, when.month, when.day,
               when.hour, when.minute, when.second)
    return g


# --------------------------------------------------- the text itself
def test_tutorial_shows_set_time_forms_that_all_work():
    """The page used to document only the positional form and warn about the
    tuple. Both now work, so it shows both - and must keep doing so."""
    src = TUTORIAL.read_text()
    assert 'set_time(2024, 1, 15, 19, 53, 0)' in src
    assert 'set_time((2024, 1, 15, 19, 53, 0))' in src


def test_tutorial_documents_the_brdf_name_matching_rule():
    """How brdf keys map to objects is the single most important thing to
    know, and it used to appear nowhere."""
    src = TUTORIAL.read_text().lower()
    assert 'case-insensitiv' in src
    assert "'default'" in src or '`default`' in src


# --------------------------------------------------- the code it shows
@pytest.mark.needs_ephemeris
def test_the_geometry_sequence_runs():
    """Observer, TLE, time - and the answers the page prints."""
    g = _geometry_at(datetime.datetime(2024, 1, 15, 19, 53, 0))
    assert g.prop_distance == pytest.approx(1229.0, rel=0.02)
    assert g.is_visible() is True
    assert float(g.eclipse_type()) == pytest.approx(1.0)


@pytest.mark.needs_ephemeris
def test_the_elements_example_runs_with_a_timezone_aware_epoch():
    from satlight.geometry import Geometry
    g = Geometry()
    g.create_observer(observer_lat=LAT, observer_lon=LON, observer_alt_m=ALT_M)
    g.create_satellite_from_elements(
        a_km=6371.0 + 35786.0, e=0.0, i_deg=0.0, raan_deg=0.0, argp_deg=0.0,
        nu_deg=100.0,
        epoch=datetime.datetime(2026, 8, 15, 22, 30,
                                tzinfo=datetime.timezone.utc))
    g.set_time(2026, 8, 15, 22, 30, 0)
    assert 30000.0 < g.prop_distance < 50000.0        # a GEO slant range


@pytest.mark.needs_ephemeris
def test_next_visibility_finds_the_pass_the_tutorial_uses():
    from satlight.geometry import Geometry
    g = Geometry()
    g.create_observer(observer_lat=LAT, observer_lon=LON, observer_alt_m=ALT_M)
    g.create_satellite(LINE1, LINE2)
    passes = g.next_visibility((2024, 1, 15, 18, 0, 0), N=24, i=20.0)
    assert passes, 'no passes found'
    starts = {(p[3], p[4]) for p in passes}
    assert (19, 53) in starts, f'the 19:53 pass is gone: {sorted(starts)}'


@pytest.mark.needs_bpy
@pytest.mark.needs_ephemeris
@pytest.mark.slow
def test_the_light_curve_brightens_as_the_pass_closes():
    """The page's headline result: the satellite brightens as the range falls
    and the phase angle drops, then goes to exactly zero in eclipse."""
    from satlight.raytracer import Renderer

    brdf = {'solar':    ('lambertian', {'albedo': 0.08}),
            'radiator': ('lambertian', {'albedo': 0.60}),
            'boom':     ('lambertian', {'albedo': 0.20}),
            'bus':      ('lambertian', {'albedo': 0.30}),
            'default':  ('lambertian', {'albedo': 0.25})}
    t0 = datetime.datetime(2024, 1, 15, 19, 53, 0)
    geom = _geometry_at(t0)
    r = Renderer(str(MODEL), geometry=geom, resolution=(160, 160),
                 brdf=brdf, add_shadows=True)
    r.initialise_scene()
    r.add_camera()
    r.set_attitude(euler_xyz=(0.0, 0.0, np.radians(30.0)))

    mags = []
    for step in range(6):
        g = _geometry_at(t0 + datetime.timedelta(seconds=30 * step))
        r.update(g)
        flux = float(np.sum(r.render()))
        assert flux > 0.0, f'step {step} unexpectedly dark'
        mags.append(-2.5 * np.log10(flux) - 18.915)

    assert mags[-1] < mags[0], 'the pass should brighten as it closes'
    assert mags[0] - mags[-1] > 1.0, f'expected >1 mag of brightening, got {mags}'


@pytest.mark.needs_bpy
def test_set_attitude_does_not_compound():
    """Called twice with the same angle, the pose must be the same as once -
    otherwise a per-frame call in a light-curve loop spins the spacecraft."""
    import bpy
    from satlight.geometry import ViewGeometry
    from satlight.raytracer import Renderer

    g = ViewGeometry(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]),
                     distance_km=550.0)
    r = Renderer(str(MODEL), geometry=g, resolution=(64, 64))
    r.initialise_scene()
    r.add_camera()
    r.set_attitude(euler_xyz=(0.0, 0.0, np.radians(30.0)))
    once = {o.name: np.array(o.matrix_world)
            for o in bpy.context.scene.objects if o.type == 'MESH'}
    r.set_attitude(euler_xyz=(0.0, 0.0, np.radians(30.0)))
    twice = {o.name: np.array(o.matrix_world)
             for o in bpy.context.scene.objects if o.type == 'MESH'}
    for name in once:
        assert np.allclose(once[name], twice[name]), f'{name} moved on the 2nd call'


@pytest.mark.needs_bpy
def test_the_tutorial_model_exists_and_has_the_documented_components():
    text = pathlib.Path(str(MODEL)).read_text()
    names = {ln.split(maxsplit=1)[1].strip() for ln in text.splitlines()
             if ln.startswith('o ')}
    assert 'bus' in names
    assert sum(n.startswith('solar') for n in names) == 6
    assert sum(n.startswith('radiator') for n in names) == 3


# ------------------------------------------- the traps, fixed at source
@pytest.mark.needs_ephemeris
def test_set_time_accepts_positional_tuple_and_datetime_alike():
    """`set_time((y, m, d, ...))` was advertised in its own docstring and did
    not work: skyfield read the one sequence as an array of YEARS, giving six
    epochs or EphemerisRangeError. All three spellings must now agree."""
    want = _geometry_at(datetime.datetime(2024, 1, 15, 19, 53, 0))

    from satlight.geometry import Geometry

    def built(setter):
        g = Geometry()
        g.create_observer(observer_lat=LAT, observer_lon=LON, observer_alt_m=ALT_M)
        g.create_satellite(LINE1, LINE2)
        setter(g)
        return g

    as_tuple = built(lambda g: g.set_time((2024, 1, 15, 19, 53, 0)))
    as_dt = built(lambda g: g.set_time(datetime.datetime(2024, 1, 15, 19, 53, 0)))
    for other in (as_tuple, as_dt):
        assert np.shape(other.positions['satellite']) == (3,), 'not a single epoch'
        assert np.allclose(other.positions['satellite'],
                           want.positions['satellite'])


@pytest.mark.needs_ephemeris
def test_set_time_still_supports_many_epochs_from_a_range():
    """Unpacking a tuple must not break the array form it coexists with."""
    from satlight.geometry import Geometry
    g = Geometry()
    g.create_observer(observer_lat=LAT, observer_lon=LON, observer_alt_m=ALT_M)
    g.create_satellite(LINE1, LINE2)
    g.set_time(2024, 1, 15, 19, 53, range(0, 4))
    assert np.shape(g.positions['satellite']) == (3, 4)


@pytest.mark.needs_ephemeris
def test_elements_epoch_accepts_a_naive_datetime():
    """A naive epoch used to raise 'can't subtract offset-naive and
    offset-aware datetimes'. Everything here is UTC, so read it as UTC."""
    from satlight.geometry import Geometry
    g = Geometry()
    g.create_observer(observer_lat=LAT, observer_lon=LON, observer_alt_m=ALT_M)
    g.create_satellite_from_elements(
        a_km=6371.0 + 35786.0, e=0.0, i_deg=0.0, raan_deg=0.0, argp_deg=0.0,
        nu_deg=100.0, epoch=datetime.datetime(2026, 8, 15, 22, 30))   # naive
    g.set_time(2026, 8, 15, 22, 30, 0)
    assert 30000.0 < g.prop_distance < 50000.0


# ------------------------------------------------------- the README
README = pathlib.Path(__file__).resolve().parent.parent / 'README.md'


@pytest.mark.needs_ephemeris
def test_readme_tle_example_gives_the_numbers_it_prints():
    """The README shows concrete outputs next to its code. If the code stops
    producing them the page is lying, which is worse than having no page."""
    g = _geometry_at(datetime.datetime(2024, 1, 15, 19, 53, 0))
    src = README.read_text()
    assert f'{g.prop_distance:.1f}' in src, 'the printed slant range is stale'
    assert g.is_visible() is True
    assert float(g.eclipse_type()) == pytest.approx(1.0)


@pytest.mark.needs_ephemeris
def test_readme_elements_example_gives_the_number_it_prints():
    from satlight.geometry import Geometry
    g = Geometry()
    g.create_observer(observer_lat=LAT, observer_lon=LON, observer_alt_m=ALT_M)
    g.create_satellite_from_elements(
        a_km=6371.0 + 35786.0, e=0.0, i_deg=0.0, raan_deg=0.0, argp_deg=0.0,
        nu_deg=100.0, epoch=datetime.datetime(2026, 8, 15, 22, 30))
    g.set_time(2026, 8, 15, 22, 30, 0)
    assert f'{g.prop_distance:.1f}' in README.read_text(), \
        'the printed GEO range is stale'


def test_readme_documents_the_brdf_name_matching_rule():
    src = README.read_text().lower()
    assert 'case-insensitiv' in src
    assert 'matched by object name' in src or 'matches if it appears' in src
