"""Gates for the extended-source earthshine in Renderer.

The defect these pin down: earthshine used to be collapsed into ONE effective
direction with one total irradiance. For a source 68 deg wide at LEO that is
wrong in a specific, silent way - a facet perpendicular to the lumped direction
collects exactly zero when it physically still sees half the Earth's disc. It
also used to discretise the Earth in equal surface AREA, which concentrates the
integral beneath the satellite and reads low at LEO.

These tests do not need bpy for the geometry parts; the render test is marked.
"""
import numpy as np
import pytest

from satlight.geometry import ViewGeometry
from satlight.raytracer import Renderer

R_EARTH = 6371.0


def _vg(sun_lvlh, alt_km=500.0):
    """Minimal geometry for an earthshine test: only the Sun and the altitude
    matter, so the observer direction is a placeholder."""
    return ViewGeometry(sun_lvlh, (0.0, 0.0, -1.0),
                        distance_km=alt_km, altitude_km=alt_km)


def _bare(**kw):
    """A Renderer with no scene, for exercising the earthshine geometry alone."""
    r = Renderer.__new__(Renderer)
    r.solar_constant = 1361.0
    r.earth_albedo = 0.3
    r.nadir_direction = np.array([0.0, 0.0, -1.0])
    r.earthshine_n_rings = 4
    r._earth_cap_cache = {}
    r.earthshine_dirs = None
    r.earthshine_patch_irradiance = None
    r.earthshine_irradiance = 0.0
    r.earthshine_direction = None
    for k, v in kw.items():
        setattr(r, k, v)
    return r


@pytest.mark.parametrize('alt_km', [100.0, 400.0, 550.0, 1200.0, 35786.0])
def test_cap_solid_angle_matches_the_analytic_sphere(alt_km):
    """The discretised cap must carry the analytic solid angle it subtends.

    This is the normalisation that ties the sum back to the whole Earth sphere.
    """
    r = _bare()
    _, omegas, _ = r._earth_cap(alt_km, 8)
    analytic = 2 * np.pi * (1 - np.cos(np.arcsin(R_EARTH / (R_EARTH + alt_km))))
    assert omegas.sum() == pytest.approx(analytic, rel=1e-3)


def test_subsolar_leo_irradiance_is_physical():
    """Over the subsolar point at 500 km a nadir plate sees ~349 W/m^2.

    Bounded above by the infinite-plane limit a*S = 408 W/m^2, and reduced from
    it by the finite angular size of the Earth. Independently reproduced by the
    equal-solid-angle disc integral in reflect_orbital/earendil_earthshine.py.
    """
    r = _bare()
    r.compute_earthshine(_vg([0.0, 0.0, 1.0]))
    assert r.earthshine_irradiance == pytest.approx(349.0, abs=2.0)
    assert r.earthshine_irradiance < 0.3 * 1361.0


def test_default_resolution_is_already_converged():
    """n_rings=4 must match a far finer grid, or the default is miscalibrated."""
    coarse, fine = _bare(), _bare()
    sun = np.array([0.0, 0.0, 1.0])
    coarse.compute_earthshine(_vg(sun), n_rings=4)
    fine.compute_earthshine(_vg(sun), n_rings=40)
    assert coarse.earthshine_irradiance == pytest.approx(
        fine.earthshine_irradiance, rel=2e-3)


def test_a_facet_edge_on_to_the_mean_direction_still_collects_light():
    """THE REGRESSION TEST for the lumped-direction defect.

    A facet whose normal is perpendicular to the flux-weighted mean earthshine
    direction still sees a large part of the Earth, so it must collect a
    non-trivial fraction of what a nadir-facing facet collects. The old lumped
    treatment gave it exactly zero.
    """
    r = _bare()
    r.compute_earthshine(_vg([0.0, 0.0, 1.0]))
    mean_dir = np.asarray(r.earthshine_direction, dtype=float)
    # a unit normal perpendicular to the mean direction
    perp = np.cross(mean_dir, np.array([1.0, 0.0, 0.0]))
    perp /= np.linalg.norm(perp)
    assert abs(perp @ mean_dir) < 1e-12

    E = r.earthshine_patch_irradiance
    dirs = r.earthshine_dirs
    on_perp = float(np.sum(E * np.clip(dirs @ perp, 0.0, None)))
    on_nadir = float(np.sum(E * np.clip(dirs @ r.nadir_direction, 0.0, None)))
    assert on_perp > 0.1 * on_nadir, (
        f'edge-on facet collected {on_perp:.3f} vs {on_nadir:.3f} W/m^2 on nadir; '
        'the extended source has collapsed back to a single direction')


def test_dark_earth_yields_no_source_rather_than_a_nan_direction():
    """Sun behind the Earth: zero patches, zero irradiance, direction None."""
    r = _bare()
    r.compute_earthshine(_vg([0.0, 0.0, -1.0]))
    assert r.earthshine_irradiance == 0.0
    assert r.earthshine_direction is None
    assert r.earthshine_patch_irradiance.size == 0


def test_irradiance_is_linear_in_earth_albedo():
    a, b = _bare(earth_albedo=0.15), _bare(earth_albedo=0.30)
    sun = np.array([0.0, 0.0, 1.0])
    a.compute_earthshine(_vg(sun))
    b.compute_earthshine(_vg(sun))
    assert 2.0 * a.earthshine_irradiance == pytest.approx(b.earthshine_irradiance, rel=1e-9)


def test_patches_rotate_with_the_nadir_direction():
    """The Earth must follow `nadir_direction`, so a rolled body frame works."""
    r = _bare(nadir_direction=np.array([0.0, -1.0, 0.0]))
    r.compute_earthshine(_vg([0.0, 1.0, 0.0]))
    # the flux-weighted mean direction must point at the Earth, i.e. along nadir
    assert np.asarray(r.earthshine_direction) @ r.nadir_direction > 0.99


def test_altitude_inside_the_earth_raises():
    with pytest.raises(ValueError):
        _bare()._earth_cap(-10.0, 4)


def test_neither_geometry_nor_explicit_vectors_is_a_clear_error():
    with pytest.raises(TypeError, match='geometry='):
        _bare().compute_earthshine()
