"""Earthshine from a discretised Earth of uniform albedo.

The Earth is split into patches of equal solid angle as seen FROM the
satellite, and each sunlit patch contributes (albedo*S/pi)*cos(incidence)*dOmega.
Everything here is pinned against an INDEPENDENT closed-form quadrature, not
against a previous run of the code, because the earlier equal-area grid fixed
to the Earth agreed with itself at every resolution while being 68% low at
550 km: it weights each patch by area*cos/r^2, which concentrates the integral
into the few patches directly beneath the satellite.

These need bpy only because satlight.raytracer imports it at module scope; the
earthshine maths itself is pure numpy and runs off a stub geometry.
"""
import numpy as np
import pytest
from types import SimpleNamespace

pytestmark = pytest.mark.needs_bpy

R_EARTH = 6371.0
S = 1361.0
ALBEDO = 0.3


@pytest.fixture
def renderer(bpy_available):
    """A Renderer with the Blender half bypassed — earthshine touches none of it."""
    from satlight.raytracer import Renderer
    r = Renderer.__new__(Renderer)
    r.solar_constant = S
    r.earth_albedo = ALBEDO
    r._patch_template = None
    r.earthshine_irradiance = 0.0
    r.earthshine_direction = None
    r._earthshine_ready = False
    return r


def stub_geometry(alt_km, sun_lvlh=(0.0, 0.0, 1.0)):
    """compute_earthshine reads only the geocentric radius and the Sun in LVLH."""
    return SimpleNamespace(
        positions={'satellite': np.array([0.0, 0.0, R_EARTH + alt_km])},
        incident_vector_lvlh=np.asarray(sun_lvlh, float),
    )


def exact_zero_phase(alt_km, n=400_001):
    """Irradiance on a NADIR-FACING PLATE directly above the subsolar point.

        dE = L cos(emission) dA cos(theta_plate) / r^2

    with, for a ring at central angle gamma (u = cos gamma) on a Lambertian
    Earth of albedo a under irradiance S:

        L               = a S u / pi          u is the solar incidence cosine
        cos(emission)   = (d u - R) / r       at the surface patch
        cos(theta_plate)= (d - R u) / r       at the satellite, from nadir
        dA              = 2 pi R^2 du

        => E = 2 R^2 a S \\int_{R/d}^{1} u (d u - R)(d - R u) / r^4 du,
           r^2 = d^2 - 2 d R u + R^2

    THE cos(theta_plate) FACTOR IS NOT OPTIONAL. An earlier version of this
    helper omitted it, keeping only the emission cosine, and so returned values
    that exceed a*S = %.1f W/m^2 - the radiosity of the surface itself, and a
    hard upper bound on what any plate can receive (it is what you get facing an
    INFINITE Lambertian plane). It read 493 W/m^2 at 550 km and 673 at 100 km,
    both impossible. With the factor restored it agrees with
    Renderer.compute_earthshine to better than 0.01 W/m^2 at every altitude.
    """ % (ALBEDO * S)
    d = R_EARTH + alt_km
    u = np.linspace(R_EARTH / d, 1.0, n)
    r = np.sqrt(d * d - 2 * d * R_EARTH * u + R_EARTH * R_EARTH)
    return 2 * R_EARTH ** 2 * ALBEDO * S * np.trapz(
        u * (d * u - R_EARTH) * (d - R_EARTH * u) / r ** 4, u)


def test_quadrature_reference_reproduces_subtended_solid_angle():
    """Guard on the reference itself, so a wrong reference can't pin wrong code."""
    for alt in (550.0, 35786.0):
        d = R_EARTH + alt
        u = np.linspace(R_EARTH / d, 1.0, 400_001)
        r = np.sqrt(d * d - 2 * d * R_EARTH * u + R_EARTH * R_EARTH)
        omega = 2 * np.pi * R_EARTH ** 2 * np.trapz((d * u - R_EARTH) / r ** 3, u)
        assert omega == pytest.approx(2 * np.pi * (1 - np.sqrt(1 - (R_EARTH / d) ** 2)),
                                      rel=1e-4)


@pytest.mark.parametrize('alt_km', [100.0, 400.0, 550.0, 1200.0, 2000.0, 35786.0])
def test_matches_exact_quadrature_at_zero_phase(renderer, alt_km):
    renderer.compute_earthshine(stub_geometry(alt_km), n_rings=40, verbose=False)
    assert renderer.earthshine_irradiance == pytest.approx(exact_zero_phase(alt_km),
                                                           rel=2e-3)


def test_points_at_nadir_at_zero_phase(renderer):
    renderer.compute_earthshine(stub_geometry(550.0), n_rings=40, verbose=False)
    assert renderer.earthshine_direction == pytest.approx([0.0, 0.0, -1.0], abs=1e-6)


def test_leo_is_order_a_quarter_of_the_solar_constant(renderer):
    """A LEO satellite over a sunlit Earth gets ~a quarter of full sunlight back.

    The scale, not just the convergence: the grid that only agreed with itself
    returned 156 W/m^2 here.

    The bound below a*S = 408 W/m^2 is physics, not tolerance. That is the
    radiosity of the Earth's surface and the most any plate can collect, reached
    only facing an infinite Lambertian plane; a sphere subtending less than a
    full hemisphere must give less. The earlier 450-520 window asserted a value
    ABOVE it, and passed only against a reference integral that had dropped the
    plate's own incidence cosine (see exact_zero_phase).
    """
    renderer.compute_earthshine(stub_geometry(550.0), n_rings=40, verbose=False)
    assert 330.0 < renderer.earthshine_irradiance < ALBEDO * S
    assert renderer.earthshine_irradiance / S == pytest.approx(0.25, abs=0.03)


def test_irradiance_is_linear_in_albedo(renderer):
    g = stub_geometry(550.0)
    renderer.compute_earthshine(g, n_rings=40, albedo=0.3, verbose=False)
    e3 = renderer.earthshine_irradiance
    renderer.compute_earthshine(g, n_rings=40, albedo=0.6, verbose=False)
    assert renderer.earthshine_irradiance == pytest.approx(2 * e3, rel=1e-12)


def test_zero_albedo_yields_no_source_rather_than_a_nan_direction(renderer):
    """render() multiplies the BRDF by this direction; a NaN would blank the frame."""
    renderer.compute_earthshine(stub_geometry(550.0), n_rings=40, albedo=0.0,
                                verbose=False)
    assert renderer.earthshine_irradiance == 0.0
    assert renderer.earthshine_direction is None


def test_dark_earth_yields_zero(renderer):
    """Sun well below the local horizon: no visible patch is sunlit."""
    renderer.compute_earthshine(stub_geometry(550.0, sun_lvlh=(0.0, 0.0, -1.0)),
                                n_rings=40, verbose=False)
    assert renderer.earthshine_irradiance == 0.0
    assert renderer.earthshine_direction is None


def test_terminator_geometry_is_between_the_two_extremes(renderer):
    """Sun on the local horizon: a sliver of the visible Earth is still lit."""
    renderer.compute_earthshine(stub_geometry(550.0, sun_lvlh=(1.0, 0.0, 0.0)),
                                n_rings=40, verbose=False)
    assert 0.0 < renderer.earthshine_irradiance < 100.0
    assert renderer.earthshine_direction[0] > 0.0      # leans toward the lit limb
    assert renderer.earthshine_direction[2] < 0.0      # still mostly downward


def test_converged_within_a_percent_by_default_resolution(renderer):
    g = stub_geometry(550.0)
    renderer.compute_earthshine(g, n_rings=40, verbose=False)
    coarse = renderer.earthshine_irradiance
    renderer.compute_earthshine(g, n_rings=200, verbose=False)
    assert coarse == pytest.approx(renderer.earthshine_irradiance, rel=1e-2)


def test_radius_inside_the_earth_raises(renderer):
    """Catches km/m and radius/altitude mix-ups instead of returning a number."""
    with pytest.raises(ValueError, match='inside the Earth'):
        renderer.compute_earthshine(stub_geometry(-100.0), verbose=False)


class TestGeometryDrivenConstruction:
    """Renderer(geometry=geom) must derive exactly what the old call passed by hand.

    The three explicit arguments were a standing trap: prop_distance is in km
    and distance_to_observer in m, and the conversion sat in every caller.
    These pin the derivation and the old form side by side, since ~30 scripts
    in examples/ still use the explicit form.
    """

    def test_derives_the_frame_from_a_geometry(self, bpy_available):
        from satlight.raytracer import Renderer
        g = SimpleNamespace(
            incident_vector_lvlh=np.array([0.0, 0.0, 1.0]),
            outgoing_vector_lvlh=np.array([0.0, 1.0, 0.0]),
            prop_distance=1234.5,                      # km
        )
        sun, obs, dist = Renderer._frame_from_geometry(g)
        assert list(sun) == pytest.approx([0.0, 0.0, 1.0])
        assert list(obs) == pytest.approx([0.0, 1.0, 0.0])
        assert dist == pytest.approx(1234.5e3)         # km -> m

    def test_array_time_geometry_is_rejected_with_its_epoch_count(self, bpy_available):
        from satlight.raytracer import Renderer
        g = SimpleNamespace(
            incident_vector_lvlh=np.zeros((3, 4)),
            outgoing_vector_lvlh=np.zeros((3, 4)),
            prop_distance=np.ones(4),
        )
        with pytest.raises(ValueError, match='4 epochs'):
            Renderer._frame_from_geometry(g)

    def test_neither_geometry_nor_vectors_is_a_clear_error(self, bpy_available, tmp_path):
        from satlight.raytracer import Renderer
        with pytest.raises(TypeError, match='geometry='):
            Renderer(str(tmp_path / 'nope.obj'))

    def test_partial_vectors_is_a_clear_error(self, bpy_available, tmp_path):
        from satlight.raytracer import Renderer
        with pytest.raises(TypeError, match='geometry='):
            Renderer(str(tmp_path / 'nope.obj'), sun_direction=np.array([0., 0., 1.]))

    def test_update_with_nothing_is_a_clear_error(self, renderer):
        with pytest.raises(TypeError, match='geometry='):
            renderer.update()

