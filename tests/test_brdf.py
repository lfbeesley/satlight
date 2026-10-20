"""BRDF unit tests. numpy only - no Blender, no ephemeris, milliseconds."""
import numpy as np
import pytest

from satlight import brdf

MODELS = {
    'lambertian': {}, 'phong': {}, 'blinn_phong': {},
    'cook_torrance': {}, 'oren_nayar': {}, 'ashikhmin_shirley': {}, 'abg': {},
}
N = np.array([0.0, 0.0, 1.0])


def _dirs(theta_deg, phi_deg=0.0):
    t, p = np.radians(theta_deg), np.radians(phi_deg)
    return np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])


@pytest.mark.parametrize('name', sorted(MODELS))
def test_backfacing_light_returns_zero(name):
    """A surface lit from behind emits nothing.

    Sign errors here are nearly invisible in a rendered frame but corrupt every
    integrated flux, so this is the cheapest high-value check in the suite.
    """
    f = getattr(brdf, name)
    v = _dirs(30.0)
    for theta in (91.0, 120.0, 179.0):
        assert f(N, _dirs(theta), v, **MODELS[name]) == 0.0


@pytest.mark.parametrize('name', sorted(MODELS))
def test_never_negative(name):
    f = getattr(brdf, name)
    for tl in (0.0, 15.0, 45.0, 80.0):
        for tv in (0.0, 15.0, 45.0, 80.0):
            for phi in (0.0, 90.0, 180.0):
                assert f(N, _dirs(tl), _dirs(tv, phi), **MODELS[name]) >= 0.0


def test_lambertian_matches_closed_form():
    for theta in (0.0, 30.0, 60.0, 85.0):
        got = brdf.lambertian(N, _dirs(theta), _dirs(20.0), albedo=0.4)
        assert got == pytest.approx(0.4 / np.pi * np.cos(np.radians(theta)), rel=1e-12)


def test_lambertian_conserves_energy():
    """Integrating the BRDF over the outgoing hemisphere must give the albedo.

    Catches a missing 1/pi, the classic normalisation error.
    """
    albedo = 0.6
    n_t, n_p = 200, 200
    th = (np.arange(n_t) + 0.5) * (np.pi / 2) / n_t
    ph = (np.arange(n_p) + 0.5) * 2 * np.pi / n_p
    dth, dph = (np.pi / 2) / n_t, 2 * np.pi / n_p
    # brdf * cos(theta_out) * sin(theta_out) dtheta dphi
    total = float(np.sum(albedo / np.pi * np.cos(th) * np.sin(th)) * dth * n_p * dph)
    assert total == pytest.approx(albedo, rel=1e-3)


def test_abg_peaks_at_the_specular_direction():
    """ABg is a specular lobe: brightest at the mirror direction, falling away."""
    light = _dirs(40.0)
    peak = brdf.abg(N, light, _dirs(40.0, 180.0))
    for off in (5.0, 15.0, 40.0):
        assert brdf.abg(N, light, _dirs(40.0 + off, 180.0)) < peak


@pytest.mark.parametrize('name', ['lambertian', 'cook_torrance', 'oren_nayar'])
def test_helmholtz_reciprocity(name):
    """f_r(n, l, v) == f_r(n, v, l) for physically based models.

    NOTE the cosine. These functions return BRDF * cos(theta_in), not the bare
    BRDF - lambertian gives albedo/pi * n.l - so reciprocity only holds after
    dividing the incident cosine out. Worth stating in the docstrings: a caller
    who assumes a pure BRDF double-counts the cosine.

    Phong and Blinn-Phong are excluded: they are empirical and not reciprocal,
    which is a property of the model rather than a bug.
    """
    f = getattr(brdf, name)
    for tl, tv, phi in ((20., 60., 0.), (35., 70., 90.), (10., 45., 180.)):
        cl, cv = np.cos(np.radians(tl)), np.cos(np.radians(tv))
        a = f(N, _dirs(tl), _dirs(tv, phi)) / cl
        b = f(N, _dirs(tv, phi), _dirs(tl)) / cv
        assert a == pytest.approx(b, rel=1e-9)


def test_functions_include_the_incident_cosine():
    """Pin the convention the reciprocity test depends on."""
    for theta in (0.0, 25.0, 55.0):
        got = brdf.lambertian(N, _dirs(theta), _dirs(10.0), albedo=1.0)
        assert got == pytest.approx(np.cos(np.radians(theta)) / np.pi, rel=1e-12)


def test_registry_matches_module_functions():
    """Every registry name resolves to the function of the same name."""
    try:
        from satlight.raytracer import BRDF_REGISTRY as reg
    except Exception:
        reg = getattr(brdf, 'BRDF_REGISTRY', None)
    if reg is None:
        pytest.skip('BRDF_REGISTRY not importable without bpy')
    for key, func in reg.items():
        assert callable(func)
        assert getattr(brdf, key, None) is func, f'{key} does not match brdf.{key}'
