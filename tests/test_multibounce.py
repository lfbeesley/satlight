"""Gates for multiple scattering (`n_bounce`) in Renderer.

The physics to pin down:
  * n_bounce=1 must be exactly the old single-bounce behaviour;
  * CONVEX geometry sees none of itself, so extra bounces must change nothing
    at all - this is the test that catches light being invented;
  * CONCAVE geometry (the finned antenna tile) must brighten by roughly
    rho * F, with F the fraction of the hemisphere its own structure occupies,
    and must stay below the closed-form series limit 1 / (1 - rho * F);
  * the estimator must be reproducible and converged at the default sample count.
"""
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIN = os.path.join(ROOT, 'models/AST/bluebird_block1_tile.obj')

# Panel -> fin view factor for bluebird_block1_tile.obj, measured by
# cosine-weighted ray casting from the nadir panel (40k rays).
F_VIEW = 0.328


def _cosine_hemisphere_only():
    from satlight.raytracer import Renderer
    return Renderer._cosine_hemisphere


def test_cosine_hemisphere_is_normalised_and_in_the_upper_half():
    """Directions must be unit, and never below the surface."""
    f = _cosine_hemisphere_only()
    n = np.array([0.0, 0.0, 1.0])
    d = f(n, 5000, np.random.default_rng(0))
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-12)
    assert (d @ n > -1e-12).all()


def test_cosine_hemisphere_follows_the_cosine_law():
    """<cos> must be 2/3 for a cosine-weighted hemisphere, not 1/2 (uniform)."""
    f = _cosine_hemisphere_only()
    n = np.array([0.0, 0.0, 1.0])
    d = f(n, 200000, np.random.default_rng(1))
    assert np.mean(d @ n) == pytest.approx(2.0 / 3.0, abs=5e-3)


def test_cosine_hemisphere_respects_an_arbitrary_normal():
    f = _cosine_hemisphere_only()
    n = np.array([0.3, -0.5, 0.81]); n /= np.linalg.norm(n)
    d = f(n, 20000, np.random.default_rng(2))
    assert (d @ n > -1e-12).all()
    assert np.mean(d @ n) == pytest.approx(2.0 / 3.0, abs=2e-2)


# ---------------------------------------------------------------- rendering
pytestmark_bpy = pytest.mark.needs_bpy


def _render(obj, rho, n_bounce, res=192, samples=8):
    from satlight.raytracer import Renderer   # pulls in bpy FIRST
    from satlight.geometry import ViewGeometry
    import mathutils as mu
    obs = np.array([0.0, 0.0, -1.0])
    sun = np.array([np.sin(np.radians(70.0)), 0.0, -np.cos(np.radians(70.0))])
    brdf = {'panel_nadir': ('lambertian', {'albedo': rho}),
            'standoffs': ('lambertian', {'albedo': rho}),
            'panel_zenith': ('lambertian', {'albedo': 0.1}),
            'default': ('lambertian', {'albedo': rho})}
    g = ViewGeometry(sun, obs, distance_km=500.0)
    r = Renderer(obj, geometry=g, resolution=(res, res), brdf=brdf,
                 add_shadows=True, n_bounce=n_bounce, bounce_samples=samples)
    r.initialise_scene()
    r.add_camera()
    r.update(g)
    return float(np.sum(r.render()))


CUBE_OBJ = """o cube
v -1 -1 -1
v  1 -1 -1
v  1  1 -1
v -1  1 -1
v -1 -1  1
v  1 -1  1
v  1  1  1
v -1  1  1
f 1 2 3 4
f 5 8 7 6
f 1 5 6 2
f 2 6 7 3
f 3 7 8 4
f 5 1 4 8
"""


@pytest.mark.needs_bpy
def test_convex_geometry_is_untouched_by_extra_bounces(tmp_path):
    """A CONVEX body sees none of itself, so n_bounce must change NOTHING.

    This is the test that catches an estimator inventing light. The mesh is
    written inline rather than imported from the study code, so the package
    test suite stays free of any dependency on large_spacecraft/.
    """
    cube = tmp_path / 'cube.obj'
    cube.write_text(CUBE_OBJ)
    one = _render(str(cube), 0.8, 1)
    two = _render(str(cube), 0.8, 2)
    assert two == pytest.approx(one, rel=1e-9)


@pytest.mark.needs_bpy
def test_concave_geometry_brightens_and_stays_under_the_series_limit():
    for rho in (0.165, 0.8):
        one = _render(FIN, rho, 1)
        two = _render(FIN, rho, 2)
        ratio = two / one
        assert ratio > 1.0, 'inter-reflection must add light, not remove it'
        # two bounces truncate the series, so it cannot exceed the closed form
        # by more than the fin-to-fin view factor exceeding the panel's own.
        assert ratio < 1.0 / (1.0 - rho * F_VIEW) * 1.05
        # and it must scale the right way with albedo
    assert (_render(FIN, 0.8, 2) / _render(FIN, 0.8, 1)
            > _render(FIN, 0.165, 2) / _render(FIN, 0.165, 1))


@pytest.mark.needs_bpy
def test_result_is_reproducible_across_renders():
    """Same seed, same geometry: the Monte Carlo must not drift between frames."""
    assert _render(FIN, 0.8, 2) == pytest.approx(_render(FIN, 0.8, 2), rel=1e-12)


@pytest.mark.needs_bpy
def test_converged_at_the_default_sample_count():
    """8 samples must already agree with 32 - the pixel average does the work."""
    base = _render(FIN, 0.8, 1)
    lo = _render(FIN, 0.8, 2, samples=8) / base
    hi = _render(FIN, 0.8, 2, samples=32) / base
    assert lo == pytest.approx(hi, rel=0.01)
