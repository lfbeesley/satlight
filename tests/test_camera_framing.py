"""Gates for the orthographic frame and the camera standoff.

Two quantities that used to be the same number and should never have been:

  ortho_scale  - the WIDTH of the orthographic frame. Must contain the
                 geometry's projection in the current view, or the body is
                 clipped.
  standoff     - how far back the camera sits. For an orthographic projection
                 this is physically irrelevant, but `render` starts its primary
                 rays ON the camera plane, so the plane must clear the body
                 along the view axis or everything behind it is silently lost.

The old code used `location = view_target + obs * ortho_scale`, i.e. a frame
WIDTH as a DISTANCE. That is safe only by accident, because the default
ortho_scale is the bounding-box diagonal and so exceeds the body's extent in
every direction. Tighten the frame and the camera plane walks into the
geometry: a 5.4 m frame on Skynet 5D, whose arrays reach 17.3 m along the view
axis, put the plane inside a solar array.
"""
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKYNET = os.path.join(ROOT, 'models/Skynet 5D/skynet.obj')
ARRAY_HALF_SPAN = 17.3      # solarPanel1/2 reach +/-17.3 m along world Y

pytestmark = pytest.mark.needs_bpy


def _renderer(res=256, ortho=None, auto=True, obs=(0.0, -1.0, 0.0)):
    from satlight.raytracer import Renderer   # pulls in bpy FIRST
    from satlight.geometry import ViewGeometry
    import mathutils as mu                    # must follow bpy, or Vector differs
    o = np.array(obs, dtype=float)
    th = np.radians(60.0)
    s = np.cos(th) * o + np.sin(th) * np.array([0.5, 0.0, 0.866])
    s /= np.linalg.norm(s)
    g = ViewGeometry(s, o, distance_km=3.75e7)
    r = Renderer(SKYNET, geometry=g, resolution=(res, res),
                 brdf={'default': ('lambertian', {'albedo': 0.3})},
                 add_shadows=True, n_bounce=1, auto_ortho_scale=auto)
    r.initialise_scene()
    if ortho:
        r.ortho_scale = ortho
    r.add_camera()
    r.update(g)
    return r


@pytest.mark.parametrize('ortho', [None, 5.41, 100.0])
def test_standoff_clears_the_body_whatever_the_frame_width(ortho):
    """The camera plane must sit outside the geometry for ANY frame width."""
    r = _renderer(ortho=ortho, auto=(ortho is None))
    assert abs(r.camera_object.location.y) > ARRAY_HALF_SPAN, (
        'camera plane is inside the solar array: primary rays start there, so '
        'everything behind it is silently dropped')


def test_standoff_is_independent_of_frame_width():
    """Standoff is a function of the along-view extent, never of ortho_scale.

    Only one Renderer can be alive at a time - initialise_scene wipes the scene
    and invalidates the previous one's objects - so read each value out first.
    """
    tight_y = _renderer(ortho=5.41, auto=False).camera_object.location.y
    wide_y = _renderer(ortho=100.0, auto=False).camera_object.location.y
    assert tight_y == pytest.approx(wide_y, rel=1e-9)


def test_repeated_updates_do_not_drift_the_camera():
    """Extents are measured about the view target, so update() is idempotent."""
    from satlight.geometry import ViewGeometry
    r = _renderer()
    first = r.camera_object.location.y
    o = np.array([0.0, -1.0, 0.0])
    g = ViewGeometry(np.array(r.sun_direction), o, distance_km=3.75e7)
    for _ in range(5):
        r.update(g)
    assert r.camera_object.location.y == pytest.approx(first, rel=1e-9)


def test_auto_frame_contains_the_body_without_touching_the_edge():
    r = _renderer()
    img = np.array(r.render())
    assert img[0, :].sum() == 0 and img[-1, :].sum() == 0
    assert img[:, 0].sum() == 0 and img[:, -1].sum() == 0


def test_auto_frame_samples_an_end_on_view_far_better():
    """The whole point: a frame fitted to the view uses the pixels it has.

    hit_count only exists once render() has run, and only one Renderer can be
    alive at a time, so render and capture before building the next.
    """
    r = _renderer(auto=True); r.render()
    auto_scale, auto_hits = r.ortho_scale, r.hit_count
    r = _renderer(auto=False); r.render()
    fixed_scale, fixed_hits = r.ortho_scale, r.hit_count
    assert fixed_scale > 5 * auto_scale          # diagonal vs projected extent
    assert auto_hits > 10 * fixed_hits


def test_auto_and_fixed_frames_agree_once_converged():
    """Fitting the frame must not change the answer, only how fast you get it."""
    a = np.sum(_renderer(res=1024, auto=True).render())
    f = np.sum(_renderer(res=1024, auto=False).render())
    ma = -2.5 * np.log10(float(a)) - 18.915
    mf = -2.5 * np.log10(float(f)) - 18.915
    assert ma == pytest.approx(mf, abs=0.05)
