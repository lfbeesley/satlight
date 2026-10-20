"""Gates for `Renderer.render()` itself, against closed-form truth.

The rest of the suite tests the pieces AROUND the raytracer - the earthshine
source terms, the cosine-hemisphere sampler, the ortho frame, the BRDF
functions. Nothing asserted that the number coming out of `render()` was
right. This file does that, and it is the file to run first when a result
looks wrong: if these pass, the forward model is sound and the problem is
upstream in the geometry or downstream in the magnitude conversion.

Two bodies have a disk-integrated Lambertian flux that can be written down:

  FLAT PLATE, area A, albedo rho, normal n, at range d:
      E = (rho/pi) * E_sun * (n.s) * A * (n.o) / d^2

  SPHERE, radius R, albedo rho, phase angle alpha:
      E = (2*rho/3) * E_sun * (R^2/d^2) * Phi(alpha)
      Phi(alpha) = [sin(alpha) + (pi - alpha) cos(alpha)] / pi
  i.e. geometric albedo 2*rho/3, and Phi normalised to Phi(0) = 1.

`render()` returns a per-pixel irradiance map in W/m^2 at the observer, so the
comparable quantity is `np.sum(render())`. Measured agreement, which sets the
tolerances below: 0.08% on the plate across incidence 0-80 deg, and 0.1-0.25%
on a subdiv-4 icosphere across phase 0-120 deg.

OBJ AXIS TRAP. Blender's OBJ importer maps file (x,y,z) -> world (x,-z,y), so a
mesh authored naively arrives rotated and every normal-dependent test silently
measures the wrong angle. `_write_obj` therefore takes WORLD/LVLH coordinates
and applies the inverse, (X,Y,Z) -> (X,Z,-Y), on the way out. The mesh
definitions and the physics formulas then share one frame.
`test_obj_round_trip_preserves_world_coordinates` guards that mapping: if it
ever changes, that one test fails loudly instead of all the others drifting.
"""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.needs_bpy

E_SUN = 1361.0          # the Renderer default, W/m^2
RHO = 0.4               # a mid albedo: far enough from 0 and 1 to catch scaling
D_KM = 500.0


# --------------------------------------------------------------- test meshes
def _write_obj(path, verts, faces, name='body'):
    """Write `verts`, given in WORLD coordinates, through the importer's inverse."""
    with open(path, 'w') as fh:
        fh.write(f'o {name}\n')
        for X, Y, Z in verts:
            fh.write(f'v {X:.10f} {Z:.10f} {-Y:.10f}\n')
        for f in faces:
            fh.write('f ' + ' '.join(str(i + 1) for i in f) + '\n')


def _plate(path, half=1.0):
    """A square plate in the world z=0 plane, normal +Z. Returns its area."""
    v = [(-half, -half, 0.0), (half, -half, 0.0),
         (half, half, 0.0), (-half, half, 0.0)]
    _write_obj(path, v, [(0, 1, 2, 3)], 'plate')
    return (2.0 * half) ** 2


def _icosphere(path, subdiv=4, R=1.0):
    """A unit icosphere. Near-uniform triangles, unlike a UV sphere whose poles
    are over-sampled and whose equator is under-sampled at the same cost."""
    t = (1.0 + 5.0 ** 0.5) / 2.0
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
         (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    v = [np.asarray(x, float) / np.linalg.norm(x) for x in v]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
         (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
         (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
         (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    for _ in range(subdiv):
        mid, nf = {}, []

        def midpoint(a, b):
            k = (min(a, b), max(a, b))
            if k not in mid:
                p = v[a] + v[b]
                v.append(p / np.linalg.norm(p))     # project back onto the sphere
                mid[k] = len(v) - 1
            return mid[k]

        for a, b, c in f:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = nf
    _write_obj(path, [tuple(R * x) for x in v], f, 'sphere')
    return R


def _plate_and_wall(path):
    """A 4x4 plate (normal +Z) and a wall at x=1 spanning z in [0,1].

    The wall's normal is +X, so it is edge-on to a +Z observer and contributes
    essentially nothing itself. Its only job is to block the Sun: with the Sun
    at 45 deg in the x-z plane, a plate point at (x,0,0) reaches the wall plane
    at height 1-x, so the strip 0 <= x <= 1 is shadowed - 4 of the 16 units of
    plate area, giving a shadowed/unshadowed ratio of exactly 0.75.
    """
    v = [(-2, -2, 0), (2, -2, 0), (2, 2, 0), (-2, 2, 0),
         (1, -2, 0), (1, 2, 0), (1, 2, 1), (1, -2, 1)]
    _write_obj(path, v, [(0, 1, 2, 3), (4, 5, 6, 7)], 'plate_wall')


# ------------------------------------------------------------------ renderer
def _render(path, sun, obs, d_km=D_KM, rho=RHO, res=128, **kw):
    """Total irradiance at the observer, W/m^2.

    `update()` is called even though `add_camera()` now places the camera
    correctly on its own, because it is what every per-frame caller does and is
    the path worth exercising. See `test_add_camera_and_update_agree`.
    """
    from satlight.raytracer import Renderer          # pulls in bpy FIRST
    from satlight.geometry import ViewGeometry
    g = ViewGeometry(np.asarray(sun, float), np.asarray(obs, float),
                     distance_km=d_km)
    r = Renderer(path, geometry=g, resolution=(res, res),
                 brdf=('lambertian', {'albedo': rho}), **kw)
    r.initialise_scene()
    r.add_camera()
    r.update(g)
    return float(np.sum(r.render()))


def _plate_truth(area, cos_i, cos_r, rho=RHO, d_km=D_KM):
    return (rho / np.pi) * E_SUN * cos_i * area * cos_r / (d_km * 1e3) ** 2


def _sphere_truth(R, alpha, rho=RHO, d_km=D_KM):
    phi = (np.sin(alpha) + (np.pi - alpha) * np.cos(alpha)) / np.pi
    return (2.0 * rho / 3.0) * E_SUN * R ** 2 / (d_km * 1e3) ** 2 * phi


# ========================================================== the axis mapping
def test_obj_round_trip_preserves_world_coordinates(tmp_path):
    """`_write_obj` must undo the importer's (x,y,z)->(x,-z,y).

    Every other test in this file states its geometry in world coordinates and
    would measure the wrong angle if this mapping changed. Assert it directly,
    so the breakage lands here rather than spreading.
    """
    import bpy
    p = str(tmp_path / 'plate.obj')
    _plate(p, half=1.0)                     # world z=0 plane, normal +Z
    _render(p, [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], res=32)

    meshes = [o for o in bpy.data.objects if o.type == 'MESH']
    assert len(meshes) == 1
    ob = meshes[0]
    got = np.array([tuple(ob.matrix_world @ v.co) for v in ob.data.vertices])
    want = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], float)
    assert np.allclose(np.sort(got, axis=0), np.sort(want, axis=0), atol=1e-6)

    n = np.array(ob.matrix_world.to_3x3() @ ob.data.polygons[0].normal)
    assert abs(abs(n[2]) - 1.0) < 1e-6, f'plate normal should be +/-Z, got {n}'


# ============================================================ the flat plate
# PIXEL QUANTISATION. A sharp-edged square is sampled by whole pixels, so its
# projected area is right only to about +/-1 pixel row out of the `span` rows it
# covers. The auto-frame makes span = res / (sqrt(2) * ortho_margin) ~= res/1.5,
# so at res=256 the envelope is ~0.6% and the error does NOT decay smoothly with
# resolution - it oscillates inside that envelope as the edge lands at different
# sub-pixel phases (measured: +2.1% at res=64, -0.9% at 181, +0.4% at 1024).
# Single-resolution plate tests therefore assert the ENVELOPE; the tight,
# phase-averaged gate is test_flat_plate_is_unbiased_over_pixel_phase below.
PLATE_PIXEL_TOL = 1.5e-2
PHASE_RESOLUTIONS = (192, 256, 341, 384, 512)


@pytest.mark.parametrize('inc_deg', [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 80.0])
def test_flat_plate_matches_the_closed_form(tmp_path, inc_deg):
    """The cleanest possible gate: one facet, one cosine, no occlusion.

    A failure here is a radiometry error - a missing pi, a factor of 2, the
    km/m confusion in `prop_distance` that is 7.5 magnitudes - and it will be
    present in every lightcurve the package has ever produced.
    """
    p = str(tmp_path / 'plate.obj')
    area = _plate(p, half=1.0)
    i = np.radians(inc_deg)
    got = _render(p, [np.sin(i), 0.0, np.cos(i)], [0.0, 0.0, 1.0], res=256)
    want = _plate_truth(area, cos_i=np.cos(i), cos_r=1.0)
    assert got == pytest.approx(want, rel=PLATE_PIXEL_TOL)


def test_flat_plate_is_unbiased_over_pixel_phase(tmp_path):
    """The tight radiometric gate on the plate.

    Averaging over resolutions averages over the sub-pixel phase of the plate
    edge, which is the only thing stopping the single-resolution tests from
    being tight. What survives is the actual radiometry, and it must be
    unbiased: measured 1.00004 over the five resolutions below, and 0.9998 over
    eleven. A systematic error of even half a percent - the kind a wrong cosine
    convention or a stale pixel area produces - would show here while hiding
    comfortably inside the per-resolution envelope.
    """
    p = str(tmp_path / 'plate.obj')
    area = _plate(p, half=1.0)
    i = np.radians(30.0)
    want = _plate_truth(area, cos_i=np.cos(i), cos_r=1.0)
    ratios = [_render(p, [np.sin(i), 0.0, np.cos(i)], [0.0, 0.0, 1.0],
                      res=res) / want for res in PHASE_RESOLUTIONS]
    assert np.mean(ratios) == pytest.approx(1.0, rel=3e-3), f'ratios={ratios}'
    # and the scatter must be the quantisation envelope, not something larger
    assert np.std(ratios) < PLATE_PIXEL_TOL


def test_flat_plate_matches_the_closed_form_for_an_oblique_observer(tmp_path):
    """Both cosines, not just the incident one. The outgoing cosine enters
    through the projected area the camera sees, so it tests the ortho frame and
    the radiometry together."""
    p = str(tmp_path / 'plate.obj')
    area = _plate(p, half=1.0)
    r_ang = np.radians(60.0)
    want = _plate_truth(area, cos_i=1.0, cos_r=np.cos(r_ang))
    ratios = [_render(p, [0.0, 0.0, 1.0],
                      [np.sin(r_ang), 0.0, np.cos(r_ang)], res=res) / want
              for res in PHASE_RESOLUTIONS]
    assert np.mean(ratios) == pytest.approx(1.0, rel=5e-3), f'ratios={ratios}'


def test_a_back_lit_plate_is_exactly_zero(tmp_path):
    """Not 'small': exactly 0.0. A clamped cosine must clamp, and a negative
    one must never leak through as a negative flux."""
    p = str(tmp_path / 'plate.obj')
    _plate(p, half=1.0)
    got = _render(p, [0.0, 0.0, -1.0], [0.0, 0.0, 1.0], res=64)
    assert got == 0.0


# ================================================================ the sphere
@pytest.mark.slow
@pytest.mark.parametrize('alpha_deg', [0.0, 20.0, 40.0, 60.0, 90.0, 120.0])
def test_lambertian_sphere_matches_the_analytic_phase_function(tmp_path, alpha_deg):
    """The strongest single test in the suite.

    A sphere exercises mesh handling, per-facet shading, the terminator, the
    limb and the disk integration at once, against a number that cannot be
    tuned: geometric albedo 2*rho/3 and Phi(alpha) are fixed by the physics.
    Getting the right answer at SIX phase angles with one albedo is very hard
    to do by accident - it gates the shape of the phase curve, not just a
    normalisation, which is exactly what an inversion leans on.
    """
    p = str(tmp_path / 'sphere.obj')
    R = _icosphere(p, subdiv=4)
    a = np.radians(alpha_deg)
    got = _render(p, [np.sin(a), 0.0, np.cos(a)], [0.0, 0.0, 1.0], res=128)
    assert got == pytest.approx(_sphere_truth(R, a), rel=1.5e-2)


@pytest.mark.slow
def test_sphere_converges_towards_the_analytic_value_with_tessellation(tmp_path):
    """Refining the mesh must move TOWARDS truth, monotonically.

    A faceted sphere is always a little dark; what matters is that the error
    shrinks when the discretisation does. If it plateaus or grows, the error is
    not discretisation at all but something systematic wearing its clothes.
    """
    a = np.radians(60.0)
    sun, obs = [np.sin(a), 0.0, np.cos(a)], [0.0, 0.0, 1.0]
    errs = []
    for sub in (2, 3, 4):
        p = str(tmp_path / f'sphere{sub}.obj')
        R = _icosphere(p, subdiv=sub)
        got = _render(p, sun, obs, res=128)
        errs.append(abs(got / _sphere_truth(R, a) - 1.0))
    assert errs[0] > errs[1] > errs[2], f'not converging: {errs}'
    assert errs[-1] < 5e-3


# ================================================== scaling and reproducibility
def test_flux_follows_the_inverse_square_law(tmp_path):
    """`prop_distance` is in km and the renderer works in m. That conversion
    being wrong is a factor of 1000 in flux - 7.5 mag - with a perfectly
    ordinary-looking image to show for it."""
    p = str(tmp_path / 'plate.obj')
    _plate(p, half=1.0)
    sun = obs = [0.0, 0.0, 1.0]
    base = _render(p, sun, obs, d_km=500.0, res=128)
    for factor in (2.0, 4.0, 10.0):
        got = _render(p, sun, obs, d_km=500.0 * factor, res=128)
        assert got == pytest.approx(base / factor ** 2, rel=1e-9)


def test_flux_is_proportional_to_area(tmp_path):
    """Doubling the plate's half-width must quadruple the flux. This is the
    gate on `pixel_area` tracking `ortho_scale`: the auto-frame resizes with
    the body, so a stale pixel area would cancel the area dependence and leave
    every object looking the same brightness.

    Exact, not approximate: the two plates are similar figures, so the
    auto-frame scales with them and both cover the same number of pixels. The
    quantisation that limits the absolute tests above cancels identically
    here."""
    small = str(tmp_path / 'small.obj')
    big = str(tmp_path / 'big.obj')
    a_small = _plate(small, half=1.0)
    a_big = _plate(big, half=2.0)
    sun = obs = [0.0, 0.0, 1.0]
    f_small = _render(small, sun, obs, res=256)
    f_big = _render(big, sun, obs, res=256)
    assert f_big / f_small == pytest.approx(a_big / a_small, rel=2e-3)


def test_flux_is_proportional_to_albedo(tmp_path):
    """Lambertian flux is strictly linear in albedo - no saturation, no
    implicit multiple scattering at n_bounce=1."""
    p = str(tmp_path / 'plate.obj')
    _plate(p, half=1.0)
    sun = obs = [0.0, 0.0, 1.0]
    f1 = _render(p, sun, obs, rho=0.1, res=128)
    f2 = _render(p, sun, obs, rho=0.8, res=128)
    assert f2 / f1 == pytest.approx(8.0, rel=1e-9)


@pytest.mark.slow
def test_result_is_converged_in_resolution(tmp_path):
    """Doubling the resolution must not move the answer meaningfully. 0.5% is
    about 0.005 mag, well inside any photometry this is compared against."""
    p = str(tmp_path / 'sphere.obj')
    _icosphere(p, subdiv=4)
    a = np.radians(60.0)
    sun, obs = [np.sin(a), 0.0, np.cos(a)], [0.0, 0.0, 1.0]
    assert _render(p, sun, obs, res=128) == pytest.approx(
        _render(p, sun, obs, res=256), rel=5e-3)


def test_render_is_bit_for_bit_reproducible(tmp_path):
    """Same geometry in, same pixels out - byte-identical, not merely close.
    Nothing in the single-bounce path is stochastic, so anything less than
    exact equality means uninitialised state is leaking between renders."""
    p = str(tmp_path / 'plate.obj')
    _plate(p, half=1.0)
    a = np.radians(35.0)
    sun, obs = [np.sin(a), 0.0, np.cos(a)], [0.0, 0.0, 1.0]
    assert _render(p, sun, obs, res=128) == _render(p, sun, obs, res=128)


def test_image_is_never_negative(tmp_path):
    """A negative pixel is a sign error that partially cancels in the sum and
    so can hide from every total-flux test above."""
    from satlight.raytracer import Renderer
    from satlight.geometry import ViewGeometry
    p = str(tmp_path / 'sphere.obj')
    _icosphere(p, subdiv=3)
    a = np.radians(100.0)
    g = ViewGeometry(np.array([np.sin(a), 0.0, np.cos(a)]),
                     np.array([0.0, 0.0, 1.0]), distance_km=D_KM)
    r = Renderer(p, geometry=g, resolution=(96, 96),
                 brdf=('lambertian', {'albedo': RHO}))
    r.initialise_scene()
    r.add_camera()
    r.update(g)
    img = r.render()
    assert np.isfinite(img).all()
    assert (img >= 0.0).all()


# ========================================================= self-shadowing
def test_self_shadowing_removes_the_predicted_area(tmp_path):
    """Shadowing must remove the geometrically correct AMOUNT, not just 'some'.

    The wall is edge-on to the observer and contributes nothing of its own, so
    the on/off ratio isolates the shadowed fraction: 12 of 16 units, 0.75.
    """
    p = str(tmp_path / 'plate_wall.obj')
    _plate_and_wall(p)
    i = np.radians(45.0)
    sun, obs = [np.sin(i), 0.0, np.cos(i)], [0.0, 0.0, 1.0]
    on = _render(p, sun, obs, res=400, add_shadows=True)
    off = _render(p, sun, obs, res=400, add_shadows=False)
    assert on / off == pytest.approx(0.75, abs=5e-3)


def test_shadows_can_only_remove_light(tmp_path):
    """Whatever the geometry, enabling shadows must never brighten anything."""
    p = str(tmp_path / 'plate_wall.obj')
    _plate_and_wall(p)
    for inc in (20.0, 45.0, 70.0):
        i = np.radians(inc)
        sun, obs = [np.sin(i), 0.0, np.cos(i)], [0.0, 0.0, 1.0]
        on = _render(p, sun, obs, res=200, add_shadows=True)
        off = _render(p, sun, obs, res=200, add_shadows=False)
        assert on <= off * (1.0 + 1e-9), f'shadows added light at inc={inc}'


def test_an_unoccluded_body_is_indifferent_to_the_shadow_flag(tmp_path):
    """A single convex plate can shadow nothing, so the flag must be a no-op.
    This is the control for the test above: it separates 'shadowing works' from
    'the shadow ray is attenuating things it never hit'."""
    p = str(tmp_path / 'plate.obj')
    _plate(p, half=1.0)
    i = np.radians(45.0)
    sun, obs = [np.sin(i), 0.0, np.cos(i)], [0.0, 0.0, 1.0]
    assert _render(p, sun, obs, res=128, add_shadows=True) == pytest.approx(
        _render(p, sun, obs, res=128, add_shadows=False), rel=1e-9)


# ========================================================== smooth normals
def test_smooth_normals_are_a_no_op_on_a_flat_plate(tmp_path):
    """A plate has no curvature, so interpolating its normals must change
    nothing - exactly. This is the gate on normal repair not perturbing flat
    geometry, which is where it would do silent damage to panels and the solar
    arrays that dominate most spacecraft's flux."""
    p = str(tmp_path / 'plate.obj')
    _plate(p, half=1.0)
    i = np.radians(40.0)
    sun, obs = [np.sin(i), 0.0, np.cos(i)], [0.0, 0.0, 1.0]
    flat = _render(p, sun, obs, res=256)
    smooth = _render(p, sun, obs, res=256,
                     smooth_normals=True, smooth_angle_deg=180.0)
    assert smooth == pytest.approx(flat, rel=1e-12)


@pytest.mark.slow
def test_smooth_and_flat_normals_converge_to_the_same_sphere(tmp_path):
    """Shading normals change the DISTRIBUTION of light over the disk, not the
    total, and the two must agree in the well-tessellated limit.

    Note the direction of the effect, measured rather than assumed: on a coarse
    icosphere smoothing makes the total slightly DARKER, not brighter, and the
    gap closes as the mesh refines (0.09% at subdiv 3, 0.04% at subdiv 4). So
    this asserts convergence, not an improvement in accuracy.
    """
    a = np.radians(60.0)
    sun, obs = [np.sin(a), 0.0, np.cos(a)], [0.0, 0.0, 1.0]
    gaps = []
    for sub in (3, 4):
        p = str(tmp_path / f'sphere{sub}.obj')
        _icosphere(p, subdiv=sub)
        flat = _render(p, sun, obs, res=128)
        smooth = _render(p, sun, obs, res=128,
                         smooth_normals=True, smooth_angle_deg=180.0)
        gaps.append(abs(smooth / flat - 1.0))
    assert gaps[1] < gaps[0], f'not converging: {gaps}'
    assert gaps[1] < 5e-3


# ==================================================== the camera/update contract
def test_add_camera_honours_the_geometry_without_update(tmp_path):
    """`initialise_scene -> add_camera -> render` must use the geometry it was given.

    It used to render from NADIR regardless, because camera placement was
    implemented twice and the `add_camera` copy hardcoded the -Z axis. With an
    oblique observer that meant looking into the BACK of the model, so the
    backface clamp returned exactly 0.0 - a silent zero in the most obvious
    first-use path. Both now go through `_place_camera`.
    """
    from satlight.raytracer import Renderer
    from satlight.geometry import ViewGeometry
    p = str(tmp_path / 'plate.obj')
    area = _plate(p, half=1.0)
    r_ang = np.radians(60.0)
    g = ViewGeometry(np.array([0.0, 0.0, 1.0]),
                     np.array([np.sin(r_ang), 0.0, np.cos(r_ang)]),
                     distance_km=D_KM)
    r = Renderer(p, geometry=g, resolution=(256, 256),
                 brdf=('lambertian', {'albedo': RHO}))
    r.initialise_scene()
    r.add_camera()                       # deliberately no update()
    got = float(np.sum(r.render()))
    want = _plate_truth(area, cos_i=1.0, cos_r=np.cos(r_ang))
    assert got > 0.0, 'rendered zero: the camera is not looking at the lit face'
    assert got == pytest.approx(want, rel=PLATE_PIXEL_TOL)


def test_add_camera_and_update_agree(tmp_path):
    """The two entry points into `_place_camera` must give the SAME frame.

    This is the regression gate on the duplication coming back: any future
    divergence between them shows up here as a flux difference.
    """
    from satlight.raytracer import Renderer
    from satlight.geometry import ViewGeometry
    p = str(tmp_path / 'sphere.obj')
    _icosphere(p, subdiv=3)
    a = np.radians(50.0)
    g = ViewGeometry(np.array([np.sin(a), 0.0, np.cos(a)]),
                     np.array([0.0, 0.0, 1.0]), distance_km=D_KM)

    def flux(call_update):
        r = Renderer(p, geometry=g, resolution=(128, 128),
                     brdf=('lambertian', {'albedo': RHO}))
        r.initialise_scene()
        r.add_camera()
        if call_update:
            r.update(g)
        return float(np.sum(r.render())), r.ortho_scale, r.pixel_area

    f_no, scale_no, px_no = flux(False)
    f_yes, scale_yes, px_yes = flux(True)
    assert scale_no == pytest.approx(scale_yes, rel=1e-12)
    assert px_no == pytest.approx(px_yes, rel=1e-12)
    assert f_no == pytest.approx(f_yes, rel=1e-12)


def test_the_removed_geometry_arguments_raise_a_pointed_error(tmp_path):
    """The old three-argument form must not be silently ignored.

    `distance_to_observer` was in metres and `prop_distance` is in kilometres,
    so a caller who got away with passing the old name would be wrong by 1000x
    in flux - 7.5 magnitudes - with a normal-looking image. Both the
    constructor and `update` must refuse, and say what to pass instead.
    """
    from satlight.raytracer import Renderer
    from satlight.geometry import ViewGeometry
    p = str(tmp_path / 'plate.obj')
    _plate(p, half=1.0)
    g = ViewGeometry(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]),
                     distance_km=D_KM)

    with pytest.raises(TypeError, match='KILOMETRES'):
        Renderer(p, geometry=g, distance_to_observer=5.0e5)
    with pytest.raises(TypeError, match='geometry='):
        Renderer(p)
    with pytest.raises(TypeError, match='unexpected keyword'):
        Renderer(p, geometry=g, no_such_option=1)

    r = Renderer(p, geometry=g, resolution=(32, 32))
    r.initialise_scene()
    r.add_camera()
    with pytest.raises(TypeError, match='KILOMETRES'):
        r.update(g, sun_direction=[0, 0, 1])
    with pytest.raises(TypeError, match='geometry='):
        r.update()
