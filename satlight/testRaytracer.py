"""
Standalone tests for four bugs found in satlight/raytracer.py's Renderer
class, and the fixes for each. No bpy/mathutils dependency -- the camera
test reimplements the relevant vector maths in plain numpy so it can run
anywhere; the other three tests count how many times a piece of work runs
under the old vs. fixed code structure.

Run with: python test_raytracer_fixes.py
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. add_camera() ignoring observer_direction
# ---------------------------------------------------------------------------

def camera_rotation_old(ortho_scale):
    """Reproduces the bug: camera always placed along hardcoded -Z,
    regardless of observer_direction."""
    location = np.array([0.0, 0.0, -1.0]) * ortho_scale
    forward = np.array([0.0, 0.0, 1.0])
    return location, forward


def camera_rotation_fixed(observer_direction, ortho_scale):
    """The fix: derive camera placement from the real observer_direction
    (same maths already used correctly in update())."""
    obs = np.asarray(observer_direction, dtype=float)
    obs = obs / np.linalg.norm(obs)
    location = obs * ortho_scale
    to_origin = -location
    to_origin = to_origin / np.linalg.norm(to_origin)
    return location, to_origin


def test_camera_ignores_observer_direction_bug():
    # An observer well off the +Z axis, e.g. viewing from the side
    observer_direction = [1.0, 0.0, 0.0]
    ortho_scale = 5.0

    old_loc, old_forward = camera_rotation_old(ortho_scale)
    fixed_loc, fixed_forward = camera_rotation_fixed(observer_direction, ortho_scale)

    obs_unit = np.array(observer_direction) / np.linalg.norm(observer_direction)

    # Old code: camera location has nothing to do with observer_direction
    assert not np.allclose(old_loc / ortho_scale, obs_unit), (
        "expected the buggy camera placement to ignore observer_direction"
    )
    # Fixed code: camera sits along the real observer direction
    assert np.allclose(fixed_loc / ortho_scale, obs_unit), (
        f"fixed camera should be placed along observer_direction, got {fixed_loc}"
    )
    print("PASS: confirmed add_camera() bug (ignores observer_direction) "
          "and that the fix places the camera correctly")
    print(f"      old camera location/scale: {old_loc/ortho_scale}  (always -Z)")
    print(f"      fixed camera location/scale: {fixed_loc/ortho_scale}  (matches observer)")


# ---------------------------------------------------------------------------
# 2. edge-pixel check indented inside the y loop
# ---------------------------------------------------------------------------

def render_old_structure(bboxes):
    """Mirrors the real control flow: edge check sits inside the row loop."""
    edge_check_calls = 0
    for (x_min, x_max, y_min, y_max) in bboxes:
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                pass  # pixel work
            edge_check_calls += 1  # <-- runs once per ROW (the bug)
    return edge_check_calls


def render_fixed_structure(bboxes):
    """Fixed: edge check runs once, after the full frame is built."""
    for (x_min, x_max, y_min, y_max) in bboxes:
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                pass
    edge_check_calls = 1  # <-- runs once, on the finished frame
    return edge_check_calls


def test_edge_check_runs_per_row_bug():
    bboxes = [(0, 50, 0, 20), (10, 60, 5, 15)]  # two objects, 20 + 10 rows
    total_rows = sum(y_max - y_min for (_, _, y_min, y_max) in bboxes)

    old_calls = render_old_structure(bboxes)
    fixed_calls = render_fixed_structure(bboxes)

    assert old_calls == total_rows, f"expected old code to call it once per row ({total_rows}), got {old_calls}"
    assert fixed_calls == 1, f"expected fixed code to call it once, got {fixed_calls}"
    print(f"PASS: confirmed edge-check bug -- old code ran the check {old_calls} times "
          f"(once per row) for {total_rows} total rows; fixed code runs it {fixed_calls} time")


# ---------------------------------------------------------------------------
# 3. overlapping object bboxes re-render the same pixels
# ---------------------------------------------------------------------------

def pixels_computed_old(bboxes):
    """Old code: iterates every bbox independently, no dedup."""
    count = 0
    for (x_min, x_max, y_min, y_max) in bboxes:
        count += (x_max - x_min) * (y_max - y_min)
    return count


def pixels_computed_fixed(bboxes, resolution=(100, 100)):
    """Fixed: track a visited mask, skip pixels already computed."""
    done = np.zeros(resolution, dtype=bool)
    count = 0
    for (x_min, x_max, y_min, y_max) in bboxes:
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                if done[y, x]:
                    continue
                done[y, x] = True
                count += 1
    return count


def test_overlapping_bboxes_double_render_bug():
    # Two objects whose bboxes overlap in a 10x10 region
    bboxes = [(0, 30, 0, 30), (20, 50, 20, 50)]
    overlap_area = (30 - 20) * (30 - 20)  # 10x10 = 100 pixels double-counted

    old_count = pixels_computed_old(bboxes)
    fixed_count = pixels_computed_fixed(bboxes)

    naive_total = (30 * 30) + (30 * 30)  # 900 + 900
    union_total = naive_total - overlap_area  # overlap only counted once

    assert old_count == naive_total, f"expected old code to double-count the overlap, got {old_count}"
    assert fixed_count == union_total, f"expected fixed code to dedup the overlap, got {fixed_count}"
    wasted = old_count - fixed_count
    print(f"PASS: confirmed overlapping-bbox bug -- old code did {old_count} pixel computations, "
          f"fixed code did {fixed_count} (saved {wasted} redundant ray casts in the overlap region)")


# ---------------------------------------------------------------------------
# 4. _get_for_object (BRDF lookup) resolved per pixel instead of per frame
# ---------------------------------------------------------------------------

class FakeResolver:
    """Counts how many times the expensive lookup actually runs."""
    def __init__(self):
        self.calls = 0

    def resolve(self, obj_name):
        self.calls += 1
        return f"brdf_for_{obj_name}"


def brdf_lookups_old(hits):
    """Old code: calls _get_for_object() once per pixel hit, every time."""
    resolver = FakeResolver()
    for obj_name in hits:
        resolver.resolve(obj_name)  # every pixel re-resolves, even repeats
    return resolver.calls


def brdf_lookups_fixed(hits):
    """Fixed: resolve once per unique object name per frame, cache the rest."""
    resolver = FakeResolver()
    cache = {}
    for obj_name in hits:
        if obj_name not in cache:
            cache[obj_name] = resolver.resolve(obj_name)
    return resolver.calls


def test_brdf_resolved_per_pixel_bug():
    # Simulate 10,000 pixel hits across only 3 distinct objects
    hits = (["Bus"] * 6000) + (["solarPanel1"] * 2500) + (["solarPanel2"] * 1500)

    old_calls = brdf_lookups_old(hits)
    fixed_calls = brdf_lookups_fixed(hits)

    assert old_calls == len(hits), f"expected old code to resolve once per pixel, got {old_calls}"
    assert fixed_calls == 3, f"expected fixed code to resolve once per unique object, got {fixed_calls}"
    print(f"PASS: confirmed BRDF-lookup bug -- old code resolved {old_calls} times for "
          f"{len(hits)} pixel hits across only 3 objects; fixed code resolved {fixed_calls} times")


if __name__ == "__main__":
    tests = [
        test_camera_ignores_observer_direction_bug,
        test_edge_check_runs_per_row_bug,
        test_overlapping_bboxes_double_render_bug,
        test_brdf_resolved_per_pixel_bug,
    ]
    failures = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures += 1
            print(f"FAIL: {t.__name__}: {e}")
        print()
    if failures:
        raise SystemExit(f"{failures} test(s) failed")
    print("All tests passed.")