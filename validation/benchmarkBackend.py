#!/usr/bin/env python3
"""
Which ray caster is faster for satlight: the current merged BVHTree, or
Blender's scene.ray_cast?

Nothing in satlight is modified. The scene-based renderer is added here as a
subclass, so you can get the answer before touching raytracer.py.

    python bench_caster.py path/to/model.obj
    python bench_caster.py path/to/model.obj --repeats 10 --resolution 500

Read the output as: whichever total is lower wins. Check that hit counts and
total flux agree between the two — if they don't, the comparison is invalid
and something else is wrong (usually the shadow-ray epsilon).
"""

import argparse
import statistics
import time

import bpy
import numpy as np
from mathutils import Vector

from satlight.raytracer import Renderer


class BenchRenderer(Renderer):
    """Renderer plus a scene.ray_cast implementation, for comparison only."""

    def _cache_brdfs(self):
        """Resolve each mesh object's BRDF once per frame, keyed by name."""
        self._brdf_cache = {
            o.name: self._get_for_object(o.name)
            for o in bpy.context.scene.objects
            if o.type == "MESH"
        }

    def render_scene(self):
        """Same physics as render(), but casting against Blender's scene.

        No per-frame BVH build: Blender holds a local-space tree per object and
        transforms the ray into it, so attitude and panel rotations are free.
        """
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        ray_cast = bpy.context.scene.ray_cast
        self._cache_brdfs()

        cam_matrix = self.camera_object.matrix_world
        ray_dir_world = (cam_matrix.to_3x3() @ Vector((0.0, 0.0, -1.0))).normalized()
        vx, vy, vz = -ray_dir_world.x, -ray_dir_world.y, -ray_dir_world.z
        view_dir = np.array((vx, vy, vz))

        self.image = np.zeros((self.resolution_y, self.resolution_x))
        self.hit_count = 0
        self.shadow_count = 0

        mask = np.zeros((self.resolution_y, self.resolution_x), dtype=bool)
        for x_min, x_max, y_min, y_max in self.get_object_pixel_bboxes():
            mask[y_min:y_max, x_min:x_max] = True

        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            self._check_frame_edges()
            return self.image

        screen_x = (2.0 * xs / self.resolution_x) - 1.0
        screen_y = 1.0 - (2.0 * ys / self.resolution_y)
        cam_local = np.column_stack((
            screen_x * (self.ortho_scale / 2),
            screen_y * (self.ortho_scale / 2) / self.aspect_ratio,
            np.zeros(xs.size),
        ))
        Mc = np.array(cam_matrix)
        origins_list = (cam_local @ Mc[:3, :3].T + Mc[:3, 3]).tolist()

        sun_dir_np = np.array(self.sun_direction, dtype=float)
        sx, sy, sz = sun_dir_np.tolist()
        sun_dir_vec = Vector((sx, sy, sz))

        es_on = self.add_earthshine and self.earthshine_direction is not None
        es_dir_np = np.array(self.earthshine_direction, dtype=float) if es_on else None

        inv_d2 = 1.0 / self.distance_to_observer ** 2
        k_sun = self.solar_constant * self.pixel_area * inv_d2
        k_es = (self.earthshine_irradiance * self.pixel_area * inv_d2) if es_on else 0.0

        brdf_cache = self._brdf_cache
        image = self.image

        for i in range(xs.size):
            ox, oy, oz = origins_list[i]
            hit, loc, normal, _face, obj, _m = ray_cast(
                depsgraph, Vector((ox, oy, oz)), ray_dir_world
            )
            if not hit:
                continue
            self.hit_count += 1

            # Plain floats: numpy on 3-vectors costs more in overhead than it saves.
            nx, ny, nz = normal.x, normal.y, normal.z
            cos_theta = nx * vx + ny * vy + nz * vz
            if cos_theta <= 0.0:
                continue

            brdf_func, brdf_kwargs = brdf_cache[obj.name]
            n_dot_l = nx * sx + ny * sy + nz * sz
            flux = 0.0
            hit_normal = None

            if n_dot_l > 0.0:
                shadow_origin = Vector(
                    (loc.x + nx * 1e-4, loc.y + ny * 1e-4, loc.z + nz * 1e-4)
                )
                if not ray_cast(depsgraph, shadow_origin, sun_dir_vec)[0]:
                    hit_normal = np.array((nx, ny, nz))
                    flux += (
                        brdf_func(hit_normal, sun_dir_np, view_dir, **brdf_kwargs)
                        * k_sun
                        * cos_theta
                    )
                else:
                    self.shadow_count += 1

            if es_on:
                if hit_normal is None:
                    hit_normal = np.array((nx, ny, nz))
                flux += (
                    brdf_func(hit_normal, es_dir_np, view_dir, **brdf_kwargs)
                    * k_es
                    * cos_theta
                )

            image[ys[i], xs[i]] = flux

        self._check_frame_edges()
        return self.image


def time_it(fn, repeats):
    """Warm up once, then return per-run wall times."""
    fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def report(label, times):
    print(
        f"  {label:<22} median {statistics.median(times):7.3f} s"
        f"   min {min(times):7.3f} s   max {max(times):7.3f} s"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("obj_path", help="path to the .obj or .stl model")
    p.add_argument("--repeats", type=int, default=5, help="timed runs per caster")
    p.add_argument("--resolution", type=int, default=500, help="square frame size")
    p.add_argument(
        "--distance", type=float, default=1000e3, help="distance to observer (m)"
    )
    p.add_argument("--brdf", default="lambertian")
    args = p.parse_args()

    sun = Vector((0.3, 0.5, 0.8)).normalized()
    obs = Vector((0.0, 0.0, 1.0)).normalized()

    r = BenchRenderer(
        args.obj_path,
        sun_direction=sun,
        observer_direction=obs,
        distance_to_observer=args.distance,
        resolution=(args.resolution, args.resolution),
        brdf=args.brdf,
    )
    r.initialise_scene()
    r.update(sun, obs, args.distance)

    n_tris = 0
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for o in bpy.context.scene.objects:
        if o.type != "MESH":
            continue
        m = o.evaluated_get(depsgraph).to_mesh()
        try:
            m.calc_loop_triangles()
        except AttributeError:
            pass
        n_tris += len(m.loop_triangles)
        o.evaluated_get(depsgraph).to_mesh_clear()

    print(f"\nmodel      {args.obj_path}")
    print(f"triangles  {n_tris:,}")
    print(f"frame      {args.resolution}x{args.resolution}   repeats {args.repeats}\n")

    # --- current path: merged BVHTree, rebuilt every frame -------------------
    print("merged BVHTree (current)")
    build_times = time_it(r.build_bvhs, args.repeats)
    total_bvh = time_it(r.render, args.repeats)
    cast_only = [t - statistics.median(build_times) for t in total_bvh]
    report("build_bvhs", build_times)
    report("cast + shade", cast_only)
    report("TOTAL", total_bvh)
    bvh_hits, bvh_shadows, bvh_flux = r.hit_count, r.shadow_count, r.image.sum()
    build_frac = 100 * statistics.median(build_times) / statistics.median(total_bvh)
    print(f"  build is {build_frac:.0f}% of the frame\n")

    # --- alternative: Blender scene.ray_cast, no rebuild ---------------------
    print("scene.ray_cast (no rebuild)")
    total_scene = time_it(r.render_scene, args.repeats)
    report("TOTAL", total_scene)
    scn_hits, scn_shadows, scn_flux = r.hit_count, r.shadow_count, r.image.sum()
    print()

    # --- agreement check -----------------------------------------------------
    print("agreement")
    print(f"  hits     {bvh_hits:>8,}  vs {scn_hits:>8,}")
    print(f"  shadowed {bvh_shadows:>8,}  vs {scn_shadows:>8,}")
    print(f"  flux     {bvh_flux:.6e}  vs {scn_flux:.6e}")
    if bvh_flux > 0:
        dev = abs(scn_flux - bvh_flux) / bvh_flux
        print(f"  relative difference {dev:.2%}")
        if dev > 0.01:
            print("  ^ over 1% — the two are not rendering the same thing, so the")
            print("    timings below mean nothing. Check the shadow-ray epsilon first.")
    print()

    # --- verdict -------------------------------------------------------------
    m_bvh, m_scn = statistics.median(total_bvh), statistics.median(total_scene)
    winner, speedup = (
        ("scene.ray_cast", m_bvh / m_scn) if m_scn < m_bvh else ("merged BVHTree", m_scn / m_bvh)
    )
    print(f"VERDICT: {winner} is {speedup:.2f}x faster. Keep it, delete the other.\n")


if __name__ == "__main__":
    main()