#!/usr/bin/env python3
"""
How much does self-shadowing change a light curve, and does it depend on shape?

Sweeps solar phase angle with the observer fixed, rendering each geometry twice
-- shadow rays on and off -- and reports the magnitude difference. At each phase
angle the body is stepped through a set of attitudes and the results averaged,
so the answer is the expected contribution for that shape rather than one lucky
orientation.

No ephemeris, no orbit, no pass visibility. Shadowing is a property of the
geometry and the illumination angle, so simulating a real night would only add
variables that have nothing to do with the claim.

All three models use the same Lambertian BRDF. Materials are held fixed on
purpose: the figure is about shape, and per-surface BRDFs would confound the
comparison with a materials difference.

This calls satlight's own Renderer and toggles `add_shadows`, so the published
figure comes from the shipped code rather than a copy of it. Anything wrong
with the shadow ray -- the epsilon in particular -- shows up here.

    python shadow_ablation.py
    python shadow_ablation.py --n-phase 37 --n-attitude 16
    python shadow_ablation.py --only starlink

Earthshine is NOT covered here -- it depends on altitude, so it needs a real
orbit and belongs in the existing per-scenario harnesses. See the note at the
bottom of this file.
"""

import argparse
import time

import bpy
import numpy as np
from mathutils import Vector

from satlight.raytracer import Renderer

BASE = "/Users/l.beesley@bham.ac.uk/Documents/Lightcurves/satlight/models"

MODELS = {
    "rocketbody": {
        "path": f"{BASE}/Falcon9 RB model/F9_Upperstage.stl",
        "label": "Falcon 9 upper stage",
        "colour": "C2",
    },
    "starlink": {
        "path": f"{BASE}/Starlink/starlink_HR/uploads_files_3710445_"
                f"SpaceXStarlinkSatelliteHighRes.obj/SpaceXStarlinkSatelliteHighRes.obj",
        "label": "Starlink",
        "colour": "C0",
    },
    "skynet": {
        "path": f"{BASE}/Skynet 5D/skynet.obj",
        "label": "Skynet 5D",
        "colour": "C1",
    },
}


def sun_at_phase(phase_deg):
    """Sun direction at a given solar phase angle, observer fixed along +Z."""
    a = np.radians(phase_deg)
    return np.array([np.sin(a), 0.0, np.cos(a)])


def attitude_matrices(n):
    """n body orientations spread over the rotation group, deterministic.

    Golden-angle spiral for the pole plus an even spin about it, so the set is
    reproducible run to run -- a random sample would make the error bars move
    every time the figure is regenerated.
    """
    import mathutils as mu

    mats = []
    ga = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n):
        z = 1.0 - 2.0 * (i + 0.5) / n
        r = np.sqrt(max(0.0, 1.0 - z * z))
        theta = ga * i
        axis = mu.Vector((r * np.cos(theta), r * np.sin(theta), z)).normalized()
        angle = 2.0 * np.pi * ((i * 0.618034) % 1.0)
        mats.append(mu.Matrix.Rotation(angle, 3, axis))
    return mats


def run_model(key, spec, phases, n_attitude, resolution, distance_m, solar_constant):
    print(f"\n{spec['label']}")
    print(f"  {spec['path']}")

    obs = Vector((0.0, 0.0, 1.0))
    r = Renderer(
        obj_path=spec["path"],
        sun_direction=Vector(sun_at_phase(0.0).tolist()),
        observer_direction=obs,
        distance_to_observer=distance_m,
        resolution=resolution,
        solar_constant=solar_constant,
        add_earthshine=False,
        brdf=("lambertian", {"albedo": 0.3}),
    )
    r.initialise_scene()
    r.update(Vector(sun_at_phase(0.0).tolist()), obs, distance_m)

    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    print(f"  {len(meshes)} mesh object(s): "
          f"{', '.join(o.name for o in meshes[:6])}"
          f"{' ...' if len(meshes) > 6 else ''}")
    
    for o in bpy.context.scene.objects:
        if o.type == 'MESH':
            bb = [o.matrix_world @ Vector(c) for c in o.bound_box]
            print(o.name, round((Vector(map(max, zip(*bb))) - Vector(map(min, zip(*bb)))).length, 1))

    mats = attitude_matrices(n_attitude)
    delta = np.full((len(phases), n_attitude), np.nan)
    shadowed_frac = np.zeros((len(phases), n_attitude))

    t0 = time.perf_counter()
    for j, mat in enumerate(mats):
        for o in meshes:
            o.rotation_euler = mat.to_euler()
        bpy.context.view_layer.update()

        for i, ph in enumerate(phases):
            sun = Vector(sun_at_phase(ph).tolist())
            r.sun_direction = sun
            r.update(sun, obs, distance_m)

            r.add_shadows = False
            f_off = float(np.sum(r.render()))

            r.add_shadows = True
            f_on = float(np.sum(r.render()))
            shadowed_frac[i, j] = r.shadow_count / max(r.hit_count, 1)

            if f_off > 0 and f_on > 0:
                delta[i, j] = -2.5 * np.log10(f_on / f_off)

        print(f"  attitude {j + 1}/{n_attitude}", end="\r", flush=True)

    n_render = 2 * len(phases) * n_attitude
    dt = time.perf_counter() - t0
    print(f"  {n_render} renders in {dt:.1f} s ({dt / n_render * 1000:.0f} ms each)")
    # --- diagnostic: the single worst-shadowed frame -------------------------
    i, j = np.unravel_index(np.nanargmax(shadowed_frac), shadowed_frac.shape)
    for o in meshes:
        o.rotation_euler = mats[j].to_euler()
    bpy.context.view_layer.update()
    sun = Vector(sun_at_phase(phases[i]).tolist())
    r.sun_direction = sun
    r.update(sun, obs, distance_m)

    r.add_shadows = False
    off = r.render().copy()
    r.add_shadows = True
    on = r.render().copy()
    lost = off - on

    ny_, nx_ = np.nonzero(off > 0)
    sl = (slice(ny_.min(), ny_.max() + 1), slice(nx_.min(), nx_.max() + 1))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    vmax = off[sl].max()
    ax[0].imshow(off[sl], cmap='gray', vmin=0, vmax=vmax)
    ax[0].set_title('shadows off')
    ax[1].imshow(on[sl], cmap='gray', vmin=0, vmax=vmax)
    ax[1].set_title('shadows on')
    ax[2].imshow(on[sl], cmap='gray', vmin=0, vmax=vmax)
    ax[2].imshow(np.ma.masked_where(lost[sl] <= 0, lost[sl]),
                 cmap='autumn', alpha=0.9)
    ax[2].set_title(f'occluded (red)  —  {shadowed_frac[i, j]*100:.1f}% of pixels')
    for a in ax:
        a.axis('off')
    fig.suptitle(f"{spec['label']}  —  phase {phases[i]:.0f}°, attitude {j}")
    fig.tight_layout()
    fig.savefig(f"shadowmask_{key}.png", dpi=150)
    plt.close(fig)
    print(f"  wrote shadowmask_{key}.png "
          f"(phase {phases[i]:.0f}°, attitude {j}, {shadowed_frac[i, j]*100:.1f}% shadowed)")
    return delta, shadowed_frac


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-phase", type=int, default=25)
    p.add_argument("--n-attitude", type=int, default=12)
    p.add_argument("--resolution", type=int, default=400)
    p.add_argument("--distance", type=float, default=1000e3)
    p.add_argument("--solar-constant", type=float, default=1361.0)
    p.add_argument("--only", default=None, help="run one model key only")
    p.add_argument("--out", default="shadow_ablation.png")
    args = p.parse_args()

    phases = np.linspace(0.0, 170.0, args.n_phase)
    keys = [args.only] if args.only else list(MODELS)

    results = {}
    for k in keys:
        results[k] = run_model(
            k, MODELS[k], phases, args.n_attitude,
            (args.resolution, args.resolution), args.distance, args.solar_constant,
        )

    print("\n\nMagnitude change from enabling self-shadowing")
    print(f"  {'model':<22} {'median':>9} {'95th pct':>10} {'peak':>9} {'max shadowed':>14}")
    for k in keys:
        d, sf = results[k]
        a = np.abs(d[np.isfinite(d)])
        if a.size == 0:
            print(f"  {MODELS[k]['label']:<22} {'no data':>9}")
            continue
        print(f"  {MODELS[k]['label']:<22} {np.median(a):9.4f} "
              f"{np.percentile(a, 95):10.4f} {a.max():9.4f} {sf.max() * 100:13.1f}%")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k in keys:
        d, _ = results[k]
        med = np.nanmedian(d, axis=1)
        lo = np.nanpercentile(d, 16, axis=1)
        hi = np.nanpercentile(d, 84, axis=1)
        ax.plot(phases, med, color=MODELS[k]["colour"], lw=1.6, label=MODELS[k]["label"])
        ax.fill_between(phases, lo, hi, color=MODELS[k]["colour"], alpha=0.18, lw=0)

    ax.axhspan(-0.05, 0.05, color="0.9", zorder=0)
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_xlabel("Solar phase angle (deg)")
    ax.set_ylabel(r"$\Delta m$ from self-shadowing")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nSaved {args.out}")
    print("Shaded band is the 16th-84th percentile across attitudes; grey strip "
          "is +/-0.05 mag, a typical photometric precision floor.")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# EARTHSHINE
#
# Earthshine is not a shape effect, so it does not belong in this sweep. Its
# size is set by altitude: at 550 km the Earth fills most of the sky below the
# spacecraft, at GEO it subtends about 17 degrees. The comparison that makes
# the point is therefore Starlink (LEO) against Skynet (GEO), run in the
# existing per-scenario harnesses where a real orbit exists.
#
# Use ablation.py there, and note two things:
#
#   1. `flux = image.sum() * illum` scales earthshine by the solar eclipse
#      fraction, which is wrong. Move illum onto the solar term inside
#      render() and leave the earthshine term alone.
#   2. Do not skip eclipsed timesteps. In eclipse the solar term vanishes and
#      earthshine is the whole signal, which is the strongest part of the case.
# ---------------------------------------------------------------------------