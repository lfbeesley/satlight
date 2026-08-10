import bpy
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
import imageio
import os
from mathutils import Vector
import mathutils as mu
from tqdm import tqdm

from satlight.geometry import Geometry
from satlight.raytracer import Renderer

# =====================================================================
# STIX font
# =====================================================================
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['mathtext.fontset'] = 'stix'

# =====================================================================
# Config
# =====================================================================
obj_path = r'/Users/l.beesley@bham.ac.uk/Documents/Lightcurves/satlight/models/Starlink/starlink_HR/uploads_files_3710445_SpaceXStarlinkSatelliteHighRes.obj/SpaceXStarlinkSatelliteHighRes.obj'
output_dir = r'/Users/l.beesley@bham.ac.uk/Documents/Lightcurves/satlight/output'

observer_lat   =  28.7622
observer_lon   = -17.8897
observer_alt_m =  2396.0

ID      = 'SYNTHETIC-STARLINK-550'
alt_km  = 550.0
inc_deg = 53.0
R_EARTH_KM = 6371.0
MU_EARTH   = 398600.4418
a_km       = R_EARTH_KM + alt_km
n_rad_s    = np.sqrt(MU_EARTH / a_km**3)

t_zenith = datetime.datetime(2025, 3, 8, 19, 50, 0)
t_start  = t_zenith - datetime.timedelta(seconds=180)
t_end    = t_zenith + datetime.timedelta(seconds=180)

exposure_s     = 10
min_elevation  = 10.0
resolution     = (512, 512)
solar_constant = 1361.0
mag_zeropoint  = -18.0

duration_s = (t_end - t_start).total_seconds()
N_frames   = int(duration_s / exposure_s)
time_grid  = [t_start + datetime.timedelta(seconds=i * exposure_s)
              for i in range(N_frames)]
print(f"Frames: {N_frames}  ({t_start} to {t_end}, {exposure_s}s cadence)")

video_fps        = 10
video_cmap       = 'inferno'
video_percentile = 99.9

# =====================================================================
# BRDF dicts
# =====================================================================
brdf = {
    'solarPanel':      ('cook_torrance', {'albedo': 0.85, 'roughness': 0.15, 'metallic': 0.95}),
    'Antenna':         ('cook_torrance', {'albedo': 0.85, 'roughness': 0.15, 'metallic': 0.95}),
    'omniAntenna':     ('cook_torrance', {'albedo': 0.7,  'roughness': 0.3,  'metallic': 0.9}),
    'bus(Aluminium)':  ('cook_torrance', {'albedo': 0.6,  'roughness': 0.25, 'metallic': 0.9}),
    'bus(MLI)':        ('oren_nayar',    {'albedo': 0.25, 'roughness': 0.5}),
    'default':         ('lambertian',    {'albedo': 0.3}),
}

brdf_darkened = {
    'solarPanel':      ('blinn_phong', {'albedo': 0.10, 'specular': 0.30, 'shininess': 16}),
    'Antenna':         ('oren_nayar',  {'albedo': 0.05, 'roughness': 0.6}),
    'omniAntenna':     ('oren_nayar',  {'albedo': 0.08, 'roughness': 0.5}),
    'bus(Aluminium)':  ('oren_nayar',  {'albedo': 0.06, 'roughness': 0.55}),
    'bus(MLI)':        ('oren_nayar',  {'albedo': 0.08, 'roughness': 0.6}),
    'default':         ('lambertian',  {'albedo': 0.05}),
}

BRDF_CONFIGS = {
    'reg':      brdf,
    'darkened': brdf_darkened,
}

# =====================================================================
# Attitude: fixed rotations (no per-frame sun tracking — static per run)
# =====================================================================
ATTITUDE_RX = np.radians(90)
ATTITUDE_RY = np.radians(0)
ATTITUDE_RZ = np.radians(180)

ATTITUDE_RX_rot = np.radians(15 + 90)   # "off" attitude: RX offset by 15 deg

ATTITUDE_CONFIGS = {
    'reg': mu.Euler((ATTITUDE_RX,     ATTITUDE_RY, ATTITUDE_RZ), 'XYZ').to_matrix(),
    'off': mu.Euler((ATTITUDE_RX_rot, ATTITUDE_RY, ATTITUDE_RZ), 'XYZ').to_matrix(),
}

# The 4 (BRDF, attitude) combinations to run:
RUNS = [
    ('reg',      'reg'),
    ('darkened', 'reg'),
    ('reg',      'off'),
    ('darkened', 'off'),
]

# =====================================================================
# Overhead-pass orbital elements
# =====================================================================
from skyfield.api import Loader
_data_dir = os.path.join(os.path.expanduser('~'), '.satlight')
load = Loader(_data_dir)
ts = load.timescale()

def gast_deg(t_utc):
    t = ts.utc(t_utc.year, t_utc.month, t_utc.day,
               t_utc.hour, t_utc.minute,
               t_utc.second + t_utc.microsecond / 1e6)
    return t.gast * 15.0

_f = 1.0 / 298.257223563
lat_gc_deg = np.degrees(np.arctan((1.0 - _f)**2
                                  * np.tan(np.radians(observer_lat))))

u0_deg = np.degrees(np.arcsin(np.sin(np.radians(lat_gc_deg))
                              / np.sin(np.radians(inc_deg))))
dlam_deg = np.degrees(np.arctan2(
    np.sin(np.radians(u0_deg)) * np.cos(np.radians(inc_deg)),
    np.cos(np.radians(u0_deg))))
raan_deg = (observer_lon + gast_deg(t_zenith) - dlam_deg) % 360.0

print(f"Overhead pass: u0={u0_deg:.2f} deg, dlam={dlam_deg:.2f} deg, "
      f"RAAN={raan_deg:.2f} deg, zenith at {t_zenith} UTC")

def make_geometry(t_utc):
    geom = Geometry()
    geom.create_observer(observer_lat, observer_lon, observer_alt_m)
    dt_s  = (t_utc - t_zenith).total_seconds()
    u_deg = (u0_deg + np.degrees(n_rad_s * dt_s)) % 360.0
    geom.create_satellite_from_elements(
        a_km=a_km, e=0.0, i_deg=inc_deg,
        raan_deg=raan_deg, argp_deg=0.0, nu_deg=u_deg, epoch=t_utc)
    geom.set_time((t_utc.year, t_utc.month, t_utc.day,
                   t_utc.hour, t_utc.minute,
                   t_utc.second + t_utc.microsecond / 1e6))
    return geom


def apply_attitude(body_mat):
    """Fixed rotation applied identically to every mesh (static per run)."""
    euler = body_mat.to_euler()
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.rotation_euler = euler
    bpy.context.view_layer.update()


# =====================================================================
# Video helpers
# =====================================================================
def frame_to_rgb(image, vmin, vmax, cmap_name=video_cmap):
    arr = np.asarray(image, dtype=float)
    gray = arr.sum(axis=-1) if arr.ndim == 3 else arr
    norm = np.clip((gray - vmin) / max(vmax - vmin, 1e-30), 0.0, 1.0)
    return (plt.get_cmap(cmap_name)(norm)[..., :3] * 255).astype(np.uint8)

def save_pass_mp4(images, out_path, vmin, vmax, fps=video_fps):
    if not images:
        print(f"  no frames to write -> {out_path} (skipped)")
        return
    with imageio.get_writer(out_path, fps=fps, codec='libx264',
                             quality=8, macro_block_size=None) as writer:
        for image in images:
            writer.append_data(frame_to_rgb(image, vmin, vmax))
    print(f"Saved video -> {out_path}")

def robust_max(images, percentile=video_percentile):
    if not images:
        return 0.0
    return max(float(np.percentile(im, percentile)) for im in images)


# =====================================================================
# Single-pass runner
# =====================================================================
def run_pass(brdf_name, attitude_name):
    print(f"\n=== Run: brdf={brdf_name}, attitude={attitude_name} ===")
    body_mat = ATTITUDE_CONFIGS[attitude_name]

    geom = make_geometry(t_start)
    sun_lvlh_0 = geom.incident_vector_lvlh
    obs_lvlh_0 = geom.outgoing_vector_lvlh
    dist_obs_0 = float(geom.prop_distance) * 1e3

    renderer = Renderer(
        obj_path             = obj_path,
        sun_direction        = Vector(sun_lvlh_0.tolist()),
        observer_direction   = Vector(obs_lvlh_0.tolist()),
        distance_to_observer = dist_obs_0,
        resolution           = resolution,
        solar_constant       = solar_constant,
        add_earthshine       = False,
        brdf                 = BRDF_CONFIGS[brdf_name],
    )
    renderer.initialise_scene()

    # Attitude is fixed for the whole run — set once, not per frame.
    apply_attitude(body_mat)

    out = {k: [] for k in ('t', 'flux', 'mag', 'phase', 'el')}
    images = []
    n_below, n_eclipsed = 0, 0

    for t_utc in tqdm(time_grid, desc=f"{brdf_name}/{attitude_name}"):
        geom = make_geometry(t_utc)

        sat_pos = geom.positions['satellite']
        obs_pos = geom.positions['observer']
        diff    = sat_pos - obs_pos
        zenith  = obs_pos / np.linalg.norm(obs_pos)
        el_deg  = float(np.degrees(np.arcsin(np.clip(
                    np.dot(diff / np.linalg.norm(diff), zenith), -1, 1))))
        if el_deg < min_elevation:
            n_below += 1
            continue

        illum     = float(geom.eclipse_type())
        phase_deg = float(np.degrees(geom.phase_angle))
        if illum < 0.01:
            n_eclipsed += 1
            continue

        sun_lvlh = geom.incident_vector_lvlh
        obs_lvlh = geom.outgoing_vector_lvlh
        dist_obs = float(geom.prop_distance) * 1e3

        renderer.update(
            sun_direction        = Vector(sun_lvlh.tolist()),
            observer_direction   = Vector(obs_lvlh.tolist()),
            distance_to_observer = dist_obs,
        )
        bpy.context.view_layer.update()

        image = renderer.render()
        flux  = float(np.sum(image)) * illum
        mag   = -2.5 * np.log10(flux) + mag_zeropoint if flux > 0 else np.nan

        out['t'].append(t_utc)
        out['flux'].append(flux)
        out['mag'].append(mag)
        out['phase'].append(phase_deg)
        out['el'].append(el_deg)
        images.append(image)

    out['sec'] = np.array([(t - t_start).total_seconds() for t in out['t']])
    for k in ('flux', 'mag', 'phase', 'el'):
        out[k] = np.array(out[k])
    out['images'] = images

    if len(out['sec']) == 0:
        print(f"  WARNING: 0 frames rendered! below_horizon={n_below}, eclipsed={n_eclipsed}")
    else:
        print(f"  rendered {len(out['sec'])} frames "
              f"(skipped: {n_below} below horizon, {n_eclipsed} eclipsed), "
              f"mag {np.nanmin(out['mag']):.2f} - {np.nanmax(out['mag']):.2f}")
    return out


# =====================================================================
# Execute all 4 runs
# =====================================================================
results = {}
for brdf_name, attitude_name in RUNS:
    results[(brdf_name, attitude_name)] = run_pass(brdf_name, attitude_name)

# =====================================================================
# Lightcurve comparison plot (STIX font)
# =====================================================================
ref_key = RUNS[0]
ref = results[ref_key]
colours = plt.cm.tab10(np.linspace(0, 1, len(RUNS)))

fig, ax = plt.subplots(figsize=(11, 4))
for (key, col) in zip(RUNS, colours):
    r = results[key]
    if len(r['sec']) == 0:
        continue
    label = f"{key[0]} / {key[1]}"
    ax.plot(r['sec'], r['mag'], '.-', ms=4, lw=1.0, color=col, label=label)

ax.invert_yaxis()
ax.set_xlabel(f"Seconds since {t_start:%H:%M:%S} UTC")
ax.set_ylabel("Apparent magnitude")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()

fig_path = os.path.join(output_dir, 'starlink_brdf_attitude_comparison.png')
plt.savefig(fig_path, dpi=150)
plt.show()
print(f"\nSaved comparison figure -> {fig_path}")

# =====================================================================
# Per-run MP4s, shared brightness scale, 512x512
# =====================================================================
video_vmin = 0.0
video_vmax = max(robust_max(results[key]['images']) for key in RUNS)
if video_vmax <= 0:
    video_vmax = 1.0
print(f"\nShared video brightness scale: vmin={video_vmin}, vmax={video_vmax:.4g}")

for key in RUNS:
    b, a = key
    video_path = os.path.join(output_dir, f'{b}_{a}_pass.mp4')
    save_pass_mp4(results[key]['images'], video_path, video_vmin, video_vmax)

# =====================================================================
# Save arrays
# =====================================================================
npz_path = os.path.join(output_dir, 'starlink_brdf_attitude_results.npz')
np.savez(npz_path, **{
    f"{b}_{a}_{field}": results[(b, a)][field]
    for (b, a) in RUNS
    for field in ('sec', 'flux', 'mag', 'phase', 'el')
})
print(f"Saved arrays -> {npz_path}")