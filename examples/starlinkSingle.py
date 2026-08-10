
import bpy
import numpy as np
import datetime
import matplotlib.pyplot as plt
import os
from mathutils import Vector
import mathutils as mu
from tqdm import tqdm

from satlight.geometry import Geometry
from satlight.raytracer import Renderer

# =====================================================================
# Config
# =====================================================================
obj_path = r'/Users/l.beesley@bham.ac.uk/Documents/Lightcurves/satlight/models/Starlink/starlink_HR/uploads_files_3710445_SpaceXStarlinkSatelliteHighRes.obj/SpaceXStarlinkSatelliteHighRes.obj'
output_dir = r'/Users/l.beesley@bham.ac.uk/Documents/Lightcurves/satlight/output'

# Observer: Roque de los Muchachos, La Palma
observer_lat   =  28.7622
observer_lon   = -17.8897
observer_alt_m =  2396.0

# Synthetic target: circular 550 km, 53 deg orbit passing through zenith
# (TLE pass replaced — original window was likely in Earth shadow: solar
# depression at 20:51 UTC was ~24 deg, and an overhead 550 km satellite is
# only sunlit for depression < arccos(R/(R+h)) ~ 23 deg.)
ID      = 'SYNTHETIC-STARLINK-550'
alt_km  = 550.0
inc_deg = 53.0
R_EARTH_KM = 6371.0
MU_EARTH   = 398600.4418            # km^3/s^2
a_km       = R_EARTH_KM + alt_km
n_rad_s    = np.sqrt(MU_EARTH / a_km**3)   # mean motion, period ~95.6 min

# Zenith crossing time — ~50 min after sunset: satellite sunlit, sky dark
t_zenith = datetime.datetime(2025, 3, 8, 19, 50, 0)

# Pass window centred on the zenith crossing
t_start = t_zenith - datetime.timedelta(seconds=180)
t_end   = t_zenith + datetime.timedelta(seconds=180)

exposure_s    = 60                 # per-frame integration
min_elevation = 10.0                    # deg
resolution    = (512, 512)
solar_constant = 1361.0
mag_zeropoint  = -18.0

solar_panel_names = ['Solar_Panel']     # articulated mesh(es)
PANEL_HINGE_AXIS  = 'Y'                 # body-frame hinge axis — VERIFY (see header)

duration_s = (t_end - t_start).total_seconds()
N_frames   = int(duration_s / exposure_s)
time_grid  = [t_start + datetime.timedelta(seconds=i * exposure_s)
              for i in range(N_frames)]
print(f"Frames: {N_frames}  ({t_start} to {t_end}, {exposure_s}s cadence)")

# =====================================================================
# BRDF configurations
# =====================================================================
# 'baseline'  : as-launched shiny spacecraft (your current values).
# 'darkened'  : DarkSat-style mitigation — diffuse low-albedo coating on
#               antennas + bus, and a lower-specular panel (AR film /
#               reduced backside reflectivity). DarkSat was measured
#               ~0.8 mag fainter than standard v1.0, so a sanity check on
#               the modelled Δmag against that is a nice validation point.
BRDF_CONFIGS = {
    'baseline': {
        'Solar_Panel':       ('blinn_phong',   {'albedo': 0.15, 'specular': 0.95, 'shininess': 64}),
        'Antenna_1_Antenna': ('cook_torrance', {'albedo': 0.85, 'roughness': 0.15, 'metallic': 0.95}),
        'Antenna_2_Antenna': ('cook_torrance', {'albedo': 0.85, 'roughness': 0.15, 'metallic': 0.95}),
        'Antenna_3_Antenna': ('cook_torrance', {'albedo': 0.85, 'roughness': 0.15, 'metallic': 0.95}),
        'Antenna_4_Antenna': ('cook_torrance', {'albedo': 0.85, 'roughness': 0.15, 'metallic': 0.95}),
        'default':           ('lambertian',    {'albedo': 0.3}),
    },
    'darkened': {
        'Solar_Panel':       ('blinn_phong',   {'albedo': 0.10, 'specular': 0.30, 'shininess': 16}),
        'Antenna_1_Antenna': ('oren_nayar',    {'albedo': 0.05, 'roughness': 0.6}),
        'Antenna_2_Antenna': ('oren_nayar',    {'albedo': 0.05, 'roughness': 0.6}),
        'Antenna_3_Antenna': ('oren_nayar',    {'albedo': 0.05, 'roughness': 0.6}),
        'Antenna_4_Antenna': ('oren_nayar',    {'albedo': 0.05, 'roughness': 0.6}),
        'default':           ('lambertian',    {'albedo': 0.05}),
    },
}

# =====================================================================
# Attitude modes
# =====================================================================
# All modes are nadir-pointing with yaw steering toward the sun (as in your
# draft: RX=90 deg, RZ = yaw - pi/2, recomputed EVERY frame since the LVLH
# sun direction rotates through a LEO pass). They differ in panel strategy:
#
#   'sun_tracking' : panel normal driven onto the projected sun direction
#   'edge_on'      : panel feathered 90 deg from sun — SpaceX knife-edge conop
#   plus an optional fixed offset in degrees.
ATTITUDE_MODES = {
    'nominal':    {'panel_strategy': 'sun_tracking', 'panel_offset_deg': 0.0},
    'feathered':  {'panel_strategy': 'edge_on',      'panel_offset_deg': 0.0},
    'offset_30':  {'panel_strategy': 'sun_tracking', 'panel_offset_deg': 30.0},
}

# The (brdf, attitude) combinations to actually run:
RUNS = [
    ('baseline', 'nominal'),
    ('baseline', 'feathered'),
    ('darkened', 'nominal'),
    ('darkened', 'feathered'),
]

# =====================================================================
# Overhead-pass orbital elements
# =====================================================================
from skyfield.api import Loader
_data_dir = os.path.join(os.path.expanduser('~'), '.satlight')
load = Loader(_data_dir)
ts = load.timescale()

def gast_deg(t_utc):
    """Apparent sidereal time in degrees (same convention as geo_raan)."""
    t = ts.utc(t_utc.year, t_utc.month, t_utc.day,
               t_utc.hour, t_utc.minute,
               t_utc.second + t_utc.microsecond / 1e6)
    return t.gast * 15.0

# Geocentric latitude of the observer (geodetic -> geocentric, ~0.16 deg
# correction at this latitude — keeps the crossing within ~1 deg of zenith)
_f = 1.0 / 298.257223563
lat_gc_deg = np.degrees(np.arctan((1.0 - _f)**2
                                  * np.tan(np.radians(observer_lat))))

# Ascending zenith crossing: argument of latitude where orbit latitude
# equals observer latitude, and the RAAN that puts it over the observer's
# longitude at t_zenith.
u0_deg = np.degrees(np.arcsin(np.sin(np.radians(lat_gc_deg))
                              / np.sin(np.radians(inc_deg))))
dlam_deg = np.degrees(np.arctan2(
    np.sin(np.radians(u0_deg)) * np.cos(np.radians(inc_deg)),
    np.cos(np.radians(u0_deg))))
raan_deg = (observer_lon + gast_deg(t_zenith) - dlam_deg) % 360.0

print(f"Overhead pass: u0={u0_deg:.2f} deg, dlam={dlam_deg:.2f} deg, "
      f"RAAN={raan_deg:.2f} deg, zenith at {t_zenith} UTC")

def make_geometry(t_utc):
    """Rebuild geometry per timestep (GEO-script pattern), advancing the
    argument of latitude by two-body mean motion from the zenith crossing."""
    geom = Geometry()
    geom.create_observer(observer_lat, observer_lon, observer_alt_m)
    dt_s  = (t_utc - t_zenith).total_seconds()
    u_deg = (u0_deg + np.degrees(n_rad_s * dt_s)) % 360.0
    geom.create_satellite_from_elements(
        a_km=a_km, e=0.0, i_deg=inc_deg,
        raan_deg=raan_deg,
        argp_deg=0.0,
        nu_deg=u_deg,          # circular orbit: nu == argument of latitude
        epoch=t_utc)
    geom.set_time((t_utc.year, t_utc.month, t_utc.day,
                   t_utc.hour, t_utc.minute,
                   t_utc.second + t_utc.microsecond / 1e6))
    return geom


def reset_blender_scene():
    """Hard reset of bpy state between configurations.

    Only needed if Renderer.initialise_scene() doesn't clear the scene
    itself — otherwise meshes/lights/cameras accumulate across runs.
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)


def body_attitude_from_sun(sun_lvlh):
    """Nadir-pointing body with yaw steering toward the sun.

    Matches the draft: RX = 90 deg tips the OBJ into LVLH, RZ = yaw - pi/2
    points the bus long axis appropriately. Returned per-frame because the
    sun's LVLH azimuth changes continuously in LEO.
    """
    s = np.asarray(sun_lvlh, dtype=float)
    s /= np.linalg.norm(s)
    yaw = float(np.arctan2(s[1], s[0]))
    euler = mu.Euler((np.radians(90.0), 0.0, yaw - np.pi / 2), 'XYZ')
    return euler.to_matrix(), yaw


def panel_drive_angle(sun_lvlh, body_mat_np, strategy, offset_deg):
    """Panel hinge angle for the chosen strategy.

    Sun is rotated into the (already yaw-steered) body frame; the nominal
    drive puts the panel normal onto the sun's projection in the body XZ
    plane (rotation about the Y hinge). 'edge_on' adds 90 deg so the panel
    presents its edge to the sun (knife-edge / shark-fin).
    """
    sun_body = body_mat_np.T @ np.asarray(sun_lvlh, dtype=float)
    ang = float(np.arctan2(sun_body[0], sun_body[2]))
    if strategy == 'edge_on':
        ang += np.pi / 2
    return ang + np.radians(offset_deg)


def apply_attitude(body_mat, panel_angle):
    """Apply body attitude to all meshes, panel drive to the panel(s)."""
    panel_set = set(solar_panel_names)
    panel_drive = mu.Matrix.Rotation(panel_angle, 3, PANEL_HINGE_AXIS)
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH':
            continue
        if obj.name in panel_set:
            obj.rotation_euler = (body_mat @ panel_drive).to_euler()
        else:
            obj.rotation_euler = body_mat.to_euler()
    bpy.context.view_layer.update()


# =====================================================================
# Single-pass runner
# =====================================================================
def run_pass(brdf_name, ATTITUDE_MODES, save_images=True):
    print(f"\n=== Run: brdf={brdf_name}, attitude={attitude_name} ===")
    mode = ATTITUDE_MODES[attitude_name]

    # Initial geometry at t_start for renderer construction
    geom = make_geometry(t_start)

    sun_lvlh_0 = geom.incident_vector_lvlh
    obs_lvlh_0 = geom.outgoing_vector_lvlh
    dist_obs_0 = float(geom.prop_distance) * 1e3

    # Fresh renderer per run so the BRDF assignment is rebuilt.
    # If meshes accumulate across runs, uncomment the reset:
    # reset_blender_scene()
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

    out = {k: [] for k in ('t', 'flux', 'mag', 'phase', 'panel', 'el', 'eq_phase')}
    images = []
    n_below, n_eclipsed = 0, 0
    el_seen = []

    for t_utc in tqdm(time_grid, desc=f"{brdf_name}/{attitude_name}"):
        geom = make_geometry(t_utc)

        # Elevation gate
        sat_pos = geom.positions['satellite']
        obs_pos = geom.positions['observer']
        diff    = sat_pos - obs_pos
        zenith  = obs_pos / np.linalg.norm(obs_pos)
        el_deg  = float(np.degrees(np.arcsin(np.clip(
                    np.dot(diff / np.linalg.norm(diff), zenith), -1, 1))))
        el_seen.append(el_deg)
        if el_deg < min_elevation:
            n_below += 1
            continue

        # Eclipse gate
        illum     = float(geom.eclipse_type())
        phase_deg = float(np.degrees(geom.phase_angle))
        if illum < 0.01:
            n_eclipsed += 1
            continue

        # Solar equatorial phase angle
        sat_pos_obs = sat_pos - obs_pos
        sun_pos     = geom.positions['sun']
        sat_ra      = float(np.degrees(np.arctan2(sat_pos_obs[1], sat_pos_obs[0])))
        sun_ra      = float(np.degrees(np.arctan2(sun_pos[1], sun_pos[0])))
        anti_sun_ra = (sun_ra + 180) % 360
        eq_phase    = ((sat_ra - anti_sun_ra + 180) % 360) - 180

        # LVLH vectors
        sun_lvlh = geom.incident_vector_lvlh
        obs_lvlh = geom.outgoing_vector_lvlh
        dist_obs = float(geom.prop_distance) * 1e3

        # Per-frame attitude (yaw steering) + panel drive
        body_mat, _ = body_attitude_from_sun(sun_lvlh)
        body_mat_np = np.array(body_mat)
        panel_angle = panel_drive_angle(sun_lvlh, body_mat_np,
                                        mode['panel_strategy'],
                                        mode['panel_offset_deg'])
        apply_attitude(body_mat, panel_angle)

        # Update renderer + render
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
        out['panel'].append(np.degrees(panel_angle))
        out['el'].append(el_deg)
        out['eq_phase'].append(eq_phase)
        if save_images:
            images.append(image)

    # seconds since t_start, for interpolation between runs
    out['sec'] = np.array([(t - t_start).total_seconds() for t in out['t']])
    for k in ('flux', 'mag', 'phase', 'panel', 'el', 'eq_phase'):
        out[k] = np.array(out[k])
    out['images'] = images

    if len(out['sec']) == 0:
        el_lo = min(el_seen) if el_seen else float('nan')
        el_hi = max(el_seen) if el_seen else float('nan')
        print(f"  WARNING: 0 frames rendered! "
              f"below_horizon={n_below}, eclipsed={n_eclipsed}, "
              f"elevation range seen: {el_lo:.1f} to {el_hi:.1f} deg "
              f"(min_elevation={min_elevation})")
    else:
        print(f"  rendered {len(out['sec'])} frames "
              f"(skipped: {n_below} below horizon, {n_eclipsed} eclipsed), "
              f"mag {np.nanmin(out['mag']):.2f} – {np.nanmax(out['mag']):.2f}")
    return out


# =====================================================================
# Execute all runs
# =====================================================================
results = {}
for brdf_name, attitude_name in RUNS:
    results[(brdf_name, attitude_name)] = run_pass(brdf_name, attitude_name)

# =====================================================================
# Comparison plots
# =====================================================================
ref_key = ('baseline', 'nominal')
ref     = results[ref_key]

populated = [k for k in RUNS if len(results[k]['sec']) > 0]
if not populated:
    raise RuntimeError(
        "No configuration produced any frames — check the diagnostic "
        "counters above (elevation gate vs eclipse gate) before plotting.")
RUNS_PLOT = populated
if ref_key not in populated:
    ref_key = populated[0]
    ref = results[ref_key]
    print(f"NOTE: baseline/nominal was empty — using {ref_key} as Δmag reference")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                               gridspec_kw={'height_ratios': [2, 1]})

colours = plt.cm.tab10(np.linspace(0, 1, len(RUNS_PLOT)))

for (key, col) in zip(RUNS_PLOT, colours):
    r = results[key]
    label = f"{key[0]} / {key[1]}"
    ax1.plot(r['sec'], r['mag'], '.-', ms=3, lw=0.8, color=col, label=label)
    if key != ref_key and len(r['sec']) > 1:
        # Δmag relative to baseline/nominal on the reference time base
        mag_interp = np.interp(ref['sec'], r['sec'], r['mag'],
                               left=np.nan, right=np.nan)
        ax2.plot(ref['sec'], mag_interp - ref['mag'], '-', lw=1.0,
                 color=col, label=label)

ax1.invert_yaxis()
ax1.set_ylabel('Apparent magnitude')
ax1.set_title(f'{ID} — attitude & BRDF comparison '
              f'({t_start:%Y-%m-%d %H:%M} UTC pass, Roque de los Muchachos)')
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

ax2.axhline(0, color='k', lw=0.5)
ax2.set_ylabel(r'$\Delta$mag vs baseline/nominal')
ax2.set_xlabel(f'Seconds since {t_start:%H:%M:%S} UTC')
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(output_dir, 'starlink_attitude_brdf_comparison.png')
plt.savefig(fig_path, dpi=150)
plt.show()
print(f"\nSaved comparison figure -> {fig_path}")

# =====================================================================
# Summary table + save results
# =====================================================================
print(f"\n{'config':<28}{'median mag':>12}{'peak mag':>10}{'mean dMag':>12}")
for key in RUNS_PLOT:
    r = results[key]
    if key == ref_key:
        dmag = 0.0
    else:
        mag_interp = np.interp(ref['sec'], r['sec'], r['mag'],
                               left=np.nan, right=np.nan)
        dmag = float(np.nanmean(mag_interp - ref['mag']))
    print(f"{key[0]+'/'+key[1]:<28}"
          f"{np.nanmedian(r['mag']):>12.2f}"
          f"{np.nanmin(r['mag']):>10.2f}"
          f"{dmag:>12.2f}")

npz_path = os.path.join(output_dir, 'starlink_attitude_brdf_results.npz')
np.savez(npz_path, **{
    f"{b}_{a}_{field}": results[(b, a)][field]
    for (b, a) in RUNS
    for field in ('sec', 'flux', 'mag', 'phase', 'panel', 'el', 'eq_phase')
})
print(f"Saved arrays -> {npz_path}")