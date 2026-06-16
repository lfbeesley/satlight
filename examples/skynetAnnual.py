import sys
import bpy
import numpy as np
import datetime
import matplotlib.pyplot as plt
from skyfield.api import Loader
import os
from mathutils import Vector
from satlight.geometry import Geometry
from satlight.raytracer import Renderer
from tqdm import tqdm
import csv 

# Config
obj_path = r'/Users/l.beesley@bham.ac.uk/Documents/Lightcurves/satlight/models/Skynet 5D/skynet.obj'
output_csv = '/Users/l.beesley@bham.ac.uk/Documents/Lightcurves/satlight/output/annual_lightcurve.csv'
output_npy = '/Users/l.beesley@bham.ac.uk/Documents/Lightcurves/satlight/output/annual_lightcurve.npy'

# Observer: Roque de los Muchachos, La Palma
observer_lat    =  28.7622
observer_lon    = -17.8897
observer_alt_m  =  2396.0

# GEO: circular equatorial at 28 W
geo_longitude   = -28.0
geo_altitude_km =  35786.0
R_EARTH_KM      =  6371.0
a_km            =  R_EARTH_KM + geo_altitude_km

# deriving RAAN for GEO sat
_data_dir = os.path.join(os.path.expanduser('~'), '.satlight')
load = Loader(_data_dir)
ts = load.timescale()

def gmst_deg(t_utc):
    """GMST in degrees at a given UTC datetime."""
    t = ts.utc(t_utc.year, t_utc.month, t_utc.day,
               t_utc.hour, t_utc.minute, t_utc.second)
    return t.gast * 15.0  # gast is in hours, *15 -> degrees

def geo_raan(target_lon_deg, t_utc):
    """RAAN needed to place a GEO satellite at target geographic longitude."""
    return (target_lon_deg + gmst_deg(t_utc)) % 360.0

# Solar panel names
solar_panel_names = ['solarPanel1', 'solarPanel2']

# Observation schedule: 52 weeks, 8 hours per night, 10-min cadence
start_date       = datetime.datetime(2025, 1, 7, 21, 0, 0)  
n_weeks          = 52
obs_duration_h   = 8
cadence_min      = 10
min_elevation    = 5.0        # degrees, below this we skip

# Renderer settings
resolution       = (300, 300)
solar_constant   = 1361.0     # W/m^2
add_earthshine   = False

mag_zeropoint    = -18

# =============================================================================
# Build time grid: 52 weeks x 8 hours x 10-min cadence
# =============================================================================

cadence     = datetime.timedelta(minutes=cadence_min)
n_steps     = int(obs_duration_h * 60 / cadence_min)   # 48 steps per night
time_grid   = []

for week in range(n_weeks):
    t0 = start_date + datetime.timedelta(weeks=week)
    for step in range(n_steps):
        time_grid.append((week, step, t0 + step * cadence))

print(f"Total timesteps: {len(time_grid)}")

# Set geometry

t0_dt = time_grid[0][2]

geom0 = Geometry()
geom0.create_observer(observer_lat, observer_lon, observer_alt_m)
geom0.create_satellite_from_elements(
    a_km=a_km, e=0.0, i_deg=0.0,
    raan_deg=geo_raan(geo_longitude, t_utc),
    argp_deg=0.0, nu_deg=0.0, epoch=t_utc,)

geom0.set_time((t0_dt.year, t0_dt.month, t0_dt.day,
                t0_dt.hour, t0_dt.minute, t0_dt.second))

# Geometry.incident_vector_lvlh is sun->sat; negate for sat->sun (toward sun)
sun_lvlh_0 = -geom0.incident_vector_lvlh
obs_lvlh_0 =  geom0.outgoing_vector_lvlh    # sat->observer
dist_obs_0 = float(geom0.prop_distance) * 1e3   # m

# Initialise Blender scene

renderer = Renderer(
    obj_path             = obj_path,
    sun_direction        = Vector(sun_lvlh_0.tolist()),
    observer_direction   = Vector(obs_lvlh_0.tolist()),
    distance_to_observer = dist_obs_0,
    resolution           = resolution,
    solar_constant       = solar_constant,
    add_earthshine       = add_earthshine,
)
renderer.initialise_scene()
print("Blender scene initialised.\n")

# Iterate over year

records    = []
n_rendered = 0
n_eclipse  = 0
n_horizon  = 0

for week_idx, step_idx, t_utc in tqdm(time_grid, desc='Lightcurve'):

    # -- Geometry for this timestep ------------------------------------------
    geom = Geometry()
    geom.create_observer(observer_lat, observer_lon, observer_alt_m)
    geom.create_satellite_from_elements(
        a_km=a_km, e=0.0, i_deg=0.0, raan_deg=0.0, argp_deg=0.0,
        nu_deg=geo_longitude, epoch=t_utc,
    )
    geom.set_time((t_utc.year, t_utc.month, t_utc.day,
                   t_utc.hour, t_utc.minute, t_utc.second))

    # -- Elevation check -----------------------------------------------------
    sat_pos  = geom.positions['satellite']
    obs_pos  = geom.positions['observer']
    diff     = sat_pos - obs_pos
    zenith   = obs_pos / np.linalg.norm(obs_pos)
    el_deg   = float(np.degrees(np.arcsin(np.clip(
                    np.dot(diff / np.linalg.norm(diff), zenith), -1, 1))))

    if el_deg < min_elevation:
        n_horizon += 1
        records.append({
            'datetime_utc': t_utc.isoformat(), 'week': week_idx, 'step': step_idx,
            'flux_W': np.nan, 'magnitude': np.nan, 'phase_angle_deg': np.nan,
            'elevation_deg': el_deg, 'illumination_fraction': np.nan,
            'sun_angle_panel_deg': np.nan, 'eclipsed': False, 'below_horizon': True,
        })
        continue

    # -- Eclipse check -------------------------------------------------------
    illum     = float(geom.eclipse_type())
    phase_deg = float(np.degrees(geom.phase_angle))

    if illum < 0.01:
        n_eclipse += 1
        records.append({
            'datetime_utc': t_utc.isoformat(), 'week': week_idx, 'step': step_idx,
            'flux_W': np.nan, 'magnitude': np.nan, 'phase_angle_deg': phase_deg,
            'elevation_deg': el_deg, 'illumination_fraction': 0.0,
            'sun_angle_panel_deg': np.nan, 'eclipsed': True, 'below_horizon': False,
        })
        continue

    # -- Directions in LVLH --------------------------------------------------
    # Geometry convention: incident_vector_lvlh = sun->sat, so negate for sat->sun
    sun_lvlh = -geom.incident_vector_lvlh   # toward sun
    obs_lvlh =  geom.outgoing_vector_lvlh   # toward observer
    dist_obs = float(geom.prop_distance) * 1e3   # m

    # -- Solar panel rotation ------------------------------------------------
    # Rotate about Y axis (orbit normal / cross-track) so panel normal (+Z, zenith)
    # tracks the sun. Project sun onto XZ plane, find signed angle from +Z.
    sun_xz   = np.array([sun_lvlh[0], 0.0, sun_lvlh[2]])
    sun_xz  /= np.linalg.norm(sun_xz) if np.linalg.norm(sun_xz) > 1e-6 else 1.0
    panel_angle = float(np.arctan2(sun_xz[0], sun_xz[2]))   # radians, about Y

    import bpy
    for name in solar_panel_names:
        obj = bpy.data.objects.get(name)
        if obj is not None:
            obj.rotation_euler[1] = panel_angle

    # -- Update Blender camera and sun direction -----------------------------
    renderer.sun_direction        = Vector(sun_lvlh.tolist())
    renderer.observer_direction   = Vector(obs_lvlh.tolist())
    renderer.distance_to_observer = dist_obs

    cam       = renderer.camera_object
    cam.location       = Vector(obs_lvlh.tolist()).normalized() * renderer.ortho_scale
    direction          = Vector((0, 0, 0)) - cam.location
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    bpy.context.view_layer.update()

    # -- Render --------------------------------------------------------------
    image = renderer.render()
    flux  = float(np.sum(image)) * illum   # scale by partial illumination fraction
    mag   = -2.5 * np.log10(flux) + mag_zeropoint if flux > 0 else np.nan
    n_rendered += 1

    records.append({
        'datetime_utc':          t_utc.isoformat(),
        'week':                  week_idx,
        'step':                  step_idx,
        'flux_W':                flux,
        'magnitude':             mag,
        'phase_angle_deg':       phase_deg,
        'elevation_deg':         el_deg,
        'illumination_fraction': illum,
        'sun_angle_panel_deg':   float(np.degrees(panel_angle)),
        'eclipsed':              False,
        'below_horizon':         False,
    })

print(f"\nRendered: {n_rendered}  |  Eclipsed: {n_eclipse}  |  Below horizon: {n_horizon}")

# =============================================================================
# Save outputs
# =============================================================================

os.makedirs(os.path.dirname(output_csv), exist_ok=True)

fieldnames = ['datetime_utc', 'week', 'step', 'flux_W', 'magnitude',
              'phase_angle_deg', 'elevation_deg', 'illumination_fraction',
              'sun_angle_panel_deg', 'eclipsed', 'below_horizon']

with open(output_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)
print(f"CSV  -> {output_csv}")

valid = [r for r in records if not r['below_horizon'] and not r['eclipsed']]
arr = np.array(
    [(r['flux_W'], r['magnitude'], r['phase_angle_deg'],
      r['elevation_deg'], r['illumination_fraction'], r['sun_angle_panel_deg'])
     for r in valid],
    dtype=[('flux_W', 'f8'), ('magnitude', 'f8'), ('phase_angle_deg', 'f8'),
           ('elevation_deg', 'f8'), ('illumination_fraction', 'f8'),
           ('sun_angle_panel_deg', 'f8')],
)
np.save(output_npy, arr)
print(f"NPY  -> {output_npy}")

# =============================================================================
# Summary plot
# =============================================================================

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)

# Magnitude over the year
datetimes = [datetime.datetime.fromisoformat(r['datetime_utc']) for r in valid]
mags      = [r['magnitude'] for r in valid]
axes[0].scatter(datetimes, mags, s=1, c='k', alpha=0.4)
axes[0].invert_yaxis()
axes[0].set_ylabel('Magnitude')
axes[0].set_title('Annual lightcurve — Skynet 5D (GEO, 28°W) from Roque de los Muchachos')

# Phase angle over the year
phases = [r['phase_angle_deg'] for r in valid]
axes[1].scatter(datetimes, phases, s=1, c='steelblue', alpha=0.4)
axes[1].set_ylabel('Phase angle (deg)')

# Solar panel angle over the year
panel_angles = [r['sun_angle_panel_deg'] for r in valid]
axes[2].scatter(datetimes, panel_angles, s=1, c='darkorange', alpha=0.4)
axes[2].set_ylabel('Panel rotation (deg)')
axes[2].set_xlabel('Date (UTC)')

plt.tight_layout()
plt.savefig(output_csv.replace('.csv', '_summary.png'), dpi=150)
plt.show()

print(f"\nMagnitude range  : {np.nanmin(mags):.2f} – {np.nanmax(mags):.2f}")
print(f"Phase angle range: {min(phases):.1f}° – {max(phases):.1f}°")