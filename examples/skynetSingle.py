import sys
import numpy as np
import datetime
import matplotlib.pyplot as plt

from mathutils import Vector
from satlight.geometry import Geometry
from satlight.raytracer import Renderer

# Config

obj_path        = r'/Users/l.beesley@bham.ac.uk/Documents/Lightcurves/satlight/models/Skynet 5D/skynet.obj'

# Observer: Roque de los Muchachos, La Palma
observer_lat    =  28.7622
observer_lon    = -17.8897
observer_alt_m  =  2396.0

# GEO: circular equatorial at 28 W
geo_longitude   = -28.0
geo_altitude_km =  35786.0
R_EARTH_KM      =  6371.0
a_km            =  R_EARTH_KM + geo_altitude_km

# Solar panel names
solar_panel_names = ['solarPanel1', 'solarPanel2']

# Single night: 2025-01-07 21:00 UTC, 8 hours, 10-min cadence
t_start        = datetime.datetime(2025, 1, 7, 21, 0, 0)
obs_duration_h = 8
cadence_min    = 10
min_elevation  = 5.0

resolution     = (100, 100)
solar_constant = 1361.0
mag_zeropoint  = 25.0

# =============================================================================
# Time grid for one night
# =============================================================================

n_steps   = int(obs_duration_h * 60 / cadence_min)
time_grid = [t_start + datetime.timedelta(minutes=i * cadence_min)
             for i in range(n_steps)]
print(f"Timesteps: {len(time_grid)}  ({t_start} to {time_grid[-1]})")

# =============================================================================
# Initialise Blender scene with first timestep geometry
# =============================================================================

def make_geometry(t_utc):
    geom = Geometry()
    geom.create_observer(observer_lat, observer_lon, observer_alt_m)
    geom.create_satellite_from_elements(
        a_km=a_km, e=0.0, i_deg=0.0, raan_deg=0.0, argp_deg=0.0,
        nu_deg=geo_longitude, epoch=t_utc,
    )
    geom.set_time((t_utc.year, t_utc.month, t_utc.day,
                   t_utc.hour, t_utc.minute, t_utc.second))
    return geom

geom0      = make_geometry(t_start)
sun_lvlh_0 = -geom0.incident_vector_lvlh
obs_lvlh_0 =  geom0.outgoing_vector_lvlh
dist_obs_0 = float(geom0.prop_distance) * 1e3

renderer = Renderer(
    obj_path             = obj_path,
    sun_direction        = Vector(sun_lvlh_0.tolist()),
    observer_direction   = Vector(obs_lvlh_0.tolist()),
    distance_to_observer = dist_obs_0,
    resolution           = resolution,
    solar_constant       = solar_constant,
    add_earthshine       = False,
)
renderer.initialise_scene()
print("Scene initialised.\n")

# =============================================================================
# Main loop — single night
# =============================================================================

import bpy

times_out    = []
fluxes       = []
magnitudes   = []
phase_angles = []
panel_angles = []
elevations   = []

for t_utc in time_grid:

    geom = make_geometry(t_utc)

    # Elevation
    sat_pos = geom.positions['satellite']
    obs_pos = geom.positions['observer']
    diff    = sat_pos - obs_pos
    zenith  = obs_pos / np.linalg.norm(obs_pos)
    el_deg  = float(np.degrees(np.arcsin(np.clip(
                np.dot(diff / np.linalg.norm(diff), zenith), -1, 1))))

    if el_deg < min_elevation:
        print(f"  {t_utc.strftime('%H:%M')}  BELOW HORIZON ({el_deg:.1f} deg)")
        continue

    # Eclipse
    illum     = float(geom.eclipse_type())
    phase_deg = float(np.degrees(geom.phase_angle))

    if illum < 0.01:
        print(f"  {t_utc.strftime('%H:%M')}  ECLIPSED")
        continue

    # LVLH directions
    sun_lvlh = -geom.incident_vector_lvlh
    obs_lvlh =  geom.outgoing_vector_lvlh
    dist_obs = float(geom.prop_distance) * 1e3

    # Solar panel rotation about Y: rotate panel normal (+Z) to face sun
    # Project sun onto XZ plane, signed angle from +Z via atan2
    sun_xz      = np.array([sun_lvlh[0], 0.0, sun_lvlh[2]])
    sun_xz_norm = np.linalg.norm(sun_xz)
    if sun_xz_norm > 1e-6:
        sun_xz /= sun_xz_norm
    panel_angle = float(np.arctan2(sun_xz[0], sun_xz[2]))

    for name in solar_panel_names:
        obj = bpy.data.objects.get(name)
        if obj is not None:
            obj.rotation_euler[1] = panel_angle
        else:
            print(f"  Warning: '{name}' not found in scene")

    # Update renderer directions
    renderer.sun_direction        = Vector(sun_lvlh.tolist())
    renderer.observer_direction   = Vector(obs_lvlh.tolist())
    renderer.distance_to_observer = dist_obs

    # Update camera — use bpy mathutils directly to avoid type clash
    import mathutils as mu
    cam      = renderer.camera_object
    obs_vec  = mu.Vector(obs_lvlh.tolist()).normalized()
    cam.location       = obs_vec * renderer.ortho_scale
    direction          = mu.Vector((0, 0, 0)) - cam.location
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    bpy.context.view_layer.update()

    # Render
    image = renderer.render()
    flux  = float(np.sum(image)) * illum
    mag   = -2.5 * np.log10(flux) + mag_zeropoint if flux > 0 else np.nan

    print(f"  {t_utc.strftime('%H:%M')}  el={el_deg:.1f}  phase={phase_deg:.1f}  "
          f"panel={np.degrees(panel_angle):.1f} deg  flux={flux:.3e} W  mag={mag:.2f}")

    times_out.append(t_utc)
    fluxes.append(flux)
    magnitudes.append(mag)
    phase_angles.append(phase_deg)
    panel_angles.append(np.degrees(panel_angle))
    elevations.append(el_deg)

# =============================================================================
# Plot
# =============================================================================

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

axes[0].plot(times_out, magnitudes, 'k.-')
axes[0].invert_yaxis()
axes[0].set_ylabel('Magnitude')
axes[0].set_title('Skynet 5D — single night lightcurve (28°W GEO, Roque de los Muchachos)')

axes[1].plot(times_out, phase_angles, 'b.-')
axes[1].set_ylabel('Phase angle (deg)')

axes[2].plot(times_out, panel_angles, 'darkorange', marker='.')
axes[2].set_ylabel('Panel rotation (deg)')
axes[2].set_xlabel('Time (UTC)')

plt.tight_layout()
plt.savefig('/Users/l.beesley@bham.ac.uk/Documents/Lightcurves/satlight/output/single_night_test.png', dpi=150)
plt.show()

print(f"\nRendered {len(fluxes)} timesteps")
print(f"Magnitude range: {np.nanmin(magnitudes):.2f} – {np.nanmax(magnitudes):.2f}")
print(f"Phase range:     {min(phase_angles):.1f}° – {max(phase_angles):.1f}°")
print(f"Panel range:     {min(panel_angles):.1f}° – {max(panel_angles):.1f}°")