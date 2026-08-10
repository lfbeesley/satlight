import sys
import bpy
import numpy as np
import datetime
import matplotlib.pyplot as plt
from skyfield.api import Loader
import os
from mathutils import Vector
from satlight.geometry import Geometry
from tqdm import tqdm
from satlight.raytracer import Renderer
import mathutils as mu
import trimesh
import multiprocessing as mp

# Config
obj_path = r'/Users/l.beesley@bham.ac.uk/Documents/Lightcurves/satlight/models/Skynet 5D/skynet.obj'

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

# Single night: 2025-01-07 21:00 UTC, 8 hours, 10-min cadence
t_start        = datetime.datetime(2025, 1, 7, 20, 0, 0)
obs_duration_h = 12
cadence_min    = 2 # integrated over range od angles
min_elevation  = 5.0

resolution     = (1024, 256)
solar_constant = 1361.0
mag_zeropoint  = -18.0

# Time grid
n_steps   = int(obs_duration_h * 60 / cadence_min)
time_grid = [t_start + datetime.timedelta(minutes=i * cadence_min)
             for i in range(n_steps)]
print(f"Timesteps: {len(time_grid)}  ({t_start} to {time_grid[-1]})")

# Initialise scene
def make_geometry(t_utc):
    geom = Geometry()
    geom.create_observer(observer_lat, observer_lon, observer_alt_m)
    geom.create_satellite_from_elements(
        a_km=a_km, e=0.0, i_deg=0.0,
        raan_deg=geo_raan(geo_longitude, t_utc), 
        argp_deg=0.0,
        nu_deg=0.0,                            
        epoch=t_utc,)
    
    geom.set_time((t_utc.year, t_utc.month, t_utc.day,
                   t_utc.hour, t_utc.minute, t_utc.second))
    return geom

geom0      = make_geometry(t_start)
sun_lvlh_0 = geom0.incident_vector_lvlh
obs_lvlh_0 = geom0.outgoing_vector_lvlh
dist_obs_0 = float(geom0.prop_distance) * 1e3

renderer = Renderer(
    obj_path             = obj_path,
    sun_direction        = Vector(sun_lvlh_0.tolist()),
    observer_direction   = Vector(obs_lvlh_0.tolist()),
    distance_to_observer = dist_obs_0,
    resolution           = resolution,
    solar_constant       = solar_constant,
    add_earthshine       = False,
    brdf = {
        'solarPanel':      ('cook_torrance',   {'albedo': 0.85, 'roughness': 0.15, 'metallic': 0.95}),
        'Antenna':         ('cook_torrance', {'albedo': 0.85, 'roughness': 0.15, 'metallic': 0.95}),  # high-gain dish, polished metal mesh
        'omniAntenna':     ('cook_torrance', {'albedo': 0.7,  'roughness': 0.3,  'metallic': 0.9}),   # smaller, less polished
        'bus(Aluminium)':  ('cook_torrance', {'albedo': 0.6,  'roughness': 0.25, 'metallic': 0.9}),   # bare/anodized aluminium structure
        'bus(MLI)':        ('oren_nayar',    {'albedo': 0.25, 'roughness': 0.5}),                      # multilayer insulation, diffuse gold/silver foil
        'default':         ('lambertian',    {'albedo': 0.3}),
    },
)

renderer.initialise_scene()
print("Scene initialised.\n")

# Axis mapping check
raw_check = trimesh.load(obj_path, force=None, process=False)
geometries_check = raw_check.geometry if hasattr(raw_check, 'geometry') else {'model': raw_check}

for obj in bpy.context.scene.objects:
    if obj.name in solar_panel_names:
        corners = [Vector(c) for c in obj.bound_box]
        xs = [c.x for c in corners]
        ys = [c.y for c in corners]
        zs = [c.z for c in corners]
        print(f"{obj.name} Blender: X={min(xs):.2f}→{max(xs):.2f}  Y={min(ys):.2f}→{max(ys):.2f}  Z={min(zs):.2f}→{max(zs):.2f}")

for name, mesh in geometries_check.items():
    if any(p.lower() in name.lower() for p in solar_panel_names):
        v = mesh.vertices
        print(f"{name} Trimesh: X={v[:,0].min():.2f}→{v[:,0].max():.2f}  Y={v[:,1].min():.2f}→{v[:,1].max():.2f}  Z={v[:,2].min():.2f}→{v[:,2].max():.2f}")

# Adjust body attitude
ATTITUDE_RX = np.radians(0)
ATTITUDE_RY = np.radians(270)
ATTITUDE_RZ = np.radians(90)
renderer.preview_lvlh_views((ATTITUDE_RX, ATTITUDE_RY, ATTITUDE_RZ))

body_euler = mu.Euler((ATTITUDE_RX, ATTITUDE_RY, ATTITUDE_RZ), 'XYZ')
body_mat   = body_euler.to_matrix()
body_mat_np = np.array(body_mat) 

# Panel geometry in raw, unrotated OBJ body frame
panel_normal_body = np.array([0, -1, 0])   # -Y is the face that points at sun
panel_hinge_body  = np.array([0, 0, 1])   # +Z is the rotation axis

panel_offsets_deg = {
    'solarPanel1': +7.0,
    'solarPanel2': -7.0,
}

# Apply body attitude to every non-panel mesh (done once here, not per-frame).
_panel_set = set(solar_panel_names)
for _obj in bpy.context.scene.objects:
    if _obj.type == 'MESH' and _obj.name not in _panel_set:
        _obj.rotation_euler = body_mat.to_euler()
bpy.context.view_layer.update()

# Main loop — single night
images = []
times_out    = []
fluxes       = []
magnitudes   = []
phase_angles = []
panel_angles = []
elevations   = []
eq_phase_angles = []
count = 0

for t_utc in tqdm(time_grid):

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

    # Solar equatorial phase angle
    sat_pos_obs = geom.positions['satellite'] - geom.positions['observer']
    sun_pos     = geom.positions['sun']
    sat_ra      = float(np.degrees(np.arctan2(sat_pos_obs[1], sat_pos_obs[0])))
    sun_ra      = float(np.degrees(np.arctan2(sun_pos[1], sun_pos[0])))
    anti_sun_ra = (sun_ra + 180) % 360
    eq_phase    = ((sat_ra - anti_sun_ra + 180) % 360) - 180

    # LVLH directions
    sun_lvlh = geom.incident_vector_lvlh
    obs_lvlh =  geom.outgoing_vector_lvlh
    dist_obs = float(geom.prop_distance) * 1e3

    # Transform sun into body frame
    sun_body = body_mat_np.T @ sun_lvlh

    # Panel normal is body +Y, hinge is body +Z
    # Project sun onto body XY plane, find angle from +Y
    sun_xy = np.array([sun_body[0], sun_body[1], 0.0])
    sun_xy /= np.linalg.norm(sun_xy)

    # angle to rotate +Y onto the sun direction, about body Z
    panel_angle = float(np.arctan2(-sun_xy[0], sun_xy[1]))

    # Apply to panels
    for name in solar_panel_names:
        obj = bpy.data.objects.get(name)
        if obj is not None:
            offset_rad  = np.radians(panel_offsets_deg.get(name, 0.0))
            panel_drive = mu.Matrix.Rotation(panel_angle + offset_rad, 3, 'Z')
            obj.rotation_euler = (body_mat @ panel_drive).to_euler()

    if count == 0:
        panel_rot = np.array(mu.Matrix.Rotation(panel_angle, 3, 'Z'))
        panel_normal_body = panel_rot @ np.array([-1, 0, 0])  # -X is normal
        panel_normal_lvlh = body_mat_np @ panel_normal_body

    count+=1

    # Update renderer directions
    renderer.sun_direction        = Vector(sun_lvlh.tolist())
    renderer.observer_direction   = Vector(obs_lvlh.tolist())
    renderer.distance_to_observer = dist_obs

    # Update camera
    renderer.update(
    sun_direction        = Vector(sun_lvlh.tolist()),
    observer_direction   = Vector(obs_lvlh.tolist()),
    distance_to_observer = dist_obs,)
    
    bpy.context.view_layer.update()
    
    # Render
    image = renderer.render()
    flux  = float(np.sum(image)) * illum
    mag   = -2.5 * np.log10(flux) + mag_zeropoint if flux > 0 else np.nan

    
    #print(f"  {t_utc.strftime('%H:%M')}  el={el_deg:.1f}  phase={phase_deg:.1f}  "
    #      f"panel={np.degrees(panel_angle):.1f} deg  flux={flux:.3e} W  mag={mag:.2f}")

    times_out.append(t_utc)
    fluxes.append(flux)
    magnitudes.append(mag)
    phase_angles.append(phase_deg)
    panel_angles.append(np.degrees(panel_angle))
    elevations.append(el_deg)
    eq_phase_angles.append(eq_phase)
    images.append(image)

# Plot
fig, axes = plt.subplots(1, 1, figsize=(12, 4), sharex=True)

axes.plot(times_out, magnitudes, 'k.-')
axes.invert_yaxis()
axes.set_ylabel('Magnitude')
axes.set_title('Skynet 5D — single night lightcurve (28°W GEO)')

plt.tight_layout()
plt.savefig('/Users/l.beesley@bham.ac.uk/Documents/Lightcurves/satlight/output/single_night_test.png', dpi=150)
plt.show()

print(f"\nRendered {len(fluxes)} timesteps")
print(f"Magnitude range: {np.nanmin(magnitudes):.2f} – {np.nanmax(magnitudes):.2f}")
print(f"Phase range:     {min(phase_angles):.1f}° – {max(phase_angles):.1f}°")
print(f"Panel range:     {min(panel_angles):.1f}° – {max(panel_angles):.1f}°")

# Save figuew
import matplotlib.animation as animation

output_path = '/Users/l.beesley@bham.ac.uk/Documents/Lightcurves/satlight/output/Skynet_14deg_solartilt.mp4'
fps = 10

# Renders are usually raw flux, not 0-255 — fix vmin/vmax across ALL frames
# so brightness doesn't flicker frame to frame (each imshow call would
# otherwise auto-scale to that frame's own min/max).
stack = np.array(images)
vmin, vmax = np.percentile(stack, [1, 99])  # robust to a few hot pixels

fig, ax = plt.subplots(figsize=(3, 12))
im = ax.imshow(images[0].T, cmap='viridis', vmin=vmin, vmax=vmax)
ax.axis('off')
title = ax.set_title('')
plt.tight_layout()

def update(frame_idx):
    im.set_data(images[frame_idx].T)
    title.set_text(f'Frame {frame_idx}')
    return im, title

ani = animation.FuncAnimation(fig, update, frames=len(images), blit=False)
ani.save(output_path, writer=animation.FFMpegWriter(fps=fps, bitrate=1800))
plt.close(fig)

print(f"Saved {len(images)} frames -> {output_path}")