"""Light curve of the geostationary satellite Intelsat 1R over one night.

Needs satlight (pip install --pre satlight) and Intelsat 1R.obj. Takes ~5 min.
The body points its antennas at the Earth; the solar panels turn to face the
Sun, each offset by PANEL_OFFSET. A panel offset by d glints at phase ~2d.
"""
import datetime

import bpy, mathutils                        # Blender, installed with satlight
import matplotlib.pyplot as plt
import numpy as np

from satlight import Geometry, Renderer
from satlight.brdf import abg

MODEL = '/path/to/Intelsat 1R.obj'           # change this to your path to the .obj

# ---- settings
SITE = (28.7622, -17.8897, 2396)             # La Palma: lat (deg N), lon (deg E), height (m)
GEO_LON = -28.0                              # longitude the satellite sits above (deg E)
START = datetime.datetime(2025, 3, 5, 20, 0, tzinfo=datetime.timezone.utc)
HOURS, STEP_S = 10, 60                       # length of the night, seconds between points
PANEL_OFFSET = {'solarPanel1': 11.5, 'solarPanel2': 18.5}   # deg
RESOLUTION = (800, 800)                      # smaller is faster but gives false jumps

# Materials, matched by part name. 'abg' is satlight's shiny (mirror-like) model:
# A/B is the brightness of the glint, g how fast it fades away from it.
MATERIALS = {
    'solarpanel':   (abg, {'A': 1.0e-4, 'B': 1.3e-5, 'g': 3.0}),
    'antennahinge': ('lambertian', {'albedo': 0.2}),
    'antenna':      (abg, {'A': 1.9e-5, 'B': 2.4e-7, 'g': 4.5}),
    'default':      ('lambertian', {'albedo': 0.3}),   # body, thruster, cameras
}

# ---- the orbit: GEO, placed above GEO_LON at START
geom = Geometry()
geom.create_observer(*SITE)
sidereal_deg = 15.0 * geom.ts.from_datetime(START).gast
geom.create_satellite_from_elements(a_km=42164.0, e=0.0, i_deg=0.0,
                                    raan_deg=(GEO_LON + sidereal_deg) % 360.0,
                                    argp_deg=0.0, nu_deg=0.0, epoch=START)

# ---- when is it observable? (above the horizon AND in sunlight), all times at once
n = int(HOURS * 3600 / STEP_S) + 1 
geom.set_time(START.year, START.month, START.day, START.hour, START.minute,
              STEP_S * np.arange(n))
times = [t.utc_datetime() for t in geom.time]
observable = np.asarray(geom.observable_fraction())  # 1 seen, 0 hidden or in shadow
print(f'{np.sum(observable > 0)} of {n} times observable')

# ---- load the model
geom.set_time(times[int(np.argmax(observable > 0))])  # the Renderer takes one time
r = Renderer(MODEL, geometry=geom, resolution=RESOLUTION, brdf=MATERIALS,
             smooth_normals=True)
r.initialise_scene()

# Turn the model so its antennas face the Earth, in the LVLH frame (X along the
# orbit, Y north, Z up). Keep each part's starting pose so moves don't add up.
body = mathutils.Euler(np.radians((270, 180, 90))).to_matrix().to_4x4() #Had to find this through trial and error.
rest = {o.name: o.matrix_world.copy() for o in bpy.data.objects if o.type == 'MESH'}
for name, M in rest.items():
    bpy.data.objects[name].matrix_world = body @ M
R = np.array(body.to_3x3())
north = R.T @ np.array([0.0, 1.0, 0.0])      # north, in the model's own frame

panels = {}
for name in PANEL_OFFSET:                    # each panel's centre and face direction
    pts = np.array([rest[name] @ v.co for v in bpy.data.objects[name].data.vertices])
    panels[name] = (pts.mean(0), np.linalg.svd(pts - pts.mean(0))[2][2])


def point_panels(sun_lvlh):
    """Turn each panel about north to face the Sun, plus its offset."""
    sun = R.T @ np.asarray(sun_lvlh)         # the Sun in the model's frame
    for name, (centre, face) in panels.items():
        angle = np.arctan2(sun @ np.cross(north, face), sun @ face)
        angle += np.radians(PANEL_OFFSET[name])
        c = mathutils.Vector(centre)
        turn = (mathutils.Matrix.Translation(c)
                @ mathutils.Matrix.Rotation(angle, 4, mathutils.Vector(north))
                @ mathutils.Matrix.Translation(-c))
        bpy.data.objects[name].matrix_world = body @ turn @ rest[name]


# ---- render every observable time
mag = np.full(n, np.nan)
for i in np.flatnonzero(observable > 0):
    geom.set_time(times[i])
    point_panels(geom.incident_vector_lvlh)  # 1. move the panels
    r.update(geom)                           # 2. Sun, observer and camera
    flux = float(np.sum(r.render()))         # 3. ray-trace: W/m^2 at the telescope
    # The Sun is magnitude -26.75 at 1361 W/m^2, so m = -2.5 log10(flux) - 18.915
    mag[i] = -2.5 * np.log10(flux) - 18.915 if flux > 0 else np.nan

# ---- plot
plt.plot([(t - START).total_seconds() / 3600 for t in times], mag, 'k-', lw=0.8)
plt.gca().invert_yaxis()                     # brighter is up
plt.xlabel(f'Hours since {START:%Y-%m-%d %H:%M} UTC')
plt.ylabel('Magnitude')
plt.savefig('geo_lightcurve.png', dpi=200, bbox_inches='tight')
