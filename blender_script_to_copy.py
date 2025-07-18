# this code should be copied and pasted into Blender's scripting window (theres probably a way to load it directly but cba figuring that out rn)
# dont worry about bpy and mathutils not being resolved, they will be in Blender!

import bpy
import math
import numpy as np
import json
from mathutils import Vector, Matrix

# === SETTINGS ===

model_name = "F9_Upperstage"
json_path = "C:/Users/gxj236/Documents/Blender Lightcurve modelling/GS_contacts/contacts.json"

# === LOAD JSON ===
with open(json_path, "r") as f:
    contacts = json.load(f)

# Use specific contact
contact = contacts[0]

frames = len(contact['timearr'])  # Number of frames (1 frame = 1 second)

# Load Cartesian position arrays (N x 3)
gs_positions = np.array(contact['gs_positions'])       # Ground station positions in km
sun_positions = np.array(contact['sun_positions'])     # Sun positions (scale arbitrary but consistent)

# === OBJECTS ===
camera = bpy.data.objects['Camera']
sun = bpy.data.objects['Light']
model = bpy.data.objects[model_name]

sun.data.type = 'SUN'
sun.data.energy = 1367  # Solar constant in W/m^2
sun.data.color = (1, 1, 1)
sun.data.angle = 0.00872665

sun_scale_distance = 1/1e8
camera_scale_distance = 1/1e3

model.location = Vector((0.0, 0.0, 0.0))

# === ROTATION SETTINGS ===
rotation_axis = Vector((1, 0, 0))
rotation_axis.normalize()

rot_period = 150  # seconds
rotation_rate_rad_per_frame = math.radians(360 / rot_period)

# === SCENE SETUP ===
bpy.context.scene.frame_end = frames

def rotation_matrix_around_axis(axis, angle):
    """Return a rotation matrix rotating by 'angle' radians around 'axis' (must be normalized)."""
    return Matrix.Rotation(angle, 4, axis)

# === ANIMATION LOOP ===
for frame in range(frames):
    bpy.context.scene.frame_set(frame + 1)

    # === CAMERA POSITION ===
    cam_vec = Vector(gs_positions[frame])
    camera.location = cam_vec * camera_scale_distance

    # === ORIENT CAMERA TOWARD ORIGIN ===
    direction = -camera.location.normalized()
    up = Vector((0, 0, 1))
    right = direction.cross(up).normalized()
    up = right.cross(direction).normalized()
    cam_rot_matrix = Matrix((right, up, direction)).transposed()
    camera.matrix_world = Matrix.Translation(camera.location) @ cam_rot_matrix.to_4x4()

    # === SATELLITE ROTATION ===
    angle = frame * rotation_rate_rad_per_frame
    rot_matrix = rotation_matrix_around_axis(rotation_axis, angle)
    model.matrix_world = Matrix.Translation(model.location) @ rot_matrix

    # === SUN POSITION ===
    sun_vec = Vector(sun_positions[frame])
    sun.location = sun_vec * sun_scale_distance

    # === KEYFRAMES ===
    camera.keyframe_insert(data_path="location", frame=frame + 1)
    camera.keyframe_insert(data_path="rotation_euler", frame=frame + 1)
    model.keyframe_insert(data_path="rotation_euler", frame=frame + 1)
    sun.keyframe_insert(data_path="location", frame=frame + 1)

# === Other stuff ===

# make sure the rays travel far enough
camera.data.clip_end = 100000

# render samples per pixel! The higher the better (at the cost of computer)
bpy.context.scene.cycles.samples = 500000
camera.data.type = 'PERSP'

# ten degree FoV - should be enough to
# make sure its only in 1 pixel - check based on target size.
bpy.context.object.data.angle = 0.174533

#smallest possible odd number resolution
bpy.context.scene.render.resolution_x = 5
bpy.context.scene.render.resolution_y = 5

# turn off rays from the world
bpy.context.scene.world.cycles_visibility.scatter = False
bpy.context.scene.world.cycles_visibility.transmission = False
bpy.context.scene.world.cycles_visibility.glossy = False
bpy.context.scene.world.cycles_visibility.diffuse = False
bpy.context.scene.world.cycles_visibility.camera = False

bpy.context.scene.render.fps = 1
