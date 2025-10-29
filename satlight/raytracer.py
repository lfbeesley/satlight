import bpy
import numpy as np
from mathutils import Vector
import trimesh
import matplotlib.pyplot as plt
from tqdm import tqdm

# ===== MANUAL SETTINGS =====
obj_path = '/Users/l.beesley@bham.ac.uk/Documents/Lightcurves/satlight/data/models/Starlink/starlink_HR/uploads_files_3710445_SpaceXStarlinkSatelliteHighRes.obj/SpaceXStarlinkSatelliteHighRes.obj'

# CAMERA PRELIM SETUP
CAMERA_DISTANCE = 30
resolution_X = 500          # Width in pixels
resolution_Y = 750          # Height in pixels

# Clear the scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Remove all lights
for light in bpy.data.lights:
    bpy.data.lights.remove(light)

# Load with trimesh to get true geometric center
mesh = trimesh.load(obj_path, force='mesh')

# If it's a scene with multiple meshes, combine them
if isinstance(mesh, trimesh.Scene):
    # Combine all meshes in the scene
    mesh = trimesh.util.concatenate([
        trimesh.Trimesh(vertices=geom.vertices, faces=geom.faces)
        for geom in mesh.geometry.values()
    ])

# Get the geometric center (centroid of vertices)
geometric_center = (0, 0, 5) #mesh.vertices.mean(axis=0)
print(f"Geometric center from trimesh: {geometric_center}")

# Import into Blender
bpy.ops.wm.obj_import(filepath=obj_path)

# Get all imported objects
imported_objects = bpy.context.selected_objects

# Move all objects so the geometric center is at origin
offset = Vector(geometric_center)
for obj in imported_objects:
    obj.location -= offset

# Add sun light at (0, 0, 20)
light_data = bpy.data.lights.new(name="Sun", type='SUN')
light_object = bpy.data.objects.new(name="Sun", object_data=light_data)
bpy.context.collection.objects.link(light_object)
light_object.location = (0, 10, 0)
sun_direction = Vector((0, 1, 0))  

# Add camera
camera_data = bpy.data.cameras.new(name="Camera")
camera_data.type = 'PERSP'
camera_data.angle = 0.5 # 0.5 #TO BE EDITTED TO SCALE THE OBJECT
camera_object = bpy.data.objects.new("Camera", camera_data)
bpy.context.collection.objects.link(camera_object)
bpy.context.scene.camera = camera_object

# Set camera position at specified distance
initial_direction = Vector((0, 1, 1)).normalized()
camera_object.location = initial_direction * CAMERA_DISTANCE

# Point camera at geometric center (which is now at origin)
direction = Vector((0, 0, 0)) - camera_object.location
camera_object.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

# Set resolution
scene = bpy.context.scene
scene.render.resolution_x = resolution_X
scene.render.resolution_y = resolution_Y

# Set sensor size
sensor_width = camera_data.sensor_width
sensor_height = sensor_width * (resolution_Y/resolution_X)

print(f"\nCamera configuration:")
print(f"Camera position: {camera_object.location}")
print(f"Camera distance: {CAMERA_DISTANCE}")
print(f"Resolution: {resolution_X}x{resolution_Y}")

# Simple Lambertian BRDF
def simple_lambertian_brdf(normal, light_dir, view_dir, albedo = 1):
    """Simple Lambertian shading"""
    n_dot_l = max(0, np.dot(normal, light_dir))
    return albedo / np.pi * n_dot_l

# Ray casting setup
depsgraph = bpy.context.evaluated_depsgraph_get()
render = scene.render
width = render.resolution_x
height = render.resolution_y

# Initialize image array
image = np.zeros((height, width))

hit_count = 0
shadow_count = 0

print(f"\nStarting ray cast for {width}x{height} pixels...")

# Get camera matrix for transforming rays
cam_matrix = camera_object.matrix_world
camera_pos = np.array(camera_object.location)

# Create ray direction in camera space
aspect_ratio_render = width / height

# Calculate camera pixel size
pixel_size_x = camera_data.ortho_scale / resolution_Y
pixel_size_y = camera_data.ortho_scale / aspect_ratio_render / resolution_X
pixel_area = pixel_size_x * pixel_size_y

projected_areas = []

distance_to_observer = 300e3

# Ray cast for each pixel
for y in tqdm(range(height)):
    for x in range(width):
        # Calculate ray for this pixel in normalised coordinates
        screen_x = (2.0 * x / width) - 1.0
        screen_y = 1.0 - (2.0 * y / height)  # Flipped Y axis
        
        # Ray direction in camera to scale to local coordinates
        ray_cam = Vector((
            screen_x * (camera_data.ortho_scale / 2),
            screen_y * (camera_data.ortho_scale / 2),
            -1.0
        ))
        
        # Transform ray to world space
        ray_dir_world = (cam_matrix.to_3x3() @ ray_cam).normalized()
        ray_dir = np.array(ray_dir_world)
        
        # Cast ray
        result, location, normal, index, obj_hit, matrix = scene.ray_cast(
            depsgraph,
            camera_pos,
            ray_dir
        )
        
        if result:
            hit_count += 1
            hit_point = np.array(location)
            hit_normal = np.array(normal)
            
            # Check shadow
            shadow_origin = hit_point + hit_normal * 0.001
            shadow_result, *_ = scene.ray_cast(
                depsgraph,
                shadow_origin,
                sun_direction
            )
            
            if shadow_result:
                shadow_count += 1
                intensity = 0 
            else:
                view_dir = -ray_dir
                intensity = simple_lambertian_brdf(hit_normal, sun_direction, view_dir)

                # Calculated projected area
                cos_theta = max(0,np.dot(hit_normal, view_dir))

                projected_area = pixel_area * cos_theta  
                projected_areas.append(projected_area)   

                intensity *= (1360 * projected_area) / (4 * np.pi * distance_to_observer**2)                   
            
            image[y, x] = intensity

print("\nRay casting complete!")
print(f"For orthographic scale of {camera_data.ortho_scale}")
print(f"Total measured Intensity: {np.sum(image):.2e} W")
print(f"Total percieved area: {np.sum(projected_areas):.4e} m^2")
print(f"Total hits: {hit_count} out of {width * height} pixels")
print(f"Pixels in shadow: {shadow_count}")
print(f"Hit percentage: {100 * hit_count / (width * height):.2f}%")

# Save or display the image
plt.figure(figsize=(8, 8 * height / width))
plt.imshow(image, cmap='gray', origin='upper')
plt.colorbar()
plt.title('Rendered Image')
plt.show()
''''''