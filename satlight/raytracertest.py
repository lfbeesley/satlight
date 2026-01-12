import bpy
import numpy as np
from mathutils import Vector
import matplotlib.pyplot as plt
from tqdm import tqdm
import BRDF
from hide_warnings import hide_warnings

obj_path = '/Users/l.beesley@bham.ac.uk/Documents/Lightcurves/satlight/data/models/Starlink/starlink_HR/uploads_files_3710445_SpaceXStarlinkSatelliteHighRes.obj/SpaceXStarlinkSatelliteHighRes.obj'

# CAMERA PRELIM SETUP
CAMERA_DISTANCE = 10
resolution_X = 500          # Width in pixels
resolution_Y = 700          # Height in pixels

# Clear the scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import into Blender and ignore .mtl file
@hide_warnings()
def add_object(file_name):
    bpy.ops.wm.obj_import(filepath=file_name, filter_obj=True, filter_glob='*.mtl', use_split_objects=False, use_split_groups=False)

add_object(obj_path)
#bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, 0))
# Get all imported objects
imported_objects = bpy.context.selected_objects

# Move all objects so the geometric center is at origin
# Collect all mesh objects
mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']

if not mesh_objects:
    raise RuntimeError("No mesh objects found in scene.")

# Compute bounding box of all meshes combined
bpy.context.view_layer.update()
all_corners = []
for obj in mesh_objects:
    all_corners.extend([obj.matrix_world @ Vector(corner) for corner in obj.bound_box])

min_corner = Vector(map(min, zip(*all_corners)))
max_corner = Vector(map(max, zip(*all_corners)))
center = (min_corner + max_corner) / 2

# Move all meshes so their center is at origin
for obj in mesh_objects:
    obj.location -= center

# Add sun light at (0, 0, 20)
light_data = bpy.data.lights.new(name="Sun", type='SUN')
light_object = bpy.data.objects.new(name="Sun", object_data=light_data)
bpy.context.collection.objects.link(light_object)
sun_direction = Vector((0, 1, 0))  

# Add camera
camera_object = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
camera_object.data.type = 'ORTHO'
bpy.context.collection.objects.link(camera_object)
bpy.context.scene.camera = camera_object

# Set camera position at specified distance
observer_direction = Vector((0, 1, 1)).normalized()
camera_object.data.ortho_scale = np.max(max_corner - min_corner)
camera_object.location = observer_direction * camera_object.data.ortho_scale
print(f"Ortho scale set to: {camera_object.data.ortho_scale}")


# Point camera at geometric center (which is now at origin)
direction = Vector((0, 0, 0)) - camera_object.location
camera_object.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler() #-Z is the cameras conventional 'look direction' point that toward origin

# Set resolution
scene = bpy.context.scene
scene.render.resolution_x = resolution_X
scene.render.resolution_y = resolution_Y


print(f"\nCamera configuration:")
print(f"Camera position: {camera_object.location}")
print(f"Camera distance: {CAMERA_DISTANCE}")
print(f"Resolution: {resolution_X}x{resolution_Y}")

# Lambertian BRDF
def lambertian_brdf(normal, light_dir, view_dir, albedo = 1):
    """Lambertian shading"""
    n_dot_l = max(0, np.dot(normal, light_dir))
    return albedo / np.pi * n_dot_l

def cook_torrance_brdf(normal, light_dir, view_dir, albedo=1, roughness=0, metallic=0.1):
    """Cook-Torrance microfacet BRDF - more realistic for metals/glossy surfaces"""
    # Convert to numpy arrays
    normal = np.array(normal)
    light_dir = np.array(light_dir)
    view_dir = np.array(view_dir)
    
    n_dot_l = max(0.001, np.dot(normal, light_dir))
    n_dot_v = max(0.001, np.dot(normal, view_dir))
    
    # Halfway vector
    halfway = (light_dir + view_dir)  # Now both are numpy arrays
    halfway = halfway / np.linalg.norm(halfway)
    n_dot_h = max(0, np.dot(normal, halfway))
    v_dot_h = max(0.001, np.dot(view_dir, halfway))
    
    # Fresnel (Schlick approximation)
    F0 = 0.04 * (1 - metallic) + albedo * metallic
    fresnel = F0 + (1 - F0) * (1 - v_dot_h) ** 5
    
    # GGX/Trowbridge-Reitz distribution
    alpha = roughness * roughness
    alpha2 = alpha * alpha
    denom = n_dot_h * n_dot_h * (alpha2 - 1) + 1
    D = alpha2 / (np.pi * denom * denom + 1e-10)  # Added epsilon for stability
    
    # Geometry term (Smith)
    def G1(n_dot_x):
        k = alpha / 2
        return n_dot_x / (n_dot_x * (1 - k) + k + 1e-10)  # Added epsilon
    
    G = G1(n_dot_l) * G1(n_dot_v)
    
    # Specular term
    specular = (D * G * fresnel) / (4 * n_dot_l * n_dot_v + 1e-10)  # Added epsilon
    
    # Diffuse term (with energy conservation)
    diffuse = (1 - fresnel) * (1 - metallic) * albedo / np.pi * n_dot_l
    
    return diffuse + specular

def phong_brdf(normal, light_dir, view_dir, albedo=1, specular=0.8, shininess=32):
    """Phong shading with diffuse + specular components"""
    # Diffuse component
    n_dot_l = max(0, np.dot(normal, light_dir))
    diffuse = albedo / np.pi * n_dot_l
    
    # Specular component (Phong)
    reflect_dir = 2 * n_dot_l * normal - light_dir
    spec_angle = max(0, np.dot(reflect_dir, view_dir))
    specular_component = specular * (shininess + 2) / (2 * np.pi) * (spec_angle ** shininess)
    
    return diffuse + specular_component

# Ray casting setup
depsgraph = bpy.context.evaluated_depsgraph_get()
width = scene.render.resolution_x
height = scene.render.resolution_y

# Initialize image array
image = np.zeros((height, width))

hit_count = 0
shadow_count = 0

print(f"\nStarting ray cast for {width}x{height} pixels...")

# Get camera matrix for transforming rays
cam_matrix = camera_object.matrix_world

# Create ray direction in camera space
aspect_ratio_render = width / height

# Calculate camera pixel size
pixel_size_x = camera_object.data.ortho_scale / resolution_X
pixel_size_y = camera_object.data.ortho_scale / aspect_ratio_render / resolution_Y
pixel_area = pixel_size_x * pixel_size_y

projected_areas = []

distance_to_observer = 550e6

ray_dir_world = (cam_matrix @ Vector((0, 0, -1)) - cam_matrix @ Vector((0, 0, 0))).normalized()
ray_dir = np.array(ray_dir_world)

# Ray cast for each pixel
for y in tqdm(range(height)):
    for x in range(width):
        # Calculate ray for this pixel in normalised coordinates
        screen_x = (2.0 * x / width) - 1.0
        screen_y = 1.0 - (2.0 * y / height)  # Flipped Y axis
        
        # Ray direction in camera to scale to local coordinates
        ray_origin_cam = Vector((
            screen_x * (camera_object.data.ortho_scale / 2) ,
            screen_y * (camera_object.data.ortho_scale / 2) / aspect_ratio_render,
            0.0
        ))
        
        ray_origin_world = cam_matrix @ ray_origin_cam
        
        result, location, normal, index, obj_hit, matrix = scene.ray_cast(
            depsgraph,
            ray_origin_world,
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
                flux = 0 
            else:
                view_dir = -ray_dir
                print(obj_hit)
                brdf = lambertian_brdf(hit_normal, sun_direction, view_dir)
                radiance = brdf * 1360

                # Calculated projected area
                cos_theta = max(0,np.dot(hit_normal, view_dir))

                projected_area = pixel_area * cos_theta  
                projected_areas.append(projected_area)   

                flux = radiance * projected_area / distance_to_observer**2            
            
            image[y, x] = flux

#I think it's nearly there, lets try simulation of a pass
print("\nRay casting complete!")
print(f"For orthographic scale of {camera_object.data.ortho_scale}")
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