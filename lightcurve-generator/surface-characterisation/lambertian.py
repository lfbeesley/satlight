import bpy
import bmesh
import mathutils
import math
import csv
import os
from mathutils import Vector, Matrix

# Scene setup -- Remove existing objects 
if bpy.context.mode != 'OBJECT':
    bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

for material in bpy.data.materials:
    bpy.data.materials.remove(material)

for light in bpy.data.lights:
    bpy.data.lights.remove(light)

for world in bpy.data.worlds:
    bpy.data.worlds.remove(world)


def create_lambertian_plane():
    """Create a 1m x 1m plane with a Lambertian (perfectly diffuse) material."""
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, 0))
    plane = bpy.context.active_object
    plane.name = "Lambertian_Reflector"
    
    # Create a new material named "Lambertian_Material"
    mat = bpy.data.materials.new(name="Lambertian_Material")
    mat.use_nodes = True
    mat.node_tree.nodes.clear()
    
    # Create a Principled BSDF shader node for the material
    bsdf = mat.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = (0.8, 0.8, 0.8, 1.0)  # 80% reflectance
    bsdf.inputs['Roughness'].default_value = 1.0  # Fully rough for perfect diffusion
    bsdf.inputs['Metallic'].default_value = 0.0   # Non-metallic
    
    # Set specular to 0 for a pure Lambertian model. The input name varies by Blender version.
    if 'Specular' in bsdf.inputs: # For Blender 3.x
        bsdf.inputs['Specular'].default_value = 0.0
    elif 'Specular IOR Level' in bsdf.inputs: # For Blender 4.x
        bsdf.inputs['Specular IOR Level'].default_value = 0.0
    
    # Add a material output node and link the BSDF shader to it
    output = mat.node_tree.nodes.new(type='ShaderNodeOutputMaterial')
    mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    # Assign the created material to the plane object
    plane.data.materials.append(mat)
    return plane

def create_sun_light(location=(2, 2, 5), power=1000.0):
    """Creates a Sun lamp and rotates it to point at the world origin (0,0,0)."""
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    bpy.ops.object.light_add(type='SUN', location=location)
    light_obj = bpy.context.active_object
    light_obj.name = "Sun_Light_Source"

    # Calculate the direction from the light's position to the origin and point the light
    direction_vector = Vector((0.0, 0.0, 0.0)) - light_obj.location
    rot_quat = direction_vector.to_track_quat('-Z', 'Y')
    light_obj.rotation_euler = rot_quat.to_euler()

    # Configure the light's properties
    light_data = light_obj.data
    light_data.energy = power # Energy is in W/m^2 for Sun lamps
    light_data.angle = math.radians(0.53) # Angular size of the sun for soft shadows
    
    print(f"Created '{light_obj.name}' at {location}, pointing towards the origin.")
    return light_obj

def create_camera(location, target=(0, 0, 0)):
    """Creates a camera at a given location and points it at a target."""
    bpy.ops.object.camera_add(location=location)
    camera_obj = bpy.context.active_object
    camera_obj.name = "Measurement_Camera"
    camera_obj.data.type = 'ORTHO'
    
    # Point the camera towards the target
    direction = Vector(target) - Vector(location)
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera_obj.rotation_euler = rot_quat.to_euler()
    
    # Set camera properties for a very wide field of view to ensure
    # the entire 1x1m plane is visible to the 1x1 pixel render.
    # A smaller lens value gives a wider field of view.
    camera = camera_obj.data
    camera.lens = 10 # Ultra-wide angle lens
    camera.sensor_width = 36
    
    return camera_obj

# --- RENDERING & MEASUREMENT ---

def setup_cycles_rendering():
    """Configures Cycles for physically accurate rendering with a black background."""
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 512
    scene.cycles.use_denoising = True
    
    # Ensure the compositor is not used, as it could alter final render values.
    scene.use_nodes = False

    # Create world if it doesn't exist
    if not scene.world:
        world = bpy.data.worlds.new("World")
        scene.world = world
    else:
        world = scene.world
    
    # Set up world background (black for controlled lighting) by clearing all nodes
    # and creating a new black background. This is more robust than assuming
    # a 'Background' node already exists.
    world.use_nodes = True
    world.node_tree.nodes.clear()
    
    background = world.node_tree.nodes.new(type='ShaderNodeBackground')
    background.inputs['Color'].default_value = (0, 0, 0, 1)  # Black background
    background.inputs['Strength'].default_value = 0.0 # Set to 0 to ensure no ambient light
    
    output = world.node_tree.nodes.new(type='ShaderNodeOutputWorld')
    world.node_tree.links.new(background.outputs['Background'], output.inputs['Surface'])

def measure_pixel_radiance(camera_obj, output_dir, alt_deg, az_deg):
    """Renders the scene as a single pixel, saves it, and measures its radiance."""
    scene = bpy.context.scene
    
    # Force render scale to 100% to ensure the output resolution is not scaled.
    scene.render.resolution_percentage = 100
    
    # Set render resolution to 1x1 pixel
    render_size = (1, 1)
    scene.render.resolution_x = render_size[0]
    scene.render.resolution_y = render_size[1]
    
    scene.render.image_settings.file_format = 'OPEN_EXR' # Use EXR for linear, high-dynamic-range data
    scene.camera = camera_obj
    
    # Set the filepath to save the EXR file in the specified directory
    filename = f"render_alt_{alt_deg}_az_{az_deg}.exr"
    filepath = os.path.join(output_dir, filename)
    scene.render.filepath = filepath

    # Render and save the file to the specified path
    bpy.ops.render.render(write_still=True)
    
    # Get the rendered image from the internal render result
    image = bpy.data.images.get('Render Result')
    if not image or not image.pixels:
        print("Warning: Render Result not found or is empty.")
        return 0.0
        
    # With a 1x1 render, there is only one pixel.
    # The pixels list will contain [R, G, B, A] for that single pixel.
    pixels = list(image.pixels)
    r, g, b = pixels[0], pixels[1], pixels[2]
    
    # In a linear EXR file, the RGB values are the radiance values.
    # We average the channels for a broadband (non-spectral) measurement.
    single_pixel_radiance = (r + g + b) / 3.0
        
    return single_pixel_radiance

def theoretical_sun_radiance(sun_energy, light_direction, surface_normal, reflectance=0.8):
    """
    Calculates the theoretical radiance for a Lambertian surface illuminated by a Sun lamp.
    Radiance L = (E * ρ * cos(θ)) / π
    Where:
      E = Irradiance from the sun lamp (sun_energy)
      ρ = Surface reflectance
      θ = Angle between the light direction and the surface normal
    """
    # Ensure vectors are normalized
    light_direction = light_direction.normalized()
    surface_normal = surface_normal.normalized()
    
    # The light direction vector points *to* the surface, so we need to invert it
    # to calculate the dot product correctly with the surface normal.
    cos_theta = max(0, -light_direction.dot(surface_normal))
    
    # Calculate radiance using the Lambertian formula
    radiance = (sun_energy * reflectance * cos_theta) / math.pi
    return radiance


def run_characterization_study():
    """Main function to run the light characterization study."""
    # --- 1. Setup the Scene ---
    setup_cycles_rendering()
    plane = create_lambertian_plane()
    
    # Define a fixed position for the light source
    light_pos = (2, 2, 5)
    sun_power = 1000.0
    light_obj = create_sun_light(light_pos, power=sun_power)
    
    # --- 2. Define Test Parameters & Output Paths ---
    results = []
    camera_radius = 2.0  # The distance of the camera from the origin (0,0,0)
    
    # Define the angles to test (from 10 to 80 degrees to avoid grazing angles)
    altitude_angles = [10, 20, 30, 45, 60, 75] # Angle from the XY plane (0-90)
    azimuth_angles = [0, 45, 90, 135, 180]    # Angle around the Z axis (0-360)

    # Define and create the output directory for the EXR files
    base_output_dir = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else os.path.expanduser("~")
    exr_output_dir = os.path.join(base_output_dir, "light-characterisation-data")
    os.makedirs(exr_output_dir, exist_ok=True)
    
    print("Starting characterization study...")
    print(f"Fixed light source at: {light_pos}")
    print(f"Saving EXR render files to: {exr_output_dir}")

    for alt_deg in altitude_angles:
        for az_deg in azimuth_angles:
            # Calculate Camera Position from Spherical Coordinates 
            alt_rad = math.radians(alt_deg)
            az_rad = math.radians(az_deg)
            
            x = camera_radius * math.cos(alt_rad) * math.cos(az_rad)
            y = camera_radius * math.cos(alt_rad) * math.sin(az_rad)
            z = camera_radius * math.sin(alt_rad)
            cam_pos = (x, y, z)

            print(f"Testing: Altitude={alt_deg}°, Azimuth={az_deg}° -> Cam Pos={tuple(round(c, 2) for c in cam_pos)}")

            # Remove previous camera
            for obj in bpy.data.objects:
                if obj.name.startswith("Measurement_Camera"):
                    bpy.data.objects.remove(obj, do_unlink=True)
            
            camera_obj = create_camera(cam_pos, target=(0, 0, 0))
            
            measured_radiance = measure_pixel_radiance(camera_obj, exr_output_dir, alt_deg, az_deg)
            
            # For the theoretical calculation
            light_direction = (Vector((0,0,0)) - Vector(light_pos)).normalized()
            surface_normal = Vector((0, 0, 1)) # Plane is at the origin, facing up
            
            theo_radiance = theoretical_sun_radiance(
                sun_energy=sun_power,
                light_direction=light_direction,
                surface_normal=surface_normal,
                reflectance=0.8
            )
            
            result = {
                'altitude_deg': alt_deg,
                'azimuth_deg': az_deg,
                'camera_pos': tuple(round(c, 3) for c in cam_pos),
                'measured_radiance': measured_radiance,
                'theoretical_radiance': theo_radiance,
                'ratio': measured_radiance / theo_radiance if theo_radiance > 0 else 0,
            }
            results.append(result)
            
            print(f"  Measured: {measured_radiance:.6f}, Theoretical: {theo_radiance:.6f}, Ratio: {result['ratio']:.3f}")

    # Save to CSV
    csv_output_path = os.path.join(base_output_dir, "lambertian_characterization_results.csv")
    
    print(f"\nSaving results to: {csv_output_path}")
    try:
        with open(csv_output_path, 'w', newline='') as csvfile:
            fieldnames = ['altitude_deg', 'azimuth_deg', 'camera_pos', 'measured_radiance', 'theoretical_radiance', 'ratio']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"✓ Results successfully saved.")
    except Exception as e:
        print(f"✗ Could not save CSV file: {e}")

    return csv_output_path, results

# --- RUN THE SCRIPT ---
if __name__ == "__main__":
    import tempfile
    saved_path, results = run_characterization_study()
    print(f"\nStudy complete! Check the CSV file at: {saved_path}")

