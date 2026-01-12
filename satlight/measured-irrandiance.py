import bpy
import math
import os
import sys
import subprocess
import csv

def setup_scene(albedo, incoming_angle_deg, irradiance):
    """A function to set up the scene for a single measurement."""
    # Clear all objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Create the plane
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, 0))
    plane = bpy.context.active_object
    mat = bpy.data.materials.new(name="Lambertian_Material")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (albedo, albedo, albedo, 1.0)
    bsdf.inputs['Roughness'].default_value = 1.0
    plane.data.materials.append(mat)

    # Create the sun
    incoming_angle_rad = math.radians(incoming_angle_deg)
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 5))
    sun = bpy.context.active_object
    sun.data.energy = irradiance
    sun.rotation_euler = (incoming_angle_rad, 0, 0)

    # Create the camera (looking straight down)
    bpy.ops.object.camera_add(location=(0, 0, 5)) # Distance doesn't matter for orthographic sun
    camera = bpy.context.active_object
    bpy.context.scene.camera = camera

    # Configure rendering
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.render.resolution_x = 128
    bpy.context.scene.render.resolution_y = 128
    bpy.context.scene.view_settings.view_transform = 'Standard'
    bpy.context.scene.view_settings.exposure = 0
    bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
    bpy.context.scene.render.image_settings.color_depth = '32'


def read_center_pixel_from_file(filepath):
    """Loads a rendered EXR image and reads the center pixel value."""
    img = bpy.data.images.load(filepath)
    pixels = img.pixels[:]
    width = img.size[0]
    center_pixel_index = ( (width * (width // 2)) + (width // 2) ) * 4
    r = pixels[center_pixel_index]
    bpy.data.images.remove(img)
    return r


def main():
    """
    Main function to run the simulation loop and generate a plot.
    """
    print("--- STARTING SIMULATION LOOP ---")
    
    # --- Simulation Parameters ---
    SURFACE_ALBEDO = 0.8
    INCOMING_IRRADIANCE = 1360
    
    # --- Data Storage ---
    angles = []
    theoretical_radiance_data = []
    measured_radiance_data = []
    
    temp_dir = bpy.app.tempdir
    
    # --- The Main Loop ---
    # Iterate from 0 to 90 degrees in 10-degree steps.
    for angle in range(0, 91, 10):
        print(f"\nSimulating for angle: {angle}°")
        
        # 1. Set up the scene for the current angle
        # CORRECTED: Fixed typo in variable name from SURFACE_ALbedo to SURFACE_ALBEDO
        setup_scene(SURFACE_ALBEDO, angle, INCOMING_IRRADIANCE)
        
        # 2. Render the scene
        output_path = os.path.join(temp_dir, f"render_{angle}.exr")
        bpy.context.scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)
        
        # 3. Calculate theoretical radiance
        incoming_angle_rad = math.radians(angle)
        theoretical_radiance = (SURFACE_ALBEDO * INCOMING_IRRADIANCE * math.cos(incoming_angle_rad)) / math.pi
        
        # 4. Measure radiance from the rendered image
        measured_radiance = read_center_pixel_from_file(output_path)
        
        # 5. Store the data
        angles.append(angle)
        theoretical_radiance_data.append(theoretical_radiance)
        measured_radiance_data.append(measured_radiance)
        
        print(f"  - Theoretical: {theoretical_radiance:.2f} W/m²/sr")
        print(f"  - Measured:    {measured_radiance:.2f} W/m²/sr")

    print("\n--- SAVING DATA TO CSV ---")

    headers = ['Angle (Degrees)', 'Theoretical Radiance (W/m²/sr)', 'Measured Radiance (W/m²/sr)']

    # Combine the lists into rows of data using zip
    # zip pairs up the elements: (angle1, theoretical1, measured1), (angle2, theoretical2, measured2), ...
    rows = zip(angles, theoretical_radiance_data, measured_radiance_data)

    # Define the path to save the file in the user's home directory
    home_dir = os.path.expanduser('~')
    data_path = os.path.join(home_dir, 'radiance_vs_angle_data.csv')

    # Write the data to the CSV file
    with open(data_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write the header row first
        writer.writerow(headers)
        
        # Write all the data rows
        writer.writerows(rows)

    print(f"\nData saved successfully to: {data_path}")

# Run the main function
if __name__ == "__main__":
    main()
    print("\nScript finished.")

