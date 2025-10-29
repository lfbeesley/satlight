"""
Test script with real Starlink v1.5 satellite model
Demonstrates ray casting, shadowing, and BRDF application

Run with: python starlink_ray_test.py
Requires: pip install bpy
"""

import bpy
import numpy as np
from pathlib import Path

print("="*70)
print("STARLINK SATELLITE RAY TRACING TEST")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to your Starlink .obj file
OBJ_FILE_PATH = "/Users/l.beesley@bham.ac.uk/Documents/LightcurveAnalysis/satlight/data/models/Starlink/starlink_HR/uploads_files_3710445_SpaceXStarlinkSatelliteHighRes.obj/SpaceXStarlinkSatelliteHighRes.obj"  # UPDATE THIS

# Simple Lambertian BRDF for testing
def simple_lambertian_brdf(normal, sun_dir, view_dir, reflectance=0.3):
    """
    Simple diffuse BRDF for testing
    
    Returns intensity based on cosine law
    """
    cos_theta = max(0, np.dot(sun_dir, normal))
    return reflectance * cos_theta

# ============================================================================
# SETUP SCENE
# ============================================================================

print("\nSetting up scene...")

# Clear existing scene
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

# Import Starlink satellite
print(f"Loading: {OBJ_FILE_PATH}")
try:
    bpy.ops.import_scene.obj(filepath=OBJ_FILE_PATH)
    print("  ✓ Satellite loaded")
except Exception as e:
    print(f"  ✗ Error loading .obj file: {e}")
    print("  Please update OBJ_FILE_PATH in the script")
    exit(1)

# Get the imported satellite object
satellite = bpy.context.selected_objects[0]
satellite.location = (0, 0, 0)  # Center at origin
print(f"  Satellite: {satellite.name}")
print(f"  Vertices: {len(satellite.data.vertices)}")
print(f"  Faces: {len(satellite.data.polygons)}")

# ============================================================================
# CAMERA AND LIGHTING SETUP
# ============================================================================

# Camera position (observer on Earth looking up)
camera_pos = np.array([0.0, 0.0, 20.0])  # 10 units away

# Sun direction (example - would come from your geometry module)
sun_direction = np.array([-0.5, 0.3, -0.8])
sun_direction = sun_direction / np.linalg.norm(sun_direction)

print(f"\nCamera position: {camera_pos}")
print(f"Sun direction: {sun_direction}")

# Get depsgraph for ray casting
depsgraph = bpy.context.evaluated_depsgraph_get()

# ============================================================================
# TEST 1: Single ray through center
# ============================================================================

print("\n" + "="*70)
print("TEST 1: Single ray toward satellite center")
print("="*70)

ray_origin = camera_pos
ray_direction = -camera_pos / np.linalg.norm(camera_pos)  # Toward origin

print(f"Ray origin: {ray_origin}")
print(f"Ray direction: {ray_direction}")

result, location, normal, index, obj, matrix = scene.ray_cast(
    depsgraph,
    ray_origin,
    ray_direction
)

if result:
    print(f"\n✓ HIT!")
    print(f"  Hit location: {np.array(location)}")
    print(f"  Hit normal: {np.array(normal)}")
    print(f"  Hit object: {obj.name}")
    print(f"  Face index: {index}")
    
    # Check if this point is illuminated
    hit_point = np.array(location)
    hit_normal = np.array(normal)
    
    # Cast shadow ray
    shadow_origin = hit_point + hit_normal * 0.001
    shadow_result, _, _, _, shadow_obj, _ = scene.ray_cast(
        depsgraph,
        shadow_origin,
        sun_direction
    )
    
    if shadow_result:
        print(f"\n  ✗ Point is in SHADOW (blocked by {shadow_obj.name})")
        print(f"    This demonstrates self-shadowing!")
    else:
        print(f"\n  ✓ Point is ILLUMINATED by sun")
        
        # Calculate BRDF
        view_dir = -ray_direction
        intensity = simple_lambertian_brdf(hit_normal, sun_direction, view_dir)
        print(f"    BRDF intensity: {intensity:.3f}")
else:
    print("\n✗ MISS - ray didn't hit satellite")

# ============================================================================
# TEST 2: Grid of rays (low resolution render)
# ============================================================================

print("\n" + "="*70)
print("TEST 2: Low resolution render (10x10 pixels)")
print("="*70)

width, height = 500, 250
focal_length = 0.5
image = np.zeros((height, width))

hit_count = 0
shadow_count = 0

print(f"\nCasting {width*height} rays...")

for y in range(height):
    for x in range(width):
        # Calculate ray for this pixel
        screen_x = (x - width/2) / width
        screen_y = (y - height/2) / height
        
        ray_dir = np.array([screen_x, screen_y, -1.0])
        ray_dir = ray_dir / np.linalg.norm(ray_dir)
        
        # Cast ray
        result, location, normal, index, obj, matrix = scene.ray_cast(
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
            shadow_result, _, _, _, _, _ = scene.ray_cast(
                depsgraph,
                shadow_origin,
                sun_direction
            )
            
            if shadow_result:
                # In shadow
                shadow_count += 1
                intensity = 0.0  # or use ambient
            else:
                # Illuminated - apply BRDF
                view_dir = -ray_dir
                intensity = simple_lambertian_brdf(hit_normal, sun_direction, view_dir)
            
            image[y, x] = intensity

# Results
total_flux = np.sum(image)

print(f"\nResults:")
print(f"  Total rays cast: {width*height}")
print(f"  Rays that hit satellite: {hit_count}")
print(f"  Hit rate: {100*hit_count/(width*height):.1f}%")
print(f"  Points in shadow: {shadow_count}")
print(f"  Shadow rate: {100*shadow_count/max(hit_count,1):.1f}% of hits")
print(f"\n  Total integrated flux: {total_flux:.3f}")
print(f"  --> This is ONE point on your light curve")

# ============================================================================
# TEST 3: Visualize which parts are visible and lit
# ============================================================================

print("\n" + "="*70)
print("TEST 3: Sample different viewing angles")
print("="*70)

# Test from different angles
test_angles = [
    ("Front", np.array([0, 0, 10])),
    ("Side", np.array([10, 0, 0])),
    ("Above", np.array([0, 10, 0])),
    ("Angled", np.array([5, 5, 5])),
]

for name, cam_pos in test_angles:
    # Ray toward satellite center
    ray_dir = -cam_pos / np.linalg.norm(cam_pos)
    
    result, location, normal, _, _, _ = scene.ray_cast(
        depsgraph,
        cam_pos,
        ray_dir
    )
    
    if result:
        hit_normal = np.array(normal)
        cos_angle = np.dot(sun_direction, hit_normal)
        print(f"{name:10s}: Hit, cos(sun angle) = {cos_angle:+.3f}", end="")
        
        if cos_angle > 0:
            print(" (illuminated)")
        else:
            print(" (facing away from sun)")
    else:
        print(f"{name:10s}: Miss")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("SUMMARY - WORKFLOW FOR LIGHT CURVE GENERATION")
print("="*70)
print("""
1. Load satellite .obj file (DONE ✓)
2. For each time step in satellite pass:
   a. Get viewing geometry from your geometry.py module
   b. Position camera using observer→satellite vector
   c. Set sun direction from geometry
   d. Render:
      - Loop over all pixels (e.g., 500×500)
      - Cast ray for each pixel
      - If hit:
        * Check shadow ray
        * Apply your BRDF
        * Store intensity
   e. Sum all pixels = total flux
   f. Store (time, flux) pair
3. Plot flux vs time = light curve

NEXT STEPS:
- Integrate with your geometry.py module for real orbital positions
- Implement your spectral BRDF function
- Loop over time range for complete light curve
""")

print("="*70)