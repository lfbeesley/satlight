import numpy as np
from PIL import Image
import math

# ============================================================================
# SCENE SETUP - Define all objects and lights
# ============================================================================

# Sphere properties
sphere_center = np.array([0.0, 0.0, -5.0])
sphere_radius = 1.0
sphere_reflectance = 0.5  # Single value: 0=black, 1=white (albedo)

# Light properties (sun/directional light)
light_direction = np.array([-1.0, 1.0, -0.5])
light_direction = light_direction / np.linalg.norm(light_direction)  # normalize
light_intensity = 10.0  # Single value for monochromatic light

# Camera properties
camera_pos = np.array([0.0, 0.0, 0.0])
focal_length = 0.5

# Image dimensions
width = 400
height = 300
aspect_ratio = height / width

print("Scene setup:")
print(f"  Sphere center: {sphere_center}")
print(f"  Sphere radius: {sphere_radius}")
print(f"  Sphere reflectance: {sphere_reflectance}")
print(f"  Light direction: {light_direction}")
print(f"  Light intensity: {light_intensity}")
print(f"  Camera position: {camera_pos}")

# ============================================================================
# SINGLE RAY TRACE - Choose a pixel and trace its ray
# ============================================================================

# Choose pixel to trace (center of image)
pixel_x = 500 
pixel_y = 300

print(f"\nTracing pixel ({pixel_x}, {pixel_y}):")

# Calculate screen space coordinates
screen_x = (pixel_x - width / 2) / width
screen_y = ((pixel_y - height / 2) / height) * aspect_ratio

print(f"  Screen space: ({screen_x:.3f}, {screen_y:.3f})")

# Calculate camera basis vectors
forward = np.array([0.0, 0.0, -1.0])
forward = forward / np.linalg.norm(forward)

up = np.array([0.0, 1.0, 0.0])
right = np.cross(up, forward)
right = right / np.linalg.norm(right)
up = np.cross(forward, right)

print(f"  Camera forward: {forward}")
print(f"  Camera right: {right}")
print(f"  Camera up: {up}")

# Calculate ray direction
ray_origin = camera_pos
ray_dir = forward * focal_length + right * screen_x + up * screen_y
ray_dir = ray_dir / np.linalg.norm(ray_dir)

print(f"  Ray origin: {ray_origin}")
print(f"  Ray direction: {ray_dir}")

# ============================================================================
# RAY-SPHERE INTERSECTION
# ============================================================================

# Calculate intersection using quadratic formula
oc = ray_origin - sphere_center
a = np.dot(ray_dir, ray_dir)
b = 2.0 * np.dot(oc, ray_dir)
c = np.dot(oc, oc) - sphere_radius * sphere_radius
discriminant = b * b - 4 * a * c

print(f"\nIntersection test:")
print(f"  a={a:.3f}, b={b:.3f}, c={c:.3f}")
print(f"  discriminant={discriminant:.3f}")

if discriminant < 0:
    print("  NO HIT - ray misses sphere")
    hit = False
else:
    # Calculate hit distance
    t = (-b - math.sqrt(discriminant)) / (2.0 * a)
    if t < 0.001:
        t = (-b + math.sqrt(discriminant)) / (2.0 * a)
    
    if t < 0.001:
        print("  NO HIT - sphere behind camera")
        hit = False
    else:
        # Calculate hit point and normal
        hit_point = ray_origin + ray_dir * t
        hit_normal = hit_point - sphere_center
        hit_normal = hit_normal / np.linalg.norm(hit_normal)
        
        hit = True
        hit_distance = t
        
        print(f"  HIT!")
        print(f"  Hit distance: {hit_distance:.3f}")
        print(f"  Hit point: {hit_point}")
        print(f"  Hit normal: {hit_normal}")

# ============================================================================
# LIGHTING CALCULATION (only if we hit something)
# ============================================================================

if hit:
    # View direction (from hit point to camera)
    view_dir = -ray_dir
    
    print(f"\nLighting calculation:")
    print(f"  View direction: {view_dir}")
    
    # Light direction (for sun, it's constant)
    light_dir = -light_direction
    print(f"  Light direction: {light_dir}")
    
    # Check shadow (cast ray from hit point toward light)
    shadow_ray_origin = hit_point + hit_normal * 0.001  # offset to avoid self-intersection
    shadow_ray_dir = light_dir
    
    # Shadow ray vs sphere intersection
    oc_shadow = shadow_ray_origin - sphere_center
    a_shadow = np.dot(shadow_ray_dir, shadow_ray_dir)
    b_shadow = 2.0 * np.dot(oc_shadow, shadow_ray_dir)
    c_shadow = np.dot(oc_shadow, oc_shadow) - sphere_radius * sphere_radius
    discriminant_shadow = b_shadow * b_shadow - 4 * a_shadow * c_shadow
    
    in_shadow = False
    if discriminant_shadow >= 0:
        t_shadow = (-b_shadow - math.sqrt(discriminant_shadow)) / (2.0 * a_shadow)
        if t_shadow > 0.001:
            in_shadow = True
    
    print(f"  In shadow: {in_shadow}")
    
    if not in_shadow:
        # Calculate incoming light intensity (for sun, no falloff)
        incoming_intensity = light_intensity
        print(f"  Incoming intensity: {incoming_intensity}")
        
        # Lambertian (diffuse) shading
        cos_theta = max(0, np.dot(light_dir, hit_normal))
        print(f"  cos(theta): {cos_theta:.3f}")
        
        # Final intensity (single channel)
        final_intensity = sphere_reflectance * incoming_intensity * cos_theta
        print(f"  Final intensity: {final_intensity}")
        
        # Clip to valid range [0, 1]
        final_intensity = np.clip(final_intensity, 0, 1)
        print(f"  Clipped intensity: {final_intensity}")
    else:
        # In shadow - use ambient only
        ambient_intensity = 0.1
        final_intensity = sphere_reflectance * ambient_intensity
        print(f"  Shadow - ambient intensity: {final_intensity}")
else:
    # No hit - black background
    final_intensity = 0.0
    print(f"\nBackground intensity: {final_intensity}")

# ============================================================================
# FULL IMAGE RENDER (using the same logic as above)
# ============================================================================

print("\n" + "="*70)
print("RENDERING FULL IMAGE...")
print("="*70)

# Single channel (grayscale) image
image = np.zeros((height, width))

for y in range(height):
    for x in range(width):
        # Calculate ray
        screen_x = (x - width / 2) / width
        screen_y = ((y - height / 2) / height) * aspect_ratio
        
        ray_origin = camera_pos
        ray_dir = forward * focal_length + right * screen_x + up * screen_y
        ray_dir = ray_dir / np.linalg.norm(ray_dir)
        
        # Intersect with sphere
        oc = ray_origin - sphere_center
        a = np.dot(ray_dir, ray_dir)
        b = 2.0 * np.dot(oc, ray_dir)
        c = np.dot(oc, oc) - sphere_radius * sphere_radius
        discriminant = b * b - 4 * a * c
        
        if discriminant >= 0:
            t = (-b - math.sqrt(discriminant)) / (2.0 * a)
            if t < 0.001:
                t = (-b + math.sqrt(discriminant)) / (2.0 * a)
            
            if t >= 0.001:
                # Hit!
                hit_point = ray_origin + ray_dir * t
                hit_normal = (hit_point - sphere_center) / np.linalg.norm(hit_point - sphere_center)
                
                # Light direction
                light_dir = -light_direction
                
                # Shadow check
                shadow_ray_origin = hit_point + hit_normal * 0.001
                oc_shadow = shadow_ray_origin - sphere_center
                a_shadow = np.dot(light_dir, light_dir)
                b_shadow = 2.0 * np.dot(oc_shadow, light_dir)
                c_shadow = np.dot(oc_shadow, oc_shadow) - sphere_radius * sphere_radius
                discriminant_shadow = b_shadow * b_shadow - 4 * a_shadow * c_shadow
                
                in_shadow = False
                if discriminant_shadow >= 0:
                    t_shadow = (-b_shadow - math.sqrt(discriminant_shadow)) / (2.0 * a_shadow)
                    if t_shadow > 0.001:
                        in_shadow = True
                
                if not in_shadow:
                    # Lambertian shading
                    incoming_intensity = light_intensity
                    cos_theta = max(0, np.dot(light_dir, hit_normal))
                    intensity = sphere_reflectance * incoming_intensity * cos_theta
                else:
                    # Ambient
                    intensity = sphere_reflectance * 0.1
                
                image[y, x] = np.clip(intensity, 0, 1)
    
    if y % 10 == 0:
        print(f"Progress: {100 * y / height:.1f}%", end='\r')

print("Progress: 100.0%")

# Save image
image_8bit = (image * 255).astype(np.uint8)
img = Image.fromarray(image_8bit)
img.save('raytraced_sphere.png')
print("\nSaved to raytraced_sphere.png")
img.show()

print("\n" + "="*70)
print("DONE! All variables are accessible for inspection.")
print("="*70)