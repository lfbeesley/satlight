"""
trimesh_raycast_test.py
Test trimesh ray casting speed vs Blender for satellite lightcurve rendering.
"""
import trimesh
import numpy as np
import time

# =============================================================================
# Load mesh
# =============================================================================

scene = trimesh.load(
    '/Users/l.beesley@bham.ac.uk/Documents/Lightcurves/satlight/models/Skynet 5D/skynet.obj',
    force='scene', split_object=True, group_material=False,
)

# Build combined mesh with face->object mapping
combined_vertices = []
combined_faces    = []
face_object_ids   = []
vertex_offset     = 0
object_names      = list(scene.geometry.keys())

for obj_idx, (name, geom) in enumerate(scene.geometry.items()):
    combined_vertices.append(geom.vertices)
    combined_faces.append(geom.faces + vertex_offset)
    face_object_ids.extend([obj_idx] * len(geom.faces))
    vertex_offset += len(geom.vertices)

mesh = trimesh.Trimesh(
    vertices=np.vstack(combined_vertices),
    faces=np.vstack(combined_faces),
    process=False,
)
face_object_ids = np.array(face_object_ids)
print(f"Mesh: {len(mesh.faces)} faces, {len(mesh.vertices)} vertices")
print(f"Objects: {object_names}")

# =============================================================================
# Camera setup — orthographic, observer from +Y looking at origin
# =============================================================================

resolution_x = 300
resolution_y = 300
aspect_ratio = 1.0
ortho_scale  = float(mesh.bounding_box.extents.max()) * 1.5

# Explicit camera basis (observer along +Y)
obs_direction = np.array([0., 1., 0.])   # normalised observer direction
cam_pos       = obs_direction * ortho_scale
cam_dir       = -obs_direction            # look direction (toward origin)
cam_x         = np.array([1., 0., 0.])   # right
cam_y         = np.array([0., 0., 1.])   # up

sun_dir = np.array([1., 1., 0.])
sun_dir = sun_dir / np.linalg.norm(sun_dir)

half_w = ortho_scale / 2
half_h = ortho_scale / 2 / aspect_ratio

print(f"\northo_scale: {ortho_scale:.3f}")
print(f"cam_pos: {cam_pos}")
print(f"cam_dir: {cam_dir}")

# =============================================================================
# Project mesh bounding box onto image plane to get pixel bounds
# =============================================================================

bounds  = mesh.bounds   # (2,3)
mins, maxs = bounds[0], bounds[1]

# 8 corners of AABB
corners = np.array([
    [mins[0], mins[1], mins[2]],
    [maxs[0], mins[1], mins[2]],
    [mins[0], maxs[1], mins[2]],
    [maxs[0], maxs[1], mins[2]],
    [mins[0], mins[1], maxs[2]],
    [maxs[0], mins[1], maxs[2]],
    [mins[0], maxs[1], maxs[2]],
    [maxs[0], maxs[1], maxs[2]],
])

# Project onto camera plane
px = corners @ cam_x    # x in image plane
py = corners @ cam_y    # y in image plane

# Convert to pixel indices
ix = ((px + half_w) / (2 * half_w) * resolution_x).astype(int)
iy = ((py + half_h) / (2 * half_h) * resolution_y).astype(int)

pad   = 5
x_min = max(0,             ix.min() - pad)
x_max = min(resolution_x,  ix.max() + pad)
y_min = max(0,             iy.min() - pad)
y_max = min(resolution_y,  iy.max() + pad)

n_rays_culled = (x_max - x_min) * (y_max - y_min)
n_rays_full   = resolution_x * resolution_y
print(f"\nBbox cull: {n_rays_culled} rays vs {n_rays_full} full grid "
      f"({100*n_rays_culled/n_rays_full:.1f}%)")
print(f"Pixel region: x=[{x_min},{x_max}]  y=[{y_min},{y_max}]")

# =============================================================================
# Build ray grid within bbox
# =============================================================================

xs = np.linspace(-half_w, half_w, resolution_x)[x_min:x_max]
ys = np.linspace(-half_h, half_h, resolution_y)[y_min:y_max]
xx, yy = np.meshgrid(xs, ys)
xx = xx.ravel()
yy = yy.ravel()

# Ray origins on image plane, ray direction toward scene
origins    = cam_pos[None, :] + xx[:, None] * cam_x[None, :] + yy[:, None] * cam_y[None, :]
directions = np.tile(cam_dir, (len(origins), 1)).astype(float)

print(f"\norigins shape:    {origins.shape}")
print(f"directions shape: {directions.shape}")
print(f"origins sample:   {origins[0]}")

# =============================================================================
# Primary ray cast
# =============================================================================

t0 = time.time()
locs, index_ray, index_tri = mesh.ray.intersects_location(
    ray_origins=origins, ray_directions=directions, multiple_hits=False
)
t1 = time.time()
print(f"\nPrimary rays : {t1-t0:.3f}s  hits: {len(locs)}/{n_rays_culled}")

if len(locs) == 0:
    print("No hits — check camera setup")
else:
    # Which object was hit
    hit_obj_ids = face_object_ids[index_tri]
    for idx, name in enumerate(object_names):
        count = (hit_obj_ids == idx).sum()
        if count > 0:
            print(f"  {name}: {count} hits")

    # =============================================================================
    # Shadow ray cast
    # =============================================================================

    normals        = mesh.face_normals[index_tri]
    shadow_origins = locs + normals * 1e-4
    shadow_dirs    = np.tile(sun_dir, (len(locs), 1))

    t2 = time.time()
    in_shadow = mesh.ray.intersects_any(
        ray_origins=shadow_origins, ray_directions=shadow_dirs
    )
    t3 = time.time()
    print(f"Shadow rays  : {t3-t2:.3f}s  in shadow: {in_shadow.sum()}/{len(locs)}")
    print(f"Total        : {(t1-t0)+(t3-t2):.3f}s")

    # =============================================================================
    # Simple Lambertian flux (sanity check)
    # =============================================================================

    solar_constant   = 1361.0   # W/m^2
    distance_obs_m   = 35786e3 * 1e3   # GEO in m
    pixel_size       = (2 * half_w / resolution_x) * (2 * half_h / resolution_y)

    flux_total = 0.0
    lit_mask   = ~in_shadow
    if lit_mask.sum() > 0:
        hit_normals = normals[lit_mask]
        cos_sun     = np.clip(hit_normals @ sun_dir, 0, None)
        cos_view    = np.clip(hit_normals @ (-cam_dir), 0, None)
        brdf        = (1.0 / np.pi) * cos_sun          # Lambertian, albedo=1
        radiance    = brdf * solar_constant             # W/m^2/sr
        proj_area   = pixel_size * cos_view             # m^2
        flux        = radiance * proj_area / distance_obs_m**2
        flux_total  = flux.sum()

    mag = -2.5 * np.log10(flux_total) - 18.0 if flux_total > 0 else np.nan
    print(f"\nFlux  : {flux_total:.4e} W/m^2")
    print(f"Mag   : {mag:.2f}")