from importlib.resources import files
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
import matplotlib.pyplot as plt
from hide_warnings import hide_warnings
import plotly.graph_objects as go
import trimesh
from satlight.brdf import (
    lambertian, phong, blinn_phong,
    cook_torrance, oren_nayar, satellite)

BRDF_REGISTRY = {
    'lambertian':   lambertian,
    'phong':        phong,
    'blinn_phong':  blinn_phong,
    'cook_torrance': cook_torrance,
    'oren_nayar':   oren_nayar,
}

class Renderer:
    """ To manage and render the object in blender and perform the ray-casting."""

    def __init__(self, obj_path, sun_direction, observer_direction, distance_to_observer, resolution=(500,500),solar_constant = 1460, add_earthshine=False, brdf='lambertian'):
        '''
        obj_path can be either .obj or .stl
        distance_to_observer in km
        Resolution in (x,y)
        sun and obs_directions are normalised

        '''
        # Set constants
        self.obj_path = obj_path
        self.resolution_x, self.resolution_y = resolution
        self.aspect_ratio = self.resolution_x / self.resolution_y # width / height
        self.solar_constant = solar_constant # in W
        self.distance_to_observer = distance_to_observer # in m
        self.sun_direction = sun_direction # may need to change to Vector(sun_direction) depending on geometry output
        self.observer_direction = observer_direction
        self.nadir_direction = np.array([0., 0., -1.])
        self.earthshine_irradiance = 0.0
        self.earthshine_direction  = None 
        self.add_earthshine = add_earthshine

        # Set up BRDFs
        if isinstance(brdf, dict):
            self.brdf_map = brdf
        else:
            self.brdf_map = {'default': brdf}

    def initialise_scene(self):
        # Set-up scene
        bpy.ops.wm.read_factory_settings(use_empty=True)
        self.ortho_scale = self._load_and_center_object(self.obj_path)
        self.add_camera()


    @hide_warnings()
    def _load_and_center_object(self, obj_path):
        '''Determine file type and import'''
        if obj_path.endswith('.obj'):
            self.obj_type = '.obj'
            bpy.ops.wm.obj_import(filepath=obj_path, filter_obj=True, use_split_objects=True, use_split_groups=False)
        elif obj_path.endswith('.stl'):
            self.obj_type = '.stl'
            bpy.ops.wm.stl_import(filepath=obj_path)
        else:
            raise ValueError(f"Unsupported file type: {obj_path}")
        
        # Store imported objects
        self.imported_objects = bpy.context.selected_objects
        
        # Calculate geometric center
        mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']

        # Compute bounding box of all combined meshes
        bpy.context.view_layer.update()
        all_corners = []
        for obj in mesh_objects:
            all_corners.extend([obj.matrix_world @ Vector(corner) for corner in obj.bound_box])

        min_corner = Vector(map(min, zip(*all_corners)))
        max_corner = Vector(map(max, zip(*all_corners)))
        center = (min_corner + max_corner) / 2
        
        extents = max_corner - min_corner
        ortho_scale = max(extents.x, extents.y, extents.z) * 1.2

        # Move all meshes so their center is at origin
        for obj in mesh_objects:
            obj.location -= center
        
        return ortho_scale

    def add_camera(self):
        import mathutils as mu

        self.camera_object = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
        self.camera_object.data.type = 'ORTHO'
        bpy.context.collection.objects.link(self.camera_object)
        bpy.context.scene.camera = self.camera_object
        self.camera_object.data.ortho_scale = self.ortho_scale

        self.camera_object.location = mu.Vector((0.0, 0.0, -1.0)) * self.ortho_scale

        # Camera looks toward origin: forward = -obs
        # Camera +Z (backward) = obs
        # right = X (along-track), up = Y (cross-track)
        forward = mu.Vector((0.0, 0.0, 1.0))   # always looking up at satellite
        right   = mu.Vector((1.0, 0.0, 0.0))
        up      = mu.Vector((0.0, 1.0, 0.0))

        # Gram-Schmidt: ensure right is orthogonal to forward
        right = (right - right.dot(forward) * forward).normalized()
        up    = forward.cross(right).normalized()

        # Rotation matrix rows = camera basis vectors in world space
        # Row 0: camera X (right), Row 1: camera Y (up), Row 2: camera Z (backward = -forward)
        rot = mu.Matrix((
            [ right.x,    right.y,    right.z   ],
            [ up.x,       up.y,       up.z      ],
            [-forward.x, -forward.y, -forward.z ],
        )).transposed()

        self.camera_object.rotation_euler = rot.to_euler()

        self.pixel_size_x = self.ortho_scale / self.resolution_x
        self.pixel_size_y = self.ortho_scale / self.aspect_ratio / self.resolution_y
        self.pixel_area   = self.pixel_size_x * self.pixel_size_y
    
    def get_object_pixel_bboxes(self):
        """Return list of (x_min, x_max, y_min, y_max) per mesh object."""
        bpy.context.view_layer.update()
        cam_inv = self.camera_object.matrix_world.inverted()
        bboxes  = []
        pad     = 2

        for obj in bpy.context.scene.objects:
            if obj.type != 'MESH':
                continue
            xs, ys = [], []
            for corner in obj.bound_box:
                wco   = obj.matrix_world @ Vector(corner)
                cco   = cam_inv @ wco
                x_ndc = cco.x / (self.ortho_scale / 2)
                y_ndc = cco.y / (self.ortho_scale / 2 / self.aspect_ratio)
                xs.append(int((x_ndc + 1) / 2 * self.resolution_x))
                ys.append(int((1 - y_ndc) / 2 * self.resolution_y))
            bboxes.append((
                max(0,                 min(xs) - pad),
                min(self.resolution_x, max(xs) + pad),
                max(0,                 min(ys) - pad),
                min(self.resolution_y, max(ys) + pad),
            ))
        return bboxes
    
    def build_bvhs(self):
        """
        Build one BVH tree per mesh object from current (evaluated) geometry.
        Call this once per frame, AFTER attitude/panel rotations are applied
        and bpy.context.view_layer.update() has been called — NOT per-pixel.
        """
        depsgraph = bpy.context.evaluated_depsgraph_get()
        self.bvh_trees = {}

        for obj in bpy.context.scene.objects:
            if obj.type != 'MESH':
                continue

            eval_obj = obj.evaluated_get(depsgraph)
            mesh = eval_obj.to_mesh()

            # Transform vertices to world space (so all BVHs share one coordinate frame)
            verts_world = [obj.matrix_world @ v.co for v in mesh.vertices]
            polys = [list(p.vertices) for p in mesh.polygons]

            self.bvh_trees[obj.name] = BVHTree.FromPolygons(verts_world, polys)

            eval_obj.to_mesh_clear()

    def _ray_cast_all(self, origin, direction):
        """
        Cast a ray against all object BVHs, return the closest hit.

        Returns
        -------
        (hit: bool, location: Vector or None, normal: Vector or None, obj_name: str or None)
        """
        best_dist = float('inf')
        best_loc, best_normal, best_name = None, None, None

        for name, bvh in self.bvh_trees.items():
            loc, normal, idx, dist = bvh.ray_cast(origin, direction)
            if loc is not None and dist < best_dist:
                best_dist = dist
                best_loc, best_normal, best_name = loc, normal, name

        return (best_loc is not None, best_loc, best_normal, best_name)

    def preview_lvlh_views(self, body_rotation=None):
        """
        Interactive 3D plotly preview of the spacecraft in the LVLH frame.

        Dashed arrows show where the model's own X/Y/Z axes land in LVLH after
        body_rotation is applied — use these to verify alignment before committing
        to attitude settings.

        Parameters
        ----------
        body_rotation : tuple (rx, ry, rz) in radians, optional
            Euler-XYZ body attitude to preview. None shows the raw imported orientation.
        """
        import math

        try:
            import mathutils as mu
            if body_rotation is None:
                rot_np = np.eye(3)
            elif isinstance(body_rotation, (tuple, list)):
                rot_np = np.array(mu.Euler(body_rotation, 'XYZ').to_matrix())
            elif hasattr(body_rotation, 'to_matrix'):
                rot_np = np.array(body_rotation.to_matrix())
            else:
                rot_np = np.array(body_rotation)
        except ImportError:
            if body_rotation is None:
                rot_np = np.eye(3)
            else:
                from scipy.spatial.transform import Rotation
                rot_np = Rotation.from_euler('xyz', body_rotation).as_matrix()

        # Blender's OBJ importer swaps Y and Z. Apply same correction to trimesh
        blender_correction = np.eye(3)

        # Full transform: correct axes first, then apply body rotation
        full_rot = rot_np @ blender_correction

        raw = trimesh.load(self.obj_path, force=None, process=False)
        geometries = raw.geometry if hasattr(raw, 'geometry') else {'model': raw}

        all_verts  = np.vstack([g.vertices for g in geometries.values()])
        centroid   = (all_verts.min(axis=0) + all_verts.max(axis=0)) / 2
        max_extent = np.max(all_verts.max(axis=0) - all_verts.min(axis=0)) / 2
        pad        = max_extent * 1.4
        arrow_len  = max_extent * 1.2
        cone_size  = max_extent * 0.08

        # ── 5. Sun / observer directions (fixed in LVLH, never rotated) ─────────
        sun_dir = np.array(self.sun_direction,      dtype=float)
        obs_dir = np.array(self.observer_direction, dtype=float)

        fig = go.Figure()

        # draw labelled arrow
        def _arrow(direction, label, color, width=5, dash=None):
            d   = np.asarray(direction, dtype=float)
            d   = d / np.linalg.norm(d)
            tip = d * arrow_len
            cs  = [[0, color], [1, color]]
            lkw = dict(color=color, width=width)
            if dash:
                lkw['dash'] = dash
            fig.add_trace(go.Scatter3d(
                x=[0, tip[0]], y=[0, tip[1]], z=[0, tip[2]],
                mode='lines', line=lkw,
                showlegend=False, hoverinfo='skip',
            ))
            fig.add_trace(go.Cone(
                x=[tip[0]], y=[tip[1]], z=[tip[2]],
                u=[d[0]], v=[d[1]], w=[d[2]],
                sizeref=cone_size, colorscale=cs,
                showscale=False, hovertext=label, hoverinfo='text',
            ))

        # Mesh — rotated by full_rot (body rotation + blender axis correction)
        for name, mesh in geometries.items():
            verts = (full_rot @ (mesh.vertices - centroid).T).T
            fig.add_trace(go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2],
                color='rgb(170, 140, 173)',
                flatshading=False, opacity=1.0,
                lighting=dict(ambient=0.4, diffuse=0.8, specular=0.3,
                            roughness=0.5, fresnel=0.2),
                lightposition=dict(x=sun_dir[0]*10000,
                                y=sun_dir[1]*10000,
                                z=sun_dir[2]*10000),
                name=name, showlegend=True, hoverinfo='skip',
            ))

            # Wireframe overlay — extract unique triangle edges and draw as a grid
            edges = set()
            for f in mesh.faces:
                for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                    edges.add((a, b) if a < b else (b, a))

            edge_x, edge_y, edge_z = [], [], []
            for a, b in edges:
                edge_x += [verts[a, 0], verts[b, 0], None]
                edge_y += [verts[a, 1], verts[b, 1], None]
                edge_z += [verts[a, 2], verts[b, 2], None]

            fig.add_trace(go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(color='rgba(0,0,0,0.5)', width=1),
                showlegend=False, hoverinfo='skip',
            ))

        # LVLH axes — fixed, never rotated
        _arrow([1, 0, 0], "Along-Track (X)", "rgb(220,  50,  50)")
        _arrow([0, 1, 0], "Orbit Normal (Y)", "rgb( 50, 160,  50)")
        _arrow([0, 0, 1], "Zenith (Z)",       "rgb( 50,  80, 220)")

        # Sun and observer — fixed in LVLH, never rotated
        _arrow(sun_dir, "Sun",      "rgb(255, 220,   0)")
        _arrow(obs_dir, "Observer", "rgb(180, 180, 180)")

        ax_style = dict(
            title="", showbackground=False, showticklabels=False,
            showgrid=True, gridcolor="rgba(200,200,200,0.4)",
            zeroline=False, range=[-pad, pad],
        )

        if body_rotation is not None and isinstance(body_rotation, (tuple, list)):
            rx, ry, rz = [math.degrees(a) for a in body_rotation]
            title_str = f"LVLH Preview — body rotation  Rx={rx:.0f}°  Ry={ry:.0f}°  Rz={rz:.0f}°"
        else:
            title_str = "LVLH Preview — raw imported orientation (no rotation)"

        fig.update_layout(
            title=dict(text=title_str, x=0.5, font=dict(size=16)),
            width=950, height=750, paper_bgcolor="white",
            scene=dict(
                xaxis=ax_style, yaxis=ax_style, zaxis=ax_style,
                aspectmode='cube', bgcolor="white",
                camera=dict(
                        eye=dict(x=-1.5, y=0, z=-1.5),
                        up=dict(x=0, y=1, z=0),
                        center=dict(x=0, y=0, z=0),
                    ),
            ),
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(x=0.99, y=0.99,
                        bgcolor="rgba(255,255,255,0.85)",
                        bordercolor="rgba(200,200,200,0.5)", borderwidth=1),
            annotations=[dict(
                text=(
                    '<b>LVLH axes (solid)</b><br>'
                    '<span style="color:rgb(220,50,50)">&#9632; X = Along-Track</span><br>'
                    '<span style="color:rgb(50,160,50)">&#9632; Y = Orbit Normal</span><br>'
                    '<span style="color:rgb(50,80,220)">&#9632; Z = Zenith</span><br>'
                    '<span style="color:rgb(255,220,0)">&#9632; Sun</span><br>'
                    '<span style="color:rgb(180,180,180)">&#9632; Observer</span><br>'
                ),
                xref="paper", yref="paper", x=0.02, y=0.98,
                showarrow=False, font=dict(size=13), align="left",
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="rgba(200,200,200,0.5)", borderwidth=1,
            )],
        )

        fig.show()

    def _build_patch_template(self, n_rings=40):
        """Build (and cache) the static Driscoll-Healy patch grid + per-patch CERES albedo.
        Independent of time/geometry, so only needs to run once."""
        if hasattr(self, '_patch_template') and self._patch_template is not None:
            return self._patch_template

        R_EARTH = 6371.0
        albedo_grid = self._load_ceres_albedo()

        sin_edges    = np.linspace(-1, 1, n_rings + 1)
        sin_mids     = (sin_edges[:-1] + sin_edges[1:]) / 2
        phi_mids     = np.arcsin(sin_mids)
        d_sinphi_arr = np.diff(sin_edges)
        n_lon        = 2 * n_rings
        d_lon        = 2 * np.pi / n_lon
        lons         = (np.arange(n_lon) + 0.5) * d_lon

        cs, As = [], []
        for phi, dsp in zip(phi_mids, d_sinphi_arr):
            cl = np.cos(phi)
            cs.append(np.stack([
                R_EARTH * cl * np.cos(lons),
                R_EARTH * cl * np.sin(lons),
                np.full(n_lon, R_EARTH * np.sin(phi))
            ], axis=1))
            As.append(np.full(n_lon, R_EARTH**2 * dsp * d_lon))

        centers_gcrs = np.vstack(cs)
        areas        = np.concatenate(As)

        # NOTE: lat/lon here are taken directly from GCRS x,y,z without correcting
        # for Earth's rotation (GMST), same simplification as test_earthshine_leo.py.
        # Fine for a demonstrator; flag as future work if it comes up.
        x, y, z  = centers_gcrs[:, 0], centers_gcrs[:, 1], centers_gcrs[:, 2]
        lats_deg = np.degrees(np.arcsin(z / R_EARTH))
        lons_deg = np.degrees(np.arctan2(y, x))

        lat_idx = np.clip(np.floor(90 - lats_deg).astype(int), 0, 179)
        lon_idx = np.clip(np.floor(lons_deg + 180).astype(int), 0, 359)
        albedos = albedo_grid[lat_idx, lon_idx]

        self._patch_template = dict(centers_gcrs=centers_gcrs, areas=areas, albedos=albedos)
        return self._patch_template

        
    def compute_earthshine(self, geometry, n_rings=40):
        """
        Compute earthshine irradiance + an effective source direction using
        CERES monthly albedo per patch. Call once per frame (geometry changes
        each timestep) — the static patch grid + albedo lookup is cached.
        """
        R_EARTH = 6371.0
        template = self._build_patch_template(n_rings)
        centers_gcrs, areas, albedos = template['centers_gcrs'], template['areas'], template['albedos']

        R_mat = np.stack([
            geometry.along_track,
            geometry.cross_track,
            -geometry.nadir
        ], axis=0)

        sat_pos_gcrs = geometry.positions['satellite']

        rel          = centers_gcrs - sat_pos_gcrs
        centers_lvlh = (R_mat @ rel.T).T

        earth_centre_lvlh = R_mat @ (np.zeros(3) - sat_pos_gcrs)
        norms_lvlh = centers_lvlh - earth_centre_lvlh
        norms_lvlh /= np.linalg.norm(norms_lvlh, axis=1, keepdims=True)

        sun_lvlh = geometry.incident_vector_lvlh
        sat_lvlh = np.zeros(3)

        to_sat  = sat_lvlh - centers_lvlh
        dist_km = np.linalg.norm(to_sat, axis=1)
        cos_sat = np.sum(norms_lvlh * (to_sat / dist_km[:, None]), axis=1)
        cos_sun = norms_lvlh @ sun_lvlh
        mask    = (cos_sun > 0) & (cos_sat > 0)

        if not np.any(mask):
            self.earthshine_irradiance = 0.0
            self.earthshine_direction  = None
            return

        dE = (albedos[mask] / np.pi) * self.solar_constant \
            * cos_sun[mask] * cos_sat[mask] \
            * (areas[mask] * 1e6) / (dist_km[mask] * 1e3)**2

        self.earthshine_irradiance = float(np.sum(dE))

        # Weighted effective direction FROM satellite TOWARD the illuminating Earth
        # (same convention as sun_direction — a unit vector pointing at the source)
        dirs = centers_lvlh[mask] / dist_km[mask][:, None]
        weighted_dir = np.sum(dE[:, None] * dirs, axis=0)
        self.earthshine_direction = weighted_dir / np.linalg.norm(weighted_dir)

        print(f"Earthshine irradiance: {self.earthshine_irradiance:.4f} W/m²  "
            f"(effective direction: {self.earthshine_direction.round(3)})")

    def _load_ceres_albedo(self):
        """Load and cache the CERES monthly albedo grid (lat x lon)."""
        if not hasattr(self, '_albedo_grid') or self._albedo_grid is None:
            path = files("satlight.data").joinpath("ceres_albedo_monthly.npy")
            self._albedo_grid = np.load(path)
        return self._albedo_grid 

    def _resolve(self, spec):
        """Turn a brdf_spec (str / callable / tuple) into (function, kwargs)."""
        if isinstance(spec, tuple):
            func_or_name, kwargs = spec
        else:
            func_or_name, kwargs = spec, {}
        func = BRDF_REGISTRY[func_or_name] if isinstance(func_or_name, str) else func_or_name
        return func, kwargs

    def _get_for_object(self, obj_name):
        """Look up the right BRDF function + kwargs for a hit object's name."""
        for key, spec in self.brdf_map.items():
            if key != 'default' and key.lower() in obj_name.lower():
                return self._resolve(spec)
        return self._resolve(self.brdf_map.get('default', 'lambertian'))

    
    def update(self, sun_direction, observer_direction, distance_to_observer):
        """Update geometry for next timestep — call before render()."""
        import mathutils as mu
        
        self.sun_direction        = sun_direction
        self.observer_direction   = observer_direction
        self.distance_to_observer = distance_to_observer

        obs = observer_direction.normalized()
        self.camera_object.location = obs * self.ortho_scale

        # Vector from camera to origin
        to_origin = mu.Vector((0,0,0)) - self.camera_object.location
        to_origin = to_origin.normalized()

        # Build rotation: camera -Z points toward origin
        abs_comps  = [abs(to_origin.x), abs(to_origin.y), abs(to_origin.z)]
        world_axes = [mu.Vector((1,0,0)), mu.Vector((0,1,0)), mu.Vector((0,0,1))]
        ref_axis   = world_axes[abs_comps.index(min(abs_comps))]  # least-parallel axis to to_origin
        right = to_origin.cross(ref_axis).normalized()
        up    = right.cross(to_origin).normalized()

        rot = mu.Matrix((
            [right.x,      right.y,      right.z     ],
            [up.x,         up.y,         up.z        ],
            [-to_origin.x, -to_origin.y, -to_origin.z],
        )).transposed()

        self.camera_object.rotation_euler = rot.to_euler()
        
    def render(self):
        # Build BVH trees once for this frame (NOT per-pixel)
        self.build_bvhs()

        cam_matrix = self.camera_object.matrix_world
        ray_dir_world = (cam_matrix.to_3x3() @ Vector((0, 0, -1))).normalized()
        ray_dir = np.array(ray_dir_world)

        self.image = np.zeros((self.resolution_y, self.resolution_x))
        self.hit_count = 0
        self.shadow_count = 0

        bboxes = self.get_object_pixel_bboxes()
        for x_min, x_max, y_min, y_max in bboxes:
            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    screen_x = (2.0 * x / self.resolution_x) - 1.0
                    screen_y = 1.0 - (2.0 * y / self.resolution_y)

                    ray_origin_cam = Vector((
                        screen_x * (self.ortho_scale / 2),
                        screen_y * (self.ortho_scale / 2) / self.aspect_ratio,
                        0.0
                    ))
                    ray_origin_world = cam_matrix @ ray_origin_cam

                    result, location, normal, obj_name = self._ray_cast_all(ray_origin_world, ray_dir_world)

                    if result:
                        self.hit_count += 1
                        hit_point  = np.array(location)
                        hit_normal = np.array(normal)

                        brdf_func, brdf_kwargs = self._get_for_object(obj_name)

                        shadow_origin = hit_point + hit_normal * 1e-4
                        shadow_result, *_ = self._ray_cast_all(Vector(shadow_origin.tolist()), self.sun_direction)
                        view_dir = -ray_dir

                        flux = 0
                        if not shadow_result:
                            sun_dir_np = np.array(self.sun_direction, dtype=float)
                            brdf_val = brdf_func(hit_normal, sun_dir_np, view_dir, **brdf_kwargs)
                            radiance = brdf_val * self.solar_constant
                            cos_theta = max(0, np.dot(hit_normal, view_dir))
                            self.projected_area = self.pixel_area * cos_theta
                            flux += radiance * self.projected_area / self.distance_to_observer**2

                        if self.add_earthshine and self.earthshine_direction is not None:
                            es_dir_np = np.array(self.earthshine_direction, dtype=float)
                            brdf_val_es = brdf_func(hit_normal, es_dir_np, view_dir, **brdf_kwargs)
                            radiance_es = brdf_val_es * self.earthshine_irradiance
                            cos_theta_es = max(0, np.dot(hit_normal, view_dir))
                            flux += radiance_es * self.pixel_area * cos_theta_es / self.distance_to_observer**2

                        self.image[y, x] = flux

                    
                edge_pixels = np.concatenate([
                self.image[0, :].ravel(),   # top row
                self.image[-1, :].ravel(),  # bottom row
                self.image[:, 0].ravel(),   # left column
                self.image[:, -1].ravel(),  # right column
                ])
                edge_threshold = 1e-6  # tweak to your noise floor
                if np.any(edge_pixels > edge_threshold):
                    print(f"  WARNING: illuminated pixels detected at frame edge "
                        f"(max edge value {edge_pixels.max():.4g}) — spacecraft may be clipped out of "
                        f"the field of view. Increase `resolution` / camera FOV and re-render.")

        return self.image

