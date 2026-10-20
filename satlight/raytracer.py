from importlib.resources import files
import bpy
import numpy as np
from mathutils import Vector
import matplotlib.pyplot as plt
from hide_warnings import hide_warnings
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'browser'
from types import SimpleNamespace
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

# The viewing geometry used to be three loose arguments. They are gone: they
# duplicated what a Geometry already knows, and put the km -> m conversion on the
# range in every caller, where dropping it is a factor-1000 error in flux
# (7.5 magnitudes) that still produces a perfectly normal-looking image.
_REMOVED_GEOMETRY_ARGS = ('sun_direction', 'observer_direction',
                          'distance_to_observer')


def _reject_legacy(kwargs, where):
    """Turn the removed arguments into a pointed error instead of a silent no-op.

    `**legacy` exists only so they can be named in the message; without it they
    would be swallowed as unexpected keywords and the geometry silently ignored.
    """
    used = [k for k in _REMOVED_GEOMETRY_ARGS if k in kwargs]
    if used:
        raise TypeError(
            f'{where} no longer takes {", ".join(used)}; pass geometry= instead. '
            f'For an orbit use a Geometry; otherwise '
            f'ViewGeometry(sun_lvlh, observer_lvlh, distance_km=...). '
            f'NOTE distance_km is in KILOMETRES, where the old '
            f'distance_to_observer was in metres.')
    if kwargs:
        raise TypeError(f'{where} got unexpected keyword argument(s): '
                        f'{", ".join(sorted(kwargs))}')


def _require_geometry(geometry, where):
    if geometry is None:
        raise TypeError(
            f'{where} requires geometry=. Pass a Geometry, or a '
            f'ViewGeometry(sun_lvlh, observer_lvlh, distance_km=...) when there '
            f'is no orbit behind the render.')

class Renderer:
    """ To manage and render the object in blender and perform the ray-casting."""
    def __init__(self, obj_path, geometry=None, *,
                 resolution=(500, 500), solar_constant=1361.0,
                 brdf='lambertian', add_shadows=True,
                 smooth_normals=False, smooth_angle_deg=30.0,
                 add_earthshine=False, earth_albedo=0.3,
                 earthshine_n_rings=4, earthshine_shadows=None,
                 n_bounce=1, bounce_samples=8, bounce_seed=0,
                 auto_ortho_scale=True, ortho_margin=1.06,
                 **legacy):
        '''
        geometry : Geometry or ViewGeometry
            REQUIRED. A `Geometry` for orbit-driven work, a `ViewGeometry` for
            BRDF studies, phase sweeps and attitude scans. Only
            `incident_vector_lvlh`, `outgoing_vector_lvlh` and `prop_distance`
            are read, so anything exposing those will do. `prop_distance` is in
            KILOMETRES; `_frame_from_geometry` does the conversion, once.
        obj_path : .obj or .stl.
        resolution : (x, y) pixels.
        solar_constant : W/m^2.
        brdf : one spec for the body, or {object_name: spec, 'default': spec}.
        add_shadows : let the body shadow itself; one shadow ray per lit pixel.
        smooth_normals, smooth_angle_deg : interpolate shading normals across
            edges meeting at less than the angle. Redistributes light over the
            disk, does not change the total.
        add_earthshine : add reflected earthshine, as an extended source.
        earth_albedo : uniform Bond albedo per Earth patch. 0.3 is the global
            mean; None restores the per-patch CERES monthly map.
        earthshine_n_rings : Earth-cap resolution, equal SOLID ANGLE per patch.
            4 gives 88 patches and already matches a 960-patch grid to 0.001
            mag. Cost is linear in the patch count.
        earthshine_shadows : whether the spacecraft occludes its own earthshine
            (a nadir panel shadowed by its stand-offs). None follows
            `add_shadows`. Expensive: one shadow ray per patch per pixel.
        bounce_samples, bounce_seed : cosine-weighted hemisphere samples per
            scattering event, and the seed that makes them reproducible. Noise
            goes as 1/sqrt(N); 8 is enough unless the indirect term is a large
            fraction of the total.
        ortho_margin : safety factor on the fitted frame; 1.06 = a 6% border.

        n_bounce
            Scattering events per path. 1 (default) is single-bounce and is
            EXACTLY the old behaviour, at no extra cost. 2 adds one
            inter-reflection, which matters for concave geometry that sees
            itself - corrugations, louvres, antenna stand-offs, box cavities -
            and captures most of a series converging like (rho * F)^k. COST is
            `bounce_samples ** (n_bounce - 1)` extra rays per lit pixel: 2 is
            cheap-ish, 3 is 64x, 4 is not worth attempting.

        auto_ortho_scale
            Re-fit the orthographic frame to the body's PROJECTED extent on
            every `update` (default). False restores the old fixed bounding-box
            DIAGONAL, which is sized for the worst-case attitude and wastes
            resolution on every other one: viewed end-on, Skynet 5D lands on
            69 px instead of 469 and reads 0.070 mag too faint. That error
            moves WITH attitude, so in a light curve it can masquerade as real
            structure.
        '''
        _reject_legacy(legacy, 'Renderer')
        _require_geometry(geometry, 'Renderer')
        sun, obs, dist_m = self._frame_from_geometry(geometry)

        # --- viewing geometry
        self.obj_path = obj_path
        self.sun_direction = sun
        self.observer_direction = obs
        self.distance_to_observer = dist_m                  # metres
        self.nadir_direction = np.array([0., 0., -1.])

        # --- sensor and framing
        self.resolution_x, self.resolution_y = resolution
        self.aspect_ratio = self.resolution_x / self.resolution_y   # width / height
        self.solar_constant = solar_constant                        # W/m^2
        self.auto_ortho_scale = bool(auto_ortho_scale)
        self.ortho_margin = float(ortho_margin)

        # --- shading
        self.add_shadows = add_shadows
        self.smooth_normals = smooth_normals
        self.smooth_angle_deg = smooth_angle_deg
        self._smooth_cache = {}
        self.brdf_map = brdf if isinstance(brdf, dict) else {'default': brdf}

        # --- earthshine. `earthshine_irradiance` and `earthshine_direction` are
        # the legacy single-source form, still populated for callers that read
        # them; the per-patch arrays are the extended-source form `render`
        # actually uses, in the SAME frame as `sun_direction`.
        self.add_earthshine = add_earthshine
        self.earth_albedo = earth_albedo
        self.earthshine_n_rings = earthshine_n_rings
        self.earthshine_shadows = (add_shadows if earthshine_shadows is None
                                   else earthshine_shadows)
        self.earthshine_irradiance = 0.0
        self.earthshine_direction = None
        self.earthshine_dirs = None
        self.earthshine_patch_irradiance = None
        self._earth_cap_cache = {}

        # --- multiple scattering
        self.n_bounce = int(n_bounce)
        self.bounce_samples = int(bounce_samples)
        self.bounce_seed = int(bounce_seed)
        self._bounce_rng = np.random.default_rng(self.bounce_seed)

    @staticmethod
    def _frame_from_geometry(geometry):
        """(sun, observer, distance_m) from a Geometry, for a SINGLE epoch.

        The km -> m conversion on the range lives here rather than in every
        caller: `Geometry.prop_distance` is in kilometres while the renderer
        works in metres, and that mismatch is a factor-1000 error in flux, i.e.
        7.5 magnitudes, with nothing to show for it in the image.
        """
        sun = np.asarray(geometry.incident_vector_lvlh, dtype=float)
        obs = np.asarray(geometry.outgoing_vector_lvlh, dtype=float)
        dist = np.asarray(geometry.prop_distance, dtype=float)
        if sun.ndim > 1 or obs.ndim > 1 or dist.ndim > 0:
            n = dist.size if dist.ndim else max(sun.shape[-1], obs.shape[-1])
            raise ValueError(
                f'Renderer renders one frame at a time, but this Geometry holds '
                f'{n} epochs. Index it (geom_at(i)) or loop over epochs, calling '
                f'update(geometry=...) per frame.')
        return sun, obs, float(dist) * 1e3

    def initialise_scene(self):
        # Set-up scene
        bpy.ops.wm.read_factory_settings(use_empty=True)
        self.ortho_scale = self._load_and_center_object(self.obj_path)
        if self.smooth_normals:
            self._repair_normals(self.smooth_angle_deg)
        self.add_camera()

    def _repair_normals(self, angle_deg=30.0):
        """Discard authored split normals and recompute them from geometry.
        """
        for o in list(bpy.context.scene.objects):
            if o.type != 'MESH':
                continue
            for other in bpy.context.scene.objects:
                other.select_set(False)
            bpy.context.view_layer.objects.active = o
            o.select_set(True)
            try:
                bpy.ops.mesh.customdata_custom_splitnormals_clear()
            except Exception:
                pass
            try:
                bpy.ops.object.shade_smooth_by_angle(angle=np.radians(angle_deg))
            except Exception:
                for p in o.data.polygons:            # older Blender fallback
                    p.use_smooth = True
                if hasattr(o.data, 'use_auto_smooth'):
                    o.data.use_auto_smooth = True
                    o.data.auto_smooth_angle = np.radians(angle_deg)
            o.select_set(False)
        self._smooth_cache = {}

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

        # Move all meshes so their center is at origin
        for obj in mesh_objects:
            obj.location -= center

        bpy.context.view_layer.update()
        radius = max((obj.matrix_world @ Vector(corner)).length
                     for obj in mesh_objects
                     for corner in obj.bound_box)
        
        return 2.0 * radius * 1.05
        

    def add_camera(self):
        """Create the orthographic camera and aim it from the current geometry."""
        self.camera_object = bpy.data.objects.new(
            'Camera', bpy.data.cameras.new('Camera'))
        self.camera_object.data.type = 'ORTHO'
        bpy.context.collection.objects.link(self.camera_object)
        bpy.context.scene.camera = self.camera_object
        self._set_pixel_scale()
        self._place_camera(self.observer_direction)

    def _set_pixel_scale(self):
        """Push `ortho_scale` to the camera and re-derive the pixel footprint.

        `pixel_area` IS the radiometry - per-pixel flux scales with it - so it
        must never be left stale behind a change of frame width.
        """
        self.camera_object.data.ortho_scale = self.ortho_scale
        self.pixel_size_x = self.ortho_scale / self.resolution_x
        self.pixel_size_y = self.ortho_scale / self.aspect_ratio / self.resolution_y
        self.pixel_area = self.pixel_size_x * self.pixel_size_y

    def _place_camera(self, observer_direction):
        """Aim the camera at the assembly from `observer_direction`, then fit the
        frame and the standoff to what it actually sees.

        The single camera-placement path. `add_camera` and `update` both need
        exactly this, and while it lived in two copies they disagreed:
        `add_camera` hardcoded the -Z axis, so `initialise_scene -> add_camera
        -> render` silently rendered from NADIR whatever geometry it was given,
        and for an oblique observer that meant looking into the back of the
        model and returning zero flux.
        """
        import mathutils as mu

        obs = mu.Vector(
            np.asarray(observer_direction, dtype=float).tolist()).normalized()
        self.view_target = self._assembly_center()
        self.camera_object.location = self.view_target + obs * self.ortho_scale

        # Camera -Z points at the target. `right` and `up` are built from the
        # world axis LEAST parallel to the view, so the basis cannot degenerate
        # whichever direction the observer sits in.
        to_target = (self.view_target - self.camera_object.location).normalized()
        comps = [abs(to_target.x), abs(to_target.y), abs(to_target.z)]
        axes = [mu.Vector((1, 0, 0)), mu.Vector((0, 1, 0)), mu.Vector((0, 0, 1))]
        ref = axes[comps.index(min(comps))]
        right = to_target.cross(ref).normalized()
        up = right.cross(to_target).normalized()
        self.camera_object.rotation_euler = mu.Matrix((
            [right.x,      right.y,      right.z     ],
            [up.x,         up.y,         up.z        ],
            [-to_target.x, -to_target.y, -to_target.z],
        )).transposed().to_euler()

        self._fit_frame_and_standoff(obs)

    def _fit_frame_and_standoff(self, obs):
        """Frame width from the PROJECTED extent, standoff from the ALONG-VIEW
        extent. Two unrelated quantities that used to be one - see
        `_safe_standoff`."""
        bpy.context.view_layer.update()   # extents are stale until reoriented
        ext = self._view_space_extents()
        if ext is None:
            return
        hx, hy, hz = ext
        if self.auto_ortho_scale:
            self.ortho_scale = self._fit_ortho_scale(hx, hy)
            self._set_pixel_scale()
        # Applied whether or not the frame is auto-fitted: the old
        # `obs * ortho_scale` placement puts the camera plane inside the body
        # for any hand-tightened frame.
        self.camera_object.location = (self.view_target
                                       + obs * self._safe_standoff(hz))
        bpy.context.view_layer.update()

    def _assembly_center(self):
        """World bbox centre of the current geometry, whatever attitude or
        articulation the user has applied. The camera targets this, so the
        assembly cannot walk out of frame."""
        bpy.context.view_layer.update()          # matrix_world is stale otherwise
        pts = [obj.matrix_world @ Vector(c)
               for obj in bpy.context.scene.objects if obj.type == 'MESH'
               for c in obj.bound_box]
        if not pts:
            return Vector((0., 0., 0.))
        return (Vector(map(min, zip(*pts))) + Vector(map(max, zip(*pts)))) / 2
    
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

        depsgraph = bpy.context.evaluated_depsgraph_get()
        geometries = {}
        for obj in bpy.context.scene.objects:
            if obj.type != 'MESH':
                continue
            me = obj.evaluated_get(depsgraph).to_mesh()
            me.calc_loop_triangles()
            v = np.empty(len(me.vertices) * 3)
            me.vertices.foreach_get('co', v)
            M = np.array(obj.matrix_world)
            verts = v.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]     # world space
            f = np.empty(len(me.loop_triangles) * 3, dtype=np.int64)
            me.loop_triangles.foreach_get('vertices', f)
            geometries[obj.name] = SimpleNamespace(vertices=verts,
                                                faces=f.reshape(-1, 3))
            obj.evaluated_get(depsgraph).to_mesh_clear()

        all_verts  = np.vstack([g.vertices for g in geometries.values()])
        centroid   = (all_verts.min(axis=0) + all_verts.max(axis=0)) / 2
        max_extent = np.max(all_verts.max(axis=0) - all_verts.min(axis=0)) / 2
        pad        = max_extent * 1.4
        arrow_len  = max_extent * 1.2
        cone_size  = max_extent * 0.08

        # Sun / observer directions (fixed in LVLH)
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

        # LVLH axes — fixed
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

    def _cache_brdfs(self):
        """Resolve each mesh object's BRDF once per frame, keyed by name."""
        self._brdf_cache = {
            o.name: self._get_for_object(o.name)
            for o in bpy.context.scene.objects
            if o.type == "MESH"
        }

    def _smooth_data(self, obj):
        """Per-object arrays needed to interpolate shading normals.

        Cached on the mesh name: the topology and the object-space normals are
        static, only the object's world matrix changes between frames.
        """
        me = obj.data
        key = me.name
        cached = self._smooth_cache.get(key)
        if cached is not None:
            return cached

        me.calc_loop_triangles()

        loop_normals, use_loops = None, False
        try:
            if not hasattr(me, 'corner_normals') and hasattr(me, 'calc_normals_split'):
                me.calc_normals_split()
            arr = np.empty(len(me.loops) * 3, dtype=np.float64)
            if hasattr(me, 'corner_normals'):
                me.corner_normals.foreach_get('vector', arr)
            else:
                me.loops.foreach_get('normal', arr)
            loop_normals = arr.reshape(-1, 3)
            use_loops = bool(np.all(np.linalg.norm(loop_normals, axis=1) > 0.5))
        except Exception:
            loop_normals, use_loops = None, False

        vert_normals = np.array([v.normal[:] for v in me.vertices], dtype=float)
        verts = np.array([v.co[:] for v in me.vertices], dtype=float)

        tri_verts, tri_loops, tri_poly = [], [], []
        for t in me.loop_triangles:
            tri_verts.append(t.vertices[:])
            tri_loops.append(t.loops[:])
            tri_poly.append(t.polygon_index)

        # polygon index -> its triangles, so a ray hit (which reports a polygon)
        # can be resolved to the specific triangle containing the hit point
        poly_tris = {}
        for ti, pi in enumerate(tri_poly):
            poly_tris.setdefault(pi, []).append(ti)

        src_n = loop_normals if use_loops else vert_normals
        idx_n = tri_loops if use_loops else tri_verts
        tri_cache = []
        for ti in range(len(tri_verts)):
            vi = tri_verts[ti]
            a = verts[vi[0]]; b = verts[vi[1]]; c = verts[vi[2]]
            e1 = b - a
            e2 = c - a
            d00 = float(e1 @ e1); d01 = float(e1 @ e2); d11 = float(e2 @ e2)
            den = d00 * d11 - d01 * d01
            inv_den = (1.0 / den) if abs(den) > 1e-20 else 0.0
            ni = idx_n[ti]
            tri_cache.append((
                float(a[0]), float(a[1]), float(a[2]),
                float(e1[0]), float(e1[1]), float(e1[2]),
                float(e2[0]), float(e2[1]), float(e2[2]),
                d00, d01, d11, inv_den,
                tuple(float(x) for x in src_n[ni[0]]),
                tuple(float(x) for x in src_n[ni[1]]),
                tuple(float(x) for x in src_n[ni[2]]),
            ))

        data = dict(verts=verts, vert_normals=vert_normals,
                    loop_normals=loop_normals, use_loops=use_loops,
                    tri_verts=np.array(tri_verts, dtype=int),
                    tri_loops=np.array(tri_loops, dtype=int),
                    tri_poly=np.array(tri_poly, dtype=int),
                    poly_tris=poly_tris, tri_cache=tri_cache)
        self._smooth_cache[key] = data
        return data

    def _cache_object_matrices(self):
        """Per-frame world-matrix inverses for the smooth-normal path.

        These change only when an object moves, but _shading_normal runs once
        per ray hit (~10k/frame), so inverting inside it cost two Matrix
        inversions per hit and dominated the render.
        """
        self._matrix_cache = {}
        for o in bpy.context.scene.objects:
            if o.type != 'MESH':
                continue
            M = o.matrix_world
            self._matrix_cache[o.name] = (
                np.array(M.inverted(), dtype=float),                      # world -> object
                np.array(M.to_3x3().inverted().transposed(), dtype=float),  # object -> world normals
            )

    def _shading_normal(self, obj, face_index, loc_world, face_normal):
        """Barycentric-interpolated shading normal at a hit point.

        Plain-float arithmetic: this runs once per ray hit, where
        numpy's per-call overhead on 3-vectors dominates the actual work.
        Returns a 3-tuple. Falls back to the geometric face normal if anything
        is degenerate.
        """
        try:
            d = self._smooth_cache.get(obj.data.name)
            if d is None:
                d = self._smooth_data(obj)
            tris = d['poly_tris'].get(face_index)
            if not tris:
                return face_normal

            mc = self._matrix_cache.get(obj.name)
            if mc is None:
                M = obj.matrix_world
                mc = (np.array(M.inverted(), dtype=float),
                      np.array(M.to_3x3().inverted().transposed(), dtype=float))
                self._matrix_cache[obj.name] = mc
            Minv, R = mc

            lx = loc_world.x; ly = loc_world.y; lz = loc_world.z
            px = Minv[0, 0] * lx + Minv[0, 1] * ly + Minv[0, 2] * lz + Minv[0, 3]
            py = Minv[1, 0] * lx + Minv[1, 1] * ly + Minv[1, 2] * lz + Minv[1, 3]
            pz = Minv[2, 0] * lx + Minv[2, 1] * ly + Minv[2, 2] * lz + Minv[2, 3]

            cache = d['tri_cache']
            best = None
            best_err = None
            for ti in tris:
                (ax, ay, az, e1x, e1y, e1z, e2x, e2y, e2z,
                 d00, d01, d11, inv_den, n0, n1, n2) = cache[ti]
                if inv_den == 0.0:
                    continue
                vx = px - ax; vy = py - ay; vz = pz - az
                d20 = vx * e1x + vy * e1y + vz * e1z
                d21 = vx * e2x + vy * e2y + vz * e2z
                v = (d11 * d20 - d01 * d21) * inv_den
                w = (d00 * d21 - d01 * d20) * inv_den
                u = 1.0 - v - w
                err = 0.0
                if u < 0.0: err -= u
                if v < 0.0: err -= v
                if w < 0.0: err -= w
                if best_err is None or err < best_err:
                    best_err = err
                    best = (u * n0[0] + v * n1[0] + w * n2[0],
                            u * n0[1] + v * n1[1] + w * n2[1],
                            u * n0[2] + v * n1[2] + w * n2[2])
                    if err <= 1e-9:
                        break

            if best is None:
                return face_normal

            bx, by, bz = best
            nx = R[0, 0] * bx + R[0, 1] * by + R[0, 2] * bz
            ny = R[1, 0] * bx + R[1, 1] * by + R[1, 2] * bz
            nz = R[2, 0] * bx + R[2, 1] * by + R[2, 2] * bz
            nrm = (nx * nx + ny * ny + nz * nz) ** 0.5
            if nrm < 1e-12:
                return face_normal
            nx /= nrm; ny /= nrm; nz /= nrm
            # keep it on the same side as the geometric normal
            if nx * face_normal[0] + ny * face_normal[1] + nz * face_normal[2] < 0.0:
                return (-nx, -ny, -nz)
            return (nx, ny, nz)
        except Exception:
            return face_normal

    def _earth_cap(self, altitude_km, n_rings):
        """Equal-SOLID-ANGLE discretisation of the Earth cap seen from the satellite.

        WHY SOLID ANGLE AND NOT AREA. The earthshine integral weights each patch
        by the solid angle it subtends at the satellite, so discretising the
        Earth in equal surface AREA (as this method used to) piles samples into
        the few patches directly beneath the satellite, under-resolves the limb,
        and reads low at LEO. Equal solid angle samples exactly what the integral
        weights by. The total is gated against the analytic spherical cap
        2*pi*(1 - cos(alpha)) in tests/test_earthshine_extended.py.

        Returns (dirs, omegas, normals): unit vectors FROM the satellite toward
        each patch, their solid angles in steradians, and the outward surface
        normal where each direction strikes. Canonical frame, nadir along -Z;
        `compute_earthshine` rotates them onto `self.nadir_direction`.
        Cached per (altitude, n_rings).
        """
        cache = self.__dict__.setdefault('_earth_cap_cache', {})
        key = (round(float(altitude_km), 1), int(n_rings))
        if key in cache:
            return cache[key]

        R_EARTH = 6371.0
        r_sat = R_EARTH + float(altitude_km)
        if r_sat <= R_EARTH:
            raise ValueError(
                f'altitude_km = {altitude_km} puts the satellite inside the Earth '
                f'(radius {R_EARTH:.0f} km); it must be positive.')
        alpha = np.arcsin(R_EARTH / r_sat)
        cos_edges = np.linspace(1.0, np.cos(alpha), int(n_rings) + 1)
        sat = np.array([0.0, 0.0, r_sat])

        dirs, omegas, norms = [], [], []
        for i in range(int(n_rings)):
            c0, c1 = cos_edges[i], cos_edges[i + 1]
            theta = np.arccos(0.5 * (c0 + c1))
            n_phi = max(1, int(round(2 * np.pi * np.sin(theta) / (c0 - c1)))) if theta > 0 else 1
            n_phi = min(n_phi, 24)
            dOmega = 2 * np.pi * (c0 - c1) / n_phi
            for j in range(n_phi):
                phi = 2 * np.pi * (j + 0.5) / n_phi
                d = np.array([np.sin(theta) * np.cos(phi),
                              np.sin(theta) * np.sin(phi),
                              -np.cos(theta)])          # nadir = -Z
                b = 2.0 * np.dot(sat, d)
                c = np.dot(sat, sat) - R_EARTH ** 2
                disc = b * b - 4.0 * c
                if disc <= 0:
                    continue                            # ray misses the Earth
                pt = sat + ((-b - np.sqrt(disc)) / 2.0) * d
                dirs.append(d)
                omegas.append(dOmega)
                norms.append(pt / np.linalg.norm(pt))

        out = (np.array(dirs), np.array(omegas), np.array(norms))
        cache[key] = out
        return out

    @staticmethod
    def _align_to_nadir(vectors, nadir_direction):
        """Rotate canonical (nadir = -Z) vectors so -Z lands on `nadir_direction`."""
        n = np.asarray(nadir_direction, dtype=float)
        n = n / np.linalg.norm(n)
        z = np.array([0.0, 0.0, -1.0])
        v = np.cross(z, n)
        c = float(np.dot(z, n))
        if np.linalg.norm(v) < 1e-12:
            R = np.eye(3) if c > 0 else -np.eye(3)
        else:
            vx = np.array([[0.0, -v[2], v[1]],
                           [v[2], 0.0, -v[0]],
                           [-v[1], v[0], 0.0]])
            R = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))
        return np.asarray(vectors, dtype=float) @ R.T

    @staticmethod
    def _altitude_from_geometry(geometry):
        """Altitude in km. Earthshine needs this specifically: the Earth's angular
        size depends on altitude, which the slant range alone does not give."""
        alt = getattr(geometry, 'altitude_km', None)
        if alt is not None:
            return float(alt)
        pos = getattr(geometry, 'positions', None)
        if pos is not None and 'satellite' in pos:
            r = np.asarray(pos['satellite'], dtype=float)
            if r.ndim > 1:
                raise ValueError(
                    f'compute_earthshine needs a single epoch, got {r.shape[-1]}.')
            return float(np.linalg.norm(r)) - 6371.0
        raise TypeError(
            'geometry carries no altitude: a Geometry supplies it through '
            'positions["satellite"], a ViewGeometry through altitude_km=.')

    def compute_earthshine(self, geometry=None, n_rings=None,
                           albedo=None, verbose=False, **legacy):
        """Earthshine as an EXTENDED source: one direction + irradiance per patch.

        A patch of Earth subtending solid angle omega with radiance L is, to any
        surface element, exactly a directional source of irradiance

            E = L * omega        arriving from that patch's direction,

        so the visible Earth becomes a set of directional sources that `render`
        sums over. This replaces the previous treatment, which collapsed the
        Earth into ONE effective direction and one total irradiance. A lumped
        direction is wrong for an extended source: a facet perpendicular to it
        collects exactly zero while physically still seeing half the disc, and no
        single direction can represent a source 68 deg wide at LEO.

        Call once per frame, before `render`. Directions are produced in the same
        frame as `self.sun_direction`, with nadir taken as `self.nadir_direction`.

        Parameters
        ----------
        geometry : satlight.geometry.Geometry, optional
            Compatibility path: altitude and Sun direction are taken from it, so
            they arrive in LVLH. If you render in a ROLLED BODY FRAME, pass
            `altitude_km` and `sun_direction` in that frame instead, or the Earth
            will sit at the wrong attitude.
        altitude_km, sun_direction :
            Explicit alternative to `geometry`, in the render frame.
        n_rings : int, optional
            Overrides `self.earthshine_n_rings` for this call.

        Sets `earthshine_dirs` (N,3) and `earthshine_patch_irradiance` (N,), and
        for backwards compatibility also `earthshine_irradiance` (the total on a
        plate facing the Earth) and `earthshine_direction` (flux-weighted mean).
        """
        R_EARTH = 6371.0
        # `compute_earthshine` is usable on a Renderer that has no Blender scene
        # (it needs only a radius and a Sun direction), so read the two settings
        # it depends on defensively rather than requiring a full __init__.
        rings = n_rings if n_rings is not None else getattr(
            self, 'earthshine_n_rings', 4)
        nadir = np.asarray(getattr(self, 'nadir_direction', (0.0, 0.0, -1.0)),
                           dtype=float)

        used = [k for k in ('altitude_km', 'sun_direction') if k in legacy]
        if used:
            raise TypeError(
                f'compute_earthshine no longer takes {", ".join(used)}; both come '
                f'off the geometry now. Use a Geometry, or ViewGeometry(..., '
                f'altitude_km=...).')
        if legacy:
            raise TypeError(f'compute_earthshine got unexpected keyword '
                            f'argument(s): {", ".join(sorted(legacy))}')
        if geometry is None:
            raise TypeError(
                'compute_earthshine requires geometry=. Pass a Geometry, or a '
                'ViewGeometry(..., altitude_km=...).')
        altitude_km = self._altitude_from_geometry(geometry)
        sun_direction = np.asarray(geometry.incident_vector_lvlh, dtype=float)

        sun = np.asarray(sun_direction, dtype=float)
        sun = sun / np.linalg.norm(sun)

        dirs, omegas, norms = self._earth_cap(altitude_km, rings)
        dirs = self._align_to_nadir(dirs, nadir)
        norms = self._align_to_nadir(norms, nadir)

        rho = self.earth_albedo if albedo is None else albedo
        if rho is not None:
            albedos = np.full(norms.shape[0], float(rho))
        else:
            # Per-patch CERES map. NOTE lat/lon come from the patch normal with
            # no GMST correction, so the continents sit at an arbitrary
            # longitude -- prefer the uniform albedo until that is fixed.
            grid = self._load_ceres_albedo()
            lat = np.degrees(np.arcsin(np.clip(norms[:, 2], -1.0, 1.0)))
            lon = np.degrees(np.arctan2(norms[:, 1], norms[:, 0]))
            albedos = grid[np.clip(np.floor(90 - lat).astype(int), 0, 179),
                           np.clip(np.floor(lon + 180).astype(int), 0, 359)]

        # Lambertian Earth; patches in darkness contribute nothing.
        cos_sun = np.clip(norms @ sun, 0.0, None)
        E = albedos * self.solar_constant * cos_sun / np.pi * omegas

        lit = E > 0.0
        self.earthshine_dirs = dirs[lit]
        self.earthshine_patch_irradiance = E[lit]

        if not np.any(lit):
            self.earthshine_irradiance = 0.0
            self.earthshine_direction = None
            return

        # Legacy scalars, kept so existing callers still read something sensible;
        # `render` uses the per-patch arrays above.
        ci = np.clip(self.earthshine_dirs @ nadir, 0.0, None)
        self.earthshine_irradiance = float(np.sum(self.earthshine_patch_irradiance * ci))
        w = self.earthshine_patch_irradiance[:, None] * self.earthshine_dirs
        mean_dir = np.sum(w, axis=0)
        self.earthshine_direction = mean_dir / np.linalg.norm(mean_dir)
        if verbose:
            print(f'Earthshine: {self.earthshine_dirs.shape[0]} lit patches, '
                  f'{self.earthshine_irradiance:.4f} W/m^2 on a nadir plate '
                  f'(mean direction {self.earthshine_direction.round(3)})')


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

    
    def _view_space_extents(self):
        """Half-extents of the geometry about the VIEW TARGET, in camera axes.

        Returns (across, up, along-view). Measured about the view target rather
        than the camera, so the result does not depend on where the camera
        currently sits - otherwise the along-view extent would include the
        camera's own standoff and grow every time `update` is called.

        Bounding-box corners bound the geometry and there are only 8 per object,
        so this is cheap enough to run every frame.
        """
        import mathutils as mu

        M = self.camera_object.matrix_world.to_3x3()
        right = M @ mu.Vector((1.0, 0.0, 0.0))
        up = M @ mu.Vector((0.0, 1.0, 0.0))
        fwd = M @ mu.Vector((0.0, 0.0, 1.0))
        tgt = self.view_target
        hx = hy = hz = 0.0
        found = False
        for obj in bpy.context.scene.objects:
            if obj.type != 'MESH':
                continue
            for corner in obj.bound_box:
                d = (obj.matrix_world @ mu.Vector(corner)) - tgt
                hx = max(hx, abs(d.dot(right)))
                hy = max(hy, abs(d.dot(up)))
                hz = max(hz, abs(d.dot(fwd)))
                found = True
        return (hx, hy, hz) if found else None

    def _fit_ortho_scale(self, hx, hy):
        """Frame width that just contains the projected geometry, plus margin."""
        return 2.0 * max(hx, hy * self.aspect_ratio) * self.ortho_margin

    def _safe_standoff(self, hz):
        """How far back the camera plane must sit to leave the body in front of it.

        THIS IS NOT A FUNCTION OF ortho_scale. An orthographic projection does not
        care where along the view axis the camera sits - but `render` starts its
        primary rays ON the camera plane, so any geometry behind that plane is
        silently dropped. The requirement is purely "clear the body along the view
        axis", which is `hz`. Deriving it from ortho_scale (a frame WIDTH) instead
        conflates two unrelated quantities and breaks as soon as the frame is
        tightened: a 5.4 m frame on a body whose arms reach 17 m along the view
        axis put the camera plane inside a solar array.
        """
        return max(hz * 2.0, 1.0)

    def update(self, geometry=None, **legacy):
        """Set the viewing geometry for the next frame - call before render().

        Takes a `Geometry` or a `ViewGeometry`, the same as the constructor.
        """
        import mathutils as mu

        _reject_legacy(legacy, 'update()')
        _require_geometry(geometry, 'update()')
        sun, obs, dist_m = self._frame_from_geometry(geometry)

        self.sun_direction = mu.Vector(np.asarray(sun).tolist())
        self.observer_direction = mu.Vector(np.asarray(obs).tolist())
        self.distance_to_observer = dist_m
        self._place_camera(self.observer_direction)

    # ------------------------------------------------------------------
    # multiple scattering
    # ------------------------------------------------------------------
    @staticmethod
    def _cosine_hemisphere(normal, n_samples, rng):
        """Cosine-weighted directions about `normal`. Returns (N,3) unit vectors.

        Cosine weighting is the natural measure here: the reflected-radiance
        integral carries a cos(theta) factor, so sampling proportional to it
        makes the estimator's weight independent of angle and kills most of the
        variance that uniform hemisphere sampling would introduce.
        """
        n = np.asarray(normal, dtype=float)
        # any vector not parallel to n, to build a tangent frame
        a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        t = np.cross(n, a); t /= np.linalg.norm(t)
        b = np.cross(n, t)
        u1 = rng.random(n_samples)
        u2 = rng.random(n_samples)
        r = np.sqrt(u1)
        phi = 2.0 * np.pi * u2
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        z = np.sqrt(np.maximum(0.0, 1.0 - u1))
        return x[:, None] * t + y[:, None] * b + z[:, None] * n

    def _direct_radiance(self, depsgraph, ray_cast, loc, n_np, obj_name, out_dir):
        """Radiance leaving a surface point toward `out_dir`, from Sun + earthshine.

        W/m^2/sr. This is the single-bounce shading, factored out so that
        secondary hits can be shaded with exactly the same physics as primary
        ones (same BRDF, same shadow test, same earthshine patches).
        """
        brdf_func, brdf_kwargs = self._brdf_cache[obj_name]
        eps = self.ortho_scale * 1e-5
        origin = Vector((loc.x + n_np[0] * eps,
                         loc.y + n_np[1] * eps,
                         loc.z + n_np[2] * eps))
        L = 0.0

        sun_np = np.asarray(self.sun_direction, dtype=float)
        if float(n_np @ sun_np) > 0.0:
            lit = True
            if self.add_shadows:
                lit = not ray_cast(depsgraph, origin,
                                   Vector(sun_np.tolist()))[0]
            if lit:
                L += brdf_func(n_np, sun_np, out_dir, **brdf_kwargs) * self.solar_constant

        if (self.add_earthshine and self.earthshine_dirs is not None
                and len(self.earthshine_dirs) > 0):
            for d, e in zip(self.earthshine_dirs, self.earthshine_patch_irradiance):
                if float(n_np @ d) <= 0.0:
                    continue
                if self.earthshine_shadows and ray_cast(
                        depsgraph, origin, Vector(np.asarray(d).tolist()))[0]:
                    continue
                L += brdf_func(n_np, d, out_dir, **brdf_kwargs) * e
        return L

    def _indirect_radiance(self, depsgraph, ray_cast, loc, n_np, obj_name,
                           out_dir, depth):
        """Radiance toward `out_dir` that arrived via at least one other surface.

        Monte Carlo estimate of

            L_ind = integral f_r(n, w, out) L_in(w) cos(theta) dw

        with cosine-weighted sampling. The BRDF callables in satlight return
        f_r * cos(theta), so dividing by cos(theta) recovers f_r and the
        cosine-weighted estimator reduces to (pi / N) * sum f_r * L_in.
        """
        if depth >= self.n_bounce:
            return 0.0
        brdf_func, brdf_kwargs = self._brdf_cache[obj_name]
        eps = self.ortho_scale * 1e-5
        origin = Vector((loc.x + n_np[0] * eps,
                         loc.y + n_np[1] * eps,
                         loc.z + n_np[2] * eps))
        dirs = self._cosine_hemisphere(n_np, self.bounce_samples, self._bounce_rng)

        total = 0.0
        for w in dirs:
            hit, loc2, nrm2, face2, obj2, _ = ray_cast(depsgraph, origin,
                                                       Vector(w.tolist()))
            if not hit or obj2 is None:
                continue                      # ray escaped to space: no incoming radiance
            n2 = np.array((nrm2.x, nrm2.y, nrm2.z))
            if self.smooth_normals:
                n2 = np.array(self._shading_normal(obj2, face2, loc2, tuple(n2)))
            back = -w                          # from the second point toward this one
            if float(n2 @ back) <= 0.0:
                n2 = -n2                       # we hit the far side; use the facing normal
            L_in = self._direct_radiance(depsgraph, ray_cast, loc2, n2,
                                         obj2.name, back)
            if self.n_bounce - depth > 1:
                L_in += self._indirect_radiance(depsgraph, ray_cast, loc2, n2,
                                                obj2.name, back, depth + 1)
            if L_in <= 0.0:
                continue
            cos_w = float(n_np @ w)
            if cos_w <= 1e-6:
                continue
            f_r = brdf_func(n_np, w, out_dir, **brdf_kwargs) / cos_w
            total += f_r * L_in
        return np.pi * total / self.bounce_samples

    def render(self):
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        ray_cast = bpy.context.scene.ray_cast
        self._cache_brdfs()
        if self.smooth_normals:
            self._cache_object_matrices()
 
        cam_matrix = self.camera_object.matrix_world
        ray_dir_world = (cam_matrix.to_3x3() @ Vector((0.0, 0.0, -1.0))).normalized()
        vx, vy, vz = -ray_dir_world.x, -ray_dir_world.y, -ray_dir_world.z
        view_dir = np.array((vx, vy, vz))
 
        self.image = np.zeros((self.resolution_y, self.resolution_x))
        self.hit_count = 0
        self.shadow_count = 0
        # reset per frame so a given geometry always renders identically
        self._bounce_rng = np.random.default_rng(self.bounce_seed)
        k_geom = self.pixel_area / self.distance_to_observer ** 2
 
        mask = np.zeros((self.resolution_y, self.resolution_x), dtype=bool)
        for x_min, x_max, y_min, y_max in self.get_object_pixel_bboxes():
            mask[y_min:y_max, x_min:x_max] = True
 
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            self._check_frame_edges()
            return self.image
 
        screen_x = (2.0 * xs / self.resolution_x) - 1.0
        screen_y = 1.0 - (2.0 * ys / self.resolution_y)
        cam_local = np.column_stack((
            screen_x * (self.ortho_scale / 2),
            screen_y * (self.ortho_scale / 2) / self.aspect_ratio,
            np.zeros(xs.size),
        ))
        Mc = np.array(cam_matrix)
        origins_list = (cam_local @ Mc[:3, :3].T + Mc[:3, 3]).tolist()
 
        sun_dir_np = np.array(self.sun_direction, dtype=float)
        sx, sy, sz = sun_dir_np.tolist()
        sun_dir_vec = Vector((sx, sy, sz))
 
        inv_d2 = 1.0 / self.distance_to_observer ** 2
        k_sun = self.solar_constant * self.pixel_area * inv_d2

        # Earthshine is an EXTENDED source: every visible, sunlit Earth patch is
        # its own directional source of irradiance L*omega. Summing them gives
        # the correct per-pixel cos(incidence) on every facet at every attitude,
        # which one lumped direction cannot: a facet perpendicular to the lump
        # would collect zero while still seeing half the disc.
        es_on = (self.add_earthshine and self.earthshine_dirs is not None
                 and len(self.earthshine_dirs) > 0)
        if es_on:
            es_dirs = np.asarray(self.earthshine_dirs, dtype=float)
            es_k = (np.asarray(self.earthshine_patch_irradiance, dtype=float)
                    * self.pixel_area * inv_d2)
            es_vecs = ([Vector(d.tolist()) for d in es_dirs]
                       if self.earthshine_shadows else None)
            es_eps = self.ortho_scale * 1e-5
 
        brdf_cache = self._brdf_cache
        image = self.image
 
        for i in range(xs.size):
            ox, oy, oz = origins_list[i]
            hit, loc, normal, _face, obj, _m = ray_cast(
                depsgraph, Vector((ox, oy, oz)), ray_dir_world
            )
            if not hit:
                continue
            self.hit_count += 1
 
            nx, ny, nz = normal.x, normal.y, normal.z
            if self.smooth_normals:
                nx, ny, nz = self._shading_normal(obj, _face, loc, (nx, ny, nz))
            cos_theta = nx * vx + ny * vy + nz * vz
            if cos_theta <= 0.0:
                continue
 
            brdf_func, brdf_kwargs = brdf_cache[obj.name]
            n_dot_l = nx * sx + ny * sy + nz * sz
            flux = 0.0
            hit_normal = None
 
            if n_dot_l > 0.0:
                lit = True
                if self.add_shadows:
                    eps = self.ortho_scale * 1e-5
                    shadow_origin = Vector((loc.x + nx*eps,
                                            loc.y + ny*eps,
                                            loc.z + nz*eps))
                    lit = not ray_cast(depsgraph, shadow_origin, sun_dir_vec)[0]
                if lit:
                    hit_normal = np.array((nx, ny, nz))
                    flux += (brdf_func(hit_normal, sun_dir_np, view_dir, **brdf_kwargs)* k_sun)
                else:
                    self.shadow_count += 1
 
            if es_on:
                if hit_normal is None:
                    hit_normal = np.array((nx, ny, nz))
                es_origin = None
                for m in range(es_dirs.shape[0]):
                    dmx, dmy, dmz = es_dirs[m, 0], es_dirs[m, 1], es_dirs[m, 2]
                    if nx * dmx + ny * dmy + nz * dmz <= 0.0:
                        continue            # this patch is behind the facet
                    if es_vecs is not None:
                        if es_origin is None:
                            es_origin = Vector((loc.x + nx * es_eps,
                                                loc.y + ny * es_eps,
                                                loc.z + nz * es_eps))
                        if ray_cast(depsgraph, es_origin, es_vecs[m])[0]:
                            continue        # spacecraft occludes the Earth here
                    flux += brdf_func(hit_normal, es_dirs[m], view_dir,
                                      **brdf_kwargs) * es_k[m]

            if self.n_bounce > 1:
                # a SHADOWED facet has no direct term and so no hit_normal yet,
                # and that is exactly where inter-reflection matters most
                if hit_normal is None:
                    hit_normal = np.array((nx, ny, nz))
                flux += self._indirect_radiance(
                    depsgraph, ray_cast, loc, hit_normal, obj.name,
                    view_dir, 1) * k_geom

            image[ys[i], xs[i]] = flux
 
        self._check_frame_edges()
        return self.image


    def _check_frame_edges(self, edge_threshold=1e-6):
        """Warn once per frame if the spacecraft is clipped by the field of view."""
        edge_pixels = np.concatenate([
            self.image[0, :].ravel(),    # top row
            self.image[-1, :].ravel(),   # bottom row
            self.image[:, 0].ravel(),    # left column
            self.image[:, -1].ravel(),   # right column
        ])
        if np.any(edge_pixels > edge_threshold):
            print(f"  WARNING: illuminated pixels detected at frame edge "
                  f"(max edge value {edge_pixels.max():.4g}) — spacecraft may be clipped out of "
                  f"the field of view. Increase `resolution` or change aspect ratio and re-render.")