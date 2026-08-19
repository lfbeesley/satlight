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

class Renderer:
    """ To manage and render the object in blender and perform the ray-casting."""

    def __init__(self, obj_path, sun_direction, observer_direction, distance_to_observer, resolution=(500,500), solar_constant = 1361, add_earthshine=False, add_shadows=True, brdf='lambertian', smooth_normals=False, smooth_angle_deg=30.0):
        '''
        obj_path can be either .obj or .stl
        distance_to_observer in m
        Resolution in (x,y)
        sun and obs_directions are normalised

        smooth_normals: scene.ray_cast returns the flat FACE normal, so a
        tessellated curved surface is shaded as a set of flat mirrors even
        when the mesh carries smooth/custom vertex normals. For a specular
        BRDF that turns one continuous glint into a train of discrete
        flashes, spaced by twice the facet-to-facet normal step. Enabling
        this interpolates the mesh's own vertex/loop normals at the hit
        point instead. Default False so existing results are unchanged.
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
        self.add_shadows = add_shadows
        self.smooth_normals = smooth_normals
        self.smooth_angle_deg = smooth_angle_deg
        self._smooth_cache = {}

        # Set up BRDFs
        if isinstance(brdf, dict):
            self.brdf_map = brdf
        else:
            self.brdf_map = {'default': brdf}

    def initialise_scene(self):
        # Set-up scene
        bpy.ops.wm.read_factory_settings(use_empty=True)
        self.ortho_scale = self._load_and_center_object(self.obj_path)
        if self.smooth_normals:
            self._repair_normals(self.smooth_angle_deg)
        self.add_camera()

    def _repair_normals(self, angle_deg=30.0):
        """Discard authored split normals and recompute them from geometry.

        Several CAD/OBJ exports carry mis-indexed vertex normals: on this
        Intelsat 1R model the median angle between a corner normal and its own
        polygon's normal is ~90 deg on the solar panels and 45 deg on the
        antenna hinges, i.e. the authored data is unusable and shading with it
        would be worse than using flat faces. Clearing it and re-deriving with
        an angle threshold restores the intent: smooth across a tessellated
        curved dish, sharp across a genuine crease.
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
        import mathutils as mu

        self.camera_object = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
        self.camera_object.data.type = 'ORTHO'
        bpy.context.collection.objects.link(self.camera_object)
        bpy.context.scene.camera = self.camera_object
        self.camera_object.data.ortho_scale = self.ortho_scale

        self.view_target = self._assembly_center()
        self.camera_object.location = self.view_target + mu.Vector((0.0, 0.0, -1.0)) * self.ortho_scale
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

        # Split ("loop"/corner) normals, which respect sharp edges -- essential
        # here, because plain vertex normals average across a thin slab's face
        # and its side walls and would tilt a flat panel by ~45 deg. Blender
        # >=4.1 computes these automatically and exposes mesh.corner_normals;
        # older versions need the now-removed calc_normals_split() first.
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

        # Flat-float per-triangle cache for the hot path. _shading_normal runs
        # once per ray hit (~10k/frame); numpy on 3-vectors costs more in
        # overhead than it saves at that size, so precompute everything the
        # barycentric solve needs as plain Python floats.
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

        Plain-float arithmetic throughout: this runs once per ray hit, where
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
        self.view_target = self._assembly_center()
        self.camera_object.location = self.view_target + obs * self.ortho_scale

        to_origin = (self.view_target - self.camera_object.location).normalized()

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
 
        es_on = self.add_earthshine and self.earthshine_direction is not None
        es_dir_np = np.array(self.earthshine_direction, dtype=float) if es_on else None
 
        inv_d2 = 1.0 / self.distance_to_observer ** 2
        k_sun = self.solar_constant * self.pixel_area * inv_d2
        k_es = (self.earthshine_irradiance * self.pixel_area * inv_d2) if es_on else 0.0
 
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
 
            # Plain floats: numpy on 3-vectors costs more in overhead than it saves.
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
                flux += (
                    brdf_func(hit_normal, es_dir_np, view_dir, **brdf_kwargs) * k_es)

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