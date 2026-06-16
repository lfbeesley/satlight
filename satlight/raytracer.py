import sys
sys.path.insert(0, '/Users/l.beesley@bham.ac.uk/Documents/Lightcurves/satlight')

from importlib.resources import files
import bpy
import numpy as np
from mathutils import Vector
import matplotlib.pyplot as plt
from satlight import tools
from tqdm import tqdm
from hide_warnings import hide_warnings
import plotly.graph_objects as go
import trimesh

def lambertian_brdf(normal, light_dir, view_dir, albedo = 1):
    """Lambertian shading"""
    n_dot_l = max(0, np.dot(normal, light_dir))
    return albedo / np.pi * n_dot_l

class Renderer:
    """ To manage and render the object in blender and perform the ray-casting."""

    def __init__(self, obj_path, sun_direction, observer_direction, distance_to_observer, resolution=(500,500),solar_constant = 1460, add_earthshine=False):
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
        self.add_earthshine = add_earthshine

    def initialise_scene(self):
        # Set-up scene
        bpy.ops.wm.read_factory_settings(use_empty=True)

        # Add/center object and return orth_scale
        self.ortho_scale = self._load_and_center_object(self.obj_path)

        # Add camera
        self.add_camera()

        # Add earthshine
        if self.add_earthshine:
            self.compute_earthshine()

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
        ortho_scale = (max_corner - min_corner).length

        # Move all meshes so their center is at origin
        for obj in mesh_objects:
            obj.location -= center
        
        return ortho_scale


    def add_camera(self):
        self.camera_object = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
        self.camera_object.data.type = 'ORTHO'
        bpy.context.collection.objects.link(self.camera_object)
        bpy.context.scene.camera = self.camera_object

        # Set camera to orthographic
        self.camera_object.data.ortho_scale = self.ortho_scale # Width of camera along x-axis

        # Set camera position at specified distance
        self.camera_object.location = self.observer_direction.normalized() * self.ortho_scale  
        direction = Vector((0, 0, 0)) - self.camera_object.location

        # Point camera at geometric center (origin)
        self.camera_object.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler() #-Z is the cameras conventional 'look direction' point that toward origin

        # Calculate camera pixel size
        self.pixel_size_x = self.ortho_scale / self.resolution_x
        self.pixel_size_y = self.ortho_scale / self.aspect_ratio / self.resolution_y 
        self.pixel_area = self.pixel_size_x * self.pixel_size_y
    
    def get_projected_bbox_pixels(self):
        """
        Return pixel bounding box of the object as seen by the camera.
        Only cast rays within this region — skips guaranteed misses.
        """
        bpy.context.view_layer.update()
        mesh_objects = [o for o in bpy.context.scene.objects if o.type == 'MESH']
        
        cam_matrix_inv = self.camera_object.matrix_world.inverted()
        xs, ys = [], []
        
        for obj in mesh_objects:
            for corner in obj.bound_box:
                world_co  = obj.matrix_world @ Vector(corner)
                cam_co    = cam_matrix_inv @ world_co
                x_ndc     = cam_co.x / (self.ortho_scale / 2)
                y_ndc     = cam_co.y / (self.ortho_scale / 2 / self.aspect_ratio)
                xs.append(int((x_ndc + 1) / 2 * self.resolution_x))
                ys.append(int((1 - y_ndc) / 2 * self.resolution_y))
        
        padding = 2
        x_min = max(0,                 min(xs) - padding)
        x_max = min(self.resolution_x, max(xs) + padding)
        y_min = max(0,                 min(ys) - padding)
        y_max = min(self.resolution_y, max(ys) + padding)
        return x_min, x_max, y_min, y_max

    def preview_lvlh_views(self):
        """
        Load the mesh via trimesh and display an interactive 3D plotly plot
        with LVLH axis arrows and labels.

        Must be called after `initialise_scene()` (uses self.obj_path).
        """

        # Load mesh
        mesh = trimesh.load(self.obj_path, force='mesh')
        mesh.vertices -= mesh.bounding_box.centroid

        # Apply the same rotation as the Blender scene
        rx, ry, rz = np.radians(self.rotation)
        Rx = trimesh.transformations.rotation_matrix(rx, [1, 0, 0])
        Ry = trimesh.transformations.rotation_matrix(ry, [0, 1, 0])
        Rz = trimesh.transformations.rotation_matrix(rz, [0, 0, 1])
        mesh.apply_transform(Rz @ Ry @ Rx)

        vertices = mesh.vertices
        faces = mesh.faces
        max_extent = max(mesh.bounding_box.extents) / 2

        # Axis padding so arrows sit just outside the mesh
        pad = max_extent * 1.3
        arrow_len = max_extent * 1.2
        cone_size = max_extent * 0.08

        # ── LVLH axis definitions ──
        axes = {
            "Along-Track":  dict(dir=[1, 0, 0], color="rgb(220, 50, 50)"),
            "Cross-Track":  dict(dir=[0, 1, 0], color="rgb(50, 160, 50)"),
            "Nadir":        dict(dir=[0, 0, 1], color="rgb(50, 80, 220)"),
            "Observer":     dict(dir=[self.observer_direction[0], self.observer_direction[1], self.observer_direction[2]], color="rgb(220, 220, 220)"),
            "Sun":          dict(dir=[self.sun_direction[0], self.sun_direction[1], self.sun_direction[2]],  color="rgb(50, 80, 220)"),
        }


        # ── Build figure with mesh ──
        fig = go.Figure(data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color='lightgrey',
                flatshading=True,
                lighting=dict(ambient=0.6, diffuse=0.7, specular=0.15),
                lightposition=dict(x=1, y=1, z=1),
                hoverinfo='skip',
            )
        ])

        # Add wireframe edges
        # Extract unique edges from faces
        edges = set()
        for f in faces:
            for i in range(3):
                edge = tuple(sorted((f[i], f[(i + 1) % 3])))
                edges.add(edge)

        # Build edge lines (with None separators for disconnected segments)
        xe, ye, ze = [], [], []
        for e0, e1 in edges:
            xe += [vertices[e0, 0], vertices[e1, 0], None]
            ye += [vertices[e0, 1], vertices[e1, 1], None]
            ze += [vertices[e0, 2], vertices[e1, 2], None]

        fig.add_trace(go.Scatter3d(
            x=xe, y=ye, z=ze,
            mode='lines',
            line=dict(color='black', width=1),
            showlegend=False,
            hoverinfo='skip',
        ))

        # ── Add axis arrows (line + cone) ──
        for label, props in axes.items():
            d = np.array(props["dir"], dtype=float)
            c = props["color"]
            colorscale = [[0, c], [1, c]]

            # Arrow shaft: origin to arrow_len along axis
            tip = d * arrow_len
            fig.add_trace(go.Scatter3d(
                x=[0, tip[0]],
                y=[0, tip[1]],
                z=[0, tip[2]],
                mode='lines',
                line=dict(color=c, width=5),
                showlegend=False,
                hoverinfo='skip',
            ))

            # Arrowhead cone
            fig.add_trace(go.Cone(
                x=[tip[0]], y=[tip[1]], z=[tip[2]],
                u=[d[0]], v=[d[1]], w=[d[2]],
                sizeref=cone_size,
                colorscale=colorscale,
                showscale=False,
                hovertext=label,
                hoverinfo='text',
            ))

        # ── Clean white layout ──
        axis_x = dict(
            title="",
            showbackground=False,
            showticklabels=False,
            showgrid=True,
            gridcolor="rgba(200,200,200,0.4)",
            zeroline=False,
            range=[-pad, pad],
        )
        axis_y = dict(
            title="",
            showbackground=False,
            showticklabels=False,
            showgrid=True,
            gridcolor="rgba(200,200,200,0.4)",
            zeroline=False,
            range=[-pad, pad],
        )
        axis_z = dict(
            title="",
            showbackground=False,
            showticklabels=False,
            showgrid=True,
            gridcolor="rgba(200,200,200,0.4)",
            zeroline=False,
            range=[-pad, pad],
        )

        fig.update_layout(
            title=dict(
                text="LVLH Coordinate Preview",
                x=0.5,
                font=dict(size=18),
            ),
            width=950,
            height=750,
            paper_bgcolor="white",
            scene=dict(
                xaxis=axis_x,
                yaxis=axis_y,
                zaxis=axis_z,
                aspectmode='cube',
                bgcolor="white",
            ),
            margin=dict(l=10, r=10, t=50, b=10),
            annotations=[
                dict(
                    text=(
                        '<span style="color:rgb(220,50,50)">&#9632; Red (X) = Along-Track</span><br>'
                        '<span style="color:rgb(50,160,50)">&#9632; Green (Y) = Cross-Track</span><br>'
                        '<span style="color:rgb(50,80,220)">&#9632; Blue (Z) = Nadir</span><br>'
                        '<br>'
                        '<span style="color:#555">If your model axes don\'t match LVLH,<br>'
                        'adjust <b>rotation=(rx, ry, rz)</b> in degrees<br>'
                        'e.g. rotation=(90, 0, 0) to rotate<br>'
                        '90° about the Along-Track axis.</span>'
                    ),
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    showarrow=False,
                    font=dict(size=14),
                    align="left",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="rgba(200,200,200,0.5)",
                    borderwidth=1,
                )
            ],
        )

        fig.show()

    def compute_earthshine(self, geometry, n_rings=40, albedo=0.30):
        """
        Compute earthshine irradiance using Earth patches in LVLH frame.
        Requires a Geometry object with calculated positions.
        """
        R_EARTH = 6371.0

        # Rotation matrix GCRS -> LVLH (already in Geometry)
        R_mat  = np.stack([
            geometry.along_track,
            geometry.cross_track,
            -geometry.nadir        # nadir points toward Earth, -nadir = up = +Z
        ], axis=0)                 # shape (3,3) for single time

        sat_pos_gcrs = geometry.positions['satellite']  # (3,) km

        # Generate Driscoll-Healy patches in GCRS
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

        centers_gcrs = np.vstack(cs)   # (N, 3) in GCRS km
        areas        = np.concatenate(As)

        # Transform patch centres to LVLH
        # Translate to satellite frame, then rotate
        rel          = centers_gcrs - sat_pos_gcrs          # (N, 3)
        centers_lvlh = (R_mat @ rel.T).T                    # (N, 3)

        # Patch normals in LVLH (outward from Earth centre, also transformed)
        earth_centre_lvlh = R_mat @ (np.zeros(3) - sat_pos_gcrs)
        norms_lvlh = centers_lvlh - earth_centre_lvlh
        norms_lvlh /= np.linalg.norm(norms_lvlh, axis=1, keepdims=True)

        # Sun direction in LVLH (already available from Geometry)
        sun_lvlh = geometry.incident_vector_lvlh   # (3,) unit vector

        # Satellite is at origin in LVLH
        sat_lvlh = np.zeros(3)

        # Visibility and illumination
        to_sat  = sat_lvlh - centers_lvlh
        dist_km = np.linalg.norm(to_sat, axis=1)
        cos_sat = np.sum(norms_lvlh * (to_sat / dist_km[:,None]), axis=1)
        cos_sun = norms_lvlh @ sun_lvlh
        mask    = (cos_sun > 0) & (cos_sat > 0)

        # Irradiance sum
        dE = (albedo / np.pi) * self.solar_constant \
            * cos_sun[mask] * cos_sat[mask] \
            * (areas[mask] * 1e6) / (dist_km[mask] * 1e3)**2

        self.earthshine_irradiance = float(np.sum(dE))
        self.nadir_direction       = np.array([0., 0., -1.])  # fixed in LVLH
        print(f"Earthshine irradiance: {self.earthshine_irradiance:.4f} W/m²")

    
    def update(self, sun_direction, observer_direction, distance_to_observer):
        """Update geometry for next timestep — call before render()."""
        self.sun_direction        = sun_direction
        self.observer_direction   = observer_direction
        self.distance_to_observer = distance_to_observer

        # Move camera to new observer position
        self.camera_object.location = observer_direction.normalized() * self.ortho_scale
        direction = Vector((0, 0, 0)) - self.camera_object.location
        self.camera_object.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
        
    def render(self):
        # Ray casting setup
        self.scene = bpy.context.scene
        self.depsgraph = bpy.context.evaluated_depsgraph_get()

        # Get camera matrix for transforming rays
        cam_matrix = self.camera_object.matrix_world
        ray_dir_world = (cam_matrix @ Vector((0, 0, -1)) - cam_matrix @ Vector((0, 0, 0))).normalized()
        ray_dir = np.array(ray_dir_world)
        
        # Output
        self.image = np.zeros((self.resolution_y, self.resolution_x))
        projected_areas = []
        self.hit_count = 0
        self.shadow_count = 0

        x_min, x_max, y_min, y_max = self.get_projected_bbox_pixels()

        # Cast rays only within bounding box
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                # Calculate ray for this pixel in normalised coordinates
                screen_x = (2.0 * x / self.resolution_x) - 1.0
                screen_y = 1.0 - (2.0 * y / self.resolution_y)  # Flipped Y axis
                
                # Ray direction in camera to scale to local coordinates
                ray_origin_cam = Vector((
                    screen_x * (self.ortho_scale / 2) ,
                    screen_y * (self.ortho_scale / 2) / self.aspect_ratio,
                    0.0
                ))
                
                ray_origin_world = cam_matrix @ ray_origin_cam
                
                result, location, normal, index, obj_hit, _ = self.scene.ray_cast(self.depsgraph, ray_origin_world, ray_dir)
                
                if result:
                    self.hit_count += 1
                    hit_point = np.array(location)
                    hit_normal = np.array(normal)
                    
                    # Check shadow
                    shadow_origin = hit_point + hit_normal * 1e-4  # Small epsilon offset
                    shadow_result, *_ = self.scene.ray_cast(self.depsgraph, shadow_origin, self.sun_direction)
                    
                    if shadow_result:
                        self.shadow_count += 1
                        flux = 0 

                        # Check earthshine
                        if self.add_earthshine:
                            shadow_earth, *_ = self.scene.ray_cast(
                                self.depsgraph, shadow_origin, self.nadir_direction
                            )
                            if not shadow_earth:
                                brdf_earth = lambertian_brdf(hit_normal, self.nadir_direction, -ray_dir)
                                radiance_earth = brdf_earth * self.earthshine_irradiance
                                cos_theta = max(0, np.dot(hit_normal, -ray_dir))
                                self.projected_area = self.pixel_area * cos_theta
                                flux = radiance_earth * self.projected_area / self.distance_to_observer**2

                    else:
                        view_dir = -ray_dir
                        # Sunlight
                        brdf = lambertian_brdf(hit_normal, self.sun_direction, view_dir)
                        radiance = brdf * self.solar_constant

                        # Earthshine on sunlit faces too
                        if self.add_earthshine:
                            shadow_earth, *_ = self.scene.ray_cast(
                                self.depsgraph, shadow_origin, self.nadir_direction
                            )
                            if not shadow_earth:
                                brdf_earth = lambertian_brdf(hit_normal, self.nadir_direction, view_dir)
                                radiance += brdf_earth * self.earthshine_irradiance

                        cos_theta = max(0, np.dot(hit_normal, view_dir))
                        self.projected_area = self.pixel_area * cos_theta
                        flux = radiance * self.projected_area / self.distance_to_observer**2
                    self.image[y, x] = flux
        return self.image



if __name__ == '__main__':
    obj_path = '/Users/l.beesley@bham.ac.uk/Documents/Lightcurves/satlight/models/reference/sphere.obj'
    sun_direction = Vector((1, 1, 0)).normalized()
    observer_direction = Vector((0, 1, 1)).normalized()

    # Without earthshine
    scene_no_es = Renderer(obj_path, sun_direction, observer_direction, 500e3,
                           add_earthshine=False)
    scene_no_es.initialise_scene()
    image_no_es = scene_no_es.render()

    # With earthshine
    scene_es = Renderer(obj_path, sun_direction, observer_direction, 500e3,
                        add_earthshine=True)
    scene_es.initialise_scene()
    image_es = scene_es.render()

    # Difference image
    diff = image_es - image_no_es

    # Plot all three
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    vmax = max(image_no_es.max(), image_es.max())
    
    axes[0].imshow(image_no_es, origin='upper', vmin=0, vmax=vmax)
    axes[0].set_title(f'Sun only\nTotal: {np.sum(image_no_es):.2e} W')
    axes[0].colorbar = plt.colorbar(axes[0].images[0], ax=axes[0])

    axes[1].imshow(image_es, origin='upper', vmin=0, vmax=vmax)
    axes[1].set_title(f'Sun + Earthshine\nTotal: {np.sum(image_es):.2e} W')
    axes[1].colorbar = plt.colorbar(axes[1].images[0], ax=axes[1])

    axes[2].imshow(diff, origin='upper', cmap='RdBu_r')
    axes[2].set_title(f'Difference (Earthshine contribution)\nTotal: {np.sum(diff):.2e} W')
    axes[2].colorbar = plt.colorbar(axes[2].images[0], ax=axes[2])

    plt.suptitle(f'Earthshine irradiance: {scene_es.earthshine_irradiance:.4f} W/m²', 
                 fontsize=11)
    plt.tight_layout()
    plt.show()

    # Sanity checks
    print(f"Sun only total:        {np.sum(image_no_es):.4e} W")
    print(f"Sun + Earthshine total:{np.sum(image_es):.4e} W")
    print(f"Earthshine contribution: {np.sum(diff):.4e} W")
    print(f"Earthshine fraction:   {np.sum(diff)/np.sum(image_no_es)*100:.2f}%")

    