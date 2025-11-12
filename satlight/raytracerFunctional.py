import bpy
import numpy as np
from mathutils import Vector
import matplotlib.pyplot as plt
from tqdm import tqdm
import BRDF
from hide_warnings import hide_warnings

def lambertian_brdf(normal, light_dir, view_dir, albedo = 1):
    """Lambertian shading"""
    n_dot_l = max(0, np.dot(normal, light_dir))
    return albedo / np.pi * n_dot_l

class Renderer:
    """ To manage and render the object in blender and perform the ray-casting."""

    def __init__(self, obj_path, sun_direction, observer_direction, distance_to_observer, resolution=(500,750), solar_constant = 1360):
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
        self.distance_to_observer = distance_to_observer * 1e3 # in m
        self.sun_direction = sun_direction # may need to change to Vector(sun_direction) depending on geometry output
        self.observer_direction = observer_direction

    def initialise_scene(self):
        # Set-up scene
        bpy.ops.wm.read_factory_settings(use_empty=True)

        # Add/center object and return orth_scale
        self.ortho_scale = self._load_and_center_object(self.obj_path) # 1.1 for 10% padding
        
        # Add lighting
        self.add_light()

        # Add camera
        self.add_camera()


    @hide_warnings()
    def _load_and_center_object(self, obj_path):
        '''Determine file type and import'''
        if obj_path.endswith('.obj'):
            self.obj_type = '.obj'
            bpy.ops.wm.obj_import(filepath=obj_path, filter_obj=True, use_split_objects=False, use_split_groups=False)
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

    def add_light(self):
        # Add sun light at sun direction
        light_data = bpy.data.lights.new(name="Sun", type='SUN')
        light_object = bpy.data.objects.new(name="Sun", object_data=light_data)
        bpy.context.collection.objects.link(light_object)
        #CHECK SUN DIRECTION


    def add_camera(self):
        self.camera_object = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
        self.camera_object.data.type = 'ORTHO'
        bpy.context.collection.objects.link(self.camera_object)
        bpy.context.scene.camera = self.camera_object

        # Set camera to orthographic
        self.camera_object.data.ortho_scale = self.ortho_scale # Width of camera along x-axis

        # Set camera position at specified distance
        self.camera_object.location = self.observer_direction.normalized() * self.ortho_scale  # multiplying to be outside of box...
        direction = Vector((0, 0, 0)) - self.camera_object.location

        # Point camera at geometric center (origin)
        self.camera_object.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler() #-Z is the cameras conventional 'look direction' point that toward origin

        # Calculate camera pixel size
        self.pixel_size_x = self.ortho_scale / self.resolution_x
        self.pixel_size_y = self.ortho_scale / self.aspect_ratio / self.resolution_y 
        self.pixel_area = self.pixel_size_x * self.pixel_size_y

        print(f"\nCamera configuration:")
        print(f"Camera position: ({self.camera_object.location.x:.3g}, {self.camera_object.location.y:.3g}, {self.camera_object.location.z:.3g}) m")
        print(f"Camera distance: {self.camera_object.location.magnitude:.3g} m ")
        print(f"Resolution: {self.resolution_x} x {self.resolution_y}")
        print(f"Field size: {self.ortho_scale:.3g} x {self.ortho_scale / self.aspect_ratio:.3g} m")
        print(f"Pixel size: {self.pixel_size_x:.3g} x {self.pixel_size_y:.3g} m")
        print(f"Pixel area: {self.pixel_area:.3g} m²")




    def render(self):
        print(f"\nStarting ray cast for {self.resolution_x}x{self.resolution_y} pixels...")

        # Ray casting setup
        self.scene = bpy.context.scene
        self.scene.render.resolution_x = self.resolution_x
        self.scene.render.resolution_y = self.resolution_y
        self.depsgraph = bpy.context.evaluated_depsgraph_get()

        # Get camera matrix for transforming rays
        cam_matrix = self.camera_object.matrix_world

        ray_dir_world = (cam_matrix @ Vector((0, 0, -1)) - cam_matrix @ Vector((0, 0, 0))).normalized()
        ray_dir = np.array(ray_dir_world)
        
        # Output
        image = np.zeros((self.resolution_y, self.resolution_x))
        projected_areas = []
        self.hit_count = 0
        self.shadow_count = 0

        # Ray cast for each pixel
        for y in tqdm(range(self.resolution_y)):
            for x in range(self.resolution_x):
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
                    shadow_origin = hit_point + hit_normal * 0.001
                    shadow_result, *_ = self.scene.ray_cast(self.depsgraph, shadow_origin, self.sun_direction)
                    
                    if shadow_result:
                        self.shadow_count += 1
                        flux = 0 
                    else:
                        view_dir = -ray_dir
                        brdf = lambertian_brdf(hit_normal, self.sun_direction, view_dir)
                        radiance = brdf * 1360

                        # Calculated projected area
                        cos_theta = max(0,np.dot(hit_normal, view_dir))

                        projected_area = self.pixel_area * cos_theta  
                        projected_areas.append(projected_area)   

                        flux = radiance * projected_area / self.distance_to_observer**2            
                    
                    image[y, x] = flux
        return image


            


if __name__ == '__main__':
    obj_path = '/Users/l.beesley@bham.ac.uk/Documents/Lightcurves/satlight/data/models/Starlink/starlink_HR/uploads_files_3710445_SpaceXStarlinkSatelliteHighRes.obj/SpaceXStarlinkSatelliteHighRes.obj'
    sun_direction = Vector((0, 10, 0)).normalized()
    camera_direction = Vector((0, 1, 1)).normalized()

    scene = Renderer(obj_path, sun_direction, camera_direction, 550e3)
    scene.initialise_scene()
    image = scene.render()

    plt.figure(figsize=(8, 8 * scene.resolution_y / scene.resolution_x))
    plt.imshow(image, cmap='gray', origin='upper')
    plt.colorbar()
    plt.title('Rendered Image')
    plt.show()

    print("\nRay casting complete!")
    print(f"Total measured Intensity: {np.sum(image):.2e} W")


