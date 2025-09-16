
import bpy
import os
import csv
import numpy as np
import glob
import OpenImageIO as oiio

def initial_setup(samples, bounce_limit):
    '''
    Setup initial scene and sample settings and raytracing using the GPU
    '''
    # wipe all existing objects and settings from the scene - very important
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.cycles.samples = samples
    scene.cycles.use_denoising = False
    scene.cycles.use_adaptive_sampling = False

    # glossy bounces are the only ones we want
    scene.cycles.max_bounces = bounce_limit
    scene.cycles.glossy_bounces = bounce_limit 

    scene.cycles.diffuse_bounces = bounce_limit
    scene.cycles.transmission_bounces = bounce_limit
    scene.cycles.volume_bounces = bounce_limit
    scene.cycles.transparent_max_bounces = bounce_limit

    # dont clamp low intensity light
    scene.cycles.light_threshold = 0.0
    scene.cycles.min_bounces = 0

    return scene

def add_cad(filepath):
    '''
    Add a CAD model based on the filepath to where it is stored on the PC.
    We centre the model centre-of-mass at the origin, and set its rotation mode to
    quaternion for later attitude adjustments.
    
    '''
    # Remove existing meshes
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            bpy.data.objects.remove(obj, do_unlink=True)
    
    # Import CAD file based on extension
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.stl':
        bpy.ops.wm.stl_import(filepath=filepath) # this is the one that seems to work best, stick to stls whenever possible
    elif ext == '.obj':
        bpy.ops.wm.obj_import(filepath=filepath)
    elif ext == '.fbx':
        bpy.ops.wm.fbx_import(filepath=filepath)
    else:
        print(f"Unsupported CAD format: {ext}")
        return None

    # Get the imported mesh object(s)
    imported_objs = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
    if not imported_objs:
        print("No mesh imported.")
        return None

    mesh_obj = imported_objs[0]

    # make sure the object is cented on its center of mass
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='MEDIAN')

    # Apply geometry shift so COM is at object origin
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bpy.ops.object.location_clear(clear_delta=False)

    mesh_obj.location = (0, 0, 0) # puts explicitly at the origin
    mesh_obj.rotation_mode = 'QUATERNION'

    return mesh_obj

def add_material(spacecraft_mesh, material_type='gloss'):
    '''
    Adding the material to the mesh that specifies the scattering mechanisms. The python lines
    that set this up are quite unintuitive compared to the GUI shader editor, but tools exist to
    convert a shader editor-generated material to a python script. 
    '''

    # create and name the material and setup links and nodes commands and variables
    mat = bpy.data.materials.new(name=material_type)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Glossy shader setup
    glossy = nodes.new("ShaderNodeBsdfGlossy"); glossy.inputs["Roughness"].default_value = 0.4; glossy.distribution = 'GGX'
    output = nodes.new("ShaderNodeOutputMaterial"); output.target = 'CYCLES'

    # Connect the shader to output
    links.new(glossy.outputs[0], output.inputs["Surface"])

    # add material to the mesh
    spacecraft_mesh.data.materials.append(mat)

def add_observer(scene, size):
    '''
    Add the orthographic camera to the scene - this is the 'transmitter' in radar terms.
    Define a few default parameters to setup later edits. Placed at the origin to start.
    '''
    bpy.ops.object.camera_add(location=(0, 0, 0), rotation=(0, 0, 0))
    observer = bpy.context.object
    observer.rotation_mode = 'QUATERNION'
    observer.data.type = 'ORTHO'
    observer.data.ortho_scale = size
    observer.data.clip_start = 0.00001
    observer.data.clip_end = 1e6 # limit is 1 million km (this takes for granted that the camera position is being scaled by 1/1000)
    scene.camera = observer

    # define an empty object at the origin to enforce tracking
    if "Origin" not in bpy.data.objects:
        origin = bpy.data.objects.new("Origin", None)
        bpy.context.collection.objects.link(origin)
        origin.location = (0.0, 0.0, 0.0)
    else:
        origin = bpy.data.objects["Origin"]

    # track origin no matter the position 
    c = observer.constraints.new(type="TRACK_TO")
    c.target = origin
    c.track_axis = 'TRACK_NEGATIVE_Z'
    c.up_axis = 'UP_Y'

    return observer

def add_sun(flux=1367, angular_size=0.00918043 ,bounce_limit=10):
    '''
    Add the sun to the scene 
    Define a few default parameters to setup later edits. Placed at the origin - sun is directional so doesnt need to move
    '''

    bpy.ops.object.light_add(type='SUN', location=(0,0,0))
    sun = bpy.context.object
    sun.rotation_mode = 'QUATERNION'
    sun.data.cycles.max_bounces = bounce_limit
    sun.data.cycles.cast_shadow = True
    sun.data.energy = flux
    sun.data.angle = angular_size

    # define an empty object at the origin to enforce tracking
    if "Origin" not in bpy.data.objects:
        origin = bpy.data.objects.new("Origin", None)
        bpy.context.collection.objects.link(origin)
        origin.location = (0.0, 0.0, 0.0)
    else:
        origin = bpy.data.objects["Origin"]

    # track origin no matter the position 
    c = sun.constraints.new(type="TRACK_TO")
    c.target = origin
    c.track_axis = 'TRACK_NEGATIVE_Z'
    c.up_axis = 'UP_Y'

    return sun

def perform_render(scene, save_dir, filename,gt=False):
    '''
    This function is where the rendering is started. Pass in the scene after all setup is done, and 
    a directory to save the files.
    '''
    
    # if ground truth, save as PNG, else save as EXR for ISAR processing
    if gt:
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGB'
        scene.render.image_settings.color_depth = '8'
    else:
        scene.render.image_settings.file_format = 'OPEN_EXR'
        scene.render.image_settings.color_mode = 'RGB'
        scene.render.image_settings.color_depth = '32'

    scene.render.filepath = os.path.join(save_dir, filename)
    bpy.ops.render.render(write_still=True)

def animate(geometry, observer, sun):
    '''
    parameters:

        geometry_array is a list of geometry objects that define the scene geometry at unique times
    
    Setup the keyframes for the animation of the frames. This means setting the camera and light positions, 
    and quaternion rotations for the camera and light. 

    The positions in this function are in the satellite-centred ICRF.

    We also scale the position by a factor of 1/1000

    We also rotate the target by a quaternion that reflects its attitude at the central moment of each frame.
    '''
    scale = 1/1000 # Scale factor for the camera positions

    for i in range(len(geometry.times)):

        # Set observer position and rotation and assign keyframes
        # print(geometry.vectors['sat_to_obs_distance'][i]*geometry.vectors['sat_to_obs_unit'][i,:])
        observer.location = geometry.vectors['sat_to_obs_distance'][i]*geometry.vectors['sat_to_obs_unit'][i,:]*scale
        observer.keyframe_insert(data_path="location", frame=i)

        # Set sun rotation and assign keyframes
        # print(geometry.vectors['sun_to_sat_unit'][i])
        sun.location = -geometry.vectors['sun_to_sat_unit'][i,:]
        sun.keyframe_insert(data_path="location", frame=i)

        # # spacecraft attitude quaternion at each time instant
        # qw = float(row['qw'])
        # qx = float(row['qx']) 
        # qy = float(row['qy'])
        # qz = float(row['qz'])
        
        # set spacecraft attitude and assign keyframes
        # spacecraft.rotation_quaternion = (qw, qx, qy, qz)
        # spacecraft.keyframe_insert(data_path="location", frame=frame)
        # spacecraft.keyframe_insert(data_path="rotation_quaternion", frame=frame)

def create_lightcurve(datapath):
    '''
    This function loads the renders and creates the lightcurve from the OpenEXR files
    '''
    
    def read_exr_channels(filename):
        inp = oiio.ImageInput.open(filename)
        spec = inp.spec()
        data = inp.read_image(format=oiio.FLOAT)
        inp.close()
        arr = np.frombuffer(data, dtype=np.float32).reshape(spec.height, spec.width, spec.nchannels)
        R, G, B = arr[..., 0], arr[..., 1], arr[..., 2] # read the red, green and blue channels of the files
        return R, G, B
    
    # # renders are loaded using glob from the renders directory
    render_files = sorted(glob.glob(os.path.join(datapath, "renders", "*.exr")))

    flux=[]
    for fname in render_files:
        red, green, blue = read_exr_channels(fname)
        
        luminance = 0.2126 * red + 0.7152 * green + 0.0722* blue
        # Total light flux (sum of luminance values)
        flux.append(np.mean(luminance))

    return flux

def render(geometry, datapath, CADpath, pixel_size=0.1, samples=1, bounce_limit=10):
    '''
    This is the main function in this file. Runs all other functions to create a blender scene with a CAD, 
    observer and sun and renders a series of frames, which are then read in sequentially to obtain the lightcurve
    '''
    # make directories for the renders
    render_dir = os.path.join(datapath, 'renders')
    ground_truth_dir = os.path.join(datapath,'ground_truth')
    os.makedirs(ground_truth_dir, exist_ok=True)
    os.makedirs(render_dir, exist_ok=True)

    # Set up the scene and add the CAD model of the target
    scene = initial_setup(samples=samples, bounce_limit=bounce_limit)
    spacecraft_mesh = add_cad(CADpath)
    
    # Calculate largest dimension of CAD to create the camera and light at the correct scale
    dims = spacecraft_mesh.dimensions
    box_size = np.sqrt(dims.x**2 + dims.y**2 + dims.z**2) * 1.1 # 10% padding for safety...
    # note that this padding may not generalise to every shape, but for most targets it seems to work.
    
    # use dimensions of the CAD to set the render resolution
    resolution = int(box_size / pixel_size)
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    
    # add the material to the mesh
    add_material(spacecraft_mesh, material_type='gloss')
    obs = add_observer(scene, size=box_size) # add observer
    sun = add_sun() # add sun
    animate(geometry, obs, sun) #setup the animation 
    
    # for each frame, render the exr files and the png ground truths and save them accordingly
    for frame in range(len(geometry.times)+1):
        scene.frame_set(frame)
        perform_render(scene, render_dir, filename=f'{frame:04d}.exr')
        perform_render(scene, ground_truth_dir, filename=f'{frame:04d}.png', gt=True)

    # generate and return lightcurve
    return create_lightcurve(datapath)



