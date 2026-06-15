from satlight import raytracer
import numpy as np
import matplotlib.pyplot as plt
from mathutils import Vector
import importlib
importlib.reload(raytracer)
import math

obj_path = r'/Users/l.beesley@bham.ac.uk/Documents/Lightcurves/satlight/data/models/Skynet 5D/skynet.stl'

sun_direction = Vector((0, 0.5, 0.5)).normalized()
observer_direction = Vector((0, 0.5, 0.5)).normalized()

scene = raytracer.Renderer(obj_path, sun_direction, observer_direction, 35786e3, resolution=(750,750), rotation=(0, 270, 0))
scene.initialise_scene()
scene.preview_lvlh_views()

image = scene.render()      

plt.figure(figsize=(8, 8 * scene.resolution_y / scene.resolution_x))
plt.imshow(image, cmap='gray', origin='upper')
plt.colorbar()
plt.title('Rendered Image')
plt.show()
