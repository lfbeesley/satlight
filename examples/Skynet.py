from satlight import raytracer
import numpy as np
import matplotlib.pyplot as plt
from mathutils import Vector


obj_path = r'/Users/l.beesley@bham.ac.uk/Documents/Lightcurves/satlight/data/models/Skynet 5D/skynet.stl'
sun_direction = Vector((0, 1,1)).normalized()
observer_direction = Vector((0, 1, 1)).normalized()

scene = raytracer.Renderer(obj_path, sun_direction, observer_direction, 3600e3)
scene.initialise_scene()
image = scene.render()

plt.figure(figsize=(8, 8 * scene.resolution_y / scene.resolution_x))
plt.imshow(image, cmap='gray', origin='upper')
plt.colorbar()
plt.title('Rendered Image')
plt.show()

