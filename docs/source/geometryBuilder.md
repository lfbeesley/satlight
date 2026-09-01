# Generating an obj model with geometryBuilder

geometryBuilder is a tool to quickly make basic satellite models for use in satlight for lightcurve simulations.
## Setup

Start by setting up the save file for the geometry by calling 'setup()'. This creates the obj and corresponding mtl file used for the object. It also sets up a global rotation which applies to all geometry created in this file.
The mtl file is only used for graphical uses and is not read by satlight, which instead uses BRDFs. The graphical distinction may be useful when identifying which parts to apply which BRDFs to.

```python
from satlight import geometryBuilder

geometryBuilder.setup(
folder = "models\\saveFolder",          # filepath
name = "satelliteName",                 # file name
x_rotation = 0, 
y_rotation = 0, 
z_rotation = 0                          # global rotations x, y, z in degrees about the origin
)    
```

## Creating geometry

### Preexisting models

Premade obj files can be read in and added to the new file in the desired size, orientation, and position. 

```python
geometryBuilder.import_model(
file = "componentName"               # file name
folder = "Models\\Components"        # filepath, if left blank the save files folder will be used
scale = 1                            # scales the model by this factor
x_angle = 0, 
y_angle = 0, 
z_angle = 0,                         # rotations x, y, z in degrees about the origin
x_offset = 0, 
y_offset = 0, 
z_offset = 0                         # translations x, y, z in metres from the origin
)
```
### Prisms

Any regular prism can be generated either solid or hollow in the desired size, orientation, and position. The base geometry is calculated from the number of sides, radius, and height.
- Cylinders are made with sufficiently high side count for the desired fidelity of the model.
- Unit cubes have radius $\frac{\sqrt2}{2}$ and height 1
- Equilateral triangles have radius $\frac{2}{3}$
  
```python
geometryBuilder.prism_geometry(
name = "Cube",    
side_count = 4,  
radius = 0.7071,            # in metres
height = 1,                 # in metres
taper = 1,                  # scales down the top vertecies by this factor 
is_hollow = False,         
wall_thickness = 0,         # if is_hollow = True, interior faces will be generated this distance in from the exterior walls in metres
material = "MLI",           # makes the obj file reference the MLI material taken from materialLibrary
x_scale = 1, 
y_scale = 1, 
z_scale = 1,                # scales up the length in each direction x, y, z by this factor
x_angle = 0, 
y_angle = 0, 
z_angle = 0,                # rotations x, y, z in degrees about the origin
x_offset = 0,
y_offset = 0, 
z_offset = 0                # translations x, y, z in metres from the origin
)
```

### Parabolas

Parabolas can be generated with a desired radius and height as well as orientation and position. Detail level can also be specified to lower or increase model accuracy. Parabolas are used for communication dishes as well as approximating engine nozzles or aerodynamic covers. 

Parabolas are 2 dimensional with the normals pointing out of the interior face. To ensure correct rendering via 'workbench' and 'satlight', it is recommended to generate a second parabola with negative height and position it below the first, as well as a hollow prism with the same number of sides positioned between the parabolas.

```python
geometryBuilder.parabola_geometry(
name = "comms dish", 
radius = 0.5,                  # in metres
height = 0.2,                  # in metres
segments = 10,                 # number of sections along the arc of the parabola between the centre and the top
fidelity = 36,                 # number of sections around the circumference of the parabola
material = "Aluminium",        # makes the obj file reference the Aluminium material from materialLibrary
x_angle = 0, 
y_angle = 0, 
z_angle = 0,                   # rotations x, y, z in degrees about the origin
x_offset = 0, 
y_offset = 0, 
z_offset = 0                   # translations x, y, z in metres from the origin
)
```
