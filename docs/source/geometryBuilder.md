# Generating an obj model with geometryBuilder

geometryBuilder is a tool to quickly make basic satellite models for use in satlight's lightcurve simulations. It uses premade models and basic procedural shapes to construct a low fidelity object, enabling easy iteration of designs to aid in reducing reflected light visible on the ground.
## Setup

Start by setting up the save file for the geometry by calling ```setup()``` . This creates the obj and corresponding mtl file used for the object. It also sets up a global rotation which applies to all geometry created in this file.
The mtl file is only used for graphical uses and is not read by satlight, which instead uses BRDFs. The graphical distinction may be useful when identifying which parts to apply which BRDFs to.

```python
from satlight import geometryBuilder

geometryBuilder.setup(
folder = "models\saveFolder",          # filepath
name = "satelliteName",                 # file name
x_rotation = 0, 
y_rotation = 0, 
z_rotation = 0                          # global rotations x, y, z in degrees about the origin
)    
```
## materialLibrary

materialLibrary.txt contains information about the materials to be rendered. It is useful for distinguishing parts when creating the model, but materials set here are not used by satlight to calculate brightness. Materials can be left blank if desired. New materials can be added by saving them to materialLibrary.txt as formatted in an mtl file.
Available materials:
- Aluminium
- Antenna
- omniAntenna
- MLI
- Solar_Panel
- Copper
- Steel
## Creating geometry

### Pre-existing models

Premade obj files can be read in and added to the new file in the desired size, orientation, and position. 

```python
geometryBuilder.import_model(
file = "componentName"               # file name
folder = "models\components"        # filepath, if left blank the save files folder will be used
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

Any regular prism can be generated either solid or hollow in the desired size, orientation, and position. The base geometry is calculated from the number of sides, radius, and height. The origin is at the geometric centre.
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

Parabolas can be generated with a desired radius and height as well as orientation and position. Detail level can also be specified to lower or increase model accuracy. The origin is at the base of the parabola. Parabolas are used for communication dishes as well as approximating engine nozzles or aerodynamic covers. 


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


## Box wing satellite example

A simple box wing satellite with an antenna on the front face comprised of only 4 parts:

```python
from satlight import geometryBuilder 

geometryBuilder.setup("models\examples", "Box wing satellite")

geometryBuilder.prism_geometry("Bus", 4, 0.7071, 1)
geometryBuilder.prism_geometry("Solar panel 1", side_count = 4, radius = 0.7071, height = 1, material = "Solar_Panel", x_scale = 2, z_scale = 0.05, x_offset = 1.5, x_angle = -60)
geometryBuilder.prism_geometry("Solar panel 2", side_count = 4, radius = 0.7071, height = 1, material = "Solar_Panel", x_scale = 2, z_scale = 0.05, x_offset = -1.5, x_angle = -60)
geometryBuilder.parabola_geometry("Comms dish", radius = 0.3, height = 0.1, segments = 10, fidelity = 18, material = "Antenna", x_angle = 90, y_offset = -0.5)
```

Rendering this object shows a main bus 1x1x1 metres, two 2x1 metre solar panels angled 30 degrees from the observer, and a 0.6 metre antenna.

![[Box wing satellite.png|700]]


## Component library

A library of premade components is available for use to enable quick, higher complexity modeling. Any model or component made can be used alongside these.
### Solar panels
- Low detail panels with a separate back face in 1x1, 1x2, and 1x3 metre sizes
- High detail panels with many separate panel segments and a separate back face in 1x1, 1x2, and 1x3 metre sizes
- 4 mounts to connect the panels to the main bus
- High and low detail Intelsat style panel, 8 metre long with triangular mount
- Low detail Skynet style panel, 14.12 metre panel with hanger mount
### Propulsion
- Ion thrusters - a low poly gridded ion thruster and hall effect thruster 
- RL10C-1, engine bell, plumbing represented as single cylinder
### Communications
- 1 metre antenna with receiver
- 1x0.5 meter antenna with receiver
- Intelsat style antenna, 2.2 metre diameter with surface mount
- Front facing omni-directional antenna, 3.55 metre long
### Buses
- Intelsat style bus
- Skynet style bus
### Example models
- Intelsat - one with high definition panels and one with low definition panels
- Skynet
- Centaur upper stage - low fidelity model
- Box wing satellite example
- 23 LEO satellites from [NASA](https://science.nasa.gov/3d-resources/)
