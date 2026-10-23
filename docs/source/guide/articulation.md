# Moving components

Solar arrays that track the Sun, an antenna that is repointed partway through a
pass: satlight renders whatever pose the spacecraft is in, but it does not move
individual parts for you. You do that with Blender's own API, `bpy`. This page
explains how the two fit together, and gives the recipe.

## Who changes what

satlight keeps no copy of the spacecraft. After `initialise_scene()` the mesh
lives in Blender's scene, and every `render()` casts rays at that scene **as it
is at that moment**. Moving a part and placing the Sun and camera are two
separate jobs that never touch each other:

```{figure} figures/pose_flow.*
:alt: Two inputs feed one shared Blender scene. On the left, you move parts by
  assigning o.matrix_world through bpy. On the right, renderer.update(geom)
  sets the Sun, observer and camera. renderer.render() then casts rays at the
  scene as it is now and returns the flux at the observer.
:width: 85%

`bpy` moves the spacecraft, `update()` moves the Sun and the camera, and
`render()` takes the picture.
```

Call | Effect
--- | ---
`o.matrix_world = ...` | Moves the part in Blender's scene. This, and only this, changes the pose.
`bpy.context.view_layer.update()` | Makes Blender apply the new matrices so the next ray cast sees them. `render()` also calls it, so this is a safety step.
`renderer.update(geom)` | Sets the Sun direction, observer direction and range for this epoch, and places the **camera**. It does not move any part.
`renderer.render()` | Casts rays through the camera at the parts where they now are, and returns the per-pixel irradiance.

## The order within a frame

```python
pose(...)                 # 1. move the parts       (bpy)
renderer.update(geom)     # 2. Sun, observer, camera (satlight)
flux = renderer.render()  # 3. take the picture
```

```{important}
Pose **before** `update()`. `update()` fits the camera frame around wherever
the parts are at that moment, so a solar array swung out afterwards can land
outside the image.
```

## Each component is its own object

The OBJ importer splits the file on its `o <name>` lines, so every named
component is a separate Blender object:

```python
import bpy
print([o.name for o in bpy.context.scene.objects if o.type == 'MESH'])
part = bpy.data.objects['antenna1']
```

These are the same names that `brdf=` matches against (see {doc}`materials`).
The importer also converts the file from Y-up to Z-up, so measure axes from the
imported objects rather than reading them off the `.obj` text.

## The recipe

Store each part's imported pose once, then rebuild every part's matrix from
it on each frame:

```python
matrix_world = body_attitude @ articulation @ rest[name]
```

Reading right to left: the pose as imported, then the part's own motion, then
the whole-spacecraft attitude. Because every frame starts again from `rest`,
nothing accumulates, and an articulation of zero restores the nominal pose.

```python
import bpy
import mathutils as mu
import numpy as np

renderer.initialise_scene()
meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
rest = {o.name: o.matrix_world.copy() for o in meshes}

body = mu.Euler(body_euler, 'XYZ').to_matrix().to_4x4()   # model -> LVLH


def rotation_about(pivot, axis, angle_rad):
    """Rotate about a line through `pivot`, not about the origin."""
    p = mu.Vector(pivot)
    return (mu.Matrix.Translation(p)
            @ mu.Matrix.Rotation(angle_rad, 4, mu.Vector(axis))
            @ mu.Matrix.Translation(-p))


def pose(local):
    """`local` maps part name -> articulation matrix; others stay rigid."""
    for o in meshes:
        o.matrix_world = body @ local.get(o.name, mu.Matrix.Identity(4)) @ rest[o.name]
    bpy.context.view_layer.update()
```

A translation is `mu.Matrix.Translation((dx, dy, dz))`, in metres.

### Finding a pivot and an axis

A principal-component fit to a part's vertices gives the axes most parts need:
the longest axis of a solar array is its hinge, and the shortest axis of a dish
is its boresight. Take the vertices in world coordinates, through
`matrix_world`. The raw `data.vertices` are in the file's frame, before
satlight re-centres the model, and a pivot taken from them is offset by that
shift.

```python
def principal_axes(name):
    obj = bpy.data.objects[name]
    v = np.empty(len(obj.data.vertices) * 3)
    obj.data.vertices.foreach_get('co', v)
    M = np.array(rest[name])
    pts = v.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
    centre = pts.mean(axis=0)
    _, _, axes = np.linalg.svd(pts - centre, full_matrices=False)
    return centre, axes          # axes[0] longest ... axes[2] shortest
```

### Sun-tracking arrays

For an array with hinge `h` and rest normal `n0`, both in the model frame, the
angle that turns the normal toward the Sun is

```python
sun_body = np.array(body.to_3x3()).T @ sun_lvlh
theta = np.arctan2(sun_body @ np.cross(h, n0), sun_body @ n0)
```

A single hinge cannot follow the Sun's component along the hinge itself, so
the normal is off by the Sun's declination relative to the array, a few degrees
for a GEO satellite.

## Do not mix with `set_attitude`

{meth}`~satlight.raytracer.Renderer.set_attitude` rotates **every** mesh from
its own stored import pose, which overwrites any articulation you have
applied. When you move components, put the body attitude into the matrix
yourself, as `body` above, and do not call `set_attitude`.
