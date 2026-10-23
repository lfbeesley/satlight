# The renderer

`Renderer` ray-traces the spacecraft mesh with Blender. It casts one ray per
pixel, evaluates the BRDF where each ray hits, and sums the result into the
irradiance at the observer.

## Calling it

```python
r = Renderer(obj_path, geometry=geom, brdf=brdf)
r.initialise_scene()          # load the mesh and create the camera (once)
r.set_attitude(euler_xyz=...) # optional
r.update(geom)                # Sun, observer and camera for this epoch
image = r.render()            # per-pixel irradiance, W/m²
flux = float(image.sum())
```

Call `update()` before each `render()`, and call `set_attitude()` before
`update()`.

## Output

`render()` returns an array of shape `(resolution_y, resolution_x)`. Each pixel
contributes

$$
\Delta E = f_r \cos\theta_i \; E_\odot \; \frac{A_{\text{pix}}}{d^2}
$$

where $E_\odot$ is the solar constant, $A_{\text{pix}}$ the area of one pixel
on the spacecraft and $d$ the range. The sum over pixels is the irradiance at
the observer in W/m².

After a render:

```python
r.hit_count       # pixels whose ray hit the spacecraft
r.shadow_count    # of those, how many were in shadow
r.image           # the last rendered array
```

## Camera

The camera is orthographic: the spacecraft is tiny as seen from the ground,
so all rays are effectively parallel, and the $1/d^2$ falloff is applied
analytically. With `auto_ortho_scale=True` (the default), `update()` sizes the
frame to the spacecraft's projected outline, plus a margin set by
`ortho_margin` (default 1.06).

A warning is printed if the spacecraft touches the edge of the frame.

## Attitude

```python
import numpy as np
r.set_attitude(euler_xyz=(0.0, 0.0, np.radians(30.0)))
```

Angles are in radians about the [LVLH axes](geometry.md#the-lvlh-frame): X
along-track, Y orbit normal, Z zenith. `(0, 0, 0)` is the mesh as imported.
Each call replaces the previous attitude rather than adding to it. To move
individual parts, see {doc}`articulation`.

## Options

`add_shadows` (default `True`)
: The spacecraft shadows itself. Adds one shadow ray per lit pixel.

`n_bounce` (default `1`)
: Number of reflections per light path. `n_bounce=2` adds light reflected
  between parts of the spacecraft, using `bounce_samples` (default 8) random
  directions per pixel. `bounce_seed` makes the result reproducible.

`smooth_normals` (default `False`)
: Interpolates surface normals across edges meeting at less than
  `smooth_angle_deg` (default 30°).

`add_earthshine` (default `False`)
: Adds sunlight reflected off the Earth. See {doc}`earthshine`.

`apply_observability` (default `True`)
: Multiplies the flux by `eclipse_type() × is_visible()` from the geometry, so
  an eclipsed or set satellite gives zero. Not available with a
  `ViewGeometry`, which has no Earth position.

`resolution` (default `(500, 500)`)
: Run time scales roughly with the number of pixels. To check convergence,
  render the same epoch at two resolutions and compare.

## Which part is which

```python
boxes = r.get_object_pixel_bboxes()   # {object_name: (xmin, ymin, xmax, ymax)}
```

gives each component's bounding box in the image, so you can tell which part
a bright region comes from.
