# Earthshine

Earthshine is sunlight reflected off the Earth onto the spacecraft. It is off
by default.

```python
r = Renderer(obj_path, geometry=geom,
             add_earthshine=True,
             earth_albedo=0.3,
             earthshine_n_rings=4)
```

It is computed when the `Renderer` is created and again on every `update()`.
Set `add_earthshine` in the constructor; setting it on an existing `Renderer`
raises an error at `render()` unless you call `update()` first.

## How it is modelled

The sunlit part of the Earth visible from the spacecraft is split into patches
of equal solid angle. Each patch is treated as a light source from its own
direction, with irradiance $E = L\,\omega$ ($L$ the patch radiance, $\omega$
its solid angle), and goes through the surface BRDF like sunlight does. The
Earth is a Lambertian reflector with a uniform albedo.

`earth_albedo` (default `0.3`)
: Uniform albedo of the Earth.

`earthshine_n_rings` (default `4`)
: Resolution of the patch grid; 4 gives 88 patches. Run time grows with the
  number of patches.

`earthshine_shadows` (default: same as `add_shadows`)
: Whether the spacecraft can shadow itself from earthshine. This casts one
  shadow ray per patch per lit pixel, so it is slow.

## With `ViewGeometry`

The size of the Earth as seen from the spacecraft depends on altitude, which a
`ViewGeometry` doesn't contain. Pass it explicitly:

```python
vg = ViewGeometry(sun_lvlh, observer_lvlh, distance_km=1200.0,
                  altitude_km=550.0)
```
