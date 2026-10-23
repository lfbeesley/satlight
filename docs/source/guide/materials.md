# Materials and BRDFs

A BRDF (bidirectional reflectance distribution function) describes how a
surface reflects light. satlight assigns one to each mesh component.

## Assigning materials by name

Pass `brdf=` a dictionary keyed by part of the object name. A key matches if it
appears anywhere in the name, ignoring case, so `'solar'` matches
`solar_left_front` and `Solar_Panel_3`. `'default'` is used for anything that
doesn't match.

```python
brdf = {
    'solar':    ('lambertian', {'albedo': 0.08}),
    'radiator': ('lambertian', {'albedo': 0.60}),
    'default':  ('lambertian', {'albedo': 0.25}),
}
r = Renderer(obj_path, geometry=geom, brdf=brdf)
```

Because matching is by substring, a short key can match more than you intend:
`'ant'` matches both `antenna` and `mount_bracket_anterior`.

`brdf=` also accepts a single spec for the whole model:

```python
brdf='lambertian'                          # a built-in model by name
brdf=('lambertian', {'albedo': 0.3})       # a model and its parameters
brdf=my_func                               # your own function (see below)
```

## Built-in models

| Name | Parameters (defaults) |
|---|---|
| `lambertian` | `albedo=0.5` |
| `phong` | `albedo=0.5`, `specular=0.5`, `shininess=32` |
| `blinn_phong` | `albedo=0.5`, `specular=0.5`, `shininess=32` |
| `cook_torrance` | `albedo=0.3`, `roughness=0.5`, `metallic=0.1` |
| `oren_nayar` | `albedo=0.5`, `roughness=0.5` |
| `ashikhmin_shirley` | `albedo=0.3`, `specular=0.05`, `nu=100`, `nv=100` |
| `abg` | `A=1e-3`, `B=1e-4`, `g=2.0` |

### Presets

`satellite` picks a model and parameters by surface type:

```python
brdf = {
    'solar':   ('satellite', {'surface_type': 'solar_panel'}),
    'antenna': ('satellite', {'surface_type': 'antenna'}),
    'default': ('satellite', {'surface_type': 'body'}),
}
```

| `surface_type` | Model and parameters |
|---|---|
| `solar_panel` | Blinn-Phong: albedo 0.15, specular 0.85, shininess 64 |
| `body` | Oren-Nayar: albedo 0.30, roughness 0.4 |
| `antenna` | Cook-Torrance: albedo 0.80, roughness 0.3, metallic 0.9 |

With `model='ashikhmin_shirley'`, all three use Ashikhmin-Shirley instead.

## Writing your own

A BRDF is a function called once per lit pixel per light source:

```python
value = brdf_func(normal, light_dir, view_dir, **kwargs)
```

- `normal`: unit surface normal at the hit point
- `light_dir`: unit vector towards the light source
- `view_dir`: unit vector towards the observer

It must return $f_r \cos\theta_i$ (the BRDF times the cosine of the incidence
angle), not $f_r$ alone, and `0.0` when the surface faces away from the light.

```python
import numpy as np

def my_lambertian(normal, light_dir, view_dir, albedo=0.3):
    n_dot_l = float(np.dot(normal, light_dir))
    if n_dot_l <= 0.0:
        return 0.0
    return albedo / np.pi * n_dot_l

r = Renderer(obj_path, geometry=geom, brdf=(my_lambertian, {'albedo': 0.3}))
```

This gives the same result as the built-in `lambertian`, which is a useful
check that a custom function is wired up correctly.
