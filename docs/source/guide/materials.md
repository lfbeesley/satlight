# Materials and BRDFs

A BRDF — bidirectional reflectance distribution function — is what turns a
geometric hit into a radiance. satlight attaches one per mesh component, and
this page covers how that attachment works, what the built-in models are, and
the contract a custom one has to satisfy.

## Materials are assigned by object name

This is the one rule worth learning properly.

```{important}
A `brdf` key matches if it appears **anywhere** in the object's name, compared
case-insensitively. `'solar'` matches `solar_left_front`, `Solar_Panel_3` and
`SOLAR`. `'default'` catches everything unmatched.
```

```python
brdf = {
    'solar':    ('lambertian', {'albedo': 0.08}),   # cells are dark
    'radiator': ('lambertian', {'albedo': 0.60}),   # white thermal surfaces
    'boom':     ('lambertian', {'albedo': 0.20}),
    'bus':      ('lambertian', {'albedo': 0.30}),
    'default':  ('lambertian', {'albedo': 0.25}),
}
r = Renderer(obj_path, geometry=geom, brdf=brdf)
```

Substring matching is convenient and occasionally surprising: a key of `'ant'`
matches `antenna` *and* `mount_bracket_anterior`. Name your components with
that in mind — it is also why the bundled example model uses explicit
prefixes.

Four spellings are accepted anywhere a material spec is wanted:

```python
brdf='lambertian'                          # a registry name, applied to everything
brdf=my_func                               # any callable (see the contract below)
brdf=('lambertian', {'albedo': 0.3})       # a (spec, kwargs) pair
brdf={'solar': ..., 'default': ...}        # per-object, keyed by name substring
```

## The built-in models

| Name | Kind | Key parameters |
|---|---|---|
`lambertian` | ideal diffuse | `albedo`
`phong` | empirical specular | `albedo`, `specular`, `shininess`
`blinn_phong` | empirical specular, half-vector | `albedo`, `specular`, `shininess`
`cook_torrance` | microfacet, physically based | `albedo`, `roughness`, `metallic`
`oren_nayar` | rough diffuse (retroreflective) | `albedo`, `roughness`
`ashikhmin_shirley` | anisotropic microfacet | `albedo`, `specular`, `nu`, `nv`
`abg` | empirical scatter (Zemax ABg) | `A`, `B`, `g`

Choosing between them, briefly:

**`lambertian`**
: The right default. Start here, and only add a specular term when you have a
  reason — it is the only model whose total reflectance you can reason about
  without measuring it.

**`oren_nayar`**
: A rough diffuse surface. Backscatters more than Lambert at high phase angle,
  which is the correct behaviour for MLI and most painted surfaces.

**`cook_torrance`**
: Use when you want a *physically based* specular lobe with Fresnel and
  geometric attenuation. The microfacet distribution conserves energy, unlike
  Phong.

**`ashikhmin_shirley`**
: Anisotropic — `nu` and `nv` differ — which is what you want for a brushed or
  grooved surface, and what much of the published SSA literature uses.

**`abg`**
: A sharp specular spike sitting on a broad shoulder, which no single
  microfacet lobe reproduces. This is the model to reach for when a measured
  light curve has a glint far narrower than Phong can make.

### Spacecraft presets

`satellite` bundles representative parameters for three common surface
classes, so you can get a plausible spacecraft without choosing seven numbers:

```python
brdf = {
    'solar':   ('satellite', {'surface_type': 'solar_panel'}),
    'antenna': ('satellite', {'surface_type': 'antenna'}),
    'default': ('satellite', {'surface_type': 'body'}),
}
```

| `surface_type` | Model used | Rationale |
|---|---|---|
`solar_panel` | Blinn-Phong, albedo 0.15, specular 0.85, shininess 64 | dark and strongly specular
`body` | Oren-Nayar, albedo 0.30, roughness 0.4 | mostly diffuse MLI
`antenna` | Cook-Torrance, albedo 0.80, roughness 0.3, metallic 0.9 | bright metal

Passing `model='ashikhmin_shirley'` renders all three with A-S instead, for
like-for-like comparison against papers that use it throughout.

## The calling contract

`render()` calls a BRDF once per lit pixel per source:

```python
value = brdf_func(normal, light_dir, view_dir, **kwargs)
```

`normal`
: unit shading normal at the hit point

`light_dir`
: unit vector pointing **to** the source — the Sun, or one Earth patch

`view_dir`
: unit vector pointing **to** the observer

All three are in the render frame (LVLH, Z zenith). The return is a single
float in sr⁻¹, so kernel × irradiance = radiance toward the observer.

### The convention that matters

```{warning}
**The returned value is $f_r \cos\theta_i$, not $f_r$.** The incident cosine is
folded into the kernel, so every built-in model ends with a `* n_dot_l`.
```

Write a custom BRDF without it and the body comes out uniformly lit however
the Sun strikes it — a wrong answer that still looks like a spacecraft, which
is the worst failure mode available.

A BRDF must also return exactly `0.0` when $\hat{n} \cdot \hat{l} \le 0$. The
renderer does not clamp for you, and a negative return would *subtract* light.

### Writing one

```python
import numpy as np

def my_brdf(normal, light_dir, view_dir, albedo=0.3):
    n_dot_l = float(np.dot(normal, light_dir))
    if n_dot_l <= 0.0:
        return 0.0                    # never negative, never unclamped
    return albedo / np.pi * n_dot_l   # f_r * cos(theta_i)

r = Renderer(obj, geometry=geom, brdf=(my_brdf, {'albedo': 0.3}))
```

That reproduces `lambertian` exactly, and is a good sanity check: swap it in,
and the flux should not change.

### It is on the hot path

| Source | Calls |
|---|---|
direct Sun | 1 per lit pixel
earthshine | 1 per lit Earth patch per lit pixel (88 patches at LEO, `n_rings=4`)
`n_bounce > 1` | `bounce_samples` extra per pixel per extra bounce

A 400-pixel frame makes roughly 9000 calls for sunlight alone. Keep a custom
BRDF **scalar and cheap** — numpy's per-call overhead on 3-vectors costs more
than the arithmetic it performs. Use `float(np.dot(...))`, not array
operations over a single vector.

## Energy conservation is your responsibility

```{warning}
The specular lobe **adds** to the diffuse floor rather than being budgeted
within a total. Raising `specular` raises the surface's overall hemispherical
reflectance too — it does not redistribute a fixed budget.
```

So `albedo=0.6, specular=0.8` is not "60% diffuse, 80% of what's left
specular". It is a surface that can reflect more light than reaches it, at
which point the light curve is unphysical no matter how plausible its shape.
This is an easy mistake to make while tuning a model to match data: the glint
gets sharper, the mean gets brighter, and nothing complains.

Two checks worth running when you tune a specular model:

1. **Integrate the hemisphere.** Total reflectance $\rho$ must be $\le 1$. A
   fitted parameter set reaching $\rho \approx 7$ is not a bright satellite,
   it is a broken one.
2. **Watch the mean, not the peak.** If sharpening a glint also raised the
   out-of-glint magnitude by a magnitude or more, the extra brightness came
   from the diffuse floor, not from the specular lobe.

```{note}
`abg`'s `peak = A/B` is particularly easy to over-drive this way, because the
peak and the integrated reflectance are controlled by the same two parameters.
```

## Which component produced the glint?

Because materials are per-object, you can ask where the light came from:

```python
boxes = r.get_object_pixel_bboxes()     # {object_name: (xmin, ymin, xmax, ymax)}
```

Combined with the rendered image, that attributes a peak to a named component
— the thing you usually actually want to know.
