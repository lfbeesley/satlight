# Validation

The point of a physically-based renderer is that you can check it against
cases with known answers. satlight's test suite does that, and this page
records what is actually gated and how accurate the forward model measurably
is.

```{important}
Read this before fitting anything. Every retraction in satlight's development
history came from fitting a model before confirming the forward model
reproduced a case with a known answer.
```

## The analytic cases

### A flat plate

A Lambertian plate of area $A$ and albedo $\rho$, illuminated at incidence
$\theta_i$ and viewed at $\theta_r$, delivers irradiance

$$
E = \frac{\rho}{\pi} E_\odot \, A \cos\theta_i \cos\theta_r \, \frac{1}{d^2}
$$

This has no free parameters, so it tests the entire radiometric chain at once:
pixel area, the $1/d^2$ falloff, the BRDF normalisation, and the projected-area
cosine.

**Measured agreement: 0.08%.**

### A sphere

A Lambertian sphere of radius $R$ at phase angle $\alpha$:

$$
E = \frac{2\rho R^2}{3\pi d^2} E_\odot \left[\sin\alpha + (\pi - \alpha)\cos\alpha\right]
$$

This additionally tests the integration over a curved surface and the
phase-angle behaviour.

**Measured agreement: 0.1–0.25%**, across phase angle.

## Two surprises worth knowing about

These are both cases where the intuitive expectation is wrong, and both are
now encoded in the tests rather than in anyone's memory.

### Plate error does not converge monotonically

```{warning}
The plate's error **oscillates at the ±1-pixel-row level** with resolution
rather than converging smoothly. A tolerance tuned at one resolution will fail
at another, and passing at your chosen resolution means nothing.
```

The cause is quantisation: the plate's edge falls between pixel centres
differently at each resolution, so the number of pixels covering it steps
rather than scaling. The test therefore gates a **phase-averaged** error
across five resolutions, not the error at any single one.

### Smooth normals make a coarse sphere *darker*

```{warning}
`smooth_normals=True` is not an accuracy improvement. On a coarsely
tessellated sphere it measurably reduces the total flux.
```

Smoothing redistributes light over the disk; it does not change what the
geometry intercepts. The intuition that interpolated normals better
approximate a curved surface does not translate into a better integrated flux.
The test asserts **convergence with mesh refinement** instead, which is the
property that actually holds.

## Earthshine

The discretised Earth cap's total solid angle is gated against the analytic
spherical cap

$$ \Omega = 2\pi\left(1 - \cos\alpha\right) $$

and the default `n_rings=4` (88 patches) agrees with a 960-patch grid to
**0.001 mag**. See {doc}`earthshine`.

## Known limitations

Documented rather than fixed, because they are genuine and you should know
about them:

**Narrow specular lobes may not be converged.**
: A glint narrower than a pixel's angular footprint is under-sampled, and the
  peak height will depend on resolution. Check by re-rendering at higher
  resolution; if the peak moves, it is not converged.

**BRDFs do not conserve energy for you.**
: The specular lobe adds to the diffuse floor. `abg_from_peak` in particular
  can be driven to a total hemispherical reflectance well above 1. See
  {doc}`materials`.

**`AB_mag` is in-band only.**
: Applied to broadband `render()` output it is ~2.2 mag too bright. See
  {doc}`photometry`.

## Running the gates yourself

```bash
pytest                                  # fast offline tier (default)
pytest -m "slow or needs_bpy"           # the Blender renders, incl. these gates
pytest -m ""                            # everything
```

The analytic gates are in `tests/test_render.py` and
`tests/test_earthshine_extended.py`. They need Blender, so they are not in the
default run.
