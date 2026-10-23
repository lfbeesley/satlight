# Validation

The test suite checks the renderer against shapes with known analytic
answers.

## Flat plate

A Lambertian plate of area $A$ and albedo $\rho$, lit at incidence angle
$\theta_i$ and viewed at $\theta_r$ from range $d$:

$$
E = \frac{\rho}{\pi} E_\odot \, A \cos\theta_i \cos\theta_r \, \frac{1}{d^2}
$$

## Sphere

A Lambertian sphere of radius $R$ at phase angle $\alpha$:

$$
E = \frac{2\rho R^2}{3\pi d^2} E_\odot \left[\sin\alpha + (\pi - \alpha)\cos\alpha\right]
$$

## Earthshine

The total solid angle of the Earth patches is checked against the spherical
cap, $\Omega = 2\pi\left(1 - \cos\alpha\right)$.

## Running the tests

```bash
pytest                                  # fast tests only (default)
pytest -m "slow or needs_bpy"           # tests that render with Blender
pytest -m ""                            # everything
```

The plate and sphere tests are in `tests/test_render.py`, and the earthshine
tests in `tests/test_earthshine_extended.py`.
