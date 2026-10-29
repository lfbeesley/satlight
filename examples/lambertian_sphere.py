"""Render a Lambertian sphere over phase angle and check it against the analytic curve.

A diffuse sphere of radius R and albedo rho, lit by a solar constant E and seen
from distance d at phase angle alpha, gives an irradiance at the observer of

    F(alpha) = E rho R^2 (2 / 3 pi) [sin(alpha) + (pi - alpha) cos(alpha)] / d^2

There is no orbit here, so the viewing geometry is given directly with a
ViewGeometry: the Sun and observer directions in the satellite's LVLH frame,
and the range in km.

Expect agreement to about 0.001 mag up to 140 deg phase. Beyond that the lit
crescent is only a few pixels wide and the difference grows (0.025 mag at
170 deg); raise the resolution if you need it.

Run from the repository root:
    python examples/lambertian_sphere.py
"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from satlight.geometry import ViewGeometry
from satlight.raytracer import Renderer

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, '..', 'models', 'reference', 'sphere.obj')   # radius 1 m

E_SUN = 1361.0          # W/m^2
RHO = 0.5
RADIUS_M = 1.0
DIST_KM = 1000.0
PHASES = np.arange(0.0, 171.0, 10.0)


def analytic_flux(alpha_deg):
    a = np.radians(alpha_deg)
    return (E_SUN * RHO * RADIUS_M ** 2 * 2 / (3 * np.pi)
            * (np.sin(a) + (np.pi - a) * np.cos(a)) / (DIST_KM * 1e3) ** 2)


def view(alpha_deg):
    """Sun overhead (+Z), observer tilted away from it by the phase angle."""
    a = np.radians(alpha_deg)
    return ViewGeometry(sun_lvlh=[0.0, 0.0, 1.0],
                        observer_lvlh=[np.sin(a), 0.0, np.cos(a)],
                        distance_km=DIST_KM)


r = Renderer(MODEL, geometry=view(0.0), resolution=(400, 400),
             brdf=('lambertian', {'albedo': RHO}), smooth_normals=True,
             add_shadows=False)
r.initialise_scene()

rendered = []
for alpha in PHASES:
    r.update(view(alpha))
    rendered.append(float(np.sum(r.render())))     # W/m^2 at the observer
rendered = np.array(rendered)
expected = analytic_flux(PHASES)

to_mag = lambda f: -2.5 * np.log10(f) - 18.915     # broadband V, see guide/photometry
print(' alpha   rendered (W/m^2)   analytic   difference (mag)')
for a, f, g in zip(PHASES, rendered, expected):
    print(f'{a:6.0f}   {f:.4e}       {g:.4e}   {to_mag(f) - to_mag(g):+.4f}')

mpl.rcParams.update({
    "mathtext.fontset": "stix", "font.family": "STIXGeneral", "font.size": 10,
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
})
fig, (ax, axr) = plt.subplots(2, 1, figsize=(3.5, 3.6), sharex=True,
                              gridspec_kw=dict(height_ratios=(2.5, 1), hspace=0.05))
fine = np.linspace(0, 175, 300)
ax.plot(fine, to_mag(analytic_flux(fine)), '-', color='0.5', lw=1.0, label='Analytic')
ax.plot(PHASES, to_mag(rendered), 'k.', ms=4, label='satlight')
ax.invert_yaxis()
ax.set_ylabel(r'$m_\mathrm{V}$ (mag)')
ax.legend(loc='lower left')
axr.axhline(0, color='0.5', lw=0.8)
axr.plot(PHASES, to_mag(rendered) - to_mag(expected), 'k.', ms=4)
axr.set_xlabel('Phase angle (deg)')
axr.set_ylabel(r'$\Delta m$ (mag)')
fig.savefig('lambertian_sphere.png', dpi=300)
print('wrote lambertian_sphere.png')
