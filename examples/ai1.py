"""Light curve of the SpaceX AI1 orbital data centre over one pass, checking first
when it is sunlit and observable.

The same pass as the tutorial: the AI1 model on the TLE below, seen from La
Palma. A satellite is only seen when three things hold at once:

    above the horizon       elevation from skyfield, or Geometry.is_visible()
    in sunlight             Geometry.eclipse_type(): 1 sunlit, 0 umbra,
                            the visible fraction of the solar disc in penumbra
    both together           Geometry.observable_fraction()
                            = eclipse_type() * is_visible()

These all take a Geometry holding many epochs, so the whole pass is checked in
one call and only the observable epochs are rendered. The Renderer applies
observable_fraction() itself and returns zero for an eclipsed frame, so the
check is about not spending time on frames you already know are dark.

Run from the repository root:
    python examples/ai1.py
"""
import datetime
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from satlight.geometry import Geometry
from satlight.raytracer import Renderer

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, '..', 'models', 'reference', 'spacex_ai1.obj')

LINE1 = "1 25544U 98067A   24015.50000000  .00016717  00000-0  10270-3 0  9990"
LINE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.49815174 10000"
SITE = dict(observer_lat=28.7622, observer_lon=-17.8897, observer_alt_m=2396)   # La Palma

BRDF = {
    'solar':    ('lambertian', {'albedo': 0.08}),
    'radiator': ('lambertian', {'albedo': 0.60}),
    'boom':     ('lambertian', {'albedo': 0.20}),
    'bus':      ('lambertian', {'albedo': 0.30}),
    'default':  ('lambertian', {'albedo': 0.25}),
}


def geometry(*time_utc):
    g = Geometry()
    g.create_observer(**SITE)
    g.create_satellite(LINE1, LINE2)
    g.set_time(*time_utc)
    return g


# ---- 1. find the pass: elevation above 10 deg, within 3 h of t0
search = geometry(2024, 1, 15, 18, 0, 0)
passes = search.next_visibility((2024, 1, 15, 18, 0, 0), N=3, i=10.0)
y, mo, d, h, mi, secs = passes[0]
print(f'pass above 10 deg starts {y}-{mo:02d}-{d:02d} {h:02d}:{mi:02d}:{secs.start:02d} UTC, '
      f'lasts {len(secs)} s')

# ---- 2. check illumination over the whole pass in one go, every 5 s
step = 5
g = geometry(y, mo, d, h, mi, np.arange(secs.start, secs.stop, step))
elev = (g.satellite - g.observer).at(g.time).altaz()[0].degrees
visible = np.asarray(g.is_visible())            # Earth not in the way
sunlit = np.asarray(g.eclipse_type())           # fraction of the Sun's disc seen
observable = np.asarray(g.observable_fraction())
t = np.array([tt.utc_datetime() for tt in g.time])

for i in range(0, len(t), 6):
    state = ('observable' if observable[i] == 1 else
             'penumbra' if observable[i] > 0 else
             'in shadow' if visible[i] else 'below horizon')
    print(f'{t[i]:%H:%M:%S}  elev {elev[i]:5.1f} deg  sunlit {sunlit[i]:.2f}  {state}')
lit = observable > 0
print(f'{lit.sum()} of {len(t)} epochs observable; '
      f'entering shadow at {t[lit][-1]:%H:%M:%S}' if lit.any() and not lit[-1] else '')

# ---- 3. render the observable epochs only
r = Renderer(MODEL, geometry=geometry(t[lit][0]), resolution=(200, 200), brdf=BRDF)
r.initialise_scene()
mag = np.full(len(t), np.nan)
for i in np.flatnonzero(lit):
    gi = geometry(t[i])
    r.update(gi)
    flux = float(np.sum(r.render()))            # W/m^2, already x observable_fraction
    mag[i] = -2.5 * np.log10(flux) - 18.915     # broadband V, see guide/photometry
    print(f'{t[i]:%H:%M:%S}  {gi.prop_distance:6.0f} km  '
          f'phase {np.degrees(gi.phase_angle):5.1f} deg  m_V {mag[i]:5.2f}')

# ---- 4. plot
mpl.rcParams.update({
    "mathtext.fontset": "stix", "font.family": "STIXGeneral", "font.size": 10,
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
})
minutes = np.array([(ti - t[0]).total_seconds() / 60 for ti in t])
fig, ax = plt.subplots(figsize=(3.5, 2.6))
ax.plot(minutes[lit], mag[lit], 'k.-', lw=0.8, ms=3)
if (~lit).any():
    ax.axvspan(minutes[~lit].min(), minutes[-1], color='0.9', lw=0)
    ax.text(0.97, 0.9, 'In shadow', transform=ax.transAxes, ha='right', fontsize=9)
ax.set_xlim(0, minutes[-1])
ax.invert_yaxis()
ax.set_xlabel(f'Time since {t[0]:%H:%M:%S} UTC (min)')
ax.set_ylabel(r'$m_\mathrm{V}$ (mag)')
fig.savefig('ai1_pass.png', dpi=300)
print('wrote ai1_pass.png')
