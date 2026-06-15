"""
test_earthshine_gto.py
Earthshine flux variation over a GTO orbit.
"""

import numpy as np
import sys
import datetime
sys.path.insert(0, '/Users/l.beesley@bham.ac.uk/Documents/Lightcurves/satlight')

from satlight.geometry import Geometry
from importlib.resources import files

# ── GTO orbital elements ─────────────────────────────────────────────────────
R_EARTH  = 6371.0
r_perigee = R_EARTH + 250      # 250km perigee
r_apogee  = R_EARTH + 35786    # GEO apogee
a = (r_perigee + r_apogee) / 2
e = (r_apogee - r_perigee) / (r_apogee + r_perigee)

print(f"GTO elements:")
print(f"  Semi-major axis: {a:.1f} km")
print(f"  Eccentricity:    {e:.4f}")
print(f"  Perigee alt:     {r_perigee - R_EARTH:.0f} km")
print(f"  Apogee alt:      {r_apogee - R_EARTH:.0f} km")

# Orbital period in hours
GM = 398600.4418
T_hours = 2 * np.pi * np.sqrt(a**3 / GM) / 3600
print(f"  Period:          {T_hours:.2f} hours\n")

# ── Geometry setup ───────────────────────────────────────────────────────────
geometry = Geometry()

# Observer at Tenerife (twilight observation site)
geometry.create_observer(28.29156, -16.62, 2070)

# Create GTO satellite
epoch = datetime.datetime(2025, 5, 4, 0, 0, 0)
geometry.create_satellite_from_elements(
    a_km     = a,
    e        = e,
    i_deg    = 28.0,
    raan_deg = 0.0,
    argp_deg = 178.0,
    nu_deg   = 0.0,
    epoch    = epoch
)

# Time array — one full orbit sampled every 5 minutes
n_steps  = int(T_hours * 60 / 5)
times    = [(2025, 5, 4, 0, i*5, 0) for i in range(n_steps)]

print(f"Propagating {n_steps} time steps over {T_hours:.1f} hour orbit...")
geometry.set_time((2025, 5, 4, 0, range(0, int(T_hours * 3600), 300), 0))