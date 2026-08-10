"""
test_earthshine_leo.py
Earthshine flux variation over one LEO SSO orbit.
"""

import numpy as np
import sys
import datetime
sys.path.insert(0, '/Users/l.beesley@bham.ac.uk/Documents/Lightcurves/satlight')

from satlight.geometry import Geometry
from importlib.resources import files
import matplotlib.pyplot as plt

# ── Constants ─────────────────────────────────────────────────────────────────
GM      = 398600.4418
R_EARTH = 6371.0

# ── SSO orbit elements ────────────────────────────────────────────────────────
a     = R_EARTH + 500   # 500km circular
e     = 0.001
i_deg = 97.4            # SSO inclination
T_sec = 2 * np.pi * np.sqrt(a**3 / GM)
T_hrs = T_sec / 3600

print(f"SSO orbit:")
print(f"  Altitude:  500 km")
print(f"  Period:    {T_hrs:.2f} hours ({T_hrs*60:.1f} minutes)")

# ── Geometry ──────────────────────────────────────────────────────────────────
geometry = Geometry()
geometry.create_observer(28.29156, -16.62, 2070)

epoch = datetime.datetime(2025, 5, 4, 0, 0, 0)
geometry.create_satellite_from_elements(
    a_km     = a,
    e        = e,
    i_deg    = i_deg,
    raan_deg = 90.0,
    argp_deg = 0.0,
    nu_deg   = 0.0,
    epoch    = epoch
)

# One full orbit, 30s cadence
dt      = 30  # seconds
n_steps = int(T_sec / dt)
geometry.set_time((2025, 5, 4, 0, 0, range(0, n_steps * dt, dt)))

print(f"  Steps:     {n_steps} at {dt}s cadence\n")
# ── CERES albedo loader ───────────────────────────────────────────────────────
def _load_ceres_albedo():
    path = files("satlight.data").joinpath("ceres_albedo_monthly.npy")
    return np.load(path)

def _lookup_albedo(grid, lat_deg, lon_deg):
    lat_idx = np.clip(int(np.floor(90 - lat_deg)), 0, 179)
    lon_idx = np.clip(int(np.floor(lon_deg + 180)), 0, 359)
    return grid[lat_idx, lon_idx]

# ── Single time step earthshine ───────────────────────────────────────────────
def compute_earthshine_single(geom, albedo_grid, solar_constant=1460, n_rings=40):
    R_EARTH = 6371.0
    R_mat = np.stack([geom.along_track, geom.cross_track, -geom.nadir], axis=0)
    sat_pos_gcrs = geom.positions['satellite']

    sin_edges    = np.linspace(-1, 1, n_rings + 1)
    sin_mids     = (sin_edges[:-1] + sin_edges[1:]) / 2
    phi_mids     = np.arcsin(sin_mids)
    d_sinphi_arr = np.diff(sin_edges)
    n_lon        = 2 * n_rings
    d_lon        = 2 * np.pi / n_lon
    lons         = (np.arange(n_lon) + 0.5) * d_lon

    cs, As = [], []
    for phi, dsp in zip(phi_mids, d_sinphi_arr):
        cl = np.cos(phi)
        cs.append(np.stack([
            R_EARTH * cl * np.cos(lons),
            R_EARTH * cl * np.sin(lons),
            np.full(n_lon, R_EARTH * np.sin(phi))
        ], axis=1))
        As.append(np.full(n_lon, R_EARTH**2 * dsp * d_lon))

    centers_gcrs = np.vstack(cs)
    areas        = np.concatenate(As)

    x, y, z  = centers_gcrs[:, 0], centers_gcrs[:, 1], centers_gcrs[:, 2]
    lats_deg = np.degrees(np.arcsin(z / R_EARTH))
    lons_deg = np.degrees(np.arctan2(y, x))
    albedos  = np.array([_lookup_albedo(albedo_grid, la, lo)
                         for la, lo in zip(lats_deg, lons_deg)])

    rel          = centers_gcrs - sat_pos_gcrs
    centers_lvlh = (R_mat @ rel.T).T
    earth_centre_lvlh = R_mat @ (np.zeros(3) - sat_pos_gcrs)
    norms_lvlh = centers_lvlh - earth_centre_lvlh
    norms_lvlh /= np.linalg.norm(norms_lvlh, axis=1, keepdims=True)

    to_sat   = -centers_lvlh
    dist_km  = np.linalg.norm(to_sat, axis=1)
    cos_sat  = np.sum(norms_lvlh * (to_sat / dist_km[:, None]), axis=1)
    sun_gcrs_unit = geom.positions['sun'].ravel() / np.linalg.norm(geom.positions['sun'])
    cos_sun_gcrs  = (centers_gcrs @ sun_gcrs_unit) / R_EARTH
    mask          = (cos_sun_gcrs > 0) & (cos_sat > 0)

    if mask.sum() == 0:
        return 0.0

    dE = (albedos[mask] / np.pi) * solar_constant \
    * cos_sun_gcrs[mask] * cos_sat[mask] \
    * (areas[mask] * 1e6) / (dist_km[mask] * 1e3)**2

    # Diagnostic — remove after checking
    if len(dE) > 0:
        print(f"  dist_km: {dist_km[mask].min():.0f}-{dist_km[mask].max():.0f} km  "
              f"areas: {(areas[mask]*1e6).min():.2e}-{(areas[mask]*1e6).max():.2e} m²  "
              f"dE max: {dE.max():.6f} W/m²")

    return float(np.sum(dE))
    return float(np.sum(dE))

# ── Loop over orbit ───────────────────────────────────────────────────────────
albedo_grid  = _load_ceres_albedo()
sat_pos      = geometry.positions['satellite']
altitude     = np.linalg.norm(sat_pos, axis=0) - R_EARTH
t_min        = np.arange(n_steps) * dt / 60  # minutes

def compute_eclipse_geometric(sat_pos_gcrs, sun_pos_gcrs):
    """
    Returns 1.0 = sunlit, 0.0 = in eclipse.
    Uses cylindrical shadow approximation.
    """
    # Unit vector from Earth to Sun
    sun_unit = sun_pos_gcrs / np.linalg.norm(sun_pos_gcrs, axis=0)
    
    # Project satellite position onto Earth-Sun axis
    sat_proj = np.sum(sat_pos_gcrs * sun_unit, axis=0)
    
    # Perpendicular distance from Earth-Sun line
    sat_perp = np.sqrt(
        np.sum(sat_pos_gcrs**2, axis=0) - sat_proj**2
    )
    
    # In shadow if on night side (sat_proj < 0) and within Earth's radius
    in_shadow = (sat_proj < 0) & (sat_perp < R_EARTH)
    return np.where(in_shadow, 0.0, 1.0)

sun_pos      = geometry.positions['sun']   # (3, N)
eclipse_frac = compute_eclipse_geometric(sat_pos, sun_pos)

print("Computing earthshine over orbit...")
es_flux = []
for i in range(n_steps):
    g = Geometry()
    g.satellite = geometry.satellite
    g.observer  = geometry.observer
    g.set_time((2025, 5, 4, 0, 0, i * dt))
    es_flux.append(compute_earthshine_single(g, albedo_grid))
    if i % 20 == 0:
        print(f"  {i}/{n_steps}  es={es_flux[-1]:.4f} W/m²  "
              f"eclipse={1-eclipse_frac[i]:.0f}")

es_flux      = np.array(es_flux)
eclipse_frac = eclipse_frac[:len(es_flux)]
t_min        = t_min[:len(es_flux)]

# Add this diagnostic before the plot
print("\nDiagnostic — sun direction at first 3 steps:")
for i in [0, 30, 60]:
    g = Geometry()
    g.satellite = geometry.satellite
    g.observer  = geometry.observer
    g.set_time((2025, 5, 4, 0, 0, i * dt))
    print(f"  t={i*dt}s  sun_lvlh={g.incident_vector_lvlh}  "
          f"eclipse={g.eclipse_type():.3f}  "
          f"sat_pos={g.positions['satellite']}")
    
# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

# Panel 1: Eclipse fraction + earthshine
ax1 = axes[0]
ax1.fill_between(t_min, eclipse_frac, alpha=0.2, color='gold', label='Sunlit fraction')
ax1.fill_between(t_min, 1 - eclipse_frac, alpha=0.2, color='navy', label='Eclipse fraction')
ax1.set_ylabel('Illumination fraction')
ax1.set_ylim(0, 1)
ax1.legend(loc='upper right')
ax1.set_title('LEO SSO 500km — Earthshine Irradiance over One Orbit')

# Panel 2: Earthshine flux
ax2 = axes[1]
ax2.plot(t_min, es_flux, color='steelblue', linewidth=1.5, label='CERES EBAF')
ax2.fill_between(t_min, es_flux, where=(eclipse_frac < 0.5),
                 alpha=0.3, color='navy', label='In eclipse')
ax2.set_ylabel('Earthshine Irradiance (W/m²)')
ax2.set_xlabel('Time from epoch (minutes)')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig('leo_earthshine.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nResults:")
print(f"  Max earthshine: {es_flux.max():.4f} W/m²  at t={t_min[np.argmax(es_flux)]:.1f} min")
print(f"  Min earthshine: {es_flux.min():.4f} W/m²  at t={t_min[np.argmin(es_flux)]:.1f} min")
in_eclipse = eclipse_frac < 0.5
if in_eclipse.any():
    print(f"  Mean (eclipse): {es_flux[in_eclipse].mean():.4f} W/m²")
else:
    print(f"  No eclipse periods found in this orbit")
print(f"  Eclipse fraction of orbit: {(1-eclipse_frac).mean()*100:.1f}%")