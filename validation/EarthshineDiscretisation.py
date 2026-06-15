"""
Earthshine discretisation comparison for SatLight
Compares: Lat/Lon grid, Equal-area rings, Icosphere
Measures: received irradiance at 1000 km altitude, timing, convergence vs
          high-resolution numerical reference.

Geometry:  Sun in +X direction.
           Satellite at (R_earth + ALT_SAT, 0, 0) — directly above subsolar
           point — to maximise earthshine and give a clean comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

# ── Constants ─────────────────────────────────────────────────────────────
R_EARTH  = 6371.0
ALT_SAT  = 1000.0
R_SAT    = R_EARTH + ALT_SAT
SOLAR_K  = 1361.0
ALBEDO   = 0.30

SUN_UNIT = np.array([1.0, 0.0, 0.0])
SAT_POS  = np.array([R_SAT, 0.0, 0.0])


def patch_irradiance(centers_km, areas_km2):
    norms   = centers_km / np.linalg.norm(centers_km, axis=1, keepdims=True)
    cos_sun = norms @ SUN_UNIT
    to_sat  = SAT_POS - centers_km
    dist_km = np.linalg.norm(to_sat, axis=1)
    cos_sat = np.sum(norms * (to_sat / dist_km[:, None]), axis=1)
    mask    = (cos_sun > 0) & (cos_sat > 0)
    dE = (ALBEDO / np.pi) * SOLAR_K \
         * cos_sun[mask] * cos_sat[mask] \
         * (areas_km2[mask] * 1e6) / (dist_km[mask] * 1e3)**2
    return float(np.sum(dE))


def reference_irradiance(n_lat=2000):
    n_lon = 2 * n_lat
    d_lat = np.pi / n_lat
    d_lon = 2 * np.pi / n_lon
    lats  = -np.pi/2 + (np.arange(n_lat) + 0.5) * d_lat
    lons  =             (np.arange(n_lon) + 0.5) * d_lon
    LAT, LON = np.meshgrid(lats, lons, indexing='ij')
    cl = np.cos(LAT)
    x  = R_EARTH * cl * np.cos(LON)
    y  = R_EARTH * cl * np.sin(LON)
    z  = R_EARTH * np.sin(LAT)
    c  = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    a  = (R_EARTH**2 * cl * d_lat * d_lon).ravel()
    return patch_irradiance(c, a)


def latlon_grid(n_lat):
    n_lon = 2 * n_lat
    d_lat = np.pi / n_lat
    d_lon = 2 * np.pi / n_lon
    lats  = -np.pi/2 + (np.arange(n_lat) + 0.5) * d_lat
    lons  =             (np.arange(n_lon) + 0.5) * d_lon
    LAT, LON = np.meshgrid(lats, lons, indexing='ij')
    cl = np.cos(LAT.ravel())
    lr = LAT.ravel(); lo = LON.ravel()
    c  = np.stack([R_EARTH*cl*np.cos(lo), R_EARTH*cl*np.sin(lo), R_EARTH*np.sin(lr)], axis=1)
    a  = R_EARTH**2 * cl * d_lat * d_lon
    return c, a


def equal_area_rings(n_rings):
    d_lat = np.pi / n_rings
    lats  = -np.pi/2 + (np.arange(n_rings) + 0.5) * d_lat
    n_eq  = max(4, round(2 * np.pi * R_EARTH / (R_EARTH * d_lat)))
    cs, As = [], []
    for lat in lats:
        n_lon = max(1, round(n_eq * abs(np.cos(lat))))
        d_lon = 2 * np.pi / n_lon
        lons  = (np.arange(n_lon) + 0.5) * d_lon
        cl    = np.cos(lat)
        cs.append(np.stack([R_EARTH*cl*np.cos(lons),
                             R_EARTH*cl*np.sin(lons),
                             np.full(n_lon, R_EARTH*np.sin(lat))], axis=1))
        As.append(np.full(n_lon, R_EARTH**2 * cl * d_lat * d_lon))
    return np.vstack(cs), np.concatenate(As)


def driscoll_healy(n_rings):
    """
    Exactly equal-area grid using equal steps in sin(φ) and λ.
    Area element: dA = R² · d(sin φ) · dλ  — no cos(φ) factor, no rounding.
    Every patch has identical area = 4πR² / N_total.
    """
    sin_edges    = np.linspace(-1, 1, n_rings + 1)
    sin_mids     = (sin_edges[:-1] + sin_edges[1:]) / 2
    phi_mids     = np.arcsin(sin_mids)
    d_sinphi_arr = np.diff(sin_edges)          # exactly 2/n_rings each
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
    return np.vstack(cs), np.concatenate(As)


def _ico_base():
    phi = (1 + np.sqrt(5)) / 2
    v = np.array([
        [-1,phi,0],[1,phi,0],[-1,-phi,0],[1,-phi,0],
        [0,-1,phi],[0,1,phi],[0,-1,-phi],[0,1,-phi],
        [phi,0,-1],[phi,0,1],[-phi,0,-1],[-phi,0,1],
    ], dtype=float)
    v /= np.linalg.norm(v[0])
    f = np.array([
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ], dtype=int)
    return v, f


def _subdivide(verts, faces):
    em, nf, vl = {}, [], list(verts)
    def mid(i, j):
        k = (min(i,j), max(i,j))
        if k not in em:
            m = (vl[i]+vl[j])/2; m /= np.linalg.norm(m)
            em[k] = len(vl); vl.append(m)
        return em[k]
    for a, b, c in faces:
        ab,bc,ca = mid(a,b),mid(b,c),mid(c,a)
        nf += [[a,ab,ca],[b,bc,ab],[c,ca,bc],[ab,bc,ca]]
    return np.array(vl), np.array(nf)


def icosphere(subdivisions):
    v, f = _ico_base()
    for _ in range(subdivisions):
        v, f = _subdivide(v, f)
    cents = (v[f[:,0]]+v[f[:,1]]+v[f[:,2]])/3
    cents /= np.linalg.norm(cents, axis=1, keepdims=True)
    return cents * R_EARTH, np.full(len(f), 4*np.pi*R_EARTH**2/len(f))


CONFIGS = {
    'latlon':     ('Lat/lon grid',          '#378ADD', [5,10,20,40,80,160,320]),
    'equal_area': ('Equal-area rings',      '#1D9E75', [5,10,20,40,80,160,320]),
    'driscoll':   ('Driscoll-Healy (exact)','#9B59B6', [5,10,20,40,80,160,320]),
    'icosphere':  ('Icosphere',             '#D85A30', [1,2,3,4,5,6]),
}
BUILDERS = {
    'latlon':     latlon_grid,
    'equal_area': equal_area_rings,
    'driscoll':   driscoll_healy,
    'icosphere':  icosphere,
}


def benchmark(E_ref):
    results = {k: [] for k in CONFIGS}
    print(f"\n{'Method':<22} {'N patches':>10} {'E (W/m²)':>11} {'Error %':>9} {'Time (ms)':>10}")
    print("─" * 68)
    for key, (label, color, params) in CONFIGS.items():
        for p in params:
            t0 = time.perf_counter()
            c, a = BUILDERS[key](p)
            E  = patch_irradiance(c, a)
            dt = (time.perf_counter() - t0) * 1000
            err = abs(E - E_ref) / E_ref * 100
            results[key].append({'N': len(c), 'E': E, 'err': err, 'dt': dt})
            print(f"{label:<22} {len(c):>10,} {E:>11.4f} {err:>9.3f} {dt:>10.2f}")
        print()
    return results


def plot_benchmark(results, E_ref):
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor('#f7f7f5')
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32)
    axes = [fig.add_subplot(gs[r,c]) for r in range(2) for c in range(2)]
    ax1, ax2, ax3, ax4 = axes

    for key, (label, color, _) in CONFIGS.items():
        r    = results[key]
        Ns   = [d['N']   for d in r]
        Es   = [d['E']   for d in r]
        errs = [d['err'] for d in r]
        dts  = [d['dt']  for d in r]
        kw   = dict(color=color, label=label, lw=1.8, ms=5)
        ax1.semilogx(Ns, Es,  'o-', **kw)
        ax2.loglog(Ns, errs,  'o-', **kw)
        ax3.loglog(Ns, dts,   'o-', **kw)
        ax4.loglog(dts, errs, 'o-', **kw)

    ax1.axhline(E_ref, color='#333', ls='--', lw=1.2, label='Reference')
    ax1.fill_between([1,1e7], E_ref*0.99, E_ref*1.01, alpha=0.07, color='#333')
    ax1.annotate(f'±1%  band', xy=(0.02,0.55), xycoords='axes fraction', fontsize=7, color='#666')

    meta = [
        ('Convergence of irradiance',       'Number of patches',        'Irradiance (W/m²)'),
        ('Accuracy vs patch count',          'Number of patches',        'Error vs reference (%)'),
        ('Compute time vs patch count',      'Number of patches',        'Computation time (ms)'),
        ('Efficiency frontier\n(lower-left = better)', 'Computation time (ms)', 'Error vs reference (%)'),
    ]
    for ax, (t, xl, yl) in zip(axes, meta):
        ax.set_facecolor('#ffffff')
        ax.spines[['top','right']].set_visible(False)
        ax.legend(fontsize=8, framealpha=0.8)
        ax.grid(True, alpha=0.2, lw=0.5)
        ax.tick_params(labelsize=8)
        ax.set_title(t, fontsize=10, fontweight='bold', pad=6)
        ax.set_xlabel(xl, fontsize=9)
        ax.set_ylabel(yl, fontsize=9)

    fig.suptitle(
        f'Earthshine discretisation — {ALT_SAT:.0f} km altitude, nadir over subsolar point\n'
        f'Albedo={ALBEDO}, Solar constant={SOLAR_K} W/m²,  Reference={E_ref:.3f} W/m²',
        fontsize=10, fontweight='bold', y=0.995)
    plt.savefig('/mnt/user-data/outputs/earthshine_comparison.png',
                dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print("Benchmark plot saved.")
    plt.close()


def plot_patches():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': '3d'})
    fig.patch.set_facecolor('#f7f7f5')

    vis_configs = [
        ('Lat/lon grid (n=20)',      latlon_grid(20),      '#378ADD'),
        ('Equal-area rings (n=20)',  equal_area_rings(20), '#1D9E75'),
        ('Icosphere (level 3)',      icosphere(3),         '#D85A30'),
    ]
    u = np.linspace(0, 2*np.pi, 60); v = np.linspace(0, np.pi, 40)
    xs = R_EARTH*np.outer(np.cos(u), np.sin(v))
    ys = R_EARTH*np.outer(np.sin(u), np.sin(v))
    zs = R_EARTH*np.outer(np.ones_like(u), np.cos(v))

    for ax, (title, (centers, areas), col) in zip(axes, vis_configs):
        ax.plot_surface(xs, ys, zs, alpha=0.07, color='#999999')
        norms   = centers / np.linalg.norm(centers, axis=1, keepdims=True)
        cos_sun = norms @ SUN_UNIT
        to_sat  = SAT_POS - centers
        cos_sat = np.sum(norms * to_sat / np.linalg.norm(to_sat, axis=1, keepdims=True), axis=1)
        mask    = (cos_sun > 0) & (cos_sat > 0)
        ax.scatter(centers[mask,0],  centers[mask,1],  centers[mask,2],
                   s=4, color=col, alpha=0.85, depthshade=False, label='Contributing')
        ax.scatter(centers[~mask,0], centers[~mask,1], centers[~mask,2],
                   s=1, color='#cccccc', alpha=0.25, depthshade=False)
        ax.quiver(0,0,0, R_EARTH*1.5,0,0, color='#EF9F27', lw=1.5, arrow_length_ratio=0.12)
        ax.text(R_EARTH*1.75, 0, 200, 'Sun', fontsize=7, color='#BA7517')
        ax.scatter(*SAT_POS, s=50, color='#7F77DD', marker='^', zorder=5)
        ax.text(SAT_POS[0]*1.03, 0, 500, 'Sat', fontsize=7, color='#534AB7')
        ax.set_title(f'{title}\nN={len(centers):,}  contributing={mask.sum()}', fontsize=8.5, pad=4)
        ax.set_facecolor('#f7f7f5')
        ax.set_xlabel('X (km)', fontsize=6); ax.set_ylabel('Y (km)', fontsize=6); ax.set_zlabel('Z (km)', fontsize=6)
        ax.tick_params(labelsize=5)
        ax.set_box_aspect([1,1,1])
        ax.view_init(elev=20, azim=25)

    fig.suptitle('Patch distributions — coloured patches contribute to earthshine at satellite',
                 fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/earthshine_patches.png',
                dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print("Patch plot saved.")
    plt.close()


def summary(results, E_ref):
    print("\n── Patches needed to reach <1% error ──────────────")
    print(f"{'Method':<22} {'N patches':>10} {'Error %':>9} {'Time (ms)':>10}")
    print("─" * 55)
    for key, (label, _, _) in CONFIGS.items():
        for d in results[key]:
            if d['err'] < 1.0:
                print(f"{label:<22} {d['N']:>10,} {d['err']:>9.3f} {d['dt']:>10.2f}")
                break
        else:
            last = results[key][-1]
            print(f"{label:<22} {last['N']:>10,}  {last['err']:>8.3f}  (never <1% in range)")


if __name__ == '__main__':
    print("=" * 68)
    print("EARTHSHINE DISCRETISATION BENCHMARK — SatLight")
    print(f"Altitude: {ALT_SAT} km  |  Albedo: {ALBEDO}  |  Solar: {SOLAR_K} W/m²")
    print("=" * 68)
    print("\nComputing reference (2000×4000 grid)...")
    t0 = time.perf_counter()
    E_ref = reference_irradiance(2000)
    print(f"Reference: {E_ref:.4f} W/m²   ({time.perf_counter()-t0:.1f}s)")

    results = benchmark(E_ref)
    summary(results, E_ref)
    plot_benchmark(results, E_ref)
    plot_patches()
    print("\nDone.")