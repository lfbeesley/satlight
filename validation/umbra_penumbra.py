import numpy as np
import matplotlib.pyplot as plt

# --- constants ---
R_E = 6371.0              # km
R_S = 695700.0            # km
AU   = 149.6e6            # km
mu_E = 398600.4418        # km^3/s^2

earth_pos = np.array([0.0,0.0,0.0])
sun_pos   = np.array([AU,0.0,0.0])   # Sun approx +X

# --- helper geometry functions ---
def sat_positions_circle(r, angles, inc):
    # returns (N,3) array of satellite positions for circular orbit rotated about X-axis by inc
    x = r * np.cos(angles)
    y = r * np.sin(angles) * np.cos(inc)
    z = r * np.sin(angles) * np.sin(inc)
    return np.stack([x,y,z], axis=-1)

def angular_radius(R, D):
    return np.arcsin(np.clip(R / D, -1.0, 1.0))

def compute_geometry_for_orbit(sat_pos):
    # sat_pos: (N,3)
    dE = earth_pos - sat_pos               # vectors from sat to Earth centre
    dS = sun_pos   - sat_pos               # vectors from sat to Sun centre
    D_E = np.linalg.norm(dE, axis=1)
    D_S = np.linalg.norm(dS, axis=1)
    theta_E = angular_radius(R_E, D_E)
    theta_S = angular_radius(R_S, D_S)
    cos_phi = np.einsum('ij,ij->i', dE, dS) / (D_E * D_S)
    cos_phi = np.clip(cos_phi, -1.0, 1.0)
    phi = np.arccos(cos_phi)
    return phi, theta_E, theta_S

# --- interpolation helper for boundary crossing ---
def interp_crossing(g, angles, idx_a, idx_b, tol=1e-12):
    """
    Linear interpolate root of g between indices idx_a and idx_b (consecutive indices,
    mod wrap). angles are sorted [0,2pi). Returns root angle in [0,2pi).
    """
    Na = angles[idx_a]
    Nb = angles[idx_b]
    ga = g[idx_a]; gb = g[idx_b]
    if abs(ga) < tol:
        return Na % (2*np.pi)
    if abs(gb) < tol:
        return Nb % (2*np.pi)
    # ensure Nb > Na in linear sense (handle wrap)
    if Nb <= Na:
        Nb = Nb + 2*np.pi
    t = ga / (ga - gb)   # fraction from a->b where g==0
    angle = Na + t * (Nb - Na)
    return angle % (2*np.pi)

# --- function to measure angular length of boolean segments (with refined boundaries) ---
def angular_length_of_true_segments(boolean_arr, angles, g1=None, g2=None):
    """
    boolean_arr: array of True/False on uniform grid angles
    g1, g2: arrays of same length used to find precise crossing roots when a boundary occurs.
            For penumbra segments we choose the boundary root from g1 or g2 depending on which changes sign.
            If g1/g2 are None the function will not attempt refined interpolation (fallback to grid count).
    """
    N = len(boolean_arr)
    b = boolean_arr
    if not np.any(b):
        return 0.0
    if np.all(b):
        return 2*np.pi

    # indices where a True-run starts and ends (start = current True & previous False)
    starts = np.where((b==1) & (np.roll(b,1)==0))[0]
    ends   = np.where((b==1) & (np.roll(b,-1)==0))[0]
    total = 0.0

    for s, e in zip(starts, ends):
        # s .. e are indices of True-run inclusive
        prev = (s - 1) % N
        nxt  = (e + 1) % N

        # start crossing: between prev and s
        start_angle = None
        if g1 is not None and g1[prev]*g1[s] < 0:
            start_angle = interp_crossing(g1, angles, prev, s)
        elif g2 is not None and g2[prev]*g2[s] < 0:
            start_angle = interp_crossing(g2, angles, prev, s)
        else:
            # fall back to grid point boundary
            start_angle = angles[s]

        # end crossing: between e and nxt
        end_angle = None
        if g1 is not None and g1[e]*g1[nxt] < 0:
            end_angle = interp_crossing(g1, angles, e, nxt)
        elif g2 is not None and g2[e]*g2[nxt] < 0:
            end_angle = interp_crossing(g2, angles, e, nxt)
        else:
            end_angle = angles[e]

        # ensure positive measure (handle wrap)
        if end_angle <= start_angle:
            end_angle += 2*np.pi
        total += (end_angle - start_angle)
    return total

# --- main computation: penumbra & umbra times vs altitude ---
def penumbra_umbra_times_vs_altitudes(alts_km, inc_rad, n_angles=4096):
    angles = np.linspace(0.0, 2*np.pi, n_angles, endpoint=False)   # rad
    pen_times = np.zeros_like(alts_km, dtype=float)
    umb_times  = np.zeros_like(alts_km, dtype=float)

    for i, alt in enumerate(alts_km):
        r = R_E + alt
        sat_pos = sat_positions_circle(r, angles, inc_rad)   # (N,3)
        phi, theta_E, theta_S = compute_geometry_for_orbit(sat_pos)
        # g1 = phi - (theta_E + theta_S)    # overlap boundary (enter/exit any overlap)
        g1 = phi - (theta_E + theta_S)
        # g2 = phi - abs(theta_E - theta_S) # umbra boundary (enter/exit total overlap)
        g2 = phi - np.abs(theta_E - theta_S)

        # boolean arrays
        penumbra_bool = (g1 < 0) & (g2 > 0)
        umbra_bool    = (theta_E > theta_S) & (phi <= (theta_E - theta_S))

        # compute angular lengths (with refined crossings)
        ang_pen = angular_length_of_true_segments(penumbra_bool, angles, g1=g1, g2=g2)
        ang_umb = angular_length_of_true_segments(umbra_bool, angles, g1=None, g2=g2)  # use g2 for umbra boundaries

        # orbital period (circular)
        a = r
        T = 2*np.pi * np.sqrt(a**3 / mu_E)   # seconds

        pen_times[i] = (ang_pen / (2*np.pi)) * T
        umb_times[i] = (ang_umb  / (2*np.pi)) * T

    return pen_times, umb_times

# --- example usage and plotting ---
if __name__ == "__main__":
    inc = np.radians(0.0)    # polar example (change as required)
    alts = np.linspace(200.0, 36000.0, 200)   # km
    pen_times, umb_times = penumbra_umbra_times_vs_altitudes(alts, inc, n_angles=4096)

    plt.figure(figsize=(8,5))
    plt.plot(alts, pen_times/60.0, label='Penumbra time per orbit (min)')
    plt.plot(alts, umb_times/60.0,  label='Umbra time per orbit (min)')
    plt.xlabel('Altitude (km)')
    plt.ylabel('Time per orbit (minutes)')
    plt.title(f'Penumbra / Umbra duration vs altitude (inc={np.degrees(inc):.0f}°)')
    plt.grid(True)
    plt.legend()
    plt.show()
