"""
Test / demo: pristine vs aged aluminium bus - GEO lightcurve comparison.

Renders the same single-night GEO pass twice with satlight, once with a
pristine aluminium bus BRDF and once with an "aged" version, then reports
whether ageing makes the material dimmer and more diffuse, as predicted by
the modelling in the PATINA proposal.

MODELLING NOTE - read before trusting the numbers
---------------------------------------------------
The proposal's pipeline is:

    exposure chamber -> spectral BRDF fit (Kubelka-Munk) ->
    n(lambda, Phi), k(lambda, Phi) -> satlight lightcurve/colour prediction

satlight's current public interface (see the brdf={...} dict in your
single-night script) only exposes SCALAR per-surface parameters -
albedo, roughness, metallic for cook_torrance; albedo, roughness for
oren_nayar. There is no per-wavelength Fresnel term visible from the
outside, so this cannot confirm the repo currently supports spectral n,k
inputs at all.

kubelka_munk_to_effective_params() below is therefore a STAND-IN: a toy
fluence -> effective-albedo / effective-roughness map, calibrated only to
get the sign and rough magnitude of the effect right (ageing lowers
broadband albedo and raises roughness), not a real K-M fit. Swap this
function's internals for a genuine Kubelka-Munk fit to your BRDF-rig data
once you have it, and if/when satlight grows a spectral BRDF class, replace
the scalar albedo here with per-band values and integrate against your
instrument bandpass instead of using a single broadband number.

This is a rendering-heavy integration/demo script (needs bpy + satlight +
your .obj model), not a fast unit test - run it directly in a notebook or
console. It reports what happened rather than asserting on it, since
brightness direction depends on your model's geometry and phase-angle
coverage as well as the material parameters - if the result doesn't match
the expected direction, that's diagnostic information, not a bug to hide.
"""

import datetime
import numpy as np
import matplotlib.pyplot as plt
from mathutils import Vector
from skyfield.api import Loader
import mathutils as mu
import os

from satlight.geometry import Geometry
from satlight.raytracer import Renderer
from tqdm import tqdm

# ---------------------------------------------------------------------
# Config - same scenario as the existing single-night script
# ---------------------------------------------------------------------
obj_path = r'/Users/l.beesley@bham.ac.uk/Documents/Lightcurves/satlight/models/Skynet 5D/skynet.obj'

observer_lat, observer_lon, observer_alt_m = 28.7622, -17.8897, 2396.0
geo_longitude, geo_altitude_km, R_EARTH_KM = -28.0, 35786.0, 6371.0
a_km = R_EARTH_KM + geo_altitude_km

_data_dir = os.path.join(os.path.expanduser('~'), '.satlight')
load = Loader(_data_dir)
ts = load.timescale()

t_start = datetime.datetime(2025, 1, 7, 20, 0, 0)
obs_duration_h, cadence_min, min_elevation = 12, 2, 5.0
resolution, solar_constant, mag_zeropoint = (1024, 256), 1361.0, -18.0

solar_panel_names = ['solarPanel1', 'solarPanel2']

n_steps = int(obs_duration_h * 60 / cadence_min)
time_grid = [t_start + datetime.timedelta(minutes=i * cadence_min) for i in range(n_steps)]


def gmst_deg(t_utc):
    t = ts.utc(t_utc.year, t_utc.month, t_utc.day, t_utc.hour, t_utc.minute, t_utc.second)
    return t.gast * 15.0


def geo_raan(target_lon_deg, t_utc):
    return (target_lon_deg + gmst_deg(t_utc)) % 360.0


def make_geometry(t_utc):
    geom = Geometry()
    geom.create_observer(observer_lat, observer_lon, observer_alt_m)
    geom.create_satellite_from_elements(
        a_km=a_km, e=0.0, i_deg=0.0,
        raan_deg=geo_raan(geo_longitude, t_utc),
        argp_deg=0.0, nu_deg=0.0, epoch=t_utc,
    )
    geom.set_time((t_utc.year, t_utc.month, t_utc.day, t_utc.hour, t_utc.minute, t_utc.second))
    return geom


# ---------------------------------------------------------------------
# Placeholder Kubelka-Munk -> effective-parameter map (see docstring)
# ---------------------------------------------------------------------
def kubelka_munk_to_effective_params(fluence_norm, base_albedo, base_roughness):
    """
    fluence_norm: 0 = pristine, 1 = fully saturated ageing (toy units).
    Returns (albedo, roughness) - saturating exponential toward a dimmer,
    rougher asymptote, standing in for a real K/S(lambda, Phi) fit.
    """
    albedo_floor = 0.55 * base_albedo
    roughness_ceiling = min(1.0, base_roughness + 0.35)

    saturation = 1 - np.exp(-3 * fluence_norm)
    albedo = base_albedo - (base_albedo - albedo_floor) * saturation
    roughness = base_roughness + (roughness_ceiling - base_roughness) * saturation
    return float(albedo), float(roughness)


def build_brdf_dict(fluence_norm):
    base_albedo, base_roughness, metallic = 0.6, 0.25, 0.9
    albedo, roughness = kubelka_munk_to_effective_params(fluence_norm, base_albedo, base_roughness)
    return {
        'solarPanel':     ('cook_torrance', {'albedo': 0.85, 'roughness': 0.15, 'metallic': 0.95}),
        'Antenna':        ('cook_torrance', {'albedo': 0.85, 'roughness': 0.15, 'metallic': 0.95}),
        'omniAntenna':    ('cook_torrance', {'albedo': 0.7,  'roughness': 0.3,  'metallic': 0.9}),
        'bus(Aluminium)': ('cook_torrance', {'albedo': albedo, 'roughness': roughness, 'metallic': metallic}),
        'bus(MLI)':       ('oren_nayar',    {'albedo': 0.25, 'roughness': 0.5}),
        'default':        ('lambertian',    {'albedo': 0.3}),
    }, albedo, roughness


# ---------------------------------------------------------------------
# Render one night's lightcurve for a given BRDF dict
# ---------------------------------------------------------------------
def render_night(brdf_dict, label):
    geom0 = make_geometry(t_start)
    sun_lvlh_0 = geom0.incident_vector_lvlh
    obs_lvlh_0 = geom0.outgoing_vector_lvlh
    dist_obs_0 = float(geom0.prop_distance) * 1e3

    renderer = Renderer(
        obj_path=obj_path,
        sun_direction=Vector(sun_lvlh_0.tolist()),
        observer_direction=Vector(obs_lvlh_0.tolist()),
        distance_to_observer=dist_obs_0,
        resolution=resolution,
        solar_constant=solar_constant,
        add_earthshine=False,
        brdf=brdf_dict,
    )
    renderer.initialise_scene()

    ATTITUDE_RX, ATTITUDE_RY, ATTITUDE_RZ = np.radians(0), np.radians(270), np.radians(90)
    body_euler = mu.Euler((ATTITUDE_RX, ATTITUDE_RY, ATTITUDE_RZ), 'XYZ')
    body_mat = body_euler.to_matrix()
    body_mat_np = np.array(body_mat)

    panel_offsets_deg = {'solarPanel1': +7.0, 'solarPanel2': -7.0}
    panel_set = set(solar_panel_names)

    import bpy
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH' and obj.name not in panel_set:
            obj.rotation_euler = body_mat.to_euler()
    bpy.context.view_layer.update()

    times_out, fluxes, magnitudes = [], [], []

    for t_utc in tqdm(time_grid, desc=label):
        geom = make_geometry(t_utc)

        sat_pos, obs_pos = geom.positions['satellite'], geom.positions['observer']
        diff = sat_pos - obs_pos
        zenith = obs_pos / np.linalg.norm(obs_pos)
        el_deg = float(np.degrees(np.arcsin(np.clip(
            np.dot(diff / np.linalg.norm(diff), zenith), -1, 1))))
        if el_deg < min_elevation:
            continue

        illum = float(geom.eclipse_type())
        if illum < 0.01:
            continue

        sun_lvlh = geom.incident_vector_lvlh
        obs_lvlh = geom.outgoing_vector_lvlh
        dist_obs = float(geom.prop_distance) * 1e3
        sun_body = body_mat_np.T @ sun_lvlh
        sun_xy = np.array([sun_body[0], sun_body[1], 0.0])
        sun_xy /= np.linalg.norm(sun_xy)
        panel_angle = float(np.arctan2(-sun_xy[0], sun_xy[1]))

        for name in solar_panel_names:
            obj = bpy.data.objects.get(name)
            if obj is not None:
                offset_rad = np.radians(panel_offsets_deg.get(name, 0.0))
                panel_drive = mu.Matrix.Rotation(panel_angle + offset_rad, 3, 'Z')
                obj.rotation_euler = (body_mat @ panel_drive).to_euler()

        renderer.update(
            sun_direction=Vector(sun_lvlh.tolist()),
            observer_direction=Vector(obs_lvlh.tolist()),
            distance_to_observer=dist_obs,
        )
        bpy.context.view_layer.update()

        image = renderer.render()
        flux = float(np.sum(image)) * illum
        mag = -2.5 * np.log10(flux) + mag_zeropoint if flux > 0 else np.nan

        times_out.append(t_utc)
        fluxes.append(flux)
        magnitudes.append(mag)

    return np.array(times_out), np.array(fluxes), np.array(magnitudes)


# ---------------------------------------------------------------------
# Run + report - no asserts, just prints what happened so you can debug
# in a notebook cell without the whole thing dying halfway through.
#
# Dimmer + more diffuse would show up as:
#   - higher mean magnitude (fainter)
#   - lower peak-to-trough amplitude (flatter phase response)
# If you see the opposite, that's worth digging into rather than
# suppressing - e.g. higher roughness on a near-normal-incidence pass can
# actually brighten a surface by spreading specular return toward the
# observer instead of past it, so "more diffuse" doesn't always mean
# "dimmer" at every phase angle. Plotting the two curves together (below)
# usually makes it obvious whether that's what's going on.
# ---------------------------------------------------------------------
def compare_aged_vs_pristine():
    brdf_pristine, alb_p, rough_p = build_brdf_dict(fluence_norm=0.0)
    brdf_aged, alb_a, rough_a = build_brdf_dict(fluence_norm=1.0)

    print(f"Pristine bus(Aluminium): albedo={alb_p:.3f} roughness={rough_p:.3f}")
    print(f"Aged     bus(Aluminium): albedo={alb_a:.3f} roughness={rough_a:.3f}")

    t_p, f_p, m_p = render_night(brdf_pristine, "pristine")
    t_a, f_a, m_a = render_night(brdf_aged, "aged")

    mean_mag_pristine = np.nanmean(m_p)
    mean_mag_aged = np.nanmean(m_a)
    amp_pristine = np.nanmax(m_p) - np.nanmin(m_p)
    amp_aged = np.nanmax(m_a) - np.nanmin(m_a)

    print(f"\nMean magnitude - pristine: {mean_mag_pristine:.3f}, aged: {mean_mag_aged:.3f} "
          f"(delta {mean_mag_aged - mean_mag_pristine:+.3f} mag)")
    print(f"Peak-to-trough amplitude - pristine: {amp_pristine:.3f}, aged: {amp_aged:.3f} "
          f"(delta {amp_aged - amp_pristine:+.3f} mag)")

    if mean_mag_aged > mean_mag_pristine:
        print("-> aged is dimmer on average, as expected")
    else:
        print("-> aged is BRIGHTER than pristine on average - check the plot, "
              "this can happen at phase angles where extra roughness scatters "
              "more light toward the observer")

    if amp_aged < amp_pristine:
        print("-> aged has a flatter lightcurve (more diffuse), as expected")
    else:
        print("-> aged has a LARGER swing than pristine - check the plot")

    return (t_p, m_p), (t_a, m_a)


def plot_comparison(pristine, aged, out_path=None):
    (t_p, m_p), (t_a, m_a) = pristine, aged
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_p, m_p, 'b.-', label='Pristine aluminium')
    ax.plot(t_a, m_a, 'r.-', label='Aged aluminium (toy K-M proxy)')
    ax.invert_yaxis()
    ax.set_ylabel('Magnitude')
    ax.set_title('Skynet 5D - pristine vs aged bus(Aluminium), single night (28W GEO)')
    ax.legend()
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
    plt.show()


if __name__ == "__main__":
    pristine, aged = compare_aged_vs_pristine()
    plot_comparison(
        pristine, aged,
        out_path='/Users/l.beesley@bham.ac.uk/Documents/Lightcurves/satlight/output/aged_vs_pristine_test.png',
    )