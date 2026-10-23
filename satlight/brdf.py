"""Bidirectional reflectance distribution functions.

HOW THE RENDERER CALLS THESE
----------------------------
`Renderer.render()` calls a BRDF once per lit pixel per source::

    value = brdf_func(normal, light_dir, view_dir, **kwargs)

with::

    normal     unit shading normal at the hit point
    light_dir  unit vector pointing TO the source (Sun, or an Earth patch)
    view_dir   unit vector pointing TO the observer
    kwargs     whatever the material spec carried

all three in the render frame (LVLH, with Z zenith), and the return a single
float.

THE ONE CONVENTION THAT MATTERS
-------------------------------
**The returned value is f_r * cos(theta_i), NOT f_r.** The incident cosine is
folded into the kernel, so every function here ends with a `* n_dot_l`. Write
a custom BRDF without it and the body comes out uniformly lit however the Sun
strikes it - a wrong answer that still looks like a spacecraft.

A BRDF must also return exactly `0.0` when `dot(normal, light_dir) <= 0`: the
renderer does not clamp for you, and a negative value would subtract light.

Units are sr^-1, so kernel * irradiance = radiance toward the observer. The
renderer multiplies by `solar_constant * pixel_area / range^2` for the Sun and
by the per-patch irradiance for earthshine.

HOW OFTEN
---------

::

    direct Sun        1 call per lit pixel
    earthshine        1 call per lit Earth patch per lit pixel
                      (88 patches at LEO with n_rings=4)
    n_bounce > 1      bounce_samples extra calls per pixel per extra bounce

A BRDF is on the hot path: a 400 px frame makes ~9000 calls for sunlight
alone. Keep it scalar and cheap - numpy's per-call overhead on 3-vectors costs
more than the arithmetic.

SPECIFYING A MATERIAL
---------------------
`Renderer(brdf=...)` accepts::

    'lambertian'                     a name from BRDF_REGISTRY
    my_func                          any callable with the signature above
    ('lambertian', {'albedo': 0.3})  a (spec, kwargs) pair
    {'solar': ..., 'default': ...}   per-object, keyed by NAME SUBSTRING

For the dict form a key matches when it appears anywhere in the object's name,
compared case-insensitively, so 'solar' catches `solar_left_front`. 'default'
catches anything unmatched.

WRITING ONE
-----------

::

    def my_brdf(normal, light_dir, view_dir, albedo=0.3):
        n_dot_l = float(np.dot(normal, light_dir))
        if n_dot_l <= 0.0:
            return 0.0                    # never negative, never unclamped
        return albedo / np.pi * n_dot_l   # f_r * cos(theta_i)

    Renderer(obj, geometry=geom, brdf=(my_brdf, {'albedo': 0.3}))

That reproduces `lambertian` exactly.
"""
import numpy as np
 
_EPS = 1e-6
 
 
def _halfway(light_dir, view_dir):
    """Unit half vector between incident and outgoing directions.
 
    Returns None in the degenerate anti-parallel case where l + v = 0.
    """
    h = np.asarray(light_dir, float) + np.asarray(view_dir, float)
    norm = np.linalg.norm(h)
    if norm < _EPS:
        return None
    return h / norm
 
 
def _orthonormal_tangent(normal, tangent=None):
    """Unit tangent perpendicular to `normal`.
 
    If `tangent` is supplied it is Gram-Schmidt orthogonalised against the
    normal; otherwise an arbitrary perpendicular is generated. An arbitrary
    tangent is fine for isotropic surfaces (nu == nv) but meaningless for
    anisotropic ones, which need an explicit per-material axis.
    """
    n = np.asarray(normal, float)
    if tangent is None:
        seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    else:
        seed = np.asarray(tangent, float)
    t = seed - np.dot(seed, n) * n
    norm = np.linalg.norm(t)
    if norm < _EPS:
        seed = np.array([0.0, 0.0, 1.0])
        t = seed - np.dot(seed, n) * n
        norm = np.linalg.norm(t)
    return t / norm
 
 
def lambertian(normal, light_dir, view_dir, albedo=0.5):
    """Ideal diffuse (Lambertian) reflectance.
 
    f_r = albedo / pi, independent of viewing direction.
 
    Reference
    ---------
    Lambert, J. H. (1760), "Photometria". English translation by
    D. L. DiLaura, IESNA, 2001.
    """
    n_dot_l = np.dot(normal, light_dir)
    if n_dot_l <= 0.0:
        return 0.0
    return albedo / np.pi * n_dot_l
 
 
def phong(normal, light_dir, view_dir, albedo=0.5, specular=0.5, shininess=32):
    """Normalised Phong: Lambertian diffuse plus a mirror-lobe specular term.
 
    Empirical, not physically derived. Retained for comparison with the
    older light curve literature; prefer `cook_torrance` or
    `ashikhmin_shirley` for physically-based work.
 
    References
    ----------
    Phong, B. T. (1975), Commun. ACM 18(6), 311-317.
    Lafortune, E. P. and Willems, Y. D. (1994), "Using the Modified Phong
    Reflectance Model for Physically Based Rendering", TR CW 197, KU Leuven.
    (normalisation factor (n+2)/2pi)
    """
    n_dot_l = np.dot(normal, light_dir)
    if n_dot_l <= 0.0:
        return 0.0
 
    reflect_dir = 2 * n_dot_l * np.asarray(normal, float) - np.asarray(light_dir, float)
    spec_angle = max(0.0, np.dot(reflect_dir, view_dir))
 
    diffuse = (1.0 - specular) * albedo / np.pi
    spec = specular * (shininess + 2) / (2 * np.pi) * spec_angle ** shininess
    return (diffuse + spec) * n_dot_l
 
 
def blinn_phong(normal, light_dir, view_dir, albedo=0.5, specular=0.5, shininess=32):
    """Blinn-Phong: as `phong` but with the lobe built on the half vector.
 
    References
    ----------
    Blinn, J. F. (1977), Computer Graphics (SIGGRAPH '77) 11(2), 192-198.
    Lafortune, E. P. and Willems, Y. D. (1994), TR CW 197, KU Leuven.
    (normalisation factor (n+8)/8pi)
    """
    n_dot_l = np.dot(normal, light_dir)
    if n_dot_l <= 0.0:
        return 0.0
 
    halfway = _halfway(light_dir, view_dir)
    if halfway is None:
        return 0.0
    n_dot_h = max(0.0, np.dot(normal, halfway))
 
    diffuse = (1.0 - specular) * albedo / np.pi
    spec = specular * (shininess + 8) / (8 * np.pi) * n_dot_h ** shininess
    return (diffuse + spec) * n_dot_l
 
 
def cook_torrance(normal, light_dir, view_dir, albedo=0.3, roughness=0.5, metallic=0.1):
    """Cook-Torrance microfacet BRDF with a GGX distribution.
 
    f_r = D * F * G / (4 cos(theta_i) cos(theta_o)), plus a Lambertian
    diffuse lobe weighted by the untransmitted fraction. The specular term
    below is divided by 4*n_dot_v only, because the module convention already
    folds the n_dot_l factor into the returned kernel.
 
    Parameters
    ----------
    roughness : GGX roughness; alpha = roughness**2.
    metallic  : 0 = dielectric (F0 = 0.04), 1 = conductor (F0 = albedo).
 
    References
    ----------
    - Torrance, K. E. and Sparrow, E. M. (1967), JOSA 57(9), 1105-1114.
    - Cook, R. L. and Torrance, K. E. (1982), ACM TOG 1(1), 7-24.
    - Trowbridge, T. S. and Reitz, K. P. (1975), JOSA 65(5), 531-536
      (the D term, later renamed GGX).
    - Walter, B. et al. (2007), EGSR '07, 195-206.
    - Smith, B. G. (1967), IEEE Trans. Antennas Propag. 15(5), 668-671
      (the G term).
    - Schlick, C. (1994), Computer Graphics Forum 13(3), 233-246
      (the F approximation).
    """
    n_dot_l = np.dot(normal, light_dir)
    n_dot_v = np.dot(normal, view_dir)
    if n_dot_l <= 0.0 or n_dot_v <= 0.0:
        return 0.0
    n_dot_l = max(_EPS, n_dot_l)
    n_dot_v = max(_EPS, n_dot_v)
 
    halfway = _halfway(light_dir, view_dir)
    if halfway is None:
        return 0.0
    n_dot_h = max(0.0, np.dot(normal, halfway))
    v_dot_h = max(_EPS, np.dot(view_dir, halfway))
 
    # Fresnel (Schlick approximation)
    F0 = 0.04 * (1 - metallic) + albedo * metallic
    fresnel = F0 + (1 - F0) * (1 - v_dot_h) ** 5
 
    # GGX / Trowbridge-Reitz normal distribution
    alpha = roughness * roughness
    alpha2 = alpha * alpha
    denom = n_dot_h * n_dot_h * (alpha2 - 1) + 1
    D = alpha2 / (np.pi * denom * denom)
 
    # Smith geometric shadowing-masking, Schlick-GGX form for direct lighting
    k = (roughness + 1) ** 2 / 8
 
    def G1(n_dot_x):
        return n_dot_x / (n_dot_x * (1 - k) + k)
 
    G = G1(n_dot_l) * G1(n_dot_v)
 
    specular = (D * G * fresnel) / (4 * n_dot_v)
    diffuse = (1 - fresnel) * (1 - metallic) * albedo / np.pi * n_dot_l
    return diffuse + specular
 
 
def oren_nayar(normal, light_dir, view_dir, albedo=0.5, roughness=0.5):
    """Oren-Nayar rough-diffuse BRDF (qualitative A/B form).
 
    Models a Lambertian surface roughened by Gaussian-distributed microfacet
    slopes of standard deviation `roughness` (radians), giving the flattened,
    backscatter-weighted response of dusty or matte surfaces.
 
    References
    ----------
    Oren, M. and Nayar, S. K. (1994), SIGGRAPH '94, 239-246.
    Oren, M. and Nayar, S. K. (1995), Int. J. Comput. Vis. 14(3), 227-251.
    """
    n_dot_l = np.dot(normal, light_dir)
    n_dot_v = np.dot(normal, view_dir)
    if n_dot_l <= 0.0 or n_dot_v <= 0.0:
        return 0.0
 
    n_dot_l = min(1.0, n_dot_l)
    n_dot_v = min(1.0, n_dot_v)
    alpha = max(np.arccos(n_dot_v), np.arccos(n_dot_l))
    beta = min(np.arccos(n_dot_v), np.arccos(n_dot_l))
 
    # Azimuthal difference between the light and view directions projected
    # into the local tangent plane.
    sin_l = np.sqrt(max(0.0, 1 - n_dot_l ** 2))
    sin_v = np.sqrt(max(0.0, 1 - n_dot_v ** 2))
    if sin_l < _EPS or sin_v < _EPS:
        cos_phi_diff = 0.0
    else:
        cos_phi_diff = np.dot(
            light_dir - n_dot_l * np.asarray(normal, float),
            view_dir - n_dot_v * np.asarray(normal, float),
        ) / (sin_l * sin_v)
        cos_phi_diff = float(np.clip(cos_phi_diff, -1.0, 1.0))
 
    sigma2 = roughness * roughness
    A = 1.0 - 0.5 * sigma2 / (sigma2 + 0.33)
    B = 0.45 * sigma2 / (sigma2 + 0.09)
 
    return (albedo / np.pi) * n_dot_l * (
        A + B * max(0.0, cos_phi_diff) * np.sin(alpha) * np.tan(beta)
    )
 
 
def ashikhmin_shirley(normal, light_dir, view_dir, tangent=None,
                      albedo=0.3, specular=0.05, nu=100.0, nv=100.0):
    """Ashikhmin-Shirley anisotropic Phong BRDF.
 
    A Fresnel-weighted anisotropic cosine lobe over a non-Lambertian diffuse
    term, normalised so directional-hemispherical reflectance stays below
    unity. Widely adopted in the SSA light curve.
 
    Parameters
    ----------
    tangent  : surface tangent defining the anisotropy axis. Required for
               nu != nv; ignored when nu == nv. If omitted an arbitrary
               perpendicular is generated, which is only valid isotropically.
    specular : normal-incidence specular reflectance Rs.
    nu, nv   : lobe exponents along the tangent and bitangent axes. Equal
               values give an isotropic lobe; large values give a tight glint.
 
    References
    ----------
    - Ashikhmin, M. and Shirley, P. (2000), J. Graphics Tools 5(2), 25-32.
    - Wetterer, C. J. et al. (2014), J. Guid. Control Dyn. 37(1), 185-196
      (adoption for space object modelling).
    """
    n = np.asarray(normal, float)
    n_dot_l = np.dot(n, light_dir)
    n_dot_v = np.dot(n, view_dir)
    if n_dot_l <= 0.0 or n_dot_v <= 0.0:
        return 0.0
 
    halfway = _halfway(light_dir, view_dir)
    if halfway is None:
        return 0.0
    n_dot_h = float(np.clip(np.dot(n, halfway), 0.0, 1.0))
    h_dot_l = max(_EPS, np.dot(halfway, light_dir))
 
    # Anisotropic exponent. Degenerates to nu at normal incidence of h.
    if abs(nu - nv) < _EPS or n_dot_h >= 1.0 - _EPS:
        exponent = nu
    else:
        t = _orthonormal_tangent(n, tangent)
        b = np.cross(n, t)
        h_dot_t = np.dot(halfway, t)
        h_dot_b = np.dot(halfway, b)
        exponent = (nu * h_dot_t ** 2 + nv * h_dot_b ** 2) / (1.0 - n_dot_h ** 2)
 
    fresnel = specular + (1 - specular) * (1 - h_dot_l) ** 5
 
    spec = (np.sqrt((nu + 1) * (nv + 1)) / (8 * np.pi)
            * n_dot_h ** exponent
            / (h_dot_l * max(n_dot_l, n_dot_v))
            * fresnel)
 
    diffuse = (28 * albedo / (23 * np.pi) * (1 - specular)
               * (1 - (1 - n_dot_l / 2) ** 5)
               * (1 - (1 - n_dot_v / 2) ** 5))
 
    return (diffuse + spec) * n_dot_l
 
 
def abg(normal, light_dir, view_dir, A=1e-3, B=1e-4, g=2.0):
    """ABg empirical scatter model (Zemax convention).
 
    ``f_r = A / (B + |beta - beta_0|^g)``, with beta the outgoing direction
    projected into the surface plane. Specular peak A/B; power-law tail of
    slope -g beyond the knee at B^(1/g). Fits a sharp spike sitting on a broad
    shoulder, which a single microfacet lobe cannot.
    """
    n = np.asarray(normal, float)
    l = np.asarray(light_dir, float)
    v = np.asarray(view_dir, float)
    n_dot_l = np.dot(n, l)
    if n_dot_l <= 0.0:
        return 0.0
    r = 2.0 * n_dot_l * n - l
    d = (v - np.dot(n, v) * n) - (r - np.dot(n, r) * n)
    return A / (B + np.linalg.norm(d) ** g) * n_dot_l
 
 
def satellite(normal, light_dir, view_dir, surface_type='solar_panel',
              model='default', tangent=None):
    """Representative material presets for common spacecraft surfaces.
 
    Parameters
    ----------
    model : 'default' uses the per-surface model listed below;
            'ashikhmin_shirley' renders every surface with A-S instead, for
            like-for-like comparison against the published SSA literature.
    """
    if surface_type not in ('solar_panel', 'body', 'antenna'):
        raise ValueError(f"unknown surface_type: {surface_type!r}")
 
    presets_as = {
        # albedo, Rs, nu, nv
        'solar_panel': (0.15, 0.85, 1000.0, 100.0),
        'body': (0.30, 0.05, 20.0, 20.0),
        'antenna': (0.80, 0.60, 400.0, 400.0),
    }
 
    if model == 'ashikhmin_shirley':
        albedo, rs, nu, nv = presets_as[surface_type]
        return ashikhmin_shirley(normal, light_dir, view_dir, tangent=tangent,
                                 albedo=albedo, specular=rs, nu=nu, nv=nv)
 
    if surface_type == 'solar_panel':
        # Specular with some diffuse
        return blinn_phong(normal, light_dir, view_dir,
                           albedo=0.15, specular=0.85, shininess=64)
    if surface_type == 'body':
        # Mostly diffuse with slight roughness
        return oren_nayar(normal, light_dir, view_dir,
                          albedo=0.3, roughness=0.4)
    # Metallic antenna with moderate roughness
    return cook_torrance(normal, light_dir, view_dir,
                         albedo=0.8, roughness=0.3, metallic=0.9)
