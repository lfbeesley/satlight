import numpy as np

def lambertian(normal, light_dir, view_dir, albedo = 1):
    """Lambertian shading"""
    n_dot_l = max(0, np.dot(normal, light_dir))
    return albedo / np.pi * n_dot_l

def phong(normal, light_dir, view_dir, albedo=0.5, specular=0.5, shininess=32):
    """Phong shading with diffuse + specular components"""
    # Diffuse component
    n_dot_l = max(0, np.dot(normal, light_dir))
    diffuse = albedo / np.pi * n_dot_l
    
    # Specular component (Phong)
    reflect_dir = 2 * n_dot_l * normal - light_dir
    spec_angle = max(0, np.dot(reflect_dir, view_dir))
    specular_component = specular * (shininess + 2) / (2 * np.pi) * (spec_angle ** shininess)
    
    return diffuse + specular_component

def blinn_phong(normal, light_dir, view_dir, albedo=0.5, specular=0.5, shininess=32):
    """Blinn-Phong shading"""
    # Diffuse
    n_dot_l = max(0, np.dot(normal, light_dir))
    diffuse = albedo / np.pi * n_dot_l
    
    # Specular (using halfway vector)
    halfway = (light_dir + view_dir) / np.linalg.norm(light_dir + view_dir)
    n_dot_h = max(0, np.dot(normal, halfway))
    specular_component = specular * (shininess + 8) / (8 * np.pi) * (n_dot_h ** shininess)
    
    return diffuse + specular_component

def cook_torrance(normal, light_dir, view_dir, albedo=0.3, roughness=0.5, metallic=0.1):
    """Cook-Torrance microfacet BRDF - more realistic for metals/glossy surfaces"""
    n_dot_l = max(0.001, np.dot(normal, light_dir))
    n_dot_v = max(0.001, np.dot(normal, view_dir))
    
    # Halfway vector
    halfway = (light_dir + view_dir) / np.linalg.norm(light_dir + view_dir)
    n_dot_h = max(0, np.dot(normal, halfway))
    v_dot_h = max(0.001, np.dot(view_dir, halfway))
    
    # Fresnel (Schlick approximation)
    F0 = 0.04 * (1 - metallic) + albedo * metallic
    fresnel = F0 + (1 - F0) * (1 - v_dot_h) ** 5
    
    # GGX/Trowbridge-Reitz distribution
    alpha = roughness * roughness
    alpha2 = alpha * alpha
    denom = n_dot_h * n_dot_h * (alpha2 - 1) + 1
    D = alpha2 / (np.pi * denom * denom)
    
    # Geometry term (Smith)
    def G1(n_dot_x):
        k = alpha / 2
        return n_dot_x / (n_dot_x * (1 - k) + k)
    
    G = G1(n_dot_l) * G1(n_dot_v)
    
    # Specular term
    specular = (D * G * fresnel) / (4 * n_dot_l * n_dot_v)
    
    # Diffuse term (with energy conservation)
    diffuse = (1 - fresnel) * (1 - metallic) * albedo / np.pi * n_dot_l
    
    return diffuse + specular

def oren_nayar(normal, light_dir, view_dir, albedo=0.5, roughness=0.5):
    """Oren-Nayar - accounts for surface roughness better than Lambertian"""
    n_dot_l = max(0, np.dot(normal, light_dir))
    n_dot_v = max(0, np.dot(normal, view_dir))
    
    # Angle between light and view in tangent plane
    angle_vn = np.arccos(n_dot_v)
    angle_ln = np.arccos(n_dot_l)
    alpha = max(angle_vn, angle_ln)
    beta = min(angle_vn, angle_ln)
    
    # Compute azimuthal difference
    cos_phi_diff = np.dot(
        light_dir - n_dot_l * normal,
        view_dir - n_dot_v * normal
    )
    if n_dot_l > 0.001 and n_dot_v > 0.001:
        cos_phi_diff /= (np.sqrt(1 - n_dot_l**2) * np.sqrt(1 - n_dot_v**2) + 1e-6)
    cos_phi_diff = max(-1, min(1, cos_phi_diff))
    
    # Oren-Nayar parameters
    sigma2 = roughness * roughness
    A = 1.0 - 0.5 * sigma2 / (sigma2 + 0.33)
    B = 0.45 * sigma2 / (sigma2 + 0.09)
    
    return (albedo / np.pi) * n_dot_l * (A + B * max(0, cos_phi_diff) * np.sin(alpha) * np.tan(beta))

def satellite(normal, light_dir, view_dir, surface_type='solar_panel'):
    """Realistic model for different satellite surfaces"""
    if surface_type == 'solar_panel':
        # Solar panels: specular with some diffuse
        return blinn_phong(normal, light_dir, view_dir, 
                               albedo=0.15, specular=0.85, shininess=64)
    elif surface_type == 'body':
        # Body: mostly diffuse with slight roughness
        return oren_nayar(normal, light_dir, view_dir,
                              albedo=0.3, roughness=0.4)
    elif surface_type == 'antenna':
        # Metal antenna: metallic with moderate roughness
        return cook_torrance(normal, light_dir, view_dir,
                                 albedo=0.8, roughness=0.3, metallic=0.9)

