"""
gpulight.py -- GPU light-curve renderer in pure PyTorch.

WHY PYTORCH AND NOT OPENGL
On a Mac the realistic GPU paths are OpenGL 4.1 (deprecated, needs GLSL) or
Metal via PyTorch's MPS backend. This uses the latter: everything here is
standard tensor ops, so the same file runs on MPS, CUDA or CPU by changing
one string. No shaders, no custom kernels, nothing to compile.

It also means the whole pipeline is differentiable. If you ever want to invert
a light curve for attitude or BRDF parameters, autograd gives you
d(flux)/d(parameters) for free -- gradient descent instead of grid search.

WHAT IT COMPUTES
  * flux    -- analytic per triangle (exact silhouette, no pixel quantisation)
  * image   -- radiance per pixel, for diagnosing WHY the magnitude moved
  * part_id -- which component owns each pixel, so a glint is a lookup
Shadows use a shadow map (orthographic rasterization from the sun), not rays.

Run `python gpulight.py` for a self-test that checks the GPU path against
known-good CPU values and benchmarks your device.
"""

import numpy as np
import torch


def pick_device(prefer='auto'):
    if prefer != 'auto':
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# =====================================================================
# BRDFs -- return f_r * (n.l)
# =====================================================================
def _d(a, b):
    return (a * b).sum(-1)


def lambertian(n, l, v, albedo=0.3):
    return (albedo / np.pi) * _d(n, l).clamp_min(0)


def blinn_phong(n, l, v, albedo=0.3, specular=0.5, shininess=32):
    ndl = _d(n, l).clamp_min(0)
    h = torch.nn.functional.normalize(l + v, dim=-1)
    k = (shininess + 8.0) / (8.0 * np.pi)
    return (albedo / np.pi) * ndl + specular * k * _d(n, h).clamp_min(0) ** shininess * ndl


def cook_torrance(n, l, v, albedo=0.6, roughness=0.3, metallic=0.9):
    eps = 1e-9
    ndl = _d(n, l).clamp_min(eps)
    ndv = _d(n, v).clamp_min(eps)
    h = torch.nn.functional.normalize(l + v, dim=-1)
    ndh = _d(n, h).clamp_min(0)
    vdh = _d(v, h).clamp_min(eps)
    a = max(roughness, 1e-3) ** 2
    a2 = a * a
    dd = a2 / (np.pi * (ndh * ndh * (a2 - 1.0) + 1.0) ** 2 + eps)
    k = a / 2.0
    g = (ndl / (ndl * (1 - k) + k)) * (ndv / (ndv * (1 - k) + k))
    f0 = 0.04 * (1 - metallic) + albedo * metallic
    f = f0 + (1.0 - f0) * (1.0 - vdh) ** 5
    return (dd * g * f / (4.0 * ndl * ndv + eps)
            + (1.0 - metallic) * albedo / np.pi) * ndl


def oren_nayar(n, l, v, albedo=0.25, roughness=0.5):
    eps = 1e-9
    ndl = _d(n, l).clamp_min(0)
    ndv = _d(n, v).clamp_min(eps)
    s2 = roughness ** 2
    A = 1.0 - 0.5 * s2 / (s2 + 0.33)
    B = 0.45 * s2 / (s2 + 0.09)
    ti = torch.arccos(ndl.clamp(-1, 1))
    tr = torch.arccos(ndv.clamp(-1, 1))
    alpha = torch.maximum(ti, tr)
    beta = torch.minimum(ti, tr)
    lp = torch.nn.functional.normalize(l - ndl.unsqueeze(-1) * n, dim=-1)
    vp = torch.nn.functional.normalize(v - ndv.unsqueeze(-1) * n, dim=-1)
    cosd = _d(lp, vp).clamp(-1, 1).clamp_min(0)
    return (albedo / np.pi) * ndl * (A + B * cosd * torch.sin(alpha) * torch.tan(beta))


REGISTRY = {'lambertian': lambertian, 'blinn_phong': blinn_phong,
            'cook_torrance': cook_torrance, 'oren_nayar': oren_nayar}


# =====================================================================
def _basis(d):
    """Orthonormal rows (right, up, forward) for a direction."""
    d = torch.nn.functional.normalize(d, dim=0)
    ref = torch.tensor([0., 0., 1.], device=d.device, dtype=d.dtype)
    if abs(float(d[2])) > 0.9:
        ref = torch.tensor([1., 0., 0.], device=d.device, dtype=d.dtype)
    r = torch.nn.functional.normalize(torch.cross(ref, d, dim=0), dim=0)
    u = torch.cross(d, r, dim=0)
    return torch.stack([r, u, d])


def rasterize(V, F, B, res, centre, half, cull=None, want_id=True):
    """
    Tensor z-buffer rasterization. Returns (depth_buf, id_buf), both (res,res),
    depth = -inf and id = -1 where empty.
    """
    dev, dt = V.device, V.dtype
    Fi = F if cull is None else F[cull]
    idx_map = None if cull is None else torch.nonzero(cull, as_tuple=True)[0]
    if Fi.shape[0] == 0:
        return (torch.full((res, res), -float('inf'), device=dev, dtype=dt),
                torch.full((res, res), -1, device=dev, dtype=torch.long))

    P = (V - centre) @ B.T
    sx = (P[:, 0] / half * 0.5 + 0.5) * (res - 1)
    sy = (P[:, 1] / half * 0.5 + 0.5) * (res - 1)

    x, y, z = sx[Fi], sy[Fi], P[:, 2][Fi]
    x0 = x.min(1).values.floor().clamp(0, res - 1).long()
    x1 = x.max(1).values.ceil().clamp(0, res - 1).long()
    y0 = y.min(1).values.floor().clamp(0, res - 1).long()
    y1 = y.max(1).values.ceil().clamp(0, res - 1).long()

    w, h = (x1 - x0 + 1).clamp_min(0), (y1 - y0 + 1).clamp_min(0)
    cnt = w * h
    keep = cnt > 0
    if not bool(keep.any()):
        return (torch.full((res, res), -float('inf'), device=dev, dtype=dt),
                torch.full((res, res), -1, device=dev, dtype=torch.long))

    kidx = torch.nonzero(keep, as_tuple=True)[0]
    ck = cnt[keep]
    tri = torch.repeat_interleave(kidx, ck)
    starts = torch.cat([torch.zeros(1, device=dev, dtype=torch.long),
                        ck.cumsum(0)[:-1]])
    local = torch.arange(int(ck.sum()), device=dev) - torch.repeat_interleave(starts, ck)
    ww = w[tri]
    px = x0[tri] + local % ww
    py = y0[tri] + local // ww

    ax, ay = x[tri, 0], y[tri, 0]
    bx, by = x[tri, 1], y[tri, 1]
    cx, cy = x[tri, 2], y[tri, 2]
    pxf, pyf = px.to(dt), py.to(dt)
    den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    den = torch.where(den.abs() < 1e-12, torch.full_like(den, 1e-12), den)
    l0 = ((by - cy) * (pxf - cx) + (cx - bx) * (pyf - cy)) / den
    l1 = ((cy - ay) * (pxf - cx) + (ax - cx) * (pyf - cy)) / den
    l2 = 1.0 - l0 - l1
    inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)

    tri, px, py = tri[inside], px[inside], py[inside]
    depth = l0[inside] * z[tri, 0] + l1[inside] * z[tri, 1] + l2[inside] * z[tri, 2]
    pix = py * res + px

    dbuf = torch.full((res * res,), -float('inf'), device=dev, dtype=dt)
    dbuf = dbuf.scatter_reduce(0, pix, depth, reduce='amax', include_self=True)

    ibuf = torch.full((res * res,), -1, device=dev, dtype=torch.long)
    if want_id:
        win = depth >= dbuf[pix] - 1e-12
        src = tri[win] if idx_map is None else idx_map[tri[win]]
        ibuf = ibuf.scatter(0, pix[win], src)
    return dbuf.view(res, res), ibuf.view(res, res)


# =====================================================================
class GPURenderer:
    def __init__(self, parts, brdf=None, solar_constant=1361.0,
                 device='auto', dtype=torch.float32):
        """
        parts : list of dicts {'name', 'V' (Nv,3), 'F' (Nf,3),
                               'pivot' (3,) optional, 'axis' (3,) optional}
        """
        self.dev = pick_device(device)
        self.dt = dtype
        self.solar_constant = solar_constant
        self.names = [p['name'] for p in parts]

        self.V, self.F, self.face_part, self.pivot, self.axis = [], [], [], [], []
        off = 0
        for i, p in enumerate(parts):
            V = torch.as_tensor(np.asarray(p['V']), dtype=dtype, device=self.dev)
            F = torch.as_tensor(np.asarray(p['F']), dtype=torch.long, device=self.dev)
            self.V.append(V)
            self.F.append(F + off)
            self.face_part.append(torch.full((F.shape[0],), i,
                                             dtype=torch.long, device=self.dev))
            self.pivot.append(torch.as_tensor(np.asarray(p.get('pivot', [0, 0, 0])),
                                              dtype=dtype, device=self.dev))
            a = p.get('axis')
            self.axis.append(None if a is None else
                             torch.as_tensor(np.asarray(a), dtype=dtype, device=self.dev))
            off += V.shape[0]
        self.vsplit = [v.shape[0] for v in self.V]
        self.V0 = torch.cat(self.V)
        self.Fc = torch.cat(self.F)
        self.face_part = torch.cat(self.face_part)
        self._resolve(brdf or {'default': ('lambertian', {})})

    def _resolve(self, brdf):
        default = brdf.get('default', ('lambertian', {}))
        unmatched = {k for k in brdf if k != 'default'}
        self.mat = []
        for nm in self.names:
            spec = default
            for k, s in brdf.items():
                if k != 'default' and k.lower() in nm.lower():
                    spec, _ = s, unmatched.discard(k)
                    break
            fn, kw = spec
            self.mat.append((REGISTRY[fn] if isinstance(fn, str) else fn, kw))
        if unmatched:
            print(f"  WARNING: BRDF keys matching no part: {sorted(unmatched)}")

    # -----------------------------------------------------------------
    def _rodrigues(self, axis, ang):
        a = torch.nn.functional.normalize(axis, dim=0)
        K = torch.zeros(3, 3, device=self.dev, dtype=self.dt)
        K[0, 1], K[0, 2] = -a[2], a[1]
        K[1, 0], K[1, 2] = a[2], -a[0]
        K[2, 0], K[2, 1] = -a[1], a[0]
        I = torch.eye(3, device=self.dev, dtype=self.dt)
        return I + torch.sin(ang) * K + (1 - torch.cos(ang)) * (K @ K)

    def transform(self, angles=None):
        """Apply hinge angles. This is the entire cost of articulation."""
        if angles is None:
            return self.V0
        out, s = [], 0
        for i, nvert in enumerate(self.vsplit):
            V = self.V0[s:s + nvert]
            a = angles[i] if i < len(angles) else None
            if a is not None and self.axis[i] is not None:
                ang = torch.as_tensor(a, dtype=self.dt, device=self.dev)
                R = self._rodrigues(self.axis[i], ang)
                V = (V - self.pivot[i]) @ R.T + self.pivot[i]
            out.append(V)
            s += nvert
        return torch.cat(out)

    # -----------------------------------------------------------------
    def render(self, sun, obs, distance, angles=None, res=256,
               shadow_res=512, want_image=True, shadows=True, pad=1.05,
               two_sided=False):
        dev, dt = self.dev, self.dt
        sun = torch.nn.functional.normalize(
            torch.as_tensor(np.asarray(sun), dtype=dt, device=dev), dim=0)
        obs = torch.nn.functional.normalize(
            torch.as_tensor(np.asarray(obs), dtype=dt, device=dev), dim=0)

        V = self.transform(angles)
        t = V[self.Fc]
        cr = torch.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0], dim=1)
        two_a = cr.norm(dim=1)
        n = cr / two_a.clamp_min(1e-30).unsqueeze(1)
        area = 0.5 * two_a

        ndv, ndl = n @ obs, n @ sun
        # two_sided=True flips normals toward the viewer, which is needed for
        # open/single-sided meshes -- but for a CLOSED mesh it makes the far
        # side front-facing too, doubling the flux unless occlusion is tested.
        # Proper backface culling is correct AND free for closed geometry.
        if two_sided:
            flip = torch.where(ndv < 0, -1.0, 1.0)
            ndv, ndl = ndv * flip, ndl * flip
        else:
            flip = torch.ones_like(ndv)
        live = (ndv > 1e-6) & (ndl > 1e-6)

        # Dynamic view volume -- tracks an actuating panel automatically.
        centre = 0.5 * (V.min(0).values + V.max(0).values)
        Bv, Bs = _basis(obs), _basis(sun)
        half_v = ((V - centre) @ Bv.T[:, :2]).abs().max() * pad
        half_s = ((V - centre) @ Bs.T[:, :2]).abs().max() * pad

        lit = torch.ones(self.Fc.shape[0], device=dev, dtype=dt)
        if shadows:
            sdep, _ = rasterize(V, self.Fc, Bs, shadow_res, centre, half_s,
                                cull=ndl > -1e-6, want_id=False)
            cen = t.mean(1)
            P = (cen - centre) @ Bs.T
            sx = ((P[:, 0] / half_s * 0.5 + 0.5) * (shadow_res - 1)).long().clamp(0, shadow_res - 1)
            sy = ((P[:, 1] / half_s * 0.5 + 0.5) * (shadow_res - 1)).long().clamp(0, shadow_res - 1)
            near = sdep.view(-1)[sy * shadow_res + sx]
            bias = 2.5 * half_s / shadow_res
            lit = (P[:, 2] >= near - bias).to(dt)

        # ---- analytic photometry (exact silhouette) --------------------
        rad = torch.zeros(self.Fc.shape[0], device=dev, dtype=dt)
        nrm = n * flip.unsqueeze(1)
        for m, (fn, kw) in enumerate(self.mat):
            sel = live & (self.face_part == m)
            if bool(sel.any()):
                rad[sel] = fn(nrm[sel], sun, obs, **kw)
        flux = (rad * self.solar_constant * ndv * area * lit
                * live.to(dt)).sum() / distance ** 2

        out = {'flux': float(flux), 'flux_t': flux}
        if want_image:
            vdep, vid = rasterize(V, self.Fc, Bv, res, centre, half_v,
                                  cull=ndv > -1e-6, want_id=True)
            img = torch.zeros(res * res, device=dev, dtype=dt)
            vf = vid.view(-1)
            hit = vf >= 0
            if bool(hit.any()):
                tf = vf[hit]
                img[hit] = (rad[tf] * self.solar_constant * ndv[tf] * lit[tf])
            out['image'] = img.view(res, res)
            out['part_id'] = torch.where(
                vf >= 0, self.face_part[vf.clamp_min(0)],
                torch.full_like(vf, -1)).view(res, res)
        return out


# =====================================================================
if __name__ == '__main__':
    import time, trimesh

    def build(subdiv=3):
        parts = []
        m = trimesh.creation.box((8.0, 2.6, 0.02)); m.apply_translation([0, 2.2, 0])
        for _ in range(subdiv): m = m.subdivide()
        parts.append({'name': 'solarPanel', 'V': m.vertices, 'F': m.faces,
                      'pivot': [0, 0.9, 0], 'axis': [1, 0, 0]})
        m = trimesh.creation.box((3.2, 1.6, 0.35))
        for _ in range(subdiv): m = m.subdivide()
        parts.append({'name': 'bus(MLI)', 'V': m.vertices, 'F': m.faces})
        for i, x in enumerate([-1.1, -0.35, 0.35, 1.1]):
            a = trimesh.creation.cylinder(radius=0.28, height=0.12, sections=48)
            a.apply_translation([x, -0.2, -0.3])
            for _ in range(subdiv - 1): a = a.subdivide()
            parts.append({'name': f'Antenna_{i}', 'V': a.vertices, 'F': a.faces})
        return parts

    BRDF = {'solarPanel': ('cook_torrance', {'albedo': 0.85, 'roughness': 0.15, 'metallic': 0.95}),
            'Antenna':    ('cook_torrance', {'albedo': 0.85, 'roughness': 0.15, 'metallic': 0.95}),
            'bus(MLI)':   ('oren_nayar',    {'albedo': 0.25, 'roughness': 0.5}),
            'default':    ('lambertian',    {'albedo': 0.3})}

    SUN = [0.4, 0.3, 0.866]
    OBS = [0.1, -0.2, 0.97]
    D = 550e3
    # Reference values from the CPU Embree implementation (analytic, no shadows)
    REF_NOSHADOW = 2.8839e-09

    for sub, lab in ((3, '14k'), (4, '55k')):
        parts = build(sub)
        ntri = sum(len(p['F']) for p in parts)
        for devname in ('cpu', 'auto'):
            r = GPURenderer(parts, brdf=BRDF, device=devname)
            if devname == 'auto' and r.dev.type == 'cpu':
                continue
            f = r.render(SUN, OBS, D, shadows=False, want_image=False)['flux']
            err = (f - REF_NOSHADOW) / REF_NOSHADOW * 100 if sub == 3 else float('nan')

            def bench(fn, n=20):
                fn()
                if r.dev.type != 'cpu':
                    torch.cuda.synchronize() if r.dev.type == 'cuda' else torch.mps.synchronize()
                ts = []
                for _ in range(n):
                    t0 = time.perf_counter(); fn()
                    if r.dev.type == 'cuda': torch.cuda.synchronize()
                    elif r.dev.type == 'mps': torch.mps.synchronize()
                    ts.append(time.perf_counter() - t0)
                return np.mean(ts) * 1e3

            t_f = bench(lambda: r.render(SUN, OBS, D, shadows=False, want_image=False))
            t_s = bench(lambda: r.render(SUN, OBS, D, shadows=True, want_image=False))
            t_i = bench(lambda: r.render(SUN, OBS, D, shadows=True, want_image=True, res=256), n=10)
            print(f"{lab:>4} tri ({ntri:>7,})  {str(r.dev):>5}  "
                  f"flux {t_f:7.2f} ms | +shadows {t_s:7.2f} ms | "
                  f"+image256 {t_i:7.2f} ms   vs CPU ref {err:+.3f}%")

    # ---- device agreement check: does your GPU match the CPU? ----------
    print("\n--- device agreement (run this before trusting MPS) ---")
    parts = build(3)
    rc = GPURenderer(parts, brdf=BRDF, device='cpu')
    rg = GPURenderer(parts, brdf=BRDF, device='auto')
    if rg.dev.type == 'cpu':
        print("  no GPU backend found -- torch.backends.mps.is_available() is False")
    else:
        worst = 0.0
        for pa in (0, 25, 50, 75):
            a = [np.radians(pa), None, None, None, None, None]
            fc = rc.render(SUN, OBS, D, angles=a, shadows=True, want_image=False)['flux']
            fg = rg.render(SUN, OBS, D, angles=a, shadows=True, want_image=False)['flux']
            rel = abs(fg - fc) / fc
            worst = max(worst, rel)
            print(f"  panel {pa:>3} deg   cpu {fc:.6e}   {rg.dev} {fg:.6e}   rel {rel:.2e}")
        print(f"  worst relative difference: {worst:.2e}"
              + ("   OK" if worst < 1e-4 else "   <-- INVESTIGATE"))

    # articulation check
    parts = build(3)
    r = GPURenderer(parts, brdf=BRDF, device='auto')
    print(f"\ndevice: {r.dev}")
    for pa in (0, 30, 60, 90):
        o = r.render(SUN, OBS, D, angles=[np.radians(pa), None, None, None, None, None],
                     res=256, shadows=True)
        ids = torch.unique(o['part_id']).tolist()
        print(f"panel {pa:>3} deg  flux {o['flux']:.4e}  "
              f"lit px {int((o['image'] > 0).sum()):>6}  parts visible {[i for i in ids if i >= 0]}")