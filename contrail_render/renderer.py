"""Spectral Monte Carlo renderer for contrails in a physically based sky.

One code base, two back-ends: the kernels are written with Taichi and compile
either for the CPU (``ti.init(arch=ti.cpu)``) or for a CUDA/Vulkan GPU.

Physics
-------
Contrail (participating medium, data from CGNS):
  beta_ext(x, lambda) = N(x) * sigma_ext(r_mean(x), lambda)          [m-1]
  with sigma_ext from Lorenz-Mie theory averaged over a log-normal
  size distribution; single-scattering albedo of ice (~1) or user value;
  tabulated Mie phase function (diffraction peak, coronae, glory).
  Light transport: unbiased volumetric path tracing with delta tracking for
  free paths, ratio tracking for shadow rays, next-event estimation of the
  sun and the sky as environment light (Novak et al. 2018, "Monte Carlo
  methods for volumetric light transport simulation").

Atmosphere (Bruneton & Neyret 2008/2017, Hillaire 2020):
  Rayleigh scattering (lambda^-4), aerosol Mie scattering (Cornette-Shanks),
  ozone absorption (Chappuis band, important at dusk), spherical Earth with
  its shadow, multiple scattering via Hillaire's LUT, Lambertian ground.
  The sky illuminates the contrail and is also seen behind it; aerial
  perspective is applied between the camera and the contrail.

Every path carries one wavelength bin; spectra are converted to CIE XYZ, then
to linear sRGB.
"""
import math

import numpy as np
import taichi as ti

from . import spectral, mie
from .mie import ANGLE_SEGMENTS

R_GROUND = 6360e3
R_TOP = 6420e3
RAYLEIGH_H = 8000.0
MIE_H = 1200.0
MIE_SCA = 3.996e-6
MIE_EXT = 4.40e-6
MIE_G = 0.8
SUN_ANG_RADIUS = math.radians(0.2667)
LIMB_U = 0.6

TR_W, TR_H = 256, 64     # transmittance LUT (mu, r)
MS_N = 32                # multiple scattering LUT
IRR_N = 64               # ground irradiance LUT (mu_s)
SKY_A, SKY_E = 256, 160  # environment (sky-view) LUT (azimuth, elevation)

PI = math.pi


@ti.data_oriented
class Renderer:
    def __init__(self, scene, volume, mie_tab, max_width=1920, max_height=1080):
        self.scene = scene
        K = int(mie_tab["sigma_ext"].shape[0])
        assert K % 4 == 0, "the number of wavelength bins must be a multiple of 4"
        self.K = K
        sp = scene

        # ------------------------------------------------------------------
        # contrail volume
        N, r = volume["N"], volume["r"]
        origin, spacing = volume["origin"].copy(), volume["spacing"].copy()
        perm = {"z": (0, 1, 2), "y": (2, 0, 1), "x": (1, 2, 0)}[sp.up_axis]
        N, r = N.transpose(perm), r.transpose(perm)
        origin, spacing = origin[list(perm)], spacing[list(perm)]
        N = N * sp.density_scale
        self.shape = N.shape
        self.origin = origin
        self.spacing = spacing
        self.period = spacing[0] * (N.shape[0] - 1)
        self.tile = max(1, int(sp.tile_x))
        ext = spacing * (np.array(N.shape) - 1)
        self.box_min = origin.copy()
        self.box_max = origin + ext
        self.box_max[0] = origin[0] + self.period * self.tile

        self.vol = ti.Vector.field(2, ti.f32, shape=N.shape)   # (N, N*r_um)
        self.vol.from_numpy(np.stack([N, N * r * 1e6], axis=-1).astype(np.float32))

        # ------------------------------------------------------------------
        # Mie tables
        rm = mie_tab["r_means"] * 1e6
        self.NR = len(rm)
        self.lnr0 = float(np.log(rm[0]))
        self.dlnr = float(np.log(rm[1]) - np.log(rm[0]))
        th = mie_tab["theta_edges"]
        self.NA = len(th) - 1
        self.sig_ext = ti.field(ti.f32, shape=(K, self.NR))      # um^2 -> stored in m^2
        self.albedo = ti.field(ti.f32, shape=(K, self.NR))
        self.phase = ti.field(ti.f32, shape=(K, self.NR, self.NA))
        self.cdf = ti.field(ti.f32, shape=(K, self.NR, self.NA + 1))
        self.versine = ti.field(ti.f32, shape=self.NA + 1)  # 1 - cos(theta) at bin edges
        self.sig_ext.from_numpy(mie_tab["sigma_ext"].astype(np.float32))
        alb = mie_tab["albedo"] if sp.albedo_override < 0 else np.full_like(mie_tab["albedo"], sp.albedo_override)
        self.albedo.from_numpy(np.clip(alb, 0, 1).astype(np.float32))
        self.phase.from_numpy(mie_tab["phase"].astype(np.float32))
        self.cdf.from_numpy(mie_tab["cdf"].astype(np.float32))
        self.versine.from_numpy((2 * np.sin(th / 2) ** 2).astype(np.float32))

        # delta-M truncation of the diffraction peak (used after the first
        # scattering event only, see scattered_radiance4): p is flattened below
        # theta_c, the removed fraction f is treated as unscattered light.
        ph = mie_tab["phase"]
        mu_e = np.cos(th)
        d_omega = 2 * np.pi * (mu_e[:-1] - mu_e[1:])
        jc = int(np.searchsorted(th, np.deg2rad(sp.delta_m_angle)))
        pt = ph.copy()
        if jc > 0:
            pt[:, :, :jc] = np.minimum(ph[:, :, :jc], ph[:, :, jc:jc + 1])
        f = np.clip(1.0 - (pt * d_omega).sum(axis=2), 0.0, 0.99)
        pt /= (pt * d_omega).sum(axis=2, keepdims=True)
        cdf_t = np.concatenate([np.zeros(pt.shape[:2] + (1,)), np.cumsum(pt * d_omega, axis=2)], axis=2)
        cdf_t /= cdf_t[:, :, -1:]
        self.trunc_fraction = f
        self.sig_ext_t = ti.field(ti.f32, shape=(K, self.NR))
        self.albedo_t = ti.field(ti.f32, shape=(K, self.NR))
        self.phase_t = ti.field(ti.f32, shape=(K, self.NR, self.NA))
        self.cdf_t = ti.field(ti.f32, shape=(K, self.NR, self.NA + 1))
        alb = np.clip(alb, 0, 1)
        self.sig_ext_t.from_numpy((mie_tab["sigma_ext"] * (1 - f * alb)).astype(np.float32))
        self.albedo_t.from_numpy((alb * (1 - f) / (1 - f * alb)).astype(np.float32))
        self.phase_t.from_numpy(pt.astype(np.float32))
        self.cdf_t.from_numpy(cdf_t.astype(np.float32))

        # majorant per wavelength (also valid for the truncated medium): max over cells of (max corner N) * sigma(max corner r)
        sig_cummax = np.maximum.accumulate(mie_tab["sigma_ext"], axis=1)
        nmax = _corner_max(N)
        with np.errstate(invalid="ignore", divide="ignore"):
            rr = np.where(N > 0, r, 0.0) * 1e6
        rmax = _corner_max(rr)
        fr = np.clip((np.log(np.maximum(rmax, 1e-6)) - self.lnr0) / self.dlnr, 0, self.NR - 1)
        ir = np.minimum(np.ceil(fr).astype(int), self.NR - 1)
        maj = np.array([(nmax * sig_cummax[k][ir]).max() for k in range(K)]) * 1.001 + 1e-12
        self.majorant = ti.field(ti.f32, shape=K)
        self.majorant.from_numpy(maj.astype(np.float32))
        self.majorant_np = maj

        # ------------------------------------------------------------------
        # spectral constants
        lam = mie_tab["lam"]
        self.lam = lam
        cmf = spectral.bin_cmf_weights(mie_tab["lam_edges"])
        self.cmf = ti.Vector.field(3, ti.f32, shape=K)
        self.cmf.from_numpy(cmf.astype(np.float32))
        self.cmf_np = cmf
        self.sun_E = ti.field(ti.f32, shape=K)
        self.ray_sca = ti.field(ti.f32, shape=K)
        self.mie_sca = ti.field(ti.f32, shape=K)
        self.mie_ext = ti.field(ti.f32, shape=K)
        self.oz_abs = ti.field(ti.f32, shape=K)
        self.sun_E.from_numpy(spectral.solar_irradiance(lam).astype(np.float32))
        self.ray_sca.from_numpy(spectral.rayleigh_scattering(lam).astype(np.float32))
        self.mie_sca.from_numpy(np.full(K, MIE_SCA * sp.aerosol_scale, np.float32))
        self.mie_ext.from_numpy(np.full(K, MIE_EXT * sp.aerosol_scale, np.float32))
        self.oz_abs.from_numpy(spectral.ozone_absorption(lam, sp.ozone_du).astype(np.float32))

        # ------------------------------------------------------------------
        # atmosphere LUTs
        self.trans_lut = ti.field(ti.f32, shape=(K, TR_W, TR_H))
        self.ms_lut = ti.field(ti.f32, shape=(K, MS_N, MS_N))
        self.irr_lut = ti.field(ti.f32, shape=(K, IRR_N))
        self.sky_lut = ti.field(ti.f32, shape=(K, SKY_A, SKY_E))

        # scalar parameters
        self.sun_dir = ti.Vector.field(3, ti.f32, shape=())
        self.par = ti.field(ti.f32, shape=16)
        self.ipar = ti.field(ti.i32, shape=8)
        self.box = ti.Vector.field(3, ti.f32, shape=4)  # box_min, box_max, origin, spacing
        self.box[0] = self.box_min.tolist()
        self.box[1] = self.box_max.tolist()
        self.box[2] = self.origin.tolist()
        self.box[3] = self.spacing.tolist()

        # image accumulation (XYZ) and display buffer
        self.max_w, self.max_h = max_width, max_height
        self.accum = ti.Vector.field(3, ti.f32, shape=(max_width, max_height))
        self.bg_accum = ti.Vector.field(3, ti.f32, shape=(max_width, max_height))
        self.spp_done = 0
        self.pass_count = 0
        self.bg_count = 0
        self.bg_max_passes = 16  # the clear sky is smooth: a few jittered passes suffice

        self.update_params()
        self._build_transmittance()
        self._build_multiscattering()
        self._build_irradiance()
        self.set_sun(sp.sun_elevation, sp.sun_azimuth)

    # ======================================================================
    # parameters
    def update_params(self):
        sp = self.scene
        p = self.par
        p[0] = sp.altitude
        p[1] = sp.ground_albedo
        p[2] = float(self.period)
        p[3] = float(sp.taper)
        p[4] = float(self.lnr0)
        p[5] = float(self.dlnr)
        p[6] = 0.0  # furnace environment radiance (test mode)
        self.ipar[0] = self.tile
        self.ipar[1] = int(sp.max_bounces)
        self.ipar[2] = 0  # furnace test mode
        self.ipar[3] = self.NR
        self.ipar[4] = self.NA

    def set_sun(self, elevation, azimuth):
        self.scene.sun_elevation, self.scene.sun_azimuth = float(elevation), float(azimuth)
        self.sun_dir[None] = self.scene.sun_direction().tolist()
        cx = 0.5 * (self.box_min + self.box_max)
        self._build_sky_lut(ti.math.vec3(*cx))
        self.reset()

    # ======================================================================
    # generic helpers
    @ti.func
    def planet_pos(self, p):
        # local frame -> planet-centred frame
        return ti.math.vec3(p.x, p.y, p.z + self.par[0] + R_GROUND)

    @ti.func
    def ray_sphere(self, r, mu, R):
        """Distances along a ray (origin radius r, cos zenith mu) to a sphere of
        radius R.  Returns (t_near, t_far, hit)."""
        b = r * mu
        c = (r - R) * (r + R)
        disc = b * b - c
        t0 = -1.0
        t1 = -1.0
        hit = 0
        if disc >= 0.0:
            s = ti.sqrt(disc)
            t0 = -b - s
            t1 = -b + s
            hit = 1
        return t0, t1, hit

    @ti.func
    def densities(self, h):
        dr = ti.exp(-h / RAYLEIGH_H)
        dm = ti.exp(-h / MIE_H)
        do = ti.max(0.0, 1.0 - ti.abs(h - 25000.0) / 15000.0)
        return dr, dm, do

    @ti.func
    def extinction_atm(self, k, h):
        dr, dm, do = self.densities(h)
        return self.ray_sca[k] * dr + self.mie_ext[k] * dm + self.oz_abs[k] * do

    # ---------------------------------------------------------------- LUTs
    @ti.func
    def tr_uv_to_rmu(self, u, v):
        H = ti.sqrt(R_TOP * R_TOP - R_GROUND * R_GROUND)
        rho = H * v
        r = ti.sqrt(rho * rho + R_GROUND * R_GROUND)
        d_min = R_TOP - r
        d_max = rho + H
        d = d_min + u * (d_max - d_min)
        mu = 1.0
        if d > 0.0:
            mu = (H * H - rho * rho - d * d) / (2.0 * r * d)
        return r, ti.math.clamp(mu, -1.0, 1.0)

    @ti.func
    def transmittance_to_top(self, k, r, mu):
        H = ti.sqrt(R_TOP * R_TOP - R_GROUND * R_GROUND)
        r = ti.math.clamp(r, R_GROUND, R_TOP)
        rho = ti.sqrt(ti.max(r * r - R_GROUND * R_GROUND, 0.0))
        disc = r * r * (mu * mu - 1.0) + R_TOP * R_TOP
        d = ti.max(-r * mu + ti.sqrt(ti.max(disc, 0.0)), 0.0)
        d_min = R_TOP - r
        d_max = rho + H
        u = ti.math.clamp((d - d_min) / ti.max(d_max - d_min, 1e-3), 0.0, 1.0)
        v = rho / H
        x = u * (TR_W - 1)
        y = v * (TR_H - 1)
        i0 = ti.min(int(x), TR_W - 2)
        j0 = ti.min(int(y), TR_H - 2)
        fx = x - i0
        fy = y - j0
        a = ti.math.mix(self.trans_lut[k, i0, j0], self.trans_lut[k, i0 + 1, j0], fx)
        b = ti.math.mix(self.trans_lut[k, i0, j0 + 1], self.trans_lut[k, i0 + 1, j0 + 1], fx)
        return ti.math.mix(a, b, fy)

    @ti.func
    def sun_transmittance(self, k, r, mu_s):
        """Transmittance towards the sun, including the Earth shadow."""
        sin_h = R_GROUND / r
        cos_h = -ti.sqrt(ti.max(1.0 - sin_h * sin_h, 0.0))
        vis = ti.math.smoothstep(-sin_h * SUN_ANG_RADIUS, sin_h * SUN_ANG_RADIUS, mu_s - cos_h)
        return self.transmittance_to_top(k, r, mu_s) * vis

    @ti.func
    def ms_lookup(self, k, r, mu_s):
        u = ti.math.clamp(mu_s * 0.5 + 0.5, 0.0, 1.0) * (MS_N - 1)
        v = ti.math.clamp((r - R_GROUND) / (R_TOP - R_GROUND), 0.0, 1.0) * (MS_N - 1)
        i0 = ti.min(int(u), MS_N - 2)
        j0 = ti.min(int(v), MS_N - 2)
        fx = u - i0
        fy = v - j0
        a = ti.math.mix(self.ms_lut[k, i0, j0], self.ms_lut[k, i0 + 1, j0], fx)
        b = ti.math.mix(self.ms_lut[k, i0, j0 + 1], self.ms_lut[k, i0 + 1, j0 + 1], fx)
        return ti.math.mix(a, b, fy)

    @ti.func
    def irr_lookup(self, k, mu_s):
        u = ti.math.clamp(mu_s * 0.5 + 0.5, 0.0, 1.0) * (IRR_N - 1)
        i0 = ti.min(int(u), IRR_N - 2)
        return ti.math.mix(self.irr_lut[k, i0], self.irr_lut[k, i0 + 1], u - i0)

    @ti.func
    def phase_rayleigh(self, nu):
        return 3.0 / (16.0 * PI) * (1.0 + nu * nu)

    @ti.func
    def phase_cs(self, nu):
        g = MIE_G
        g2 = g * g
        return 3.0 / (8.0 * PI) * (1.0 - g2) * (1.0 + nu * nu) / (
            (2.0 + g2) * ti.pow(ti.max(1.0 + g2 - 2.0 * g * nu, 1e-6), 1.5))

    @ti.kernel
    def _build_transmittance(self):
        for k, i, j in self.trans_lut:
            r, mu = self.tr_uv_to_rmu(i / (TR_W - 1.0), j / (TR_H - 1.0))
            _, t1, _ = self.ray_sphere(r, mu, R_TOP)
            n = 500
            dt = ti.max(t1, 0.0) / n
            tau = 0.0
            for s in range(n):
                t = (s + 0.5) * dt
                rr = ti.sqrt(r * r + 2.0 * r * mu * t + t * t)
                tau += self.extinction_atm(k, rr - R_GROUND) * dt
            self.trans_lut[k, i, j] = ti.exp(-tau)

    @ti.kernel
    def _build_multiscattering(self):
        # Hillaire 2020: isotropic 2nd-order luminance and transfer factor
        for k, i, j in self.ms_lut:
            mu_s = i / (MS_N - 1.0) * 2.0 - 1.0
            r = R_GROUND + (j + 0.5) / MS_N * (R_TOP - R_GROUND)
            nd = 16
            L2 = 0.0
            fms = 0.0
            for a, b in ti.ndrange(nd, nd):
                cth = 1.0 - 2.0 * (a + 0.5) / nd
                phi = 2.0 * PI * (b + 0.5) / nd
                sth = ti.sqrt(ti.max(1.0 - cth * cth, 0.0))
                # local frame: up = z, sun in the xz plane
                d = ti.math.vec3(sth * ti.cos(phi), sth * ti.sin(phi), cth)
                sdir = ti.math.vec3(ti.sqrt(ti.max(1.0 - mu_s * mu_s, 0.0)), 0.0, mu_s)
                P = ti.math.vec3(0.0, 0.0, r)
                _, tt, _ = self.ray_sphere(r, cth, R_TOP)
                tg0, _, hg = self.ray_sphere(r, cth, R_GROUND)
                tend = tt
                ground = 0
                if hg == 1 and tg0 > 0.0:
                    tend = tg0
                    ground = 1
                ns = 30
                dt = tend / ns
                T = 1.0
                Lp = 0.0
                Lf = 0.0
                for s in range(ns):
                    t = (s + 0.5) * dt
                    Q = P + t * d
                    rq = Q.norm()
                    dr, dm, do = self.densities(rq - R_GROUND)
                    sig_s = self.ray_sca[k] * dr + self.mie_sca[k] * dm
                    sig_t = self.ray_sca[k] * dr + self.mie_ext[k] * dm + self.oz_abs[k] * do
                    mus = Q.dot(sdir) / rq
                    Ts = self.sun_transmittance(k, rq, mus)
                    st = ti.exp(-sig_t * dt)
                    integ = (1.0 - st) / ti.max(sig_t, 1e-12)
                    Lp += T * sig_s * Ts / (4.0 * PI) * integ
                    Lf += T * sig_s * integ
                    T *= st
                if ground == 1:
                    Q = P + tend * d
                    rq = Q.norm()
                    mus = Q.dot(sdir) / rq
                    Lp += T * self.sun_transmittance(k, rq, mus) * ti.max(mus, 0.0) * self.par[1] / PI
                L2 += Lp / (nd * nd)
                fms += Lf / (nd * nd)
            self.ms_lut[k, i, j] = L2 / ti.max(1.0 - fms, 1e-3)

    # ---------------------------------------------------------------- 4-lane helpers
    # A path carries 4 wavelength bins kv = (k0, k0+K/4, k0+K/2, k0+3K/4)
    # ("hero wavelength" sampling, Wilkie et al. 2014).
    @ti.func
    def gather(self, f: ti.template(), kv):
        out = ti.math.vec4(0.0)
        for l in ti.static(range(4)):
            out[l] = f[kv[l]]
        return out

    @ti.func
    def sun_transmittance4(self, kv, r, mu_s):
        out = ti.math.vec4(0.0)
        for l in ti.static(range(4)):
            out[l] = self.sun_transmittance(kv[l], r, mu_s)
        return out

    @ti.func
    def ms_lookup4(self, kv, r, mu_s):
        out = ti.math.vec4(0.0)
        for l in ti.static(range(4)):
            out[l] = self.ms_lookup(kv[l], r, mu_s)
        return out

    @ti.func
    def atmo_march4(self, kv, p_local, d, tmax, nsteps: ti.template(), with_ground: ti.template()):
        """Atmospheric in-scattering (single scattering + multiple-scattering LUT)
        and transmittance along a ray, from p_local up to distance tmax (or the
        top of the atmosphere / the ground).  Returns (S, T) as vec4 over lanes."""
        sdir = self.sun_dir[None]
        P = self.planet_pos(p_local)
        r = P.norm()
        mu = P.dot(d) / r
        S = ti.math.vec4(0.0)
        T = ti.math.vec4(1.0)
        t_in = 0.0
        ok = 1
        if r > R_TOP:
            t0, _, hit = self.ray_sphere(r, mu, R_TOP)
            if hit == 0 or t0 < 0.0:
                ok = 0
            else:
                t_in = t0
        if ok == 1:
            _, t_top, _ = self.ray_sphere(r, mu, R_TOP)
            tg0, _, hg = self.ray_sphere(r, mu, R_GROUND)
            t_end = t_top
            hit_ground = 0
            if hg == 1 and tg0 > 0.0:
                t_end = tg0
                hit_ground = 1
            if tmax < t_end:
                t_end = tmax
                hit_ground = 0
            nu = d.dot(sdir)
            pr = self.phase_rayleigh(nu)
            pm = self.phase_cs(nu)
            E = self.gather(self.sun_E, kv)
            cr = self.gather(self.ray_sca, kv)
            cms = self.gather(self.mie_sca, kv)
            cme = self.gather(self.mie_ext, kv)
            co = self.gather(self.oz_abs, kv)
            dt = ti.max(t_end - t_in, 0.0) / nsteps
            for s in range(nsteps):
                t = t_in + (s + 0.5) * dt
                Q = P + t * d
                rq = Q.norm()
                dr, dm, do = self.densities(rq - R_GROUND)
                srs = cr * dr
                sms = cms * dm
                sig_t = srs + cme * dm + co * do
                mus = Q.dot(sdir) / rq
                Ts = self.sun_transmittance4(kv, rq, mus)
                scat = (srs * pr + sms * pm) * Ts + (srs + sms) * self.ms_lookup4(kv, rq, mus)
                st = ti.exp(-sig_t * dt)
                S += T * scat * E * (1.0 - st) / ti.max(sig_t, 1e-12)
                T *= st
            if ti.static(with_ground):
                if hit_ground == 1:
                    Q = P + t_end * d
                    rq = Q.norm()
                    mus = Q.dot(sdir) / rq
                    irr = ti.math.vec4(0.0)
                    for l in ti.static(range(4)):
                        irr[l] = self.irr_lookup(kv[l], mus)
                    irr += self.sun_transmittance4(kv, rq, mus) * ti.max(mus, 0.0)
                    S += T * self.par[1] / PI * irr * E
                    T = ti.math.vec4(0.0)
        return S, T

    @ti.kernel
    def _build_irradiance(self):
        # sky irradiance on the ground (per unit solar irradiance), vs. sun zenith cosine
        for k, i in self.irr_lut:
            mu_s = i / (IRR_N - 1.0) * 2.0 - 1.0
            sdir = ti.math.vec3(ti.sqrt(ti.max(1.0 - mu_s * mu_s, 0.0)), 0.0, mu_s)
            E = 0.0
            na = 8
            nb = 16
            for a, b in ti.ndrange(na, nb):
                cth = (a + 0.5) / na
                phi = 2.0 * PI * (b + 0.5) / nb
                sth = ti.sqrt(1.0 - cth * cth)
                d = ti.math.vec3(sth * ti.cos(phi), sth * ti.sin(phi), cth)
                r = R_GROUND + 1.0
                P = ti.math.vec3(0.0, 0.0, r)
                _, t_top, _ = self.ray_sphere(r, cth, R_TOP)
                nu = d.dot(sdir)
                pr = self.phase_rayleigh(nu)
                pm = self.phase_cs(nu)
                ns = 24
                dt = t_top / ns
                T = 1.0
                S = 0.0
                for s in range(ns):
                    t = (s + 0.5) * dt
                    Q = P + t * d
                    rq = Q.norm()
                    dr, dm, do = self.densities(rq - R_GROUND)
                    srs = self.ray_sca[k] * dr
                    sms = self.mie_sca[k] * dm
                    sig_t = srs + self.mie_ext[k] * dm + self.oz_abs[k] * do
                    mus = Q.dot(sdir) / rq
                    Ts = self.sun_transmittance(k, rq, mus)
                    scat = (srs * pr + sms * pm) * Ts + (srs + sms) * self.ms_lookup(k, rq, mus)
                    st = ti.exp(-sig_t * dt)
                    S += T * scat * (1.0 - st) / ti.max(sig_t, 1e-12)
                    T *= st
                # cosine-weighted hemispherical integral (uniform in cos theta)
                E += S * cth * (2.0 * PI / (na * nb))
            self.irr_lut[k, i] = E

    @ti.func
    def lanes(self, k0):
        q = self.K // 4
        return ti.math.ivec4(k0, k0 + q, k0 + 2 * q, k0 + 3 * q)

    @ti.func
    def sky_dir_to_uv(self, d):
        az = ti.atan2(d.y, d.x)
        if az < 0.0:
            az += 2.0 * PI
        el = ti.asin(ti.math.clamp(d.z, -1.0, 1.0))
        v = 0.5 + 0.5 * ti.math.sign(el) * ti.sqrt(ti.abs(el) / (0.5 * PI))
        return az / (2.0 * PI), v

    @ti.kernel
    def _build_sky_lut(self, center: ti.math.vec3):
        # environment radiance seen from the contrail (sun disk excluded)
        for g, i, j in ti.ndrange(self.K // 4, SKY_A, SKY_E):
            az = (i + 0.5) / SKY_A * 2.0 * PI
            v = j / (SKY_E - 1.0)
            x = 2.0 * v - 1.0
            el = ti.math.sign(x) * x * x * 0.5 * PI
            d = ti.math.vec3(ti.cos(el) * ti.cos(az), ti.cos(el) * ti.sin(az), ti.sin(el))
            kv = self.lanes(g)
            S, _ = self.atmo_march4(kv, center, d, 1e30, 48, True)
            for l in ti.static(range(4)):
                self.sky_lut[kv[l], i, j] = S[l]

    @ti.func
    def sky_env4(self, kv, d):
        u, v = self.sky_dir_to_uv(d)
        x = u * SKY_A - 0.5
        y = v * (SKY_E - 1)
        i0 = int(ti.floor(x))
        fx = x - i0
        i1 = (i0 + 1) % SKY_A
        i0 = (i0 + SKY_A) % SKY_A
        j0 = ti.min(int(y), SKY_E - 2)
        fy = y - j0
        L = ti.math.vec4(0.0)
        for l in ti.static(range(4)):
            k = kv[l]
            a = ti.math.mix(self.sky_lut[k, i0, j0], self.sky_lut[k, i1, j0], fx)
            b = ti.math.mix(self.sky_lut[k, i0, j0 + 1], self.sky_lut[k, i1, j0 + 1], fx)
            L[l] = ti.math.mix(a, b, fy)
        if self.ipar[2] == 1:
            L = ti.math.vec4(self.par[6])
        return L

    # ======================================================================
    # contrail medium
    @ti.func
    def box_hit(self, o, d):
        bmin = self.box[0]
        bmax = self.box[1]
        dd = ti.select(ti.abs(d) > 1e-9, d, ti.select(d >= 0.0, 1e-9, -1e-9))
        inv = 1.0 / dd
        ta = (bmin - o) * inv
        tb = (bmax - o) * inv
        tmin = ti.min(ta, tb)
        tmax = ti.max(ta, tb)
        t0 = ti.max(ti.max(tmin.x, tmin.y), ti.max(tmin.z, 0.0))
        t1 = ti.min(ti.min(tmax.x, tmax.y), tmax.z)
        return t0, t1

    @ti.func
    def medium_at(self, p):
        """Returns (number density [#/m3], mean radius [um]) at p (trilinear)."""
        org = self.box[2]
        h = self.box[3]
        xs = p.x - org.x
        period = self.par[2]
        total = period * self.ipar[0]
        n_out = 0.0
        r_out = 1.0
        if xs >= 0.0 and xs <= total:
            fade = 1.0
            taper = self.par[3]
            if taper > 0.0:
                fade = ti.math.smoothstep(0.0, taper, xs) * ti.math.smoothstep(0.0, taper, total - xs)
            xl = xs - period * ti.floor(xs / period)
            gx = xl / h.x
            gy = (p.y - org.y) / h.y
            gz = (p.z - org.z) / h.z
            nx, ny, nz = self.vol.shape
            if gy >= 0.0 and gz >= 0.0 and gy <= ny - 1 and gz <= nz - 1:
                i = ti.min(int(gx), nx - 2)
                j = ti.min(int(gy), ny - 2)
                kk = ti.min(int(gz), nz - 2)
                fx = gx - i
                fy = gy - j
                fz = gz - kk
                c00 = ti.math.mix(self.vol[i, j, kk], self.vol[i + 1, j, kk], fx)
                c10 = ti.math.mix(self.vol[i, j + 1, kk], self.vol[i + 1, j + 1, kk], fx)
                c01 = ti.math.mix(self.vol[i, j, kk + 1], self.vol[i + 1, j, kk + 1], fx)
                c11 = ti.math.mix(self.vol[i, j + 1, kk + 1], self.vol[i + 1, j + 1, kk + 1], fx)
                c0 = ti.math.mix(c00, c10, fy)
                c1 = ti.math.mix(c01, c11, fy)
                c = ti.math.mix(c0, c1, fz)
                n_out = c[0] * fade
                if c[0] > 1e-20:
                    r_out = c[1] / c[0]
        return n_out, r_out

    @ti.func
    def radius_index(self, r_um):
        fr = (ti.log(ti.max(r_um, 1e-6)) - self.par[4]) / self.par[5]
        return ti.math.clamp(fr, 0.0, self.ipar[3] - 1.001)

    @ti.func
    def beta4(self, kv, p, trunc: ti.template()):
        """Extinction coefficient [m-1] for the 4 lanes (delta-M scaled if trunc)."""
        n, r = self.medium_at(p)
        b = ti.math.vec4(0.0)
        if n > 0.0:
            fr = self.radius_index(r)
            i = int(fr)
            f = fr - i
            for l in ti.static(range(4)):
                if ti.static(trunc):
                    b[l] = n * ti.math.mix(self.sig_ext_t[kv[l], i], self.sig_ext_t[kv[l], i + 1], f)
                else:
                    b[l] = n * ti.math.mix(self.sig_ext[kv[l], i], self.sig_ext[kv[l], i + 1], f)
        return b

    @ti.func
    def majorant4(self, kv):
        m = 0.0
        for l in ti.static(range(4)):
            m = ti.max(m, self.majorant[kv[l]])
        return m

    @ti.func
    def spectral_track(self, kv, o, d, t0, t1, trunc: ti.template()):
        """Spectral delta tracking (Kutz et al. 2017) with a majorant common to
        the 4 lanes and collision probability = mean of the lane ratios.
        Returns (hit, t, weights)."""
        maj = self.majorant4(kv)
        t = t0
        hit = 0
        w = ti.math.vec4(1.0)
        if t1 > t0:
            while True:
                t -= ti.log(1.0 - ti.random(ti.f32)) / maj
                if t >= t1:
                    break
                ratio = self.beta4(kv, o + t * d, trunc) / maj
                pc = 0.25 * (ratio[0] + ratio[1] + ratio[2] + ratio[3])
                if pc > 0.0:
                    if ti.random(ti.f32) < pc:
                        w *= ratio / pc
                        hit = 1
                        break
                    w *= (1.0 - ratio) / (1.0 - pc)
        return hit, t, w

    @ti.func
    def ratio_track4(self, kv, o, d, trunc: ti.template()):
        t0, t1 = self.box_hit(o, d)
        maj = self.majorant4(kv)
        T = ti.math.vec4(1.0)
        t = t0
        if t1 > t0:
            while True:
                t -= ti.log(1.0 - ti.random(ti.f32)) / maj
                if t >= t1:
                    break
                T *= 1.0 - self.beta4(kv, o + t * d, trunc) / maj
                if ti.max(ti.max(T[0], T[1]), ti.max(T[2], T[3])) < 0.05:  # Russian roulette
                    if ti.random(ti.f32) < 0.5:
                        T = ti.math.vec4(0.0)
                        break
                    T *= 2.0
        return T

    # ---------------------------------------------------------------- phase
    @ti.func
    def angle_bin(self, theta):
        deg = theta * (180.0 / PI)
        j = 0
        if deg < ANGLE_SEGMENTS[1][0]:
            j = int((deg - ANGLE_SEGMENTS[0][0]) / ANGLE_SEGMENTS[0][1]) + ANGLE_SEGMENTS[0][2]
        elif deg < ANGLE_SEGMENTS[2][0]:
            j = int((deg - ANGLE_SEGMENTS[1][0]) / ANGLE_SEGMENTS[1][1]) + ANGLE_SEGMENTS[1][2]
        elif deg < ANGLE_SEGMENTS[3][0]:
            j = int((deg - ANGLE_SEGMENTS[2][0]) / ANGLE_SEGMENTS[2][1]) + ANGLE_SEGMENTS[2][2]
        else:
            j = int((deg - ANGLE_SEGMENTS[3][0]) / ANGLE_SEGMENTS[3][1]) + ANGLE_SEGMENTS[3][2]
        return ti.math.clamp(j, 0, self.ipar[4] - 1)

    @ti.func
    def angle_between(self, a, b):
        # accurate for tiny angles (unlike acos of the dot product in float32)
        return ti.atan2(a.cross(b).norm(), a.dot(b))

    @ti.func
    def phase4(self, kv, ir, j, trunc: ti.template()):
        out = ti.math.vec4(0.0)
        for l in ti.static(range(4)):
            if ti.static(trunc):
                out[l] = self.phase_t[kv[l], ir, j]
            else:
                out[l] = self.phase[kv[l], ir, j]
        return out

    @ti.func
    def albedo4(self, kv, ir, trunc: ti.template()):
        out = ti.math.vec4(0.0)
        for l in ti.static(range(4)):
            if ti.static(trunc):
                out[l] = self.albedo_t[kv[l], ir]
            else:
                out[l] = self.albedo[kv[l], ir]
        return out

    @ti.func
    def onb(self, n):
        sign = 1.0 if n.z >= 0.0 else -1.0
        a = -1.0 / (sign + n.z)
        b = n.x * n.y * a
        t = ti.math.vec3(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x)
        bb = ti.math.vec3(b, sign + n.y * n.y * a, -n.y)
        return t, bb

    @ti.func
    def dir_from_versine(self, d, v):
        """Direction at angle theta = acos(1-v) from d, uniform azimuth.
        The versine keeps float32 precision for angles of a few 0.01 deg."""
        theta = 2.0 * ti.asin(ti.math.clamp(ti.sqrt(ti.max(v, 0.0) * 0.5), 0.0, 1.0))
        phi = 2.0 * PI * ti.random(ti.f32)
        t, b = self.onb(d)
        st = ti.sin(theta)
        return (ti.cos(theta) * d + st * (ti.cos(phi) * t + ti.sin(phi) * b)).normalized()

    @ti.func
    def cdf_at(self, k, ir, j, trunc: ti.template()):
        v = 0.0
        if ti.static(trunc):
            v = self.cdf_t[k, ir, j]
        else:
            v = self.cdf[k, ir, j]
        return v

    @ti.func
    def phase_sample(self, k, ir, d, trunc: ti.template()):
        """Sample the tabulated Mie phase function; returns (direction, angle bin)."""
        u = ti.random(ti.f32)
        lo = 0
        hi = self.ipar[4]
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.cdf_at(k, ir, mid, trunc) <= u:
                lo = mid
            else:
                hi = mid
        c0 = self.cdf_at(k, ir, lo, trunc)
        c1 = self.cdf_at(k, ir, lo + 1, trunc)
        f = ti.math.clamp((u - c0) / ti.max(c1 - c0, 1e-12), 0.0, 1.0)
        v = ti.math.mix(self.versine[lo], self.versine[lo + 1], f)
        return self.dir_from_versine(d, v), lo

    @ti.func
    def pick_radius_bin(self, r_um):
        fr = self.radius_index(r_um)
        i = int(fr)
        if ti.random(ti.f32) < fr - i:
            i += 1
        return i

    # ---------------------------------------------------------------- transport
    @ti.func
    def sun_nee4(self, kv, p, wo, ir, trunc: ti.template()):
        """Direct sunlight (next-event estimation) scattered at p, for a path
        travelling along wo.  The phase function evaluated at the exact
        scattering angle produces the diffraction corona around the sun."""
        L = ti.math.vec4(0.0)
        if self.ipar[2] == 0:
            sdir = self.sun_dir[None]
            v_cone = 2.0 * ti.sin(0.5 * SUN_ANG_RADIUS) ** 2
            ws = self.dir_from_versine(sdir, ti.random(ti.f32) * v_cone)
            P = self.planet_pos(p)
            rq = P.norm()
            Ta = self.sun_transmittance4(kv, rq, P.dot(ws) / rq)
            if ti.max(ti.max(Ta[0], Ta[1]), ti.max(Ta[2], Ta[3])) > 1e-7:
                j = self.angle_bin(self.angle_between(wo, ws))
                L = self.phase4(kv, ir, j, trunc) * self.gather(self.sun_E, kv) * Ta * self.ratio_track4(kv, p, ws, True)
        return L

    @ti.func
    def scattered_radiance4(self, kv, p0, d0):
        """Radiance arriving from a real collision at p0 for a path travelling
        along d0 (multiple scattering, albedo included)."""
        p = p0
        d = d0
        L = ti.math.vec4(0.0)
        thr = ti.math.vec4(1.0)
        max_b = self.ipar[1]
        for depth in range(max_b):
            n, r = self.medium_at(p)
            ir = self.pick_radius_bin(r)
            # one-sample spectral MIS over the 4 lanes' phase functions
            c = ti.min(int(ti.random(ti.f32) * 4.0), 3)
            dn = ti.math.vec3(0.0)
            j = 0
            pv = ti.math.vec4(0.0)
            if depth == 0:
                # first event: full Mie phase function (coronae, iridescence, glory)
                thr *= self.albedo4(kv, ir, False)
                L += thr * self.sun_nee4(kv, p, d, ir, False)
                dn, j = self.phase_sample(kv[c], ir, d, False)
                pv = self.phase4(kv, ir, j, False)
            else:
                # multiple scattering: delta-M truncated medium (low variance)
                thr *= self.albedo4(kv, ir, True)
                L += thr * self.sun_nee4(kv, p, d, ir, True)
                dn, j = self.phase_sample(kv[c], ir, d, True)
                pv = self.phase4(kv, ir, j, True)
            pm = 0.25 * (pv[0] + pv[1] + pv[2] + pv[3])
            thr *= pv / ti.max(pm, 1e-30)
            d = dn
            t0, t1 = self.box_hit(p, d)
            hit, t, w = self.spectral_track(kv, p, d, t0, t1, True)
            thr *= w
            if hit == 0:
                L += thr * self.sky_env4(kv, d)
                break
            p = p + t * d
            if depth > 128:
                if ti.random(ti.f32) > 0.9:
                    break
                thr /= 0.9
        return L

    @ti.func
    def sun_disk4(self, kv, d):
        sdir = self.sun_dir[None]
        th = self.angle_between(d, sdir)
        L = ti.math.vec4(0.0)
        if th < SUN_ANG_RADIUS:
            mu = ti.sqrt(ti.max(1.0 - (th / SUN_ANG_RADIUS) ** 2, 0.0))
            L = self.gather(self.sun_E, kv) / (PI * SUN_ANG_RADIUS ** 2 * (1.0 - LIMB_U / 3.0)) * (
                1.0 - LIMB_U * (1.0 - mu))
        return L

    @ti.func
    def background_parts4(self, kv, o, d):
        """Clear-sky radiance without contrail: (sky + ground, attenuated sun disk)."""
        L = ti.math.vec4(self.par[6])
        Ls = ti.math.vec4(0.0)
        if self.ipar[2] == 0:
            S, T = self.atmo_march4(kv, o, d, 1e30, 32, True)
            L = S
            Ls = T * self.sun_disk4(kv, d)
        return L, Ls

    @ti.func
    def background4(self, kv, o, d):
        L, Ls = self.background_parts4(kv, o, d)
        return L + Ls

    @ti.func
    def contrail_delta4(self, kv, o, d):
        """Monte Carlo estimate of (radiance with contrail) - (clear-sky radiance).
        The clear sky is integrated deterministically for all wavelengths in
        _render_background, so this term is zero for rays missing the box and
        the sky is noise-free (control variate)."""
        out = ti.math.vec4(0.0)
        t0, t1 = self.box_hit(o, d)
        if t1 > t0:
            hit, t, w = self.spectral_track(kv, o, d, t0, t1, False)
            bg, sun = self.background_parts4(kv, o, d)
            if hit == 1:
                Lc = self.scattered_radiance4(kv, o + t * d, d)
                if self.ipar[2] == 0:
                    S, T = self.atmo_march4(kv, o, d, t, 24, False)
                    out = w * (S + T * Lc) - bg
                else:
                    out = w * Lc - bg
            else:
                out = (w - 1.0) * bg
            # The sun disk (~1e7 times brighter than the sky) is kept out of the
            # control variate: its attenuation by the contrail is estimated by
            # ratio tracking, which is smooth and never negative.
            if sun[0] + sun[1] + sun[2] + sun[3] > 0.0:
                out += (self.ratio_track4(kv, o, d, False) - 1.0) * sun
        return out

    @ti.func
    def hash(self, x: ti.u32):
        x ^= x >> 16
        x *= ti.u32(0x7feb352d)
        x ^= x >> 15
        x *= ti.u32(0x846ca68b)
        x ^= x >> 16
        return x

    @ti.func
    def camera_ray(self, px, py, W, H, cam_f, cam_r, cam_u, tan_half):
        sx = (px + ti.random(ti.f32)) / W * 2.0 - 1.0
        sy = (py + ti.random(ti.f32)) / H * 2.0 - 1.0
        return (cam_f + sx * tan_half * (W / H) * cam_r + sy * tan_half * cam_u).normalized()

    @ti.func
    def lanes_to_xyz(self, kv, L):
        acc = ti.math.vec3(0.0)
        for l in ti.static(range(4)):
            v = L[l]
            if not (v > -1e12 and v < 1e12):  # NaN / inf guard
                v = 0.0
            acc += v * self.cmf[kv[l]]
        return acc

    @ti.kernel
    def _render(self, x0: ti.i32, y0: ti.i32, w: ti.i32, h: ti.i32, W: ti.i32, H: ti.i32,
                spp: ti.i32, pass_index: ti.i32,
                cam_o: ti.math.vec3, cam_f: ti.math.vec3, cam_r: ti.math.vec3, cam_u: ti.math.vec3,
                tan_half: ti.f32):
        q = self.K // 4
        for i, j in ti.ndrange(w, h):
            px = x0 + i
            py = y0 + j
            pix_hash = self.hash(ti.u32(px * 7919 + py * 104729 + 1))
            acc = ti.math.vec3(0.0)
            for s in range(spp):
                k0 = int((ti.u32(pass_index * spp + s) + pix_hash) % ti.u32(q))
                kv = self.lanes(k0)
                d = self.camera_ray(px, py, W, H, cam_f, cam_r, cam_u, tan_half)
                acc += self.lanes_to_xyz(kv, self.contrail_delta4(kv, cam_o, d)) * q
            self.accum[i, j] += acc

    @ti.kernel
    def _render_background(self, x0: ti.i32, y0: ti.i32, w: ti.i32, h: ti.i32, W: ti.i32, H: ti.i32,
                           cam_o: ti.math.vec3, cam_f: ti.math.vec3, cam_r: ti.math.vec3,
                           cam_u: ti.math.vec3, tan_half: ti.f32):
        for i, j in ti.ndrange(w, h):
            d = self.camera_ray(x0 + i, y0 + j, W, H, cam_f, cam_r, cam_u, tan_half)
            acc = ti.math.vec3(0.0)
            for g in range(self.K // 4):
                kv = self.lanes(g)
                acc += self.lanes_to_xyz(kv, self.background4(kv, cam_o, d))
            self.bg_accum[i, j] += acc

    # ======================================================================
    # Python API
    def render_pass(self, camera, W, H, spp=1, x0=0, y0=0, w=None, h=None):
        """Accumulate `spp` samples per pixel (plus one clear-sky sample during the
        first bg_max_passes passes) for the
        tile [x0,x0+w)x[y0,y0+h) of a WxH image.  Call reset() between tiles
        or after a camera move."""
        w = W if w is None else w
        h = H if h is None else h
        assert w <= self.max_w and h <= self.max_h
        f, r, u = camera.basis()
        args = (ti.math.vec3(*camera.position), ti.math.vec3(*f), ti.math.vec3(*r),
                ti.math.vec3(*u), math.tan(math.radians(camera.fov) / 2))
        if self.bg_count < self.bg_max_passes:
            self._render_background(x0, y0, w, h, W, H, *args)
            self.bg_count += 1
        self._render(x0, y0, w, h, W, H, spp, self.pass_count, *args)
        self.spp_done += spp
        self.pass_count += 1

    def get_xyz(self, w, h):
        """Mean radiance per pixel, CIE XYZ of spectral radiance [W m-2 sr-1 nm-1]
        integrated against the CMFs; arrays indexed [x, y] (y up)."""
        bg = self.bg_accum.to_numpy()[:w, :h] / max(self.bg_count, 1)
        c = self.accum.to_numpy()[:w, :h] / max(self.spp_done, 1)
        return bg + c

    def reset(self):
        self.accum.fill(0)
        self.bg_accum.fill(0)
        self.spp_done = 0
        self.pass_count = 0
        self.bg_count = 0

    def furnace(self, env=1.0):
        """White-furnace test: constant environment radiance, no sun nor atmosphere."""
        self.ipar[2] = 1
        self.par[6] = env
        self.reset()


def _corner_max(a):
    m = a[:-1, :-1, :-1]
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                m = np.maximum(m, a[di:a.shape[0] - 1 + di, dj:a.shape[1] - 1 + dj, dk:a.shape[2] - 1 + dk])
    return m


# --------------------------------------------------------------------------
# post-processing
def xyz_to_srgb_linear(xyz):
    return xyz @ spectral.XYZ_TO_SRGB.T


def luminance(xyz):
    return spectral.LUMINOUS_EFFICACY * xyz[..., 1]  # cd/m2


def auto_exposure(xyz, key=0.22):
    Y = luminance(xyz)
    Y = Y[np.isfinite(Y)]
    if Y.size == 0:
        return 1.0
    hi = np.percentile(Y, 99.0)  # ignore the sun disk
    Y = np.clip(Y, 1e-3, hi)
    return key / np.exp(np.mean(np.log(Y)))


def aces_film(x):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


def srgb_oetf(x):
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 0), 1 / 2.4) - 0.055)


def tonemap(xyz, exposure):
    """XYZ (radiometric) -> display sRGB in [0,1].  exposure multiplies luminance in cd/m2."""
    rgb = xyz_to_srgb_linear(xyz) * spectral.LUMINOUS_EFFICACY * exposure
    rgb = np.maximum(rgb, 0.0)
    return srgb_oetf(aces_film(rgb))
