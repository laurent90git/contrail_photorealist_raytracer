"""Lorenz-Mie scattering tables for ice spheres.

For each wavelength bin and each *mean* radius r_mean of a log-normal number
size distribution, we tabulate
  - sigma_ext  : extinction cross-section per particle [m2]
  - albedo     : single-scattering albedo
  - phase      : phase function p(theta) [sr-1], piecewise constant in
                 cos(theta) over a non-uniform angular grid (very fine in the
                 forward diffraction peak and around the glory)
  - cdf        : cumulative distribution of the phase function, for sampling

The full Mie solution contains diffraction exactly (forward peak, corona
rings, glory), so no separate diffraction model is needed.
"""
import os
import hashlib
from multiprocessing import Pool

import numpy as np

from . import spectral


def angle_edges():
    """Bin edges in theta [rad], from 0 (forward) to pi (backward)."""
    deg = np.concatenate([
        np.arange(0.0, 5.0, 0.01),      # diffraction peak / coronae
        np.arange(5.0, 20.0, 0.05),
        np.arange(20.0, 160.0, 0.25),
        np.arange(160.0, 180.0, 0.05),  # glory
        [180.0]])
    return np.deg2rad(deg)


# Segments allowing O(1) bin lookup on the GPU: (theta_start_deg, step_deg, first_bin)
ANGLE_SEGMENTS = [(0.0, 0.01, 0), (5.0, 0.05, 500), (20.0, 0.25, 800), (160.0, 0.05, 1360)]


def mie_single(m, x, mu):
    """Bohren & Huffman BHMIE.  Returns Qext, Qsca, |S1|^2+|S2|^2 at cos angles mu."""
    nstop = int(x + 4.0 * x ** (1.0 / 3.0) + 2.0)
    y = m * x
    nmx = int(max(nstop, abs(y))) + 25 + int(1.5 * np.sqrt(abs(y)))
    D = np.zeros(nmx + 1, dtype=np.complex128)
    for n in range(nmx, 0, -1):
        D[n - 1] = n / y - 1.0 / (D[n] + n / y)
    # The upward recurrence of psi_n is unstable for large x; use the stable
    # psi_n = psi_{n-1} / (n/x + Dx_n) with Dx the log-derivative at the real x.
    Dx = np.zeros(nmx + 1)
    for n in range(nmx, 0, -1):
        Dx[n - 1] = n / x - 1.0 / (Dx[n] + n / x)
    psi0, psi1 = np.cos(x), np.sin(x)
    chi0, chi1 = -np.sin(x), np.cos(x)
    xi1 = psi1 - 1j * chi1
    pi0 = np.zeros_like(mu)
    pi1 = np.ones_like(mu)
    S1 = np.zeros(mu.shape, dtype=np.complex128)
    S2 = np.zeros(mu.shape, dtype=np.complex128)
    qext = 0.0
    qsca = 0.0
    for n in range(1, nstop + 1):
        fn = (2.0 * n + 1.0) / (n * (n + 1.0))
        psi = psi1 / (n / x + Dx[n])
        chi =(2.0 * n - 1.0) * chi1 / x - chi0
        xi = psi - 1j * chi
        da = D[n] / m + n / x
        db = D[n] * m + n / x
        an = (da * psi - psi1) / (da * xi - xi1)
        bn = (db * psi - psi1) / (db * xi - xi1)
        qext += (2 * n + 1) * (an.real + bn.real)
        qsca += (2 * n + 1) * (abs(an) ** 2 + abs(bn) ** 2)
        pi = pi1
        tau = n * mu * pi - (n + 1) * pi0
        S1 += fn * (an * pi + bn * tau)
        S2 += fn * (an * tau + bn * pi)
        psi0, psi1 = psi1, psi
        chi0, chi1 = chi1, chi
        xi1 = psi1 - 1j * chi1
        pi1 = ((2.0 * n + 1.0) * mu * pi - (n + 1.0) * pi0) / n
        pi0 = pi
    qext *= 2.0 / x ** 2
    qsca *= 2.0 / x ** 2
    return qext, qsca, np.abs(S1) ** 2 + np.abs(S2) ** 2


def _radius_grid():
    return np.exp(np.linspace(np.log(0.005e-6), np.log(150e-6), 1100))


def _lognormal_weights(r_grid, r_means, sigma_g):
    """Number weights w[i_mean, i_r] (sum to 1) of a log-normal distribution
    whose arithmetic mean radius is r_mean."""
    ls = np.log(sigma_g)
    r_med = r_means * np.exp(-0.5 * ls ** 2)
    lr = np.log(r_grid)
    dlr = np.gradient(lr)
    w = np.exp(-0.5 * ((lr[None, :] - np.log(r_med)[:, None]) / ls) ** 2) * dlr[None, :]
    w /= w.sum(axis=1, keepdims=True)
    w[w < 1e-9] = 0.0
    w /= w.sum(axis=1, keepdims=True)
    return w


def _one_wavelength(args):
    lam_nm, r_grid, needed, mu_eval = args
    lam = lam_nm * 1e-9
    m = complex(spectral.ice_refractive_index(lam_nm))
    k = 2 * np.pi / lam
    n_r = len(r_grid)
    cext = np.zeros(n_r)
    csca = np.zeros(n_r)
    s = np.zeros((n_r, len(mu_eval)))
    for i in np.nonzero(needed)[0]:
        x = k * r_grid[i]
        qe, qs, s12 = mie_single(m, x, mu_eval)
        geo = np.pi * r_grid[i] ** 2
        cext[i] = qe * geo
        csca[i] = qs * geo
        s[i] = s12 / (2.0 * k ** 2)  # differential scattering cross-section [m2/sr]
    return cext, csca, s


def build_tables(K=32, n_rmean=56, rmean_min=0.05e-6, rmean_max=20e-6, sigma_g=1.3,
                 cache_dir=None, processes=None, verbose=True):
    edges_l, lam = spectral.wavelength_bins(K)
    r_means = np.exp(np.linspace(np.log(rmean_min), np.log(rmean_max), n_rmean))
    th = angle_edges()
    key = f"mie_K{K}_R{n_rmean}_{rmean_min:.3e}_{rmean_max:.3e}_sg{sigma_g:.4f}_A{len(th)}_v1"
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, hashlib.md5(key.encode()).hexdigest()[:12] + ".npz")
        if os.path.exists(path):
            return dict(np.load(path))

    r_grid = _radius_grid()
    w = _lognormal_weights(r_grid, r_means, sigma_g)
    needed = (w > 0).any(axis=0)
    # Evaluate at edges and mid-points -> Simpson average inside each bin
    th_mid = 0.5 * (th[1:] + th[:-1])
    th_eval = np.empty(2 * len(th) - 1)
    th_eval[0::2] = th
    th_eval[1::2] = th_mid
    mu_eval = np.cos(th_eval)

    if verbose:
        print(f"[mie] computing {needed.sum()} radii x {K} wavelengths x {len(mu_eval)} angles "
              f"(sigma_g={sigma_g}) ... this is cached afterwards")
    jobs = [(l, r_grid, needed, mu_eval) for l in lam]
    with Pool(processes or os.cpu_count()) as pool:
        res = pool.map(_one_wavelength, jobs)

    n_a = len(th) - 1
    mu_edges = np.cos(th)
    d_omega = 2 * np.pi * (mu_edges[:-1] - mu_edges[1:])
    sig_ext = np.zeros((K, n_rmean))
    albedo = np.zeros((K, n_rmean))
    gasym = np.zeros((K, n_rmean))
    phase = np.zeros((K, n_rmean, n_a))
    for kk, (cext, csca, s) in enumerate(res):
        se = w @ cext
        ss = w @ csca
        dcs = w @ s  # [n_rmean, n_eval]
        p_edges = dcs[:, 0::2]
        p_mid = dcs[:, 1::2]
        p_bin = (p_edges[:, :-1] + 4 * p_mid + p_edges[:, 1:]) / 6.0
        norm = (p_bin * d_omega[None, :]).sum(axis=1, keepdims=True)
        p_bin /= norm
        sig_ext[kk] = se
        albedo[kk] = ss / se
        phase[kk] = p_bin
        gasym[kk] = (p_bin * d_omega * np.cos(th_mid)).sum(axis=1)
        if verbose:
            print(f"[mie]  lambda={lam[kk]:.1f} nm  quadrature/Qsca check={(norm[:, 0] / ss).min():.4f}"
                  f"..{(norm[:, 0] / ss).max():.4f}")
    cdf = np.concatenate([np.zeros((K, n_rmean, 1)), np.cumsum(phase * d_omega, axis=2)], axis=2)
    cdf /= cdf[:, :, -1:]
    out = dict(lam=lam, lam_edges=edges_l, r_means=r_means, theta_edges=th,
               sigma_ext=sig_ext, albedo=albedo, g=gasym, phase=phase, cdf=cdf,
               sigma_g=np.array(sigma_g))
    if cache_dir:
        np.savez_compressed(path, **out)
    return out


# ---------------------------------------------------------------------------
# Hybrid "rough crystal" phase function
# ---------------------------------------------------------------------------
# Ice spheres scatter far too little sideways (p(90 deg) ~ 0.003 sr-1) compared
# with real, non-spherical and roughened ice crystals (~0.02-0.04 sr-1).  For
# large size parameters the scattering splits into diffraction, which depends
# only on the projected area and is thus nearly shape-independent, and a
# geometric-optics part, which depends on the shape.  The hybrid keeps the Mie
# solution in the diffraction region (theta < blend[0]: coronae, iridescence)
# and replaces it beyond blend[1] by a smooth two-term Henyey-Greenstein shape
#     q = w HG(g1) + (1 - w) HG(-g2)
# carrying exactly the same scattered energy (smoothstep blend in between).
# Default parameters were fitted to give a total asymmetry factor g ~ 0.74
# and p(90) ~ 0.023 sr-1 for r = 4.5 um, i.e. the values reported for
# severely roughened ice crystals in the visible (Yang et al. 2013, Baum et
# al. 2014).  This is a parametric approximation, not a ray-tracing solution.
ROUGH_HG_DEFAULT = (0.40, 0.90, 0.996)  # g1, g2, w
ROUGH_BLEND_DEFAULT = (10.0, 20.0)      # deg


def _hg(g, c):
    return (1 - g * g) / (4 * np.pi * (1 + g * g - 2 * g * c) ** 1.5)


def make_rough(tab, hg=ROUGH_HG_DEFAULT, blend=ROUGH_BLEND_DEFAULT):
    """Return a copy of the Mie tables with the hybrid phase function."""
    g1, g2, w = hg
    th_e = tab["theta_edges"]
    th = 0.5 * (th_e[1:] + th_e[:-1])
    mu = np.cos(th_e)
    d_omega = 2 * np.pi * (mu[:-1] - mu[1:])
    s = np.clip((np.rad2deg(th) - blend[0]) / (blend[1] - blend[0]), 0, 1)
    s = s * s * (3 - 2 * s)
    q = w * _hg(g1, np.cos(th)) + (1 - w) * _hg(-g2, np.cos(th))
    p = tab["phase"]
    E = (s * p * d_omega).sum(axis=2, keepdims=True)  # energy redistributed
    Q = (s * q * d_omega).sum()
    ph = (1 - s) * p + s * q * (E / Q)
    ph /= (ph * d_omega).sum(axis=2, keepdims=True)
    cdf = np.concatenate([np.zeros(ph.shape[:2] + (1,)), np.cumsum(ph * d_omega, axis=2)], axis=2)
    cdf /= cdf[:, :, -1:]
    out = dict(tab)
    out.update(phase=ph, cdf=cdf, g=(ph * d_omega * np.cos(th)).sum(axis=2))
    return out
