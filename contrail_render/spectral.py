"""Spectral data: wavelength bins, CIE colour matching, solar spectrum,
atmospheric constituents and refractive index of ice.

All the renderer works on K discrete wavelength bins; a Monte Carlo path
carries a single bin (chosen stochastically), which makes dispersion effects
(coronae, iridescence, sky colours) come out naturally.
"""
import numpy as np

LAMBDA_MIN = 380.0  # nm
LAMBDA_MAX = 780.0  # nm


def wavelength_bins(K=32):
    edges = np.linspace(LAMBDA_MIN, LAMBDA_MAX, K + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    return edges, centers


# --- CIE 1931 2deg colour matching functions --------------------------------
# Multi-lobe analytic fit of Wyman, Sloan & Shirley (JCGT 2013).
def _g(lam, mu, s1, s2):
    s = np.where(lam < mu, s1, s2)
    return np.exp(-0.5 * ((lam - mu) / s) ** 2)


def cie_xyz(lam):
    lam = np.asarray(lam, dtype=np.float64)
    x = (1.056 * _g(lam, 599.8, 37.9, 31.0) + 0.362 * _g(lam, 442.0, 16.0, 26.7)
         - 0.065 * _g(lam, 501.1, 20.4, 26.2))
    y = 0.821 * _g(lam, 568.8, 46.9, 40.5) + 0.286 * _g(lam, 530.9, 16.3, 31.1)
    z = 1.217 * _g(lam, 437.0, 11.8, 36.0) + 0.681 * _g(lam, 459.0, 26.0, 13.8)
    return np.stack([x, y, z], axis=-1)


def bin_cmf_weights(edges, n_sub=50):
    """Integral of the CMFs over each bin [nm].  XYZ = sum_k L_k * W_k."""
    W = np.zeros((len(edges) - 1, 3))
    for k in range(len(edges) - 1):
        lam = np.linspace(edges[k], edges[k + 1], n_sub)
        W[k] = np.trapezoid(cie_xyz(lam), lam, axis=0)
    return W


# XYZ -> linear sRGB (D65)
XYZ_TO_SRGB = np.array([[3.2404542, -1.5371385, -0.4985314],
                        [-0.9692660, 1.8760108, 0.0415560],
                        [0.0556434, -0.2040259, 1.0572252]])

LUMINOUS_EFFICACY = 683.0  # lm/W, converts W/(m2 sr) weighted by ybar to cd/m2

# --- Solar spectrum and ozone ------------------------------------------------
# Values tabulated every 10 nm from 360 to 830 nm, as used by Bruneton's
# "Precomputed Atmospheric Scattering" reference implementation (2017).
_TAB_LAMBDA = np.arange(360.0, 831.0, 10.0)
# Extra-terrestrial solar spectral irradiance [W m-2 nm-1]
_SOLAR = np.array([
    1.11776, 1.14259, 1.01249, 1.14716, 1.72765, 1.73054, 1.6887, 1.61253,
    1.91198, 2.03474, 2.02042, 2.02212, 1.93377, 1.95809, 1.91686, 1.8298,
    1.8685, 1.8931, 1.85149, 1.8504, 1.8341, 1.8345, 1.8147, 1.78158, 1.7533,
    1.6965, 1.68194, 1.64654, 1.6048, 1.52143, 1.55622, 1.5113, 1.474, 1.4482,
    1.41018, 1.36775, 1.34188, 1.31429, 1.28303, 1.26758, 1.2367, 1.2082,
    1.18737, 1.14683, 1.12362, 1.1058, 1.07124, 1.04992])
# Ozone absorption cross section [m2/molecule]
_OZONE = np.array([
    1.18e-27, 2.182e-28, 2.818e-28, 6.636e-28, 1.527e-27, 2.763e-27, 5.52e-27,
    8.451e-27, 1.582e-26, 2.316e-26, 3.669e-26, 4.924e-26, 7.752e-26, 9.016e-26,
    1.48e-25, 1.602e-25, 2.139e-25, 2.755e-25, 3.091e-25, 3.5e-25, 4.266e-25,
    4.672e-25, 4.398e-25, 4.701e-25, 5.019e-25, 4.305e-25, 3.74e-25, 3.215e-25,
    2.662e-25, 2.238e-25, 1.852e-25, 1.473e-25, 1.209e-25, 9.423e-26, 7.455e-26,
    6.566e-26, 5.105e-26, 4.15e-26, 4.228e-26, 3.237e-26, 2.451e-26, 2.801e-26,
    2.534e-26, 1.624e-26, 1.465e-26, 2.078e-26, 1.383e-26, 7.105e-30])
DOBSON_UNIT = 2.687e20  # molecules / m2


def solar_irradiance(lam):
    return np.interp(lam, _TAB_LAMBDA, _SOLAR)


def ozone_absorption(lam, total_column_du=300.0, layer_width=15000.0):
    """Peak ozone absorption coefficient [m-1] (tent profile centred at 25 km)."""
    n_max = total_column_du * DOBSON_UNIT / layer_width
    return n_max * np.interp(lam, _TAB_LAMBDA, _OZONE)


def rayleigh_scattering(lam):
    """Sea-level Rayleigh scattering coefficient [m-1]."""
    lam_um = np.asarray(lam) * 1e-3
    return 1.24062e-6 / lam_um ** 4


# --- Ice refractive index ----------------------------------------------------
# Real part: Cauchy fit of Warren & Brandt (2008) over the visible.
# Imaginary part: log-interpolated rough values of the same compilation.
_ICE_K_LAMBDA = np.array([380, 400, 450, 500, 550, 600, 650, 700, 750, 800.0])
_ICE_K = np.array([3e-11, 2.4e-11, 1e-11, 4e-10, 2.9e-9, 1.2e-8, 2.0e-8,
                   3.0e-8, 1.0e-7, 1.3e-7])


def ice_refractive_index(lam):
    lam_um = np.asarray(lam) * 1e-3
    n = 1.3008 + 0.00297 / lam_um ** 2
    k = np.exp(np.interp(lam, _ICE_K_LAMBDA, np.log(_ICE_K)))
    return n + 1j * k
