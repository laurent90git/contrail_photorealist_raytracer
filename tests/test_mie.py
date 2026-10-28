"""Validate the in-house BHMIE implementation against miepython, and the
hybrid rough-crystal phase function (normalisation, untouched diffraction peak)."""
import numpy as np
import miepython
from contrail_render import mie


def test_against_miepython():
    th = mie.angle_edges()
    print("n angle edges", len(th), "(expect 1761)")
    assert len(th) == 1761
    mu = np.cos(np.linspace(0, np.pi, 181))
    for m, x in [(1.31 + 1e-8j, 0.5), (1.31 + 1e-8j, 10.0), (1.31 + 1e-9j, 80.0), (1.31 + 0j, 600.0), (1.31 + 1e-9j, 1500.0)]:
        qe, qs, s12 = mie.mie_single(m, x, mu)
        qe_ref, qs_ref, _, _ = miepython.efficiencies_mx(m, x)
        s1, s2 = miepython.S1_S2(m, x, mu, norm='wiscombe')
        ref = np.abs(s1) ** 2 + np.abs(s2) ** 2
        err = np.max(np.abs(s12 - ref) / ref)
        print(f"m={m} x={x}: Qext {qe:.6f} vs {qe_ref:.6f}, Qsca {qs:.6f} vs {qs_ref:.6f}, max rel err S {err:.2e}")
        assert abs(qe - qe_ref) < 1e-7 * max(1, qe_ref)
        assert err < 1e-5



def test_tables_and_rough_hybrid():
    from contrail_render import app
    tab = mie.build_tables(sigma_g=1.3, cache_dir=app.default_cache_dir(), verbose=False)
    th = tab["theta_edges"]
    d_omega = 2 * np.pi * (np.cos(th[:-1]) - np.cos(th[1:]))
    assert np.allclose((tab["phase"] * d_omega).sum(axis=2), 1.0, atol=1e-6)
    assert tab["albedo"].min() > 0.9999
    rough = mie.make_rough(tab)
    assert np.allclose((rough["phase"] * d_omega).sum(axis=2), 1.0, atol=1e-6)
    j10 = np.searchsorted(th, np.deg2rad(10.0)) - 1
    assert np.allclose(rough["phase"][:, :, :j10], tab["phase"][:, :, :j10], rtol=1e-6)  # diffraction kept
    j90 = np.searchsorted(th, np.deg2rad(90.0))
    ir = np.argmin(abs(tab["r_means"] - 4.5e-6))
    print("p(90 deg) sphere %.4f -> rough %.4f, g %.3f -> %.3f" % (
        tab["phase"][14, ir, j90], rough["phase"][14, ir, j90], tab["g"][14, ir], rough["g"][14, ir]))
    assert rough["phase"][14, ir, j90] > 5 * tab["phase"][14, ir, j90]
    assert 0.70 < rough["g"][14, ir] < 0.78


if __name__ == "__main__":
    test_against_miepython()
    test_tables_and_rough_hybrid()
