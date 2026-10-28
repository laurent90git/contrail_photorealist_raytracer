"""Precompute (and cache) the Mie tables, e.g. on a login node before a batch job.

  contrail-mie                      # default distribution width sigma_g = 1.3
  contrail-mie --sigma-g 1.1 1.3 1.6
"""
import argparse

from . import app, mie


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sigma-g", type=float, nargs="+", default=[1.3])
    p.add_argument("--wavelengths", type=int, default=32)
    p.add_argument("--processes", type=int, default=None, help="worker processes [all cores]")
    p.add_argument("--cache", default=app.default_cache_dir())
    a = p.parse_args()
    for sg in a.sigma_g:
        tab = mie.build_tables(K=a.wavelengths, sigma_g=sg, cache_dir=a.cache, processes=a.processes)
        print(f"[mie] sigma_g={sg}: ready in {a.cache} "
              f"(albedo >= {tab['albedo'].min():.6f}, g in [{tab['g'].min():.3f}, {tab['g'].max():.3f}])")


if __name__ == "__main__":
    main()
