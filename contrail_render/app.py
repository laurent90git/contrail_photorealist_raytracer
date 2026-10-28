"""Shared command-line plumbing for render.py and viewer.py."""
import argparse
import os

import taichi as ti

from . import scene as S, mie, cgns_reader

ARCHS = {"cpu": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan, "gpu": ti.gpu}


def default_cache_dir():
    return os.environ.get("CONTRAIL_CACHE",
                          os.path.join(os.path.expanduser("~"), ".cache", "contrail_render"))


def add_common_args(p: argparse.ArgumentParser):
    p.add_argument("--scene", help="scene JSON (camera, sun, physics); saved by the viewer")
    p.add_argument("--cgns", help="CGNS file (overrides the scene)")
    p.add_argument("--arch", default="gpu", choices=list(ARCHS),
                   help="Taichi back-end: cpu, cuda, vulkan or gpu (= best available) [gpu]")
    p.add_argument("--sun-elevation", type=float, help="deg; negative values = sun below the horizon")
    p.add_argument("--sun-azimuth", type=float, help="deg from +x (flight axis) towards +y")
    p.add_argument("--altitude", type=float, help="altitude of the CGNS origin [m]")
    p.add_argument("--sigma-g", type=float, help="log-normal geometric std-dev of the size distribution")
    p.add_argument("--crystal", choices=["rough", "sphere"],
                   help="phase function: Mie spheres, or Mie diffraction + rough crystals [rough]")
    p.add_argument("--albedo", type=float, help="override the Mie single-scattering albedo")
    p.add_argument("--tile-x", type=int, help="repeat the periodic field n times along x")
    p.add_argument("--taper", type=float, help="fade length [m] at both ends of the contrail")
    p.add_argument("--up-axis", choices=["x", "y", "z"])
    p.add_argument("--density-scale", type=float)
    p.add_argument("--aerosol-scale", type=float, help="turbidity factor of the aerosol layer")
    p.add_argument("--ground-albedo", type=float)
    p.add_argument("--camera", type=float, nargs=3, metavar=("X", "Y", "Z"), help="camera position [m]")
    p.add_argument("--yaw", type=float)
    p.add_argument("--pitch", type=float)
    p.add_argument("--fov", type=float)
    p.add_argument("--ev", type=float, help="exposure compensation [EV]")
    p.add_argument("--wavelengths", type=int, help="number of spectral bins (multiple of 4) [32]")
    p.add_argument("--cache", default=default_cache_dir(),
                   help="directory for the Mie tables cache [$CONTRAIL_CACHE or ~/.cache/contrail_render]")
    return p


def scene_from_args(a):
    sc = S.Scene.from_json(a.scene) if a.scene else S.Scene()
    m = {"cgns": "cgns", "sun_elevation": "sun_elevation", "sun_azimuth": "sun_azimuth",
         "altitude": "altitude", "crystal": "crystal", "sigma_g": "sigma_g", "albedo": "albedo_override",
         "tile_x": "tile_x", "taper": "taper", "up_axis": "up_axis", "density_scale": "density_scale",
         "aerosol_scale": "aerosol_scale", "ground_albedo": "ground_albedo", "ev": "exposure_ev",
         "wavelengths": "n_wavelengths"}
    for k, v in m.items():
        if getattr(a, k, None) is not None:
            setattr(sc, v, getattr(a, k))
    if a.camera is not None:
        sc.camera.position = list(a.camera)
    for k in ("yaw", "pitch", "fov"):
        if getattr(a, k) is not None:
            setattr(sc.camera, k, getattr(a, k))
    return sc


def init_taichi(arch_name):
    ti.init(arch=ARCHS[arch_name], default_fp=ti.f32, offline_cache=True,
            device_memory_fraction=0.6)
    print(f"[taichi] running on {ti.lang.impl.current_cfg().arch}")


def load(sc, cache, max_w, max_h):
    from .renderer import Renderer
    vol = cgns_reader.read_contrail(sc.cgns, zone=sc.zone or None, number_field=sc.number_field or None,
                                    radius_field=sc.radius_field or None)
    tab = mie.build_tables(K=sc.n_wavelengths, sigma_g=sc.sigma_g, cache_dir=cache)
    if sc.crystal == "rough":
        tab = mie.make_rough(tab, sc.rough_hg, sc.rough_blend)
    elif sc.crystal != "sphere":
        raise ValueError(f"unknown crystal model {sc.crystal!r}")
    return Renderer(sc, vol, tab, max_w, max_h)
