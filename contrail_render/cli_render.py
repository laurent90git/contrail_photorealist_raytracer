"""Offline high-resolution rendering of a contrail.

Examples
--------
  contrail-render --scene examples/day_ground.json --width 7680 --height 4320 --spp 1024
  contrail-render --arch cpu --width 960 --height 540 --spp 64 --sun-elevation -2

Outputs (in --out, default renders/):
  <name>.png         8-bit sRGB, tone-mapped
  <name>_16bit.tif   16-bit sRGB, tone-mapped
  <name>_xyz.tif     32-bit float linear CIE XYZ radiance (for post-processing)
  <name>.json        the scene used (re-render with --scene)
"""
import argparse
import os
import sys
import time

import numpy as np

from . import app


def main():
    p = app.add_common_args(argparse.ArgumentParser(description=__doc__,
                                                    formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--width", type=int, default=3840)
    p.add_argument("--height", type=int, default=2160)
    p.add_argument("--spp", type=int, default=256, help="samples per pixel (each carries 4 wavelengths)")
    p.add_argument("--tile", type=int, default=1024, help="tile size in pixels (memory bound)")
    p.add_argument("--batch", type=int, default=8, help="samples per kernel launch")
    p.add_argument("--out", default="renders")
    p.add_argument("--name", default=None)
    a = p.parse_args()

    sc = app.scene_from_args(a)
    app.init_taichi(a.arch)
    W, H = a.width, a.height
    tw, th = min(a.tile, W), min(a.tile, H)
    rd = app.load(sc, a.cache, tw, th)
    rd.bg_max_passes = 64

    xyz = np.zeros((W, H, 3), np.float32)
    tiles = [(x0, y0) for y0 in range(0, H, th) for x0 in range(0, W, tw)]
    t_start = time.time()
    for it, (x0, y0) in enumerate(tiles):
        w, h = min(tw, W - x0), min(th, H - y0)
        rd.reset()
        done = 0
        while done < a.spp:
            n = min(a.batch, a.spp - done)
            rd.render_pass(sc.camera, W, H, n, x0, y0, w, h)
            done += n
        xyz[x0:x0 + w, y0:y0 + h] = rd.get_xyz(w, h)
        el = time.time() - t_start
        eta = el / (it + 1) * (len(tiles) - it - 1)
        print(f"\r[render] tile {it + 1}/{len(tiles)}  elapsed {el:7.1f}s  eta {eta:7.1f}s", end="")
        sys.stdout.flush()
    print()

    from .renderer import auto_exposure, tonemap
    exposure = sc.exposure if (not sc.auto_exposure and sc.exposure > 0) else auto_exposure(xyz)
    exposure *= 2.0 ** sc.exposure_ev
    img = tonemap(xyz, exposure).transpose(1, 0, 2)[::-1]  # [y, x], top row first

    os.makedirs(a.out, exist_ok=True)
    name = a.name or time.strftime("contrail_%Y%m%d_%H%M%S")
    base = os.path.join(a.out, name)
    from PIL import Image
    import tifffile
    Image.fromarray((img * 255 + 0.5).astype(np.uint8)).save(base + ".png")
    tifffile.imwrite(base + "_16bit.tif", (img * 65535 + 0.5).astype(np.uint16), compression="zlib")
    tifffile.imwrite(base + "_xyz.tif", xyz.transpose(1, 0, 2)[::-1].astype(np.float32), compression="zlib")
    sc.exposure, sc.auto_exposure = float(exposure / 2.0 ** sc.exposure_ev), False
    sc.to_json(base + ".json")
    print(f"[render] {W}x{H}, {a.spp} spp in {time.time() - t_start:.1f}s -> {base}.png / _16bit.tif / _xyz.tif")


if __name__ == "__main__":
    main()
