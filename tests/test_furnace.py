"""White-furnace test: with albedo = 1 and a uniform environment of radiance 1
(no sun, no atmosphere), radiance must be exactly 1 everywhere, whatever the
contrail density -> validates delta/spectral tracking, phase sampling and the
spectral MIS weights."""
import sys
import numpy as np
import taichi as ti

sys.path.insert(0, ".")


def test_furnace(arch=ti.cpu):
    ti.init(arch=arch, random_seed=3)
    from contrail_render import scene as S, app
    sc = S.Scene(albedo_override=1.0, density_scale=5.0)  # 5x denser: tau ~ 13 laterally
    sc.camera = S.Camera(position=[1000.0, -1200.0, 0.0], yaw=90.0, pitch=0.0, fov=40.0)
    W, H = 64, 48
    rd = app.load(sc, app.default_cache_dir(), W, H)  # default crystal model ("rough" hybrid phase function)
    rd.furnace(1.0)
    for _ in range(16):
        rd.render_pass(sc.camera, W, H, spp=16)
    xyz = rd.get_xyz(W, H)
    ref = rd.cmf_np.sum(axis=0)
    ratio = xyz / ref
    print("furnace: mean", ratio.mean(axis=(0, 1)), "min", ratio.min(), "max", ratio.max())
    assert np.allclose(ratio.mean(axis=(0, 1)), 1.0, atol=0.01)


if __name__ == "__main__":
    test_furnace(getattr(ti, sys.argv[1]) if len(sys.argv) > 1 else ti.cpu)
