"""Interactive, progressive viewer: fly a camera through the contrail.

    contrail-viewer                        # best available GPU (CUDA/Vulkan), else CPU
    contrail-viewer --arch cpu --scale 4   # CPU, internal resolution = window / 4

Controls
--------
  mouse drag (left)    look around
  W A S D              move forward / left / back / right
  E / Q                move up / down
  wheel  or  Z / X     speed x2 / /2
  arrows               sun elevation (up/down) and azimuth (left/right), 1 deg (x5 with shift)
  - / =                exposure -/+ 0.5 EV
  F / G                field of view -/+ 5 deg
  1 2 3 4              internal resolution scale (1 = full window resolution)
  O                    save the view (scene JSON) to views/
  P                    save a screenshot of the current preview to renders/
  H                    save the view and launch a high-resolution render in the background
                       (resolution/spp set by --hires-width/--hires-height/--hires-spp)
  ESC                  quit

The image refines progressively when the camera is still; any change restarts
the accumulation.  While moving, the image is rendered at half the internal
resolution to stay responsive.
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np
import taichi as ti

from . import app
from .renderer import auto_exposure, tonemap


class Viewer:
    def __init__(self, rd, scene, W, H, scale, hires):
        self.rd, self.sc = rd, scene
        self.W, self.H = W, H
        self.scale = scale
        self.hires = hires
        self.speed = 50.0  # m per frame
        self.exposure = None
        self.dirty = True
        self.moving = False
        self.last_mouse = None
        self.msg, self.msg_t = "", 0.0

    # ------------------------------------------------------------------
    def res(self):
        s = self.scale * (2 if self.moving else 1)
        return max(16, self.W // s), max(16, self.H // s)

    def notify(self, m):
        print(m)
        self.msg, self.msg_t = m, time.time()

    def save_view(self, prefix="view"):
        os.makedirs("views", exist_ok=True)
        path = os.path.join("views", time.strftime(f"{prefix}_%Y%m%d_%H%M%S.json"))
        if self.exposure is not None:
            self.sc.exposure = float(self.exposure)
            self.sc.auto_exposure = False
        self.sc.to_json(path)
        self.sc.auto_exposure = True
        return path

    def handle(self, gui):
        cam = self.sc.camera
        moved = False
        sun_changed = False
        for e in gui.get_events():
            if e.type == ti.GUI.PRESS:
                k = e.key
                step = 5.0 if gui.is_pressed(ti.GUI.SHIFT) else 1.0
                if k == ti.GUI.ESCAPE:
                    gui.running = False
                elif k == ti.GUI.UP:
                    self.sc.sun_elevation += step; sun_changed = True
                elif k == ti.GUI.DOWN:
                    self.sc.sun_elevation -= step; sun_changed = True
                elif k == ti.GUI.LEFT:
                    self.sc.sun_azimuth -= step; sun_changed = True
                elif k == ti.GUI.RIGHT:
                    self.sc.sun_azimuth += step; sun_changed = True
                elif k == "z":
                    self.speed *= 2
                elif k == "x":
                    self.speed /= 2
                elif k == "-":
                    self.sc.exposure_ev -= 0.5
                elif k == "=":
                    self.sc.exposure_ev += 0.5
                elif k == "f":
                    cam.fov = max(1.0, cam.fov - 5); moved = True
                elif k == "g":
                    cam.fov = min(150.0, cam.fov + 5); moved = True
                elif k in ("1", "2", "3", "4"):
                    self.scale = {"1": 1, "2": 2, "3": 3, "4": 4}[k]; moved = True
                elif k == "o":
                    self.notify(f"view saved: {self.save_view()}")
                elif k == "p":
                    self.screenshot()
                elif k == "h":
                    self.launch_hires()
            elif e.type == ti.GUI.MOTION and e.key == ti.GUI.WHEEL:
                self.speed *= 2.0 if e.delta[1] > 0 else 0.5
        # mouse look
        if gui.is_pressed(ti.GUI.LMB):
            mx, my = gui.get_cursor_pos()
            if self.last_mouse is not None:
                dx, dy = mx - self.last_mouse[0], my - self.last_mouse[1]
                if dx or dy:
                    cam.yaw -= dx * cam.fov * self.W / self.H
                    cam.pitch = float(np.clip(cam.pitch + dy * cam.fov, -89.9, 89.9))
                    moved = True
            self.last_mouse = (mx, my)
        else:
            self.last_mouse = None
        # keyboard motion
        f, r, u = cam.basis()
        v = np.zeros(3)
        for key, vec in (("w", f), ("s", -f), ("d", r), ("a", -r), ("e", [0, 0, 1]), ("q", [0, 0, -1])):
            if gui.is_pressed(key):
                v += vec
        if np.any(v):
            cam.position = (np.array(cam.position) + self.speed * v).tolist()
            moved = True
        if sun_changed:
            self.sc.sun_elevation = float(np.clip(self.sc.sun_elevation, -18, 90))
            self.rd.set_sun(self.sc.sun_elevation, self.sc.sun_azimuth)
            moved = True
        if moved != self.moving:
            self.dirty = True
        self.moving = moved
        if moved:
            self.dirty = True

    def frame(self):
        if self.dirty:
            self.rd.reset()
            self.dirty = False
        w, h = self.res()
        self.rd.render_pass(self.sc.camera, w, h, spp=1)
        xyz = self.rd.get_xyz(w, h)
        e = auto_exposure(xyz)
        if self.exposure is None or self.rd.spp_done <= 1:
            self.exposure = e if self.exposure is None else 0.7 * self.exposure + 0.3 * e
        img = tonemap(xyz, self.exposure * 2.0 ** self.sc.exposure_ev).astype(np.float32)
        img = np.repeat(np.repeat(img, self.W // w + 1, axis=0), self.H // h + 1, axis=1)
        return img[:self.W, :self.H], xyz

    def screenshot(self):
        os.makedirs("renders", exist_ok=True)
        from PIL import Image
        w, h = self.res()
        xyz = self.rd.get_xyz(w, h)
        img = tonemap(xyz, self.exposure * 2.0 ** self.sc.exposure_ev).transpose(1, 0, 2)[::-1]
        path = os.path.join("renders", time.strftime("preview_%Y%m%d_%H%M%S.png"))
        Image.fromarray((img * 255 + 0.5).astype(np.uint8)).save(path)
        self.notify(f"screenshot: {path} ({w}x{h}, {self.rd.spp_done} spp)")

    def launch_hires(self):
        path = self.save_view("hires")
        W, H, spp, arch = self.hires
        cmd = [sys.executable, "-m", "contrail_render.cli_render",
               "--scene", path, "--width", str(W), "--height", str(H), "--spp", str(spp), "--arch", arch,
               "--name", os.path.splitext(os.path.basename(path))[0]]
        log = open(os.path.splitext(path)[0] + ".log", "w")
        subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        self.notify(f"hi-res render launched ({W}x{H}, {spp} spp): log in {log.name}")

    def hud(self, gui, fps):
        sc, c = self.sc, self.sc.camera
        lines = [
            f"spp {self.rd.spp_done:5d}   {fps:4.1f} fps   internal {self.res()[0]}x{self.res()[1]}",
            f"cam ({c.position[0]:.0f}, {c.position[1]:.0f}, {c.position[2]:.0f}) m  yaw {c.yaw:.1f}  pitch {c.pitch:.1f}  fov {c.fov:.0f}",
            f"sun elev {sc.sun_elevation:.1f}  az {sc.sun_azimuth:.1f}   EV {sc.exposure_ev:+.1f}   speed {self.speed:.1f} m",
        ]
        if time.time() - self.msg_t < 4:
            lines.append(self.msg)
        for i, l in enumerate(lines):
            gui.text(l, pos=(0.01, 0.99 - 0.03 * i), font_size=16, color=0xFFFFFF)


def main():
    p = app.add_common_args(argparse.ArgumentParser(description=__doc__,
                                                    formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--scale", type=int, default=2, help="internal resolution = window / scale")
    p.add_argument("--hires-width", type=int, default=7680)
    p.add_argument("--hires-height", type=int, default=4320)
    p.add_argument("--hires-spp", type=int, default=512)
    p.add_argument("--hires-arch", default=None, help="back-end of the high-res render [same as --arch]")
    p.add_argument("--frames", type=int, default=0, help="quit after n frames (testing)")
    p.add_argument("--headless", action="store_true", help="no window (testing)")
    a = p.parse_args()

    sc = app.scene_from_args(a)
    app.init_taichi(a.arch)
    rd = app.load(sc, a.cache, a.width, a.height)
    v = Viewer(rd, sc, a.width, a.height, a.scale,
               (a.hires_width, a.hires_height, a.hires_spp, a.hires_arch or a.arch))
    gui = ti.GUI("Contrail viewer", res=(a.width, a.height), fast_gui=False, show_gui=not a.headless)
    t, n = time.time(), 0
    fps = 0.0
    while gui.running:
        v.handle(gui)
        img, _ = v.frame()
        gui.set_image(img)
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t, 1e-6) if n else 1.0 / max(now - t, 1e-6)
        t = now
        v.hud(gui, fps)
        gui.show()
        n += 1
        if a.frames and n >= a.frames:
            break


if __name__ == "__main__":
    main()
