"""Scene parameters (physics, sun, camera), saved/loaded as JSON so that a view
explored in the interactive viewer can be re-rendered at high resolution."""
import json
import os
from dataclasses import dataclass, field, asdict

import numpy as np


@dataclass
class Camera:
    # position in the local frame of the CGNS domain [m]; z is up (see Scene.up_axis)
    position: list = field(default_factory=lambda: [-600.0, -1500.0, -250.0])
    yaw: float = 50.0      # deg, from +x towards +y
    pitch: float = 8.0     # deg, positive = looking up
    fov: float = 60.0      # vertical field of view, deg

    def basis(self):
        y, p = np.deg2rad(self.yaw), np.deg2rad(self.pitch)
        f = np.array([np.cos(p) * np.cos(y), np.cos(p) * np.sin(y), np.sin(p)])
        r = np.cross(f, [0.0, 0.0, 1.0])
        if np.linalg.norm(r) < 1e-6:
            r = np.array([np.sin(y), -np.cos(y), 0.0])
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        return f, r, u


@dataclass
class Scene:
    # --- contrail data ---
    cgns: str = "ressources/contrail.cgns"
    zone: str = ""
    number_field: str = ""
    radius_field: str = ""
    up_axis: str = "z"          # which CGNS axis is vertical (x, y or z)
    density_scale: float = 1.0  # multiplies the number density (sensitivity studies)
    tile_x: int = 1             # repeat the (periodic) field along the flight axis
    taper: float = 0.0          # [m] smooth fade-in/out at both ends of the tiled contrail
    # --- microphysics / optics ---
    sigma_g: float = 1.3        # geometric std-dev of the log-normal size distribution
    albedo_override: float = -1.0  # <0: use Mie single-scattering albedo of ice (~0.99999)
    n_wavelengths: int = 32
    crystal: str = "rough"      # "sphere": pure Mie; "rough": Mie diffraction + rough-crystal
                                # side/back scattering (see mie.make_rough)
    rough_hg: list = field(default_factory=lambda: [0.40, 0.90, 0.996])  # g1, g2, w
    rough_blend: list = field(default_factory=lambda: [10.0, 20.0])      # deg
    delta_m_angle: float = 5.0  # deg; diffraction peak truncated below this angle after the
                                # first scattering event (variance reduction, 0 = off)
    # --- geometry of the world ---
    altitude: float = 10500.0   # [m] altitude of the domain origin
    # --- sun ---
    sun_elevation: float = 25.0  # deg, at the contrail location; may be negative (dusk)
    sun_azimuth: float = 120.0   # deg, from +x towards +y
    # --- atmosphere ---
    aerosol_scale: float = 1.0   # multiplies the aerosol (Mie) optical depth (turbidity)
    ozone_du: float = 300.0
    ground_albedo: float = 0.15
    # --- rendering ---
    max_bounces: int = 512
    exposure_ev: float = 0.0     # exposure compensation [EV]
    auto_exposure: bool = True
    exposure: float = -1.0       # fixed linear exposure (used when auto_exposure is false)
    camera: Camera = field(default_factory=Camera)

    def sun_direction(self):
        e, a = np.deg2rad(self.sun_elevation), np.deg2rad(self.sun_azimuth)
        return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            d = json.load(f)
        cam = Camera(**d.pop("camera", {}))
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        sc = cls(camera=cam, **known)
        # a relative CGNS path is looked up from the working directory, then
        # next to the JSON file and in its parent directory
        if not os.path.isabs(sc.cgns) and not os.path.exists(sc.cgns):
            base = os.path.dirname(os.path.abspath(path))
            for cand in (os.path.join(base, sc.cgns), os.path.join(base, "..", sc.cgns)):
                if os.path.exists(cand):
                    sc.cgns = os.path.normpath(cand)
                    break
        return sc
