"""Reader for structured CGNS/HDF5 files holding the ice-crystal fields.

Only h5py is required (CGNS >= 3 files are HDF5).  The reader returns the
fields on a *uniform* Cartesian grid, resampling them if the CGNS grid is
stretched (the renderer does trilinear lookups on a regular grid).
"""
import numpy as np
import h5py

DEFAULT_NUMBER = ("IceNumberDensity", "NumberDensity", "n_ice", "N_ice", "Nice")
DEFAULT_RADIUS = ("IceMeanRadius", "MeanRadius", "r_ice", "R_ice", "Rice")


def _data(node):
    return node[" data"][()]


def _label(node):
    v = node.attrs.get("label")
    return v.tobytes().decode(errors="ignore").rstrip("\x00") if v is not None else ""


def _string(node):
    return bytes(_data(node)).decode(errors="ignore").strip()


def list_zones(path):
    out = []
    with h5py.File(path, "r") as f:
        for bname, base in f.items():
            if not isinstance(base, h5py.Group) or _label(base) != "CGNSBase_t":
                continue
            for zname, zone in base.items():
                if isinstance(zone, h5py.Group) and _label(zone) == "Zone_t":
                    out.append(f"{bname}/{zname}")
    return out


def _find_field(sol, names):
    for n in names:
        if n in sol:
            return n
    lower = {k.lower(): k for k in sol.keys()}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    raise KeyError(f"none of {names} found in FlowSolution (available: {list(sol.keys())})")


def read_contrail(path, zone=None, number_field=None, radius_field=None, solution=None,
                  verbose=True):
    """Return dict with N [#/m3], r [m] on a uniform grid (arrays indexed [i,j,k]
    = [x,y,z]), origin (3,) and spacing (3,) in metres."""
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(f"CGNS file not found: {path!r} (use --cgns or the 'cgns' entry of the scene JSON)")
    zones = list_zones(path)
    if not zones:
        raise ValueError(f"no CGNS zone found in {path}")
    zone = zone or zones[0]
    with h5py.File(path, "r") as f:
        z = f[zone]
        ztype = _string(z["ZoneType"]) if "ZoneType" in z else "Structured"
        if ztype != "Structured":
            raise NotImplementedError("only structured zones are supported "
                                      "(resample unstructured data on a Cartesian grid first)")
        gc = z["GridCoordinates"]
        # HDF5 stores CGNS arrays in C order -> reverse to [i,j,k]
        X, Y, Z = (np.ascontiguousarray(_data(gc[f"Coordinate{c}"]).T) for c in "XYZ")
        sols = [k for k, v in z.items() if isinstance(v, h5py.Group) and _label(v) == "FlowSolution_t"]
        sol = z[solution or sols[0]]
        loc = _string(sol["GridLocation"]) if "GridLocation" in sol else "Vertex"
        fn = _find_field(sol, [number_field] if number_field else DEFAULT_NUMBER)
        fr = _find_field(sol, [radius_field] if radius_field else DEFAULT_RADIUS)
        N = np.ascontiguousarray(_data(sol[fn]).T).astype(np.float64)
        r = np.ascontiguousarray(_data(sol[fr]).T).astype(np.float64)

    if loc == "CellCenter":
        X, Y, Z = (0.125 * sum(A[a:A.shape[0] - 1 + a, b:A.shape[1] - 1 + b, c:A.shape[2] - 1 + c]
                               for a in (0, 1) for b in (0, 1) for c in (0, 1)) for A in (X, Y, Z))
    x1, y1, z1 = X[:, 0, 0], Y[0, :, 0], Z[0, 0, :]
    # Rectilinear check: x only depends on i, etc.
    if not (np.allclose(X, x1[:, None, None]) and np.allclose(Y, y1[None, :, None])
            and np.allclose(Z, z1[None, None, :])):
        raise NotImplementedError("curvilinear grids are not supported; the grid must be rectilinear")

    N = np.nan_to_num(np.maximum(N, 0.0))
    r = np.nan_to_num(np.maximum(r, 0.0))
    axes = [x1, y1, z1]
    uniform = all(np.allclose(np.diff(a), np.diff(a)[0], rtol=1e-4) for a in axes)
    if not uniform:
        from scipy.interpolate import RegularGridInterpolator
        h = min(np.diff(a).min() for a in axes)
        new_axes = [np.linspace(a[0], a[-1], int(round((a[-1] - a[0]) / h)) + 1) for a in axes]
        pts = np.stack(np.meshgrid(*new_axes, indexing="ij"), axis=-1)
        # interpolate N and N*r so that the mean radius stays consistent
        Nr = RegularGridInterpolator(axes, N * r)(pts)
        N = RegularGridInterpolator(axes, N)(pts)
        r = np.where(N > 0, Nr / np.maximum(N, 1e-30), 0.0)
        axes = new_axes
        if verbose:
            print(f"[cgns] stretched grid resampled to uniform {N.shape}")
    origin = np.array([a[0] for a in axes])
    spacing = np.array([a[1] - a[0] for a in axes])
    if verbose:
        ext = spacing * (np.array(N.shape) - 1)
        print(f"[cgns] {path}:{zone} fields '{fn}', '{fr}' ({loc}), grid {N.shape}, "
              f"extent {ext} m, N max {N.max():.3e} /m3, r max {r.max() * 1e6:.2f} um")
    return dict(N=N, r=r, origin=origin, spacing=spacing)
