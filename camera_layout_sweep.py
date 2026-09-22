"""
Camera layout sweep for ArUco tracking in a square arena (pinhole model, no rendering needed).

For every candidate rig (cam radius r, height h, aim offset, separation) it finds the *tightest* fovy that keeps
the whole +-0.9 m arena (marker corners, every yaw) inside the image of BOTH cameras, then reports the
worst-case pixels-per-ArUco-bit (detectability) and the ideal-noise xyz uncertainty from fusing both cameras.

aim_off > 0 aims past the arena centre (away from the camera), < 0 aims short of it (toward the camera).
NOTE: sub-mm numbers assume 0.15 px Gaussian centre noise; use them to compare layouts, not as absolute accuracy.
Edit Z0/HALF/GRID below for other plate heights / marker sizes / arena sizes.
"""
import numpy as np, itertools
Z0 = 0.056            # marker top surface height
HALF = 0.032          # inner pattern half-size (0.064 m)
CELLS = 6             # 4x4 bits + 2 border cells
SIG = 0.15            # px noise on marker centre (0.3px corner noise / 2)
gx = np.linspace(-0.9, 0.9, 10)
GRID = np.array([[x, y, Z0] for x in gx for y in gx])
yaws = np.deg2rad(np.arange(0, 360, 45))
sq = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]]) * HALF

def corners_world():
    out = []
    for p in GRID:
        for a in yaws:
            c, s = np.cos(a), np.sin(a)
            R = np.array([[c, -s], [s, c]])
            xy = sq @ R.T + p[:2]
            out.append(np.c_[xy, np.full(4, Z0)])
    return np.array(out)             # (N,4,3)
CW = corners_world()

def lookat(C, T):
    f = T - C; f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1]); r /= np.linalg.norm(r)
    d = np.cross(f, r)
    return np.array([r, d, f])       # world->cam rotation

def make_cam(az_deg, r, h, aim_off):
    az = np.deg2rad(az_deg); u = np.array([np.cos(az), np.sin(az), 0])
    C = np.array([r*u[0], r*u[1], h])
    T = np.array([0, 0, Z0]) - aim_off*u        # aim_off>0 -> aim past centre, away from cam
    return C, lookat(C, T)

def norm_coords(C, R):
    pc = (CW - C) @ R.T
    return pc

def evaluate(cams, W, H, fovy=None, margin=8):
    # required focal length so every marker corner (all yaws) is inside image
    fs = []
    for C, R in cams:
        pc = norm_coords(C, R)
        if (pc[..., 2] < 0.2).any(): return None
        xn = np.abs(pc[..., 0] / pc[..., 2]).max(); yn = np.abs(pc[..., 1] / pc[..., 2]).max()
        fs.append(min((W/2 - margin)/xn, (H/2 - margin)/yn))
    f_need = min(fs)
    if fovy is None: f = f_need
    else:
        f = (H/2)/np.tan(np.deg2rad(fovy)/2)
        if f > f_need*1.0001: return None   # doesn't cover
    fov = np.rad2deg(2*np.arctan(H/2/f))
    # marker cell size (worst over arena & yaw, worst camera)
    cell_min = 1e9; cell_mean = []
    for C, R in cams:
        pc = norm_coords(C, R); uv = f*pc[..., :2]/pc[..., 2:3]
        s = lambda a, b: np.linalg.norm(uv[:, a]-uv[:, b], axis=1)
        e1 = (s(0, 1)+s(3, 2))/2; e2 = (s(1, 2)+s(0, 3))/2
        cell = np.minimum(e1, e2)/CELLS
        cell_min = min(cell_min, cell.min()); cell_mean.append(cell.mean())
    # xyz covariance from fusing cams, at yaw-independent centres
    sx = []; sz = []; sxy_planar = []
    for P in GRID:
        I = np.zeros((3, 3))
        for C, R in cams:
            pc = R @ (P - C)
            dU = np.array([[f/pc[2], 0, -f*pc[0]/pc[2]**2], [0, f/pc[2], -f*pc[1]/pc[2]**2]])
            J = dU @ R; I += J.T @ J / SIG**2
        cov = np.linalg.inv(I)
        sx.append(np.sqrt(cov[0, 0]+cov[1, 1])); sz.append(np.sqrt(cov[2, 2]))
        Ixy = I[:2, :2]; sxy_planar.append(np.sqrt(np.trace(np.linalg.inv(Ixy))))
    return dict(fovy=fov, hfov=np.rad2deg(2*np.arctan(W/2/f)), cell_min=cell_min,
                cell_mean=np.mean(cell_mean),
                xy_free_mm=np.max(sx)*1e3, z_free_mm=np.max(sz)*1e3,
                xy_z_known_mm=np.max(sxy_planar)*1e3, xy_z_known_mean=np.mean(sxy_planar)*1e3)

def fmt(tag, e):
    if e is None: return f"{tag}: cannot cover arena"
    return (f"{tag}: fovy={e['fovy']:.1f} hfov={e['hfov']:.1f} | cell px min/mean={e['cell_min']:.2f}/{e['cell_mean']:.2f} | "
            f"worst xy(z free)={e['xy_free_mm']:.1f}mm z={e['z_free_mm']:.1f}mm | xy(z known) worst={e['xy_z_known_mm']:.1f} mean={e['xy_z_known_mean']:.1f}mm")


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    print("Ranked by worst-case px/bit, full arena in view of both cameras, fovy = tightest that covers it")
    for W, H in [(640, 480), (1280, 960), (1920, 1080)]:
        res = []
        for az2 in (180, 90):                                   # cameras opposite / 90 deg apart
            for r, h, a in itertools.product(np.arange(0.5, 2.51, 0.25), np.arange(1.2, 2.81, 0.2), np.arange(-0.4, 0.81, 0.2)):
                try: e = evaluate([make_cam(-90, r, h, a), make_cam(-90 + az2, r, h, a)], W, H)
                except Exception: continue
                if e and np.isfinite(e["fovy"]) and e["cell_min"] < 1e6: res.append((e["cell_min"], az2, r, h, a, e))
        res.sort(key=lambda t: -t[0])
        print(f"\n{W}x{H}")
        for az2 in (180, 90):
            for t in [x for x in res if x[1] == az2][:3]:
                e = t[5]
                print(f"  sep={az2:3d} r={t[2]:.2f} h={t[3]:.1f} aim_off={t[4]:+.1f} -> fovy={e['fovy']:.1f} "
                      f"min px/bit={e['cell_min']:.2f} | xyz worst (ideal): xy={e['xy_free_mm']:.2f} z={e['z_free_mm']:.2f} mm")
    print("\nYour mount point (r=0.774, h=1.659) retuned, 1280x960:")
    for a in (-0.4, -0.2, 0.0, 0.2, 0.4):
        e = evaluate([make_cam(-90, .774, 1.659, a), make_cam(90, .774, 1.659, a)], 1280, 960)
        print(f"  aim_off={a:+.1f}: " + ("not coverable" if e is None else f"fovy={e['fovy']:.1f} min px/bit={e['cell_min']:.2f}"))
