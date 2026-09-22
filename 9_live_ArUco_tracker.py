"""
Dual slanted-camera ArUco tracker for MuJoCo (v2)

What changed vs. the solvePnP version
-------------------------------------
1. Pose comes from *plane back-projection* instead of solvePnP. The marker plate is
   horizontal at a known height, so each of the 4 sub-pixel corners is ray-cast onto
   the plane z = PLATE_Z. Centre = mean of the 4 world points, yaw = direction of
   the TL->TR edge. This removes the IPPE two-solution flip and the noisy PnP depth.
2. z is triangulated from the two camera rays through the marker centre (needs both
   cameras). It doubles as a sanity check: it should stay ~PLATE_Z.
3. Per-observation quality gate: the reconstructed marker side must be ~MARKER_LEN.
   Wrong detections / wrong plane height / bad corners fail this test.
4. Fusion weight = marker area in pixels (variance ~ 1/area), applied to all
   observations, not just the first two. Yaw is fused on the unit circle.
5. Camera intrinsics/extrinsics are computed once (cameras are static), and use the
   OpenCV pixel-centre convention (cx = (W-1)/2) which removes a 0.5 px bias.
6. Resolution is 1280x960 by default (640x480 gives only ~1.4-2 px per ArUco bit
   at the arena edges, which is at the detection limit).
7. HUD is drawn on the down-scaled canvas so text never overlaps; heading error
   and z error are reported too.

Run:   python aruco_tracker_v2.py [scene.xml]
Headless smoke test (no window):   python aruco_tracker_v2.py scene.xml --frames 200 --no-gui
"""
import argparse
import time
from dataclasses import dataclass

import cv2
import mujoco
import numpy as np

# ----------------------------------------------------------------------------- config
CAM_NAMES = ["cam_south", "cam_north"]
CAM_W, CAM_H = 1280, 720          # must fit <visual><global offwidth/offheight>
DISPLAY_SCALE = 0.65               # imshow scale (2 x 1280 wide is too large for a screen)
MARKER_LEN = 0.08 * (400.0 / 500.0)   # inner black pattern = 0.064 m
MAX_SCALE_ERR = 0.08              # reject if reconstructed marker size is >8 % off
MIN_TRI_ANGLE_DEG = 15.0          # need this much ray divergence to trust triangulated z
PHYSICS_STEPS_PER_FRAME = 10
ROBOTS = {0: "robot1_aruco_plate", 1: "robot2_aruco_plate"}   # marker id -> plate geom
YAW_OFFSET_DEG = None             # None = auto-calibrate on first detection, or hardcode

R_OPT = np.diag([1.0, -1.0, -1.0])    # MuJoCo cam (x right,y up,-z fwd) -> OpenCV cam


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


# ----------------------------------------------------------------------------- camera
class StaticCam:
    """Pinhole model of a fixed MuJoCo camera, expressed in OpenCV conventions."""

    def __init__(self, model, data, name, W, H):
        self.name = name
        self.cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        f = (H / 2.0) / np.tan(np.deg2rad(model.cam_fovy[self.cid]) / 2.0)
        # OpenCV pixel *centres* sit on integers, so the optical centre is (W-1)/2.
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
        self.K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        self.Kinv = np.linalg.inv(self.K)
        self.C = data.cam_xpos[self.cid].copy()                                # world position
        self.R = data.cam_xmat[self.cid].reshape(3, 3).copy() @ R_OPT          # OpenCV cam -> world

    def rays(self, px):
        """(N,2) pixels -> (N,3) unit ray directions in world frame."""
        d = (self.Kinv @ np.c_[px, np.ones(len(px))].T).T @ self.R.T
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def to_plane(self, px, z):
        """Intersect pixel rays with the horizontal plane at height z."""
        d = self.rays(px)
        s = (z - self.C[2]) / d[:, 2]
        return self.C + s[:, None] * d


# ----------------------------------------------------------------------------- measurement
@dataclass
class Obs:
    cam: StaticCam
    centre: np.ndarray      # world xyz of marker centre (on the plate plane)
    yaw: float              # rad, direction of marker x-axis (TL->TR) in world
    weight: float           # marker area in px^2
    cell_px: float          # pixels per ArUco bit (detectability indicator)
    scale_err: float        # reconstructed marker size / MARKER_LEN - 1
    centre_px: np.ndarray   # perspective-correct marker centre in the image


def diag_intersection(c):
    """Intersection of the two diagonals = image of the marker's true centre."""
    p0, p1, p2, p3 = c
    A = np.array([p2 - p0, -(p3 - p1)]).T
    t = np.linalg.solve(A, p1 - p0)
    return p0 + t[0] * (p2 - p0)


def measure(cam, corners, plate_z):
    c = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    P = cam.to_plane(c, plate_z)
    centre = P.mean(axis=0)
    ex = (P[1] - P[0]) + (P[2] - P[3])                       # marker x-axis (TL->TR) in world
    yaw = np.arctan2(ex[1], ex[0])
    scale_err = np.linalg.norm(P - np.roll(P, -1, axis=0), axis=1).mean() / MARKER_LEN - 1.0
    d1, d2 = c[2] - c[0], c[3] - c[1]
    area = 0.5 * abs(d1[0] * d2[1] - d1[1] * d2[0])
    return Obs(cam, centre, yaw, area, np.sqrt(area) / 6.0, scale_err, diag_intersection(c))


def triangulate(obs):
    """Least-squares point closest to all centre rays. Returns (xyz, max ray angle deg)."""
    A, b, dirs = np.zeros((3, 3)), np.zeros(3), []
    for o in obs:
        d = o.cam.rays(o.centre_px[None])[0]
        M = np.eye(3) - np.outer(d, d)
        A += M
        b += M @ o.cam.C
        dirs.append(d)
    ang = np.rad2deg(np.arccos(np.clip(dirs[0] @ dirs[1], -1, 1))) if len(dirs) > 1 else 0.0
    return np.linalg.solve(A, b), ang


def fuse(obs, yaw_offset_deg):
    w = np.array([o.weight for o in obs])
    w = w / w.sum()
    xy = sum(wi * o.centre[:2] for wi, o in zip(w, obs))
    s = sum(wi * np.sin(o.yaw) for wi, o in zip(w, obs))
    c = sum(wi * np.cos(o.yaw) for wi, o in zip(w, obs))
    yaw = wrap_deg(np.rad2deg(np.arctan2(s, c)) - yaw_offset_deg)
    est = dict(xy=xy, yaw=yaw, cams=len(obs), cell=min(o.cell_px for o in obs), z_tri=None)
    if len(obs) >= 2:
        p, ang = triangulate(obs)
        if ang >= MIN_TRI_ANGLE_DEG:
            est["z_tri"] = p[2]
    return est


# ----------------------------------------------------------------------------- detector
def make_detector():
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.cornerRefinementWinSize = 4
    p.cornerRefinementMaxIterations = 50
    p.cornerRefinementMinAccuracy = 0.01
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 23
    p.adaptiveThreshWinSizeStep = 10
    p.minMarkerPerimeterRate = 0.02      # small markers at the far edge of the arena
    p.polygonalApproxAccuracyRate = 0.05
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), p)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xml", nargs="?", default="robots12_arena_2cam.xml")
    ap.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = run until q/ESC)")
    ap.add_argument("--no-gui", action="store_true")
    ap.add_argument("--snapshot", default="", help="save last combined frame to this PNG")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    for name, xy in (("robot1_root", (-0.5, -0.5)), ("robot2_root", (0.5, 0.5))):
        adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        data.qpos[adr:adr + 2] = xy
    mujoco.mj_forward(model, data)

    gids = {m: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g) for m, g in ROBOTS.items()}
    # plate top surface height (box half-thickness = size[2]); flat robots -> constant
    plate_z = float(data.geom_xpos[gids[0]][2] + model.geom_size[gids[0]][2])

    cams = [StaticCam(model, data, n, CAM_W, CAM_H) for n in CAM_NAMES]
    renderer = mujoco.Renderer(model, height=CAM_H, width=CAM_W)
    detector = make_detector()
    yaw_offset = YAW_OFFSET_DEG

    if not args.no_gui:
        win = "BotArena: Dual Slanted Tracker v2"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, int(CAM_W * 2 * DISPLAY_SCALE), int(CAM_H * DISPLAY_SCALE))

    stats = {m: [] for m in ROBOTS}
    frame, t0, fps, combined = 0, time.time(), 0.0, None
    print(f"plate z = {plate_z:.4f} m | cameras: {CAM_NAMES} | {CAM_W}x{CAM_H}")

    try:
        while True:
            # Drive commands: robot1 = ctrl[0:2], robot2 = ctrl[2:4]
            data.ctrl[:4] = (0.02, 0.035, 0.03, 0.02)
            for _ in range(PHYSICS_STEPS_PER_FRAME):
                mujoco.mj_step(model, data)
            frame += 1

            gt = {m: data.geom_xpos[g].copy() for m, g in gids.items()}
            gt_yaw = {m: np.rad2deg(np.arctan2(data.geom_xmat[g][3], data.geom_xmat[g][0]))
                      for m, g in gids.items()}                    # xmat row-major: [1,0]=idx 3, [0,0]=idx 0

            per_robot = {m: [] for m in ROBOTS}
            views, labels = [], []
            for idx, cam in enumerate(cams):
                renderer.update_scene(data, camera=cam.cid)
                rgb = renderer.render()
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                corners, ids, _ = detector.detectMarkers(gray)
                view = None if args.no_gui and not args.snapshot else cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                if ids is not None:
                    for c, mid in zip(corners, ids.flatten()):
                        mid = int(mid)
                        if mid not in ROBOTS:
                            continue
                        o = measure(cam, c, plate_z)
                        ok = abs(o.scale_err) < MAX_SCALE_ERR
                        if ok:
                            per_robot[mid].append(o)
                        if view is not None:
                            col = (0, 255, 0) if ok else (0, 0, 255)
                            cv2.polylines(view, [c.reshape(-1, 2).astype(np.int32)], True, col, 2)
                            x, y = c.reshape(-1, 2)[0]
                            labels.append((idx, f"R{mid} {o.cell_px:.1f}px/bit", (x, y - 10)))
                if view is not None:
                    views.append(view)

            # -------- fusion + logging
            fused = {}
            for mid, obs in per_robot.items():
                if not obs:
                    continue
                if yaw_offset is None:      # marker frame vs robot heading is a constant offset
                    yaw_offset = wrap_deg(np.rad2deg(fuse(obs, 0.0)["yaw"]) - gt_yaw[mid])
                    yaw_offset = 90.0 * round(yaw_offset / 90.0)
                    print(f"auto-calibrated YAW_OFFSET_DEG = {yaw_offset:.0f}  (hardcode it)")
                est = fuse(obs, yaw_offset)
                est["err_xy"] = np.linalg.norm(est["xy"] - gt[mid][:2]) * 1000.0
                est["err_yaw"] = wrap_deg(est["yaw"] - gt_yaw[mid])
                est["err_z"] = None if est["z_tri"] is None else (est["z_tri"] - plate_z) * 1000.0
                fused[mid] = est
                stats[mid].append((est["err_xy"], est["err_yaw"], est["err_z"], est["cams"]))

            if frame % 30 == 0:
                fps = 30 / max(time.time() - t0, 1e-6)
                t0 = time.time()
                for mid, e in fused.items():
                    zs = "n/a" if e["err_z"] is None else f"{e['err_z']:+.1f}mm"
                    print(f"[{frame:04d}] R{mid} {e['cams']}cam xy=({e['xy'][0]:+.3f},{e['xy'][1]:+.3f}) "
                          f"errXY={e['err_xy']:.1f}mm errYaw={e['err_yaw']:+.1f}deg errZ={zs} "
                          f"min={e['cell']:.1f}px/bit  {fps:.1f} fps")

            # -------- display
            if views:
                combined = np.hstack(views)
                small = cv2.resize(combined, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_AREA)
                for idx, txt, (x, y) in labels:
                    p = (int((x + idx * CAM_W) * DISPLAY_SCALE), int(y * DISPLAY_SCALE))
                    cv2.putText(small, txt, p, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                for idx, n in enumerate(CAM_NAMES):
                    cv2.putText(small, n, (int(idx * small.shape[1] / 2) + 12, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                # two-line HUD per robot, each in its own half of the canvas
                h, w = small.shape[:2]
                cv2.rectangle(small, (0, h - 56), (w, h), (15, 15, 15), -1)
                for mid in ROBOTS:
                    x0 = 12 + mid * w // 2
                    if mid in fused:
                        e = fused[mid]
                        col = (0, 255, 0) if e["err_xy"] < 5 else (0, 165, 255)
                        z = "z n/a" if e["z_tri"] is None else f"z {e['z_tri']:.3f}m"
                        l1 = f"R{mid} [{e['cams']} cam]  xy ({e['xy'][0]:+.3f}, {e['xy'][1]:+.3f}) m  {z}"
                        l2 = f"yaw {e['yaw']:+.1f} deg | err xy {e['err_xy']:.1f} mm  yaw {e['err_yaw']:+.1f} deg"
                    else:
                        col, l1, l2 = (0, 0, 255), f"R{mid}: NO DETECTION", ""
                    cv2.putText(small, l1, (x0, h - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
                    cv2.putText(small, l2, (x0, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
                combined = small
                if not args.no_gui:
                    cv2.imshow(win, small)
                    if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                        break
            if args.frames and frame >= args.frames:
                break
    finally:
        if args.snapshot and combined is not None:
            cv2.imwrite(args.snapshot, combined)
        for mid, s in stats.items():
            if s:
                a = np.array([[x if x is not None else np.nan for x in r] for r in s], dtype=float)
                print(f"R{mid}: detected {len(s)}/{frame} frames | xy RMS {np.sqrt(np.mean(a[:, 0] ** 2)):.2f} mm "
                      f"| yaw RMS {np.sqrt(np.mean(a[:, 1] ** 2)):.2f} deg "
                      f"| z RMS {np.sqrt(np.nanmean(a[:, 2] ** 2)) if not np.all(np.isnan(a[:, 2])) else float('nan'):.2f} mm")
            else:
                print(f"R{mid}: never detected")
        renderer.close()
        if not args.no_gui:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
