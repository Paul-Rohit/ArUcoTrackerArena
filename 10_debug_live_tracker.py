import argparse
import os
import time
from dataclasses import dataclass

import cv2
import mujoco
import numpy as np

# ----------------------------------------------------------------------------- Configuration
CAM_NAMES = ["cam_south", "cam_north"]
CAM_W, CAM_H = 1280, 720          # 16:9 native 720p matching physical webcams
DISPLAY_SCALE = 0.5
MARKER_LEN = 0.08 * (400.0 / 500.0)  # 0.064 m inner black pattern
MAX_SCALE_ERR = 0.08              # Reject observation if side deviates > 8%
MIN_TRI_ANGLE_DEG = 15.0
PHYSICS_STEPS_PER_FRAME = 10
ROBOTS = {0: "robot1_aruco_plate", 1: "robot2_aruco_plate"}
YAW_OFFSET_DEG = None

# Worst-case anomaly triggers for automatic dumping
WORST_ERR_XY_THRESH_MM = 20.0     # Save snapshot if error exceeds 20mm
WORST_ERR_YAW_THRESH_DEG = 10.0   # Save snapshot if yaw error exceeds 10 deg
DUMP_DIR = "worst_case_dumps"
os.makedirs(DUMP_DIR, exist_ok=True)

R_OPT = np.diag([1.0, -1.0, -1.0])


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


# ----------------------------------------------------------------------------- Camera Model
class StaticCam:
    def __init__(self, model, data, name, W, H):
        self.name = name
        self.cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        f = (H / 2.0) / np.tan(np.deg2rad(model.cam_fovy[self.cid]) / 2.0)
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
        self.K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        self.Kinv = np.linalg.inv(self.K)
        self.C = data.cam_xpos[self.cid].copy()
        self.R = data.cam_xmat[self.cid].reshape(3, 3).copy() @ R_OPT

    def rays(self, px):
        d = (self.Kinv @ np.c_[px, np.ones(len(px))].T).T @ self.R.T
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def to_plane(self, px, z):
        d = self.rays(px)
        s = (z - self.C[2]) / d[:, 2]
        return self.C + s[:, None] * d


# ----------------------------------------------------------------------------- Measurements
@dataclass
class Obs:
    cam: StaticCam
    centre: np.ndarray
    yaw: float
    weight: float
    cell_px: float
    scale_err: float
    centre_px: np.ndarray
    raw_corners: np.ndarray
    incidence_deg: float


def diag_intersection(c):
    p0, p1, p2, p3 = c
    A = np.array([p2 - p0, -(p3 - p1)]).T
    t = np.linalg.solve(A, p1 - p0)
    return p0 + t[0] * (p2 - p0)


def measure(cam, corners, plate_z):
    c = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    P = cam.to_plane(c, plate_z)
    centre = P.mean(axis=0)
    ex = (P[1] - P[0]) + (P[2] - P[3])
    yaw = np.arctan2(ex[1], ex[0])
    
    # Calculate reconstructed side length error against known 0.064m
    sides = np.linalg.norm(P - np.roll(P, -1, axis=0), axis=1)
    scale_err = (sides.mean() / MARKER_LEN) - 1.0
    
    d1, d2 = c[2] - c[0], c[3] - c[1]
    area = 0.5 * abs(d1[0] * d2[1] - d1[1] * d2[0])
    
    # Incidence angle relative to the arena floor upward normal [0, 0, 1]
    ray_to_center = centre - cam.C
    dist = np.linalg.norm(ray_to_center)
    inc_deg = np.rad2deg(np.arccos(np.clip(-ray_to_center[2] / dist, -1.0, 1.0)))
    
    return Obs(
        cam, centre, yaw, area, np.sqrt(area) / 6.0, scale_err,
        diag_intersection(c), c, inc_deg
    )


def triangulate(obs):
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
    
    est = dict(
        xy=xy, yaw=yaw, cams=len(obs),
        min_cell=min(o.cell_px for o in obs),
        max_scale_err=max(abs(o.scale_err) for o in obs),
        max_inc=max(o.incidence_deg for o in obs),
        z_tri=None
    )
    if len(obs) >= 2:
        p, ang = triangulate(obs)
        if ang >= MIN_TRI_ANGLE_DEG:
            est["z_tri"] = p[2]
    return est


def make_detector():
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.cornerRefinementWinSize = 5
    p.cornerRefinementMaxIterations = 40
    p.cornerRefinementMinAccuracy = 0.01
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 23
    p.adaptiveThreshWinSizeStep = 4
    p.minMarkerPerimeterRate = 0.02
    p.polygonalApproxAccuracyRate = 0.05
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), p)


# ----------------------------------------------------------------------------- Diagnostic Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xml", nargs="?", default="robots12_arena_2cam.xml")
    ap.add_argument("--frames", type=int, default=0)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    for name, xy in (("robot1_root", (-0.5, -0.5)), ("robot2_root", (0.5, 0.5))):
        adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        data.qpos[adr:adr + 2] = xy
    mujoco.mj_forward(model, data)

    gids = {m: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g) for m, g in ROBOTS.items()}
    plate_z = float(data.geom_xpos[gids[0]][2] + model.geom_size[gids[0]][2])

    cams = [StaticCam(model, data, n, CAM_W, CAM_H) for n in CAM_NAMES]
    renderer = mujoco.Renderer(model, height=CAM_H, width=CAM_W)
    detector = make_detector()
    yaw_offset = YAW_OFFSET_DEG

    win = "BotArena: v2 Worst-Case Diagnostic Workbench"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, int(CAM_W * 2 * DISPLAY_SCALE), int(CAM_H * DISPLAY_SCALE) + 120)

    # Tracking Worst-Case Analytics
    worst_case = {
        m: {
            "max_xy_err_mm": 0.0,
            "max_yaw_err_deg": 0.0,
            "max_z_err_mm": 0.0,
            "min_cell_px": 999.0,
            "max_scale_err_pct": 0.0,
            "max_incidence_deg": 0.0,
            "dropouts": 0,
            "scale_gate_rejections": 0,
            "total_frames": 0,
        }
        for m in ROBOTS
    }

    frame, t0 = 0, time.time()
    last_dump_time = {0: 0.0, 1: 0.0}

    print("=" * 75)
    print(" V2 PLANE RAY-CAST TRACKER: WORST-CASE BENCHMARK RUNNING")
    print(f" Output logs and anomaly snapshots directed to: ./{DUMP_DIR}/")
    print("=" * 75)

    try:
        while True:
            # Trajectory driving loop
            data.ctrl[:4] = (0.02, 0.035, 0.03, 0.02)
            for _ in range(PHYSICS_STEPS_PER_FRAME):
                mujoco.mj_step(model, data)
            frame += 1

            gt = {m: data.geom_xpos[g].copy() for m, g in gids.items()}
            gt_yaw = {
                m: np.rad2deg(np.arctan2(data.geom_xmat[g][3], data.geom_xmat[g][0]))
                for m, g in gids.items()
            }

            per_robot = {m: [] for m in ROBOTS}
            rejected_by_gate = {m: 0 for m in ROBOTS}
            views = []

            for idx, cam in enumerate(cams):
                renderer.update_scene(data, camera=cam.cid)
                rgb = renderer.render()
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                corners, ids, _ = detector.detectMarkers(gray)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                if ids is not None:
                    for c, mid in zip(corners, ids.flatten()):
                        mid = int(mid)
                        if mid not in ROBOTS:
                            continue
                        o = measure(cam, c, plate_z)
                        ok = abs(o.scale_err) < MAX_SCALE_ERR

                        if ok:
                            per_robot[mid].append(o)
                            col = (0, 255, 0)
                        else:
                            rejected_by_gate[mid] += 1
                            col = (0, 0, 255)

                        cv2.polylines(bgr, [c.reshape(-1, 2).astype(np.int32)], True, col, 2)
                        cp = c.reshape(-1, 2)[0]
                        cv2.putText(
                            bgr, f"R{mid} dS:{o.scale_err*100:+.1f}% inc:{o.incidence_deg:.0f}d",
                            (int(cp[0]), int(cp[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1
                        )

                views.append(bgr)

            # Fusion and Worst-Case Tracking
            combined_full = np.hstack(views)
            fused = {}

            for mid in ROBOTS:
                worst_case[mid]["total_frames"] += 1
                obs = per_robot[mid]

                if not obs:
                    worst_case[mid]["dropouts"] += 1
                    if rejected_by_gate[mid] > 0:
                        worst_case[mid]["scale_gate_rejections"] += 1
                    continue

                if yaw_offset is None:
                    yaw_offset = wrap_deg(np.rad2deg(fuse(obs, 0.0)["yaw"]) - gt_yaw[mid])
                    yaw_offset = 90.0 * round(yaw_offset / 90.0)

                est = fuse(obs, yaw_offset)
                err_xy = float(np.linalg.norm(est["xy"] - gt[mid][:2]) * 1000.0)
                err_yaw = float(abs(wrap_deg(est["yaw"] - gt_yaw[mid])))
                err_z = (
                    abs(est["z_tri"] - plate_z) * 1000.0 if est["z_tri"] is not None else None
                )

                est["err_xy"] = err_xy
                est["err_yaw"] = err_yaw
                est["err_z"] = err_z
                fused[mid] = est

                # Record peak metrics
                m_stat = worst_case[mid]
                m_stat["max_xy_err_mm"] = max(m_stat["max_xy_err_mm"], err_xy)
                m_stat["max_yaw_err_deg"] = max(m_stat["max_yaw_err_deg"], err_yaw)
                if err_z is not None:
                    m_stat["max_z_err_mm"] = max(m_stat["max_z_err_mm"], err_z)
                m_stat["min_cell_px"] = min(m_stat["min_cell_px"], est["min_cell"])
                m_stat["max_scale_err_pct"] = max(m_stat["max_scale_err_pct"], est["max_scale_err"] * 100.0)
                m_stat["max_incidence_deg"] = max(m_stat["max_incidence_deg"], est["max_inc"])

                # Automatic Worst-Case Trigger Snapshot
                now = time.time()
                is_anomaly = (err_xy > WORST_ERR_XY_THRESH_MM) or (err_yaw > WORST_ERR_YAW_THRESH_DEG)
                if is_anomaly and (now - last_dump_time[mid] > 2.0):
                    last_dump_time[mid] = now
                    fname = f"{DUMP_DIR}/worst_R{mid}_f{frame:05d}_xy{int(err_xy)}mm_yaw{int(err_yaw)}deg.png"
                    cv2.imwrite(fname, combined_full)
                    print(f"\n[!] WORST-CASE SNAPSHOT CAPTURED: {fname}")
                    print(f"    R{mid} | errXY: {err_xy:.1f}mm | errYaw: {err_yaw:.1f}deg | GT: {gt[mid][:2]}")

            # Assemble Workbench HUD
            small = cv2.resize(combined_full, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_AREA)
            sh, sw = small.shape[:2]
            hud = np.zeros((120, sw, 3), dtype=np.uint8)
            hud[:] = 18

            for mid in ROBOTS:
                x_off = 15 + mid * (sw // 2)
                st = worst_case[mid]
                if mid in fused:
                    e = fused[mid]
                    color = (0, 255, 0) if e["err_xy"] < 10.0 else (0, 165, 255)
                    l1 = f"Robot {mid} [{e['cams']} Cam]: XY: ({e['xy'][0]:+.3f}, {e['xy'][1]:+.3f})m"
                    l2 = f"Now -> ErrXY: {e['err_xy']:.1f}mm | ErrYaw: {e['err_yaw']:.1f}d | MinCell: {e['min_cell']:.1f}px"
                else:
                    color = (0, 0, 255)
                    l1 = f"Robot {mid}: BLIND SPOT / LOST"
                    l2 = f"Gate Drops: {st['scale_gate_rejections']} | Total Dropouts: {st['dropouts']}"

                l3 = (
                    f"WORST-CASE -> MaxXY: {st['max_xy_err_mm']:.1f}mm | "
                    f"MaxYaw: {st['max_yaw_err_deg']:.1f}d | MaxScaleDev: {st['max_scale_err_pct']:.1f}%"
                )
                l4 = (
                    f"Reliability: {((frame - st['dropouts']) / max(frame, 1))*100:.1f}% | "
                    f"MaxIncidence: {st['max_incidence_deg']:.1f}d (Limit 58d)"
                )

                cv2.putText(hud, l1, (x_off, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
                cv2.putText(hud, l2, (x_off, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
                cv2.putText(hud, l3, (x_off, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 165, 255), 1, cv2.LINE_AA)
                cv2.putText(hud, l4, (x_off, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

            display_canvas = np.vstack([small, hud])
            cv2.imshow(win, display_canvas)

            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
            if args.frames and frame >= args.frames:
                break

    finally:
        print("\n" + "=" * 75)
        print(" FINAL BENCHMARK SUMMARY")
        print("=" * 75)
        for mid, st in worst_case.items():
            tot = max(st["total_frames"], 1)
            valid = tot - st["dropouts"]
            print(f"Robot {mid}:")
            print(f"  Frame Tracking Ratio  : {valid}/{tot} ({valid/tot*100:.2f}%)")
            print(f"  Worst Peak XY Error   : {st['max_xy_err_mm']:.2f} mm")
            print(f"  Worst Peak Yaw Error  : {st['max_yaw_err_deg']:.2f} deg")
            print(f"  Worst Peak Z Error    : {st['max_z_err_mm']:.2f} mm")
            print(f"  Worst Marker Foreshort: {st['min_cell_px']:.2f} px/bit")
            print(f"  Worst Metric Deviation: {st['max_scale_err_pct']:.2f} % (Scale Gate Limit: 8.0%)")
            print(f"  Worst Incidence Angle : {st['max_incidence_deg']:.1f} deg")
            print(f"  Scale Gate Rejections : {st['scale_gate_rejections']}")
            print("-" * 75)

        renderer.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()