"""
Dual Slanted-Camera ArUco Tracker (v2 Stress Benchmark)
File: 12_worst_case_stress_test.py

Features:
- Adversarial trajectory generation: Boundary diving, rapid in-place spins, cross-passes.
- Plane back-projection (z = PLATE_Z) + least-squares dual-ray triangulation for z.
- Metric scale gate validation (MAX_SCALE_ERR).
- Camera name overlays and telemetry HUD optimized for 1920-wide displays.
- Automatic anomaly snapshot capture upon dropout or error spikes.
"""

import argparse
import os
import time
from dataclasses import dataclass

import cv2
import mujoco
import numpy as np

# ----------------------------------------------------------------------------- Configuration
CAM_NAMES = ["cam_south", "cam_north"]
CAM_W, CAM_H = 1280, 720              # Native 16:9 720p
DISPLAY_SCALE = 0.70                  # Fitted for 1920x1080 / 1920x1200 screens
MARKER_LEN = 0.08 * (400.0 / 500.0)   # 0.064 m inner black pattern
MAX_SCALE_ERR = 0.08                  # Reject if reconstructed size is >8% off
MIN_TRI_ANGLE_DEG = 15.0              # Ray convergence threshold for valid triangulated z
PHYSICS_STEPS_PER_FRAME = 10
ROBOTS = {0: "robot1_aruco_plate", 1: "robot2_aruco_plate"}
YAW_OFFSET_DEG = None

# Worst-case incident dump thresholds
WORST_ERR_XY_THRESH_MM = 15.0         # Capture snapshot if XY error > 15 mm
WORST_ERR_Z_THRESH_MM = 12.0          # Capture snapshot if Z error > 12 mm
WORST_ERR_YAW_THRESH_DEG = 8.0        # Capture snapshot if Yaw error > 8 deg
DUMP_DIR = "worst_case_dumps"
os.makedirs(DUMP_DIR, exist_ok=True)

R_OPT = np.diag([1.0, -1.0, -1.0])


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


# ----------------------------------------------------------------------------- Adversarial Motion
def get_stress_control(t, r0_pos, r1_pos):
    """
    Generates 3 distinct stress-testing motion regimes:
      Phase 1 (0-12s)  : Perimeter & corner push (high speed, extreme camera foreshortening).
      Phase 2 (12-22s) : Aggressive in-place yaw reversals (corner refinement jitter test).
      Phase 3 (22-32s) : Center crossing & mutual passes (mutual camera line-of-sight test).
    """
    cycle_t = t % 32.0

    if cycle_t < 12.0:
        # Phase 1: Push toward arena extremes
        u0_left = 0.045 + 0.015 * np.sin(0.6 * t)
        u0_right = 0.045 - 0.015 * np.sin(0.6 * t)
        u1_left = 0.040 - 0.018 * np.cos(0.5 * t)
        u1_right = 0.040 + 0.018 * np.cos(0.5 * t)
    elif cycle_t < 22.0:
        # Phase 2: Rapid alternating spins
        spin_dir0 = np.sign(np.sin(2.0 * np.pi * 0.5 * t))
        spin_dir1 = -np.sign(np.cos(2.0 * np.pi * 0.5 * t))
        u0_left, u0_right = 0.038 * spin_dir0, -0.038 * spin_dir0
        u1_left, u1_right = 0.038 * spin_dir1, -0.038 * spin_dir1
    else:
        # Phase 3: Center crossing figure-8
        u0_left = 0.035 + 0.020 * np.sin(1.1 * t)
        u0_right = 0.035 - 0.020 * np.sin(1.1 * t)
        u1_left = 0.035 - 0.020 * np.sin(1.1 * t)
        u1_right = 0.035 + 0.020 * np.sin(1.1 * t)

    # Perimeter wall bounce barrier: keep within arena limits (|x|, |y| < 0.82m)
    cmd = [u0_left, u0_right, u1_left, u1_right]
    if abs(r0_pos[0]) > 0.80 or abs(r0_pos[1]) > 0.80:
        cmd[0], cmd[1] = -0.030, 0.030
    if abs(r1_pos[0]) > 0.80 or abs(r1_pos[1]) > 0.80:
        cmd[2], cmd[3] = 0.030, -0.030

    return tuple(cmd)


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

    sides = np.linalg.norm(P - np.roll(P, -1, axis=0), axis=1)
    scale_err = (sides.mean() / MARKER_LEN) - 1.0

    d1, d2 = c[2] - c[0], c[3] - c[1]
    area = 0.5 * abs(d1[0] * d2[1] - d1[1] * d2[0])

    ray_to_center = centre - cam.C
    dist = np.linalg.norm(ray_to_center)
    inc_deg = np.rad2deg(np.arccos(np.clip(-ray_to_center[2] / dist, -1.0, 1.0)))

    return Obs(
        cam, centre, yaw, area, np.sqrt(area) / 6.0, scale_err,
        diag_intersection(c), inc_deg
    )


def triangulate(obs):
    A, b, dirs = np.zeros((3, 3)), np.zeros(3), []
    for o in obs:
        d = o.cam.rays(o.centre_px[None])[0]
        M = np.eye(3) - np.outer(d, d)
        A += M
        b += M @ o.cam.C
        dirs.append(d)
    ang = np.rad2deg(np.arccos(np.clip(dirs[0] @ dirs[1], -1.0, 1.0))) if len(dirs) > 1 else 0.0
    return np.linalg.solve(A, b), ang


def fuse(obs, yaw_offset_deg, plate_z):
    w = np.array([o.weight for o in obs])
    w = w / w.sum()
    xy = sum(wi * o.centre[:2] for wi, o in zip(w, obs))
    s = sum(wi * np.sin(o.yaw) for wi, o in zip(w, obs))
    c = sum(wi * np.cos(o.yaw) for wi, o in zip(w, obs))
    yaw = wrap_deg(np.rad2deg(np.arctan2(s, c)) - yaw_offset_deg)

    z_tri, tri_angle = None, 0.0
    if len(obs) >= 2:
        p, ang = triangulate(obs)
        tri_angle = ang
        if ang >= MIN_TRI_ANGLE_DEG:
            z_tri = float(p[2])

    pose_3d = np.array([xy[0], xy[1], z_tri if z_tri is not None else plate_z])

    return dict(
        xy=xy,
        z_tri=z_tri,
        pose_3d=pose_3d,
        yaw=yaw,
        cams=len(obs),
        tri_angle=tri_angle,
        min_cell=min(o.cell_px for o in obs),
        max_scale_err=max(abs(o.scale_err) for o in obs),
        max_inc=max(o.incidence_deg for o in obs),
    )


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


# ----------------------------------------------------------------------------- Main
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

    win = "BotArena: v2 3D Pose Worst-Case Stress Benchmark"
    canvas_w = int(CAM_W * 2 * DISPLAY_SCALE)
    canvas_h = int(CAM_H * DISPLAY_SCALE) + 135

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, canvas_w, canvas_h)

    worst_case = {
        m: {
            "max_xy_err_mm": 0.0,
            "max_z_err_mm": 0.0,
            "max_yaw_err_deg": 0.0,
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
    sim_time = 0.0
    last_dump_time = {0: 0.0, 1: 0.0}

    print("=" * 80)
    print(" V2 ARUCO TRACKER: DYNAMIC STRESS-TEST BENCHMARK INITIALIZED")
    print(f" Arena Plate Ground Truth Z: {plate_z:.4f} m | Native Resolution: {CAM_W}x{CAM_H}")
    print(f" Target Incident Dump Directory: ./{DUMP_DIR}/")
    print("=" * 80)

    try:
        while True:
            # -------------------------------------------------------------
            # ADVERSARIAL DRIVING COMMANDS
            # -------------------------------------------------------------
            r0_pos = data.geom_xpos[gids[0]][:2]
            r1_pos = data.geom_xpos[gids[1]][:2]
            data.ctrl[:4] = get_stress_control(sim_time, r0_pos, r1_pos)

            for _ in range(PHYSICS_STEPS_PER_FRAME):
                mujoco.mj_step(model, data)
                sim_time += model.opt.timestep
            frame += 1

            # Ground truth 3D poses (center top face of plate)
            gt_3d = {
                m: np.array([
                    data.geom_xpos[g][0],
                    data.geom_xpos[g][1],
                    data.geom_xpos[g][2] + model.geom_size[g][2]
                ])
                for m, g in gids.items()
            }
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

                # ---------------------------------------------------------
                # CAMERA NAME OVERLAY BADGE
                # ---------------------------------------------------------
                cv2.rectangle(bgr, (16, 16), (280, 56), (15, 15, 15), -1)
                cv2.rectangle(bgr, (16, 16), (280, 56), (90, 90, 90), 1)
                cv2.putText(
                    bgr, f"CAM: {CAM_NAMES[idx]}", (30, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA
                )

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
                            (int(cp[0]), int(cp[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA
                        )

                views.append(bgr)

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
                    yaw_offset = wrap_deg(np.rad2deg(fuse(obs, 0.0, plate_z)["yaw"]) - gt_yaw[mid])
                    yaw_offset = 90.0 * round(yaw_offset / 90.0)

                est = fuse(obs, yaw_offset, plate_z)
                err_xy = float(np.linalg.norm(est["xy"] - gt_3d[mid][:2]) * 1000.0)
                err_yaw = float(abs(wrap_deg(est["yaw"] - gt_yaw[mid])))
                err_z = (
                    float(abs(est["z_tri"] - gt_3d[mid][2]) * 1000.0)
                    if est["z_tri"] is not None else None
                )

                est["err_xy"] = err_xy
                est["err_yaw"] = err_yaw
                est["err_z"] = err_z
                fused[mid] = est

                # Record worst-case peak deviations
                st = worst_case[mid]
                st["max_xy_err_mm"] = max(st["max_xy_err_mm"], err_xy)
                st["max_yaw_err_deg"] = max(st["max_yaw_err_deg"], err_yaw)
                if err_z is not None:
                    st["max_z_err_mm"] = max(st["max_z_err_mm"], err_z)
                st["min_cell_px"] = min(st["min_cell_px"], est["min_cell"])
                st["max_scale_err_pct"] = max(st["max_scale_err_pct"], est["max_scale_err"] * 100.0)
                st["max_incidence_deg"] = max(st["max_incidence_deg"], est["max_inc"])

                # Trigger automatic anomaly dump if limits are breached
                now = time.time()
                is_anomaly = (
                    (err_xy > WORST_ERR_XY_THRESH_MM)
                    or (err_yaw > WORST_ERR_YAW_THRESH_DEG)
                    or (err_z is not None and err_z > WORST_ERR_Z_THRESH_MM)
                )
                if is_anomaly and (now - last_dump_time[mid] > 2.0):
                    last_dump_time[mid] = now
                    z_tag = f"_z{int(err_z)}mm" if err_z is not None else "_zNA"
                    fname = f"{DUMP_DIR}/anomaly_R{mid}_f{frame:05d}_xy{int(err_xy)}mm{z_tag}_yaw{int(err_yaw)}d.png"
                    cv2.imwrite(fname, combined_full)
                    print(f"\n[!] ADVERSARIAL STRESS ANOMALY DUMP: {fname}")
                    print(f"    R{mid} Pose: ({est['pose_3d'][0]:+.3f}, {est['pose_3d'][1]:+.3f}, {est['pose_3d'][2]:+.3f}) | ErrXY: {err_xy:.1f}mm | ErrZ: {err_z if err_z is not None else -1:.1f}mm")

            # -------------------------------------------------------------
            # HUD TELEMETRY CANVAS
            # -------------------------------------------------------------
            small = cv2.resize(combined_full, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_AREA)
            sh, sw = small.shape[:2]
            hud = np.zeros((135, sw, 3), dtype=np.uint8)
            hud[:] = 18

            for mid in ROBOTS:
                x_off = 15 + mid * (sw // 2)
                st = worst_case[mid]
                if mid in fused:
                    e = fused[mid]
                    color = (0, 255, 0) if e["err_xy"] < 10.0 else (0, 165, 255)
                    z_err_str = f"ErrZ: {e['err_z']:.1f}mm" if e["err_z"] is not None else "ErrZ: n/a (1 cam)"
                    l1 = f"Robot {mid} [{e['cams']} Cam] 3D Pose: ({e['pose_3d'][0]:+.3f}, {e['pose_3d'][1]:+.3f}, {e['pose_3d'][2]:+.3f})m"
                    l2 = f"Now -> ErrXY: {e['err_xy']:.1f}mm | {z_err_str} | ErrYaw: {e['err_yaw']:.1f}d | Yaw: {e['yaw']:+.1f}d"
                else:
                    color = (0, 0, 255)
                    l1 = f"Robot {mid}: NO DETECTION / BLIND SPOT"
                    l2 = f"Gate Drops: {st['scale_gate_rejections']} | Total Lost: {st['dropouts']}"

                l3 = f"WORST PEAKS -> MaxXY: {st['max_xy_err_mm']:.1f}mm | MaxZ: {st['max_z_err_mm']:.1f}mm | MaxYaw: {st['max_yaw_err_deg']:.1f}d"
                l4 = f"ScaleDev: {st['max_scale_err_pct']:.1f}% | IncAngle: {st['max_incidence_deg']:.1f}d | Uptime: {((frame - st['dropouts']) / max(frame, 1))*100:.1f}%"

                cv2.putText(hud, l1, (x_off, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
                cv2.putText(hud, l2, (x_off, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
                cv2.putText(hud, l3, (x_off, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 165, 255), 1, cv2.LINE_AA)
                cv2.putText(hud, l4, (x_off, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

            display_canvas = np.vstack([small, hud])
            cv2.imshow(win, display_canvas)

            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
            if args.frames and frame >= args.frames:
                break

    finally:
        print("\n" + "=" * 80)
        print(" FINAL WORST-CASE BENCHMARK REPORT")
        print("=" * 80)
        for mid, st in worst_case.items():
            tot = max(st["total_frames"], 1)
            valid = tot - st["dropouts"]
            print(f"Robot {mid}:")
            print(f"  Tracked Frames        : {valid}/{tot} ({valid/tot*100:.2f}%)")
            print(f"  Peak Trans Error (XY) : {st['max_xy_err_mm']:.2f} mm")
            print(f"  Peak Z Error (Triang) : {st['max_z_err_mm']:.2f} mm")
            print(f"  Peak Heading Error    : {st['max_yaw_err_deg']:.2f} deg")
            print(f"  Worst Marker Foreshort: {st['min_cell_px']:.2f} px/bit")
            print(f"  Worst Metric Dev      : {st['max_scale_err_pct']:.2f} % (Scale Gate Limit: 8.0%)")
            print(f"  Worst Incidence Angle : {st['max_incidence_deg']:.1f} deg")
            print(f"  Scale Gate Rejections : {st['scale_gate_rejections']}")
            print("-" * 80)

        renderer.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()