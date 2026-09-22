import os
import time
import cv2
import mujoco
import numpy as np

# -----------------------------------------------------------------------------
# 1. Load Model & Simulation State
# -----------------------------------------------------------------------------
xml_path = "robots12_arena_2cam.xml"
if not os.path.exists(xml_path):
    raise FileNotFoundError(f"Cannot find XML file: {xml_path}")

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Set starting positions: Robot 1 (-0.5, -0.5), Robot 2 (0.5, 0.5)
r1_qpos_adr = model.jnt_qposadr[
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot1_root")
]
r2_qpos_adr = model.jnt_qposadr[
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot2_root")
]
data.qpos[r1_qpos_adr : r1_qpos_adr + 2] = [-0.5, -0.5]
data.qpos[r2_qpos_adr : r2_qpos_adr + 2] = [0.5, 0.5]
mujoco.mj_forward(model, data)

marker_geom_ids = {
    0: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "robot1_aruco_plate"),
    1: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "robot2_aruco_plate"),
}

# -----------------------------------------------------------------------------
# 2. Camera Setup (720p Native 16:9)
# -----------------------------------------------------------------------------
cam_names = ["cam_south", "cam_north"]
cam_ids = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    for name in cam_names
]

cam_w, cam_h = 1280, 720
renderer = mujoco.Renderer(model, height=cam_h, width=cam_w)

cam_matrices = []
for cid in cam_ids:
    fovy_rad = np.deg2rad(model.cam_fovy[cid])
    f = (cam_h / 2.0) / np.tan(fovy_rad / 2.0)
    K = np.array(
        [[f, 0, cam_w / 2.0], [0, f, cam_h / 2.0], [0, 0, 1.0]],
        dtype=np.float64,
    )
    cam_matrices.append(K)

# -----------------------------------------------------------------------------
# 3. ArUco Detector Configuration
# -----------------------------------------------------------------------------
params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
params.cornerRefinementWinSize = 5
params.cornerRefinementMaxIterations = 30
params.cornerRefinementMinAccuracy = 0.05

params.adaptiveThreshWinSizeMin = 3
params.adaptiveThreshWinSizeMax = 25
params.adaptiveThreshWinSizeStep = 4

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector = cv2.aruco.ArucoDetector(dictionary, params)

# Setup output directory for automated blind spot dumps
dump_dir = "blindspot_dumps"
os.makedirs(dump_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# 4. Interactive Diagnostic Workbench UI
# -----------------------------------------------------------------------------
window_name = "ArUco Dropout Diagnostic Workbench"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1300, 750)

focused_robot = 0
paused = False
last_auto_dump_time = {0: 0.0, 1: 0.0}
DUMP_COOLDOWN_SEC = 2.0  # Avoid writing 50 screenshots per second during a blind spot

# MuJoCo camera coordinate adapter: +X right, +Y down, +Z forward
R_opt = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)

print("=" * 70)
print(" ARUCO BLIND-SPOT DIAGNOSTIC WORKBENCH INITIALIZED")
print("=" * 70)
print(" Controls:")
print("   [SPACE] : Pause / Resume simulation")
print("   [0]     : Track / Focus Robot 0")
print("   [1]     : Track / Focus Robot 1")
print("   [s]     : Manually save current screen to blindspot_dumps/")
print("   [q/ESC] : Quit")
print(f" Automatic snapshots will save to ./{dump_dir}/ upon detection dropout.")
print("=" * 70)

try:
    while True:
        if not paused:
            # Step physics (differential circle trajectory)
            for _ in range(10):
                data.ctrl[0] = 0.02
                data.ctrl[1] = 0.035
                data.ctrl[2] = 0.03
                data.ctrl[3] = 0.02
                mujoco.mj_step(model, data)

        # Ground truth 3D position of focused robot's ArUco plate
        geom_id = marker_geom_ids[focused_robot]
        gt_plate_pos = data.geom_xpos[geom_id].copy()

        diagnostic_tiles = []
        cam_detection_flags = []
        cam_telemetry_records = []

        for idx, cid in enumerate(cam_ids):
            renderer.update_scene(data, camera=cid)
            frame_rgb = renderer.render()
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            cam_pos = data.cam_xpos[cid]
            cam_rot = data.cam_xmat[cid].reshape(3, 3)
            K = cam_matrices[idx]

            # 1. Project ground truth 3D position onto image plane
            rel_pos = gt_plate_pos - cam_pos
            p_cam = R_opt.T @ (cam_rot.T @ rel_pos)

            # Prevent division by zero if behind camera
            z_cam = p_cam[2] if abs(p_cam[2]) > 1e-4 else 1e-4
            u = int(K[0, 0] * (p_cam[0] / z_cam) + K[0, 2])
            v = int(K[1, 1] * (p_cam[1] / z_cam) + K[1, 2])

            dist_3d = np.linalg.norm(rel_pos)
            cos_inc = rel_pos[2] / (dist_3d if dist_3d > 1e-4 else 1.0)
            incidence_deg = np.rad2deg(np.arccos(np.clip(cos_inc, -1.0, 1.0)))

            # 2. Run detection with candidate tracking
            corners, ids, rejected = detector.detectMarkers(gray)
            detected_ids = [int(x) for x in ids.flatten()] if ids is not None else []
            is_detected = focused_robot in detected_ids
            cam_detection_flags.append(is_detected)

            # Record telemetry for logging
            cam_telemetry_records.append({
                "name": cam_names[idx],
                "detected": is_detected,
                "dist": dist_3d,
                "incidence": incidence_deg,
                "pixel_pos": (u, v),
                "rejected_count": len(rejected)
            })

            # Draw standard detected markers
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(bgr, corners, ids)

            # Draw rejected quad contours in RED to see what OpenCV threw away
            if len(rejected) > 0:
                for r_poly in rejected:
                    cv2.polylines(
                        bgr, [r_poly.astype(np.int32)], True, (0, 0, 255), 1
                    )

            # Draw ground truth center marker in CYAN
            cv2.drawMarker(
                bgr,
                (u, v),
                (255, 255, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=18,
                thickness=2,
            )

            # 3. Create high-magnification zoomed crop (160x160 px around GT center)
            crop_size = 80
            u1, u2 = max(0, u - crop_size), min(cam_w, u + crop_size)
            v1, v2 = max(0, v - crop_size), min(cam_h, v + crop_size)

            if (u2 - u1 > 10) and (v2 - v1 > 10):
                crop_bgr = bgr[v1:v2, u1:u2]
                crop_gray = gray[v1:v2, u1:u2]

                crop_bin = cv2.adaptiveThreshold(
                    crop_gray,
                    255,
                    cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY,
                    params.adaptiveThreshWinSizeMin + 4,
                    7,
                )
                crop_bin_bgr = cv2.cvtColor(crop_bin, cv2.COLOR_GRAY2BGR)
            else:
                # Out of bounds fallback
                crop_bgr = np.zeros((crop_size * 2, crop_size * 2, 3), dtype=np.uint8)
                crop_bin_bgr = crop_bgr.copy()

            zoom_bgr = cv2.resize(
                crop_bgr, (260, 260), interpolation=cv2.INTER_NEAREST
            )
            zoom_bin = cv2.resize(
                crop_bin_bgr, (260, 260), interpolation=cv2.INTER_NEAREST
            )

            # 4. Status Panel Overlay
            status_color = (0, 255, 0) if is_detected else (0, 0, 255)
            status_text = "DETECTED" if is_detected else "DROPOUT / BLIND SPOT"

            cv2.rectangle(bgr, (10, 10), (460, 125), (15, 15, 15), -1)
            cv2.putText(
                bgr,
                f"CAM: {cam_names[idx]} | Robot {focused_robot}: {status_text}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                status_color,
                2,
            )
            cv2.putText(
                bgr,
                f"Incidence Angle: {incidence_deg:.1f} deg (Safe: < 58 deg)",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255) if incidence_deg > 58 else (255, 255, 255),
                1,
            )
            cv2.putText(
                bgr,
                f"3D Distance: {dist_3d:.2f} m | Projected Center: ({u}, {v})",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                bgr,
                f"Rejected Quad Contours: {len(rejected)} (Red Lines)",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 165, 255),
                1,
            )

            # Assemble diagnostic column for this camera
            bgr_preview = cv2.resize(bgr, (640, 360))
            side_inspect = np.vstack([zoom_bgr, zoom_bin])
            side_inspect = cv2.resize(side_inspect, (260, 360))
            cam_composite = np.hstack([bgr_preview, side_inspect])
            diagnostic_tiles.append(cam_composite)

        # Combine both views vertically
        final_ui = np.vstack(diagnostic_tiles)

        # Global Footer
        footer_y = final_ui.shape[0] - 12
        status_bar = f"[SPACE]: {'RESUME' if paused else 'PAUSE'} | [0/1]: FOCUS R{focused_robot} | [s]: MANUAL SNAPSHOT | Red Lines: Discarded Quads"
        cv2.putText(
            final_ui,
            status_bar,
            (25, footer_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        # ---------------------------------------------------------------------
        # 5. Automated Blind-Spot Detection & Logging Trigger
        # ---------------------------------------------------------------------
        is_blind_spot = not any(cam_detection_flags)
        curr_time = time.time()

        if is_blind_spot and (curr_time - last_auto_dump_time[focused_robot] > DUMP_COOLDOWN_SEC):
            last_auto_dump_time[focused_robot] = curr_time
            file_stamp = f"blindspot_R{focused_robot}_t{int(data.time * 100):05d}"
            save_path = os.path.join(dump_dir, f"{file_stamp}.png")
            cv2.imwrite(save_path, final_ui)

            print("\n" + "!" * 70)
            print(f"[BLIND SPOT TRIGGERED] Robot {focused_robot} lost by BOTH cameras!")
            print(f"  Simulation Time : {data.time:.3f} s")
            print(f"  Ground Truth 3D : ({gt_plate_pos[0]:.3f}, {gt_plate_pos[1]:.3f}, {gt_plate_pos[2]:.3f}) m")
            for rec in cam_telemetry_records:
                print(
                    f"  [{rec['name']}] Dist: {rec['dist']:.2f}m | "
                    f"Incidence: {rec['incidence']:.1f}deg | "
                    f"Pixel: {rec['pixel_pos']} | "
                    f"Rejected Quads: {rec['rejected_count']}"
                )
            print(f"  --> Saved full workbench screenshot to: {save_path}")
            print("!" * 70 + "\n")

        cv2.imshow(window_name, final_ui)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord("q"), 27]:
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("0"):
            focused_robot = 0
            print("[STATUS] Focus switched to Robot 0")
        elif key == ord("1"):
            focused_robot = 1
            print("[STATUS] Focus switched to Robot 1")
        elif key == ord("s"):
            manual_file = os.path.join(dump_dir, f"manual_snap_R{focused_robot}_{int(time.time())}.png")
            cv2.imwrite(manual_file, final_ui)
            print(f"[SAVED] Manual snapshot saved to: {manual_file}")

finally:
    cv2.destroyAllWindows()