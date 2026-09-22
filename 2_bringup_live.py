import cv2
import mujoco
import numpy as np

# 1. Load model and simulation state
model = mujoco.MjModel.from_xml_path("robots12_arena.xml")
data = mujoco.MjData(model)

# Initial placement: Robot 1 (-0.5, -0.5), Robot 2 (0.5, 0.5)
r1_qpos_adr = model.jnt_qposadr[
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot1_root")
]
r2_qpos_adr = model.jnt_qposadr[
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot2_root")
]
data.qpos[r1_qpos_adr : r1_qpos_adr + 2] = [-0.5, -0.5]
data.qpos[r2_qpos_adr : r2_qpos_adr + 2] = [0.5, 0.5]
mujoco.mj_forward(model, data)

# 2. Camera setup
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_cam")
width, height = 1280, 720
renderer = mujoco.Renderer(model, height=height, width=width)

fovy_rad = np.deg2rad(model.cam_fovy[cam_id])
f = (height / 2.0) / np.tan(fovy_rad / 2.0)
K = np.array(
    [[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]], dtype=np.float64
)
dist = np.zeros((4, 1), dtype=np.float64)

# 3. ArUco Marker Setup (0.064m inner pattern)
marker_len = 0.08 * (400.0 / 500.0)
obj_pts = np.array(
    [
        [-marker_len / 2, marker_len / 2, 0],
        [marker_len / 2, marker_len / 2, 0],
        [marker_len / 2, -marker_len / 2, 0],
        [-marker_len / 2, -marker_len / 2, 0],
    ],
    dtype=np.float32,
)

params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params
)
R_opt = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)

# Exact marker geoms for zero-lever-arm ground truth comparison
marker_geom_ids = {
    0: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "robot1_aruco_plate"),
    1: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "robot2_aruco_plate"),
}

window_name = "BotArena Overhead View"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 960, 540)

physics_steps_per_frame = 10
frame_count = 0

try:
    while True:
        # Step physics forward
        for _ in range(physics_steps_per_frame):
            data.ctrl[0] = 0.02
            data.ctrl[1] = 0.035
            data.ctrl[2] = 0.03
            data.ctrl[3] = 0.02
            mujoco.mj_step(model, data)

        frame_count += 1

        # Render frame
        renderer.update_scene(data, camera=cam_id)
        gt_positions = {
            i: data.geom_xpos[marker_geom_ids[i]][:2].copy() for i in (0, 1)
        }

        frame_rgb = renderer.render()
        display_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

        # Detect markers
        corners, ids, _ = detector.detectMarkers(gray)

        # Top HUD Dashboard background
        cv2.rectangle(display_frame, (10, 10), (520, 110), (20, 20, 20), -1)
        cv2.rectangle(display_frame, (10, 10), (520, 110), (100, 100, 100), 1)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(display_frame, corners, ids)
            cam_pos = data.cam_xpos[cam_id]
            cam_rot = data.cam_xmat[cam_id].reshape(3, 3)

            for i, mid in enumerate(ids.flatten()):
                mid = int(mid)
                if mid not in marker_geom_ids:
                    continue

                success, rvec, tvec = cv2.solvePnP(
                    obj_pts,
                    corners[i],
                    K,
                    dist,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )

                if success:
                    # Draw 3D coordinate frame (5cm length)
                    cv2.drawFrameAxes(
                        display_frame, K, dist, rvec, tvec, 0.05, 2
                    )

                    # Transform estimated pose to world frame
                    p_world = cam_pos + cam_rot @ (R_opt @ tvec.flatten())
                    R_m, _ = cv2.Rodrigues(rvec)
                    R_world = cam_rot @ (R_opt @ R_m)
                    yaw_deg = np.rad2deg(
                        np.arctan2(R_world[1, 0], R_world[0, 0])
                    )

                    # Calculate ground truth error
                    gt_xy = gt_positions[mid]
                    error_mm = np.linalg.norm(p_world[:2] - gt_xy) * 1000.0

                    # 1. Overlay telemetry text next to the robot
                    c = corners[i][0][0].astype(int)
                    label_robot = f"R{mid}: ({p_world[0]:.2f}, {p_world[1]:.2f})m | {yaw_deg:.0f}deg"
                    err_robot = f"Err: {error_mm:.1f} mm"

                    cv2.putText(
                        display_frame,
                        label_robot,
                        (c[0] - 40, c[1] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2,
                    )
                    cv2.putText(
                        display_frame,
                        err_robot,
                        (c[0] - 40, c[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255) if error_mm > 25 else (0, 255, 0),
                        2,
                    )

                    # 2. Write to top HUD summary box
                    hud_y = 45 + mid * 35
                    hud_text = (
                        f"R{mid} Est: ({p_world[0]:>5.2f}, {p_world[1]:>5.2f})m | "
                        f"Err: {error_mm:>4.1f}mm"
                    )
                    cv2.putText(
                        display_frame,
                        hud_text,
                        (20, hud_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        1,
                    )

                    # Periodic console logging
                    if frame_count % 30 == 0:
                        print(
                            f"[Time {data.time:6.2f}s] Robot {mid} | "
                            f"Est: (X={p_world[0]:6.3f}, Y={p_world[1]:6.3f}, Yaw={yaw_deg:5.1f}deg) | "
                            f"GT: (X={gt_xy[0]:6.3f}, Y={gt_xy[1]:6.3f}) | "
                            f"Err: {error_mm:5.2f} mm"
                        )

        cv2.imshow(window_name, display_frame)
        if (cv2.waitKey(1) & 0xFF) in [ord("q"), 27]:
            break

finally:
    cv2.destroyAllWindows()