import cv2
import mujoco
import numpy as np

# 1. Load model and simulation state
model = mujoco.MjModel.from_xml_path("robots12_arena_2cam.xml")
data = mujoco.MjData(model)

# Initial placements: Robot 1 at (-0.5, -0.5), Robot 2 at (0.5, 0.5)
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

# 2. Dual Slanted Camera Setup (Native 720p 16:9)
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

dist = np.zeros((4, 1), dtype=np.float64)

# 3. ArUco Marker Metric Setup (0.064m inner black pattern)
marker_len = 0.08 * (400.0 / 500.0)  # 0.064 m
obj_pts = np.array(
    [
        [-marker_len / 2, marker_len / 2, 0],
        [marker_len / 2, marker_len / 2, 0],
        [marker_len / 2, -marker_len / 2, 0],
        [-marker_len / 2, -marker_len / 2, 0],
    ],
    dtype=np.float32,
)

# Robust DetectorParameters specifically tuned for slanted perspectives
params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
params.cornerRefinementWinSize = 5
params.cornerRefinementMaxIterations = 30
params.cornerRefinementMinAccuracy = 0.05

# Finer adaptive threshold window steps to resolve foreshortened bit patterns
params.adaptiveThreshWinSizeMin = 3
params.adaptiveThreshWinSizeMax = 25
params.adaptiveThreshWinSizeStep = 4

# Tolerances for non-orthogonal quadrilateral extraction
params.polygonalApproxAccuracyRate = 0.05
params.minMarkerPerimeterRate = 0.02
params.perspectiveRemovePixelPerCell = 8
params.perspectiveRemoveIgnoredMarginPerCell = 0.13

detector = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params
)

# MuJoCo camera coordinate adapter: +X right, +Y down, +Z forward
R_opt = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)

# 4. Display Window & Pre-allocated Canvas Setup
disp_w, disp_h = 640, 360
window_name = "BotArena: Dual Slanted Multi-View Tracker"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, disp_w * 2, disp_h + 40)

combined_display = np.zeros((disp_h + 40, disp_w * 2, 3), dtype=np.uint8)

physics_steps_per_frame = 10
frame_count = 0

print("Tracker running. Press 'q' or ESC to exit.")

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

        # Ground truth positions at exact render instant
        gt_positions = {
            i: data.geom_xpos[marker_geom_ids[i]][:2].copy() for i in (0, 1)
        }

        multi_obs = {0: [], 1: []}

        # Process each slanted camera view
        for idx, cid in enumerate(cam_ids):
            renderer.update_scene(data, camera=cid)
            frame_rgb = renderer.render()

            # Proper luminance-preserving grayscale conversion
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            corners, ids, _ = detector.detectMarkers(gray)

            cam_pos = data.cam_xpos[cid]
            cam_rot = data.cam_xmat[cid].reshape(3, 3)

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(bgr, corners, ids)

                for i, mid in enumerate(ids.flatten()):
                    mid = int(mid)
                    if mid not in marker_geom_ids:
                        continue

                    success, rvec, tvec = cv2.solvePnP(
                        obj_pts,
                        corners[i],
                        cam_matrices[idx],
                        dist,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE,
                    )

                    if success:
                        cv2.drawFrameAxes(
                            bgr, cam_matrices[idx], dist, rvec, tvec, 0.05, 2
                        )

                        # World coordinate transformation
                        p_world = cam_pos + cam_rot @ (R_opt @ tvec.flatten())

                        # Heading extraction
                        R_m, _ = cv2.Rodrigues(rvec)
                        R_world = cam_rot @ (R_opt @ R_m)
                        yaw_deg = np.rad2deg(
                            np.arctan2(R_world[1, 0], R_world[0, 0])
                        )
                        dist_to_cam = float(np.linalg.norm(tvec))

                        multi_obs[mid].append(
                            (p_world[:2], yaw_deg, dist_to_cam)
                        )

                        c = corners[i][0][0].astype(int)
                        cv2.putText(
                            bgr,
                            f"R{mid} d={dist_to_cam:.2f}m",
                            (c[0] - 25, c[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2,
                        )

            # Camera header
            cv2.rectangle(bgr, (10, 10), (320, 50), (20, 20, 20), -1)
            cv2.putText(
                bgr,
                f"Camera: {cam_names[idx]}",
                (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            # Downsample for window display
            bgr_small = cv2.resize(bgr, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
            combined_display[0:disp_h, idx * disp_w : (idx + 1) * disp_w] = bgr_small

        # Multi-View Inverse Distance Squared Fusion: w_i = 1 / d_i^2
        fused_results = {}
        for mid in (0, 1):
            obs = multi_obs[mid]
            if len(obs) == 0:
                continue
            elif len(obs) == 1:
                fused_results[mid] = {
                    "xy": obs[0][0],
                    "yaw": obs[0][1],
                    "cams": 1,
                }
            else:
                w0 = 1.0 / (obs[0][2] ** 2)
                w1 = 1.0 / (obs[1][2] ** 2)
                fused_xy = (w0 * obs[0][0] + w1 * obs[1][0]) / (w0 + w1)

                sin_yaw = (
                    w0 * np.sin(np.deg2rad(obs[0][1]))
                    + w1 * np.sin(np.deg2rad(obs[1][1]))
                ) / (w0 + w1)
                cos_yaw = (
                    w0 * np.cos(np.deg2rad(obs[0][1]))
                    + w1 * np.cos(np.deg2rad(obs[1][1]))
                ) / (w0 + w1)
                fused_yaw = np.rad2deg(np.arctan2(sin_yaw, cos_yaw))

                fused_results[mid] = {
                    "xy": fused_xy,
                    "yaw": fused_yaw,
                    "cams": 2,
                }

        # Clear HUD bar
        combined_display[disp_h:, :] = 20

        for mid in (0, 1):
            if mid in fused_results:
                est = fused_results[mid]
                gt = gt_positions[mid]
                err_mm = np.linalg.norm(est["xy"] - gt) * 1000.0

                color = (0, 255, 0) if err_mm < 25 else (0, 165, 255)
                text = (
                    f"R{mid} [{est['cams']} Cams]: "
                    f"({est['xy'][0]:>5.2f}, {est['xy'][1]:>5.2f})m | "
                    f"Yaw:{est['yaw']:>5.1f}d | "
                    f"Err:{err_mm:>4.1f}mm"
                )
            else:
                color = (0, 0, 255)
                text = f"Robot {mid}: NO DETECTION (Blind Spot)"

            cv2.putText(
                combined_display,
                text,
                (20 + mid * disp_w, disp_h + 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                1,
            )

        cv2.imshow(window_name, combined_display)
        if (cv2.waitKey(1) & 0xFF) in [ord("q"), 27]:
            break

finally:
    cv2.destroyAllWindows()