import collections
import cv2
import matplotlib.pyplot as plt
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

# 2. Camera Intrinsics
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_cam")
width, height = 1280, 720
renderer = mujoco.Renderer(model, height=height, width=width)

fovy_rad = np.deg2rad(model.cam_fovy[cam_id])
f = (height / 2.0) / np.tan(fovy_rad / 2.0)
K = np.array(
    [[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]], dtype=np.float64
)
dist = np.zeros((4, 1), dtype=np.float64)

# 3. ArUco Marker Metric Setup (0.064m effective pattern)
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

detector = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
    cv2.aruco.DetectorParameters(),
)
R_opt = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)

body_ids = {
    0: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot1"),
    1: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot2"),
}

# 4. Matplotlib Setup
plt.ion()
fig, (ax_traj, ax_err) = plt.subplots(1, 2, figsize=(11, 5))
fig.canvas.manager.set_window_title("BotArena: Pose Estimation Performance")

# Left Plot: Trajectory
ax_traj.set_xlim(-1.1, 1.1)
ax_traj.set_ylim(-1.1, 1.1)
ax_traj.set_aspect("equal")
ax_traj.set_title("2D Arena Trajectory (m)")
ax_traj.set_xlabel("X (m)")
ax_traj.set_ylabel("Y (m)")
ax_traj.grid(True, linestyle="--", alpha=0.5)

(line_gt_r0,) = ax_traj.plot([], [], "b-", lw=1.8, label="R0 Ground Truth")
(line_est_r0,) = ax_traj.plot([], [], "c--", lw=1.2, label="R0 Estimated")
(line_gt_r1,) = ax_traj.plot([], [], "m-", lw=1.8, label="R1 Ground Truth")
(line_est_r1,) = ax_traj.plot([], [], "y--", lw=1.2, label="R1 Estimated")
ax_traj.legend(loc="upper right", fontsize=8)

# Right Plot: Error (Y-axis scaled to 100 mm)
ax_err.set_xlim(0, 10)
ax_err.set_ylim(0, 100)
ax_err.set_title("Absolute Tracking Error")
ax_err.set_xlabel("Time (s)")
ax_err.set_ylabel("Error (mm)")
ax_err.grid(True, linestyle="--", alpha=0.5)

(line_err_r0,) = ax_err.plot([], [], "c-", label="R0 Error")
(line_err_r1,) = ax_err.plot([], [], "m-", label="R1 Error")
ax_err.legend(loc="upper right", fontsize=8)

# Independent Buffers (keeps last 300 samples per robot)
BUF_LEN = 300
traj = {
    0: {
        "times": collections.deque(maxlen=BUF_LEN),
        "gt_x": collections.deque(maxlen=BUF_LEN),
        "gt_y": collections.deque(maxlen=BUF_LEN),
        "est_x": collections.deque(maxlen=BUF_LEN),
        "est_y": collections.deque(maxlen=BUF_LEN),
        "err": collections.deque(maxlen=BUF_LEN),
    },
    1: {
        "times": collections.deque(maxlen=BUF_LEN),
        "gt_x": collections.deque(maxlen=BUF_LEN),
        "gt_y": collections.deque(maxlen=BUF_LEN),
        "est_x": collections.deque(maxlen=BUF_LEN),
        "est_y": collections.deque(maxlen=BUF_LEN),
        "err": collections.deque(maxlen=BUF_LEN),
    },
}

# 5. Live Simulation Loop
window_name = "Overhead Camera View"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 640, 360)

step_count = 0
physics_steps_per_frame = 10

try:
    while True:
        # Step physics forward
        for _ in range(physics_steps_per_frame):
            data.ctrl[0] = 0.02
            data.ctrl[1] = 0.035
            data.ctrl[2] = 0.03
            data.ctrl[3] = 0.02
            mujoco.mj_step(model, data)

        step_count += 1
        current_time = data.time

        # Render Overhead Frame
        renderer.update_scene(data, camera=cam_id)
        frame_rgb = renderer.render()
        display_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

        # Detect ArUco Markers
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(display_frame, corners, ids)
            cam_pos = data.cam_xpos[cam_id]
            cam_rot = data.cam_xmat[cam_id].reshape(3, 3)

            for i, mid in enumerate(ids.flatten()):
                mid = int(mid)
                if mid not in traj:
                    continue

                success, rvec, tvec = cv2.solvePnP(
                    obj_pts,
                    corners[i],
                    K,
                    dist,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )

                if success:
                    cv2.drawFrameAxes(
                        display_frame, K, dist, rvec, tvec, 0.05, 2
                    )

                    # Transform estimated pose to world frame
                    p_world = cam_pos + cam_rot @ (R_opt @ tvec.flatten())
                    est_xy = p_world[:2]
                    gt_xy = data.xpos[body_ids[mid]][:2]

                    error_mm = np.linalg.norm(est_xy - gt_xy) * 1000.0

                    # Append to this robot's specific synchronized history
                    traj[mid]["times"].append(current_time)
                    traj[mid]["est_x"].append(est_xy[0])
                    traj[mid]["est_y"].append(est_xy[1])
                    traj[mid]["gt_x"].append(gt_xy[0])
                    traj[mid]["gt_y"].append(gt_xy[1])
                    traj[mid]["err"].append(error_mm)

        # Update Plots (every 5 visual frames)
        if step_count % 5 == 0:
            if len(traj[0]["times"]) > 1:
                line_gt_r0.set_data(traj[0]["gt_x"], traj[0]["gt_y"])
                line_est_r0.set_data(traj[0]["est_x"], traj[0]["est_y"])
                line_err_r0.set_data(traj[0]["times"], traj[0]["err"])

            if len(traj[1]["times"]) > 1:
                line_gt_r1.set_data(traj[1]["gt_x"], traj[1]["gt_y"])
                line_est_r1.set_data(traj[1]["est_x"], traj[1]["est_y"])
                line_err_r1.set_data(traj[1]["times"], traj[1]["err"])

            # Scroll time axis
            ax_err.set_xlim(max(0, current_time - 10), current_time + 0.5)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

        # Display window
        cv2.imshow(window_name, display_frame)
        if (cv2.waitKey(1) & 0xFF) in [ord("q"), 27]:
            break

finally:
    plt.ioff()
    plt.show()
    cv2.destroyAllWindows()