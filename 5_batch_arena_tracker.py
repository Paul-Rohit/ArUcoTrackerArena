import cv2
import matplotlib.pyplot as plt
import mujoco
import numpy as np

# ==============================================================================
# 1. 2D EXTENDED KALMAN FILTER
# ==============================================================================
def wrap_angle(angle):
    """Normalize angle to [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class RobotEKF2D:
    def __init__(self, init_state, wheel_radius=0.031, track_width=0.105):
        self.state = np.array(init_state, dtype=np.float64)  # [x, y, theta]
        self.r = wheel_radius
        self.L = track_width

        self.P = np.diag([0.01, 0.01, 0.05])
        self.Q = np.diag([1e-4, 1e-4, 4e-4])          # Process noise (wheel slip)
        self.R = np.diag([2e-5, 2e-5, 1e-3])          # Camera measurement noise

    def predict(self, omega_left, omega_right, dt):
        v = (self.r / 2.0) * (omega_right + omega_left)
        omega = (self.r / self.L) * (omega_right - omega_left)
        theta = self.state[2]

        self.state[0] += v * dt * np.cos(theta)
        self.state[1] += v * dt * np.sin(theta)
        self.state[2] = wrap_angle(self.state[2] + omega * dt)

        F = np.array([
            [1.0, 0.0, -v * dt * np.sin(theta)],
            [0.0, 1.0,  v * dt * np.cos(theta)],
            [0.0, 0.0,  1.0]
        ])
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        y = np.array(z, dtype=np.float64) - self.state
        y[2] = wrap_angle(y[2])

        S = self.P + self.R
        K = self.P @ np.linalg.inv(S)

        self.state += K @ y
        self.state[2] = wrap_angle(self.state[2])
        self.P = (np.eye(3) - K) @ self.P


# ==============================================================================
# 2. SIMULATION SETUP
# ==============================================================================
model = mujoco.MjModel.from_xml_path("robots12_arena.xml")
data = mujoco.MjData(model)

# Set starting positions: R0 at (-0.5, -0.5), R1 at (0.5, 0.5)
r1_qpos_adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot1_root")]
r2_qpos_adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot2_root")]
data.qpos[r1_qpos_adr : r1_qpos_adr + 2] = [-0.5, -0.5]
data.qpos[r2_qpos_adr : r2_qpos_adr + 2] = [0.5, 0.5]
mujoco.mj_forward(model, data)

# Wheel joint DOF indices
wheel_dofs = {
    0: {
        "l": model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot1_left_wheel_joint")],
        "r": model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot1_right_wheel_joint")]
    },
    1: {
        "l": model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot2_left_wheel_joint")],
        "r": model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot2_right_wheel_joint")]
    }
}

marker_geom_ids = {
    0: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "robot1_aruco_plate"),
    1: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "robot2_aruco_plate")
}

# Overhead camera
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_cam")
width, height = 1280, 720
renderer = mujoco.Renderer(model, height=height, width=width)

fovy_rad = np.deg2rad(model.cam_fovy[cam_id])
f = (height / 2.0) / np.tan(fovy_rad / 2.0)
K = np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]], dtype=np.float64)
dist = np.zeros((4, 1), dtype=np.float64)

# ArUco parameters
marker_len = 0.08 * (400.0 / 500.0)  # 0.064m
obj_pts = np.array([
    [-marker_len / 2,  marker_len / 2, 0],
    [ marker_len / 2,  marker_len / 2, 0],
    [ marker_len / 2, -marker_len / 2, 0],
    [-marker_len / 2, -marker_len / 2, 0]
], dtype=np.float32)

params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params)
R_opt = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)

ekfs = {
    0: RobotEKF2D(init_state=[-0.5, -0.5, 0.0]),
    1: RobotEKF2D(init_state=[ 0.5,  0.5, 0.0])
}

# Complete Run Telemetry Buffers (unbounded lists)
history = {
    0: {"time": [], "gt_x": [], "gt_y": [], "ekf_x": [], "ekf_y": [], "raw_x": [], "raw_y": [], "raw_t": [], "err": []},
    1: {"time": [], "gt_x": [], "gt_y": [], "ekf_x": [], "ekf_y": [], "raw_x": [], "raw_y": [], "raw_t": [], "err": []}
}

# ==============================================================================
# 3. LIVE SIMULATION LOOP (FULL SPEED)
# ==============================================================================
window_name = "BotArena (Press 'q' or ESC to stop & view plot)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 960, 540)

physics_dt = model.opt.timestep
substeps_per_frame = 10

print("Simulation started. Press 'q' or ESC in the OpenCV window to finish and view the analysis.")

try:
    while True:
        # A. Physics & EKF Prediction (500 Hz)
        for _ in range(substeps_per_frame):
            data.ctrl[0] = 0.02   # R0 left motor
            data.ctrl[1] = 0.035  # R0 right motor
            data.ctrl[2] = 0.03   # R1 left motor
            data.ctrl[3] = 0.02   # R1 right motor
            mujoco.mj_step(model, data)

            for i in (0, 1):
                w_l = data.qvel[wheel_dofs[i]["l"]]
                w_r = data.qvel[wheel_dofs[i]["r"]]
                ekfs[i].predict(omega_left=w_l, omega_right=w_r, dt=physics_dt)

        current_time = data.time

        # B. Camera Render & ArUco Detection (50 Hz)
        renderer.update_scene(data, camera=cam_id)
        gt_positions = {i: data.geom_xpos[marker_geom_ids[i]][:2].copy() for i in (0, 1)}

        frame_rgb = renderer.render()
        display_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(display_frame, corners, ids)
            cam_pos = data.cam_xpos[cam_id]
            cam_rot = data.cam_xmat[cam_id].reshape(3, 3)

            for i, mid in enumerate(ids.flatten()):
                mid = int(mid)
                if mid not in ekfs:
                    continue

                success, rvec, tvec = cv2.solvePnP(
                    obj_pts, corners[i], K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
                )

                if success:
                    cv2.drawFrameAxes(display_frame, K, dist, rvec, tvec, 0.05, 2)

                    p_world = cam_pos + cam_rot @ (R_opt @ tvec.flatten())
                    R_m, _ = cv2.Rodrigues(rvec)
                    R_world = cam_rot @ (R_opt @ R_m)
                    yaw_measured = np.arctan2(R_world[1, 0], R_world[0, 0])

                    ekfs[mid].update([p_world[0], p_world[1], yaw_measured])

                    # Log raw vision measurements
                    history[mid]["raw_x"].append(p_world[0])
                    history[mid]["raw_y"].append(p_world[1])
                    history[mid]["raw_t"].append(current_time)

        # Log synchronized states
        for i in (0, 1):
            gt_xy = gt_positions[i]
            ekf_xy = ekfs[i].state[:2]
            err_mm = np.linalg.norm(ekf_xy - gt_xy) * 1000.0

            history[i]["time"].append(current_time)
            history[i]["gt_x"].append(gt_xy[0])
            history[i]["gt_y"].append(gt_xy[1])
            history[i]["ekf_x"].append(ekf_xy[0])
            history[i]["ekf_y"].append(ekf_xy[1])
            history[i]["err"].append(err_mm)

            cv2.putText(
                display_frame,
                f"R{i} Err: {err_mm:.1f}mm",
                (30 + i * 220, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

        cv2.imshow(window_name, display_frame)
        if (cv2.waitKey(1) & 0xFF) in [ord("q"), 27]:
            break

finally:
    cv2.destroyAllWindows()
    print("Simulation closed. Preparing post-run summary plot...")

# ==============================================================================
# 4. POST-RUN MATPLOTLIB REPORT
# ==============================================================================
fig, (ax_traj, ax_err) = plt.subplots(1, 2, figsize=(14, 6))
fig.canvas.manager.set_window_title("BotArena Post-Run Trajectory & Error Analysis")

# --- Left Subplot: Full 2D Trajectories ---
ax_traj.set_xlim(-1.1, 1.1)
ax_traj.set_ylim(-1.1, 1.1)
ax_traj.set_aspect("equal")
ax_traj.set_title("Full Arena Trajectories (m)")
ax_traj.set_xlabel("X (m)")
ax_traj.set_ylabel("Y (m)")
ax_traj.grid(True, linestyle="--", alpha=0.6)

# Robot 0
ax_traj.plot(history[0]["gt_x"], history[0]["gt_y"], "b-", lw=2.2, label="R0 Ground Truth")
ax_traj.plot(history[0]["ekf_x"], history[0]["ekf_y"], "g--", lw=1.6, label="R0 EKF Fused")
ax_traj.scatter(history[0]["raw_x"], history[0]["raw_y"], c="cyan", s=8, alpha=0.3, label="R0 Raw ArUco")

# Robot 1
ax_traj.plot(history[1]["gt_x"], history[1]["gt_y"], "m-", lw=2.2, label="R1 Ground Truth")
ax_traj.plot(history[1]["ekf_x"], history[1]["ekf_y"], c="pink", lw=1.6, label="R1 EKF Fused")
ax_traj.scatter(history[1]["raw_x"], history[1]["raw_y"], c="orange", s=8, alpha=0.3, label="R1 Raw ArUco")

ax_traj.legend(loc="upper right", fontsize=8)

# --- Right Subplot: Error vs Time ---
ax_err.set_title("Absolute Tracking Error vs. Time")
ax_err.set_xlabel("Time (s)")
ax_err.set_ylabel("Error (mm)")
ax_err.grid(True, linestyle="--", alpha=0.6)

if history[0]["time"]:
    ax_err.plot(history[0]["time"], history[0]["err"], "c-", lw=1.4, label=f"R0 Error (Mean: {np.mean(history[0]['err']):.1f} mm)")
if history[1]["time"]:
    ax_err.plot(history[1]["time"], history[1]["err"], "m-", lw=1.4, label=f"R1 Error (Mean: {np.mean(history[1]['err']):.1f} mm)")

ax_err.set_ylim(0, 100)
ax_err.legend(loc="upper right", fontsize=9)

plt.tight_layout()
plt.show()