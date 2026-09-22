import collections
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
# 2. SIMULATION & SYSTEM INITIALIZATION
# ==============================================================================
model = mujoco.MjModel.from_xml_path("robots12_arena.xml")
data = mujoco.MjData(model)

# Set initial positions: R0 at (-0.5, -0.5), R1 at (0.5, 0.5)
r1_qpos_adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot1_root")]
r2_qpos_adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot2_root")]
data.qpos[r1_qpos_adr : r1_qpos_adr + 2] = [-0.5, -0.5]
data.qpos[r2_qpos_adr : r2_qpos_adr + 2] = [0.5, 0.5]
mujoco.mj_forward(model, data)

# Query wheel joint dof indices
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

# Ground truth geom targets (marker top surfaces)
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

# Initialize EKFs
ekfs = {
    0: RobotEKF2D(init_state=[-0.5, -0.5, 0.0]),
    1: RobotEKF2D(init_state=[ 0.5,  0.5, 0.0])
}

# ==============================================================================
# 3. MATPLOTLIB REAL-TIME DASHBOARD
# ==============================================================================
plt.ion()
fig, (ax_traj, ax_err) = plt.subplots(1, 2, figsize=(12, 5))
fig.canvas.manager.set_window_title("BotArena: Ground Truth vs. EKF Tracking")

# 2D Arena Trajectory
ax_traj.set_xlim(-1.1, 1.1)
ax_traj.set_ylim(-1.1, 1.1)
ax_traj.set_aspect("equal")
ax_traj.set_title("2D Arena Map (m)")
ax_traj.set_xlabel("X (m)")
ax_traj.set_ylabel("Y (m)")
ax_traj.grid(True, linestyle="--", alpha=0.5)

(line_gt_r0,)  = ax_traj.plot([], [], "b-",  lw=2.0, label="R0 Ground Truth")
(line_ekf_r0,) = ax_traj.plot([], [], "c--", lw=1.5, label="R0 EKF Fused")
(line_gt_r1,)  = ax_traj.plot([], [], "m-",  lw=2.0, label="R1 Ground Truth")
(line_ekf_r1,) = ax_traj.plot([], [], "y--", lw=1.5, label="R1 EKF Fused")
ax_traj.legend(loc="upper right", fontsize=8)

# Tracking Error Plot
ax_err.set_xlim(0, 10)
ax_err.set_ylim(0, 100)  # 0 to 100 mm range
ax_err.set_title("Absolute Tracking Error (mm)")
ax_err.set_xlabel("Time (s)")
ax_err.set_ylabel("Error (mm)")
ax_err.grid(True, linestyle="--", alpha=0.5)

(line_err_r0,) = ax_err.plot([], [], "c-", lw=1.5, label="R0 EKF Error")
(line_err_r1,) = ax_err.plot([], [], "m-", lw=1.5, label="R1 EKF Error")
ax_err.legend(loc="upper right", fontsize=8)

# Rolling data buffers
BUF_LEN = 300
traj = {
    i: {
        "times": collections.deque(maxlen=BUF_LEN),
        "gt_x":  collections.deque(maxlen=BUF_LEN),
        "gt_y":  collections.deque(maxlen=BUF_LEN),
        "ekf_x": collections.deque(maxlen=BUF_LEN),
        "ekf_y": collections.deque(maxlen=BUF_LEN),
        "err":   collections.deque(maxlen=BUF_LEN)
    } for i in (0, 1)
}

# ==============================================================================
# 4. MAIN INTERACTIVE EXECUTION LOOP
# ==============================================================================
window_name = "BotArena Live Overhead View"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 640, 360)

physics_dt = model.opt.timestep        # 0.002s (500 Hz)
substeps_per_frame = 10               # 10 steps = 0.02s (50 Hz camera frame)
step_count = 0

try:
    while True:
        # --- A. PHYSICS & EKF PREDICTION (500 Hz) ---
        for _ in range(substeps_per_frame):
            # Actuator drive torques
            data.ctrl[0] = 0.02   # R0 left motor
            data.ctrl[1] = 0.035  # R0 right motor (circles counter-clockwise)
            data.ctrl[2] = 0.03   # R1 left motor
            data.ctrl[3] = 0.02   # R1 right motor (circles clockwise)
            mujoco.mj_step(model, data)

            # High-frequency wheel odometry integration
            for i in (0, 1):
                w_l = data.qvel[wheel_dofs[i]["l"]]
                w_r = data.qvel[wheel_dofs[i]["r"]]
                ekfs[i].predict(omega_left=w_l, omega_right=w_r, dt=physics_dt)

        step_count += 1
        current_time = data.time

        # --- B. CAMERA RENDER & EKF CORRECTION (50 Hz) ---
        renderer.update_scene(data, camera=cam_id)
        
        # Synchronous ground truth capture at the exact render instant
        gt_positions = {
            i: data.geom_xpos[marker_geom_ids[i]][:2].copy() for i in (0, 1)
        }

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

                    # Compute World Frame Pose
                    p_world = cam_pos + cam_rot @ (R_opt @ tvec.flatten())
                    R_m, _ = cv2.Rodrigues(rvec)
                    R_world = cam_rot @ (R_opt @ R_m)
                    yaw_measured = np.arctan2(R_world[1, 0], R_world[0, 0])

                    # EKF Measurement Update
                    ekfs[mid].update([p_world[0], p_world[1], yaw_measured])

        # Record synchronized metrics for both robots
        for i in (0, 1):
            gt_xy = gt_positions[i]
            ekf_xy = ekfs[i].state[:2]
            error_mm = np.linalg.norm(ekf_xy - gt_xy) * 1000.0

            traj[i]["times"].append(current_time)
            traj[i]["gt_x"].append(gt_xy[0])
            traj[i]["gt_y"].append(gt_xy[1])
            traj[i]["ekf_x"].append(ekf_xy[0])
            traj[i]["ekf_y"].append(ekf_xy[1])
            traj[i]["err"].append(error_mm)

            # Live text overlay on OpenCV feed
            label = f"R{i} Err: {error_mm:.1f}mm"
            px = 30 + i * 220
            cv2.putText(display_frame, label, (px, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        # --- C. REFRESH MATPLOTLIB DASHBOARD (~10 Hz) ---
        if step_count % 5 == 0:
            if len(traj[0]["times"]) > 1:
                line_gt_r0.set_data(traj[0]["gt_x"], traj[0]["gt_y"])
                line_ekf_r0.set_data(traj[0]["ekf_x"], traj[0]["ekf_y"])
                line_err_r0.set_data(traj[0]["times"], traj[0]["err"])

            if len(traj[1]["times"]) > 1:
                line_gt_r1.set_data(traj[1]["gt_x"], traj[1]["gt_y"])
                line_ekf_r1.set_data(traj[1]["ekf_x"], traj[1]["ekf_y"])
                line_err_r1.set_data(traj[1]["times"], traj[1]["err"])

            ax_err.set_xlim(max(0, current_time - 10), current_time + 0.5)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

        # Display window (press 'q' or ESC to exit)
        cv2.imshow(window_name, display_frame)
        if (cv2.waitKey(1) & 0xFF) in [ord("q"), 27]:
            break

finally:
    plt.ioff()
    plt.show()
    cv2.destroyAllWindows()