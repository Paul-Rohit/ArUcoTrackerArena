import cv2
import mujoco
import numpy as np
from ekf2d import RobotEKF2D, wrap_angle

# 1. Load simulation
model = mujoco.MjModel.from_xml_path("robots12_arena.xml")
data = mujoco.MjData(model)

# Place Robot 1
r1_qpos_adr = model.jnt_qposadr[
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot1_root")
]
data.qpos[r1_qpos_adr : r1_qpos_adr + 2] = [-0.5, -0.5]
mujoco.mj_forward(model, data)

# Joint indices for wheel velocity readout
lw_joint_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_JOINT, "robot1_left_wheel_joint"
)
rw_joint_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_JOINT, "robot1_right_wheel_joint"
)
lw_qvel_adr = model.jnt_dofadr[lw_joint_id]
rw_qvel_adr = model.jnt_dofadr[rw_joint_id]
marker_geom_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_GEOM, "robot1_aruco_plate"
)

# 2. Camera setup
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_cam")
width, height = 1280, 720
renderer = mujoco.Renderer(model, height=height, width=width)
f = (height / 2.0) / np.tan(np.deg2rad(model.cam_fovy[cam_id]) / 2.0)
K_mat = np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]])
dist = np.zeros((4, 1))

# ArUco detector
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

# 3. Initialize EKF
ekf = RobotEKF2D(
    init_state=[-0.5, -0.5, 0.0], wheel_radius=0.031, track_width=0.105
)

dt_physics = model.opt.timestep  # 0.002s (500 Hz)
vision_interval = 10  # Run camera every 10 steps (50 Hz)

for step in range(1000):
    # Drive Robot 1 in an arc
    data.ctrl[0] = 0.02
    data.ctrl[1] = 0.035
    mujoco.mj_step(model, data)

    # A. PREDICTION STEP (Every physics sub-step @ 500Hz)
    w_l = data.qvel[lw_qvel_adr]
    w_r = data.qvel[rw_qvel_adr]
    ekf.predict(omega_left=w_l, omega_right=w_r, dt=dt_physics)

    # B. CORRECTION STEP (Every vision interval @ 50Hz)
    if step % vision_interval == 0:
        renderer.update_scene(data, camera=cam_id)
        frame = renderer.render()
        corners, ids, _ = detector.detectMarkers(
            cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        )

        if ids is not None and 0 in ids.flatten():
            idx = np.where(ids.flatten() == 0)[0][0]
            success, rvec, tvec = cv2.solvePnP(
                obj_pts,
                corners[idx],
                K_mat,
                dist,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )

            if success:
                # Transform to World Frame
                cam_pos = data.cam_xpos[cam_id]
                cam_rot = data.cam_xmat[cam_id].reshape(3, 3)
                p_world = cam_pos + cam_rot @ (R_opt @ tvec.flatten())

                R_m, _ = cv2.Rodrigues(rvec)
                R_world = cam_rot @ (R_opt @ R_m)
                yaw_measured = np.arctan2(R_world[1, 0], R_world[0, 0])

                # EKF measurement update
                z = [p_world[0], p_world[1], yaw_measured]
                ekf.update(z)

    # Validation against MuJoCo Ground Truth
    if step % 50 == 0:
        gt_xy = data.geom_xpos[marker_geom_id][:2]
        err_mm = np.linalg.norm(ekf.state[:2] - gt_xy) * 1000.0
        print(
            f"Step {step:04d} | "
            f"EKF: ({ekf.state[0]:.3f}, {ekf.state[1]:.3f}, {np.rad2deg(ekf.state[2]):.1f}°) | "
            f"GT: ({gt_xy[0]:.3f}, {gt_xy[1]:.3f}) | "
            f"Error: {err_mm:.2f} mm"
        )