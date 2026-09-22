import cv2
import mujoco
import numpy as np

# 1. Load model and simulation state
model = mujoco.MjModel.from_xml_path("robots12_arena.xml")
data = mujoco.MjData(model)

# 2. Set distinct starting positions
# Robot 1 at (-0.5, -0.5), Robot 2 at (0.5, 0.5)
r1_qpos_adr = model.jnt_qposadr[
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot1_root")
]
r2_qpos_adr = model.jnt_qposadr[
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot2_root")
]
data.qpos[r1_qpos_adr : r1_qpos_adr + 2] = [-0.5, -0.5]
data.qpos[r2_qpos_adr : r2_qpos_adr + 2] = [0.5, 0.5]
mujoco.mj_forward(model, data)

# 3. Camera intrinsics setup
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_cam")
width, height = 1280, 720
renderer = mujoco.Renderer(model, height=height, width=width)

fovy_rad = np.deg2rad(model.cam_fovy[cam_id])
f = (height / 2.0) / np.tan(fovy_rad / 2.0)
K = np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]], dtype=np.float64)
dist = np.zeros((4, 1), dtype=np.float64)

# 4. Marker metric dimensions (8cm plate with 400/500 inner black pattern)
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

detector = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
    cv2.aruco.DetectorParameters(),
)

# Optical-to-MuJoCo camera coordinate adapter
R_opt = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)


def get_agent_poses():
    renderer.update_scene(data, camera=cam_id)
    rgb = renderer.render()
    corners, ids, _ = detector.detectMarkers(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    )

    poses = {}
    if ids is not None:
        cam_pos = data.cam_xpos[cam_id]
        cam_rot = data.cam_xmat[cam_id].reshape(3, 3)

        for i, mid in enumerate(ids.flatten()):
            _, rvec, tvec = cv2.solvePnP(
                obj_pts,
                corners[i],
                K,
                dist,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )

            # Transform translation to arena world coordinates
            p_world = cam_pos + cam_rot @ (R_opt @ tvec.flatten())

            # Transform orientation to arena world coordinates
            R_m, _ = cv2.Rodrigues(rvec)
            R_world = cam_rot @ (R_opt @ R_m)
            yaw = np.arctan2(R_world[1, 0], R_world[0, 0])

            poses[int(mid)] = {
                "x": float(p_world[0]),
                "y": float(p_world[1]),
                "z": float(p_world[2]),
                "yaw_deg": float(np.rad2deg(yaw)),
            }
    return poses


if __name__ == "__main__":
    # Let contacts settle
    mujoco.mj_step(model, data)

    poses = get_agent_poses()

    r1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot1")
    r2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot2")
    gt = {
        0: data.xpos[r1_id][:2],
        1: data.xpos[r2_id][:2],
    }

    print("-" * 65)
    print(f"{'Agent':<8} | {'Estimated (X, Y)':<22} | {'Ground Truth (X, Y)':<22} | {'Error (mm)':<10}")
    print("-" * 65)

    for mid, pose in sorted(poses.items()):
        est_xy = np.array([pose["x"], pose["y"]])
        gt_xy = gt[mid]
        error_mm = np.linalg.norm(est_xy - gt_xy) * 1000.0
        print(
            f"Robot {mid:<2} | ({pose['x']:>6.3f}, {pose['y']:>6.3f}) m      | "
            f"({gt_xy[0]:>6.3f}, {gt_xy[1]:>6.3f}) m      | {error_mm:>6.1f} mm"
        )
    print("-" * 65)

    # Save validation image
    renderer.update_scene(data, camera=cam_id)
    frame = renderer.render()
    cv2.imwrite("debug_overhead.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print("Saved 'debug_overhead.png'.")