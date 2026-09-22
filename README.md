# ArUcoTrackerArena

# ArUcoTrackerArena

A MuJoCo-based multi-robot tracking and evaluation environment using
OpenCV ArUco markers.

ArUcoTrackerArena provides a controlled simulation environment for
developing, debugging, and benchmarking visual tracking of multiple
differential-drive robots. The project combines MuJoCo ground truth with
OpenCV ArUco detection so that estimated robot poses can be compared
directly against the simulated state.

The repository contains a progression of tracking implementations,
starting with basic ArUco detection and single-camera pose estimation and
progressing to dual-camera tracking using plane back-projection,
metric-scale validation, ray-based Z triangulation, telemetry, debugging,
and adversarial stress testing.

---

## Features

- Two independently modeled robots in MuJoCo
- ArUco marker IDs `0` and `1`
- OpenCV `DICT_4X4_50` marker detection
- MuJoCo-rendered camera images
- Single-camera overhead tracking
- Dual slanted-camera tracking
- Sub-pixel ArUco corner refinement
- Camera/world coordinate-frame conversion
- Metric marker-size validation
- Plane-based marker back-projection
- Dual-camera ray triangulation for Z estimation
- Area-weighted multi-camera observation fusion
- Circular yaw fusion
- Ground-truth comparison against MuJoCo
- Live position, yaw, Z, and error telemetry
- Matplotlib trajectory/error visualization
- 2D Extended Kalman Filter for wheel-odometry/camera fusion
- Camera-layout analysis and parameter sweep
- Automated worst-case snapshot collection
- Stress trajectories designed to expose tracking failure modes
- Headless frame-based testing

---

## System Overview

The simulated system consists of:

```text
                 cam_south
                    \
                     \
                      \       Robot 1
                       \       [ID 0]
                        \
                         +----------------+
                         |     Arena      |
                         |                |
                         |     Robot 2    |
                         |       [ID 1]   |
                         +----------------+
                                   \
                                    \
                                     \
                                  cam_north


The two robots carry textured ArUco marker plates:

Robot	ArUco ID	Marker asset
Robot 1	0	aruco_0.png
Robot 2	1	aruco_1.png

The dual-camera configuration uses:

cam_south

cam_north

Both cameras render the same MuJoCo scene from different slanted viewpoints.

The tracker detects each marker independently in each camera image,
reconstructs its position in the arena coordinate frame, and then fuses
the observations.

Repository Structure
ArUcoTrackerArena/
│
├── #0_check_aruco.py
├── 0_gen_ArUco.py
│
├── 1_bringup_hook.py
├── 2_bringup_live.py
├── 3_bringup_live_withPlot.py
├── 4_bringup_withOdom.py
│
├── 5_batch_arena_tracker.py
├── 6_live_arena_tracker.py
├── 7_live_slanted_tracker.py
├── 8_debug_live_tracker_logger.py
├── 9_live_ArUco_tracker.py
│
├── 10_debug_live_tracker.py
├── 11_debug_v2.py
├── 12_worst_case_stress_test.py
├── 13_test_with_metrics.py
│
├── camera_layout_sweep.py
├── check_dependencies.py
├── ekf2d.py
│
├── aruco_0.png
├── aruco_1.png
│
├── robot1.xml
├── robot2.xml
├── robots12_arena.xml
├── robots12_arena_2cam.xml
│
├── LICENSE
└── README.md


###Requirements###
The project is written in Python and uses:

Python 3

MuJoCo >= 3.0.0

OpenCV Contrib >= 4.8.0.76

NumPy >= 1.24.0

Matplotlib >= 3.7.0

The repository includes check_dependencies.py to verify the runtime
environment and check specifically that OpenCV contains the aruco
module.

Important: install opencv-contrib-python, not only
opencv-python, because the project requires cv2.aruco.

Installation
Clone the repository:

git clone https://github.com/Paul-Rohit/ArUcoTrackerArena.git
cd ArUcoTrackerArena

Create and activate a virtual environment:

python -m venv .venv

Linux/macOS
source .venv/bin/activate

Windows
.venv\Scripts\activate

Install the dependencies:

pip install -r requirements.txt

If requirements.txt is not available or needs to be regenerated,
run:

python check_dependencies.py

The dependency checker looks for:

mujoco
opencv-contrib-python
numpy
matplotlib

Verify the Environment
Run:

python check_dependencies.py

The script checks that the required Python modules are available and,
for OpenCV, verifies that:

cv2.aruco

exists.

ArUco Marker Setup
The project uses the OpenCV dictionary:

cv2.aruco.DICT_4X4_50

Two markers are used:

Marker 0 -> Robot 1
Marker 1 -> Robot 2

Generate the markers
Run:

python 0_gen_ArUco.py

This generates:

aruco_0.png
aruco_1.png

Each marker is generated with:

400 × 400 pixel ArUco pattern

50 pixel white padding

500 × 500 pixel final image

3-channel BGR representation

The conversion to a 3-channel image is intentional because the marker
textures are used as MuJoCo textures.

The script also performs an automatic self-detection test after
generation.

Verify an Existing Marker
Run:

python "#0_check_aruco.py"

The script loads:

aruco_0.png

and verifies:

Image shape: (500, 500, 3)
Detected ID: [[0]]

This is a quick sanity check that the marker asset is compatible with
the expected OpenCV ArUco detector.

MuJoCo Environment
Arena
robots12_arena.xml defines the primary two-robot arena.

The environment includes:

arena floor

perimeter walls

lighting

ArUco marker textures

robot instances

camera configuration

The arena floor is modeled as a 2 m × 2 m working area.

The simulation uses:

integrator="RK4"
timestep="0.002"
gravity="0 0 -9.81"

The model also configures MuJoCo's rendering buffers for high-resolution
camera output.

Dual-Camera Arena
robots12_arena_2cam.xml is the main scene used by the dual slanted-camera
trackers.

It contains:

cam_south
cam_north

The two cameras provide different viewing rays for the same marker.

This enables the tracker to estimate the marker's 3D Z coordinate through
ray triangulation rather than relying entirely on single-camera PnP
depth.

The marker textures are explicitly assigned to the robot plates:

robot1_aruco_plate -> aruco_0.png
robot2_aruco_plate -> aruco_1.png

The marker materials disable specular/reflective properties to reduce
visual artifacts during detection.

Robot Models
The robot models are defined separately:

robot1.xml
robot2.xml

Each robot uses a differential-drive configuration with:

left wheel

right wheel

wheel joints

wheel geometry

robot body

ArUco marker plate

The tracking code uses a wheel radius of approximately:

0.031 m

and a track width of:

0.105 m

These parameters are also used by the 2D EKF.

Tracking Pipeline
The tracking system evolved through several stages.

Stage 1 — Basic ArUco Detection
The initial scripts verify that:

marker images can be generated,

the images have the expected dimensions,

OpenCV can detect the expected marker IDs.

Stage 2 — Single-Camera Pose Estimation
The early bring-up scripts use:

cv2.solvePnP(...)

with:

cv2.SOLVEPNP_IPPE_SQUARE

to estimate the marker pose from its four detected corners.

The estimated camera-frame pose is transformed into the MuJoCo world
coordinate frame.

An optical-to-MuJoCo coordinate adapter is used:

R_OPT = np.diag([1.0, -1.0, -1.0])

This accounts for the difference between the OpenCV optical coordinate
convention and the MuJoCo camera coordinate system.

Marker Metric Scale
The generated marker image contains a 400 px inner pattern inside a
500 px image.

The simulated physical marker plate is 8 cm across.

Therefore, the effective black ArUco pattern size used by the pose
calculation is:

0.08 × (400 / 500)
= 0.064 m

So the tracker uses:

MARKER_LEN = 0.064

as the physical marker length.

This distinction is important because the white border surrounding the
ArUco pattern is not part of the black coded pattern used for geometric
measurement.

V2 Tracking Method
The later tracker implementation replaces the original PnP-based depth
estimation with a geometry-based approach.

The v2 tracker is implemented in:

9_live_ArUco_tracker.py
11_debug_v2.py
12_worst_case_stress_test.py
13_test_with_metrics.py

The method has several key components.

1. Plane Back-Projection
The ArUco marker plate is horizontal and has a known height:

z = PLATE_Z

Instead of directly using the depth returned by solvePnP, each detected
image corner is converted into a camera ray and intersected with the
known marker plane.

For a camera center C and normalized world-space ray d:

P = C + s d

where:

s = (z_plate - C_z) / d_z

This produces a world-space point for each marker corner.

The marker center is then calculated from the reconstructed corners.

This makes the XY estimate explicitly constrained to the known physical
marker plane.

2. Marker Yaw
The marker orientation is estimated from the reconstructed top-left to
top-right edge.

The tracker computes the world-space direction of the marker's X-axis
and obtains yaw using:

yaw = atan2(dy, dx)

When observations from multiple cameras are available, yaw is fused on
the unit circle rather than averaging angles directly.

This avoids problems at the -180° / +180° wraparound.

3. Metric Scale Gate
After projecting the detected marker corners onto the known plane, the
tracker reconstructs the marker's physical side length.

The reconstructed size is compared against:

MARKER_LEN = 0.064 m

The v2 configuration uses:

MAX_SCALE_ERR = 0.08

meaning an observation is rejected when its reconstructed marker size
deviates by more than 8% from the expected physical size.

This provides a geometric quality check for:

incorrect detections

poor corner localization

incorrect plane assumptions

severe perspective/foreshortening

other unreliable observations

4. Dual-Ray Z Triangulation
Plane back-projection provides a strong XY estimate when the marker
height is known, but the system also independently estimates Z using the
two camera observations.

For each camera, the marker center produces a world-space ray.

The tracker computes the least-squares point closest to the available
camera rays.

Conceptually:

Camera South
      \
       \
        \        Marker
         \       *
          \     /
           \   /
            \ /
             X
            / \
           /   \
          /     \
         /       \
Camera North

The intersection is not assumed to be exact. Instead, the tracker solves
for the point minimizing the distance to both rays.