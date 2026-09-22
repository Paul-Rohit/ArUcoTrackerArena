import numpy as np


def wrap_angle(angle):
    """Normalize angle to [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class RobotEKF2D:

    def __init__(self, init_state, wheel_radius=0.031, track_width=0.105):
        self.state = np.array(init_state, dtype=np.float64)  # [x, y, theta]
        self.r = wheel_radius
        self.L = track_width

        # State error covariance
        self.P = np.diag([0.01, 0.01, 0.05])

        # Process noise (wheel slip / actuator uncertainty)
        self.Q = np.diag([1e-4, 1e-4, 4e-4])

        # Measurement noise covariance (ArUco camera uncertainty)
        self.R = np.diag([2e-5, 2e-5, 1e-3])  # ~4.5mm stdev, ~1.8 deg stdev

    def predict(self, omega_left, omega_right, dt):
        """High-frequency wheel odometry prediction step."""
        v = (self.r / 2.0) * (omega_right + omega_left)
        omega = (self.r / self.L) * (omega_right - omega_left)
        theta = self.state[2]

        # State update
        self.state[0] += v * dt * np.cos(theta)
        self.state[1] += v * dt * np.sin(theta)
        self.state[2] = wrap_angle(self.state[2] + omega * dt)

        # Jacobian F
        F = np.array(
            [[1.0, 0.0, -v * dt * np.sin(theta)], [0.0, 1.0, v * dt * np.cos(theta)], [0.0, 0.0, 1.0]]
        )

        # Covariance propagation
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        """Asynchronous ArUco measurement update step: z = [x, y, theta]."""
        y = np.array(z, dtype=np.float64) - self.state
        y[2] = wrap_angle(y[2])  # Angle residual normalization

        # H = I, so S = P + R
        S = self.P + self.R
        K = self.P @ np.linalg.inv(S)

        self.state += K @ y
        self.state[2] = wrap_angle(self.state[2])
        self.P = (np.eye(3) - K) @ self.P