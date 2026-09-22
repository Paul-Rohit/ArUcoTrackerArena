import cv2
import numpy as np


def generate_aruco_marker(
    marker_id: int,
    marker_size_px: int = 400,
    border_px: int = 50,
    dict_type=cv2.aruco.DICT_4X4_50,
) -> np.ndarray:
  dictionary = cv2.aruco.getPredefinedDictionary(dict_type)

  # 1. Generate raw marker (single channel)
  raw_marker = cv2.aruco.generateImageMarker(
      dictionary, marker_id, marker_size_px
  )

  # 2. Add white padding
  padded_marker = cv2.copyMakeBorder(
      raw_marker,
      top=border_px,
      bottom=border_px,
      left=border_px,
      right=border_px,
      borderType=cv2.BORDER_CONSTANT,
      value=255,
  )

  # 3. CRITICAL: Convert to 3-channel 24-bit BGR for MuJoCo texture compatibility
  bgr_marker = cv2.cvtColor(padded_marker, cv2.COLOR_GRAY2BGR)

  return bgr_marker


if __name__ == "__main__":
  # Generate and save both markers
  marker_0 = generate_aruco_marker(
      marker_id=0, marker_size_px=400, border_px=50
  )
  cv2.imwrite("aruco_0.png", marker_0)
  print("Saved aruco_0.png (500x500 24-bit RGB)")

  marker_1 = generate_aruco_marker(
      marker_id=1, marker_size_px=400, border_px=50
  )
  cv2.imwrite("aruco_1.png", marker_1)
  print("Saved aruco_1.png (500x500 24-bit RGB)")

  # Verification test
  detector = cv2.aruco.ArucoDetector(
      cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
  )
  for mid, img in [(0, marker_0), (1, marker_1)]:
    corners, ids, _ = detector.detectMarkers(img)
    assert ids is not None and ids[0][0] == mid, (
        f"Verification failed for marker {mid}"
    )
    print(f"Verified Marker ID {mid}: Self-detection successful.")