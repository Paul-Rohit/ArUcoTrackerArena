import cv2

img = cv2.imread("aruco_0.png")
print("aruco_1.png dimensions:", img.shape)  # Must print: (500, 500, 3)

detector = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
)
corners, ids, _ = detector.detectMarkers(img)
print("Detected IDs:", ids)  # Must print: [[0]]