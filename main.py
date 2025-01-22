import cv2
import cv2.aruco as aruco
import numpy as np

# Initialize video capture
cap = cv2.VideoCapture(0)

# Load pre-calibrated camera matrix and distortion coefficients
camera_matrix = np.array([[800, 0, 320],
                          [0, 800, 240],
                          [0, 0, 1]], dtype=np.float32)  # Example camera matrix
dist_coeffs = np.array([0, 0, 0, 0], dtype=np.float32)  # Example distortion coefficients

# Define ArUco parameters
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
aruco_params = aruco.DetectorParameters()

# Function to detect and estimate pose of ArUco markers
def findArucoMarkers(img, marker_size=6, draw=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    if ids is not None:
        for corner, marker_id in zip(corners, ids):
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corner, marker_size, camera_matrix, dist_coeffs)
            if draw:
                # Draw marker borders
                aruco.drawDetectedMarkers(img, [corner])
                # Draw axis for each marker
                aruco.drawAxis(img, camera_matrix, dist_coeffs, rvec, tvec, 10)

                # Display the coordinates
                x, y, z = tvec[0][0]
                cv2.putText(img, f"ID: {marker_id[0]} X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}",
                            (int(corner[0][0][0]), int(corner[0][0][1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                print(f"Marker ID: {marker_id[0]} | X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}")

    return img

# Main loop
while True:
    ret, img = cap.read()
    if not ret:
        break

    img = findArucoMarkers(img, marker_size=6)  # Adjust marker size based on your ArUco markers

    cv2.imshow("ArUco Detection", img)

    if cv2.waitKey(1) == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
