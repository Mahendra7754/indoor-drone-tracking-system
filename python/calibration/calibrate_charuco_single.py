import cv2
import numpy as np
import glob
import os

# =========================
# USER SETTINGS
# =========================
img_dir = "calibration_cam1"          # folder with calibration images (*.png)

squaresX = 7                         # number of chessboard squares in X direction
squaresY = 5                         # number of chessboard squares in Y direction
squareLength = 0.030                 # meters
markerLength = 0.022                 # meters
dictionary_id = cv2.aruco.DICT_4X4_1000

min_charuco_corners = 15             # reject images with fewer detected ChArUco corners
min_markers = 4                      # reject images with too few markers

SHOW_DETECTION = True                # set False if you don't want imshow windows
DELAY_MS = 50                        # delay for visualization

# =========================
# Load images
# =========================
image_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
print(f"Found {len(image_paths)} images in '{img_dir}'")

if len(image_paths) == 0:
    raise RuntimeError("No images found. Check img_dir and file extension (e.g., *.png).")

# =========================
# Create board & detectors
# =========================
aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)

board = cv2.aruco.CharucoBoard(
    (squaresX, squaresY),
    squareLength,
    markerLength,
    aruco_dict
)

# Detector parameters (good defaults + subpixel refinement)
params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, params)
charuco_detector = cv2.aruco.CharucoDetector(board)

all_charuco_corners = []
all_charuco_ids = []
img_size = None

kept = 0
rejected = 0

# =========================
# Process images
# =========================
for p in image_paths:
    img = cv2.imread(p)
    if img is None:
        print(f"❌ Reject (cannot read): {os.path.basename(p)}")
        rejected += 1
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img_size is None:
        img_size = (gray.shape[1], gray.shape[0])  # (width, height)

    # 1) Detect ArUco markers (we do this to use our detector params)
    marker_corners, marker_ids, _ = aruco_detector.detectMarkers(gray)

    if marker_ids is None or len(marker_ids) < min_markers:
        print(f"❌ Reject (few markers): {os.path.basename(p)}")
        rejected += 1
        continue

    # 2) Interpolate ChArUco corners from detected markers
    # IMPORTANT: correct arg order for OpenCV 4.x:
    # detectBoard(image[, charucoCorners[, charucoIds[, markerCorners[, markerIds]]]])
    try:
        ch_corners, ch_ids, _, _ = charuco_detector.detectBoard(
            gray, None, None, marker_corners, marker_ids
        )
    except cv2.error as e:
        # Fallback: let CharucoDetector detect markers internally (less control, but robust)
        # Returns: charucoCorners, charucoIds, markerCorners, markerIds
        ch_corners, ch_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)

    if ch_ids is None or len(ch_ids) < min_charuco_corners:
        print(f"❌ Reject (few charuco corners): {os.path.basename(p)}")
        rejected += 1
        continue

    all_charuco_corners.append(ch_corners)
    all_charuco_ids.append(ch_ids)
    kept += 1
    print(f"✅ Kept: {os.path.basename(p)}  (markers={len(marker_ids)}, charuco={len(ch_ids)})")

    # Visualization
    if SHOW_DETECTION:
        vis = img.copy()
        cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)
        cv2.aruco.drawDetectedCornersCharuco(vis, ch_corners, ch_ids, (0, 255, 0))
        cv2.imshow("Detection (press q to quit preview)", vis)
        key = cv2.waitKey(DELAY_MS) & 0xFF
        if key == ord('q'):
            SHOW_DETECTION = False

cv2.destroyAllWindows()

print(f"\nKept: {kept}, Rejected: {rejected}")

if kept < 10:
    raise RuntimeError("Too few good images for calibration. Try taking more/better images.")

# =========================
# Calibration
# =========================
rms, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
    charucoCorners=all_charuco_corners,
    charucoIds=all_charuco_ids,
    board=board,
    imageSize=img_size,
    cameraMatrix=None,
    distCoeffs=None
)

print("\n====== Calibration Result ======")
print(f"RMS reprojection error: {rms:.4f} px")
print("K (intrinsics):\n", K)
print("dist (distortion):\n", dist.ravel())

# Save
out_path = os.path.join(img_dir, "intrinsics_charuco.npz")
np.savez(
    out_path,
    K=K,
    dist=dist,
    img_size=np.array(img_size),
    rms=np.array([rms])
)
print(f"\n✅ Intrinsics saved to: {out_path}")

# =========================
# Undistortion preview
# =========================
test = cv2.imread(image_paths[0])
h, w = test.shape[:2]
print('printing h,w')
print(h)
print(w)
newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1.0)
und = cv2.undistort(test, K, dist, None, newK)

cv2.imshow("Original", test)
cv2.imshow("Undistorted", und)
cv2.waitKey(0)
cv2.destroyAllWindows()
