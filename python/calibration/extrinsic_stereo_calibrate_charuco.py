import cv2
import numpy as np
import glob
import os

# ============================================================
# Extrinsic calibration (Stereo) using ChArUco
# Computes R, T between cameras (left->right)
# ============================================================

# ------------------ PATHS (based on your folder listing) ------------------
LEFT_DIR  = "stereo_pairs/left"
RIGHT_DIR = "stereo_pairs/right"

INTR_LEFT  = "Left_camera/intrinsics_charuco.npz"
INTR_RIGHT = "Right_camera/intrinsics_charuco.npz"

OUT_FILE = "stereo_extrinsics_charuco.npz"
# ----------------------------------------------------- squaresY = 7, 5
squaresX = 7                         # number of chessboard squares in X direction
squaresY = 5                         # number of chessboard squares in Y direction
squareLength = 0.0535              # meters
markerLength = 0.039             # meters
dictionary_id = cv2.aruco.DICT_4X4_1000

min_common_corners = 12
SHOW_RECTIFIED_PREVIEW = True
# ----------------------------------------------------


def assert_exists(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n❌ {label} not found:\n  {path}\n\n"
            f"Tip: run in the Drone folder (where Left_camera/ Right_camera/ exist)\n"
            f"and make sure the file name is exactly intrinsics_charuco.npz"
        )


def detect_charuco(gray, aruco_detector, charuco_detector):
    m_corners, m_ids, _ = aruco_detector.detectMarkers(gray)
    if m_ids is None or len(m_ids) < 4:
        return None, None

    try:
        ch_corners, ch_ids, _, _ = charuco_detector.detectBoard(
            gray, None, None, m_corners, m_ids
        )
    except cv2.error:
        ch_corners, ch_ids, _, _ = charuco_detector.detectBoard(gray)

    if ch_ids is None or len(ch_ids) < 4:
        return None, None

    return ch_corners, ch_ids


def main():
    # Check files exist
    assert_exists(INTR_LEFT,  "LEFT intrinsics")
    assert_exists(INTR_RIGHT, "RIGHT intrinsics")

    # Load intrinsics
    L = np.load(INTR_LEFT)
    R = np.load(INTR_RIGHT)

    K1 = L["K"].astype(np.float64)
    d1 = L["dist"].ravel().astype(np.float64)

    K2 = R["K"].astype(np.float64)
    d2 = R["dist"].ravel().astype(np.float64)

    # Use imageSize from intrinsics if present
    imageSize = None
    if "img_size" in L and "img_size" in R:
        sL = tuple(L["img_size"].tolist())
        sR = tuple(R["img_size"].tolist())
        if sL != sR:
            raise RuntimeError(f"❌ Intrinsics img_size mismatch: {sL} vs {sR}")
        imageSize = sL  # (w,h)
        print("Using imageSize from intrinsics:", imageSize)

    # Load stereo pair paths
    left_paths  = sorted(glob.glob(os.path.join(LEFT_DIR, "left_*.png")))
    right_paths = sorted(glob.glob(os.path.join(RIGHT_DIR, "right_*.png")))

    if len(left_paths) == 0:
        raise RuntimeError(f"❌ No left images in {LEFT_DIR}")
    if len(left_paths) != len(right_paths):
        raise RuntimeError("❌ Left/right image count mismatch")

    print(f"Found {len(left_paths)} stereo pairs")

    # Create board/detectors
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
    board = cv2.aruco.CharucoBoard((squaresX, squaresY), squareLength, markerLength, aruco_dict)

    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    charuco_detector = cv2.aruco.CharucoDetector(board)

    obj_corners_3d = board.getChessboardCorners()  # (N,3)

    objectPoints = []
    imgPointsL = []
    imgPointsR = []

    kept, rej = 0, 0

    for lp, rp in zip(left_paths, right_paths):
        imgL = cv2.imread(lp)
        imgR = cv2.imread(rp)
        if imgL is None or imgR is None:
            rej += 1
            continue

        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        if imageSize is None:
            imageSize = (grayL.shape[1], grayL.shape[0])  # (w,h)
            print("Inferred imageSize from images:", imageSize)

        if (grayL.shape[1], grayL.shape[0]) != imageSize or (grayR.shape[1], grayR.shape[0]) != imageSize:
            rej += 1
            continue

        chL, idsL = detect_charuco(grayL, aruco_detector, charuco_detector)
        chR, idsR = detect_charuco(grayR, aruco_detector, charuco_detector)

        if idsL is None or idsR is None:
            rej += 1
            continue

        idsL = idsL.flatten()
        idsR = idsR.flatten()

        mapL = {int(i): chL[k, 0, :] for k, i in enumerate(idsL)}
        mapR = {int(i): chR[k, 0, :] for k, i in enumerate(idsR)}

        common_ids = sorted(set(mapL.keys()).intersection(set(mapR.keys())))
        if len(common_ids) < min_common_corners:
            rej += 1
            continue

        obj = np.array([obj_corners_3d[i] for i in common_ids], dtype=np.float32)
        ptsL = np.array([mapL[i] for i in common_ids], dtype=np.float32)
        ptsR = np.array([mapR[i] for i in common_ids], dtype=np.float32)

        objectPoints.append(obj)
        imgPointsL.append(ptsL)
        imgPointsR.append(ptsR)
        kept += 1

    print(f"Kept pairs: {kept}, Rejected pairs: {rej}")
    if kept < 10:
        raise RuntimeError("❌ Too few good stereo pairs. Capture 30–60 pairs.")

    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)

    rms, _, _, _, _, Rmat, Tvec, E, F = cv2.stereoCalibrate(
        objectPoints, imgPointsL, imgPointsR,
        K1, d1, K2, d2,
        imageSize,
        criteria=criteria,
        flags=flags
    )

    baseline = float(np.linalg.norm(Tvec))

    print("\n====== EXTRINSIC RESULT ======")
    print(f"Stereo RMS reprojection error: {rms:.4f} px")
    print("T (meters):", Tvec.ravel())
    print(f"Baseline |T| = {baseline:.4f} m")

    # Rectification maps (useful for later triangulation)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, d1, K2, d2, imageSize,
        Rmat, Tvec,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.3
    )

    map1x, map1y = cv2.initUndistortRectifyMap(K1, d1, R1, P1, imageSize, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, d2, R2, P2, imageSize, cv2.CV_32FC1)

    np.savez_compressed(
        OUT_FILE,
        K1=K1, d1=d1, K2=K2, d2=d2,
        R=Rmat, T=Tvec, E=E, F=F,
        imageSize=np.array(imageSize),
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        map1x=map1x, map1y=map1y, map2x=map2x, map2y=map2y,
        rms=np.array([rms]),
        baseline=np.array([baseline])
    )
    print("roi1:", roi1)
    print("roi2:", roi2)
    print(f"\n✅ Saved extrinsics to: {OUT_FILE}")

    if SHOW_RECTIFIED_PREVIEW:
        testL = cv2.imread(left_paths[0])
        testR = cv2.imread(right_paths[0])

        rectL = cv2.remap(testL, map1x, map1y, cv2.INTER_LINEAR)
        rectR = cv2.remap(testR, map2x, map2y, cv2.INTER_LINEAR)

        vis = cv2.hconcat([rectL, rectR])
        h_vis = vis.shape[0]
        for y in range(0, h_vis, 40):
            cv2.line(vis, (0, y), (vis.shape[1], y), (0, 255, 0), 1)

        cv2.imshow("Rectified (LEFT | RIGHT) - points should lie on same green line", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
