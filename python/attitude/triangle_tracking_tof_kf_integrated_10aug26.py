# -*- coding: utf-8 -*-
# ============================================================
# Stereo 3-IR-LED Triangle Tracking + Rear Center / Front Vector + Crazyflie ToF Logger
#
#
# Main idea:
# 1) Detect multiple bright IR LED blobs in left and right cameras.
# 2) Try all possible 3-LED left/right stereo matches.
# 3) Triangulate each possible 3D triangle.
# 4) Select the triangle whose side lengths match your drone LED geometry:
#       rear-left  <-> rear-right = 14.5 cm
#       front      <-> rear-right =  8.5 cm
#       front      <-> rear-left  = 10.5 cm
# 5) Compute:
#       rear_center = (rear_left + rear_right) / 2
#       front_vector = front - rear_center
#       front_unit_vector = front_vector / ||front_vector||
#
# Important coordinate note:
# - X_undist/Y_undist/Z_undist are in your stereo calibration coordinate frame.
# - In the usual OpenCV camera frame: X = right, Y = down, Z = forward/depth.
# - yaw_xz_deg is the heading angle of the front vector projected onto the X-Z plane.
# ============================================================
# @ 03/07/2026 separate navigation-frame conversion and ZYX Euler pose functions added
#
# Euler convention used below:
#   Navigation/body axes: X forward, Y right, Z down
#   R_NB = Rz(yaw) @ Ry(pitch) @ Rx(roll)
#

import cv2
import numpy as np
import time
import os
import csv
import threading
import logging
import itertools
import math
from datetime import datetime

from openpyxl import Workbook

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper


# ===================== USER SETTINGS =====================
LEFT_CAM_INDEX  = 2
RIGHT_CAM_INDEX = 0

W, H = 800, 600
FPS = 20
FORCE_MJPG = True

EXTR_FILE = "stereo_extrinsics_charuco.npz"

GAUSS_K = (5, 5)
THRESH_T = 100
THRESH_MAX = 255

# Blob/centroid settings
MIN_BLOB_AREA = 1
MAX_BLOB_AREA = 500
USE_INTENSITY_WEIGHTED_CENTROID = True

# For 3-LED triangle detection
REQUIRED_LED_COUNT = 3
MAX_CANDIDATE_BLOBS = 6          # Increase only if you have many false detections
MIN_BLOB_DISTANCE_PX = 4.0       # Reject almost-duplicate blob centers

# Your measured physical LED geometry on the drone
# Front LED + two rear LEDs make a triangle.
LED_REAR_BASE_M = 0.145                 # rear-left to rear-right = ~14.5 cm
LED_FRONT_TO_REAR_RIGHT_M = 0.085       # front to rear-right = ~8.5 cm
LED_FRONT_TO_REAR_LEFT_M = 0.105        # front to rear-left = ~10.5 cm

# Triangle matching tolerance.
# This is intentionally not too strict because your physical measurements are approximate
# and stereo scale may still contain some error.
MAX_TRIANGLE_SHAPE_RMS_RATIO = 0.45
MIN_TRIANGLE_SCALE = 0.40
MAX_TRIANGLE_SCALE = 1.80

#scaling error factors
scale_x= 0.9113 # scaling in x axis
offset_x= 0.00685

scale_y= 0.89558 # scaling in y axis
offset_y= 0.00767

scale_z= 1.1037275 # scaling in z axis
offset_z= -0.001538

# Pair validity checks
ENABLE_DY_CHECK = False
MAX_RAW_DY_PX = 80.0          # Tune for your setup. Increase if raw cameras are vertically offset.
MIN_Z_M = 0.05
MAX_Z_M = 10.0
MAX_ABS_XYZ_M = 20.0
MIN_HOMOG_W = 1e-9

PRINT_EVERY_SEC = 0.2
LOG_EVERY_SEC = 0.1

DISPLAY_SCALE = 0.5
SHOW_THRESHOLD_BOTTOM = False     # False = bottom row shows original frames, True = threshold debug
LOG_FOLDER = "logs"

CF_URI = uri_helper.uri_from_env(
    default="radio://0/80/2M/E7E7E7E7E7"
)

TOF_PERIOD_MS = 100
# =========================================================

logging.basicConfig(level=logging.ERROR)


# =========================================================
# Coordinate-frame convention
# =========================================================
# OpenCV left-camera frame C:
#   X_C = right, Y_C = down, Z_C = forward
#
# Camera-aligned navigation frame N:
#   X_N = forward, Y_N = right, Z_N = down
#
# Therefore:
#   [X_N, Y_N, Z_N]^T = [Z_C, X_C, Y_C]^T
#   v_N = R_NC @ v_C
#
# This is only an axis conversion. If the stereo rig is tilted relative to the
# room/gravity frame, replace R_NC with a calibrated camera-to-world rotation.
R_NC = np.array(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


# =========================================================
# Crazyflie ToF Logger
# =========================================================
class TOFLogger:
    _NONE = -1

    def __init__(self, link_uri):
        self._cf = Crazyflie(rw_cache="./cache")
        self._lock = threading.Lock()
        self.is_connected = False

        self._front = self._NONE
        self._front_local_time = None       # local monotonic time when callback arrived
        self._front_cf_timestamp = None     # Crazyflie log timestamp
        self._packet_count = 0

        self._cf.connected.add_callback(self._connected)
        self._cf.disconnected.add_callback(self._disconnected)
        self._cf.connection_failed.add_callback(self._connection_failed)
        self._cf.connection_lost.add_callback(self._connection_lost)

        print(f"[ToF] Connecting to {link_uri} ...")
        self._cf.open_link(link_uri)

    @property
    def latest(self):
        with self._lock:
            return {
                "tof_front": self._front,
                "tof_local_time": self._front_local_time,
                "tof_cf_timestamp": self._front_cf_timestamp,
                "tof_packet_count": self._packet_count,
                "tof_valid": self._front != self._NONE
            }

    def _connected(self, link_uri):
        print(f"[ToF] Connected to {link_uri}")
        self.is_connected = True

        self._lg = LogConfig(name="ToFData", period_in_ms=TOF_PERIOD_MS)
        self._lg.add_variable("range.front", "uint16_t")

        try:
            self._cf.log.add_config(self._lg)
            self._lg.data_received_cb.add_callback(self._log_data)
            self._lg.error_cb.add_callback(self._log_error)
            self._lg.start()
            print("[ToF] Started range.front logging.")
        except KeyError as e:
            print(f"[ToF] Variable not found: {e}")
        except AttributeError:
            print("[ToF] Bad log configuration.")

    def _log_data(self, timestamp, data, logconf):
        with self._lock:
            self._front = data["range.front"]
            self._front_local_time = time.perf_counter()
            self._front_cf_timestamp = timestamp
            self._packet_count += 1

    def _log_error(self, logconf, msg):
        print(f"[ToF] Log error ({logconf.name}): {msg}")

    def _connection_failed(self, link_uri, msg):
        print(f"[ToF] Connection failed: {msg}")
        self.is_connected = False

    def _connection_lost(self, link_uri, msg):
        print(f"[ToF] Connection lost: {msg}")
        self.is_connected = False

    def _disconnected(self, link_uri):
        print(f"[ToF] Disconnected from {link_uri}")
        self.is_connected = False

    def close(self):
        try:
            self._cf.close_link()
        except Exception:
            pass


# =========================================================
# Logger Helpers
# =========================================================
def get_headers():
    return [
        "time_s",
        "loop_dt_s",
        "log_dt_s",
        "cam_grab_dt_ms",
        "cam_retrieve_dt_ms",

        "n_blobs_left",
        "n_blobs_right",

        "triangle_detected",
        "triangle_valid",
        "reject_reason",
        "triangle_score",
        "triangle_shape_rms_ratio",
        "triangle_scale",

        # Front LED pixels
        "front_xL_px", "front_yL_px",
        "front_xR_px", "front_yR_px",

        # Rear-left LED pixels
        "rear_left_xL_px", "rear_left_yL_px",
        "rear_left_xR_px", "rear_left_yR_px",

        # Rear-right LED pixels
        "rear_right_xL_px", "rear_right_yL_px",
        "rear_right_xR_px", "rear_right_yR_px",

        # Front 3D
        "front_X_m", "front_Y_m", "front_Z_m",

        # Rear-left 3D
        "rear_left_X_m", "rear_left_Y_m", "rear_left_Z_m",

        # Rear-right 3D
        "rear_right_X_m", "rear_right_Y_m", "rear_right_Z_m",

        # Calculated rear center and front vector
        "rear_center_X_m", "rear_center_Y_m", "rear_center_Z_m",

        # Linear Kalman filter outputs
        # State = [X, Y, Z, Vx, Vy, Vz]^T, using rear_center as measurement.
        "kf_initialized",
        "kf_measurement_used",
        "kf_prediction_only",
        "kf_X_m", "kf_Y_m", "kf_Z_m",
        "kf_Vx_mps", "kf_Vy_mps", "kf_Vz_mps",
        "kf_sigma_X_m", "kf_sigma_Y_m", "kf_sigma_Z_m",
        "kf_sigma_Vx_mps", "kf_sigma_Vy_mps", "kf_sigma_Vz_mps",

        "front_vector_X_m", "front_vector_Y_m", "front_vector_Z_m",
        "front_vector_norm_m",
        "front_unit_X", "front_unit_Y", "front_unit_Z",

        # Useful heading projections
        "yaw_xz_deg",
        "yaw_xy_deg",

        # Camera-aligned navigation-frame debugging values
        "rear_center_N_X_m", "rear_center_N_Y_m", "rear_center_N_Z_m",
        "front_vector_N_X_m", "front_vector_N_Y_m", "front_vector_N_Z_m",
        "body_x_N_X", "body_x_N_Y", "body_x_N_Z",
        "body_y_N_X", "body_y_N_Y", "body_y_N_Z",
        "body_z_N_X", "body_z_N_Y", "body_z_N_Z",

        # Full drone attitude using ZYX Euler convention:
        # R_NB = Rz(yaw) @ Ry(pitch) @ Rx(roll)
        "euler_zyx_valid",
        "euler_zyx_reason",
        "yaw_zyx_deg",
        "pitch_zyx_deg",
        "roll_zyx_deg",
        "euler_gimbal_lock",
        "R_NB_determinant",
        "R_NB_orthogonality_error",

        # Measured triangle side lengths
        "rear_base_m",
        "front_to_rear_left_m",
        "front_to_rear_right_m",

        # ToF
        "tof_front_mm",
        "tof_valid",
        "tof_age_ms",
        "tof_packet_count",
        "tof_cf_timestamp"
    ]


def create_loggers():
    os.makedirs(LOG_FOLDER, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(LOG_FOLDER, f"triangle_tracking_tof_{timestamp}")
    xlsx_file = base + ".xlsx"
    csv_file = base + ".csv"

    headers = get_headers()

    # Excel logger, saved at the beginning and end.
    wb = Workbook()
    ws = wb.active
    ws.title = "Triangle_Log"
    ws.append(headers)
    wb.save(xlsx_file)

    # CSV logger, flushed every row. This is the safer primary measurement log.
    csv_fh = open(csv_file, mode="w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_fh)
    csv_writer.writerow(headers)
    csv_fh.flush()

    print(f"[LOG] Excel file created: {xlsx_file}")
    print(f"[LOG] CSV file created:   {csv_file}")

    return wb, ws, xlsx_file, csv_fh, csv_writer, csv_file


def safe_float(v):
    if v is None:
        return -1.0
    try:
        vf = float(v)
        if not math.isfinite(vf):
            return -1.0
        return vf
    except Exception:
        return -1.0


def append_log_row(ws, csv_writer, csv_fh, row):
    # Convert numpy values to native Python values for safer Excel/CSV writing.
    clean_row = []
    for v in row:
        if isinstance(v, np.generic):
            clean_row.append(v.item())
        elif isinstance(v, np.ndarray):
            clean_row.append(v.tolist())
        else:
            clean_row.append(v)

    # For speed, CSV is the primary live logger.
    # Uncomment the next line if you really want live Excel rows too, but it can slow the loop.
    # ws.append(clean_row)

    csv_writer.writerow(clean_row)
    csv_fh.flush()


# =========================================================
# Camera Helpers
# =========================================================
def setup_camera(index, name):
    cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open {name} camera index={index}")

    if FORCE_MJPG:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    # Some camera backends ignore this, but it is still worth requesting.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # These properties are backend/camera dependent. Confirm actual behavior manually.
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)
    cap.set(cv2.CAP_PROP_GAIN, 0)

    # Flush old buffered frames.
    for _ in range(20):
        cap.read()

    ret, frame = cap.read()
    if not ret or frame is None:
        raise RuntimeError(f"{name} camera gave no frame")

    ah, aw = frame.shape[:2]
    fps_rep = cap.get(cv2.CAP_PROP_FPS)

    print(f"[{name}] requested {W}x{H}@{FPS}")
    print(f"[{name}] actual    {aw}x{ah}@{fps_rep:.1f}")

    # Critical: calibration is only valid at the same image resolution.
    if aw != W or ah != H:
        cap.release()
        raise RuntimeError(
            f"{name} camera actual resolution {aw}x{ah} does not match "
            f"requested/calibrated resolution {W}x{H}. Measurement would be invalid."
        )

    return cap


def grab_retrieve_pair(capL, capR):
    """
    Better than sequential read() for stereo timing.
    Still not true hardware sync, but reduces left/right timing mismatch.
    """
    t_grab_start = time.perf_counter()

    okL = capL.grab()
    okR = capR.grab()

    t_grab_done = time.perf_counter()

    if not okL or not okR:
        return False, None, None, t_grab_done, -1.0, -1.0

    retL, frameL = capL.retrieve()
    t_after_L = time.perf_counter()

    retR, frameR = capR.retrieve()
    t_after_R = time.perf_counter()

    if not retL or frameL is None or not retR or frameR is None:
        return False, None, None, t_grab_done, -1.0, -1.0

    cam_grab_dt_ms = (t_grab_done - t_grab_start) * 1000.0
    cam_retrieve_dt_ms = (t_after_R - t_after_L) * 1000.0

    return True, frameL, frameR, t_grab_done, cam_grab_dt_ms, cam_retrieve_dt_ms


def open_camera_settings(cap, name):
    print(f"Opening settings for {name}")
    ok = cap.set(cv2.CAP_PROP_SETTINGS, 1)
    print(f"[{name}] settings result: {ok}")


# =========================================================
# Vision Helpers
# =========================================================
def detect_led_blobs(frame_bgr):
    """
    Detect bright IR LED blobs.
    Returns:
        blobs: list of dictionaries sorted by area descending.
        thresh: threshold image for debugging.
    Each blob dictionary contains:
        cx, cy, area, bbox, kp
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, GAUSS_K, 0)

    _, thresh = cv2.threshold(
        gray_blur,
        THRESH_T,
        THRESH_MAX,
        cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    blobs = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (MIN_BLOB_AREA <= area <= MAX_BLOB_AREA):
            continue

        mask = np.zeros_like(gray_blur, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)

        if USE_INTENSITY_WEIGHTED_CENTROID:
            ys, xs = np.nonzero(mask)
            weights = gray_blur[ys, xs].astype(np.float64)
            wsum = np.sum(weights)

            if wsum > 1e-12:
                cx = np.sum(xs * weights) / wsum
                cy = np.sum(ys * weights) / wsum
            else:
                M = cv2.moments(cnt)
                if abs(M["m00"]) < 1e-12:
                    continue
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
        else:
            M = cv2.moments(cnt)
            if abs(M["m00"]) < 1e-12:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

        x, y, w, h = cv2.boundingRect(cnt)
        kp_size = float(max(w, h))
        kp = cv2.KeyPoint(float(cx), float(cy), kp_size)

        blobs.append({
            "cx": float(cx),
            "cy": float(cy),
            "area": float(area),
            "bbox": (int(x), int(y), int(w), int(h)),
            "kp": kp
        })

    # Sort strongest/largest first
    blobs.sort(key=lambda b: b["area"], reverse=True)

    # Remove almost duplicate centers
    filtered = []
    for b in blobs:
        duplicate = False
        for kept in filtered:
            d = math.hypot(b["cx"] - kept["cx"], b["cy"] - kept["cy"])
            if d < MIN_BLOB_DISTANCE_PX:
                duplicate = True
                break
        if not duplicate:
            filtered.append(b)

    return filtered[:MAX_CANDIDATE_BLOBS], thresh


def make_threshold_bgr(thresh):
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


def resize_to_match(img, ref):
    return cv2.resize(
        img,
        (ref.shape[1], ref.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )


# =========================================================
# Triangulation Helpers
# =========================================================
def homogeneous_to_3d(X_h):
    w = float(X_h[3, 0])
    if abs(w) < MIN_HOMOG_W:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)

    X = (X_h[:3, 0] / w).astype(np.float64)
    return X


def triangulate_raw_direct(uL, uR, K1, K2, R, T):
    """
    Method 1:
    Raw distorted pixel coordinates directly.
    This intentionally ignores lens distortion, so use only for comparison.
    """
    ptsL = np.array([[uL[0]], [uL[1]]], dtype=np.float64)
    ptsR = np.array([[uR[0]], [uR[1]]], dtype=np.float64)

    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K2 @ np.hstack((R, T.reshape(3, 1)))

    X_h = cv2.triangulatePoints(P1, P2, ptsL, ptsR)
    return homogeneous_to_3d(X_h)


def triangulate_point_undistorted(uL, uR, K1, d1, K2, d2, R, T):
    """
    Preferred method:
    Raw LED detection -> point-only undistortion -> triangulation in normalized coordinates.
    """
    ptsL = np.array([[uL]], dtype=np.float64)
    ptsR = np.array([[uR]], dtype=np.float64)

    ptsL_norm = cv2.undistortPoints(ptsL, K1, d1)
    ptsR_norm = cv2.undistortPoints(ptsR, K2, d2)

    xL = ptsL_norm.reshape(2, 1)
    xR = ptsR_norm.reshape(2, 1)

    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((R, T.reshape(3, 1)))

    X_h = cv2.triangulatePoints(P1, P2, xL, xR)

    # converting homogeneous point to normal 3D point
    X_3d= homogeneous_to_3d(X_h)

    X_3d[0]=scale_x*X_3d[0] + offset_x
    X_3d[1]=scale_y*X_3d[1] + offset_y
    X_3d[2]=scale_z*X_3d[2] + offset_z
    return X_3d


def validate_3d_point(X):
    if not np.all(np.isfinite(X)):
        return False, "nan_or_inf"

    if abs(X[0]) > MAX_ABS_XYZ_M or abs(X[1]) > MAX_ABS_XYZ_M or abs(X[2]) > MAX_ABS_XYZ_M:
        return False, "xyz_out_of_range"

    if X[2] < MIN_Z_M:
        return False, "z_too_small_or_behind"

    if X[2] > MAX_Z_M:
        return False, "z_too_large"

    return True, "ok"


def validate_pair_before_triangulation(xL_meas, yL_meas, xR_meas, yR_meas):
    if xL_meas is None or yL_meas is None or xR_meas is None or yR_meas is None:
        return False, -1.0, "missing_left_or_right_detection"

    raw_dy = abs(yL_meas - yR_meas)

    if ENABLE_DY_CHECK and raw_dy > MAX_RAW_DY_PX:
        return False, raw_dy, "raw_dy_too_large"

    return True, raw_dy, "ok"


# =========================================================
# Triangle / Pose Helpers
# =========================================================
def distance_3d(a, b):
    return float(np.linalg.norm(a - b))


def triangle_side_lengths(points3d):
    """
    For points [P0, P1, P2], returns:
        {(0,1): d01, (0,2): d02, (1,2): d12}
    """
    return {
        (0, 1): distance_3d(points3d[0], points3d[1]),
        (0, 2): distance_3d(points3d[0], points3d[2]),
        (1, 2): distance_3d(points3d[1], points3d[2]),
    }


def score_triangle_geometry(points3d):
    """
    Scores whether the 3D points match the expected LED triangle.

    To tolerate stereo scale error, this function fits one scalar scale between
    expected side lengths and measured side lengths, then scores the shape error.
    """
    expected_sorted = np.array(sorted([
        LED_FRONT_TO_REAR_RIGHT_M,
        LED_FRONT_TO_REAR_LEFT_M,
        LED_REAR_BASE_M
    ]), dtype=np.float64)

    sides = triangle_side_lengths(points3d)
    measured_sorted = np.array(sorted(sides.values()), dtype=np.float64)

    if not np.all(np.isfinite(measured_sorted)):
        return False, float("inf"), {
            "reason": "triangle_side_nan",
            "shape_rms_ratio": float("inf"),
            "scale": -1.0,
            "sides": sides
        }

    # Least-squares scalar: measured ≈ scale * expected
    denom = float(np.dot(expected_sorted, expected_sorted))
    if denom < 1e-12:
        scale = 1.0
    else:
        scale = float(np.dot(measured_sorted, expected_sorted) / denom)

    expected_scaled = scale * expected_sorted
    rms = float(np.sqrt(np.mean((measured_sorted - expected_scaled) ** 2)))
    denom_rms = float(np.sqrt(np.mean(expected_scaled ** 2))) + 1e-12
    shape_rms_ratio = rms / denom_rms

    if scale < MIN_TRIANGLE_SCALE or scale > MAX_TRIANGLE_SCALE:
        return False, shape_rms_ratio, {
            "reason": "triangle_scale_out_of_range",
            "shape_rms_ratio": shape_rms_ratio,
            "scale": scale,
            "sides": sides
        }

    if shape_rms_ratio > MAX_TRIANGLE_SHAPE_RMS_RATIO:
        return False, shape_rms_ratio, {
            "reason": "triangle_shape_mismatch",
            "shape_rms_ratio": shape_rms_ratio,
            "scale": scale,
            "sides": sides
        }

    return True, shape_rms_ratio, {
        "reason": "ok",
        "shape_rms_ratio": shape_rms_ratio,
        "scale": scale,
        "sides": sides
    }


def label_triangle_points(points3d):
    """
    Identify front, rear-left, rear-right from the 3 measured LED positions.

    Logic:
    - The longest side should be the rear base: rear-left <-> rear-right.
    - The remaining point is front.
    - Of the two rear points:
        front-to-rear-right is shorter, approximately 7.3 cm.
        front-to-rear-left  is longer,  approximately 8.3 cm.
      So the rear point closer to front is labelled rear_right.
    """
    sides = triangle_side_lengths(points3d)

    # Longest side = rear base
    rear_pair = max(sides.keys(), key=lambda k: sides[k])
    all_indices = {0, 1, 2}
    front_idx = list(all_indices - set(rear_pair))[0]

    rear_a_idx, rear_b_idx = rear_pair
    d_a = distance_3d(points3d[front_idx], points3d[rear_a_idx])
    d_b = distance_3d(points3d[front_idx], points3d[rear_b_idx])

    if d_a <= d_b:
        rear_right_idx = rear_a_idx
        rear_left_idx = rear_b_idx
    else:
        rear_right_idx = rear_b_idx
        rear_left_idx = rear_a_idx

    front = points3d[front_idx]
    rear_left = points3d[rear_left_idx]
    rear_right = points3d[rear_right_idx]

    rear_center = 0.5 * (rear_left + rear_right)
    front_vector = front - rear_center
    front_vector_norm = float(np.linalg.norm(front_vector))

    if front_vector_norm > 1e-12:
        front_unit = front_vector / front_vector_norm
    else:
        front_unit = np.array([np.nan, np.nan, np.nan], dtype=np.float64)

    # Heading projections.
    # yaw_xz_deg: projection onto X-Z plane, useful if Y is vertical in your stereo/world frame.
    # yaw_xy_deg: projection onto X-Y plane, useful for image-plane style debugging.
    yaw_xz_deg = math.degrees(math.atan2(front_vector[0], front_vector[2]))
    yaw_xy_deg = math.degrees(math.atan2(front_vector[1], front_vector[0]))

    result = {
        "front_idx": front_idx,
        "rear_left_idx": rear_left_idx,
        "rear_right_idx": rear_right_idx,

        "front": front,
        "rear_left": rear_left,
        "rear_right": rear_right,
        "rear_center": rear_center,
        "front_vector": front_vector,
        "front_vector_norm": front_vector_norm,
        "front_unit": front_unit,
        "yaw_xz_deg": float(yaw_xz_deg),
        "yaw_xy_deg": float(yaw_xy_deg),

        "rear_base_m": distance_3d(rear_left, rear_right),
        "front_to_rear_left_m": distance_3d(front, rear_left),
        "front_to_rear_right_m": distance_3d(front, rear_right)
    }

    return result


def camera_to_navigation(values_c):
    """
    Convert a 3D vector or an array of 3D points from the OpenCV camera frame
    into the camera-aligned navigation frame.

    Camera frame C:      X right, Y down, Z forward
    Navigation frame N:  X forward, Y right, Z down

    Accepted shapes:
        (3,)   -> one vector/point
        (N, 3) -> multiple row-wise vectors/points
    """
    values_c = np.asarray(values_c, dtype=np.float64)

    if values_c.shape == (3,):
        return R_NC @ values_c

    if values_c.ndim == 2 and values_c.shape[1] == 3:
        return (R_NC @ values_c.T).T

    raise ValueError(
        f"camera_to_navigation expects shape (3,) or (N, 3), got {values_c.shape}"
    )


def calculate_zyx_euler_angles(front_c, rear_left_c, rear_right_c):
    """
    Calculate drone yaw, pitch and roll from the three triangulated LEDs.

    Frames
    ------
    OpenCV left-camera frame C:
        X_C = camera right
        Y_C = camera down
        Z_C = camera forward

    Camera-aligned navigation frame N:
        X_N = forward
        Y_N = right
        Z_N = down

    Drone body frame B:
        X_B = drone forward
        Y_B = drone right
        Z_B = drone down

    The function builds the body axes from the LED triangle:
        X_B: rear-center -> front LED
        Y_B: rear-left -> rear-right, made perpendicular to X_B
        Z_B: X_B cross Y_B

    Rotation and Euler convention
    -----------------------------
    R_NB maps body-frame coordinates into navigation-frame coordinates:

        v_N = R_NB @ v_B

    Intrinsic ZYX Euler convention (equivalent matrix product):

        R_NB = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    Returns
    -------
    Dictionary containing angles in degrees, body axes in N, R_NB, and
    validity diagnostics. At gimbal lock, yaw and roll are not individually
    unique; this function sets yaw to zero and returns one valid roll value.
    """
    nan3 = np.full(3, np.nan, dtype=np.float64)
    nan33 = np.full((3, 3), np.nan, dtype=np.float64)

    invalid_result = {
        "valid": False,
        "reason": "invalid_input",
        "yaw_deg": np.nan,
        "pitch_deg": np.nan,
        "roll_deg": np.nan,
        "gimbal_lock": False,
        "rear_center_n": nan3.copy(),
        "front_vector_n": nan3.copy(),
        "body_x_n": nan3.copy(),
        "body_y_n": nan3.copy(),
        "body_z_n": nan3.copy(),
        "R_nb": nan33.copy(),
        "determinant": np.nan,
        "orthogonality_error": np.nan,
    }

    try:
        front_c = np.asarray(front_c, dtype=np.float64).reshape(3)
        rear_left_c = np.asarray(rear_left_c, dtype=np.float64).reshape(3)
        rear_right_c = np.asarray(rear_right_c, dtype=np.float64).reshape(3)
    except (TypeError, ValueError):
        invalid_result["reason"] = "bad_led_point_shape"
        return invalid_result

    if not (
        np.all(np.isfinite(front_c))
        and np.all(np.isfinite(rear_left_c))
        and np.all(np.isfinite(rear_right_c))
    ):
        invalid_result["reason"] = "non_finite_led_point"
        return invalid_result

    # Convert all measured LED coordinates into navigation coordinates first.
    front_n, rear_left_n, rear_right_n = camera_to_navigation(
        np.vstack((front_c, rear_left_c, rear_right_c))
    )

    rear_center_n = 0.5 * (rear_left_n + rear_right_n)
    front_vector_n = front_n - rear_center_n

    # Body X axis: rear-center -> front LED.
    x_norm = float(np.linalg.norm(front_vector_n))
    if x_norm < 1e-9:
        invalid_result["reason"] = "front_axis_too_small"
        return invalid_result
    body_x_n = front_vector_n / x_norm

    # Initial right-axis reference: rear-left -> rear-right.
    y_reference_n = rear_right_n - rear_left_n
    y_reference_norm = float(np.linalg.norm(y_reference_n))
    if y_reference_norm < 1e-9:
        invalid_result["reason"] = "rear_axis_too_small"
        return invalid_result
    y_reference_n = y_reference_n / y_reference_norm

    # Gram-Schmidt: remove any X component from the measured rear baseline.
    # This is important because the measured scalene triangle and stereo noise
    # do not guarantee an exactly perpendicular forward axis and rear baseline.
    body_y_raw_n = y_reference_n - np.dot(y_reference_n, body_x_n) * body_x_n
    y_norm = float(np.linalg.norm(body_y_raw_n))
    if y_norm < 1e-9:
        invalid_result["reason"] = "led_geometry_nearly_collinear"
        return invalid_result
    body_y_n = body_y_raw_n / y_norm

    # Right-handed frame: X_B cross Y_B = Z_B.
    body_z_n = np.cross(body_x_n, body_y_n)
    z_norm = float(np.linalg.norm(body_z_n))
    if z_norm < 1e-9:
        invalid_result["reason"] = "body_z_axis_too_small"
        return invalid_result
    body_z_n = body_z_n / z_norm

    # Recompute Y once to reduce accumulated floating-point error.
    body_y_n = np.cross(body_z_n, body_x_n)
    body_y_n = body_y_n / np.linalg.norm(body_y_n)

    # Columns of R_NB are the body axes expressed in navigation coordinates.
    R_nb = np.column_stack((body_x_n, body_y_n, body_z_n))

    determinant = float(np.linalg.det(R_nb))
    orthogonality_error = float(
        np.linalg.norm(R_nb.T @ R_nb - np.eye(3), ord="fro")
    )

    result_common = {
        "rear_center_n": rear_center_n,
        "front_vector_n": front_vector_n,
        "body_x_n": body_x_n,
        "body_y_n": body_y_n,
        "body_z_n": body_z_n,
        "R_nb": R_nb,
        "determinant": determinant,
        "orthogonality_error": orthogonality_error,
    }

    if (
        not np.isfinite(determinant)
        or abs(determinant - 1.0) > 1e-5
        or orthogonality_error > 1e-5
    ):
        invalid_result.update(result_common)
        invalid_result["reason"] = "invalid_rotation_matrix"
        return invalid_result

    # ZYX extraction for R_NB = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    horizontal = math.hypot(R_nb[0, 0], R_nb[1, 0])
    pitch = math.atan2(-R_nb[2, 0], horizontal)
    gimbal_lock = horizontal < 1e-6

    if not gimbal_lock:
        yaw = math.atan2(R_nb[1, 0], R_nb[0, 0])
        roll = math.atan2(R_nb[2, 1], R_nb[2, 2])
    else:
        # At pitch = +/-90 degrees, yaw and roll are coupled.
        # Choose yaw = 0 as a deterministic representation.
        yaw = 0.0
        roll = math.atan2(-R_nb[1, 2], R_nb[1, 1])

    return {
        "valid": True,
        "reason": "gimbal_lock" if gimbal_lock else "ok",
        "yaw_deg": float(math.degrees(yaw)),
        "pitch_deg": float(math.degrees(pitch)),
        "roll_deg": float(math.degrees(roll)),
        "gimbal_lock": bool(gimbal_lock),
        **result_common,
    }

def find_best_led_triangle(blobsL, blobsR, K1, d1, K2, d2, R, T):
    """
    Finds the best 3D LED triangle by trying possible stereo correspondences.

    Returns:
        result dictionary with:
            valid, reason, score, geometry, labels, selected_left_blobs, selected_right_blobs
    """
    if len(blobsL) < REQUIRED_LED_COUNT or len(blobsR) < REQUIRED_LED_COUNT:
        return {
            "valid": False,
            "reason": "need_3_blobs_in_both_cameras",
            "score": float("inf"),
            "geometry": None,
            "labels": None,
            "selected_left_blobs": None,
            "selected_right_blobs": None,
            "points3d": None
        }

    best = {
        "valid": False,
        "reason": "no_valid_triangle_candidate",
        "score": float("inf"),
        "geometry": None,
        "labels": None,
        "selected_left_blobs": None,
        "selected_right_blobs": None,
        "points3d": None
    }

    left_indices = range(len(blobsL))
    right_indices = range(len(blobsR))

    for left_combo in itertools.combinations(left_indices, REQUIRED_LED_COUNT):
        for right_combo in itertools.combinations(right_indices, REQUIRED_LED_COUNT):
            # Try all possible left->right assignments for the selected 3 points
            for right_perm in itertools.permutations(right_combo, REQUIRED_LED_COUNT):

                points3d = []
                pair_bad = False
                pair_reasons = []

                for li, ri in zip(left_combo, right_perm):
                    bL = blobsL[li]
                    bR = blobsR[ri]

                    pair_ok, raw_dy, pair_reason = validate_pair_before_triangulation(
                        bL["cx"], bL["cy"],
                        bR["cx"], bR["cy"]
                    )

                    if not pair_ok:
                        pair_bad = True
                        pair_reasons.append(pair_reason)
                        break

                    X = triangulate_point_undistorted(
                        (bL["cx"], bL["cy"]),
                        (bR["cx"], bR["cy"]),
                        K1, d1, K2, d2, R, T
                    )

                    valid_3d, reason_3d = validate_3d_point(X)
                    if not valid_3d:
                        pair_bad = True
                        pair_reasons.append(reason_3d)
                        break

                    points3d.append(X)

                if pair_bad:
                    if best["reason"] == "no_valid_triangle_candidate" and len(pair_reasons) > 0:
                        best["reason"] = pair_reasons[0]
                    continue

                points3d = np.array(points3d, dtype=np.float64)

                geom_ok, geom_score, geom_info = score_triangle_geometry(points3d)

                # Add a small preference for larger blob areas because LEDs should be bright/clear.
                # This term is tiny compared with geometry score.
                area_sum = sum(blobsL[i]["area"] for i in left_combo) + sum(blobsR[i]["area"] for i in right_combo)
                area_penalty = 1.0 / max(area_sum, 1.0)
                total_score = float(geom_score + 0.001 * area_penalty)

                if geom_ok and total_score < best["score"]:
                    labels = label_triangle_points(points3d)

                    selected_left = [blobsL[i] for i in left_combo]
                    selected_right = [blobsR[i] for i in right_perm]

                    best = {
                        "valid": True,
                        "reason": "ok",
                        "score": total_score,
                        "geometry": geom_info,
                        "labels": labels,
                        "selected_left_blobs": selected_left,
                        "selected_right_blobs": selected_right,
                        "points3d": points3d
                    }
                elif (not best["valid"]) and geom_score < best["score"]:
                    best["score"] = float(geom_score)
                    best["reason"] = geom_info.get("reason", "triangle_rejected")
                    best["geometry"] = geom_info

    return best


def get_label_blob(result, label_name, side):
    """
    Returns selected left/right blob for a labelled LED.
    label_name: 'front_idx', 'rear_left_idx', or 'rear_right_idx'
    side: 'L' or 'R'
    """
    if not result["valid"] or result["labels"] is None:
        return None

    idx = result["labels"][label_name]

    if side.upper() == "L":
        return result["selected_left_blobs"][idx]
    else:
        return result["selected_right_blobs"][idx]


def draw_blob_candidates(vis, blobs, color=(255, 0, 0)):
    for i, b in enumerate(blobs):
        x, y, w, h = b["bbox"]
        cx = int(round(b["cx"]))
        cy = int(round(b["cy"]))
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 1)
        cv2.circle(vis, (cx, cy), 4, color, -1)
        cv2.putText(
            vis,
            f"{i}",
            (cx + 6, cy - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1
        )


def draw_labelled_triangle(vis, result, side):
    if not result["valid"]:
        return

    labels_to_draw = [
        ("F", "front_idx", (0, 0, 255)),
        ("RL", "rear_left_idx", (255, 0, 255)),
        ("RR", "rear_right_idx", (0, 255, 255)),
    ]

    pts = {}

    for text, key, color in labels_to_draw:
        b = get_label_blob(result, key, side)
        if b is None:
            continue

        cx = int(round(b["cx"]))
        cy = int(round(b["cy"]))
        pts[text] = (cx, cy)

        cv2.circle(vis, (cx, cy), 8, color, 2)
        cv2.putText(
            vis,
            text,
            (cx + 10, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )

    # Draw triangle lines
    if "F" in pts and "RL" in pts:
        cv2.line(vis, pts["F"], pts["RL"], (0, 255, 0), 2)
    if "F" in pts and "RR" in pts:
        cv2.line(vis, pts["F"], pts["RR"], (0, 255, 0), 2)
    if "RL" in pts and "RR" in pts:
        cv2.line(vis, pts["RL"], pts["RR"], (0, 255, 0), 2)


def vector_or_nan3(result, key):
    if not result["valid"] or result["labels"] is None:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    return result["labels"][key]



## Kalman filter and state estimation class


class PositionKalmanFilter3D:
    """
    3D Constant-Velocity Linear Kalman Filter

    State:
        x = [X, Y, Z, Vx, Vy, Vz]^T

    Measurement:
        z = [X_measured, Y_measured, Z_measured]^T

    Model:
        X(k+1) = X(k) + Vx(k) * dt
        Y(k+1) = Y(k) + Vy(k) * dt
        Z(k+1) = Z(k) + Vz(k) * dt

        Vx(k+1) = Vx(k)
        Vy(k+1) = Vy(k)
        Vz(k+1) = Vz(k)
    """

    def __init__(
        self,
        sigma_x=0.02,
        sigma_y=0.02,
        sigma_z=0.03,
        sigma_a=1.0
    ):
        """
        Parameters
        ----------
        sigma_x : float
            Standard deviation of stereo X measurement [m].

        sigma_y : float
            Standard deviation of stereo Y measurement [m].

        sigma_z : float
            Standard deviation of stereo Z measurement [m].

        sigma_a : float
            Assumed standard deviation of unknown drone acceleration [m/s^2].
            This controls process noise Q.
        """

        # ---------------------------------------------------------
        # 1. STATE
        #
        # x = [X, Y, Z, Vx, Vy, Vz]^T
        # ---------------------------------------------------------

        self.x = np.zeros((6, 1))

        # ---------------------------------------------------------
        # 2. MEASUREMENT MATRIX H
        #
        # Stereo camera measures only X, Y, Z.
        # ---------------------------------------------------------

        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        ])

        # ---------------------------------------------------------
        # 3. MEASUREMENT NOISE COVARIANCE R
        #
        # R contains variance, therefore sigma^2.
        # ---------------------------------------------------------

        self.R = np.diag([
            sigma_x ** 2,
            sigma_y ** 2,
            sigma_z ** 2
        ])

        # ---------------------------------------------------------
        # 4. INITIAL STATE COVARIANCE P
        #
        # Position is measured, so uncertainty can be relatively low.
        # Velocity is unknown initially, so uncertainty is larger.
        # ---------------------------------------------------------

        position_variance = 0.1 ** 2
        velocity_variance = 1.0 ** 2

        self.P = np.diag([
            position_variance,
            position_variance,
            position_variance,
            velocity_variance,
            velocity_variance,
            velocity_variance
        ])

        # Assumed acceleration uncertainty
        self.sigma_a = sigma_a

        # Identity matrix for covariance update
        self.I = np.eye(6)

        # Filter initialization flag
        self.initialized = False

    # =============================================================
    # STATE TRANSITION MATRIX
    # =============================================================

    def make_F(self, dt):
        """
        Construct state-transition matrix F for the current dt.
        """

        F = np.array([
            [1.0, 0.0, 0.0, dt,  0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, dt,  0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, dt ],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ])

        return F

    # =============================================================
    # PROCESS NOISE MATRIX
    # =============================================================

    def make_Q(self, dt):
        """
        Construct process noise covariance matrix Q.

        The constant-velocity model assumes acceleration is unknown
        process noise.

        sigma_a represents how much acceleration we expect the drone
        may experience.
        """

        q = self.sigma_a ** 2

        dt2 = dt ** 2
        dt3 = dt ** 3
        dt4 = dt ** 4

        Q = q * np.array([
            [dt4 / 4, 0.0,       0.0,       dt3 / 2, 0.0,       0.0],
            [0.0,     dt4 / 4,   0.0,       0.0,     dt3 / 2,   0.0],
            [0.0,     0.0,       dt4 / 4,   0.0,     0.0,       dt3 / 2],

            [dt3 / 2, 0.0,       0.0,       dt2,     0.0,       0.0],
            [0.0,     dt3 / 2,   0.0,       0.0,     dt2,       0.0],
            [0.0,     0.0,       dt3 / 2,   0.0,     0.0,       dt2]
        ])

        return Q

    # =============================================================
    # INITIALIZATION
    # =============================================================

    def initialize(self, X, Y, Z):
        """
        Initialize the filter using the first valid stereo measurement.

        Initial velocity is assumed to be zero.
        """

        self.x = np.array([
            [X],
            [Y],
            [Z],
            [0.0],
            [0.0],
            [0.0]
        ])

        self.initialized = True

    # =============================================================
    # PREDICTION
    # =============================================================

    def predict(self, dt):
        """
        Kalman filter prediction step.
        """

        F = self.make_F(dt)
        Q = self.make_Q(dt)

        # State prediction
        #
        # x_pred = F x
        self.x = F @ self.x

        # Covariance prediction
        #
        # P_pred = F P F^T + Q
        self.P = F @ self.P @ F.T + Q

    # =============================================================
    # MEASUREMENT UPDATE
    # =============================================================

    def update(self, X_measured, Y_measured, Z_measured):
        """
        Kalman correction/update using new stereo XYZ measurement.
        """

        z = np.array([
            [X_measured],
            [Y_measured],
            [Z_measured]
        ])

        # ---------------------------------------------------------
        # Innovation / measurement residual
        #
        # y = z - H x_pred
        # ---------------------------------------------------------

        y = z - self.H @ self.x

        # ---------------------------------------------------------
        # Innovation covariance
        #
        # S = H P H^T + R
        # ---------------------------------------------------------

        S = self.H @ self.P @ self.H.T + self.R

        # ---------------------------------------------------------
        # Kalman gain
        #
        # K = P H^T S^-1
        #
        # np.linalg.solve is numerically better than explicitly
        # calculating inverse(S).
        # ---------------------------------------------------------

        PHt = self.P @ self.H.T

        K = np.linalg.solve(
            S.T,
            PHt.T
        ).T

        # ---------------------------------------------------------
        # State correction
        #
        # x = x_pred + K y
        # ---------------------------------------------------------

        self.x = self.x + K @ y

        # ---------------------------------------------------------
        # Covariance correction
        #
        # Joseph form:
        #
        # P = (I-KH)P(I-KH)^T + K R K^T
        #
        # Slightly more computationally expensive than
        # P = (I-KH)P
        # but numerically safer.
        # ---------------------------------------------------------

        IKH = self.I - K @ self.H

        self.P = (
            IKH @ self.P @ IKH.T
            + K @ self.R @ K.T
        )

    # =============================================================
    # COMPLETE FILTER STEP
    # =============================================================

    def step(self, X_measured, Y_measured, Z_measured, dt):
        """
        Perform one complete Kalman-filter cycle.

        Input:
            measured stereo X, Y, Z
            dt

        Output:
            filtered position and estimated velocity
        """

        # First measurement initializes filter
        if not self.initialized:

            self.initialize(
                X_measured,
                Y_measured,
                Z_measured
            )

        else:

            # Prediction
            self.predict(dt)

            # Measurement correction
            self.update(
                X_measured,
                Y_measured,
                Z_measured
            )

        return self.get_state()

    # =============================================================
    # OUTPUT
    # =============================================================

    def get_state(self):
        """
        Return the current filtered state.
        """

        return {
            "X": float(self.x[0, 0]),
            "Y": float(self.x[1, 0]),
            "Z": float(self.x[2, 0]),

            "Vx": float(self.x[3, 0]),
            "Vy": float(self.x[4, 0]),
            "Vz": float(self.x[5, 0])
        }
# =========================================================
# Main
# =========================================================
def main():
    # ---------- Start Crazyflie ----------
    cflib.crtp.init_drivers()
    tof = TOFLogger(CF_URI)

    # Allow time for radio connection and first ToF packet.
    time.sleep(2.0)

    # ---------- Loggers ----------
    wb, ws, xlsx_file, csv_fh, csv_writer, csv_file = create_loggers()

    capL = None
    capR = None

    try:
        # ---------- Load calibration ----------
        data = np.load(EXTR_FILE)

        print("Loaded calibration keys:")
        print(data.files)

        K1 = data["K1"].astype(np.float64)
        d1 = data["d1"].astype(np.float64).ravel()
        K2 = data["K2"].astype(np.float64)
        d2 = data["d2"].astype(np.float64).ravel()
        R = data["R"].astype(np.float64)
        T = data["T"].astype(np.float64).reshape(3, 1)

        print("\nBaseline from T:", np.linalg.norm(T), "m")

        if "imageSize" in data.files:
            print("Calibration imageSize:", data["imageSize"])

        # ---------- Open cameras ----------
        capL = setup_camera(LEFT_CAM_INDEX, "LEFT")
        capR = setup_camera(RIGHT_CAM_INDEX, "RIGHT")

        print("\nControls:")
        print("l = left settings")
        print("r = right settings")
        print("b = both settings")
        print("s = save Excel now")
        print("q = quit\n")

        start_time = time.perf_counter()
        prev_loop_time = start_time
        last_print = start_time
        last_log = start_time

        # ---------- Kalman filter initialization ----------
        # Replace sigma_x/y/z with standard deviations from a stationary
        # stereo experiment. The values below are placeholders for now.
        kf = PositionKalmanFilter3D(
            sigma_x=0.02,  # stereo X measurement std [m]
            sigma_y=0.02,  # stereo Y measurement std [m]
            sigma_z=0.03,  # stereo Z measurement std [m]
            sigma_a=1.0    # unmodelled acceleration std [m/s^2]
        )

        while True:
            ok, frameL, frameR, frame_time, cam_grab_dt_ms, cam_retrieve_dt_ms = grab_retrieve_pair(capL, capR)
            if not ok:
                continue

            now = frame_time
            t = now - start_time
            loop_dt_s = now - prev_loop_time
            prev_loop_time = now

            visL = frameL.copy()
            visR = frameR.copy()

            # ---------- Detect LED blobs ----------
            blobsL, thrL_display = detect_led_blobs(frameL)
            blobsR, thrR_display = detect_led_blobs(frameR)

            # Draw all candidate detections
            draw_blob_candidates(visL, blobsL, color=(255, 0, 0))
            draw_blob_candidates(visR, blobsR, color=(255, 0, 0))

            cv2.putText(
                visL,
                f"LEFT candidates: {len(blobsL)}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2
            )
            cv2.putText(
                visR,
                f"RIGHT candidates: {len(blobsR)}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2
            )

            # ---------- Find and label 3D LED triangle ----------
            triangle = find_best_led_triangle(blobsL, blobsR, K1, d1, K2, d2, R, T)

            triangle_detected = 1 if (len(blobsL) >= 3 and len(blobsR) >= 3) else 0
            triangle_valid = 1 if triangle["valid"] else 0
            reject_reason = triangle["reason"]

            # Default values for logging
            front = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
            rear_left = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
            rear_right = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
            rear_center = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
            front_vector = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
            front_unit = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
            front_vector_norm = np.nan
            yaw_xz_deg = np.nan
            yaw_xy_deg = np.nan

            # Navigation-frame and full ZYX Euler attitude defaults
            rear_center_n = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
            front_vector_n = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
            body_x_n = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
            body_y_n = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
            body_z_n = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
            euler_zyx_valid = 0
            euler_zyx_reason = "triangle_invalid"
            yaw_zyx_deg = np.nan
            pitch_zyx_deg = np.nan
            roll_zyx_deg = np.nan
            euler_gimbal_lock = 0
            r_nb_determinant = np.nan
            r_nb_orthogonality_error = np.nan

            rear_base_m = np.nan
            front_to_rear_left_m = np.nan
            front_to_rear_right_m = np.nan
            shape_rms_ratio = np.nan
            triangle_scale = np.nan

            # Kalman filter defaults for this loop.
            # They remain NaN until the filter receives its first valid measurement.
            kf_measurement_used = 0
            kf_prediction_only = 0

            kf_X = np.nan
            kf_Y = np.nan
            kf_Z = np.nan
            kf_Vx = np.nan
            kf_Vy = np.nan
            kf_Vz = np.nan

            kf_sigma_X = np.nan
            kf_sigma_Y = np.nan
            kf_sigma_Z = np.nan
            kf_sigma_Vx = np.nan
            kf_sigma_Vy = np.nan
            kf_sigma_Vz = np.nan

            # Pixel blobs for logging
            bFL = bFR = bRLL = bRLR = bRRL = bRRR = None

            tof_snap = tof.latest
            if tof_snap["tof_local_time"] is not None:
                tof_age_ms = (now - tof_snap["tof_local_time"]) * 1000.0
            else:
                tof_age_ms = -1.0

            if triangle["valid"]:
                draw_labelled_triangle(visL, triangle, side="L")
                draw_labelled_triangle(visR, triangle, side="R")

                labels = triangle["labels"]
                geom = triangle["geometry"]

                front = labels["front"]
                rear_left = labels["rear_left"]
                rear_right = labels["rear_right"]
                rear_center = labels["rear_center"]

                # ---------------------------------------------------------
                # LINEAR KALMAN FILTER
                # Measurement z = [rear_center_X, rear_center_Y, rear_center_Z]^T
                #
                # Scaling correction has already been applied inside
                # triangulate_point_undistorted(), therefore rear_center is
                # the corrected stereo measurement for the filter.
                # ---------------------------------------------------------
                state = kf.step(
                    X_measured=float(rear_center[0]),
                    Y_measured=float(rear_center[1]),
                    Z_measured=float(rear_center[2]),
                    dt=loop_dt_s
                )
                kf_measurement_used = 1

                kf_X = state["X"]
                kf_Y = state["Y"]
                kf_Z = state["Z"]
                kf_Vx = state["Vx"]
                kf_Vy = state["Vy"]
                kf_Vz = state["Vz"]

                # sqrt(diag(P)) gives 1-sigma uncertainty for each state.
                kf_sigma_X = math.sqrt(max(float(kf.P[0, 0]), 0.0))
                kf_sigma_Y = math.sqrt(max(float(kf.P[1, 1]), 0.0))
                kf_sigma_Z = math.sqrt(max(float(kf.P[2, 2]), 0.0))
                kf_sigma_Vx = math.sqrt(max(float(kf.P[3, 3]), 0.0))
                kf_sigma_Vy = math.sqrt(max(float(kf.P[4, 4]), 0.0))
                kf_sigma_Vz = math.sqrt(max(float(kf.P[5, 5]), 0.0))

                front_vector = labels["front_vector"]
                front_vector_norm = labels["front_vector_norm"]
                front_unit = labels["front_unit"]
                yaw_xz_deg = labels["yaw_xz_deg"]
                yaw_xy_deg = labels["yaw_xy_deg"]

                # Calculate full roll, pitch and yaw in the navigation frame.
                euler = calculate_zyx_euler_angles(
                    front,
                    rear_left,
                    rear_right
                )
                euler_zyx_valid = 1 if euler["valid"] else 0
                euler_zyx_reason = euler["reason"]
                yaw_zyx_deg = euler["yaw_deg"]
                pitch_zyx_deg = euler["pitch_deg"]
                roll_zyx_deg = euler["roll_deg"]
                euler_gimbal_lock = 1 if euler["gimbal_lock"] else 0
                rear_center_n = euler["rear_center_n"]
                front_vector_n = euler["front_vector_n"]
                body_x_n = euler["body_x_n"]
                body_y_n = euler["body_y_n"]
                body_z_n = euler["body_z_n"]
                r_nb_determinant = euler["determinant"]
                r_nb_orthogonality_error = euler["orthogonality_error"]

                rear_base_m = labels["rear_base_m"]
                front_to_rear_left_m = labels["front_to_rear_left_m"]
                front_to_rear_right_m = labels["front_to_rear_right_m"]

                shape_rms_ratio = geom["shape_rms_ratio"]
                triangle_scale = geom["scale"]

                bFL = get_label_blob(triangle, "front_idx", "L")
                bFR = get_label_blob(triangle, "front_idx", "R")
                bRLL = get_label_blob(triangle, "rear_left_idx", "L")
                bRLR = get_label_blob(triangle, "rear_left_idx", "R")
                bRRL = get_label_blob(triangle, "rear_right_idx", "L")
                bRRR = get_label_blob(triangle, "rear_right_idx", "R")

                cv2.putText(
                    visL,
                    f"Rear center: X={rear_center[0]:+.3f} Y={rear_center[1]:+.3f} Z={rear_center[2]:+.3f} m",
                    (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    visL,
                    f"KF pos: X={kf_X:+.3f} Y={kf_Y:+.3f} Z={kf_Z:+.3f} m | "
                    f"V=[{kf_Vx:+.2f}, {kf_Vy:+.2f}, {kf_Vz:+.2f}] m/s",
                    (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    visL,
                    f"Front vec: X={front_vector[0]:+.3f} Y={front_vector[1]:+.3f} Z={front_vector[2]:+.3f} | yaw_xz={yaw_xz_deg:+.1f} deg",
                    (10, 135),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    visL,
                    f"ZYX attitude: yaw={yaw_zyx_deg:+.1f} pitch={pitch_zyx_deg:+.1f} roll={roll_zyx_deg:+.1f} deg",
                    (10, 165),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (0, 255, 0) if euler_zyx_valid else (0, 0, 255),
                    2
                )

                cv2.putText(
                    visL,
                    f"base={rear_base_m*100:.1f}cm | F-RL={front_to_rear_left_m*100:.1f}cm | F-RR={front_to_rear_right_m*100:.1f}cm | ToF={tof_snap['tof_front']}mm",
                    (10, 195),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (255, 255, 0),
                    2
                )

                if now - last_print > PRINT_EVERY_SEC:
                    print(
                        f"t={t:.2f}s | "
                        f"rear_center=[{rear_center[0]:+.3f}, {rear_center[1]:+.3f}, {rear_center[2]:+.3f}] m | "
                        f"KF=[{kf_X:+.3f}, {kf_Y:+.3f}, {kf_Z:+.3f}] m | "
                        f"V=[{kf_Vx:+.3f}, {kf_Vy:+.3f}, {kf_Vz:+.3f}] m/s | "
                        f"front_vector=[{front_vector[0]:+.3f}, {front_vector[1]:+.3f}, {front_vector[2]:+.3f}] m | "
                        f"|v|={front_vector_norm:.3f} m | "
                        f"yaw_xz={yaw_xz_deg:+.1f} deg | "
                        f"ZYX[yaw={yaw_zyx_deg:+.1f}, pitch={pitch_zyx_deg:+.1f}, "
                        f"roll={roll_zyx_deg:+.1f}] deg | "
                        f"sides: base={rear_base_m*100:.1f}cm, "
                        f"F-RL={front_to_rear_left_m*100:.1f}cm, "
                        f"F-RR={front_to_rear_right_m*100:.1f}cm | "
                        f"ToF={tof_snap['tof_front']} mm age={tof_age_ms:.0f} ms"
                    )
                    last_print = now

            else:
                # No valid stereo measurement in this frame.
                # If initialized, run prediction only. Never feed zeros as a
                # fake position measurement.
                if kf.initialized and loop_dt_s > 0.0:
                    kf.predict(loop_dt_s)
                    state = kf.get_state()
                    kf_prediction_only = 1

                    kf_X = state["X"]
                    kf_Y = state["Y"]
                    kf_Z = state["Z"]
                    kf_Vx = state["Vx"]
                    kf_Vy = state["Vy"]
                    kf_Vz = state["Vz"]

                    kf_sigma_X = math.sqrt(max(float(kf.P[0, 0]), 0.0))
                    kf_sigma_Y = math.sqrt(max(float(kf.P[1, 1]), 0.0))
                    kf_sigma_Z = math.sqrt(max(float(kf.P[2, 2]), 0.0))
                    kf_sigma_Vx = math.sqrt(max(float(kf.P[3, 3]), 0.0))
                    kf_sigma_Vy = math.sqrt(max(float(kf.P[4, 4]), 0.0))
                    kf_sigma_Vz = math.sqrt(max(float(kf.P[5, 5]), 0.0))

                cv2.putText(
                    visL,
                    f"No valid 3-LED triangle: {reject_reason}",
                    (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2
                )

                if now - last_print > PRINT_EVERY_SEC:
                    print(
                        f"t={t:.2f}s | no valid triangle | "
                        f"L blobs={len(blobsL)}, R blobs={len(blobsR)} | reason={reject_reason}"
                    )
                    last_print = now

            # ---------- Log row ----------
            if now - last_log >= LOG_EVERY_SEC:
                log_dt_s = now - last_log

                def bx(blob):
                    return safe_float(blob["cx"]) if blob is not None else -1.0

                def by(blob):
                    return safe_float(blob["cy"]) if blob is not None else -1.0

                row = [
                    t,
                    loop_dt_s,
                    log_dt_s,
                    cam_grab_dt_ms,
                    cam_retrieve_dt_ms,

                    len(blobsL),
                    len(blobsR),

                    triangle_detected,
                    triangle_valid,
                    reject_reason,
                    safe_float(triangle["score"]),
                    safe_float(shape_rms_ratio),
                    safe_float(triangle_scale),

                    bx(bFL), by(bFL),
                    bx(bFR), by(bFR),

                    bx(bRLL), by(bRLL),
                    bx(bRLR), by(bRLR),

                    bx(bRRL), by(bRRL),
                    bx(bRRR), by(bRRR),

                    safe_float(front[0]), safe_float(front[1]), safe_float(front[2]),

                    safe_float(rear_left[0]), safe_float(rear_left[1]), safe_float(rear_left[2]),

                    safe_float(rear_right[0]), safe_float(rear_right[1]), safe_float(rear_right[2]),

                    safe_float(rear_center[0]), safe_float(rear_center[1]), safe_float(rear_center[2]),

                    1 if kf.initialized else 0,
                    kf_measurement_used,
                    kf_prediction_only,
                    safe_float(kf_X), safe_float(kf_Y), safe_float(kf_Z),
                    safe_float(kf_Vx), safe_float(kf_Vy), safe_float(kf_Vz),
                    safe_float(kf_sigma_X), safe_float(kf_sigma_Y), safe_float(kf_sigma_Z),
                    safe_float(kf_sigma_Vx), safe_float(kf_sigma_Vy), safe_float(kf_sigma_Vz),

                    safe_float(front_vector[0]), safe_float(front_vector[1]), safe_float(front_vector[2]),
                    safe_float(front_vector_norm),
                    safe_float(front_unit[0]), safe_float(front_unit[1]), safe_float(front_unit[2]),

                    safe_float(yaw_xz_deg),
                    safe_float(yaw_xy_deg),

                    safe_float(rear_center_n[0]), safe_float(rear_center_n[1]), safe_float(rear_center_n[2]),
                    safe_float(front_vector_n[0]), safe_float(front_vector_n[1]), safe_float(front_vector_n[2]),
                    safe_float(body_x_n[0]), safe_float(body_x_n[1]), safe_float(body_x_n[2]),
                    safe_float(body_y_n[0]), safe_float(body_y_n[1]), safe_float(body_y_n[2]),
                    safe_float(body_z_n[0]), safe_float(body_z_n[1]), safe_float(body_z_n[2]),

                    euler_zyx_valid,
                    euler_zyx_reason,
                    safe_float(yaw_zyx_deg),
                    safe_float(pitch_zyx_deg),
                    safe_float(roll_zyx_deg),
                    euler_gimbal_lock,
                    safe_float(r_nb_determinant),
                    safe_float(r_nb_orthogonality_error),

                    safe_float(rear_base_m),
                    safe_float(front_to_rear_left_m),
                    safe_float(front_to_rear_right_m),

                    tof_snap["tof_front"],
                    1 if tof_snap["tof_valid"] else 0,
                    safe_float(tof_age_ms),
                    tof_snap["tof_packet_count"],
                    tof_snap["tof_cf_timestamp"] if tof_snap["tof_cf_timestamp"] is not None else -1
                ]

                append_log_row(ws, csv_writer, csv_fh, row)
                last_log = now

            # =================================================
            # DISPLAY
            # =================================================
            # Top row: annotated left/right frames
            top = cv2.hconcat([visL, visR])

            # Bottom row:
            # - default: original camera frames, matching your older display preference
            # - set SHOW_THRESHOLD_BOTTOM=True to debug blob thresholding
            if SHOW_THRESHOLD_BOTTOM:
                thrL_bgr = resize_to_match(make_threshold_bgr(thrL_display), visL)
                thrR_bgr = resize_to_match(make_threshold_bgr(thrR_display), visR)
                bot = cv2.hconcat([thrL_bgr, thrR_bgr])
            else:
                origL_bgr = resize_to_match(frameL.copy(), visL)
                origR_bgr = resize_to_match(frameR.copy(), visR)
                bot = cv2.hconcat([origL_bgr, origR_bgr])

            vis = cv2.vconcat([top, bot])

            cv2.putText(
                vis,
                "Stereo 3-IR-LED triangle tracking | F=front, RL=rear-left, RR=rear-right | q=quit | s=save Excel",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

            vis_display = cv2.resize(
                vis,
                None,
                fx=DISPLAY_SCALE,
                fy=DISPLAY_SCALE,
                interpolation=cv2.INTER_AREA
            )

            cv2.imshow("Stereo 3-IR-LED Triangle Tracking + ToF", vis_display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("s"):
                wb.save(xlsx_file)
                print(f"[LOG] Saved Excel: {xlsx_file}")
                print(f"[LOG] CSV already live: {csv_file}")
            elif key == ord("l"):
                open_camera_settings(capL, "LEFT")
            elif key == ord("r"):
                open_camera_settings(capR, "RIGHT")
            elif key == ord("b"):
                open_camera_settings(capL, "LEFT")
                time.sleep(1.0)
                open_camera_settings(capR, "RIGHT")

    finally:
        try:
            wb.save(xlsx_file)
            print(f"[LOG] Final Excel saved: {xlsx_file}")
            print(f"[LOG] Final CSV saved:   {csv_file}")
        except Exception as e:
            print(f"[LOG] Could not save Excel: {e}")

        try:
            csv_fh.close()
        except Exception:
            pass

        if capL is not None:
            capL.release()
        if capR is not None:
            capR.release()

        cv2.destroyAllWindows()

        print("Closing Crazyflie link...")
        tof.close()


if __name__ == "__main__":
    main()
