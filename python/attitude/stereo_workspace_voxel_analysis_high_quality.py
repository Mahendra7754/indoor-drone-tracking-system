# -*- coding: utf-8 -*-
"""
Stereo Camera 3D Workspace / Voxel Analysis
===========================================

Purpose
-------
1. Use the calibrated left-camera pose in the world frame.
2. Load stereo extrinsics R, T from the same calibration file used by the
   real-time tracking script.
3. Derive the right-camera pose in the world frame automatically.
4. Construct the left and right camera viewing frustums from calibrated FOVs.
5. Generate a 3D voxel grid only around the camera-visible region.
6. Classify each voxel as:
       - visible in left camera
       - visible in right camera
       - visible in BOTH cameras
7. Respect the physical floor / marker-center offset.
8. Visualize the stereo-visible voxel volume with PyVista.
9. Save the centers of stereo-visible voxels to CSV.

Coordinate conventions
----------------------
Stereo calibration convention used in your tracking code:

    X_R = R_RL @ X_L + T_RL

because triangulation uses:
    P1 = [I | 0]
    P2 = [R | T]

World-camera pose convention used here:

    X_W = R_WC @ X_C + t_WC

Therefore:
    X_C = R_WC.T @ (X_W - t_WC)

World-frame assumption for floor clipping:
    Z_W is vertical/up, floor is Z_W = 0.

If your latest world calibration differs from the values below,
replace R_WL and t_WL.
"""

import math
import os
import numpy as np

try:
    import pyvista as pv
except ImportError as exc:
    raise SystemExit(
        "\nPyVista is not installed.\n"
        "Install it with:\n\n"
        "    pip install pyvista\n\n"
        "Then run this script again.\n"
    ) from exc

# Global text style for publication figures.
try:
    pv.global_theme.font.family = "times"
    pv.global_theme.font.color = "black"
except Exception:
    pass


# ============================================================
# USER SETTINGS
# ============================================================

# Same stereo calibration file used by your tracking program.
EXTR_FILE = "stereo_extrinsics_charuco.npz"

# ------------------------------------------------------------
# Left-camera pose in WORLD frame
#
# Convention:
#   P_W = R_WL @ P_L + t_WL
#
# These are the values from your room/world calibration discussed
# previously. Replace them if you have a newer calibration.
# ------------------------------------------------------------
R_WL = np.array([
    [-0.9986339, -0.0284713, -0.0438152],
    [ 0.0180450,  0.5990224, -0.8005289],
    [ 0.0490383, -0.8002260, -0.5976903],
], dtype=np.float64)

t_WL = np.array(
    [0.5633495, 2.1895329, 2.9490722],
    dtype=np.float64
)

# ------------------------------------------------------------
# FOV values calculated from your camera calibration
# ------------------------------------------------------------
HFOV_LEFT_DEG  = 54.0
VFOV_LEFT_DEG  = 41.0

HFOV_RIGHT_DEG = 52.0
VFOV_RIGHT_DEG = 40.0

# ------------------------------------------------------------
# Frustum depth limits measured along each camera's optical Z axis.
#
# A camera FOV alone extends infinitely, so a finite far plane is
# necessary to create a finite workspace.
#
# For the FIRST geometric visualization we use 4.8 m as an upper
# analysis limit. Your experimentally validated 2.5--4.5 m tracking
# range should be applied later as a second constraint.
# ------------------------------------------------------------
NEAR_Z_CAMERA_M = 0.30
FAR_Z_CAMERA_M  = 4.80

# Keep a small angular margin from image borders.
# Set 0.0 first if you want the pure theoretical FOV intersection.
FOV_MARGIN_DEG = 0.0

# Voxel edge length.
# Start with 0.10 m for development.
# Later try 0.05 m for a finer final figure.
VOXEL_SIZE_M = 0.10

# ------------------------------------------------------------
# Physical floor and tracked-marker limits in WORLD frame
#
# IMPORTANT:
# The calibrated world reference was defined using the LED marker.
# When the drone rests on the physical floor, the tracked marker
# reference point is approximately 3.1 cm ABOVE the floor.
#
# Therefore:
#     physical floor       : Z_W = -0.031 m
#     minimum marker height: Z_W =  0.000 m
#
# These must NOT be treated as the same plane.
#
# The physical floor is used for visualization.
# The marker minimum is used when classifying valid drone-marker voxels.
# ------------------------------------------------------------
CLIP_TO_PHYSICAL_BOUNDS = True

FLOOR_Z_W_M = -0.031          # physical room floor [m]
MARKER_MIN_Z_W_M = 0.0        # lowest physically possible tracked marker center [m]

CEILING_Z_W_M = None          # physical ceiling, if known, e.g. 3.60
MARKER_MAX_Z_W_M = None       # optional maximum tracked-marker height

# Extra padding around the automatically computed frustum bounds.
GRID_PADDING_M = 0.10

# Save visible voxel centers.
OUTPUT_CSV = "stereo_visible_voxels_world.csv"

# Visualization
SHOW_LEFT_ONLY = False
SHOW_RIGHT_ONLY = False
SHOW_STEREO_VOXELS = True
SHOW_VOXEL_EDGES = False

# ------------------------------------------------------------
# Publication-quality figure settings
# ------------------------------------------------------------
# The colors below follow Farben.m used in the project.
# rb1 = blue, ro = orange, rg1/rg2 = green, hg1/hg2 = greys, hb1 = dark blue.
COLOR_LEFT_FOV = "#0000FF"       # rb1
COLOR_RIGHT_FOV = "#FF7400"      # ro
COLOR_STEREO = "#00B050"         # rg1
COLOR_STEREO_EDGE = "#006400"    # rg2
COLOR_FLOOR = "#E3E3E3"          # hg2
COLOR_GRID = "#6E6F6E"           # hg1
COLOR_TEXT = "#002F5D"           # hb1
COLOR_MARKER_PLANE = "#F39100"   # ho
COLOR_AXIS_X = "#FF0000"         # rr1
COLOR_AXIS_Y = "#00B050"         # rg1
COLOR_AXIS_Z = "#0000FF"         # rb1

BACKGROUND_COLOR = "white"

# Make the voxels more visible than in the original plot.
STEREO_VOXEL_OPACITY = 0.58
FOV_SURFACE_OPACITY = 0.045
FLOOR_OPACITY = 0.32
MARKER_PLANE_OPACITY = 0.10

# Save high-resolution figures automatically.
SAVE_HIGH_RES_FIGURES = True
FIGURE_OUTPUT_DIR = "workspace_figures"
FIGURE_WINDOW_SIZE = (1800, 1200)
SCREENSHOT_SCALE = 2   # 1800x1200 -> 3600x2400 PNG

# Views exported automatically.
# iso : 3-D isometric view
# xz  : side view in the World X-Z plane
# yz  : side view in the World Y-Z plane
# xy  : top view in the World X-Y plane
EXPORT_VIEWS = ("iso", "xz", "yz", "xy")

# Keep an interactive 3-D window after saving the figures.
SHOW_INTERACTIVE_WINDOW = True


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def check_rotation_matrix(R, name="R"):
    """Basic diagnostic for a 3x3 rotation matrix."""
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"{name} must be 3x3, got {R.shape}")

    det = np.linalg.det(R)
    ortho_error = np.linalg.norm(R.T @ R - np.eye(3), ord="fro")

    print(f"{name}: det={det:.8f}, orthogonality error={ortho_error:.3e}")

    if abs(det - 1.0) > 1e-3 or ortho_error > 1e-3:
        print(
            f"WARNING: {name} does not look like a clean rotation matrix. "
            "Check the calibration convention."
        )


def derive_right_camera_pose_world(R_WL, t_WL, R_RL, T_RL):
    """
    Derive the RIGHT-camera pose in the world frame.

    Stereo extrinsic convention:
        P_R = R_RL @ P_L + T_RL

    Camera center of RIGHT expressed in LEFT coordinates:
        C_R_in_L = -R_RL.T @ T_RL

    Right-camera orientation in world:
        R_WR = R_WL @ R_RL.T

    Right-camera center in world:
        t_WR = R_WL @ C_R_in_L + t_WL
    """
    R_RL = np.asarray(R_RL, dtype=np.float64).reshape(3, 3)
    T_RL = np.asarray(T_RL, dtype=np.float64).reshape(3)

    C_R_in_L = -R_RL.T @ T_RL

    R_WR = R_WL @ R_RL.T
    t_WR = R_WL @ C_R_in_L + t_WL

    return R_WR, t_WR


def world_to_camera(points_w, R_WC, t_WC):
    """
    Convert N world-frame points into a camera frame.

    Input:
        points_w : shape (N, 3)

    World-camera convention:
        P_W = R_WC @ P_C + t_WC

    Therefore:
        P_C = R_WC.T @ (P_W - t_WC)
    """
    points_w = np.asarray(points_w, dtype=np.float64)
    return (R_WC.T @ (points_w - t_WC).T).T


def camera_to_world(points_c, R_WC, t_WC):
    """
    Convert camera-frame points into world coordinates.

        P_W = R_WC @ P_C + t_WC
    """
    points_c = np.asarray(points_c, dtype=np.float64)
    return (R_WC @ points_c.T).T + t_WC


def frustum_corners_camera(hfov_deg, vfov_deg, near_z, far_z):
    """
    Return 8 frustum corners in CAMERA coordinates.

    OpenCV camera frame:
        X = right
        Y = down
        Z = forward

    At a given optical depth z:

        half_width  = z * tan(HFOV/2)
        half_height = z * tan(VFOV/2)

    Corner order:
        0..3 = near plane
        4..7 = far plane
    """
    h = math.radians(hfov_deg / 2.0)
    v = math.radians(vfov_deg / 2.0)

    def plane(z):
        x = z * math.tan(h)
        y = z * math.tan(v)

        return np.array([
            [-x, -y, z],
            [ x, -y, z],
            [ x,  y, z],
            [-x,  y, z],
        ], dtype=np.float64)

    return np.vstack((plane(near_z), plane(far_z)))


def inside_camera_fov(
    points_w,
    R_WC,
    t_WC,
    hfov_deg,
    vfov_deg,
    near_z,
    far_z,
    margin_deg=0.0,
):
    """
    Test whether world points lie inside one camera frustum.

    Instead of explicitly calculating atan2 for every voxel, use

        |X/Z| <= tan(HFOV/2)
        |Y/Z| <= tan(VFOV/2)

    which is mathematically equivalent for Z > 0 and is efficient
    for large NumPy arrays.

    Returns
    -------
    mask : boolean array, shape (N,)
    points_c : points expressed in this camera frame
    """
    points_c = world_to_camera(points_w, R_WC, t_WC)

    Xc = points_c[:, 0]
    Yc = points_c[:, 1]
    Zc = points_c[:, 2]

    half_h_deg = hfov_deg / 2.0 - margin_deg
    half_v_deg = vfov_deg / 2.0 - margin_deg

    if half_h_deg <= 0 or half_v_deg <= 0:
        raise ValueError("FOV margin is too large.")

    tan_h = math.tan(math.radians(half_h_deg))
    tan_v = math.tan(math.radians(half_v_deg))

    depth_ok = (Zc >= near_z) & (Zc <= far_z)

    # Avoid division entirely:
    horizontal_ok = np.abs(Xc) <= Zc * tan_h
    vertical_ok   = np.abs(Yc) <= Zc * tan_v

    mask = depth_ok & horizontal_ok & vertical_ok

    return mask, points_c


def frustum_polydata(corners_w):
    """Create PyVista line geometry for a frustum."""
    corners_w = np.asarray(corners_w, dtype=np.float64)

    # Near rectangle, far rectangle, and four connecting edges.
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    lines = []
    for a, b in edges:
        lines.extend([2, a, b])

    poly = pv.PolyData()
    poly.points = corners_w
    poly.lines = np.array(lines, dtype=np.int64)
    return poly


def add_camera_axes(plotter, R_WC, t_WC, name, axis_length=0.28):
    """Draw compact camera axes and an opaque camera label."""
    axis_specs = [
        (0, COLOR_AXIS_X),
        (1, COLOR_AXIS_Y),
        (2, COLOR_AXIS_Z),
    ]

    for axis_index, color in axis_specs:
        end = t_WC + axis_length * R_WC[:, axis_index]
        plotter.add_mesh(
            pv.Line(t_WC, end),
            color=color,
            line_width=4,
        )

    # Opaque white label background prevents the text from looking washed out.
    plotter.add_point_labels(
        np.array([t_WC]),
        [name],
        point_size=10,
        point_color=COLOR_TEXT,
        text_color=COLOR_TEXT,
        font_size=15,
        shape_color="white",
        shape_opacity=1.0,
        fill_shape=True,
        always_visible=True,
    )


def frustum_surface_polydata(corners_w):
    """Create closed frustum faces for a faint transparent FOV fill."""
    corners_w = np.asarray(corners_w, dtype=np.float64)

    # Four-point faces.  VTK face format: [4, i0, i1, i2, i3, ...].
    faces = np.hstack([
        [4, 0, 1, 2, 3],  # near plane
        [4, 4, 7, 6, 5],  # far plane
        [4, 0, 4, 5, 1],
        [4, 1, 5, 6, 2],
        [4, 2, 6, 7, 3],
        [4, 3, 7, 4, 0],
    ]).astype(np.int64)

    return pv.PolyData(corners_w, faces)


def apply_publication_view(plotter, view_name):
    """Set a reproducible camera view for screenshots."""
    view_name = view_name.lower()

    if view_name == "iso":
        plotter.camera_position = "iso"
        plotter.camera.zoom(1.08)
    elif view_name == "xy":
        plotter.view_xy()
        plotter.enable_parallel_projection()
    elif view_name == "xz":
        plotter.view_xz()
        plotter.enable_parallel_projection()
    elif view_name == "yz":
        plotter.view_yz()
        plotter.enable_parallel_projection()
    else:
        raise ValueError(f"Unknown view '{view_name}'. Use iso, xy, xz, or yz.")

    plotter.reset_camera()


def build_workspace_plotter(
    *,
    grid,
    stereo_cells,
    inside_left,
    inside_right,
    left_corners_w,
    right_corners_w,
    t_WL,
    t_WR,
    R_WL,
    R_WR,
    grid_min,
    grid_max,
    approx_stereo_volume,
    off_screen=False,
):
    """Build the same polished scene for interactive and saved figures."""
    plotter = pv.Plotter(
        window_size=FIGURE_WINDOW_SIZE,
        off_screen=off_screen,
    )
    plotter.set_background(BACKGROUND_COLOR)

    # Supersampling anti-aliasing makes edges and text much cleaner in PNGs.
    try:
        plotter.enable_anti_aliasing("ssaa")
    except Exception:
        # Older PyVista/VTK versions may not provide SSAA.
        try:
            plotter.enable_anti_aliasing()
        except Exception:
            pass

    # Camera centres
    plotter.add_mesh(
        pv.Sphere(radius=0.065, center=t_WL, theta_resolution=48, phi_resolution=48),
        color=COLOR_LEFT_FOV,
        smooth_shading=True,
    )
    plotter.add_mesh(
        pv.Sphere(radius=0.065, center=t_WR, theta_resolution=48, phi_resolution=48),
        color=COLOR_RIGHT_FOV,
        smooth_shading=True,
    )

    add_camera_axes(plotter, R_WL, t_WL, "Left camera")
    add_camera_axes(plotter, R_WR, t_WR, "Right camera")

    # FOV surfaces are intentionally faint; the wireframes remain dominant.
    plotter.add_mesh(
        frustum_surface_polydata(left_corners_w),
        color=COLOR_LEFT_FOV,
        opacity=FOV_SURFACE_OPACITY,
        show_edges=False,
    )
    plotter.add_mesh(
        frustum_surface_polydata(right_corners_w),
        color=COLOR_RIGHT_FOV,
        opacity=FOV_SURFACE_OPACITY,
        show_edges=False,
    )

    plotter.add_mesh(
        frustum_polydata(left_corners_w),
        color=COLOR_LEFT_FOV,
        line_width=4,
        label="Left camera FOV",
    )
    plotter.add_mesh(
        frustum_polydata(right_corners_w),
        color=COLOR_RIGHT_FOV,
        line_width=4,
        label="Right camera FOV",
    )

    # Optional one-camera-only regions.
    if SHOW_LEFT_ONLY:
        left_only_mask = inside_left & ~inside_right
        grid.cell_data["tmp_left_only"] = left_only_mask.astype(np.uint8)
        left_only_cells = grid.threshold(
            value=(0.5, 1.5), scalars="tmp_left_only", preference="cell"
        )
        plotter.add_mesh(
            left_only_cells, color="#8495DC", opacity=0.10, show_edges=False,
            label="Left only"
        )

    if SHOW_RIGHT_ONLY:
        right_only_mask = inside_right & ~inside_left
        grid.cell_data["tmp_right_only"] = right_only_mask.astype(np.uint8)
        right_only_cells = grid.threshold(
            value=(0.5, 1.5), scalars="tmp_right_only", preference="cell"
        )
        plotter.add_mesh(
            right_only_cells, color="#F39100", opacity=0.10, show_edges=False,
            label="Right only"
        )

    if SHOW_STEREO_VOXELS:
        plotter.add_mesh(
            stereo_cells,
            color=COLOR_STEREO,
            opacity=STEREO_VOXEL_OPACITY,
            show_edges=SHOW_VOXEL_EDGES,
            edge_color=COLOR_STEREO_EDGE,
            line_width=0.5,
            ambient=0.50,
            diffuse=0.55,
            specular=0.05,
            label="Stereo-visible voxels",
        )

    if CLIP_TO_PHYSICAL_BOUNDS:
        floor_center = (
            0.5 * (grid_min[0] + grid_max[0]),
            0.5 * (grid_min[1] + grid_max[1]),
            FLOOR_Z_W_M,
        )
        floor = pv.Plane(
            center=floor_center,
            direction=(0, 0, 1),
            i_size=(grid_max[0] - grid_min[0]),
            j_size=(grid_max[1] - grid_min[1]),
            i_resolution=10,
            j_resolution=10,
        )
        plotter.add_mesh(
            floor,
            color=COLOR_FLOOR,
            opacity=FLOOR_OPACITY,
            show_edges=True,
            edge_color=COLOR_GRID,
            line_width=0.6,
            label=f"Physical floor Z_W={FLOOR_Z_W_M:+.3f} m",
        )

        marker_plane_center = (
            0.5 * (grid_min[0] + grid_max[0]),
            0.5 * (grid_min[1] + grid_max[1]),
            MARKER_MIN_Z_W_M,
        )
        marker_plane = pv.Plane(
            center=marker_plane_center,
            direction=(0, 0, 1),
            i_size=(grid_max[0] - grid_min[0]),
            j_size=(grid_max[1] - grid_min[1]),
        )
        plotter.add_mesh(
            marker_plane,
            color=COLOR_MARKER_PLANE,
            opacity=MARKER_PLANE_OPACITY,
            show_edges=False,
            label=f"Minimum marker plane Z_W={MARKER_MIN_Z_W_M:+.3f} m",
        )

    # Clean thesis-style bounds.  Keeping the text fully opaque is important.
    plotter.show_grid(
        xtitle="World X [m]",
        ytitle="World Y [m]",
        ztitle="World Z [m]",
        color=COLOR_GRID,
        font_size=16,
        bold=False,
        grid="back",
        location="outer",
        all_edges=True,
    )

    plotter.add_axes(
        xlabel="X", ylabel="Y", zlabel="Z",
        color=COLOR_TEXT,
        line_width=3,
    )

    plotter.add_legend(
        bcolor="white",
        size=(0.28, 0.22),
        loc="upper right",
    )

    plotter.add_text(
        "Theoretical stereo-visible workspace",
        position="upper_left",
        font_size=20,
        color=COLOR_TEXT,
        shadow=False,
    )
    plotter.add_text(
        f"Voxel size: {VOXEL_SIZE_M*100:.0f} cm    "
        f"Approx. volume: {approx_stereo_volume:.2f} m³",
        position=(20, 65),
        font_size=14,
        color="black",
        shadow=False,
    )

    return plotter


def save_publication_views(plotter_builder, output_dir):
    """Render and save 4 high-resolution, reproducible views."""
    os.makedirs(output_dir, exist_ok=True)

    for view_name in EXPORT_VIEWS:
        p = plotter_builder(off_screen=True)
        apply_publication_view(p, view_name)

        out_file = os.path.join(
            output_dir,
            f"stereo_workspace_{view_name}.png",
        )

        # Render once, then save a supersampled screenshot.
        p.show(auto_close=False)
        p.screenshot(
            out_file,
            transparent_background=False,
            scale=SCREENSHOT_SCALE,
        )
        p.close()
        print(f"Saved high-resolution figure: {out_file}")


# ============================================================
# MAIN
# ============================================================

def main():
    # --------------------------------------------------------
    # 1. Load stereo calibration
    # --------------------------------------------------------
    if not os.path.exists(EXTR_FILE):
        raise FileNotFoundError(
            f"\nCould not find '{EXTR_FILE}'.\n"
            "Put this script in the same folder as your stereo calibration "
            "file, or change EXTR_FILE at the top of the script.\n"
        )

    data = np.load(EXTR_FILE)

    if "R" not in data.files or "T" not in data.files:
        raise KeyError(
            f"{EXTR_FILE} must contain R and T. Available keys: {data.files}"
        )

    # In your tracking code:
    #     P2 = [R | T]
    # so these map LEFT-camera coordinates -> RIGHT-camera coordinates.
    R_RL = data["R"].astype(np.float64)
    T_RL = data["T"].astype(np.float64).reshape(3)

    print("\n--- Rotation diagnostics ---")
    check_rotation_matrix(R_WL, "R_WL")
    check_rotation_matrix(R_RL, "R_RL")

    # --------------------------------------------------------
    # 2. Derive right-camera world pose
    # --------------------------------------------------------
    R_WR, t_WR = derive_right_camera_pose_world(
        R_WL=R_WL,
        t_WL=t_WL,
        R_RL=R_RL,
        T_RL=T_RL,
    )

    check_rotation_matrix(R_WR, "R_WR")

    print("\n--- Camera poses in WORLD frame ---")
    print("Left camera center t_WL [m]:")
    print(t_WL)

    print("\nRight camera center t_WR [m]:")
    print(t_WR)

    print("\nR_WR:")
    print(R_WR)

    baseline_world = np.linalg.norm(t_WR - t_WL)
    print(f"\nWorld-frame stereo baseline = {baseline_world:.4f} m")

    print("\n--- Physical reference levels ---")
    print(f"Physical floor Z_W       = {FLOOR_Z_W_M:+.3f} m")
    print(f"Minimum marker Z_W       = {MARKER_MIN_Z_W_M:+.3f} m")
    print(
        f"Marker-to-floor offset   = "
        f"{(MARKER_MIN_Z_W_M - FLOOR_Z_W_M)*100.0:.1f} cm"
    )

    # --------------------------------------------------------
    # 3. Construct the two camera frustums
    # --------------------------------------------------------
    left_corners_c = frustum_corners_camera(
        HFOV_LEFT_DEG,
        VFOV_LEFT_DEG,
        NEAR_Z_CAMERA_M,
        FAR_Z_CAMERA_M,
    )

    right_corners_c = frustum_corners_camera(
        HFOV_RIGHT_DEG,
        VFOV_RIGHT_DEG,
        NEAR_Z_CAMERA_M,
        FAR_Z_CAMERA_M,
    )

    left_corners_w = camera_to_world(left_corners_c, R_WL, t_WL)
    right_corners_w = camera_to_world(right_corners_c, R_WR, t_WR)

    # --------------------------------------------------------
    # 4. Automatically choose a grid around both frustums
    #
    # We do NOT voxelize the full room.
    # --------------------------------------------------------
    all_frustum_points = np.vstack((
        left_corners_w,
        right_corners_w,
        t_WL.reshape(1, 3),
        t_WR.reshape(1, 3),
    ))

    grid_min = np.min(all_frustum_points, axis=0) - GRID_PADDING_M
    grid_max = np.max(all_frustum_points, axis=0) + GRID_PADDING_M

    # Optional physical/world clipping.
    #
    # The voxel grid represents possible MARKER-center positions, so
    # its lower Z limit should be MARKER_MIN_Z_W_M rather than the
    # physical floor level.
    if CLIP_TO_PHYSICAL_BOUNDS:
        grid_min[2] = max(grid_min[2], MARKER_MIN_Z_W_M)

    if MARKER_MAX_Z_W_M is not None:
        grid_max[2] = min(grid_max[2], MARKER_MAX_Z_W_M)
    elif CEILING_Z_W_M is not None:
        # Conservative fallback if only the physical ceiling is known.
        grid_max[2] = min(grid_max[2], CEILING_Z_W_M)

    print("\n--- Automatically selected voxel-grid bounds ---")
    print(f"X_W: {grid_min[0]:+.3f} to {grid_max[0]:+.3f} m")
    print(f"Y_W: {grid_min[1]:+.3f} to {grid_max[1]:+.3f} m")
    print(f"Z_W: {grid_min[2]:+.3f} to {grid_max[2]:+.3f} m")

    # Number of cells along each dimension.
    cell_counts = np.ceil(
        (grid_max - grid_min) / VOXEL_SIZE_M
    ).astype(int)

    nx, ny, nz = cell_counts.tolist()

    if min(nx, ny, nz) <= 0:
        raise RuntimeError(
            "Invalid grid dimensions. Check camera pose and clipping settings."
        )

    total_voxels = nx * ny * nz

    print("\n--- Voxel grid ---")
    print(f"Voxel size = {VOXEL_SIZE_M:.3f} m")
    print(f"Grid cells = {nx} x {ny} x {nz}")
    print(f"Total candidate voxels = {total_voxels:,}")

    # PyVista ImageData dimensions refer to POINTS.
    # Number of cells = dimensions - 1.
    grid = pv.ImageData(
        dimensions=(nx + 1, ny + 1, nz + 1),
        spacing=(VOXEL_SIZE_M, VOXEL_SIZE_M, VOXEL_SIZE_M),
        origin=tuple(grid_min),
    )

    # These are exactly the centers of the VTK cells, in correct ordering.
    voxel_centers_w = grid.cell_centers().points

    # --------------------------------------------------------
    # 5. Test every voxel against LEFT camera
    # --------------------------------------------------------
    inside_left, points_left_c = inside_camera_fov(
        voxel_centers_w,
        R_WL,
        t_WL,
        HFOV_LEFT_DEG,
        VFOV_LEFT_DEG,
        NEAR_Z_CAMERA_M,
        FAR_Z_CAMERA_M,
        FOV_MARGIN_DEG,
    )

    # --------------------------------------------------------
    # 6. Test every voxel against RIGHT camera
    # --------------------------------------------------------
    inside_right, points_right_c = inside_camera_fov(
        voxel_centers_w,
        R_WR,
        t_WR,
        HFOV_RIGHT_DEG,
        VFOV_RIGHT_DEG,
        NEAR_Z_CAMERA_M,
        FAR_Z_CAMERA_M,
        FOV_MARGIN_DEG,
    )

    # --------------------------------------------------------
    # 7. Stereo-visible region = intersection
    # --------------------------------------------------------
    stereo_visible = inside_left & inside_right

    # Physical feasibility test for the TRACKED MARKER center.
    #
    # A marker-center voxel at Z_W < 0 would require the drone to
    # penetrate the physical floor, because the marker sits 3.1 cm
    # above the floor when the drone is resting on it.
    if CLIP_TO_PHYSICAL_BOUNDS:
        stereo_visible &= voxel_centers_w[:, 2] >= MARKER_MIN_Z_W_M

    if MARKER_MAX_Z_W_M is not None:
        stereo_visible &= voxel_centers_w[:, 2] <= MARKER_MAX_Z_W_M
    elif CEILING_Z_W_M is not None:
        stereo_visible &= voxel_centers_w[:, 2] <= CEILING_Z_W_M

    # Classification codes:
    # 0 = neither
    # 1 = left only
    # 2 = right only
    # 3 = both
    classification = np.zeros(total_voxels, dtype=np.uint8)
    classification[inside_left & ~inside_right] = 1
    classification[~inside_left & inside_right] = 2
    classification[stereo_visible] = 3

    grid.cell_data["visibility_class"] = classification
    grid.cell_data["stereo_visible"] = stereo_visible.astype(np.uint8)
    grid.cell_data["left_visible"] = inside_left.astype(np.uint8)
    grid.cell_data["right_visible"] = inside_right.astype(np.uint8)

    visible_points_w = voxel_centers_w[stereo_visible]

    n_left = int(np.count_nonzero(inside_left))
    n_right = int(np.count_nonzero(inside_right))
    n_stereo = int(np.count_nonzero(stereo_visible))

    voxel_volume = VOXEL_SIZE_M ** 3
    approx_stereo_volume = n_stereo * voxel_volume

    print("\n--- Visibility results ---")
    print(f"Left-visible voxels   = {n_left:,}")
    print(f"Right-visible voxels  = {n_right:,}")
    print(f"Stereo-visible voxels = {n_stereo:,}")
    print(
        f"Approx. voxelized stereo-visible volume = "
        f"{approx_stereo_volume:.3f} m^3"
    )

    if n_stereo == 0:
        raise RuntimeError(
            "\nNo stereo-visible voxels were found.\n"
            "Most likely causes:\n"
            "1. wrong camera/world transform convention,\n"
            "2. wrong stereo R,T direction,\n"
            "3. FOV values or far plane incorrect.\n"
        )

    visible_min = np.min(visible_points_w, axis=0)
    visible_max = np.max(visible_points_w, axis=0)

    print("\n--- Bounding box of stereo-visible voxel CENTERS ---")
    print(f"X_W: {visible_min[0]:+.3f} to {visible_max[0]:+.3f} m")
    print(f"Y_W: {visible_min[1]:+.3f} to {visible_max[1]:+.3f} m")
    print(f"Z_W: {visible_min[2]:+.3f} to {visible_max[2]:+.3f} m")

    # --------------------------------------------------------
    # 8. Save visible voxel centers
    # --------------------------------------------------------
    np.savetxt(
        OUTPUT_CSV,
        visible_points_w,
        delimiter=",",
        header="X_W_m,Y_W_m,Z_W_m",
        comments="",
        fmt="%.6f",
    )

    print(f"\nSaved stereo-visible voxel centers to:\n{OUTPUT_CSV}")

    # --------------------------------------------------------
    # 9. Publication-quality PyVista visualization
    # --------------------------------------------------------
    stereo_cells = grid.threshold(
        value=(0.5, 1.5),
        scalars="stereo_visible",
        preference="cell",
    )

    def make_plotter(off_screen=False):
        return build_workspace_plotter(
            grid=grid,
            stereo_cells=stereo_cells,
            inside_left=inside_left,
            inside_right=inside_right,
            left_corners_w=left_corners_w,
            right_corners_w=right_corners_w,
            t_WL=t_WL,
            t_WR=t_WR,
            R_WL=R_WL,
            R_WR=R_WR,
            grid_min=grid_min,
            grid_max=grid_max,
            approx_stereo_volume=approx_stereo_volume,
            off_screen=off_screen,
        )

    if SAVE_HIGH_RES_FIGURES:
        save_publication_views(
            plotter_builder=make_plotter,
            output_dir=FIGURE_OUTPUT_DIR,
        )

    if SHOW_INTERACTIVE_WINDOW:
        plotter = make_plotter(off_screen=False)
        apply_publication_view(plotter, "iso")
        print("\nClose the PyVista window to finish.")
        plotter.show()


if __name__ == "__main__":
    main()
