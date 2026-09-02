from pathlib import Path

import numpy as np
import open3d as o3d


# ==========================================================
# Configuration
# ==========================================================

DATASET_FOLDER = "datasetStatic/scans"

# Size of each voxel in meters.
# Try:
#   0.05 -> detailed
#   0.10 -> balanced
#   0.20 -> coarse
VOXEL_SIZE = 0.10

OUTPUT_FILE = "datasetStatic/global_voxel_map.ply"


# ==========================================================
# Load All Scan Frames
# ==========================================================

def load_all_scans(folder):

    folder = Path(folder)

    files = sorted(
        folder.glob("scan_*.csv")
    )

    if len(files) == 0:
        raise RuntimeError(
            f"No scan files found in: {folder}"
        )

    all_points = []

    print(f"Found {len(files)} scan frames.\n")

    for i, file in enumerate(files):

        # Your CSV format contains:
        #
        # # timestamp=...
        # # sensor_x=...
        # # sensor_y=...
        # # sensor_z=...
        # x,y,z
        #
        # Therefore skip the first 5 rows.

        points = np.loadtxt(
            file,
            delimiter=",",
            skiprows=5
        ).astype(np.float32)

        # Handle files containing only one point
        if points.ndim == 1:
            points = points.reshape(1, 3)

        all_points.append(points)

        print(
            f"Loaded {i + 1}/{len(files)} "
            f"| {file.name} "
            f"| {len(points)} points"
        )

    # Merge every frame into one Nx3 array
    merged_points = np.vstack(
        all_points
    )

    return merged_points


# ==========================================================
# Create Global Point Cloud
# ==========================================================

def create_global_point_cloud(points):

    pcd = o3d.geometry.PointCloud()

    pcd.points = (
        o3d.utility.Vector3dVector(points)
    )

    return pcd


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    # ------------------------------------------------------
    # Load all frames
    # ------------------------------------------------------

    merged_points = load_all_scans(
        DATASET_FOLDER
    )

    print()
    print(
        "Total points before downsampling:",
        len(merged_points)
    )

    # ------------------------------------------------------
    # Create global point cloud
    # ------------------------------------------------------

    global_pcd = create_global_point_cloud(
        merged_points
    )

    # ------------------------------------------------------
    # Voxel downsampling
    # ------------------------------------------------------

    voxel_map = global_pcd.voxel_down_sample(
        voxel_size=VOXEL_SIZE
    )

    print(
        "Points after voxel downsampling:",
        len(voxel_map.points)
    )

    # ------------------------------------------------------
    # Save voxel map
    # ------------------------------------------------------

    o3d.io.write_point_cloud(
        OUTPUT_FILE,
        voxel_map
    )

    print()
    print(
        f"Voxel map saved to:\n{OUTPUT_FILE}"
    )

    # ------------------------------------------------------
    # Visualization
    # ------------------------------------------------------

    voxel_map.paint_uniform_color(
        [0.7, 0.7, 0.7]
    )

    o3d.visualization.draw_geometries(
        [voxel_map],
        window_name="Global Voxel Map",
        width=1280,
        height=720
    )