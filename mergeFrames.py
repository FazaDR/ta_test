from pathlib import Path

import numpy as np
import open3d as o3d


# ==========================================================
# Configuration
# ==========================================================

DATASET_FOLDER = "datasetStatic/scans"

OUTPUT_FILE = "datasetStatic/merged_point_cloud.ply"


# ==========================================================
# Load and Merge All Scans
# ==========================================================

def load_and_merge_scans(folder):

    folder = Path(folder)

    files = sorted(
        folder.glob("scan_*.csv")
    )

    if len(files) == 0:

        raise RuntimeError(
            f"No scan files found in: {folder}"
        )

    all_points = []

    print(
        f"Found {len(files)} scan files.\n"
    )

    for i, file in enumerate(files):

        # Skip:
        #
        # # timestamp
        # # sensor_x
        # # sensor_y
        # # sensor_z
        # x,y,z

        points = np.loadtxt(
            file,
            delimiter=",",
            skiprows=5
        ).astype(np.float32)

        # Handle a scan containing only one point
        if points.ndim == 1:

            points = points.reshape(
                1,
                3
            )

        all_points.append(
            points
        )

        print(
            f"Loaded {i + 1}/{len(files)} "
            f"| {file.name} "
            f"| {len(points)} points"
        )

    # Merge all frames

    merged_points = np.vstack(
        all_points
    )

    return merged_points


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    # ------------------------------------------------------
    # Merge scans
    # ------------------------------------------------------

    merged_points = load_and_merge_scans(
        DATASET_FOLDER
    )

    print()
    print(
        "Total merged points:",
        len(merged_points)
    )

    # ------------------------------------------------------
    # Create Open3D point cloud
    # ------------------------------------------------------

    merged_pcd = o3d.geometry.PointCloud()

    merged_pcd.points = (
        o3d.utility.Vector3dVector(
            merged_points
        )
    )

    # ------------------------------------------------------
    # Save merged point cloud
    # ------------------------------------------------------

    o3d.io.write_point_cloud(
        OUTPUT_FILE,
        merged_pcd
    )

    print(
        f"Merged point cloud saved to:\n"
        f"{OUTPUT_FILE}"
    )

    # ------------------------------------------------------
    # Visualize
    # ------------------------------------------------------

    merged_pcd.paint_uniform_color(
        [0.7, 0.7, 0.7]
    )

    o3d.visualization.draw_geometries(
        [merged_pcd],
        window_name="Merged Point Cloud",
        width=1280,
        height=720
    )