import numpy as np
import open3d as o3d


# ==========================================================
# Configuration
# ==========================================================

VOXEL_POINT_CLOUD = (
    "../datasetShowcase/global_voxel_map.ply"
)

MAPS = [
    (
        "LOCAL ATC",
        "mapsShow/local_atc_long_1.npz",
        "positions"
    ),
    (
        "LOCAL MLATC LAYER 1",
        "mapsShow/local_mlatc_long_1.npz",
        "layer_0_positions"
    )
]

COVERAGE_THRESHOLDS = [
    0.5,
    0.75,
    1.00
]


# ==========================================================
# Load Voxel Point Cloud
# ==========================================================

def load_point_cloud(filename):

    print(
        f"Loading point cloud:\n{filename}"
    )

    pcd = o3d.io.read_point_cloud(
        filename
    )

    points = np.asarray(
        pcd.points
    ).astype(
        np.float32
    )

    if len(points) == 0:

        raise RuntimeError(
            f"No points found in:\n{filename}"
        )

    print(
        f"Loaded {len(points)} points.\n"
    )

    return points


# ==========================================================
# Load Map Positions
# ==========================================================

def load_map_positions(filename, key):

    print(
        f"Loading topology map:\n{filename}"
    )

    data = np.load(
        filename
    )

    if key not in data:
        raise KeyError(
            f"'{key}' not found in topology map: {filename}"
        )

    positions = data[key].astype(np.float32)

    if len(positions) == 0:

        raise RuntimeError(
            "No nodes found in topology map."
        )

    print(
        f"Nodes: {len(positions)}\n"
    )

    return positions


# ==========================================================
# Nearest Node Distance
# ==========================================================

def compute_nearest_node_distances(
    points,
    nodes
):

    node_pcd = o3d.geometry.PointCloud()

    node_pcd.points = (
        o3d.utility.Vector3dVector(
            nodes
        )
    )

    node_tree = o3d.geometry.KDTreeFlann(
        node_pcd
    )

    distances = np.empty(
        len(points),
        dtype=np.float32
    )

    for i, point in enumerate(points):

        k, indices, squared_distances = (
            node_tree.search_knn_vector_3d(
                point,
                1
            )
        )

        if k > 0:

            distances[i] = np.sqrt(
                squared_distances[0]
            )

        else:

            distances[i] = np.inf

    return distances


def compute_average_node_distance(nodes):

    node_pcd = o3d.geometry.PointCloud()

    node_pcd.points = (
        o3d.utility.Vector3dVector(
            nodes
        )
    )

    node_tree = o3d.geometry.KDTreeFlann(
        node_pcd
    )

    distances = []

    for i, node in enumerate(nodes):

        k, indices, squared_distances = (
            node_tree.search_knn_vector_3d(
                node,
                2
            )
        )

        # k=2 because the first result is the node itself
        if k >= 2:

            nearest_distance = np.sqrt(
                squared_distances[1]
            )

            distances.append(
                nearest_distance
            )

    if len(distances) == 0:
        return np.nan

    return np.mean(distances)

# ==========================================================
# Evaluate
# ==========================================================

def evaluate(
    reference_points,
    map_nodes,
    map_name
):

    distances = compute_nearest_node_distances(
        reference_points,
        map_nodes
    )

    mean_distance = np.mean(
        distances
    )

    rmse = np.sqrt(
        np.mean(
            distances ** 2
        )
    )

    max_distance = np.max(
        distances
    )

    # Average distance between Layer-1 nodes
    avg_node_distance = compute_average_node_distance(
        map_nodes
    )

    print(
        "=" * 70
    )

    print(
        f"{map_name} EVALUATION"
    )

    print(
        "=" * 70
    )

    print(
        f"Reference points : {len(reference_points)}"
    )

    print(
        f"Map nodes        : {len(map_nodes)}"
    )

    print()

    print(
        f"Mean distance    : "
        f"{mean_distance:.6f} m"
    )

    print(
        f"RMSE             : "
        f"{rmse:.6f} m"
    )

    print(
        f"Maximum distance : "
        f"{max_distance:.6f} m"
    )

    print(
        f"Avg node distance: "
        f"{avg_node_distance:.6f} m"
    )

    print()

    for threshold in COVERAGE_THRESHOLDS:

        covered = np.sum(
            distances <= threshold
        )

        coverage = (
            covered /
            len(distances)
        ) * 100.0

        print(
            f"Coverage <= "
            f"{threshold:.2f} m : "
            f"{coverage:.2f}%"
        )

    print(
        "=" * 70
    )


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    # ------------------------------------------------------
    # Load voxel reference
    # ------------------------------------------------------

    voxel_points = load_point_cloud(
        VOXEL_POINT_CLOUD
    )

    for map_name, filename, key in MAPS:

        map_nodes = load_map_positions(
            filename,
            key
        )

        evaluate(
            voxel_points,
            map_nodes,
            map_name
        )