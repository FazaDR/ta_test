from pathlib import Path

import numpy as np
import open3d as o3d


# ==========================================================
# Configuration
# ==========================================================

# ----------------------------------------------------------
# Reference point clouds
# ----------------------------------------------------------

MERGED_POINT_CLOUD = (
    "../datasetStatic/merged_point_cloud.ply"
)

VOXEL_POINT_CLOUD = (
    "../datasetStatic/global_voxel_map.ply"
)


# ----------------------------------------------------------
# Topology map files
# ----------------------------------------------------------

VIGILANCES = [
    0.1,
    0.3,
    0.5,
    0.7
]


BASE_MAPS = {

    0.1: "baseATC_static_0_1.npz",
    0.3: "baseATC_static_0_3.npz",
    0.5: "baseATC_static_0_5.npz",
    0.7: "baseATC_static_0_7.npz"

}


FREE_AREA_MAPS = {

    0.1: "freeArea_static_0_1.npz",
    0.3: "freeArea_static_0_3.npz",
    0.5: "freeArea_static_0_5.npz",
    0.7: "freeArea_static_0_7.npz"

}


# ----------------------------------------------------------
# Coverage thresholds
#
# A point is considered covered if there is a topology node
# within this distance.
# ----------------------------------------------------------

COVERAGE_THRESHOLDS = [
    0.25,
    0.50,
    1.00
]


# ==========================================================
# Load Point Cloud
# ==========================================================

def load_point_cloud(filename):

    print(
        f"Loading:\n{filename}"
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
# Load Topology
# ==========================================================

def load_topology(filename):

    print(
        f"Loading topology:\n{filename}"
    )

    data = np.load(
        filename
    )

    positions = data[
        "positions"
    ].astype(
        np.float32
    )

    if len(positions) == 0:

        raise RuntimeError(
            f"No topology nodes found in:\n{filename}"
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

    """
    For every point:

        point -> nearest topology node

    Returns:

        distances[i]

    where distances[i] is the Euclidean distance
    from points[i] to its nearest topology node.
    """

    # ------------------------------------------------------
    # Convert nodes to Open3D point cloud
    # ------------------------------------------------------

    node_pcd = o3d.geometry.PointCloud()

    node_pcd.points = (
        o3d.utility.Vector3dVector(
            nodes
        )
    )

    # ------------------------------------------------------
    # KD Tree
    # ------------------------------------------------------

    node_tree = o3d.geometry.KDTreeFlann(
        node_pcd
    )

    # ------------------------------------------------------
    # Distance storage
    # ------------------------------------------------------

    distances = np.empty(
        len(points),
        dtype=np.float32
    )

    # ------------------------------------------------------
    # Find nearest node
    # ------------------------------------------------------

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


# ==========================================================
# Evaluate Map
# ==========================================================

def evaluate_map(
    reference_points,
    topology_nodes
):

    distances = (
        compute_nearest_node_distances(
            reference_points,
            topology_nodes
        )
    )

    # ------------------------------------------------------
    # Mean distance
    # ------------------------------------------------------

    mean_distance = np.mean(
        distances
    )

    # ------------------------------------------------------
    # RMSE
    # ------------------------------------------------------

    rmse = np.sqrt(
        np.mean(
            distances ** 2
        )
    )

    # ------------------------------------------------------
    # Maximum distance
    # ------------------------------------------------------

    max_distance = np.max(
        distances
    )

    # ------------------------------------------------------
    # Coverage
    # ------------------------------------------------------

    coverage = {}

    for threshold in COVERAGE_THRESHOLDS:

        covered = np.sum(
            distances <= threshold
        )

        percentage = (
            covered /
            len(distances)
        ) * 100.0

        coverage[threshold] = percentage

    return {

        "mean_distance":
            mean_distance,

        "rmse":
            rmse,

        "max_distance":
            max_distance,

        "coverage":
            coverage,

        "num_nodes":
            len(topology_nodes),

        "num_points":
            len(reference_points)

    }


# ==========================================================
# Print Individual Result
# ==========================================================

def print_result(
    method,
    vigilance,
    result
):

    print(
        "=" * 65
    )

    print(
        f"{method}"
    )

    print(
        f"Vigilance: {vigilance}"
    )

    print(
        "-" * 65
    )

    print(
        f"Reference points : "
        f"{result['num_points']}"
    )

    print(
        f"Topology nodes   : "
        f"{result['num_nodes']}"
    )

    print()

    print(
        f"Mean distance    : "
        f"{result['mean_distance']:.6f} m"
    )

    print(
        f"RMSE             : "
        f"{result['rmse']:.6f} m"
    )

    print(
        f"Maximum distance : "
        f"{result['max_distance']:.6f} m"
    )

    print()

    for threshold, percentage in (
        result["coverage"].items()
    ):

        print(
            f"Coverage <= "
            f"{threshold:.2f} m : "
            f"{percentage:.2f}%"
        )

    print()


# ==========================================================
# Print Comparison Table
# ==========================================================

def print_comparison(
    title,
    base_results,
    free_results
):

    print()
    print("=" * 140)

    print(title)

    print("=" * 140)

    print(
        f"{'Vig.':<8}"
        f"{'Method':<14}"
        f"{'Nodes':>10}"
        f"{'Mean (m)':>14}"
        f"{'RMSE (m)':>14}"
        f"{'Max (m)':>14}"
        f"{'Cov 0.25m':>14}"
        f"{'Cov 0.50m':>14}"
        f"{'Cov 1.00m':>14}"
    )

    print(
        "-" * 140
    )

    for vigilance in VIGILANCES:

        # ==================================================
        # Base ATC-DT
        # ==================================================

        result = base_results[
            vigilance
        ]

        coverage_025 = (
            result["coverage"][0.25]
        )

        coverage_050 = (
            result["coverage"][0.50]
        )

        coverage_100 = (
            result["coverage"][1.00]
        )

        print(
            f"{vigilance:<8.1f}"
            f"{'Base ATC-DT':<14}"
            f"{result['num_nodes']:>10}"
            f"{result['mean_distance']:>14.6f}"
            f"{result['rmse']:>14.6f}"
            f"{result['max_distance']:>14.6f}"
            f"{coverage_025:>13.2f}%"
            f"{coverage_050:>13.2f}%"
            f"{coverage_100:>13.2f}%"
        )

        # ==================================================
        # Free-Area ATC-DT
        # ==================================================

        result = free_results[
            vigilance
        ]

        coverage_025 = (
            result["coverage"][0.25]
        )

        coverage_050 = (
            result["coverage"][0.50]
        )

        coverage_100 = (
            result["coverage"][1.00]
        )

        print(
            f"{vigilance:<8.1f}"
            f"{'Free Area':<14}"
            f"{result['num_nodes']:>10}"
            f"{result['mean_distance']:>14.6f}"
            f"{result['rmse']:>14.6f}"
            f"{result['max_distance']:>14.6f}"
            f"{coverage_025:>13.2f}%"
            f"{coverage_050:>13.2f}%"
            f"{coverage_100:>13.2f}%"
        )

        print(
            "-" * 140
        )


# ==========================================================
# Evaluate One Reference Map
# ==========================================================

def evaluate_reference(
    reference_name,
    reference_points
):

    print()
    print("#" * 70)

    print(
        f"EVALUATING: {reference_name}"
    )

    print("#" * 70)

    base_results = {}

    free_results = {}

    # ======================================================
    # Evaluate Base ATC-DT
    # ======================================================

    print()
    print(
        "BASE ATC-DT EVALUATION"
    )

    print(
        "=" * 70
    )

    for vigilance in VIGILANCES:

        filename = BASE_MAPS[
            vigilance
        ]

        nodes = load_topology(
            filename
        )

        result = evaluate_map(
            reference_points,
            nodes
        )

        base_results[
            vigilance
        ] = result

        print_result(
            "Base ATC-DT",
            vigilance,
            result
        )

    # ======================================================
    # Evaluate Free-Area ATC-DT
    # ======================================================

    print()
    print(
        "FREE-AREA ATC-DT EVALUATION"
    )

    print(
        "=" * 70
    )

    for vigilance in VIGILANCES:

        filename = FREE_AREA_MAPS[
            vigilance
        ]

        nodes = load_topology(
            filename
        )

        result = evaluate_map(
            reference_points,
            nodes
        )

        free_results[
            vigilance
        ] = result

        print_result(
            "Free-Area ATC-DT",
            vigilance,
            result
        )

    # ======================================================
    # Comparison Table
    # ======================================================

    print_comparison(
        f"COMPARISON: {reference_name}",
        base_results,
        free_results
    )

    return (
        base_results,
        free_results
    )


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    # ======================================================
    # Load reference point clouds
    # ======================================================

    merged_points = load_point_cloud(
        MERGED_POINT_CLOUD
    )

    voxel_points = load_point_cloud(
        VOXEL_POINT_CLOUD
    )

    # ======================================================
    # Evaluate against merged point cloud
    # ======================================================

    merged_base_results, merged_free_results = (
        evaluate_reference(
            "STATIC MERGED POINT CLOUD",
            merged_points
        )
    )

    # ======================================================
    # Evaluate against voxel point cloud
    # ======================================================

    voxel_base_results, voxel_free_results = (
        evaluate_reference(
            "STATIC VOXEL POINT CLOUD",
            voxel_points
        )
    )

    print()
    print("=" * 70)

    print(
        "EVALUATION FINISHED"
    )

    print("=" * 70)