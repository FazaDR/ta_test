from __future__ import annotations

import glob
import os

import numpy as np
import open3d as o3d


class DBL_GNG:

    def __init__(
        self,
        feature_number=3,
        maxNodeLength=68,
        L1=0.33,
        L2=0.05,
        newNodeFreq=50,
        maxAge=15,
        newNodeFactor=0.05,
        errorNodeFactor=0.5,
    ):
        self.W = []
        self.c = []
        self.M = maxNodeLength
        self.alpha = L1
        self.beta = L2
        self.gamma = newNodeFreq
        self.theta = maxAge
        self.rho = newNodeFactor
        self.delta = errorNodeFactor
        self.newNodeCount = 0

    def getConnectedNodes(self, index):
        return np.unique(
            np.concatenate(
                (
                    self.c[self.c[:, 0] == index, 1],
                    self.c[self.c[:, 1] == index, 0],
                )
            )
        )

    def pushData(self, x):
        if len(self.W) < 2:
            self.W.append(np.append(x, [0]))
            if len(self.W) == 2:
                self.W = np.asarray(self.W, dtype=np.float32)
                self.c = np.asarray([[0, 1, 0]])
            return

        diff = x - self.W[:, :-1]
        dist = np.linalg.norm(diff, axis=1)
        s1 = np.argmin(dist)
        winner_distance = dist[s1]
        dist[s1] = np.inf
        s2 = np.argmin(dist)
        dist[s1] = winner_distance

        connected = np.logical_or(
            np.logical_and(self.c[:, 0] == s1, self.c[:, 1] == s2),
            np.logical_and(self.c[:, 0] == s2, self.c[:, 1] == s1),
        )
        if np.any(connected):
            self.c[connected, 2] = 0
        else:
            self.c = np.vstack((self.c, [s1, s2, 0]))

        self.c[
            np.logical_or(self.c[:, 0] == s1, self.c[:, 1] == s1),
            2,
        ] += 1
        self.W[s1, -1] += self.alpha * dist[s1]
        self.W[s1, :-1] += self.alpha * (x - self.W[s1, :-1])

        connected_nodes = self.getConnectedNodes(s1)
        self.W[connected_nodes, :-1] += (
            self.beta * (x - self.W[connected_nodes, :-1])
        )

        expired = self.c[self.c[:, 2] >= self.theta, :2]
        self.c = self.c[self.c[:, 2] < self.theta]
        nodes_to_delete = np.unique(expired)
        final_delete = [
            node for node in nodes_to_delete if node not in self.c[:, :2]
        ]

        for node in sorted(final_delete, reverse=True):
            self.c = self.c[
                np.logical_not(
                    np.logical_or(
                        self.c[:, 0] == node,
                        self.c[:, 1] == node,
                    )
                )
            ]
            self.c[self.c[:, 0] > node, 0] -= 1
            self.c[self.c[:, 1] > node, 1] -= 1
            self.W = np.delete(self.W, node, axis=0)

        self.W[:, -1] *= self.delta
        self.newNodeCount += 1

        if self.newNodeCount >= self.gamma and len(self.W) < self.M:
            self.newNodeCount = 0
            q1 = np.argmax(self.W[:, -1])
            connected_nodes = self.getConnectedNodes(q1)
            if len(connected_nodes) == 0:
                return

            q2 = connected_nodes[
                np.argmax(self.W[connected_nodes, -1])
            ]
            self.W[q1, -1] *= self.rho
            self.W[q2, -1] *= self.rho
            q3 = len(self.W)
            new_node = (self.W[q1] + self.W[q2]) * 0.5
            self.W = np.vstack((self.W, new_node))

            self.c = self.c[
                np.logical_not(
                    np.logical_or(
                        np.logical_and(
                            self.c[:, 0] == q1,
                            self.c[:, 1] == q2,
                        ),
                        np.logical_and(
                            self.c[:, 0] == q2,
                            self.c[:, 1] == q1,
                        ),
                    )
                )
            ]
            self.c = np.vstack(
                (self.c, [q1, q3, 0], [q2, q3, 0])
            )

    def initialize_random(self, points, rng):
        if len(points) < 2:
            raise ValueError("Need at least two points")

        indices = rng.choice(len(points), size=2, replace=False)

        self.W = np.asarray(
            [
                np.append(points[indices[0]], 0),
                np.append(points[indices[1]], 0),
            ],
            dtype=np.float32,
        )

        self.c = np.asarray([[0, 1, 0]], dtype=np.int32)

class DBLGNGVisualizer:

    def __init__(self, first_scan):

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            window_name="DBL-GNG",
            width=1400,
            height=900,
        )

        self.scan_cloud = o3d.geometry.PointCloud()
        self.scan_cloud.points = o3d.utility.Vector3dVector(first_scan)
        self.scan_cloud.paint_uniform_color([0.75, 0.75, 0.75])

        self.node_cloud = o3d.geometry.PointCloud()
        self.line_set = o3d.geometry.LineSet()

        self.vis.add_geometry(self.scan_cloud)
        self.vis.add_geometry(self.node_cloud)
        self.vis.add_geometry(self.line_set)

    def update(self, nodes, edges, scan=None):

        if scan is not None:
            self.scan_cloud.points = o3d.utility.Vector3dVector(scan)
            self.scan_cloud.paint_uniform_color([0.75, 0.75, 0.75])
            self.vis.update_geometry(self.scan_cloud)

        points = np.asarray(nodes[:, :-1], dtype=np.float32)
        self.node_cloud.points = o3d.utility.Vector3dVector(points)
        self.node_cloud.paint_uniform_color([1.0, 0.0, 0.0])

        edges = np.asarray(edges[:, :2], dtype=np.int32)
        if edges.size == 0:
            edges = np.empty((0, 2), dtype=np.int32)

        self.line_set.points = o3d.utility.Vector3dVector(points)
        self.line_set.lines = o3d.utility.Vector2iVector(edges)
        self.line_set.colors = o3d.utility.Vector3dVector(
            np.tile([0.0, 1.0, 0.0], (len(edges), 1))
        )

        self.vis.update_geometry(self.node_cloud)
        self.vis.update_geometry(self.line_set)

    def run(self, update_callback):
        self.vis.register_animation_callback(update_callback)
        self.vis.run()

    def close(self):
        self.vis.destroy_window()


SCAN_DIR = "datasetLong/scans"
scan_files = sorted(glob.glob(os.path.join(SCAN_DIR, "scan_*.csv")))

if not scan_files:
    raise FileNotFoundError(
        f"No scan files found in {os.path.abspath(SCAN_DIR)}"
    )

first_scan = np.loadtxt(
    scan_files[0],
    delimiter=",",
    skiprows=5,
).astype(np.float32)

dbl_gng = DBL_GNG(
    feature_number=3,
    maxNodeLength=500,
)

rng = np.random.default_rng(36)

visualizer = DBLGNGVisualizer(first_scan)
frame_id = 0
current_points = None
current_indices = None
point_cursor = 0
chunk_size = 250


def update_frame(_vis):

    global frame_id, current_points, current_indices, point_cursor

    if frame_id >= len(scan_files):
        return True

    if current_points is None:
        current_points = np.loadtxt(
            scan_files[frame_id],
            delimiter=",",
            skiprows=5,
        ).astype(np.float32)
        current_indices = rng.permutation(len(current_points))
        point_cursor = 0

        if frame_id == 0:
            dbl_gng.initialize_random(current_points, rng)

    end_cursor = min(
        point_cursor + chunk_size,
        len(current_indices),
    )

    for index in current_indices[point_cursor:end_cursor]:
        dbl_gng.pushData(current_points[index])

    point_cursor = end_cursor

    visualizer.update(
        np.asarray(dbl_gng.W, dtype=np.float32),
        np.asarray(dbl_gng.c, dtype=np.int32),
        current_points,
    )

    print(
        f"\rFrame: {frame_id:4d} / {len(scan_files) - 1:4d} | "
        f"Points: {point_cursor:5d} / {len(current_points):5d} | "
        f"Nodes: {len(dbl_gng.W):5d} | "
        f"Edges: {len(dbl_gng.c):5d}",
        end="",
    )

    if point_cursor >= len(current_indices):
        current_points = None
        current_indices = None
        point_cursor = 0
        frame_id += 1

    return True


visualizer.run(update_frame)
