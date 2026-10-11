from __future__ import annotations

from collections import defaultdict
from typing import Dict
import numpy as np
from pathlib import Path
import open3d as o3d


class TopologicalMap:

    def __init__(self):

        self.node_count = 0

        self.positions = np.empty((0, 3), dtype=np.float32)

        self.graph: Dict[int, Dict[int, int]] = defaultdict(dict)

    def edges(self):

        visited = set()
        edges = []

        for a in self.graph:

            for b in self.graph[a]:

                if (b, a) in visited:
                    continue

                visited.add((a, b))
                edges.append((a, b))

        return edges

    def edge_ages_of(self, node_id):

        return np.asarray(
            list(self.graph[node_id].values()),
            dtype=np.float32
        )

    def add_node(self, position: np.ndarray) -> int:

        position = np.asarray(position, dtype=np.float32)

        self.positions = np.vstack([
            self.positions,
            position.reshape(1, 3)
        ])

        node_id = self.node_count
        self.node_count += 1

        return node_id

    def has_edge(self, a: int, b: int) -> bool:

        return b in self.graph[a]

    def add_edge(self, a: int, b: int):

        self.graph[a][b] = 1
        self.graph[b][a] = 1

    def remove_edge(self, a: int, b: int):

        self.graph[a].pop(b, None)
        self.graph[b].pop(a, None)

    def neighbors(self, node_id: int):

        return self.graph[node_id].keys()

    def num_edges(self):

        count = 0
        visited = set()

        for a in self.graph:

            for b in self.graph[a]:

                if (b, a) in visited:
                    continue

                visited.add((a, b))
                count += 1

        return count


class ATCDT:

    def __init__(
        self,
        vigilance=0.5,
        lambda_points=4000
    ):

        self.vigilance = vigilance
        self.lambda_points = lambda_points

        self.deleted_edge_ages = []

        self.map = TopologicalMap()

        self.winner_count = np.empty(
            (0,),
            dtype=np.int32
        )

    def sample_points(self, points):

        n = len(points)

        if n <= self.lambda_points:
            return points

        idx = np.random.choice(
            n,
            self.lambda_points,
            replace=False
        )

        return points[idx]

    def add_node(self, point):

        node_id = self.map.add_node(point)

        self.winner_count = np.concatenate(
            (
                self.winner_count,
                np.array([1], dtype=np.int32)
            )
        )

        return node_id

    def initialize(self, points):

        idx = np.random.choice(
            len(points),
            2,
            replace=False
        )

        self.add_node(points[idx[0]])
        self.add_node(points[idx[1]])

    def winner_search(self, point):

        if self.map.node_count == 0:
            return None, None, np.inf, np.inf

        dist = np.linalg.norm(
            self.map.positions - point,
            axis=1
        )

        if self.map.node_count == 1:

            d = np.linalg.norm(
                self.map.positions[0] - point
            )

            return 0, None, d, np.inf

        order = np.argpartition(dist, 1)

        s1 = order[0]
        s2 = order[1]

        if dist[s2] < dist[s1]:
            s1, s2 = s2, s1

        d1 = dist[s1]
        d2 = dist[s2]

        return s1, s2, d1, d2

    # def update_existing_node(
    #     self,
    #     point,
    #     winner
    # ):

    #     self.winner_count[winner] += 1

    #     lr = 1.0 / (
    #         10 * self.winner_count[winner]
    #     )

    #     self.map.positions[winner] += \
    #         lr * (
    #             point -
    #             self.map.positions[winner]
    #         )

    #     for neighbor in self.map.neighbors(winner):

    #         lr = 1.0 / (
    #             100 * self.winner_count[neighbor]
    #         )

    #         self.map.positions[neighbor] += \
    #             lr * (
    #                 point -
    #                 self.map.positions[neighbor]
    #             )

    # def update_existing_node(self, point, winner, update_neighbors):
    #     self.winner_count[winner] += 1
    #     lr = 1.0 / (10 * self.winner_count[winner])
    #     self.map.positions[winner] += lr * (point - self.map.positions[winner])

    #     if update_neighbors:
    #         for neighbor in self.map.neighbors(winner):
    #             lr = 1.0 / (100 * self.winner_count[neighbor])
    #             self.map.positions[neighbor] += lr * (point - self.map.positions[neighbor])

    def update_neighbors(self, point, winner):

        for neighbor in self.map.neighbors(winner):

            neighbor_slot = self.map.get_slot(neighbor)

            # Eq. (5)
            lr = 1.0 / (
                100 * self.winner_count[neighbor_slot]
            )

            self.map.positions[neighbor_slot] += (
                lr *
                (point - self.map.positions[neighbor_slot])
            )

    def update_winner(self, point, winner):

        winner_slot = self.map.get_slot(winner)

        # M_s1 = M_s1 + 1
        self.winner_count[winner_slot] += 1

        # Eq. (4)
        lr = 1.0 / (
            10 * self.winner_count[winner_slot]
        )

        self.map.positions[winner_slot] += (
            lr *
            (point - self.map.positions[winner_slot])
        )

    def update_edge(self, s1, s2):

        for neighbor in list(
            self.map.neighbors(s1)
        ):

            self.map.graph[s1][neighbor] += 1
            self.map.graph[neighbor][s1] += 1

        if self.map.has_edge(s1, s2):

            self.map.graph[s1][s2] = 1
            self.map.graph[s2][s1] = 1

        else:

            self.map.add_edge(s1, s2)

    def remove_old_edges(self, s1, gmax):

        remove_edges = []

        for neighbor in list(self.map.neighbors(s1)):

            age = self.map.graph[s1][neighbor]

            if age > gmax:

                self.deleted_edge_ages.append(age)

                remove_edges.append(
                    (s1, neighbor)
                )

        for a, b in remove_edges:

            self.map.remove_edge(a, b)

    def process_frame(self, point_cloud):

        sampled = self.sample_points(point_cloud)

        for point in sampled:

            s1, s2, d1, d2 = self.winner_search(point)

            if d1 > self.vigilance:

                self.add_node(point)
                continue


            self.update_winner(
                point,
                s1
            )

            if d2 < self.vigilance:

                self.update_neighbors(
                    point,
                    s1
                )

            for neighbor in list(
                self.map.neighbors(s1)
            ):

                self.map.graph[s1][neighbor] += 1
                self.map.graph[neighbor][s1] += 1

            if d2 < self.vigilance:

                if self.map.has_edge(s1, s2):

                    self.map.graph[s1][s2] = 1
                    self.map.graph[s2][s1] = 1

                else:

                    self.map.add_edge(s1, s2)

            gmax = self.compute_gmax(s1)

            self.remove_old_edges(
                s1,
                gmax
            )

    def compute_gthr(self, gamma):

        if len(gamma) == 0:
            return np.inf

        q1 = np.percentile(gamma, 25)
        q3 = np.percentile(gamma, 75)

        iqr = q3 - q1

        return q3 + iqr

    def compute_gmax(self, s1):

        gamma = self.map.edge_ages_of(s1)

        if len(gamma) == 0:
            return np.inf

        gthr = self.compute_gthr(gamma)

        if len(self.deleted_edge_ages) == 0:
            return gthr

        gdel = np.mean(
            self.deleted_edge_ages
        )

        total = (
            len(gamma) +
            len(self.deleted_edge_ages)
        )

        weight = (
            len(self.deleted_edge_ages) /
            total
        )

        return (
            gdel * weight
            +
            gthr * (1.0 - weight)
        )


class ScanDataset:

    def __init__(self, folder):

        self.folder = Path(folder)

        self.files = sorted(
            self.folder.glob("scan_*.csv")
        )

    def __len__(self):

        return len(self.files)

    def __getitem__(self, idx):

        return np.loadtxt(
            self.files[idx],
            delimiter=",",
            comments="#",
            skiprows=5
        ).astype(np.float32)


def visualize_map(atc: ATCDT):

    vis = o3d.visualization.Visualizer()
    vis.create_window("ATC-DT Result")

    node_cloud = o3d.geometry.PointCloud()

    node_cloud.points = o3d.utility.Vector3dVector(
        atc.map.positions
    )

    node_cloud.paint_uniform_color(
        [1, 0, 0]
    )

    vis.add_geometry(node_cloud)

    lines = o3d.geometry.LineSet()

    lines.points = o3d.utility.Vector3dVector(
        atc.map.positions
    )

    edges = np.asarray(
        atc.map.edges(),
        dtype=np.int32
    )

    if len(edges) > 0:

        lines.lines = o3d.utility.Vector2iVector(
            edges
        )

        colors = np.tile(
            np.array([[0, 1, 0]]),
            (len(edges), 1)
        )

        lines.colors = o3d.utility.Vector3dVector(
            colors
        )

    vis.add_geometry(lines)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":

    dataset = ScanDataset(
        "datasetStatic/scans",
    )

    atc = ATCDT(
        vigilance=0.8,
        lambda_points=5700
    )

    for i in range(len(dataset)):

        scan = dataset[i]

        print(
            f"Processing frame {i} "
            f"({len(scan)} points)"
        )

        if atc.map.node_count == 0:

            atc.initialize(scan)

        sampled = atc.sample_points(scan)

        atc.process_frame(sampled)

    print()
    print("Finished.")
    print("Nodes :", atc.map.node_count)
    print("Edges :", atc.map.num_edges())

    visualize_map(atc)