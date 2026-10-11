from __future__ import annotations

from collections import defaultdict
from typing import Dict
from pathlib import Path

import numpy as np
import open3d as o3d
import time
import matplotlib.pyplot as plt



# ==========================================================
# Topological Map
# ==========================================================

class TopologicalMap:
    """
    Positional topological map.

        G = (V, {h_i}, E)

    where:

        V         : node IDs
        positions : node positions
        graph     : positional edges
    """

    def __init__(self):

        # Number of currently active nodes
        self.node_count = 0

        # Next globally unique node ID
        self.next_node_id = 0

        # Allocated storage capacity
        self.capacity = 1024

        # Dense position storage.
        # Only [0 : node_count] contains active nodes.
        self.positions = np.empty(
            (self.capacity, 3),
            dtype=np.float32
        )

        # Node attributes
        self.normals = np.zeros(
            (self.capacity, 3),
            dtype=np.float32
        )

        self.slope_angles = np.zeros(
            self.capacity,
            dtype=np.float32
        )

        self.traversable = np.zeros(
            self.capacity,
            dtype=bool
        )

        self.contour = np.zeros(
            self.capacity,
            dtype=bool
        )

        # Stable node ID stored at each array slot.
        self.node_ids = np.empty(
            self.capacity,
            dtype=np.int32
        )
        # Stable node ID -> array slot
        self.id_to_slot: Dict[int, int] = {}

        # graph[a][b] = edge age
        self.graph: Dict[int, Dict[int, int]] = defaultdict(dict)

    # ==========================================================
    # Capacity
    # ==========================================================

    def _grow_capacity(self):

        new_capacity = self.capacity * 2

        new_positions = np.empty(
            (new_capacity, 3),
            dtype=np.float32
        )

        new_normals = np.zeros(
            (new_capacity, 3),
            dtype=np.float32
        )

        new_slope = np.zeros(
            new_capacity,
            dtype=np.float32
        )

        new_trav = np.zeros(
            new_capacity,
            dtype=bool
        )

        new_contour = np.zeros(
            new_capacity,
            dtype=bool
        )

        new_node_ids = np.empty(
            new_capacity,
            dtype=np.int32
        )

        new_positions[:self.node_count] = (
            self.positions[:self.node_count]
        )

        new_normals[:self.node_count] = (
            self.normals[:self.node_count]
        )

        new_slope[:self.node_count] = (
            self.slope_angles[:self.node_count]
        )

        new_trav[:self.node_count] = (
            self.traversable[:self.node_count]
        )

        new_contour[:self.node_count] = (
            self.contour[:self.node_count]
        )

        new_node_ids[:self.node_count] = (
            self.node_ids[:self.node_count]
        )

        self.positions = new_positions
        self.normals = new_normals
        self.slope_angles = new_slope
        self.traversable = new_trav
        self.contour = new_contour
        self.node_ids = new_node_ids

        self.capacity = new_capacity

    # ==========================================================
    # Node Operations
    # ==========================================================

    def add_node(self, position: np.ndarray) -> int:

        if self.node_count >= self.capacity:
            self._grow_capacity()

        position = np.asarray(
            position,
            dtype=np.float32
        )

        slot = self.node_count

        node_id = self.next_node_id
        self.next_node_id += 1

        self.positions[slot] = position
        self.node_ids[slot] = node_id

        self.id_to_slot[node_id] = slot

        self.node_count += 1

        return node_id

    def remove_node(self, node_id):

        if node_id not in self.id_to_slot:
            return

        slot = self.id_to_slot[node_id]

        # Remove all graph connections
        for neighbor in list(self.graph[node_id]):
            self.remove_edge(node_id, neighbor)

        self.graph.pop(node_id, None)

        last_slot = self.node_count - 1

        # Swap-delete
        if slot != last_slot:

            moved_id = self.node_ids[last_slot]

            self.positions[slot] = (
                self.positions[last_slot]
            )

            self.normals[slot] = (
                self.normals[last_slot]
            )

            self.slope_angles[slot] = (
                self.slope_angles[last_slot]
            )

            self.traversable[slot] = (
                self.traversable[last_slot]
            )

            self.contour[slot] = (
                self.contour[last_slot]
            )

            self.node_ids[slot] = moved_id

            self.id_to_slot[moved_id] = slot

        del self.id_to_slot[node_id]

        self.node_count -= 1

    # ==========================================================
    # Edge Operations
    # ==========================================================

    def has_edge(self, a: int, b: int) -> bool:

        return b in self.graph[a]

    def add_edge(self, a: int, b: int):

        self.graph[a][b] = 0
        self.graph[b][a] = 0

    def remove_edge(self, a: int, b: int):

        self.graph[a].pop(b, None)
        self.graph[b].pop(a, None)

    def neighbors(self, node_id: int):

        return self.graph[node_id].keys()

    # ==========================================================
    # Utilities
    # ==========================================================

    def get_position(self, node_id: int) -> np.ndarray:

        slot = self.id_to_slot[node_id]

        return self.positions[slot]

    def get_slot(self, node_id: int) -> int:

        return self.id_to_slot[node_id]

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

    def edge_ages(self):

        ages = []
        visited = set()

        for a in self.graph:

            for b, age in self.graph[a].items():

                if (b, a) in visited:
                    continue

                visited.add((a, b))

                ages.append(age)

        return np.asarray(
            ages,
            dtype=np.float32
        )

    
    def edge_ages_of(self, node_id: int):
    
            return np.asarray(
                list(self.graph[node_id].values()),
                dtype=np.float32
            )

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


# ==========================================================
# ATC-DT
# ==========================================================

class ATCDT:

    def __init__(
        self,
        vigilance=0.5,
        lambda_points=4000,
        free_area_sectors=24,
        horizontal_band=0.15,
        free_area_distance=0.5,
        deg_max=np.deg2rad(30),
        theta_thr=np.pi,
        gravity=None
    ):

        self.vigilance = vigilance
        self.lambda_points = lambda_points

        self.free_area_sectors = free_area_sectors
        self.horizontal_band = horizontal_band
        self.free_area_distance = free_area_distance
        self.deg_max = deg_max
        self.theta_thr = theta_thr

        self.gravity = (
            np.array(
                [0.0, -1.0, 0.0],
                dtype=np.float32
            )
            if gravity is None
            else np.asarray(
                gravity,
                dtype=np.float32
            )
        )

        self.deleted_edge_count = 0
        self.deleted_edge_mean = 0.0

        self.map = TopologicalMap()

        self.winner_count = np.zeros(
            self.map.capacity,
            dtype=np.int32
        )

    def initialize(self, point_cloud):

        idx = np.random.choice(
            len(point_cloud),
            2,
            replace=False
        )

        self.add_node(point_cloud[idx[0]])
        self.add_node(point_cloud[idx[1]])

    # ==========================================================
    # Point Sampling
    # ==========================================================

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

    # ==========================================================
    # Node Operations
    # ==========================================================

    def add_node(self, point):

        old_capacity = self.map.capacity

        node_id = self.map.add_node(point)

        if self.map.capacity != old_capacity:

            new_winner_count = np.zeros(
                self.map.capacity,
                dtype=np.int32
            )

            new_winner_count[:self.map.node_count - 1] = (
                self.winner_count[
                    :self.map.node_count - 1
                ]
            )

            self.winner_count = new_winner_count

        slot = self.map.get_slot(node_id)

        # Paper Eq. (3)
        self.winner_count[slot] = 1

        return node_id

    def remove_node(self, node_id):

        if node_id not in self.map.id_to_slot:
            return

        slot = self.map.get_slot(node_id)

        last_slot = self.map.node_count - 1

        # Keep winner count aligned with swap-delete
        if slot != last_slot:

            self.winner_count[slot] = (
                self.winner_count[last_slot]
            )

        self.map.remove_node(node_id)

    # ==========================================================
    # Winner Search
    # ==========================================================

    def winner_search(self, point):

        if self.map.node_count == 0:

            return (
                None,
                None,
                np.inf,
                np.inf
            )

        active_positions = (
            self.map.positions[
                :self.map.node_count
            ]
        )

        dist = np.linalg.norm(
            active_positions - point,
            axis=1
        )

        if self.map.node_count == 1:

            node_id = self.map.node_ids[0]

            return (
                int(node_id),
                None,
                dist[0],
                np.inf
            )

        order = np.argpartition(
            dist,
            1
        )

        s1_slot = order[0]
        s2_slot = order[1]

        if dist[s2_slot] < dist[s1_slot]:

            s1_slot, s2_slot = (
                s2_slot,
                s1_slot
            )

        d1 = dist[s1_slot]
        d2 = dist[s2_slot]

        s1 = int(
            self.map.node_ids[s1_slot]
        )

        s2 = int(
            self.map.node_ids[s2_slot]
        )

        return (
            s1,
            s2,
            d1,
            d2
        )

    # ==========================================================
    # Node Adaptation
    # ==========================================================

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


    def estimate_normal(self, s1):

        neighbor_ids = list(
            self.map.neighbors(s1)
        )

        if len(neighbor_ids) < 3:
            return

        s1_slot = self.map.get_slot(s1)

        p_s1 = self.map.positions[s1_slot]

        neighbor_slots = [
            self.map.get_slot(n)
            for n in neighbor_ids
        ]

        diffs = (
            self.map.positions[neighbor_slots]
            - p_s1
        )

        cov = diffs.T @ diffs

        eigvals, eigvecs = np.linalg.eigh(cov)

        # Eigenvector corresponding to
        # smallest eigenvalue
        self.map.normals[s1_slot] = (
            eigvecs[:, 0]
        )

    def estimate_traversability(self, s1):

        s1_slot = self.map.get_slot(s1)

        normal = self.map.normals[s1_slot]

        norm = np.linalg.norm(normal)

        if norm < 1e-8:
            return

        cos_theta = np.clip(
            np.dot(normal, self.gravity) / norm,
            -1.0,
            1.0
        )

        angle = np.arccos(cos_theta)

        # PCA normal has arbitrary sign
        angle = min(
            angle,
            np.pi - angle
        )

        self.map.slope_angles[s1_slot] = angle

        self.map.traversable[s1_slot] = (
            angle < self.deg_max
        )

    def detect_contour(
        self,
        s1,
        exclude_untraversable=True
    ):

        neighbor_ids = list(
            self.map.neighbors(s1)
        )

        s1_slot = self.map.get_slot(s1)

        if exclude_untraversable:

            confirmed = []

            for nid in neighbor_ids:

                nslot = self.map.get_slot(nid)

                has_normal = (
                    np.linalg.norm(
                        self.map.normals[nslot]
                    ) > 1e-8
                )

                if (
                    has_normal
                    and
                    self.map.traversable[nslot]
                ):

                    confirmed.append(nid)

            neighbor_ids = confirmed

        if len(neighbor_ids) < 2:

            self.map.contour[s1_slot] = True

            return

        p_s1 = self.map.positions[s1_slot]

        g = (
            self.gravity /
            np.linalg.norm(self.gravity)
        )

        ref = np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32
        )

        if abs(np.dot(ref, g)) > 0.9:

            ref = np.array(
                [0.0, 1.0, 0.0],
                dtype=np.float32
            )

        u = np.cross(g, ref)
        u /= np.linalg.norm(u)

        v = np.cross(g, u)

        angles = []

        for nid in neighbor_ids:

            diff = (
                self.map.positions[
                    self.map.get_slot(nid)
                ]
                - p_s1
            )

            diff_proj = (
                diff -
                np.dot(diff, g) * g
            )

            angles.append(
                np.arctan2(
                    np.dot(diff_proj, v),
                    np.dot(diff_proj, u)
                )
            )

        angles = np.sort(
            np.asarray(angles)
        )

        gaps = np.diff(angles)

        gaps = np.append(
            gaps,
            (angles[0] + 2 * np.pi)
            - angles[-1]
        )

        theta_max = np.max(gaps)

        self.map.contour[s1_slot] = (
            theta_max > self.theta_thr
        )




    # ==========================================================
    # Edge Aging
    # ==========================================================

    def remove_old_edges(self, s1, gmax):
    
            remove_edges = []
    
            for neighbor in list(
                self.map.neighbors(s1)
            ):
    
                age = self.map.graph[s1][neighbor]
    
                if age > gmax:
    
                    # ------------------------------------------
                    # Update running mean of deleted edge ages
                    # ------------------------------------------
    
                    self.deleted_edge_count += 1
    
                    self.deleted_edge_mean += (
                        age - self.deleted_edge_mean
                    ) / self.deleted_edge_count
    
                    remove_edges.append(
                        (s1, neighbor)
                    )
    
            for a, b in remove_edges:
    
                self.map.remove_edge(a, b)

    def compute_gthr(self, gamma):

        if len(gamma) == 0:
            return np.inf

        q3 = np.percentile(
            gamma,
            75
        )

        q1 = np.percentile(
            gamma,
            25
        )

        iqr = q3 - q1

        return q3 + iqr


    def compute_gmax(self, s1):

        gamma = self.map.edge_ages_of(s1)

        if len(gamma) == 0:
            return np.inf

        gthr = self.compute_gthr(gamma)

        # No deleted-edge history yet
        if self.deleted_edge_count == 0:
            return gthr

        # Mean age of all previously deleted edges
        gdel = self.deleted_edge_mean

        total = (
            self.deleted_edge_count +
            len(gamma)
        )

        weight_deleted = (
            self.deleted_edge_count /
            total
        )

        gmax = (
            gdel * weight_deleted
            +
            gthr * (1.0 - weight_deleted)
        )

        return gmax
    # ==========================================================
    # Free Area Detection
    # ==========================================================

    def horizontal_scan(
        self,
        point_cloud,
        robot_position
    ):

        point_cloud = np.asarray(
            point_cloud,
            dtype=np.float32
        )

        robot_position = np.asarray(
            robot_position,
            dtype=np.float32
        )

        vertical_difference = np.abs(
            point_cloud[:, 1] -
            robot_position[1]
        )

        mask = (
            vertical_difference <=
            self.horizontal_band
        )

        return point_cloud[mask]

    def find_proximity_points(
        self,
        point_cloud,
        robot_position
    ):

        scan = self.horizontal_scan(
            point_cloud,
            robot_position
        )

        if len(scan) == 0:
            return []

        dx = (
            scan[:, 0] -
            robot_position[0]
        )

        dz = (
            scan[:, 2] -
            robot_position[2]
        )

        distances = np.sqrt(
            dx * dx +
            dz * dz
        )

        angles = np.arctan2(
            dz,
            dx
        )

        angles = np.mod(
            angles,
            2.0 * np.pi
        )

        sector_width = (
            2.0 * np.pi /
            self.free_area_sectors
        )

        sector_indices = (
            angles / sector_width
        ).astype(np.int32)

        proximity_points = []

        for sector in range(
            self.free_area_sectors
        ):

            mask = (
                sector_indices == sector
            )

            if not np.any(mask):
                continue

            sector_indices_local = np.where(
                mask
            )[0]

            closest_local = (
                sector_indices_local[
                    np.argmin(
                        distances[
                            sector_indices_local
                        ]
                    )
                ]
            )

            proximity_points.append(
                scan[closest_local]
            )

        return proximity_points

    def node_in_free_area(
        self,
        node_position,
        proximity_points,
        robot_position
    ):

        if len(proximity_points) == 0:
            return False

        node = np.asarray(
            node_position,
            dtype=np.float32
        )

        robot = np.asarray(
            robot_position,
            dtype=np.float32
        )

        node_2d = np.array([
            node[0],
            node[2]
        ])

        robot_2d = np.array([
            robot[0],
            robot[2]
        ])

        for r in proximity_points:

            r_2d = np.array([
                r[0],
                r[2]
            ])

            q = r_2d - robot_2d

            norm_q = np.linalg.norm(q)

            if norm_q < 1e-6:
                continue

            q /= norm_q

            value = np.dot(
                q,
                r_2d - node_2d
            )

            if value <= 0:
                return False

        return True

    # ==========================================================
    # Free Area Node Deletion
    # ==========================================================

    def delete_free_area_nodes(
        self,
        point_cloud,
        robot_position
    ):

        proximity_points = (
            self.find_proximity_points(
                point_cloud,
                robot_position
            )
        )

        if len(proximity_points) < 2:
            return

        nodes_to_delete = []

        active_positions = (
            self.map.positions[
                :self.map.node_count
            ]
        )

        active_ids = (
            self.map.node_ids[
                :self.map.node_count
            ]
        )

        for slot, node_id in enumerate(active_ids):

            node_position = active_positions[
                slot
            ]

            if not self.node_in_free_area(
                node_position,
                proximity_points,
                robot_position
            ):
                continue

            distances = np.linalg.norm(
                point_cloud - node_position,
                axis=1
            )

            min_distance = np.min(
                distances
            )

            if (
                min_distance >
                self.free_area_distance
            ):

                nodes_to_delete.append(
                    int(node_id)
                )

        # Delete after testing all nodes
        for node_id in nodes_to_delete:

            self.remove_node(node_id)

    # ==========================================================
    # Main Processing
    # ==========================================================

    def process_frame(
        self,
        point_cloud,
        robot_position
    ):

        sampled = self.sample_points(
            point_cloud
        )

        for point in sampled:

            s1, s2, d1, d2 = (
                self.winner_search(point)
            )

            # ----------------------------------------------
            # Case (a):
            # Add a new node
            # ----------------------------------------------

            if d1 > self.vigilance:

                self.add_node(point)

                continue

            # --------------------------------------------------
            # Case 2: Update s1
            # --------------------------------------------------

            self.update_winner(
                point,
                s1
            )

            # --------------------------------------------------
            # Case 3:
            # Update neighbors only when s2
            # is within vigilance
            # --------------------------------------------------

            if d2 < self.vigilance:

                self.update_neighbors(
                    point,
                    s1
                )

            # ----------------------------------------------
            # Age edges connected to s1
            # ----------------------------------------------

            for neighbor in list(
                self.map.neighbors(s1)
            ):

                self.map.graph[s1][neighbor] += 1

                self.map.graph[neighbor][s1] += 1

            # ----------------------------------------------
            # Case (c):
            # Add or reset s1-s2 edge
            # ----------------------------------------------

            if d2 <= self.vigilance:

                if self.map.has_edge(s1, s2):

                    self.map.graph[s1][s2] = 1
                    self.map.graph[s2][s1] = 1

                else:

                    self.map.add_edge(s1, s2)
            

            # ----------------------------------------------
            # Update adaptive edge threshold
            # ----------------------------------------------

            gmax = self.compute_gmax(s1)

            # ----------------------------------------------
            # Remove obsolete edges
            # ----------------------------------------------

            self.remove_old_edges(s1, gmax)

            self.estimate_normal(s1)
            self.estimate_traversability(s1)
            self.detect_contour(s1)

        

        # ----------------------------------------------
        # Free-area node deletion
        # ----------------------------------------------

        self.delete_free_area_nodes(
            point_cloud,
            robot_position
        )


# ==========================================================
# Dataset
# ==========================================================

class ScanDataset:

    def __init__(
        self,
        folder,
        pose_file
    ):

        self.folder = Path(folder)

        self.files = sorted(
            self.folder.glob(
                "scan_*.csv"
            )
        )

        self.poses = np.loadtxt(
            pose_file,
            delimiter=",",
            skiprows=1
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

    def get_pose(self, idx):

        return self.poses[
            idx,
            2:5
        ].astype(np.float32)


# ==========================================================
# Final Map Visualization
# ==========================================================

def visualize_map(atc: ATCDT):

    vis = o3d.visualization.Visualizer()

    vis.create_window(
        "ATC-DT Final Result"
    )

    # ------------------------------------------------------
    # Nodes
    # ------------------------------------------------------

    node_cloud = o3d.geometry.PointCloud()

    positions = atc.map.positions[
        :atc.map.node_count
    ]

    node_cloud.points = (
        o3d.utility.Vector3dVector(
            positions
        )
    )

    # Red nodes
    node_cloud.paint_uniform_color(
        [1, 0, 0]
    )

    vis.add_geometry(
        node_cloud
    )

    # ------------------------------------------------------
    # Edges
    # ------------------------------------------------------

    lines = o3d.geometry.LineSet()

    lines.points = (
        o3d.utility.Vector3dVector(
            positions
        )
    )

    edge_slots = []

    for a, b in atc.map.edges():

        if (
            a not in atc.map.id_to_slot or
            b not in atc.map.id_to_slot
        ):
            continue

        edge_slots.append([
            atc.map.get_slot(a),
            atc.map.get_slot(b)
        ])

    if len(edge_slots) == 0:

        edge_slots = np.empty(
            (0, 2),
            dtype=np.int32
        )

    else:

        edge_slots = np.asarray(
            edge_slots,
            dtype=np.int32
        )

    lines.lines = (
        o3d.utility.Vector2iVector(
            edge_slots
        )
    )

    # Green edges
    colors = np.tile(
        np.array([[0, 1, 0]]),
        (len(edge_slots), 1)
    )

    lines.colors = (
        o3d.utility.Vector3dVector(
            colors
        )
    )

    vis.add_geometry(
        lines
    )

    # ------------------------------------------------------
    # Open window only here
    # ------------------------------------------------------

    vis.run()

    vis.destroy_window()

def node_colors(atc: ATCDT):

    n = atc.map.node_count

    colors = np.tile(
        np.array(
            [0.6, 0.6, 0.6],
            dtype=np.float32
        ),
        (n, 1)
    )

    normal_norms = np.linalg.norm(
        atc.map.normals[:n],
        axis=1
    )

    has_normal = (
        normal_norms > 1e-8
    )

    trav = atc.map.traversable[:n]
    contour = atc.map.contour[:n]

    # Traversable
    colors[
        has_normal & trav
    ] = [0.0, 0.0, 1.0]

    # Traversable + contour
    colors[
        has_normal & trav & contour
    ] = [0.0, 1.0, 0.0]

    # Untraversable
    colors[
        has_normal & ~trav
    ] = [1.0, 0.0, 0.0]

    return colors

def normal_lines(atc: ATCDT, length=0.3):
    """
    Build points/segments for visualizing each node's normal axis
    as a line centered on the node, extending length/2 in both
    directions along h_nor. Sign-agnostic: +n and -n draw identically.
    """
    n = atc.map.node_count

    normal_norms = np.linalg.norm(atc.map.normals[:n], axis=1)
    has_normal = normal_norms > 1e-8

    idx = np.where(has_normal)[0]

    if len(idx) == 0:
        return None, None

    base_points = atc.map.positions[idx]
    unit_normals = atc.map.normals[idx] / normal_norms[idx, None]

    half = length / 2.0
    end_a = base_points - unit_normals * half
    end_b = base_points + unit_normals * half

    points = np.empty((2 * len(idx), 3), dtype=np.float32)
    points[0::2] = end_a
    points[1::2] = end_b

    lines = np.array(
        [[2 * k, 2 * k + 1] for k in range(len(idx))],
        dtype=np.int32
    )

    return points, lines

def visualize_map_attributes(atc: ATCDT, show_normals=False, normal_length=0.3):

    vis = o3d.visualization.Visualizer()
    vis.create_window("ATC-DT Attributes")

    n = atc.map.node_count
    colors = node_colors(atc)

    # -----------------------------
    # Nodes
    # -----------------------------

    node_cloud = o3d.geometry.PointCloud()
    node_cloud.points = o3d.utility.Vector3dVector(atc.map.positions[:n])
    node_cloud.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(node_cloud)

    # -----------------------------
    # Edges — color follows endpoint average
    # -----------------------------

    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(atc.map.positions[:n])

    edge_slots = []
    edge_colors = []

    for a, b in atc.map.edges():

        if a not in atc.map.id_to_slot or b not in atc.map.id_to_slot:
            continue

        sa = atc.map.get_slot(a)
        sb = atc.map.get_slot(b)

        edge_slots.append([sa, sb])
        edge_colors.append((colors[sa] + colors[sb]) / 2.0)

    if len(edge_slots) == 0:

        edge_slots = np.empty(
            (0, 2),
            dtype=np.int32
        )

        edge_colors = np.empty(
            (0, 3),
            dtype=np.float32
        )

    else:

        edge_slots = np.asarray(
            edge_slots,
            dtype=np.int32
        )

        edge_colors = np.asarray(
            edge_colors,
            dtype=np.float32
        )

    lines.lines = o3d.utility.Vector2iVector(
        edge_slots
    )

    lines.colors = o3d.utility.Vector3dVector(
        edge_colors
    )
    vis.add_geometry(lines)

    # -----------------------------
    # Normal direction lines
    # -----------------------------

    if show_normals:

        points, segs = normal_lines(atc, length=normal_length)

        if points is not None:

            normal_set = o3d.geometry.LineSet()
            normal_set.points = o3d.utility.Vector3dVector(points)
            normal_set.lines = o3d.utility.Vector2iVector(segs)
            normal_set.paint_uniform_color([0.0, 0.0, 0.0])  # yellow

            vis.add_geometry(normal_set)

    vis.run()
    vis.destroy_window()


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    dataset = ScanDataset(
        "../datasetLong/scans",
        "../datasetLong/poses.csv"
    )

    atc = ATCDT(
        vigilance=1.0,
        lambda_points=3500
    )

    process_times = []
    node_counts = []
    edge_counts = []

    for i in range(len(dataset)):

        scan = dataset[i]

        pose = dataset.get_pose(i)
        print(
            f"Processing frame {i} "
            f"({len(scan)} points) "
            # f"pose={pose}"
        )

        if atc.map.node_count == 0:

            atc.initialize(scan)
            
        start_time = time.perf_counter()


        atc.process_frame(
            scan,
            pose
        )

        process_time = (
            time.perf_counter()
            - start_time
        )

        node_count = atc.map.node_count

        edge_count = atc.map.num_edges()

        process_times.append(
            process_time
        )

        node_counts.append(
            node_count
        )

        edge_counts.append(
            edge_count
        )

        print(
            f"Frame {i} complete"
        )

        print(
            f"  Process time : "
            f"{process_time:.6f} s"
        )

        print(
            f"  Nodes        : "
            f"{node_count}"
        )

        print(
            f"  Edges        : "
            f"{edge_count}"
        )

    print()
    print("Finished.")
    print("Nodes :", atc.map.node_count)
    print("Edges :", atc.map.num_edges())

    plt.figure()
    plt.scatter(
        node_counts,
        process_times
    )
    plt.xlabel(
        "Node Count"
    )
    plt.ylabel(
        "Process Time (s)"
    )
    plt.title(
        "ATC-DT with Free-Area Deletion: "
        "Process Time vs Node Count"
    )
    plt.grid(
        True
    )
    plt.tight_layout()
    plt.show()

    visualize_map_attributes(
        atc
    )
