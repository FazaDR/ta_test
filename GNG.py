from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Dict

import numpy as np
import open3d as o3d
import glob
import os
import time


# ==========================================================
# TOPOLOGICAL MAP
# ==========================================================

class TopologicalMap:

    def __init__(self, capacity=1024):

        self.node_count = 0
        self.next_node_id = 0

        self.capacity = capacity

        # --------------------------------------------------
        # Node positions
        # --------------------------------------------------

        self.positions = np.empty(
            (self.capacity, 3),
            dtype=np.float32
        )

        # --------------------------------------------------
        # Persistent node IDs
        # --------------------------------------------------

        self.node_ids = np.empty(
            self.capacity,
            dtype=np.int32
        )

        self.id_to_slot: Dict[int, int] = {}

        # --------------------------------------------------
        # Accumulated GNG error E_i
        # --------------------------------------------------

        self.errors = np.zeros(
            self.capacity,
            dtype=np.float32
        )

        # --------------------------------------------------
        # Position topology C_pos
        #
        # graph[a][b] = edge age
        # --------------------------------------------------

        self.graph = defaultdict(dict)


    # ======================================================
    # CAPACITY
    # ======================================================

    def _grow_capacity(self):

        new_capacity = self.capacity * 2

        new_positions = np.empty(
            (new_capacity, 3),
            dtype=np.float32
        )

        new_node_ids = np.empty(
            new_capacity,
            dtype=np.int32
        )

        new_errors = np.zeros(
            new_capacity,
            dtype=np.float32
        )

        new_positions[
            :self.node_count
        ] = self.positions[
            :self.node_count
        ]

        new_node_ids[
            :self.node_count
        ] = self.node_ids[
            :self.node_count
        ]

        new_errors[
            :self.node_count
        ] = self.errors[
            :self.node_count
        ]

        self.positions = new_positions
        self.node_ids = new_node_ids
        self.errors = new_errors

        self.capacity = new_capacity


    # ======================================================
    # NODE OPERATIONS
    # ======================================================

    def add_node(self, position):

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
        self.errors[slot] = 0.0

        self.id_to_slot[node_id] = slot

        self.node_count += 1

        return node_id


    def remove_node(self, node_id):

        if node_id not in self.id_to_slot:
            return

        slot = self.id_to_slot[node_id]

        # --------------------------------------------------
        # Remove all graph connections
        # --------------------------------------------------

        neighbors = list(
            self.graph[node_id].keys()
        )

        for neighbor in neighbors:
            self.remove_edge(
                node_id,
                neighbor
            )

        self.graph.pop(
            node_id,
            None
        )

        # --------------------------------------------------
        # Compact arrays by moving last node into slot
        # --------------------------------------------------

        last_slot = self.node_count - 1

        if slot != last_slot:

            moved_id = int(
                self.node_ids[last_slot]
            )

            self.positions[slot] = (
                self.positions[last_slot]
            )

            self.node_ids[slot] = (
                self.node_ids[last_slot]
            )

            self.errors[slot] = (
                self.errors[last_slot]
            )

            self.id_to_slot[moved_id] = slot

        # --------------------------------------------------
        # Remove ID
        # --------------------------------------------------

        del self.id_to_slot[node_id]

        self.node_count -= 1


    # ======================================================
    # EDGE OPERATIONS
    # ======================================================

    def add_edge(self, a, b):

        if a == b:
            return

        self.graph[a][b] = 0
        self.graph[b][a] = 0


    def remove_edge(self, a, b):

        self.graph[a].pop(
            b,
            None
        )

        self.graph[b].pop(
            a,
            None
        )


    def has_edge(self, a, b):

        return b in self.graph[a]


    def neighbors(self, node_id):

        return self.graph[node_id].keys()


    # ======================================================
    # POSITION ACCESS
    # ======================================================

    def position(self, node_id):

        slot = self.id_to_slot[node_id]

        return self.positions[slot]


    # ======================================================
    # INITIALIZATION
    # ======================================================

    def initialize_from_points(
        self,
        points,
        rng
    ):

        if len(points) < 2:
            raise ValueError(
                "At least two points are required."
            )

        idx = rng.choice(
            len(points),
            size=2,
            replace=False
        )

        n1 = self.add_node(
            points[idx[0]]
        )

        n2 = self.add_node(
            points[idx[1]]
        )

        # Step 0
        self.add_edge(
            n1,
            n2
        )


# ==========================================================
# GNG-DT
# ==========================================================

class GNGDT:

    def __init__(
        self,
        eta1=0.05,
        eta2=0.006,
        lambda_points=4000,
        sample_size=4000,
        alpha_dec=0.5,
        beta=0.0005,
        gmax=2,
        seed=36
    ):

        self.eta1 = eta1
        self.eta2 = eta2

        self.lambda_points = lambda_points
        self.sample_size = sample_size

        self.alpha_dec = alpha_dec
        self.beta = beta

        self.gmax = gmax
        self.rng = np.random.default_rng(seed)

        self.map = TopologicalMap()

        self.input_count = 0


    # ======================================================
    # INITIALIZATION
    # ======================================================

    def initialize(self, points):

        self.map.initialize_from_points(
            points,
            self.rng
        )


    # ======================================================
    # WINNER SEARCH
    # ======================================================

    def winner_search(self, point):

        positions = self.map.positions[
            :self.map.node_count
        ]

        distances = np.linalg.norm(
            positions - point,
            axis=1
        )

        idx = np.argpartition(distances, 1)[:2]

        if distances[idx[0]] > distances[idx[1]]:
            idx = idx[::-1]

        s1_slot, s2_slot = int(idx[0]), int(idx[1])

        s1 = int(
            self.map.node_ids[s1_slot]
        )

        s2 = int(
            self.map.node_ids[s2_slot]
        )

        d1 = float(
            distances[s1_slot]
        )

        d2 = float(
            distances[s2_slot]
        )

        return s1, s2, d1, d2


    # ======================================================
    # STEP 3
    # ======================================================

    def update_error(
        self,
        winner,
        distance
    ):

        slot = self.map.id_to_slot[
            winner
        ]

        self.map.errors[slot] += (
            distance ** 2
        )


    # ======================================================
    # STEP 4
    # ======================================================

    def update_winner(
        self,
        point,
        winner
    ):

        slot = self.map.id_to_slot[
            winner
        ]

        self.map.positions[slot] += (
            self.eta1
            *
            (
                point
                -
                self.map.positions[slot]
            )
        )


    def update_neighbors(
        self,
        point,
        winner
    ):

        for neighbor in list(
            self.map.neighbors(winner)
        ):

            slot = self.map.id_to_slot[
                neighbor
            ]

            self.map.positions[slot] += (
                self.eta2
                *
                (
                    point
                    -
                    self.map.positions[slot]
                )
            )


    # ======================================================
    # STEP 5
    # ======================================================

    def connect_winners(
        self,
        s1,
        s2
    ):

        if self.map.has_edge(
            s1,
            s2
        ):

            self.map.graph[s1][s2] = 0
            self.map.graph[s2][s1] = 0

        else:

            self.map.add_edge(
                s1,
                s2
            )


    # ======================================================
    # STEP 6
    # ======================================================

    def age_edges(
        self,
        winner
    ):

        for neighbor in list(
            self.map.neighbors(winner)
        ):

            self.map.graph[
                winner
            ][neighbor] += 1

            self.map.graph[
                neighbor
            ][winner] += 1


    # ======================================================
    # STEP 7
    # ======================================================

    def remove_old_edges(self, winner):

        old = [
            n for n, age in self.map.graph[winner].items()
            if age > self.gmax
        ]

        for n in old:
            self.map.remove_edge(winner, n)

        # delete nodes that lost their last position edge
        for node_id in [winner] + old:
            if (
                node_id in self.map.id_to_slot
                and len(self.map.graph[node_id]) == 0
            ):
                self.map.remove_node(node_id)


    # ======================================================
    # STEP 8
    # HIGHEST ERROR NODE
    # ======================================================

    def highest_error_node(self):

        if self.map.node_count == 0:
            return None

        errors = self.map.errors[
            :self.map.node_count
        ]

        slot = int(
            np.argmax(errors)
        )

        return int(
            self.map.node_ids[slot]
        )


    # ======================================================
    # STEP 8
    # HIGHEST ERROR NEIGHBOR
    # ======================================================

    def highest_error_neighbor(
        self,
        u
    ):

        neighbors = list(
            self.map.neighbors(u)
        )

        if not neighbors:
            return None

        return max(
            neighbors,
            key=lambda node:
                self.map.errors[
                    self.map.id_to_slot[node]
                ]
        )


    # ======================================================
    # STEP 8
    # INSERT NODE
    # ======================================================

    def insert_node(
        self,
        u,
        f
    ):

        u_slot = self.map.id_to_slot[
            u
        ]

        f_slot = self.map.id_to_slot[
            f
        ]

        # --------------------------------------------------
        # Eq. (8)
        #
        # h_r = 0.5(h_u + h_f)
        # --------------------------------------------------

        new_position = 0.5 * (
            self.map.positions[u_slot]
            +
            self.map.positions[f_slot]
        )

        r = self.map.add_node(
            new_position
        )

        # --------------------------------------------------
        # Remove original u-f edge
        # --------------------------------------------------

        self.map.remove_edge(
            u,
            f
        )

        # --------------------------------------------------
        # Add u-r and r-f
        # --------------------------------------------------

        self.map.add_edge(
            u,
            r
        )

        self.map.add_edge(
            r,
            f
        )

        return r


    # ======================================================
    # STEP 8
    # ERROR DECAY FOR u AND f
    # ======================================================

    def decay_two_errors(
        self,
        u,
        f
    ):

        u_slot = self.map.id_to_slot[
            u
        ]

        f_slot = self.map.id_to_slot[
            f
        ]

        self.map.errors[u_slot] *= (
            1.0 - self.alpha_dec
        )

        self.map.errors[f_slot] *= (
            1.0 - self.alpha_dec
        )


    # ======================================================
    # STEP 8
    # COMPLETE INSERTION
    # ======================================================

    def insert_highest_error_node(self):

        if self.map.node_count < 2:      # was < 3, this was blocking all growth
            return

        u = self.highest_error_node()
        if u is None:
            return

        f = self.highest_error_neighbor(u)
        if f is None:
            return

        r = self.insert_node(u, f)

        self.decay_two_errors(u, f)

        # new node inherits the (already decayed) error of u
        self.map.errors[self.map.id_to_slot[r]] = (
            self.map.errors[self.map.id_to_slot[u]]
        )


    # ======================================================
    # STEP 9
    # DECAY ALL ERRORS
    # ======================================================

    def decay_all_errors(self):

        self.map.errors[
            :self.map.node_count
        ] *= (
            1.0 - self.beta
        )


    # ======================================================
    # PROCESS ONE INPUT
    # ======================================================

    def process_point(
        self,
        point
    ):

        # --------------------------------------------------
        # Step 2
        # --------------------------------------------------

        s1, s2, d1, d2 = (
            self.winner_search(
                point
            )
        )

        # --------------------------------------------------
        # Step 3
        # --------------------------------------------------

        self.update_error(
            s1,
            d1
        )

        # --------------------------------------------------
        # Step 4
        # --------------------------------------------------

        self.update_winner(
            point,
            s1
        )

        self.update_neighbors(
            point,
            s1
        )

        # --------------------------------------------------
        # Step 5
        # --------------------------------------------------

        self.connect_winners(
            s1,
            s2
        )

        # --------------------------------------------------
        # Step 6
        # --------------------------------------------------

        self.age_edges(
            s1
        )

        # --------------------------------------------------
        # Step 7
        # --------------------------------------------------

        self.remove_old_edges(s1)

        # --------------------------------------------------
        # Count input
        # --------------------------------------------------

        self.input_count += 1

        # --------------------------------------------------
        # Step 8
        # --------------------------------------------------

        if (
            self.input_count
            % self.lambda_points
            == 0
        ):

            self.insert_highest_error_node()

        # --------------------------------------------------
        # Step 9
        # --------------------------------------------------

        self.decay_all_errors()


    # ======================================================
    # SAMPLE POINTS
    # ======================================================

    def sample_points(self, points):

        if len(points) <= self.sample_size:
            return points

        indices = self.rng.choice(
            len(points),
            self.sample_size,
            replace=False
        )

        return points[indices]


    # ======================================================
    # PROCESS FRAME
    # ======================================================

    def process_frame(
        self,
        point_cloud
    ):

        point_cloud = np.asarray(
            point_cloud,
            dtype=np.float32
        )

        if self.map.node_count == 0:

            self.initialize(
                point_cloud
            )

        sampled = self.sample_points(
            point_cloud
        )

        for point in sampled:

            self.process_point(
                point
            )


# ==========================================================
# LIVE OPEN3D VISUALIZATION
# ==========================================================

# ==========================================================
# GNG-DT VISUALIZER
# ==========================================================

class GNGVisualizer:

    def __init__(self, first_scan):

        self.vis = o3d.visualization.Visualizer()

        self.vis.create_window(
            window_name="GNG-DT",
            width=1400,
            height=900,
        )

        # ---------------------------------
        # Current LiDAR scan
        # ---------------------------------

        self.scan_cloud = (
            o3d.geometry.PointCloud()
        )

        self.scan_cloud.points = (
            o3d.utility.Vector3dVector(
                first_scan
            )
        )

        self.scan_cloud.paint_uniform_color(
            [0.75, 0.75, 0.75]
        )

        # ---------------------------------
        # GNG nodes
        # ---------------------------------

        self.node_cloud = (
            o3d.geometry.PointCloud()
        )

        # ---------------------------------
        # GNG edges
        # ---------------------------------

        self.line_set = (
            o3d.geometry.LineSet()
        )

        # ---------------------------------
        # Add geometries
        # ---------------------------------

        self.vis.add_geometry(
            self.scan_cloud
        )

        self.vis.add_geometry(
            self.node_cloud
        )

        self.vis.add_geometry(
            self.line_set
        )


    # ======================================================
    # UPDATE
    # ======================================================

    def update(self, gng, scan=None):

        # ---------------------------------
        # Current scan
        # ---------------------------------

        if scan is not None:

            self.scan_cloud.points = (
                o3d.utility.Vector3dVector(
                    scan
                )
            )

            self.scan_cloud.paint_uniform_color(
                [0.75, 0.75, 0.75]
            )

            self.vis.update_geometry(
                self.scan_cloud
            )


        # ---------------------------------
        # GNG nodes
        # ---------------------------------

        if gng.map.node_count == 0:

            pts = np.empty(
                (0, 3),
                dtype=np.float32
            )

        else:

            pts = gng.map.positions[
                :gng.map.node_count
            ]


        self.node_cloud.points = (
            o3d.utility.Vector3dVector(
                pts
            )
        )

        self.node_cloud.paint_uniform_color(
            [1, 0, 0]
        )


        # ---------------------------------
        # GNG edges
        # ---------------------------------

        self.line_set.points = (
            o3d.utility.Vector3dVector(
                pts
            )
        )

        edge_slots = []

        for a in gng.map.graph:

            if a not in gng.map.id_to_slot:
                continue

            for b in gng.map.graph[a]:

                if b not in gng.map.id_to_slot:
                    continue

                # Avoid drawing the same undirected
                # edge twice.
                if a >= b:
                    continue

                edge_slots.append([
                    gng.map.id_to_slot[a],
                    gng.map.id_to_slot[b]
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


        self.line_set.lines = (
            o3d.utility.Vector2iVector(
                edge_slots
            )
        )


        # ---------------------------------
        # Edge color
        # ---------------------------------

        colors = np.zeros(
            (len(edge_slots), 3),
            dtype=np.float64
        )

        colors[:] = [0, 1, 0]

        self.line_set.colors = (
            o3d.utility.Vector3dVector(
                colors
            )
        )


        # ---------------------------------
        # Update Open3D
        # ---------------------------------

        self.vis.update_geometry(
            self.node_cloud
        )

        self.vis.update_geometry(
            self.line_set
        )

    # ======================================================
    # CLOSE
    # ======================================================

    def run(self, update_callback):

        self.vis.register_animation_callback(
            update_callback
        )
        self.vis.run()

    def close(self):

        self.vis.destroy_window()


# ==========================================================
# DATASET
# ==========================================================
# ==========================================================
# DATASET
# ==========================================================

parser = argparse.ArgumentParser(
    description="Process and visualize a GNG-DT dataset."
)
parser.add_argument(
    "--dataset",
    default="datasetShowcase",
    help="Dataset directory containing a scans/ folder.",
)
args = parser.parse_args()

SCAN_DIR = os.path.join(args.dataset, "scans")

scan_files = sorted(
    glob.glob(
        os.path.join(
            SCAN_DIR,
            "scan_*.csv"
        )
    )
)

print(
    "Number of scan frames:",
    len(scan_files)
)

if not scan_files:
    raise FileNotFoundError(
        f"No scan files found in {os.path.abspath(SCAN_DIR)}"
    )


# ==========================================================
# GNG-DT
# ==========================================================

gng = GNGDT(
    eta1=0.5,
    eta2=0.06,
    lambda_points=500,
    sample_size=4000,
    alpha_dec=0.5,
    beta=0.0005,
    gmax=50,
    seed=36
)


# ==========================================================
# FIRST SCAN
# ==========================================================

first_scan = np.loadtxt(
    scan_files[0],
    delimiter=",",
    skiprows=5
).astype(
    np.float32
)


# ==========================================================
# VISUALIZER
# ==========================================================

visualizer = GNGVisualizer(
    first_scan
)


# ==========================================================
# FRAME LOOP
# ==========================================================

frame_id = 0
current_points = None
current_sampled = None
point_cursor = 0
chunk_size = 250
simulation_delay = 0.09


def update_frame(_vis):

    global frame_id, current_points, current_sampled, point_cursor

    if frame_id >= len(scan_files):
        return True

    if current_points is None:
        current_points = np.loadtxt(
            scan_files[frame_id],
            delimiter=",",
            skiprows=5,
        ).astype(np.float32)

        if gng.map.node_count == 0:
            gng.initialize(current_points)

        current_sampled = gng.sample_points(current_points)
        point_cursor = 0

    end_cursor = min(
        point_cursor + chunk_size,
        len(current_sampled),
    )

    for point in current_sampled[point_cursor:end_cursor]:
        gng.process_point(point)

    point_cursor = end_cursor


    # ---------------------------------
    # Update visualization
    # ---------------------------------

    visualizer.update(
        gng,
        current_points
    )
    time.sleep(simulation_delay)


    # ---------------------------------
    # Statistics
    # ---------------------------------

    edge_count = sum(
        len(v)
        for v in gng.map.graph.values()
    ) // 2

    print(
        f"\r"
        f"Frame: {frame_id:4d} / {len(scan_files)-1:4d} | "
        f"Nodes: {gng.map.node_count:5d} | "
        f"Points: {point_cursor:5d} / {len(current_sampled):5d} | "
        f"Edges: {edge_count:5d} | "
        f"Inputs: {gng.input_count:7d}",
        end=""
    )

    if point_cursor >= len(current_sampled):
        current_points = None
        current_sampled = None
        point_cursor = 0
        frame_id += 1

    return True


visualizer.run(
    update_frame
)


print()
print("Finished.")
