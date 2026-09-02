from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Set

import numpy as np
import matplotlib.pyplot as plt


class TopologicalMap:

    def __init__(self):

        # ======================================================
        # Node storage
        # ======================================================

        self.node_count = 0

        # Stable ID generator
        self.next_node_id = 0

        # Dense storage capacity
        self.capacity = 1024

        # Node positions
        #
        # positions[slot] -> position of the node occupying
        # that slot.
        self.positions = np.empty(
            (self.capacity, 3),
            dtype=np.float32
        )

        # Stable node ID stored at each slot
        #
        # node_ids[slot] -> stable node ID
        self.node_ids = np.empty(
            self.capacity,
            dtype=np.int32
        )

        # Stable node ID -> array slot
        #
        # id_to_slot[node_id] -> slot
        self.id_to_slot: Dict[int, int] = {}

        # ======================================================
        # Positional graph
        # ======================================================

        # graph[a][b] = edge age
        #
        # Edges are stored in both directions.
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

        new_node_ids = np.empty(
            new_capacity,
            dtype=np.int32
        )

        new_positions[:self.node_count] = \
            self.positions[:self.node_count]

        new_node_ids[:self.node_count] = \
            self.node_ids[:self.node_count]

        self.positions = new_positions
        self.node_ids = new_node_ids

        self.capacity = new_capacity

    # ==========================================================
    # Node operations
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

    # ==========================================================
    # Edge operations
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
    # Queries
    # ==========================================================

    def get_slot(self, node_id: int) -> int:

        return self.id_to_slot[node_id]

    def get_position(self, node_id: int) -> np.ndarray:

        return self.positions[
            self.id_to_slot[node_id]
        ]

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

    def num_edges(self) -> int:

        count = 0
        visited = set()

        for a in self.graph:

            for b in self.graph[a]:

                if (b, a) in visited:
                    continue

                visited.add((a, b))
                count += 1

        return count