from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

import numpy as np
from pathlib import Path
import time




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


class TopologicalMap:

    def __init__(self):

        # ==================================================
        # Node storage
        # ==================================================

        self.node_count = 0

        # Globally unique node ID.
        # Never reused after deletion.
        self.next_node_id = 0

        # Dense storage capacity.
        self.capacity = 1024

        # positions[slot] -> xyz
        self.positions = np.empty(
            (self.capacity, 3),
            dtype=np.float32
        )

        # node_ids[slot] -> stable node ID
        self.node_ids = np.empty(
            self.capacity,
            dtype=np.int32
        )

        # stable node ID -> dense slot
        self.id_to_slot: Dict[int, int] = {}

        # ==================================================
        # Positional graph
        # ==================================================

        # graph[a][b] = edge age
        #
        # Edges are stored in both directions.
        self.graph: Dict[int, Dict[int, int]] = {}

    # ======================================================
    # Capacity
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

        new_positions[:self.node_count] = (
            self.positions[:self.node_count]
        )

        new_node_ids[:self.node_count] = (
            self.node_ids[:self.node_count]
        )

        self.positions = new_positions
        self.node_ids = new_node_ids
        self.capacity = new_capacity

    # ======================================================
    # Node operations
    # ======================================================

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
        self.graph[node_id] = {}

        self.node_count += 1

        return node_id

    def get_slot(self, node_id: int) -> int:

        return self.id_to_slot[node_id]

    def node_ids_active(self) -> np.ndarray:

        return self.node_ids[:self.node_count]

    def remove_node(self, node_id: int):

        if node_id not in self.id_to_slot:
            return

        slot = self.id_to_slot[node_id]

        # Remove all graph connections.
        for neighbor in list(
            self.graph.get(node_id, {})
        ):
            self.remove_edge(
                node_id,
                neighbor
            )

        self.graph.pop(
            node_id,
            None
        )

        # --------------------------------------------------
        # Swap-delete
        # --------------------------------------------------

        last_slot = self.node_count - 1

        if slot != last_slot:

            moved_id = self.node_ids[last_slot]

            self.positions[slot] = (
                self.positions[last_slot]
            )

            self.node_ids[slot] = moved_id

            self.id_to_slot[moved_id] = slot

        del self.id_to_slot[node_id]

        self.node_count -= 1



    # ======================================================
    # Edge operations
    # ======================================================

    def has_edge(self, a: int, b: int) -> bool:

        return b in self.graph.get(a, {})

    def add_edge(self, a: int, b: int) -> bool:

        new_edge = not self.has_edge(a, b)

        # self.graph[a][b] = 0
        # self.graph[b][a] = 0
        
        if new_edge:
            self.graph[a][b] = 0
            self.graph[b][a] = 0

        return new_edge

    def remove_edge(self, a: int, b: int):

        self.graph[a].pop(b, None)
        self.graph[b].pop(a, None)

    def neighbors(self, node_id: int):

        return self.graph[node_id].keys()

    def edges(self) -> List[Tuple[int, int]]:

        visited = set()
        edges = []

        for a in self.graph:

            for b in self.graph[a]:

                if (b, a) in visited:
                    continue

                visited.add((a, b))
                edges.append((a, b))

        return edges

    def edge_ages(self) -> np.ndarray:

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


class MLATCLayer:

    def __init__(
        self,
        layer_index: int,
        vigilance: float
    ):

        self.layer_index = layer_index

        self.vigilance = vigilance

        # Positional graph
        self.map = TopologicalMap()

        # m_i
        self.winner_count = np.zeros(
            self.map.capacity,
            dtype=np.int32
        )

        # Γ_del
        self.deleted_edge_ages = []

        # Current g_max
        self.gmax = np.inf

        # g_del
        self.gdel = 0

    def _resize_winner_count(self):

        new_winner_count = np.zeros(
            self.map.capacity,
            dtype=np.int32
        )

        # Copy only the entries that actually exist
        # in the old winner_count array.
        copy_count = min(
            len(self.winner_count),
            self.map.node_count
        )

        new_winner_count[:copy_count] = (
            self.winner_count[:copy_count]
        )

        self.winner_count = new_winner_count

    def add_node(self, position: np.ndarray) -> int:

        old_capacity = self.map.capacity

        node_id = self.map.add_node(position)

        if self.map.capacity != old_capacity:
            self._resize_winner_count()

        slot = self.map.get_slot(node_id)

        # A newly created node represents the current
        # learning event, so initialize m_i to one.
        self.winner_count[slot] = 1

        return node_id

    def remove_node(self, node_id: int):

        if node_id not in self.map.id_to_slot:
            return

        slot = self.map.get_slot(node_id)

        last_slot = self.map.node_count - 1

        if slot != last_slot:

            self.winner_count[slot] = (
                self.winner_count[last_slot]
            )

        self.winner_count[last_slot] = 0

        self.map.remove_node(node_id)

    def get_position(self, node_id: int) -> np.ndarray:

        slot = self.map.get_slot(node_id)

        return self.map.positions[slot]


    def get_winner_count(self, node_id: int) -> int:

        slot = self.map.get_slot(node_id)

        return int(self.winner_count[slot])


    def increment_winner_count(self, node_id: int):

        slot = self.map.get_slot(node_id)

        self.winner_count[slot] += 1


    def set_position(
        self,
        node_id: int,
        position: np.ndarray
    ):

        slot = self.map.get_slot(node_id)

        self.map.positions[slot] = np.asarray(
            position,
            dtype=np.float32
        )

    def update_by_winners(
        self,
        p: np.ndarray,
        s1: Optional[int],
        s2: Optional[int],
        d_s1: float,
        d_s2: float
    ) -> Tuple[int, bool]:

        # ==================================================
        # Case (a): Add new node
        # ds1 > vigilance
        # ==================================================

        if s1 is None or d_s1 > self.vigilance:

            new_node_id = self.add_node(p)

            return new_node_id, True

        # ==================================================
        # Case (b) / (c):
        # Existing winner
        # ==================================================

        self.increment_winner_count(s1)

        m_s1 = self.get_winner_count(s1)

        # --------------------------------------------------
        # Update winner node
        #
        # Eq. (6)
        # --------------------------------------------------

        winner_position = self.get_position(s1)

        learning_rate = 1.0 / (10.0 * m_s1)

        new_position = (
            winner_position
            + learning_rate * (p - winner_position)
        )

        self.set_position(
            s1,
            new_position
        )

        # ==================================================
        # Case (c): Connect first and second winner
        # ds2 <= vigilance
        # ==================================================

        new_edge = False

        if (
            s2 is not None
            and d_s2 < self.vigilance
        ):

            new_edge = self.map.add_edge(
                s1,
                s2
            )

        # ==================================================
        # Update neighbors of s1
        # Eq. (7)
        # ==================================================

        for k in list(
            self.map.neighbors(s1)
        ):

            m_k = self.get_winner_count(k)

            neighbor_position = self.get_position(k)

            if m_k <= 0:
                continue

            learning_rate_neighbor = (
                1.0 / (100.0 * m_k)
            )

            new_neighbor_position = (
                neighbor_position
                + learning_rate_neighbor
                * (p - neighbor_position)
            )

            self.set_position(
                k,
                new_neighbor_position
            )

        # ==================================================
        # Age edges connected to s1
        # ==================================================

        for k in list(
            self.map.neighbors(s1)
        ):

            self.map.graph[s1][k] += 1
            self.map.graph[k][s1] += 1

        # ==================================================
        # Remove old edges
        # ==================================================

        self._remove_old_edges()

        return s1, False

    def _update_gmax(self):

        current_ages = self.map.edge_ages()

        deleted_ages = np.asarray(
            self.deleted_edge_ages,
            dtype=np.float32
        )

        n_current = len(current_ages)
        n_deleted = len(deleted_ages)

        # No edge information yet
        if n_current == 0 and n_deleted == 0:
            self.gmax = np.inf
            return

        # No current edges:
        # Eq. (10) reduces to gmax = gdel
        if n_current == 0:
            self.gmax = float(self.gdel)
            return

        q75 = np.percentile(current_ages, 75)
        q25 = np.percentile(current_ages, 25)

        iqr = q75 - q25

        gthr = q75 + iqr

        total = n_current + n_deleted

        deleted_ratio = n_deleted / total

        self.gmax = (
            self.gdel * deleted_ratio
            + gthr * (1.0 - deleted_ratio)
        )

    def _remove_old_edges(self):

        # --------------------------------------------------
        # First determine gmax from current edge ages
        # and previously deleted edge ages.
        # --------------------------------------------------

        self._update_gmax()

        edges_to_remove = []

        for a, b in self.map.edges():

            age = self.map.graph[a][b]

            if age > self.gmax:

                edges_to_remove.append(
                    (a, b, age)
                )

        # --------------------------------------------------
        # Remove old edges
        # --------------------------------------------------

        for a, b, age in edges_to_remove:

            self.deleted_edge_ages.append(age)

            # g_del
            self.gdel = age

            self.map.remove_edge(
                a,
                b
            )

        # --------------------------------------------------
        # Recalculate gmax after deletion
        # --------------------------------------------------

        self._update_gmax()
    

class MLATC:

    def __init__(
        self,
        base_vigilance: float = 0.2,
        alpha: float = 3.5,
        lambda_points: int = 4000,
        seed: int = 42
    ):

        self.base_vigilance = base_vigilance
        self.alpha = alpha
        self.lambda_points = lambda_points
        self.rng = np.random.default_rng(seed)

        # ==================================================
        # Hierarchical layers
        # ==================================================

        self.layers: List[MLATCLayer] = []

        self.num_layers = 0

        # ==================================================
        # Parent-child relationships
        # ==================================================

        # --------------------------------------------------
        # Children
        # --------------------------------------------------
        #
        # (upper_layer, upper_node_id)
        #     ->
        # set of (lower_layer, lower_node_id)
        #
        # Example:
        #
        # children[(1, 4)] = {
        #     (0, 12),
        #     (0, 18),
        #     (0, 23)
        # }
        #
        # Meaning:
        # Layer 1 node 4 has Layer 0 nodes
        # 12, 18, and 23 as children.
        #
        self.children: Dict[
            Tuple[int, int],
            Set[Tuple[int, int]]
        ] = defaultdict(set)


        # --------------------------------------------------
        # Parent
        # --------------------------------------------------
        #
        # (lower_layer, lower_node_id)
        #     ->
        # (upper_layer, upper_node_id)
        #
        # Each lower-layer node has at most one parent.
        #
        # Example:
        #
        # parent[(0, 18)] = (1, 4)
        #
        self.parent: Dict[
            Tuple[int, int],
            Tuple[int, int]
        ] = {}

        # Start with Layer 1.
        self._add_layer()

    def _add_layer(self):

        layer_index = self.num_layers

        vigilance = (
            self.base_vigilance *
            (self.alpha ** layer_index)
        )

        layer = MLATCLayer(
            layer_index=layer_index,
            vigilance=vigilance
        )

        self.layers.append(layer)

        self.num_layers += 1

        return layer

    def sample_points(
        self,
        points: np.ndarray
    ) -> np.ndarray:

        if len(points) <= self.lambda_points:
            return points

        indices = self.rng.choice(
            len(points),
            size=self.lambda_points,
            replace=False
        )

        return points[indices]

    def get_layer(self, layer_index: int) -> MLATCLayer:

        return self.layers[layer_index]

    def add_parent_child(
        self,
        upper_layer: int,
        upper_node_id: int,
        lower_layer: int,
        lower_node_id: int  
    ):
        """
        Establish a parent-child relationship between
        two adjacent MLATC layers.
        """

        upper = (upper_layer, upper_node_id)
        lower = (lower_layer, lower_node_id)

        if upper_layer != lower_layer + 1:
            raise ValueError(
                "Parent and child must belong to adjacent layers."
            )
        
        # A lower-layer node may have only one parent.
        if lower in self.parent:

            existing_parent = self.parent[lower]

            if existing_parent != upper:
                raise ValueError(
                    f"Node {lower} already has parent "
                    f"{existing_parent}"
                )

            return

        self.parent[lower] = upper
        self.children[upper].add(lower)

    def get_parent(
        self,
        layer: int,
        node_id: int
    ) -> Optional[Tuple[int, int]]:

        return self.parent.get(
            (layer, node_id)
        )

    def remove_parent_child(
        self,
        upper_layer: int,
        upper_node_id: int,
        lower_layer: int,
        lower_node_id: int
    ):
        upper = (upper_layer, upper_node_id)
        lower = (lower_layer, lower_node_id)

        self.children[upper].discard(lower)

        if lower in self.parent:
            if self.parent[lower] == upper:
                del self.parent[lower]


    def _add_layer_from_top(self):
        """
        Create a new layer when the current top layer
        reaches two nodes.

        The first node of the old top layer is inherited
        as the root of the new top layer.

        The second old-top node is returned as an input
        that must be learned by the new layer.
        """

        old_top_index = self.num_layers - 1
        old_top_layer = self.layers[old_top_index]

        if old_top_layer.map.node_count != 2:
            raise RuntimeError(
                "A new layer can only be created when "
                "the top layer contains exactly two nodes."
            )

        node_ids = (
            old_top_layer.map
            .node_ids_active()
            .copy()
            .tolist()
        )

        first_id = node_ids[0]
        second_id = node_ids[1]

        first_position = (
            old_top_layer
            .get_position(first_id)
            .copy()
        )

        second_position = (
            old_top_layer
            .get_position(second_id)
            .copy()
        )

        # Create new upper layer.
        new_layer = self._add_layer()

        # Inherit first old-top node as root.
        inherited_root = new_layer.add_node(
            first_position
        )

        return {
            "old_layer": old_top_index,
            "new_layer": self.num_layers - 1,
            "first_id": first_id,
            "second_id": second_id,
            "root_id": inherited_root,
            "second_position": second_position,
        }
    
    def _search_threshold(self, layer_index: int) -> float:

        return sum(
            self.layers[i].vigilance
            for i in range(layer_index + 1)
        )

    def hierarchical_nns(
        self,
        p: np.ndarray
    ):

        p = np.asarray(
            p,
            dtype=np.float32
        )

        winner_sets = [
            []
            for _ in range(self.num_layers)
        ]

        # ==================================================
        # Top layer
        # ==================================================

        top_index = self.num_layers - 1
        top_layer = self.layers[top_index]

        top_ids = (
            top_layer.map.node_ids_active()
            .copy()
            .tolist()
        )

        for node_id in top_ids:

            position = top_layer.get_position(
                node_id
            )

            distance = np.linalg.norm(
                p - position
            )

            winner_sets[top_index].append(
                (distance, node_id)
            )

        winner_sets[top_index].sort(
            key=lambda x: x[0]
        )

        # ==================================================
        # Descend hierarchy
        # ==================================================

        for layer_index in range(
            top_index - 1,
            -1,
            -1
        ):

            layer = self.layers[layer_index]

            search_threshold = (
                self._search_threshold(
                    layer_index
                )
            )

            candidate_ids = set()

            upper_winners = winner_sets[
                layer_index + 1
            ]

            # --------------------------------------------------
            # Collect children
            # --------------------------------------------------

            for _, upper_node_id in upper_winners:

                upper_key = (
                    layer_index + 1,
                    upper_node_id
                )

                for child_layer, child_id in (
                    self.children.get(
                        upper_key,
                        set()
                    )
                ):

                    if child_layer == layer_index:

                        candidate_ids.add(
                            child_id
                        )

            # --------------------------------------------------
            # Distance filtering
            # --------------------------------------------------

            candidates = []

            for node_id in candidate_ids:

                position = layer.get_position(
                    node_id
                )

                distance = np.linalg.norm(
                    p - position
                )

                if distance <= search_threshold:

                    candidates.append(
                        (distance, node_id)
                    )

            candidates.sort(
                key=lambda x: x[0]
            )

            winner_sets[layer_index] = candidates

        return winner_sets

    def _get_winners(
        self,
        candidates
    ):
        if not candidates:
            return (
                None,
                None,
                np.inf,
                np.inf
            )

        s1 = candidates[0][1]
        d_s1 = candidates[0][0]

        if len(candidates) >= 2:
            s2 = candidates[1][1]
            d_s2 = candidates[1][0]
        else:
            s2 = None
            d_s2 = np.inf

        return (
            s1,
            s2,
            d_s1,
            d_s2
        )

    def _learn_upper_input(
        self,
        upper_layer_index: int,
        input_position: np.ndarray,
        candidates
    ):
        """
        Learn an input position in an existing upper layer
        using the candidate set preserved during exploration.
        """

        upper_layer = self.layers[
            upper_layer_index
        ]

        (
            s1,
            s2,
            d_s1,
            d_s2
        ) = self._get_winners(
            candidates
        )

        return upper_layer.update_by_winners(
            input_position,
            s1,
            s2,
            d_s1,
            d_s2
        )

    def learn_point(
        self,
        p: np.ndarray
    ):

        p = np.asarray(
            p,
            dtype=np.float32
        )

        # ==================================================
        # Exploration
        # ==================================================

        winner_sets = self.hierarchical_nns(p)

        # ==================================================
        # Learning
        # ==================================================

        layer_index = 0

        while layer_index < self.num_layers:

            layer = self.layers[layer_index]

            candidates = winner_sets[layer_index]

            (
                s1,
                s2,
                d_s1,
                d_s2
            ) = self._get_winners(
                candidates
            )

            # --------------------------------------------------
            # Learn current layer.
            # --------------------------------------------------

            node_id, new_node = (
                layer.update_by_winners(
                    p,
                    s1,
                    s2,
                    d_s1,
                    d_s2
                )
            )

            # --------------------------------------------------
            # Existing node:
            #
            # Learning stops here.
            # --------------------------------------------------

            if not new_node:
                break

            # Position of newly created node.
            input_position = (
                layer.get_position(node_id)
                .copy()
            )

            # ==================================================
            # Current layer is top layer
            # ==================================================

            if layer_index == self.num_layers - 1:

                # --------------------------------------------------
                # A newly-created node in the top layer means that
                # the top layer now contains two nodes.
                #
                # Create a new upper layer whose root is the first
                # node of the old top layer, then learn the second
                # old-top node into the new layer.
                # --------------------------------------------------

                if layer.map.node_count == 2:

                    transition = self._add_layer_from_top()

                    old_index = transition["old_layer"]
                    new_index = transition["new_layer"]

                    first_id = transition["first_id"]
                    second_id = transition["second_id"]

                    root_id = transition["root_id"]
                    second_position = transition["second_position"]

                    # --------------------------------------------------
                    # First old-top node -> inherited root
                    # --------------------------------------------------

                    self.add_parent_child(
                        upper_layer=new_index,
                        upper_node_id=root_id,
                        lower_layer=old_index,
                        lower_node_id=first_id
                    )

                    # --------------------------------------------------
                    # Learn second old-top node into new layer
                    # --------------------------------------------------

                    upper_layer = self.layers[new_index]

                    root_position = (
                        upper_layer
                        .get_position(root_id)
                        .copy()
                    )

                    root_distance = np.linalg.norm(
                        second_position - root_position
                    )

                    (
                        upper_node_id,
                        upper_new
                    ) = upper_layer.update_by_winners(
                        second_position,
                        root_id,
                        None,
                        root_distance,
                        np.inf
                    )

                    # --------------------------------------------------
                    # Second old-top node -> new upper winner
                    # --------------------------------------------------

                    self.add_parent_child(
                        upper_layer=new_index,
                        upper_node_id=upper_node_id,
                        lower_layer=old_index,
                        lower_node_id=second_id
                    )

                    # --------------------------------------------------
                    # IMPORTANT:
                    #
                    # If the new layer itself contains two nodes,
                    # continue the hierarchy construction.
                    # --------------------------------------------------

                    while self.layers[new_index].map.node_count == 2:

                        transition = self._add_layer_from_top()

                        old_index = transition["old_layer"]
                        new_index = transition["new_layer"]

                        first_id = transition["first_id"]
                        second_id = transition["second_id"]

                        root_id = transition["root_id"]
                        second_position = transition["second_position"]

                        # First old-top node becomes the new root
                        self.add_parent_child(
                            upper_layer=new_index,
                            upper_node_id=root_id,
                            lower_layer=old_index,
                            lower_node_id=first_id
                        )

                        # Learn second old-top node
                        upper_layer = self.layers[new_index]

                        root_position = (
                            upper_layer
                            .get_position(root_id)
                            .copy()
                        )

                        root_distance = np.linalg.norm(
                            second_position - root_position
                        )

                        (
                            upper_node_id,
                            upper_new
                        ) = upper_layer.update_by_winners(
                            second_position,
                            root_id,
                            None,
                            root_distance,
                            np.inf
                        )

                        # Second old-top node gets its parent
                        self.add_parent_child(
                            upper_layer=new_index,
                            upper_node_id=upper_node_id,
                            lower_layer=old_index,
                            lower_node_id=second_id
                        )

                break

            # ==================================================
            # Propagate newly-created node upward
            # ==================================================

            upper_index = layer_index + 1

            upper_layer = self.layers[
                upper_index
            ]

            upper_candidates = (
                winner_sets[upper_index]
            )

            (
                upper_s1,
                upper_s2,
                upper_d_s1,
                upper_d_s2
            ) = self._get_winners(
                upper_candidates
            )

            (
                upper_node_id,
                upper_new
            ) = upper_layer.update_by_winners(
                input_position,
                upper_s1,
                upper_s2,
                upper_d_s1,
                upper_d_s2
            )

            # --------------------------------------------------
            # Connect newly-created lower node to upper winner.
            # --------------------------------------------------

            self.add_parent_child(
                upper_layer=upper_index,
                upper_node_id=upper_node_id,
                lower_layer=layer_index,
                lower_node_id=node_id
            )

            # --------------------------------------------------
            # Existing upper winner:
            # propagation stops.
            # --------------------------------------------------

            if not upper_new:
                break

            # --------------------------------------------------
            # Continue learning upward.
            # --------------------------------------------------

            layer_index += 1


    def print_hierarchy(self):

        print("\n========== MLATC HIERARCHY ==========")

        top_index = self.num_layers - 1

        for layer_index, layer in enumerate(self.layers):

            node_ids = (
                layer.map.node_ids_active()
                .copy()
                .tolist()
            )

            print(
                f"Layer {layer_index + 1}: "
                f"nodes={layer.map.node_count}, "
                f"vigilance={layer.vigilance:.4f}"
                + ("  <-- TOP / ROOT" if layer_index == top_index else "")
            )

            for node_id in node_ids:

                key = (layer_index, node_id)

                children = sorted(
                    self.children.get(key, set())
                )

                parent = self.parent.get(key)

                print(
                    f"  Node {node_id}: "
                    f"parent={parent}, "
                    f"children={children}"
                )

        print("======================================\n")

    def process_frame(
        self,
        points: np.ndarray
    ):

        start = time.perf_counter()

        sampled_points = self.sample_points(
            points
        )

        for p in sampled_points:

            self.learn_point(p)

        elapsed = (
            time.perf_counter() -
            start
        )

        stats = self.stats()

        stats["time"] = elapsed
        stats["input_points"] = len(points)
        stats["sampled_points"] = len(sampled_points)

        return stats
    
    def stats(self):

        return {
            "num_layers": self.num_layers,

            "nodes_per_layer": [
                layer.map.node_count
                for layer in self.layers
            ],

            "edges_per_layer": [
                len(layer.map.edges())
                for layer in self.layers
            ]
        }

class MLATCVisualizer:
    """
    Matplotlib visualization for the positional MLATC graphs.

    One 3D figure is created for each MLATC layer.
    The visualization does not modify the MLATC data.
    """

    def __init__(self, mlatc: MLATC):
        self.mlatc = mlatc

        self.figures: Dict[int, plt.Figure] = {}
        self.axes: Dict[int, object] = {}

    def _create_layer_window(self, layer_index: int):

        fig = plt.figure(
            num=f"MLATC Layer {layer_index + 1}",
            figsize=(8, 7)
        )

        ax = fig.add_subplot(
            111,
            projection="3d"
        )

        ax.set_title(
            f"MLATC Layer {layer_index + 1}"
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        self.figures[layer_index] = fig
        self.axes[layer_index] = ax

    def _sync_layers(self):

        current_layer_count = len(self.mlatc.layers)

        # Create windows for newly added layers
        for layer_index in range(current_layer_count):

            if layer_index not in self.figures:
                self._create_layer_window(layer_index)

    def _draw_layer(self, layer_index: int):

        layer = self.mlatc.layers[layer_index]
        ax = self.axes[layer_index]

        ax.clear()

        ax.set_title(
            f"MLATC Layer {layer_index + 1}"
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        node_ids = layer.map.node_ids_active()

        # --------------------------------------------------
        # Draw edges first
        # --------------------------------------------------

        for node_a, node_b in layer.map.edges():

            position_a = layer.get_position(node_a)
            position_b = layer.get_position(node_b)

            ax.plot(
                [position_a[0], position_b[0]],
                [position_a[1], position_b[1]],
                [position_a[2], position_b[2]],
                linewidth=1.0
            )

        # --------------------------------------------------
        # Draw nodes
        # --------------------------------------------------

        if len(node_ids) > 0:

            positions = np.asarray(
                [
                    layer.get_position(node_id)
                    for node_id in node_ids
                ]
            )

            ax.scatter(
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
                s=40
            )

        # --------------------------------------------------
        # Draw node IDs
        # --------------------------------------------------

        for node_id in node_ids:

            position = layer.get_position(node_id)

            ax.text(
                position[0],
                position[1],
                position[2],
                str(node_id),
                fontsize=8
            )

        # --------------------------------------------------
        # Show basic graph statistics
        # --------------------------------------------------

        ax.text2D(
            0.02,
            0.95,
            f"Nodes: {len(node_ids)}\n"
            f"Edges: {len(layer.map.edges())}\n"
            f"ρ: {layer.vigilance:.3f}",
            transform=ax.transAxes,
            verticalalignment="top"
        )

    def update(self):

        """
        Synchronize the number of windows with the current
        number of MLATC layers and redraw every layer.
        """

        self._sync_layers()

        for layer_index in range(len(self.mlatc.layers)):

            self._draw_layer(layer_index)

            self.figures[layer_index].canvas.draw_idle()

        plt.pause(0.001)

    def close(self):

        for fig in self.figures.values():
            plt.close(fig)

        self.figures.clear()
        self.axes.clear()




if __name__ == "__main__":

    # ==========================================================
    # Dataset
    # ==========================================================

    dataset_path = Path(
        "dataset/scans"
    )

    dataset = ScanDataset(
        dataset_path
    )

    # ==========================================================
    # MLATC
    # ==========================================================

    mlatc = MLATC(
        base_vigilance=0.2,
        alpha=3.5,
        lambda_points=4000,
        seed=42
    )

    # ==========================================================
    # Matplotlib
    # ==========================================================

    plt.ion()

    visualizer = MLATCVisualizer(
        mlatc
    )

    # ==========================================================
    # Process frames
    # ==========================================================

    try:

        for frame_idx in range(len(dataset)):

            points = dataset[frame_idx]

            stats = mlatc.process_frame(
                points
            )

            # Update all MLATC layer windows
            

            # --------------------------------------------------
            # Console information
            # --------------------------------------------------

            print(
                f"Frame {frame_idx:03d} | "
                f"time={stats['time']:.4f}s | "
                f"layers={stats['num_layers']} | "
                f"nodes={stats['nodes_per_layer']} | "
                f"edges={stats['edges_per_layer']}"
            )

        visualizer.update()
        mlatc.print_hierarchy()

    except KeyboardInterrupt:

        print("\nProcessing interrupted by user.")

    finally:

        plt.ioff()

        plt.show()