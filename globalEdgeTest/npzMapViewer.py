import numpy as np
import open3d as o3d


class MapViewer:

    def __init__(self, filename):

        self.filename = filename

        data = np.load(
            filename,
            allow_pickle=False
        )

        self.node_ids = data["node_ids"]
        self.positions = data["positions"]
        self.winner_counts = data["winner_counts"]

        self.edges = data["edges"]
        self.edge_ages = data["edge_ages"]

        self.vigilance = float(
            data["vigilance"]
        )

        self.lambda_points = int(
            data["lambda_points"]
        )

        self.seed = int(
            data["seed"]
        )

        # Stable node ID -> array slot
        self.id_to_slot = {
            int(node_id): i
            for i, node_id in enumerate(self.node_ids)
        }

    def create_node_cloud(self):

        node_cloud = o3d.geometry.PointCloud()

        node_cloud.points = o3d.utility.Vector3dVector(
            self.positions
        )

        node_cloud.paint_uniform_color(
            [1, 0, 0]
        )

        return node_cloud

    def create_edge_lines(self):

        lines = o3d.geometry.LineSet()

        lines.points = o3d.utility.Vector3dVector(
            self.positions
        )

        edge_slots = []

        valid_edge_ages = []

        for i, (a, b) in enumerate(self.edges):

            a = int(a)
            b = int(b)

            if (
                a not in self.id_to_slot
                or
                b not in self.id_to_slot
            ):
                continue

            edge_slots.append([
                self.id_to_slot[a],
                self.id_to_slot[b]
            ])

            valid_edge_ages.append(
                self.edge_ages[i]
            )

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

        lines.lines = o3d.utility.Vector2iVector(
            edge_slots
        )

        colors = np.tile(
            np.array([[0, 1, 0]]),
            (
                len(edge_slots),
                1
            )
        )

        lines.colors = o3d.utility.Vector3dVector(
            colors
        )

        return lines

    def visualize(self):

        print()
        print("Map")
        print("--------------------------------")
        print(
            "File       :",
            self.filename
        )
        print(
            "Nodes      :",
            len(self.node_ids)
        )
        print(
            "Edges      :",
            len(self.edges)
        )
        print(
            "Vigilance  :",
            self.vigilance
        )
        print(
            "Lambda     :",
            self.lambda_points
        )
        print(
            "Seed       :",
            self.seed
        )
        print("--------------------------------")

        vis = o3d.visualization.Visualizer()

        vis.create_window(
            window_name="ATC-DT Map Viewer",
            width=1280,
            height=720
        )

        node_cloud = self.create_node_cloud()

        lines = self.create_edge_lines()

        vis.add_geometry(
            node_cloud
        )

        vis.add_geometry(
            lines
        )

        # Coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0,
            origin=[0, 0, 0]
        )

        vis.add_geometry(
            coordinate_frame
        )

        # Render settings
        render_option = vis.get_render_option()

        render_option.point_size = 5.0

        vis.run()

        vis.destroy_window()


if __name__ == "__main__":

    viewer = MapViewer(
        "maps/baseATC_static_0_1.npz"
    )

    viewer.visualize()