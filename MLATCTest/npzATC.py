import argparse

import numpy as np
import open3d as o3d


DEFAULT_NODE_COLOR = np.array([1.0, 0.0, 0.0], dtype=np.float32)
EDGE_COLOR = np.array([0.0, 1.0, 0.0], dtype=np.float32)


def load_map(npz_path):

	with np.load(npz_path) as data:

		required = {
			"positions",
			"node_ids",
			"edges",
			"normals",
			"traversable",
			"contour"
		}

		missing = required.difference(data.files)

		if missing:
			raise KeyError(
				f"Missing arrays in {npz_path}: {sorted(missing)}"
			)

		positions = data["positions"].astype(np.float32)
		node_ids = data["node_ids"]
		normals = data["normals"].astype(np.float32)
		traversable = data["traversable"].astype(bool)
		contour = data["contour"].astype(bool)
		edges = data["edges"]

	if len(positions) == 0:
		raise ValueError(f"No nodes found in {npz_path}")

	id_to_slot = {
		int(node_id): slot
		for slot, node_id in enumerate(node_ids)
	}

	edge_slots = [
		[id_to_slot[int(a)], id_to_slot[int(b)]]
		for a, b in edges
		if int(a) in id_to_slot and int(b) in id_to_slot
	]

	return {
		"positions": positions,
		"normals": normals,
		"traversable": traversable,
		"contour": contour,
		"edge_slots": np.asarray(edge_slots, dtype=np.int32)
		if edge_slots else np.empty((0, 2), dtype=np.int32)
	}


def node_colors(atc_map):

	n = len(atc_map["positions"])
	colors = np.tile(
		np.array([0.6, 0.6, 0.6], dtype=np.float32),
		(n, 1)
	)

	has_normal = np.linalg.norm(atc_map["normals"], axis=1) > 1e-8
	traversable = atc_map["traversable"]
	contour = atc_map["contour"]

	colors[has_normal & traversable] = [0.0, 0.0, 1.0]
	colors[has_normal & traversable & contour] = [0.0, 1.0, 0.0]
	colors[has_normal & ~traversable] = [1.0, 0.0, 0.0]

	return colors


def normal_lines(atc_map, length=0.3):

	positions = atc_map["positions"]
	normals = atc_map["normals"]
	normal_norms = np.linalg.norm(normals, axis=1)
	indices = np.where(normal_norms > 1e-8)[0]

	if len(indices) == 0:
		return None, None

	unit_normals = normals[indices] / normal_norms[indices, None]
	half_length = length / 2.0
	starts = positions[indices] - unit_normals * half_length
	ends = positions[indices] + unit_normals * half_length

	points = np.empty((2 * len(indices), 3), dtype=np.float32)
	points[0::2] = starts
	points[1::2] = ends

	lines = np.asarray(
		[[2 * index, 2 * index + 1] for index in range(len(indices))],
		dtype=np.int32
	)

	return points, lines


class ATCViewer:

	def __init__(self, atc_map, show_normals=False, normal_length=0.3):

		self.atc_map = atc_map
		self.show_normals = show_normals
		self.normal_length = normal_length
		self.vis = o3d.visualization.VisualizerWithKeyCallback()
		self.vis.create_window("ATC-DT NPZ Viewer")

		colors = node_colors(atc_map)

		self.node_cloud = o3d.geometry.PointCloud()
		self.node_cloud.points = o3d.utility.Vector3dVector(
			atc_map["positions"]
		)
		self.node_cloud.colors = o3d.utility.Vector3dVector(colors)
		self.vis.add_geometry(self.node_cloud)

		self.edge_set = o3d.geometry.LineSet()
		self.edge_set.points = o3d.utility.Vector3dVector(
			atc_map["positions"]
		)

		edge_slots = atc_map["edge_slots"]

		if len(edge_slots) > 0:
			self.edge_set.lines = o3d.utility.Vector2iVector(edge_slots)
			edge_colors = np.asarray(
				[
					(colors[a] + colors[b]) / 2.0
					for a, b in edge_slots
				],
				dtype=np.float32
			)
			self.edge_set.colors = o3d.utility.Vector3dVector(edge_colors)
		else:
			self.edge_set.paint_uniform_color(EDGE_COLOR.tolist())

		self.vis.add_geometry(self.edge_set)
		self.normal_set = None

		self.vis.register_key_callback(ord("n"), self._toggle_normals)
		self.vis.register_key_callback(ord("N"), self._toggle_normals)

		if show_normals:
			self._toggle_normals(self.vis)

	def _toggle_normals(self, vis):

		if self.normal_set is None:
			points, lines = normal_lines(
				self.atc_map,
				self.normal_length
			)

			if points is None:
				print("No estimated normals to display.")
				return False

			self.normal_set = o3d.geometry.LineSet()
			self.normal_set.points = o3d.utility.Vector3dVector(points)
			self.normal_set.lines = o3d.utility.Vector2iVector(lines)
			self.normal_set.paint_uniform_color([0.0, 0.0, 0.0])
			self.vis.add_geometry(self.normal_set)
			print("Normals: shown")
		else:
			self.vis.remove_geometry(self.normal_set)
			self.normal_set = None
			print("Normals: hidden")

		return False

	def run(self):
		self.vis.run()
		self.vis.destroy_window()


def main():

	parser = argparse.ArgumentParser(
		description="View a base ATC-DT .npz map."
	)
	parser.add_argument(
		"npz_path",
		nargs="?",
		default="maps/local_atc_long_1.npz",
		help="Path to the ATC-DT .npz map."
	)
	parser.add_argument(
		"--show-normals",
		action="store_true",
		help="Show estimated node normals at startup."
	)
	parser.add_argument(
		"--normal-length",
		type=float,
		default=0.3,
		help="Length of each displayed normal line."
	)

	args = parser.parse_args()
	atc_map = load_map(args.npz_path)

	print(f"Loaded {len(atc_map['positions'])} nodes")
	print(f"Loaded {len(atc_map['edge_slots'])} edges")
	print("Press N to toggle normals; Q or Esc exits.")

	ATCViewer(
		atc_map,
		show_normals=args.show_normals,
		normal_length=args.normal_length
	).run()


if __name__ == "__main__":
	main()
