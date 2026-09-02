import open3d as o3d
import numpy as np
import time
from scipy.spatial import KDTree


# ==========================================
# ATC graph
# ==========================================

nodes = []
wins = []
edges = set()

VTH = 1.5      # vigilance parameter
EDGE_MAX_AGE = 30

edge_age = {}

waypoints = [
    np.array([5, 1, 5], dtype=np.float32),
    np.array([15, 1, 5], dtype=np.float32),
    np.array([15, 1, 15], dtype=np.float32),
    np.array([5, 1, 15], dtype=np.float32),
]

current_wp = 0
speed = 0.05




def animate(origin):

    global current_wp

    target = waypoints[current_wp]

    direction = target - origin

    dist = np.linalg.norm(direction)

    if dist < 0.2:
        current_wp = (current_wp + 1) % len(waypoints)
        return origin

    direction /= dist

    origin += direction * speed

    return origin

def atc_update(points):

    global nodes
    global wins
    global edges
    global edge_age

    for v in points:

        # first nodes
        if len(nodes) < 2:

            nodes.append(v.copy())
            wins.append(1)

            continue

        s1, s2 = nearest_two_nodes(v)

        d1 = np.linalg.norm(
            v - nodes[s1]
        )

        # ==================================
        # STEP 3
        # add node
        # ==================================

        if d1 > VTH:

            nodes.append(v.copy())
            wins.append(1)

            continue

        # ==================================
        # STEP 4
        # winner update
        # ==================================

        wins[s1] += 1

        lr = 1.0 / (10.0 * wins[s1])

        nodes[s1] += lr * (
            v - nodes[s1]
        )

        # ==================================
        # STEP 5
        # edge update
        # ==================================

        edge = tuple(
            sorted((s1, s2))
        )

        edges.add(edge)

        edge_age[edge] = 0

        for e in list(edge_age.keys()):

            if s1 in e:

                edge_age[e] += 1

        # ==================================
        # STEP 6
        # remove old edges
        # ==================================

        remove_list = []

        for e, age in edge_age.items():

            if age > EDGE_MAX_AGE:

                remove_list.append(e)

        for e in remove_list:

            edge_age.pop(e)

            if e in edges:
                edges.remove(e)

def nearest_two_nodes(point):

    if len(nodes) < 2:
        return None, None

    positions = np.array(nodes)

    dist = np.linalg.norm(
        positions - point,
        axis=1
    )

    idx = np.argsort(dist)

    return idx[0], idx[1]

def lidar_scan(scene, origin, dirs):

    origins = np.repeat(origin[None, :], len(dirs), axis=0)

    rays = np.hstack([origins, dirs])

    rays = o3d.core.Tensor(
        rays,
        dtype=o3d.core.Dtype.Float32
    )

    ans = scene.cast_rays(rays)

    t_hit = ans["t_hit"].numpy()

    valid = np.isfinite(t_hit)

    hit_points = (
        origins[valid]
        + dirs[valid] * t_hit[valid, None]
    )

    # blind zone around lidar
    min_range = 2.0

    dist = np.linalg.norm(
        hit_points - origin,
        axis=1
    )

    hit_points = hit_points[dist > min_range]

    return hit_points

points = np.array([
    [0, 0, 0],
    [50, 0, 0],
    [0, 50, 0],
    [0, 0, 50]
])

lines = np.array([
    [0,1],
    [0,2],
    [0,3]
])

colors = np.array([
    [1, 0, 0],  # X = red
    [0, 1, 0],  # Y = green
    [0, 0, 1]   # Z = blue
])

line = o3d.geometry.LineSet()
line.points = o3d.utility.Vector3dVector(points)
line.lines = o3d.utility.Vector2iVector(lines)
line.colors = o3d.utility.Vector3dVector(colors)





floor = o3d.geometry.TriangleMesh.create_box(width = 20, height = 0.5, depth = 20 )

pillar = o3d.geometry.TriangleMesh.create_box(width = 2, height = 4, depth = 2)


wall1 = o3d.geometry.TriangleMesh.create_box(width = 20, height = 6, depth = 0.5)
wall1.translate([0,0,0])
wall2 = o3d.geometry.TriangleMesh.create_box(width = 0.5, height = 6, depth = 20)
wall2.translate([20,0,0])
wall3 = o3d.geometry.TriangleMesh.create_box(width = 20, height = 6, depth = 0.5)
wall3.translate([0,0,20])
wall4 = o3d.geometry.TriangleMesh.create_box(width = 0.5, height = 6, depth = 20)
wall4.translate([0,0,0])


floor.translate([0, 0, 0])
pillar.translate([0, 0.5, 0])




env = floor + pillar + wall1 + wall2 + wall3 + wall4

env.compute_vertex_normals()


# ==========================================================
# 3D LiDAR
# ==========================================================

tmesh = o3d.t.geometry.TriangleMesh.from_legacy(env)

scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(tmesh)

# lidar position
origin = np.array([10, 1, 10], dtype=np.float32)

lidar = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
lidar.compute_vertex_normals()
lidar.paint_uniform_color([1, 0, 1])

lidar.translate(origin)

N = 10000

phi = np.pi * (3.0 - np.sqrt(5.0))

dirs = []

for i in range(N):
    y = 1 - (i / float(N - 1)) * 2
    radius = np.sqrt(1 - y * y)

    theta = phi * i

    x = np.cos(theta) * radius
    z = np.sin(theta) * radius

    dirs.append([x, y, z])

dirs = np.array(dirs, dtype=np.float32)



hit_points = lidar_scan(scene, origin, dirs)

# create point cloud from lidar returns
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(hit_points)

# optional: make lidar points yellow
pcd.paint_uniform_color([1, 1, 0])
free_pcd = o3d.geometry.PointCloud()
free_pcd.paint_uniform_color([0, 1, 0])

nodes_pcd = o3d.geometry.PointCloud()
nodes_pcd.paint_uniform_color([1,0,0])

graph_lines = o3d.geometry.LineSet()

# ==========================================================
# Visualize
# ==========================================================

vis = o3d.visualization.Visualizer()
vis.create_window()

vis.add_geometry(env)
vis.add_geometry(line)
# vis.add_geometry(pcd)
# vis.add_geometry(free_pcd)
vis.add_geometry(lidar)
vis.add_geometry(nodes_pcd)
vis.add_geometry(graph_lines)


prev_origin = origin.copy()

while vis.poll_events():
    # move lidar
    origin = animate(origin)

    delta = origin - prev_origin

    lidar.translate(delta)

    # new scan
    hit_points = lidar_scan(scene, origin, dirs)

    scan_pcd = o3d.geometry.PointCloud()
    scan_pcd.points = o3d.utility.Vector3dVector(hit_points)

    scan_pcd = scan_pcd.voxel_down_sample(
        voxel_size=0.5
    )

    scan_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=1.0,
            max_nn=30
        )
    )
    down_points = np.asarray(scan_pcd.points)
    normals = np.asarray(scan_pcd.normals)
    up = np.array([0, 1, 0])

    cos_theta = np.abs(normals @ up)

    slope_deg = np.degrees(
        np.arccos(
            np.clip(cos_theta, 0, 1)
        )
    )

    traversable = slope_deg < 30
    free_points = down_points[traversable]
    
    atc_update(free_points)

    if len(nodes) > 1:

        graph_lines.points = (
            o3d.utility.Vector3dVector(
                np.array(nodes)
            )
        )

    graph_lines.lines = (
        o3d.utility.Vector2iVector(
            np.array(list(edges))
        )
    )

    colors = np.tile(
        [0,1,0],
        (len(edges),1)
    )

    graph_lines.colors = (
        o3d.utility.Vector3dVector(colors)
    )
    
    print(
        "nodes:",
        len(nodes),
        "edges:",
        len(edges)
    )

    free_scan = o3d.geometry.PointCloud()
    # free_scan.points = o3d.utility.Vector3dVector(
    #     free_points
    # )
    
    free_pcd.points = o3d.utility.Vector3dVector(
        free_points
    )

    if len(nodes):

        nodes_pcd.points = (
            o3d.utility.Vector3dVector(
                np.array(nodes)
            )
        )

    # pcd.points = o3d.utility.Vector3dVector(hit_points)

    vis.update_geometry(lidar)
    vis.update_geometry(pcd)
    vis.update_geometry(free_pcd)
    vis.update_geometry(nodes_pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.update_geometry(graph_lines)

    prev_origin = origin.copy()
    time.sleep(0.1)

vis.destroy_window()