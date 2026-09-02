import open3d as o3d
import numpy as np
import time
import os


waypoints = [
    np.array([5, 1, 5], dtype=np.float32),
    # np.array([15, 1, 5], dtype=np.float32),
    # np.array([15, 1, 15], dtype=np.float32),
    # np.array([5, 1, 15], dtype=np.float32),
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

def lidar_scan(scene, origin, dirs):

    origins = np.repeat(
        origin[None, :],
        len(dirs),
        axis=0
    )

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

    distances = np.linalg.norm(
        hit_points - origin,
        axis=1
    )

    MIN_RANGE = 0.1
    MAX_RANGE = 20.0

    mask = (
        (distances >= MIN_RANGE) &
        (distances <= MAX_RANGE)
    )

    hit_points = hit_points[mask]

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
pillar.translate([10, 0.5, 2.5])

env = floor + pillar + wall1 + wall2 + wall3 + wall4

env.compute_vertex_normals()


# ==========================================================
# 3D LiDAR
# ==========================================================

tmesh = o3d.t.geometry.TriangleMesh.from_legacy(env)

scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(tmesh)

# lidar position
origin = np.array([10, 2, 10], dtype=np.float32)

lidar = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
lidar.compute_vertex_normals()
lidar.paint_uniform_color([1, 0, 1])

lidar.translate(origin)

# ==========================================================
# LiDAR specification
# ==========================================================

HFOV = 360.0          # degrees
VUPFOV = 30.0           # degrees
VDOWNFOV = 30.0         # degrees

NUM_H = 256          # horizontal rays
NUM_V = 64            # vertical channels

# vertical angles
v_angles = np.linspace(
    -VDOWNFOV,
    VUPFOV,
    NUM_V
)

# horizontal angles
h_angles = np.linspace(
    0,
    HFOV,
    NUM_H,
    endpoint=False
)

dirs = []

for v_deg in v_angles:

    v = np.deg2rad(v_deg)

    for h_deg in h_angles:

        h = np.deg2rad(h_deg)

        x = np.cos(v) * np.cos(h)
        y = np.sin(v)
        z = np.cos(v) * np.sin(h)

        dirs.append([x, y, z])

dirs = np.asarray(
    dirs,
    dtype=np.float32
)

print("Total rays:", len(dirs))



hit_points = lidar_scan(scene, origin, dirs)

# create point cloud from lidar returns
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(hit_points)

# optional: make lidar points yellow
pcd.paint_uniform_color([1, 1, 0])

# ==========================================================
# Dataset output
# ==========================================================

os.makedirs("dataset/scans", exist_ok=True)

pose_file = open(
    "dataset/poses.csv",
    "w"
)

pose_file.write(
    "frame,timestamp,x,y,z\n"
)

frame_id = 0

# ==========================================================
# Visualize
# ==========================================================

vis = o3d.visualization.Visualizer()
vis.create_window()

vis.add_geometry(env)
vis.add_geometry(line)
vis.add_geometry(pcd)
vis.add_geometry(lidar)


prev_origin = origin.copy()

while vis.poll_events():

    # move lidar
    origin = animate(origin)

    delta = origin - prev_origin

    lidar.translate(delta)

    # new scan
    hit_points = lidar_scan(scene, origin, dirs)

    timestamp = time.time()

    # --------------------------------------------------
    # Save point cloud as CSV
    # --------------------------------------------------

    scan_filename = (
        f"dataset/scans/scan_{frame_id:06d}.csv"
    )

    header = (
        f"# timestamp={timestamp}\n"
        f"# sensor_x={origin[0]}\n"
        f"# sensor_y={origin[1]}\n"
        f"# sensor_z={origin[2]}\n"
        f"x,y,z"
    )

    np.savetxt(
        scan_filename,
        hit_points,
        delimiter=",",
        header=header,
        comments=""
    )

    # --------------------------------------------------
    # Save pose log
    # --------------------------------------------------

    pose_file.write(
        f"{frame_id},"
        f"{timestamp},"
        f"{origin[0]},"
        f"{origin[1]},"
        f"{origin[2]}\n"
    )

    frame_id += 1

    # update visualization
    pcd.points = o3d.utility.Vector3dVector(hit_points)

    

    vis.update_geometry(lidar)
    vis.update_geometry(pcd)

    vis.poll_events()
    vis.update_renderer()

    prev_origin = origin.copy()
    time.sleep(0.1)

vis.destroy_window()
pose_file.close()
