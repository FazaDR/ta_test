import open3d as o3d
import numpy as np


# ============================================================
# Utility
# ============================================================
def create_box(width, depth, height, center, color):
    mesh = o3d.geometry.TriangleMesh.create_box(width, depth, height)
    mesh.compute_vertex_normals()

    mesh.translate([
        center[0] - width / 2,
        center[1] - depth / 2,
        center[2] - height / 2
    ])

    mesh.paint_uniform_color(color)
    return mesh


# ============================================================
# Parameters
# ============================================================
ROOM_SIZE = 12
WALL_THICKNESS = 0.2
WALL_HEIGHT = 2.5

geometries = []

# ============================================================
# Floor
# ============================================================
floor = create_box(
    ROOM_SIZE,
    ROOM_SIZE,
    0.05,
    [0, 0, -0.025],
    [0.8, 0.8, 0.8]
)
geometries.append(floor)

# ============================================================
# Outer walls
# ============================================================
wall_color = [0.65, 0.65, 0.65]   

# Left
geometries.append(create_box(
    WALL_THICKNESS,
    ROOM_SIZE,
    WALL_HEIGHT,
    [-ROOM_SIZE/2, 0, WALL_HEIGHT/2],
    wall_color
))

# Right
geometries.append(create_box(
    WALL_THICKNESS,
    ROOM_SIZE,
    WALL_HEIGHT,
    [ROOM_SIZE/2, 0, WALL_HEIGHT/2],
    wall_color
))

# Bottom
geometries.append(create_box(
    ROOM_SIZE,
    WALL_THICKNESS,
    WALL_HEIGHT,
    [0, -ROOM_SIZE/2, WALL_HEIGHT/2],
    wall_color
))

# Top
geometries.append(create_box(
    ROOM_SIZE,
    WALL_THICKNESS,
    WALL_HEIGHT,
    [0, ROOM_SIZE/2, WALL_HEIGHT/2],
    wall_color
))

# ============================================================
# Interior walls
# ============================================================

# Lower wall
lower_wall = create_box(
    WALL_THICKNESS,
    4.9,
    WALL_HEIGHT,
    [0, -3.65, WALL_HEIGHT/2],
    [0.1, 0.1, 0.1]
)
geometries.append(lower_wall)

# Upper wall
upper_wall = create_box(
    WALL_THICKNESS,
    1.8,
    WALL_HEIGHT,
    [0, 2.5, WALL_HEIGHT/2],
    [0.1, 0.1, 0.1]
)
geometries.append(upper_wall)

# ============================================================
# Disappearing wall (orange)
# ============================================================
# Comment the append() below to simulate the wall disappearing.

disappearing_wall = create_box(
    WALL_THICKNESS,
    2.8,
    WALL_HEIGHT,
    [0, 0.2, WALL_HEIGHT/2],
    [1.0, 0.4, 0.0]
)

# geometries.append(disappearing_wall)

# ============================================================
# Square exploration path
# ============================================================

path = np.array([
    [-4.6, -4.7, 0.05],
    [-4.6,  3.8, 0.05],
    [-3.8,  4.8, 0.05],
    [-1.0,  5.0, 0.05],
    [ 1.5,  5.0, 0.05],
    [ 3.5,  4.6, 0.05],
    [ 4.2,  2.5, 0.05],
    [ 4.2, -4.5, 0.05],
])

# ------------------------------------------------------------
# Dashed line
# ------------------------------------------------------------
dash = 0.25
gap = 0.15

dash_points = []
dash_lines = []

idx = 0

for i in range(len(path)-1):

    p0 = path[i]
    p1 = path[i+1]

    vec = p1 - p0
    length = np.linalg.norm(vec)
    direction = vec / length

    d = 0

    while d < length:

        a = p0 + direction*d
        b = p0 + direction*min(d+dash, length)

        dash_points.append(a)
        dash_points.append(b)

        dash_lines.append([idx, idx+1])

        idx += 2

        d += dash + gap

line = o3d.geometry.LineSet()
line.points = o3d.utility.Vector3dVector(np.asarray(dash_points))
line.lines = o3d.utility.Vector2iVector(np.asarray(dash_lines))
line.colors = o3d.utility.Vector3dVector(
    np.tile([0.70, 0.20, 0.80], (len(dash_lines), 1))
)

geometries.append(line)

# ============================================================
# Start / End
# ============================================================

start = o3d.geometry.TriangleMesh.create_sphere(radius=0.18)
start.compute_vertex_normals()
start.translate([-4.6, -4.8, 0.18])
start.paint_uniform_color([0.0, 0.8, 0.0])

end = o3d.geometry.TriangleMesh.create_sphere(radius=0.18)
end.compute_vertex_normals()
end.translate([4.2, -4.8, 0.18])
end.paint_uniform_color([0.9, 0.0, 0.0])

geometries.extend([start, end])

# ============================================================
# Coordinate Frame
# ============================================================

# frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.8)
# geometries.append(frame)

# ============================================================
# Visualize
# ============================================================

o3d.visualization.draw_geometries(
    geometries,
    window_name="Adaptability Environment",
    width=1400,
    height=900,
)