import open3d as o3d
import numpy as np


# -------------------------------------------------
# Utility
# -------------------------------------------------
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


ROOM = 20
WALL = 0.20
HEIGHT = 2.5

wall_color = [0.60, 0.60, 0.60]
floor_color = [0.92, 0.92, 0.92]

geometries = []


# -------------------------------------------------
# Floor
# -------------------------------------------------

geometries.append(
    create_box(
        ROOM,
        ROOM,
        0.05,
        [0, 0, -0.025],
        floor_color
    )
)

# -------------------------------------------------
# Outer walls
# -------------------------------------------------

outer = [

    (-ROOM/2,0,WALL,ROOM),
    ( ROOM/2,0,WALL,ROOM),
    (0,-ROOM/2,ROOM,WALL),
    (0, ROOM/2,ROOM,WALL),

]

for x,y,w,d in outer:
    geometries.append(
        create_box(
            w,d,HEIGHT,
            [x,y,HEIGHT/2],
            wall_color
        )
    )

# -------------------------------------------------
# Internal walls
# -------------------------------------------------

# Horizontal walls

horizontal = [

    (-2, 6.5, 16),
    (2, 2.5, 16),
    (-2,-1.5, 16),
    (2,-5.5, 16),

]

for x,y,length in horizontal:

    geometries.append(
        create_box(
            length,
            WALL,
            HEIGHT,
            [x,y,HEIGHT/2],
            wall_color
        )
    )

# Vertical connectors (alternating ends)

vertical = [

    # ( 7, 4.5, 4),
    # (-7, 0.5, 4),
    # ( 7,-3.5, 4),

]

for x,y,length in vertical:

    geometries.append(
        create_box(
            WALL,
            length,
            HEIGHT,
            [x,y,HEIGHT/2],
            wall_color
        )
    )

# -------------------------------------------------
# Mapping path
# -------------------------------------------------

path = np.array([

    [-8.5, 8.5, 0.05],
    [ 8.0, 8.5, 0.05],
    [ 8.0, 4.5, 0.05],
    [-8.0, 4.5, 0.05],
    [-8.0, 0.5, 0.05],
    [ 8.0, 0.5, 0.05],
    [ 8.0,-3.5, 0.05],
    [-8.0,-3.5, 0.05],
    [-8.0,-8.5, 0.05],
    [ 8.5,-8.5, 0.05]

])

dash = 0.35
gap = 0.18

pts = []
lines = []

idx = 0

for i in range(len(path)-1):

    a = path[i]
    b = path[i+1]

    v = b-a
    L = np.linalg.norm(v)
    d = v/L

    s = 0

    while s < L:

        p1 = a + d*s
        p2 = a + d*min(s+dash, L)

        pts.extend([p1,p2])
        lines.append([idx,idx+1])

        idx += 2
        s += dash + gap


line = o3d.geometry.LineSet()
line.points = o3d.utility.Vector3dVector(np.asarray(pts))
line.lines = o3d.utility.Vector2iVector(np.asarray(lines))
line.colors = o3d.utility.Vector3dVector(
    np.tile([0.8,0.3,0.9], (len(lines),1))
)

geometries.append(line)

# -------------------------------------------------
# Start / End
# -------------------------------------------------

start = o3d.geometry.TriangleMesh.create_sphere(radius=0.22)
start.compute_vertex_normals()
start.translate([-8.5,8.5,0.22])
start.paint_uniform_color([0,0.8,0])

end = o3d.geometry.TriangleMesh.create_sphere(radius=0.22)
end.compute_vertex_normals()
end.translate([8.5,-8.5,0.22])
end.paint_uniform_color([1,0,0])

geometries.extend([start,end])

# -------------------------------------------------
# Visualize
# -------------------------------------------------

o3d.visualization.draw_geometries(
    geometries,
    
    width=1400,
    height=900,
    mesh_show_back_face=True
)