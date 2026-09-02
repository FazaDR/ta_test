import pybullet as pb
import pybullet_data
import numpy as np
import time


# ==========================
# Start physics
# ==========================

pb.connect(pb.GUI)

pb.setAdditionalSearchPath(
    pybullet_data.getDataPath()
)

pb.setGravity(
    0,0,-9.8
)


# ==========================
# Create objects
# ==========================

def create_box(pos, size):

    collision = pb.createCollisionShape(
        pb.GEOM_BOX,
        halfExtents=[
            size[0]/2,
            size[1]/2,
            size[2]/2
        ]
    )

    visual = pb.createVisualShape(
        pb.GEOM_BOX,
        halfExtents=[
            size[0]/2,
            size[1]/2,
            size[2]/2
        ],
        rgbaColor=[
            0.7,0.7,0.7,1
        ]
    )


    return pb.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=pos
    )


# floor

create_box(
    [5,5,-0.1],
    [10,10,0.2]
)


# room walls

create_box(
    [5,0,1.5],
    [10,0.2,3]
)

create_box(
    [5,10,1.5],
    [10,0.2,3]
)

create_box(
    [0,5,1.5],
    [0.2,10,3]
)

create_box(
    [10,5,1.5],
    [0.2,10,3]
)


# extending wall from side

create_box(
    [3,3,1.5],
    [0.2,4,3]
)


# object sitting on floor

create_box(
    [7,7,0.5],
    [1,1,1]
)

# ==========================
# Raycast sensor
# ==========================

sensor = np.array(
    [4,4,1],
    dtype=float
)


# generate fixed spherical rays once

ray_directions = []


for i in range(500):

    direction = np.random.normal(
        size=3
    )

    direction /= np.linalg.norm(direction)

    ray_directions.append(direction)



ray_directions = np.array(
    ray_directions
)



def lidar(origin):

    points = []


    for direction in ray_directions:


        end = origin + direction*20


        hit = pb.rayTest(
            origin,
            end
        )[0]


        if hit[0] != -1:

            points.append(
                hit[3]
            )


    return np.array(points)

# ==========================
# Visualization
# ==========================

point_ids = []


# ==========================
# Main update loop
# ==========================

while True:

    keys = pb.getKeyboardEvents()

    move = np.zeros(3)


    if pb.B3G_UP_ARROW in keys and keys[pb.B3G_UP_ARROW] & pb.KEY_IS_DOWN:
        move[1] += 0.05


    if pb.B3G_DOWN_ARROW in keys and keys[pb.B3G_DOWN_ARROW] & pb.KEY_IS_DOWN:
        move[1] -= 0.05


    if pb.B3G_LEFT_ARROW in keys and keys[pb.B3G_LEFT_ARROW] & pb.KEY_IS_DOWN:
        move[0] -= 0.05


    if pb.B3G_RIGHT_ARROW in keys and keys[pb.B3G_RIGHT_ARROW] & pb.KEY_IS_DOWN:
        move[0] += 0.05


    if ord('w') in keys and keys[ord('w')] & pb.KEY_IS_DOWN:
        move[2] += 0.05


    if ord('s') in keys and keys[ord('s')] & pb.KEY_IS_DOWN:
        move[2] -= 0.05


    sensor += move


    points = lidar(sensor)
    print(len(points))


    # remove old points

    for p in point_ids:
        pb.removeUserDebugItem(p)

    point_ids.clear()


    # draw lidar points

    for point in points:

        pid = pb.addUserDebugPoints(
            [point],
            [[1,0,0]],
            pointSize=5,
            lifeTime=0.2
        )

        point_ids.append(pid)


    # draw sensor

    sid = pb.addUserDebugPoints(
        [sensor],
        [[0,0,1]],
        pointSize=15,
        lifeTime=1
    )


    pb.stepSimulation()


    time.sleep(0.01)