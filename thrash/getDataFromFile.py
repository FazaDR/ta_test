import os
import numpy as np
import pandas as pd




class SyntheticLidarDataset:

    def __init__(self, dataset_path):

        self.dataset_path = dataset_path

        self.scans_path = os.path.join(
            dataset_path,
            "scans"
        )

        self.poses = pd.read_csv(
            os.path.join(
                dataset_path,
                "poses.csv"
            )
        )

    def __len__(self):

        return len(self.poses)

    def get_frame(self, frame_id):

        pose_row = self.poses.iloc[frame_id]

        scan_file = os.path.join(
            self.scans_path,
            f"scan_{frame_id:06d}.csv"
        )

        points = np.loadtxt(
            scan_file,
            delimiter=",",
            comments="#",
            skiprows=5
        )

        position = np.array([
            pose_row["x"],
            pose_row["y"],
            pose_row["z"]
        ], dtype=np.float32)

        timestamp = pose_row["timestamp"]

        return {
            "frame": frame_id,
            "timestamp": timestamp,
            "position": position,
            "points": points
        }

    def __iter__(self):

        for frame_id in range(len(self)):
            yield self.get_frame(frame_id)

def sample_points(points, num_samples=4000):

    n = len(points)

    if n <= num_samples:
        return points

    idx = np.random.choice(
        n,
        size=num_samples,
        replace=False
    )

    return points[idx]


dataset = SyntheticLidarDataset("dataset")

print(len(dataset))

for frame in dataset:
    
    points = frame["points"]


    sampled_points = sample_points(
        points,
        num_samples=4000
    )
    
    sensor_pos = frame["position"]


    print(sensor_pos)
    print(sampled_points.shape)

    # for point in sampled_points:
        # process_point(point, sensor_pos)
        # ATC-DT update
        # pass

    # print(
    #     frame["frame"],
    #     points.shape,
    #     sensor_pos
    # )