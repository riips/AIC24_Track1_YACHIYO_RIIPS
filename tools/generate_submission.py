import json
import argparse
import os
import numpy as np

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def convert_coordinates_2world(x, y):
    vector_xyz = np.array([x, y, z])
    vector_xyz_3d = np.dot(np.linalg.inv(homography_matrix), vector_xyz.T)
    vector_xyz_3d = vector_xyz_3d / vector_xyz_3d[2]
    return vector_xyz_3d[0], vector_xyz_3d[1]

def load_calibration(calib_path):
    data = read_json_file(calib_path)
    global camera_projection_matrix
    global homography_matrix
    camera_projection_matrix = np.array(data["camera projection matrix"])
    homography_matrix =  np.array(data["homography matrix"])

def generate_submission(json_path, data_root="", save_path=""):
    json_path = os.path.join(data_root, json_path)
    submission_path = os.path.join(data_root, save_path )
    if not os.path.exists(submission_path):
            os.makedirs(submission_path)
    submission_path = os.path.join(submission_path, 'track1.txt')
    json_data = read_json_file(json_path)
    ret_data = []
    for cam in json_data:
        print(f"processing camera : {cam.zfill(3)}")
        for seq in json_data[cam]:
            item = json_data[cam][seq]
            if "GlobalOfflineID" in item:
                ret_line = [cam, \
                            item["GlobalOfflineID"], \
                            (item["Frame"] - 1), \
                            item["Coordinate"]["x1"], \
                            item["Coordinate"]["y1"], \
                            (item["Coordinate"]["x2"] - item["Coordinate"]["x1"]), \
                            (item["Coordinate"]["y2"] - item["Coordinate"]["y1"]), \
                            "{:.6f}".format(item["WorldCoordinate"]["x"]), \
                            "{:.6f}".format(item["WorldCoordinate"]["y"])]
                ret_data.append(ret_line)
    ret_data = sorted(ret_data, key=lambda x: (int(x[0]), int(x[2]), int(x[1])))
    np.savetxt(submission_path, ret_data, delimiter=' ', fmt="%s")


if __name__ == "__main__":
    print("create track1.txt")
    scenes = os.listdir("./Tracking/")
    for sc in scenes:
        print(f"processing scene : {sc}")
        generate_submission(json_path=os.path.join(f"Tracking", sc,"fixed_whole_tracking_results.json"), save_path=os.path.join(f"Submission", sc))

    print("merge track1.txt")
    with open(os.path.join("Submission", "track1.txt"), "w") as merged_file:
        for file_path in scenes:
            with open(os.path.join("Submission", f"{file_path}/track1.txt"), "r") as file:
                merged_file.write(file.read())