import os
import json
import numpy as np
import PIL.Image as Image
import cv2
from multiprocessing import Pool
from sys import stdout
import argparse
import os.path as osp

def make_parser():
    parser = argparse.ArgumentParser("reid")
    parser.add_argument("root_path", type=str, default=None)
    parser.add_argument("-s", "--scene", type=str, default=None)
    return parser

args = make_parser().parse_args()
data_root = osp.join(args.root_path, "Original")
scene = args.scene

fprint, endl = stdout.write, "\n"

IMAGE_FORMAT = ".jpg"


def video2image(parameter_set):
    scenario, camera, camera_dir = parameter_set
    fprint(f"[Processing] {scenario} {camera}{endl}")
    imgs_dir = f"{camera_dir}/Frame"
    if not os.path.exists(imgs_dir):
        os.makedirs(imgs_dir)
    print("camera_dir:" + camera_dir)
    cap = cv2.VideoCapture(f"{camera_dir}/video.mp4")
    current_frame = 1
    ret, frame = cap.read()
    while ret:
        frame_file_name = f"{str(current_frame).zfill(6)}{IMAGE_FORMAT}"
        cv2.imwrite(f"{imgs_dir}/{frame_file_name}", frame)
        ret, frame = cap.read()
        current_frame += 1
    fprint(f"[Done] {scenario} {camera}{endl}")


def main():
    parameter_sets = []
    scenario_dir = osp.join(data_root, scene)
    cameras = os.listdir(scenario_dir)
    for each_camera in cameras:
        cam = each_camera
        if "map" in each_camera:
            continue
        camera_dir = f"{scenario_dir}/{each_camera}"                
        parameter_sets.append(
            [scene, each_camera, camera_dir]
        )

    pool = Pool(processes=len(parameter_sets))
    pool.map(video2image, parameter_sets)
    pool.close()


if __name__ == "__main__":
    main()

