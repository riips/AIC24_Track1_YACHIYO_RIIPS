import os
import json
import argparse
from datetime import datetime

from tracking import Tracker
from utils import DetectedObjects

def run_scpt(feature_data_root, out_dir="outdir", tracking_params={}):
    # Load and generate "detected object list"
    tracking_results = {}
    if not os.path.isdir(feature_data_root):
        raise Exception(f"No such directory: {feature_data_root}")
    if os.path.basename(feature_data_root).startswith("camera_"):
        camera_ids = [os.path.basename(feature_data_root)]
        feature_data_root = os.path.dirname(feature_data_root)
        is_multi = False
    else:
        camera_ids = [cam_id for cam_id in os.listdir(feature_data_root) if cam_id[:7] == "camera_"]
        is_multi = True

    # loading detections
    for camera_id in camera_ids:
        data_dir = os.path.join(feature_data_root, camera_id)
        camera_id = int(camera_id[7:])
        detected_objects = load_detections(data_dir)
        tracking_results[camera_id] = detected_objects.to_trackingdict()
        del detected_objects
    
    # Run SCT on all detections of all cameras
    for camera_id in tracking_results:
        tracking_dict = tracking_results[camera_id]
        start_time = datetime.now()
        tracker = Tracker(tracking_params)
        tracking_results[camera_id] = tracker.scpt(tracking_dict) # tracking returns tracking_dict
        end_time = datetime.now()
        print(f"Camera{camera_id} elapsed time: {end_time - start_time}")

        # Dump the result
        out_json = os.path.join(out_dir, f'camera{camera_id:03d}_tracking_results.json')
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, mode='w') as f:
            json.dump(tracking_results[camera_id], f)        

def run_mcpt(scene_id, json_dir,out_dir="outdir", tracking_params={}):
    start_time = datetime.now()
    tracker = Tracker(tracking_params)
    whole_tracking_result = tracker.mcpt(scene_id, json_dir,out_dir)
    
    # Dump the result
    out_file = os.path.join(out_dir, 'whole_tracking_results.json')
    with open(out_file, mode='w') as f:
        json.dump(whole_tracking_result, f)
    end_time = datetime.now()
    print(f"Elapsed_time: {end_time - start_time}")


def correct_scpt_result(scene_id, json_dir, out_dir=None, tracking_params={}):
    if not os.path.isdir(json_dir):
        raise Exception(f"The directory '{json_dir}' does not exist.")
    if out_dir == None:
        out_dir = json_dir
    
    json_files = [f for f in os.listdir(json_dir) if os.path.splitext(f)[1].lower() == ".json" and f.startswith("camera")]
    json_files = sorted(json_files)
    for json_file in json_files:
        camera_id = int(json_file.split("_")[0][6:])
        with open(os.path.join(json_dir, json_file)) as f:
            tracking_dict = json.load(f)
        tracker = Tracker(tracking_params)
        tracking_dict = tracker.correcting_scpt_result(tracking_dict) 
        out_file = os.path.join(out_dir, "fixed_"+os.path.basename(json_file))
        with open(out_file, mode='w') as f:
            json.dump(tracking_dict, f)

def correct_mcpt_result(scene_id,json_dir,out_dir,tracking_params={}):
    with open(os.path.join(json_dir, 'whole_tracking_results.json')) as f:
        tracking_results = json.load(f)
    with open(os.path.join(json_dir, f"representative_nodes_scene{str(scene_id)}.json")) as f:
        representative_nodes = json.load(f)
    tracker = Tracker(tracking_params)
    tracking_resuluts = tracker.correcting_mcpt_result(scene_id,tracking_results,representative_nodes)
    out_file = os.path.join(out_dir, "fixed_whole_tracking_results.json")
    with open(out_file, mode='w') as f:
        json.dump(tracking_resuluts, f)


def load_detections(data_root, debug=False):
    print(f"Loading detections from {data_root}.")
    detected_objects = DetectedObjects()
    detected_objects.load_from_directory(feature_root=data_root)
    print(f"Found {len(detected_objects.objects)} frames, and {detected_objects.num_objects} objects.")
    if debug:
        frames = sorted(detected_objects.objects)
        min_num_obj = 9999999
        max_num_obj = 0
        for frame in frames:
            obj = detected_objects[frame]
            num = len(obj)
            min_num_obj = min(min_num_obj, num)
            max_num_obj = max(max_num_obj, num)
        print(f"###  MIN num detections: {min_num_obj},  MAX num detections: {max_num_obj} ###\n")

    return detected_objects

def get_args():
    parser = argparse.ArgumentParser(description='Offline Tracker sample app.')
    parser.add_argument('-d', '--data', default='EmbedFeature/scene_001', type=str)
    parser.add_argument('-o', '--outdir', default='output', type=str)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    run(feature_data_root=args.data, out_dir=args.outdir, tracking_params={})
