import os
import json
import glob
import argparse
from datetime import datetime
import sys

from utils import DetectedObjects, JSONHandler, get_camera_ids

import pose
from pose import IdentifiabilityEvaluator

def run_preprocess(feature_data_root, camera_id=None, out_dir="outdir",common_params={}):
    if not os.path.isdir(feature_data_root):
        raise Exception(f"No such directory: {feature_data_root}")
    # loading detections
    tracking_results = {}
    detected_objects = load_detections(feature_data_root,common_params=common_params)
    tracking_results[camera_id] = detected_objects.to_trackingdict()
    del detected_objects
    os.makedirs(out_dir, exist_ok=True)
    JSONHandler.save_json(tracking_results,os.path.join(out_dir, f'camera{camera_id:03d}_results.json'))

    identifiability_results = {}
    evaluator = IdentifiabilityEvaluator(tracking_dict=tracking_results[camera_id], common_params=common_params)
    # evaluator.load_keypoint_result()
    identifiability_results[camera_id] = evaluator.eval_identifiabilities(tracking_results[camera_id])
    JSONHandler.save_json(identifiability_results,os.path.join(out_dir, f'camera{camera_id:03d}_identifiability.json'))

def run_scpt(scene_id,camera_id,json_dir=None,out_dir="outdir",common_params={}, tracking_params={}):
    from scpt import OfflineSingleCameraTracker
    # Load and generate "detected object list"
    
    if not os.path.isdir(json_dir):
        raise Exception(f"The directory '{json_dir}' does not exist.")
    if out_dir == None:
        out_dir = json_dir
    start_time = datetime.now()

    tracking_results = JSONHandler.open_json(os.path.join(json_dir, f"camera{str(camera_id).zfill(3)}_results.json")) 
    identifiability_results = JSONHandler.open_json(os.path.join(json_dir, f"camera{str(camera_id).zfill(3)}_identifiability.json")) 

    tracker = OfflineSingleCameraTracker(tracking_results[str(camera_id)],identifiability_results=identifiability_results[str(camera_id)],common_params=common_params, tracking_params=tracking_params)
    tracking_results[str(camera_id)] = tracker.run_tracking() # tracking returns tracking_dict
    end_time = datetime.now()
    print(f"Camera{camera_id} elapsed time: {end_time - start_time}")

    # Dump the result
    os.makedirs(out_dir, exist_ok=True)
    JSONHandler.save_json(tracking_results,os.path.join(out_dir, f'camera{camera_id:03d}_results.json'))

def run_mcpt(scene_id, json_dir, out_dir="outdir", common_params={},tracking_params={}):
    from mcpt import OfflineMultiCameraTracker
    start_time = datetime.now()

    tracker = OfflineMultiCameraTracker(JSONHandler.concat_jsons(json_dir,startwith="postprocessed_camera",endwith="results"),
                                        JSONHandler.concat_jsons(json_dir,endwith="identifiability"),
                                        out_dir,common_params=common_params, tracking_params=tracking_params)
    whole_tracking_result = tracker.run_tracking() #scene_id, json_dir,out_dir
    
    # Dump the result
    JSONHandler.save_json(whole_tracking_result,os.path.join(out_dir, 'multi_camera_results.json'))
    end_time = datetime.now()
    print(f"Elapsed_time: {end_time - start_time}\n")

def run_scpt_postprocess(scene_id,camera_id, json_dir, out_dir=None, params={}):
    from scpt import SCPTPostProcessor
    
    if not os.path.isdir(json_dir):
        raise Exception(f"The directory '{json_dir}' does not exist.")
    if out_dir == None:
        out_dir = json_dir

    json_file = f"camera{str(camera_id).zfill(3)}_results.json"
    tracking_results = JSONHandler.open_json(os.path.join(json_dir, json_file))

    processor = SCPTPostProcessor(tracking_results[str(camera_id)], params=params)
    tracking_results[str(camera_id)] = processor.run_scpt_postprocess()
    JSONHandler.save_json(tracking_results, os.path.join(out_dir, "postprocessed_"+os.path.basename(json_file)))


def run_mcpt_postprocess(scene_id,json_dir,out_dir,params={}): #common_params={},tracking_
    from mcpt import MCPTPostProcessor

    processor = MCPTPostProcessor(tracking_results = JSONHandler.open_json(os.path.join(json_dir, 'multi_camera_results.json')),
                                  representative_nodes = JSONHandler.open_json(os.path.join(json_dir, f'representative_nodes.json')),
                                  out_dir=out_dir,params=params)
    tracking_results = processor.run_mcpt_postprocess() #scene_id,tracking_results
    JSONHandler.save_json(tracking_results, os.path.join(out_dir, "postprocessed_multi_camera_results.json"))

def load_detections(data_root,common_params={}):
    print(f"Loading detections from {data_root}.")
    detected_objects = DetectedObjects(params=common_params)
    detected_objects.load_from_directory(feature_root=data_root)
    print(f"Found {len(detected_objects.objects)} frames, and {detected_objects.num_objects} objects.")
    return detected_objects

def get_args():
    parser = argparse.ArgumentParser(description='Offline Tracker sample app.')
    parser.add_argument('-s', '--scene', type=int, required=True)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    # Common parameters
    common_params={"scene_id":args.scene,"camera_id":12}

    # Preprocessing parameters
    preprocess_params = {"keypoint_th":0.7}

    # SCPT parameters
    scpt_params={"epsilon": 0.15,"is_print_parameters":False,"frame_period":90,"is_exclude_low_identifiable_img":True} #,"is_use_motion":False
    scpt_postprocess_params={"short_tracklet_th":20, "alpha":0.3, "is_run_sequential_nms":False, "is_run_separate_warp": False, "is_run_exclude_short_track": False, "is_run_exclude_motionless_track": False}

    # MCPT parameters
    mcpt_params={"scene_id":args.scene,"epsilon": 0.35,"keypoint_th":0.7,"resample_representative":True}
    mcpt_postprocess_params={"scene_id":args.scene,"is_print_parameters":True,"delete_distant_person":False,"interpolate_track":False,"remove_noise_image":False}

    emb_model = "osnet_x1_0" #   "synthetic_osnet"
    out_sct_dir = f"scene_{args.scene:02}"

    # Getting camera IDs from scene ID
    camera_ids = get_camera_ids(args.scene, "tracking/config/scene_2_camera_id_file.json")
    print(f"Target scene ID: {args.scene}, camera IDs: {camera_ids}")

    # SCPT for all cameras
    for i, camera_id in enumerate(camera_ids):
        scpt_params["is_print_parameters"] = True if i == 0 else False
        common_params["camera_id"] = camera_id
        feature_data_root=os.path.join("EmbedFeature", emb_model, f"Scene{str(args.scene).zfill(2)}",f"Camera{str(camera_id).zfill(3)}")
        run_preprocess(feature_data_root=feature_data_root, out_dir=out_sct_dir, camera_id=camera_id, common_params=common_params)

        scpt_params["is_print_parameters"] = True if i == 0 else False
        common_params["camera_id"] = camera_id
        run_scpt(scene_id=args.scene, camera_id=camera_id, out_dir=out_sct_dir, json_dir=out_sct_dir, common_params=common_params, tracking_params=scpt_params)

        scpt_postprocess_params["is_print_parameters"] = True if i == 0 else False
        run_scpt_postprocess(scene_id=args.scene, camera_id=camera_id, json_dir=out_sct_dir, out_dir=out_sct_dir, params=scpt_postprocess_params)

    # MCPT
    run_mcpt(scene_id=args.scene, json_dir=out_sct_dir, out_dir=out_sct_dir, common_params=common_params, tracking_params=mcpt_params)  
    run_mcpt_postprocess(scene_id=args.scene, json_dir=out_sct_dir, out_dir=out_sct_dir, params=mcpt_postprocess_params)
