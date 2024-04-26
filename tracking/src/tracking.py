import os
import sys
import tqdm
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity

from utils import DetectedObjects
from scpt import *
from mcpt import *

class Tracker():
    """
    This class represents YOTM, aka Yoshida Offline Tracking Method.
    """
    def __init__(self, params={}):
        self.camera_ids = []
        self.tracking_dicts = {}
        self._init_parameters()
        self.update_parameters(**params)
        self.frame_period = self.parameters["time_period"] * self.parameters["fps"]

    def _init_parameters(self):

        #self.parameters[""]: = 
        self.parameters = {}
        self.parameters["image_size"] = (1920,1080)

        # sct parameters
        self.parameters["time_period"]:int = 3
        self.parameters["fps"]:int = 30
        self.parameters["epsilon_scpt"]:float = 0.1
        self.parameters["min_samples"]:int = 4
        self.parameters["remove_noise_cluster"]:bool = True
        self.parameters["overlap_suppression"]:bool = True
        self.parameters["num_candidates"]:int = 10
        self.parameters["clustering_method"]:str = "agglomerative" #agglomerative or dbsacn
        self.parameters["debug"]:bool = False
        
        #fix_sct parameters
        self.parameters["sequential_nms"]:bool = True
        self.parameters["temporally_snms_th"]:float = 0.6
        self.parameters["spatially_snms_th"]:float = 0.6
        self.parameters["merge_nonoverlap"]:bool = True

        self.parameters["separate_warp"]:bool = True
        self.parameters["warp_th"]:int = 40
        self.parameters["alpha"]:float = 0.5

        self.parameters["exclude_short_track"]:bool = False
        self.parameters["short_tracklet_th"]:int = 120

        self.parameters["exclude_motionless_track"]:bool = False        
        self.parameters["stop_track_th"]:int = 25

        # mct parameters
        self.parameters["epsilon_mcpt"]:float = 0.4
        self.parameters["keypoint_th"]:float = 0.8
        self.parameters["keypoint_condition_th"]:float = 1
        self.parameters["distance_th"]:int = 5

        self.parameters["check_sc_overlap"]:bool = False
        self.parameters["distance_type"]:str = "max" #max or mean or min
        self.parameters["replace_similarity_by_wcoordinate"]:bool = False
        self.parameters["replace_value"]: float = -10
        self.parameters["representative_selection_method"]:str = "keypoint" #keypoint or centrality
        self.parameters["aspect_th"]:float =0.5 
        
        # fix mct parameters
        self.parameters["reassign_global_id"]:bool = True
        self.parameters["short_track_th"]:int = 120
        self.parameters["delete_gid_th"]:int = 6000 
        self.parameters["assign_all_tracklet"]:bool = False
        self.parameters["sim_th"]:float = 0.75
        self.parameters["delete_few_camera_cluster"]:bool = False 
        
        self.parameters["measure_wcoordinate"]:bool = False

        self.parameters["remove_noise_image"]:bool = True

        self.parameters["delete_distant_person"]:bool = True

        self.parameters["interpolate_track"]:bool = True
        self.parameters["max_interpolate_interval"]:int = 15
        

    def update_parameter(self, parameter, value):
        if not parameter in self.parameters:
            print(f"Unknown parameter: {parameter}.")
            sys.exit()
            return
        self.parameters[parameter] = value
    
    def update_parameters(self, **params):
        for key in params:
            self.update_parameter(key, params[key])
    
    def scpt(self, tracking_dict):
        """
        This performs object tracking with single camera dataset.
        Most of code below are just copied from '20240214_OfflineTracking-Debug.ipynb' and tweaked few.
        """

        frame_period = self.parameters["time_period"] * self.parameters["fps"]
        epsilon = self.parameters["epsilon_scpt"]

        max_offlineid = -1
        last_frame = get_max_value_of_dict(tracking_dict, "Frame")
        time_section_serial_dict = {timesection:[] for timesection in range(last_frame//frame_period+1) }

        for serial in tracking_dict.keys():
            frame = tracking_dict[serial]["Frame"]
            time_section = frame // frame_period
            time_section_serial_dict[time_section].append(serial)

        for time_section in range(last_frame//frame_period+1): 
            serials = time_section_serial_dict[time_section]
            if len(serials) == 0: continue
            clusters = tracking_by_clustering(tracking_dict,serials, **self.parameters)

            clusters = [cluster+max_offlineid+1 if cluster != -1 else -i for i,cluster in enumerate(clusters)]
            max_offlineid = max(clusters) if max(clusters) > 0 else max_offlineid

            if time_section == 0:
                for serial,cluster in zip(serials,clusters):
                    tracking_dict[serial]["OfflineID"] = int(cluster)
            elif time_section > 0:
                past_serials = time_section_serial_dict[time_section-1]
                tracking_dict = associate_cluster_between_period(tracking_dict, clusters, serials, past_serials, **self.parameters)  

        # We have tracking results in TrackingDict, yet will gather results for debugging. Could be deleted.
        offline_ids = [tracking_dict[serial]["OfflineID"] for serial in tracking_dict]
        new_offline_ids_dict = {key:i for i,key in enumerate(set(offline_ids)) if key != -1}
        new_offline_ids_dict[-1] = -1        

        for serial in tracking_dict:
            offline_id = tracking_dict[serial]["OfflineID"]
            tracking_dict[serial]["OfflineID"] = new_offline_ids_dict[offline_id]

        return tracking_dict
    
    def correcting_scpt_result(self,tracking_dict,**kwargs): 

        sequential_nms  = self.parameters["sequential_nms"]
        separate_warp = self.parameters["separate_warp"]
        exclude_short_track = self.parameters["exclude_short_track"]
        exclude_motionless_track = self.parameters["exclude_motionless_track"]    
        print("sequential_nms:",sequential_nms)
        print("separate_warp:",separate_warp)
        print("exclude_short_track:",exclude_short_track)
        print("exclude_motionless_track:",exclude_motionless_track)    

        if sequential_nms:
            tracking_dict = sequential_non_maximum_suppression(tracking_dict, **self.parameters) 
        if separate_warp:
            tracking_dict = separate_warp_tracklet(tracking_dict, **self.parameters)
        if exclude_short_track:
            tracking_dict = exclude_short_tracklet(tracking_dict, **self.parameters)
        if exclude_motionless_track:
            tracking_dict = exclude_motionless_tracklet(tracking_dict, **self.parameters)
        return tracking_dict

    def mcpt(self,scene_id, json_dir,out_dir):
        epsilon = self.parameters["epsilon_mcpt"]

        if not os.path.isdir(json_dir):
            raise Exception(f"The directory '{json_dir}' does not exist.")
        if out_dir == None:
            out_dir = json_dir
        tracking_results = {}
        json_files = [f for f in os.listdir(json_dir) if os.path.splitext(f)[1].lower() == ".json" and f.startswith("fixed_camera")]
        json_files = sorted(json_files)
        for json_file in json_files:
            camera_id = int(json_file.split("_")[1][6:])
            with open(os.path.join(json_dir, json_file)) as f:
                tracking_dict = json.load(f)
            print(f"{json_file} len(serials):{len(tracking_dict)}")
            tracking_results[camera_id] = tracking_dict
        tracking_results = multi_camera_people_tracking(tracking_results, scene_id=scene_id, json_dir=json_dir, out_dir=out_dir, **self.parameters)

        return tracking_results

    def correcting_mcpt_result(self,scene_id,tracking_results,represntative_nodes,**kwargs):
        reassign_global_id  = self.parameters["reassign_global_id"]
        measure_wcoordinate = self.parameters["measure_wcoordinate"]
        interpolate_track = self.parameters["interpolate_track"]
        remove_noise_image = self.parameters["remove_noise_image"]
        delete_distant_person = self.parameters["delete_distant_person"]
        print("reassign_global_id:",reassign_global_id)
        print("measure_wcoordinate:",measure_wcoordinate)
        print("interpolate_track:",interpolate_track)
        print("delete_distant_person:",delete_distant_person)
        
        if reassign_global_id:
            tracking_results = global_id_reassignment(tracking_results,represntative_nodes,scene_id,**self.parameters)
        if measure_wcoordinate:
            tracking_results = measure_world_coordinate(scene_id,tracking_results,**self.parameters)
        if remove_noise_image:
            tracking_results = remove_noise_images(scene_id,tracking_results,**self.parameters)
        if delete_distant_person:
            tracking_results = delete_distant_persons(tracking_results,**self.parameters)
        if interpolate_track:
            tracking_results = interpolate_tracklet(tracking_results,represntative_nodes,**self.parameters) 
        
        return tracking_results
