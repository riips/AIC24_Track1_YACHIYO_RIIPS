import sys
import os
import json
import numpy as np
from datetime import datetime
from collections import defaultdict, Counter
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from scipy.interpolate import RegularGridInterpolator
from itertools import combinations
from scipy.stats import mode
from scipy.spatial.distance import pdist, squareform

#original module
from utils import TrackingFunctions, ParameterManager,JSONHandler
from pose import IdentifiabilityEvaluator
from parameters import CommonParameters, MCPTParameters, MCPTPostprocessParameters

"""
Definitions for clustering to muilt-camera tracking.
"""

class OfflineMultiCameraTracker(TrackingFunctions,ParameterManager,CommonParameters,MCPTParameters):
    def __init__(self, tracking_results, identifiability_results, out_dir, common_params={},tracking_params={}):
        TrackingFunctions.__init__(self)
        ParameterManager.__init__(self)
        CommonParameters.__init__(self)
        MCPTParameters.__init__(self)
        if common_params:
            self.update_params(common_params)
        if tracking_params:
            self.update_params(tracking_params)

        self.tracking_results = tracking_results
        self.append_tmp_globalid()
        self.identifiability_results = identifiability_results
        self.out_dir = out_dir

    def run_tracking(self): #, scene_id, json_dir, out_dir,
        # perform mcpt using tracking_results
        # tracking_results contains tracking_dict, which contains results of scpt in each camera
        print("=== start mcpt ===")
        if self.is_print_parameters:
            print("**mcpt main parameters**")
            print("scene_id:",self.scene_id)
            print("epsilon:",self.epsilon)
            print("is_use_motion_feature:",self.is_use_motion_feature)
            print("keypoint_condition_th:",self.keypoint_condition_th)
            print("keypoint_th:",self.keypoint_th,"\n")

        self._localid_int2str()
       
        # Representative image extraction
        representative_nodes = self._get_representative_nodes()
        
        similarity_matrix = self.create_similarity_matrix_mcpt(representative_nodes)
        similarity_matrix[similarity_matrix < (1-self.epsilon)] = 0    
        
        if self.is_use_motion_feature:
            """
            Todo 
            SCPTに倣ってOptimizeをSimilarityMatrixを作るときに実行する
            """
            print("replace similarity using motion feature")
            similarity_matrix = self.replace_similarity(representative_nodes,similarity_matrix)
        clusters = list(range(len(similarity_matrix)))

        print("number of tracklet:",len(set(clusters)))

        #perform Re-identification using hieralchical clustering with average linkage
        clusters = self.associate_cluster(clusters, similarity_matrix, epsilon=self.epsilon, cost_function=2) 

        self.assign_global_id(representative_nodes,clusters)
        return self.tracking_results

    def replace_similarity(self, representative_nodes,similarity_matrix,**params):        
        if self.check_sc_overlap:
            camera_localid_dict = self.make_camera_localid_dict(representative_nodes)
            similarity_matrix = self.minimize_similarity_by_sc_overlap(representative_nodes,similarity_matrix,camera_localid_dict)
        if self.replace_similarity_by_wcoordinate:
            min_distance_matrix,max_distance_matrix,mean_distance_matrix = self.create_distance_matrix(representative_nodes)

            similarity_matrix = np.where(np.less(mean_distance_matrix,0.5), 1, similarity_matrix)

            min_distance_matrix = np.where(min_distance_matrix == np.inf, -np.inf, min_distance_matrix)
            similarity_matrix = np.where(np.greater(min_distance_matrix,self.distance_th), self.replace_value, similarity_matrix)            
        return similarity_matrix

    def minimize_similarity_by_sc_overlap(self,representative_nodes,matrix,camera_localid_dict):
        # minimize similarity if tracklets are overlapping in SCPT results
        replace_value = -1
        for camera_id in representative_nodes:
            setattr(self,"tracking_dict",self.tracking_results[camera_id])
            indices = camera_localid_dict[camera_id]["indices"]
            unique_local_ids = camera_localid_dict[camera_id]["unique_local_ids"]
            local_ids,frames = zip(*[(self.tracking_dict[serial]["LocalID"],self.tracking_dict[serial]["Frame"]) for serial in self.tracking_dict])
            local_id_frame_dict = self.make_dict(local_ids,frames)
            for index1 in range(len(indices)-1):
                local_id1 = unique_local_ids[index1]
                id1_frames = local_id_frame_dict[local_id1]
                id1_index = indices[index1]
                for index2 in range(index1+1,len(indices)):   
                    local_id2 = unique_local_ids[index2]
                    id2_frames = local_id_frame_dict[local_id2]
                    common_frames = set(id1_frames).intersection(set(id2_frames))
                    if len(common_frames) == 0: continue
                    id2_index = indices[index2]
                    matrix[id1_index,id2_index] = replace_value
                    matrix[id2_index,id1_index] = replace_value     
        return matrix

    def append_tmp_globalid(self,):
        for camera_id,tracking_dict in self.tracking_results.items():  
            for serial,value in tracking_dict.items(): 
                value["GlobalID"] = -1
            self.tracking_results[camera_id] = tracking_dict

    def _localid_int2str(self,):
        for camera_id,tracking_dict in self.tracking_results.items():  
            for serial,value in tracking_dict.items(): 
                value["LocalID"] = str(value["LocalID"])
            self.tracking_results[camera_id] = tracking_dict

    def _get_representative_nodes(self,scene_id=None, print_txt=True):
        scene_id = self.scene_id if scene_id is None else scene_id
        # Get cached representative nodes info if any
        representative_node_json = os.path.join(self.out_dir, f"representative_nodes.json")
        if not os.path.isfile(representative_node_json) or self.resample_representative:
            print(f"make representative_nodes_scene{self.scene_id}.json")
            representative_nodes = self.make_representative_nodes()
            json_path = os.path.join(self.out_dir, f"representative_nodes.json")
            if self.save_representative_json:
                JSONHandler.save_json(representative_nodes,json_path)
            return representative_nodes
        else:
            if print_txt:
                print(f"load representative_nodes.json")
            representative_nodes = JSONHandler.open_json(representative_node_json)
            return representative_nodes

    def select_identifiable_node(self,serials):
        results = [tuple(self.identifiability_results[self.camera_id][serial].values()) for serial in serials]
        intersects, conditions,  image_scores = zip(*results)
        min_condition = int(np.min(conditions))

        indeices_and_intersects = np.array([(i,intersect) for i,(condition,intersect) in enumerate(zip(conditions,intersects)) if condition ==  min_condition])
        index,min_intersect = indeices_and_intersects[np.argmin(indeices_and_intersects[:,1])]

        serial = serials[int(index)]
        return serial, min_condition

    def get_representative_serial(self,serials):
        # serial, score = self.eval_identifiable_by_keypoint(serials)
        serial, score = self.select_identifiable_node(serials)
        return serial, score

    def make_representative_nodes(self):
        representative_nodes = {}
        for camera_id,tracking_dict in self.tracking_results.items():
            setattr(self,"camera_id",camera_id)
            representative_nodes[camera_id] = {}
        
            serials, local_ids = zip(*[(serial,tracking_dict[serial]["LocalID"]) for serial in tracking_dict]) 
            local_id_serials_dict = self.make_dict(local_ids,serials)
            del serials, local_ids            

            # self.load_keypoint_result(self.scene_id,camera_id)
            for local_id,serials in local_id_serials_dict.items():
                serial, score = self.get_representative_serial(serials)
                if serial == None: continue 
                representative_node = {"serial": serial,"score":score, "npy_path": tracking_dict[serial]["NpyPath"]}
                representative_nodes[camera_id][local_id] = {"representative_node": representative_node, "all_serials": serials}
        self.delete_variable("keypoints_results")
        return representative_nodes

    def make_camera_localid_dict(self, representative_nodes,**params):
        camera_localid_dict = {camera_id:{"indices":[],"unique_local_ids":[]} for camera_id in representative_nodes}
        max_id = 0
        for camera_id, value in representative_nodes.items():
            local_ids = []
            for local_id,representative_info in value.items():
                serials = representative_info["all_serials"]
                if self.representative_selection_method == "keypoint" and representative_info["representative_node"]["score"] > self.keypoint_condition_th:
                    continue
                local_ids.append(local_id)
            unique_local_ids = sorted(list(set(local_ids)))
            camera_localid_dict[camera_id]["indices"] += list(range(max_id,max_id+len(unique_local_ids)))
            camera_localid_dict[camera_id]["unique_local_ids"] += unique_local_ids
            max_id += len(unique_local_ids)
        return camera_localid_dict 

    def judge_is_representative(self,representative_node):
        is_representative = not (self.representative_selection_method == "keypoint" and representative_node["score"] > self.keypoint_condition_th)
        return is_representative

    def list_from_representative_nodes(self,representative_nodes, append_func):
        result = []
        for camera_id,value in representative_nodes.items():
            for local_id,representative_info in value.items():
                representative_node = representative_info["representative_node"]
                if self.judge_is_representative(representative_node):
                    result.append(append_func(camera_id, local_id, representative_node))
        return result

    def create_similarity_matrix_mcpt(self,representative_nodes,**params):
        # create similarity matrix from representative feature
        npy_paths = self.list_from_representative_nodes(representative_nodes, lambda camera_id, local_id, node: node["npy_path"])
        if len(npy_paths) == 0:
            raise Exception("All images are considered low identifiable.") 
        return self.similarity_matrix_by_npypaths(npy_paths)
    
    def assign_global_id(self,representative_nodes,clusters): #camera_localid_dict,
        representatives = self.list_from_representative_nodes(representative_nodes, lambda camera_id, local_id, node: (camera_id, local_id))

        for (camera_id,local_id),cluster in zip(representatives,clusters):
            for serial in representative_nodes[camera_id][local_id]["all_serials"]:
                self.tracking_results[camera_id][serial]["GlobalID"] = int(cluster)

    def create_distance_matrix_sub(self,representative_nodes,camera_localid_dict):
        index_frames_dict = defaultdict(list)
        index_wcoordinates_dict = defaultdict(list)
        
        for camera_id in representative_nodes:
            indices = camera_localid_dict[camera_id]["indices"]
            unique_local_ids = camera_localid_dict[camera_id]["unique_local_ids"]
            local_ids,serials = zip(*[(self.tracking_results[camera_id][serial]["LocalID"],serial) for serial in self.tracking_results[camera_id]])
            local_id_serials_dict = self.make_dict(local_ids,serials)
            
            for tmp_index in range(len(indices)):
                local_id = unique_local_ids[tmp_index]
                serials = local_id_serials_dict[local_id]
                frames = [self.tracking_results[camera_id][serial]["Frame"] for serial in serials]
                wcoordinates = [list(self.tracking_results[camera_id][serial]["WorldCoordinate"].values()) for serial in serials]

                frames, serials,wcoordinates = zip(*sorted(zip(frames,serials,wcoordinates)))
                index = indices[tmp_index]
                index_frames_dict[index].append(frames)
                index_wcoordinates_dict[index].append(wcoordinates)
        return index_frames_dict, index_wcoordinates_dict

    def create_distance_matrix(self, representative_nodes,  **params):
        # create a Euclidean distance matrix showing the Euclidean distance between each tracklet
        camera_localid_dict = self.make_camera_localid_dict(representative_nodes)

        index_frames_dict, index_wcoordinates_dict = self.create_distance_matrix_sub(representative_nodes,camera_localid_dict)

        shape = np.sum([len(camera_localid_dict[camera_id]["indices"]) for camera_id in camera_localid_dict])
        max_distance_matrix  = np.full((shape, shape), np.inf, dtype=np.float16)
        mean_distance_matrix  = np.full((shape, shape), np.inf, dtype=np.float16)
        min_distance_matrix  = np.full((shape, shape), np.inf, dtype=np.float16)

        for id1_index in range(len(max_distance_matrix)-1):
            id1_frames = index_frames_dict[id1_index]
            if id1_frames == []:
                continue
            id1_wcoordinates = index_wcoordinates_dict[id1_index]
            
            for id2_index in range(id1_index+1,len(max_distance_matrix)):   
                id2_frames = index_frames_dict[id2_index]
                if id2_frames == []:
                    continue
                common_frames = set(id1_frames).intersection(set(id2_frames))
                if len(common_frames) < 1: continue
                id2_wcoordinates = index_wcoordinates_dict[id2_index]
                id1_lap_indices = [i for i,id1_frame in enumerate(id1_frames) if id1_frame in common_frames]
                id2_lap_indices = [i for i,id2_frame in enumerate(id2_frames) if id2_frame in common_frames]
                id1_lap_wcoordinates = [id1_wcoordinates[id1_lap_index] for id1_lap_index in id1_lap_indices]
                id2_lap_wcoordinates = [id2_wcoordinates[id2_lap_index] for id2_lap_index in id2_lap_indices]

                euclid_distances = self.measure_euclidean_distances(id1_lap_wcoordinates,id2_lap_wcoordinates)
                min_distance = np.min(euclid_distances)
                mean_distance = np.mean(euclid_distances)
                max_distance = np.max(euclid_distances)
                min_distance_matrix[id1_index,id2_index] = min_distance
                min_distance_matrix[id2_index,id1_index] = min_distance
                if len(common_frames) > 120:
                    mean_distance_matrix[id1_index,id2_index] = mean_distance
                    mean_distance_matrix[id2_index,id1_index] = mean_distance
                    max_distance_matrix[id1_index,id2_index] = max_distance
                    max_distance_matrix[id2_index,id1_index] = max_distance
        return min_distance_matrix,max_distance_matrix,mean_distance_matrix



    ######################################################################################################
    ######################################################################################################

class MCPTPostProcessor(TrackingFunctions, ParameterManager ,CommonParameters, MCPTPostprocessParameters):
    def __init__(self, tracking_results,representative_nodes, out_dir=None, params={}):
        TrackingFunctions.__init__(self)
        # OfflineMultiCameraTracker.__init__(self)
        CommonParameters.__init__(self)
        MCPTPostprocessParameters.__init__(self)
        if params:
            self.update_params(params)

        self.tracking_results = tracking_results
        self.representative_nodes = representative_nodes
        self.out_dir = out_dir

    def run_mcpt_postprocess(self):
        print("=== start mcpt post process ===")
        if self.is_print_parameters:
            print("** mcpt post process contents **")
            print("is_reassign_global_id:",self.is_reassign_global_id)
            print("measure_wcoordinate:",self.measure_wcoordinate)
            print("remove_noise_image:",self.remove_noise_image)
            print("delete_distant_person:",self.delete_distant_person)
            print("interpolate_track:",self.interpolate_track,"\n")
        
        if self.is_reassign_global_id:
            self.global_id_reassignment() 
        if self.measure_wcoordinate:
            tracking_results = self.measure_world_coordinate()
        if self.remove_noise_image:
            self.remove_noise_images()
        if self.delete_distant_person:
            self.delete_distant_persons()
        if self.interpolate_track:
            self.interpolate_tracklet() 
        return self.tracking_results
    
    def global_id_reassignment(self,):
        # perform delete_small_global_id() and assign_global_id() for reassigning unclustered tracklets
        unique_global_ids = self.get_unique_global_ids()
        global_serials_dict = self.make_globalid_dict(additional_keys=["LocalID"]) 
        self.exclude_small_global_id(global_serials_dict)
        self.reassign_global_id()
    
    def get_global_ids(self,):
        global_ids = []
        for camera_id,value in self.representative_nodes.items():
            for local_id,representative_info in value.items():
                serial = representative_info["representative_node"]["serial"]
                global_id = self.tracking_results[camera_id][serial].get("GlobalID",-1)
                global_ids.append(self.tracking_results[camera_id][serial]["GlobalID"])
        return global_ids

    def get_unique_global_ids(self,):
        # get unique global ids from tracking_results 
        global_ids = self.get_global_ids()
        return sorted(list(set(global_ids)))

    def make_globalid_dict(self,additional_keys=[]):
        dict_ = defaultdict(list)
        for camera_id, tracking_dict in self.tracking_results.items():
            for serial, value in tracking_dict.items():
                tmp_list = [camera_id,serial]
                gid = value["GlobalID"]
                for key in additional_keys:
                    tmp_list.append(value[key])
                dict_[gid].append(tuple(tmp_list))
        return dict_

    def divide_save_or_delete(self,global_serial_dict):
        
        print("delete_gid_th:",self.delete_gid_th)
        print("is_delete_few_camera_cluster:",self.is_delete_few_camera_cluster)
        delete_global_ids = []
        save_global_ids = []    
        
        for global_id,value in global_serial_dict.items():
            camera_ids,serials,local_ids = zip(*value)
            serial_counter = 0
            for camera_id, serial, local_id in zip(camera_ids,serials,local_ids):
                serial_counter += len(self.representative_nodes[camera_id][local_id]["all_serials"])

            if serial_counter < self.delete_gid_th:
                delete_global_ids.append(global_id)
                continue
            if self.is_delete_few_camera_cluster and len(set(camera_ids)) < self.min_camera_appearance:
                delete_global_ids.append(global_id)
                continue
            save_global_ids.append(global_id)
        return save_global_ids, delete_global_ids
    
    def replace_result_value(self,key,delete_list,replace_value=-1):
        for camera_id,tracking_dict in self.tracking_results.items():
            for serial,value in tracking_dict.items():
                target_value = tracking_dict.get(key)        
                if target_value in delete_list:
                    self.tracking_results[camera_id][serial][key] = replace_value

    def exclude_small_global_id(self,global_serials_dict):
        # delete global id that contains only a little serials from tracking_results
        save_global_ids, delete_global_ids = self.divide_save_or_delete(global_serials_dict)
        if save_global_ids == []:
            raise ValueError("all global_ids are deleted.")
        self.replace_result_value("GlobalID",delete_global_ids)        

    def reassign_global_id(self,):
        # assign unclustered tracklets to global id
        print(f"sim_th:",self.sim_th)
        assigned_tracks, unassigned_tracks = self.divide_assigned_or_unassigned()
        global_ids,camera_ids,local_ids,serials = zip(*assigned_tracks)
        
        feature_stack = self.create_mcpt_feature_stack(camera_ids,serials)
        feature_stack_T = feature_stack.T
        feature_stack_norm = np.linalg.norm(feature_stack, axis=1)

        for (camera_id,local_id) in unassigned_tracks:
            representative_info = self.representative_nodes[camera_id][local_id]
            if representative_info["representative_node"]["score"] > 4:
                continue
            feature = np.load(representative_info["representative_node"]["npy_path"])
            cos_sims = np.dot(feature,feature_stack_T)/ (np.linalg.norm(feature)*feature_stack_norm)
            new_global_id = self.get_new_globalid(cos_sims,global_ids)
            if new_global_id == None: continue       
            serials =self.representative_nodes[camera_id][local_id]["all_serials"]
            for serial in serials:
                self.tracking_results[camera_id][serial]["GlobalID"] = int(new_global_id)
        # print(f"{counter} tracklets are reassigned")

    def get_new_globalid(self,cos_sims,global_ids):
        if self.assign_all_tracklet:
            similar_idndex = np.argmax(cos_sims)
            return global_ids[similar_idndex]
        if np.max(cos_sims) < self.sim_th:
            return None
        if self.reassign_criterion == "mode":
            similar_indices = list(np.where(cos_sims >= self.sim_th)[0])
            tmp_global_ids = [global_id for i,global_id in enumerate(global_ids) if i in similar_indices]
            new_global_id = mode(tmp_global_ids, keepdims=False).mode
        elif self.reassign_criterion == "max":
            similar_idndex = np.argmax(cos_sims)
            new_global_id = global_ids[similar_idndex]
        return new_global_id

    def create_mcpt_feature_stack(self,camera_ids,serials):
        npy_paths = [self.tracking_results[camera_id][serial]["NpyPath"] for camera_id,serial in zip(camera_ids,serials)]
        return np.vstack(self.loop(npy_paths,self.load_npy_from_npypath))

    def divide_assigned_or_unassigned(self,):
        assigned_tracks = []
        unassigned_tracks = []
        for camera_id, value in self.representative_nodes.items():
            for local_id,representative_info in value.items():
                serial = representative_info["representative_node"]["serial"]
                tracking_value = self.tracking_results[camera_id][serial]
                global_id = tracking_value["GlobalID"]
                if global_id != -1:                    
                    assigned_tracks.append((global_id,camera_id,local_id,serial))
                else:
                    unassigned_tracks.append((camera_id,local_id))
        return assigned_tracks, unassigned_tracks

    def delete_tracking_results(self,delete_list):
        for (camera_id,serial) in delete_list:
            del self.tracking_results[camera_id][serial] #["GlobalID"]
        
    def remove_noise_by_pose(self,):
        del_serials = []
        for serial in self.tracking_dict:
            value = self.tracking_dict[serial]
            kp = self.keypoints_results.get_keypoints(serial)
            if kp == None:
                del_serials.append(serial)
                continue
            keypoints = kp['Keypoints']
            coordinate = list(value["Coordinate"].values())
            w,h = coordinate[2]-coordinate[0],coordinate[3]-coordinate[1]
            aspect = h/w
            if  aspect < 1/3 or aspect > 5:
                del_serials.append(serial)
                continue
            # condition = self.eval_noise_level(keypoints)
            # if condition >= 2:
            #     if condition==2 and min(w,h) < 100:
            #         continue
            #     del_serials.append(serial)
        for serial in del_serials:
            self.tracking_dict[serial]["GlobalID"] = -1
    
    def remove_long_interval(self,):
        interval_th = 30
        local_ids = [self.tracking_dict[serial]["LocalID"] for serial in self.tracking_dict]

        local_id_serials_dict = self.make_dict(local_ids,self.tracking_dict.keys())
        local_id_frames_dict = self.make_dict(local_ids,[self.tracking_dict[serial]["Frame"] for serial in self.tracking_dict])

        target_serials = []
        for local_id in local_id_serials_dict:
            if type(local_id)==str:
                raise Exception("")

            if local_id == -1:
                continue
            frames, serials = zip(*sorted(zip(local_id_frames_dict[local_id], local_id_serials_dict[local_id])))
            for i in range(len(frames) - 3 + 1):
                past_frame,frame,future_frame = frames[i:i + 3]
                if (frame - past_frame >interval_th) and (future_frame - frame > interval_th):
                    del_serials.append(serials[i])
        for serial in target_serials:
            del self.tracking_dict[serial]["GlobalID"]

    def remove_noise_images(self,):
        # remove noise images based on pose estimation
        for camera_id,tracking_dict in self.tracking_results.items():
            setattr(self,"tracking_dict",tracking_dict)
            # self.remove_noise_by_pose()
            self.remove_long_interval()
            self.tracking_results[camera_id] = self.tracking_dict

        self.delete_variable("keypoints_results")
        self.delete_variable("tracking_dict")
        return 
    
    def get_distant_persons(self,gid_serials):
        print("distance_th:",self.distance_th)
        delete_list= []
        for gid,value in gid_serials.items():
            camera_ids,serials,frames = zip(*value)
            frames, serials,camera_ids = zip(*sorted(zip(frames, serials, camera_ids)))
            frame_counter = Counter(frames)
            if max(frame_counter.values()) < 2:
                continue
            start_index=0
            for frame in frame_counter:
                number = frame_counter[frame]
                if number <= 2:
                    pass
                else:
                    tmp_frames = frames[start_index:start_index+number]
                    tmp_serials = serials[start_index:start_index+number]
                    tmp_camera_ids = camera_ids[start_index:start_index+number]
                    
                    world_coordinates = [tuple(self.tracking_results[camera_id][serial]["WorldCoordinate"].values()) for camera_id,serial in zip(tmp_camera_ids,tmp_serials)]
                    distance_matrix = squareform(pdist(np.array(world_coordinates), 'euclidean'))
                    np.fill_diagonal(distance_matrix, np.inf)
                    min_distances = np.min(distance_matrix,axis=1)
                    max_distance = np.max(min_distances)
                    max_index = np.argmax(min_distances)
                    if max_distance > self.distance_th:
                        delete_list.append((tmp_camera_ids[max_index],tmp_serials[max_index]))
                start_index += number
        return delete_list

    def delete_distant_persons(self,):
        # delete the node that has long distances to other nodes with the same global id

        gid_serials = self.make_globalid_dict(keys=["Frame"])
        delete_list = self.get_distant_persons(gid_serials)
        replace_result_value(self,"GlobalID",delete_list,replace_value=-1)
        # self.delete_tracking_results(delete_list)

    def get_missing_frames(self, frames):
        missing_frames = []
        max_interval = self.max_interpolate_interval
        sampling_freq = self.frame_sampling_freq
        
        for frame, next_frame in zip(frames[:-1], frames[1:]):
            gap = next_frame - frame
            if gap > max_interval: 
                continue
            missing_frames.extend(range(frame + sampling_freq, next_frame, sampling_freq))
        return missing_frames

    def interpolate_tracklet(self,):
        # interpolate missing detections for each tracklet
        for camera_id,tracking_dict in self.tracking_results.items():
            local_ids = [tracking_dict[serial]["LocalID"] for serial in tracking_dict]
            unique_local_ids = sorted(list(set(local_ids)))
            if min(unique_local_ids) == -1: unique_local_ids.remove(-1)
            local_id_serials_dict = self.make_dict(local_ids, [serial for serial in tracking_dict])
            local_id_frames_dict = self.make_dict(local_ids, [tracking_dict[serial]["Frame"] for serial in tracking_dict])

            max_serial = int(max(tracking_dict.keys()))
            for local_id in unique_local_ids:
                if local_id == "-1": continue

                frames, serials = zip(*sorted(zip(local_id_frames_dict[local_id], local_id_serials_dict[local_id])))
                missing_frames = self.get_missing_frames(frames)
                if missing_frames==[]: continue
                # global_id = tracking_dict[serials[0]]["GlobalID"] if "GlobalID" in tracking_dict[serials[0]] else None
                global_id = tracking_dict[serials[0]].get("GlobalID", None)
                if global_id == None: continue
                coordinates = [list(tracking_dict[serial]["Coordinate"].values())+list(tracking_dict[serial]["WorldCoordinate"].values()) for serial in serials]
                interpolator = RegularGridInterpolator((np.array(frames),), np.array(coordinates), method='linear') 
                for frame in missing_frames:
                    x1,y1,x2,y2,w_x,w_y = interpolator([frame])[0]
                    (x1, y1, x2, y2), (w_x,w_y) = map(int, [x1, y1, x2, y2]),map(float,[w_x,w_y])
                    max_serial += 1
                    self.tracking_results[camera_id][str(max_serial)] = {"Frame": frame, "Coordinate": {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}, "WorldCoordinate": {'x': w_x, 'y': w_y}, "LocalID": local_id}

    def measure_world_coordinate(self,):
        # measur world coordinates in each node
        for camera_id,tracking_dict in self.tracking_results.items():   
            with open(f"Calibration/scene_{str(self.scene_id).zfill(3)}/camera_{str(camera_id).zfill(4)}.json") as f:
                calibration_json = json.load(f)
            homography_matrix = np.array(calibration_json['homography matrix'])
            for serial,value in tracking_dict.items():
                x1,y1,x2,y2 = value["Coordinate"].values()
                x,y =  (x2+x1)/2,y2
                fp_world_coordinate = self.translate_world_coordinate(x,y, homography_matrix)
                self.tracking_results[camera_id][serial]["WoorldCoordinate"] = {"x":fp_world_coordinate[0],"y":fp_world_coordinate[1]}

