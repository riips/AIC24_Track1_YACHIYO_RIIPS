import sys
import os
import json
import numpy as np
from datetime import datetime
from collections import Counter
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from scipy.interpolate import RegularGridInterpolator
from itertools import combinations
from scipy.stats import mode
from scipy.spatial.distance import pdist, squareform

from scpt import associate_cluster,agglomerative_clustering 
import pose

"""
Definitions for clustering to muilt-camera tracking.
"""

def get_max_value_of_dict(dictionary, key):
    # get max value of any key from nested dictionary
    max_value = float('-inf')  
    for k, v in dictionary.items():
        if isinstance(v, dict):
            max_value = max(max_value, get_max_value_of_dict(v, key))
        elif k == key:  
            max_value = max(max_value, v)
    return max_value

def create_similarity_matrix_mcpt(representative_nodes,**kwargs):
    # create similarity matrix from representative feature
    short_track_th = kwargs.get('short_track_th', 0)
    representative_selection_method = kwargs.get("representative_selection_method","keypoint") 
    keypoint_condition_th = kwargs.get("keypoint_condition_th",2)
    feature_stack = None
    for camera_id in representative_nodes:
        tmp_representative_nodes = representative_nodes[camera_id]
        for local_id in tmp_representative_nodes:
            value = tmp_representative_nodes[local_id]
            representative_node = value["representative_node"]
            serials =  value["all_serials"]

            if len(serials) < short_track_th:
                continue
            if representative_selection_method == "keypoint":
                score = representative_node["score"]
                if score > keypoint_condition_th:
                    continue

            feature = np.load(representative_node["npy_path"])
            if feature_stack is None:
                feature_stack = np.empty((0, len(feature.flatten())))
            feature_stack  = np.append(feature_stack , feature.reshape(1, -1) , axis=0)
    similarity_matrix = cosine_similarity(feature_stack)
    similarity_matrix = similarity_matrix.astype(np.float16)
    return similarity_matrix

def measure_intersect_area(rectangle1, rectangle2):
    # measure intersect area
    intersect_width = min(rectangle1[2], rectangle2[2]) - max(rectangle1[0], rectangle2[0])
    intersect_height = min(rectangle1[3], rectangle2[3]) - max(rectangle1[1], rectangle2[1])
    intersect_area = max(intersect_width, 0) * max(intersect_height, 0)
    return intersect_area

def eval_keypoints(serial,other_serials,keypoints_results,**kwargs):
    # evaluate results of pose estimation
    """
    condition = 1: All keypoints has high confidence 
    condition = 2: half of keypoint has high confidence (left or right half of the body)
    condition = 3: part of the keypoint has high confidence in left or right half of the body
    condition = 4: almost keypoints has low confidence 
    """
    keypoint_th = kwargs.get("keypoint_th",0.7)

    kp = keypoints_results.get_keypoints(serial)
    if kp == None:
        condition, intersect_ratio, score,area = 4, 1 , 0, 0
    else:
        x1,y1,x2,y2,bbox_confidence = kp["bbox"]
        keypoints = kp["Keypoints"]
        area = (x2-x1)*(y2-y1)
        x_list, y_list, scores = zip(*keypoints)

        intersect_area = 0
        for other_serial in other_serials :
            other_kp =keypoints_results.get_keypoints(other_serial)
            if other_kp==None: continue
            x1_,y1_,x2_,y2_,bbox_confidence = other_kp["bbox"]
            tmp_intersect_area = measure_intersect_area([x1,y1,x2,y2],[x1_,y1_,x2_,y2_]) 
            intersect_area = max(intersect_area,tmp_intersect_area)
        intersect_ratio = intersect_area/((x2-x1)*(y2-y1))

        if np.min(scores) >= keypoint_th:
            score = np.mean(scores)
            condition = 1
        else:
            right_scores = [score for i,score in enumerate(scores) if i%2==0]
            left_scores = [score for i,score in enumerate(scores) if i%2==1]
            nose_score = right_scores.pop(0)
            min_right_scores = np.min(right_scores)
            min_left_scores = np.min(left_scores)
            target_scores = left_scores if min_left_scores > min_right_scores else right_scores
            min_score = np.min(target_scores)
            score = np.mean(target_scores)
            if min_score >= keypoint_th:
                condition = 2
            else:
                count = len([tmp_score for tmp_score in target_scores if tmp_score >= keypoint_th])
                if count/len(target_scores) > 0.7:
                    condition =3
                else: 
                    condition = 4
    return condition, intersect_ratio, score, area

def find_high_confidence_keypoint_node(tracking_dict,serials,keypoints_results,frame_serials_dict,**kwargs):
    keypoint_th = kwargs.get("keypoint_th",0.7)

    conditions = []
    intersects = []
    image_scores = []
    areas = []
    
    for k,serial in enumerate(serials):
        frame = tracking_dict[serial]["Frame"]
        other_serials = frame_serials_dict[frame]
        other_serials.remove(serial)

        condition,intersect_ratio ,image_score,area = eval_keypoints(serial,other_serials,keypoints_results)
        conditions.append(condition)
        intersects.append(intersect_ratio)
        image_scores.append(image_score)
        areas.append(area)
    min_condition = np.min(conditions)
    index_area = np.array([(i,area) for i,(condition,area) in enumerate(zip(conditions,areas)) if condition ==  min_condition])
    max_index = np.argmax(index_area[:,1])
    index,max_area = index_area[max_index]

    serial = serials[int(index)]
    feature = np.load(tracking_dict[serial]["NpyPath"])
    return serial, feature, int(min_condition)

def decide_representative_nodes(tracking_results,out_dir,scene_id,**kwargs):
    # decide representative nodes from each tracklet
    epsilon = kwargs.get('epsilon_mcpt', 0.3)
    representative_selection_method = kwargs.get("representative_selection_method","centrality") 
    short_track_th = kwargs.get("short_track_th",20) 
    model = kwargs.get("model","mmpose_hrnet")
    keypoint_th = kwargs.get("keypoint_th",0.7)

    representative_nodes = {}
    for camera_id in tracking_results:
        representative_nodes[camera_id] = {}
        tracking_dict = tracking_results[camera_id]
        if representative_selection_method == "keypoint":
            keypoints_results = pose.PoseKeypoints(f"Pose/scene_{str(scene_id).zfill(3)}/camera_{str(camera_id).zfill(4)}/camera_{str(camera_id).zfill(4)}_out_keypoint.json")
            keypoints_results.assign_serial_from_tracking_dict(tracking_dict=tracking_dict)
            max_frame = get_max_value_of_dict(tracking_dict,"Frame")
            frame_serials_dict = {n+1:[] for n in range(max_frame)}
            [frame_serials_dict[tracking_dict[serial]["Frame"]].append(serial) for serial in tracking_dict if tracking_dict[serial]["OfflineID"] != -1]

        # Get each clusters, we need to iterate tracking_dict to extract cluster-wise data
        local_ids = [tracking_dict[serial]["OfflineID"] for serial in tracking_dict]
        unique_local_ids = sorted(set(local_ids))
        if -1 in unique_local_ids:
            unique_local_ids.remove(-1)
        local_id_serials_dict = {local_id:[] for local_id in unique_local_ids}
        [local_id_serials_dict[local_id].append(serial) for local_id,serial in zip(local_ids,tracking_dict) if local_id >= 0]

        # Get the representative node of each clusters
        for local_id in local_id_serials_dict:
            serials = local_id_serials_dict[local_id]
            if representative_selection_method == "centrality":
                serials, serial, feature = find_highest_centrality_node(tracking_dict, serials, epsilon=epsilon)
                if serial != None:
                    representative_node = {"serial": serial, "npy_path": tracking_dict[serial]["NpyPath"]}
            elif representative_selection_method == "keypoint":
                serial, feature, score = find_high_confidence_keypoint_node(tracking_dict,serials,keypoints_results,frame_serials_dict,keypoint_th = keypoint_th)
                representative_node = {"serial": serial,"score":score, "npy_path": tracking_dict[serial]["NpyPath"]}
            else:
                print("representative_selection_method is wrong")
                sys.exit()
            # Save result out to json
            if serials !=[]:
                
                representative_nodes[camera_id][local_id] = {"representative_node": representative_node, "all_serials": serials}
    json_path = os.path.join(out_dir, f"representative_nodes_scene{scene_id}.json")
    with open(json_path, "w") as f:
        json.dump(representative_nodes, f)

    return representative_nodes
        
def multi_camera_people_tracking(tracking_results, scene_id, json_dir, out_dir,**kwargs):
    # perform mcpt using tracking_results
    # tracking_results contains tracking_dict, which contains results of scpt in each camera
    print("running multi_camera_people_tracking")
    
    appearance_based_tracking = kwargs.get("appearance_based_tracking",True)
    distance_type = kwargs.get("distance_type","max")
    distance_th = kwargs.get("distance_th",5)
    epsilon = kwargs.get("epsilon_mcpt",0.4)
    representative_selection_method = kwargs.get("representative_selection_method","keypoint") 
    short_track_th = kwargs.get("short_track_th",0) 
    keypoint_th = kwargs.get("keypoint_th",0.7)
    keypoint_condition_th = kwargs.get("keypoint_condition_th",2)
    replace_similarity_by_wcoordinate = kwargs.get("replace_similarity_by_wcoordinate",True)
    replace_value = kwargs.get('replace_value', -10)

    print("representative_selection_method:",representative_selection_method)
    print("short_track_th:",short_track_th)
    print("epsilon:",epsilon)
    if representative_selection_method == "keypoint":
        print("keypoint_condition_th:",keypoint_condition_th)

    # Representative image extraction
    representative_nodes = get_representative_nodes_cache(scene_id=scene_id, out_dir=out_dir)
    if representative_nodes == None:
        representative_nodes = decide_representative_nodes(tracking_results,out_dir,scene_id,epsilon=epsilon,representative_selection_method=representative_selection_method,short_track_th=short_track_th,keypoint_th=keypoint_th)
    else:
        print(f"Found repsentative_nodes cache file. Got {len(representative_nodes)} camera(s) info.")
    print("representative feature is selected")
    
    similarity_matrix = create_similarity_matrix_mcpt(representative_nodes,short_track_th=short_track_th,representative_selection_method=representative_selection_method,keypoint_condition_th=keypoint_condition_th)
    similarity_matrix[similarity_matrix < (1-epsilon)] = 0    
    clusters = list(range(len(similarity_matrix)))
    print("number of tracklet:",len(set(clusters)))
    similarity_matrix = replace_similarity(representative_nodes,similarity_matrix,tracking_results,clusters,distance_th=distance_th, 
                                           distance_type=distance_type,replace_similarity_by_wcoordinate=replace_similarity_by_wcoordinate,
                                           short_track_th = short_track_th, keypoint_condition_th=keypoint_condition_th,
                                           representative_selection_method=representative_selection_method)
    # perform Re-identification using hieralchical clustering with average linkage
    clusters = associate_cluster(clusters, similarity_matrix, epsilon=epsilon, cost_function=2, minimize=False)
    del similarity_matrix

    print("unique_clusters:",len(set(clusters)))

    camera_dict = create_camera_dict(representative_nodes,short_track_th = short_track_th,
                                     keypoint_condition_th=keypoint_condition_th, representative_selection_method=representative_selection_method)

    for camera_id in camera_dict:
        tracking_dict = tracking_results[int(camera_id)]
        indices = camera_dict[camera_id]["indices"]
        local_ids = camera_dict[camera_id]["unique_local_ids"]
        tmp_clusters = [clusters[index] for index in indices]
        local_id_cluster_dict = {local_id:cluster for local_id,cluster in zip(local_ids,tmp_clusters)}

        local_ids = [tracking_dict[serial]["OfflineID"] for serial in tracking_dict]
        unique_local_ids = sorted(set(local_ids))
        if -1 in unique_local_ids:
            unique_local_ids.remove(-1)
        local_id_serials_dict = {local_id:[] for local_id in unique_local_ids}
        [local_id_serials_dict[local_id].append(serial) for local_id,serial in zip(local_ids,tracking_dict) if local_id >= 0]
        for local_id in unique_local_ids:
            for serial in local_id_serials_dict[local_id]:
                value = tracking_dict[serial]
                if local_id in local_id_cluster_dict:
                    value["GlobalOfflineID"] = int(local_id_cluster_dict[local_id])
    return tracking_results

def get_representative_nodes_cache(scene_id, out_dir):
    # Get cached representative nodes info if any
    representative_node_json = os.path.join(out_dir, f"representative_nodes_scene{scene_id}.json")
    if os.path.isfile(representative_node_json):
        with open(representative_node_json, "r") as f:
            representative_nodes = json.load(f)
            return representative_nodes
    return None

def get_unique_global_ids(tracking_results,representative_nodes):
    # get unique global ids from tracking_results 
    global_ids = []
    for camera_id in representative_nodes:
        tracking_dict = tracking_results[camera_id]
        for local_id in representative_nodes[camera_id]:
            serial = representative_nodes[camera_id][local_id]["representative_node"]["serial"]
            if "GlobalOfflineID" in tracking_dict[serial]:
                global_ids.append(tracking_dict[serial]["GlobalOfflineID"])
    unique_global_ids = sorted(list(set(global_ids)))
    return unique_global_ids

def get_serials_each_global_id(tracking_results,representative_nodes,unique_global_ids):
    # get serials assigned each global id
    global_serial_dict = {} #global_id: {camera_id:(local_id, serial)}
    for global_id in unique_global_ids:
        tmp_dict = {}
        for camera_id in representative_nodes:
            tmp_dict[camera_id] = []
        global_serial_dict[global_id] = tmp_dict
    for camera_id in representative_nodes:
        tracking_dict = tracking_results[camera_id]
        for local_id in representative_nodes[camera_id]:
            serial = representative_nodes[camera_id][local_id]["representative_node"]["serial"]
            if "GlobalOfflineID" in tracking_dict[serial]:
                global_id = tracking_dict[serial]["GlobalOfflineID"]
                global_serial_dict[global_id][camera_id].append((local_id,serial))
    return global_serial_dict

def create_camera_dict(representative_nodes,**kwargs):
    #
    short_track_th = kwargs.get('short_track_th', 0)
    representative_selection_method = kwargs.get("representative_selection_method","keypoint") 
    keypoint_condition_th = kwargs.get("keypoint_condition_th",2)

    camera_dict = {camera_id:{"indices":[],"unique_local_ids":[]} for camera_id in representative_nodes}
    max_id = 0
    for camera_id in representative_nodes:
        tmp_representative_nodes = representative_nodes[camera_id]
        local_ids = []
        for local_id in tmp_representative_nodes:
            serials =  tmp_representative_nodes[local_id]["all_serials"]
            if len(serials) < short_track_th:
                continue
            if representative_selection_method == "keypoint":
                score = tmp_representative_nodes[local_id]["representative_node"]["score"]
                if score > keypoint_condition_th:
                    continue
            local_ids.append(int(local_id))
        unique_local_ids = sorted(list(set(local_ids)))
        camera_dict[camera_id]["indices"] += list(range(max_id,max_id+len(unique_local_ids)))
        camera_dict[camera_id]["unique_local_ids"] += unique_local_ids
        max_id += len(unique_local_ids)
    return camera_dict 

def create_mcpt_feature_stack(tracking_results,target_list):
    feature_stack = None
    for camera_id, serial in target_list:
        feature = np.load(tracking_results[camera_id][serial]["NpyPath"])
        if feature_stack is None:
            feature_stack = np.empty((0, len(feature.flatten())))
        feature_stack  = np.append(feature_stack , feature.reshape(1, -1), axis=0)
    return feature_stack



def assign_global_id(tracking_results,representative_nodes,**kwargs):
    # assign unclustered tracklets to global id

    epsilon = kwargs.get('epsilon_mcpt', 0.3)
    assign_all_tracklet = kwargs.get('assign_all_tracklet', False)
    sim_th = kwargs.get('sim_th', 0.9)
    print("sim_th:",sim_th)
    print("assign_all_tracklet:",assign_all_tracklet)
    model = kwargs.get("model","mmpose_hrnet")

    counter = 0
    assigned_tracks = []
    unassigned_tracks = []

    for camera_id in representative_nodes:
        tracking_dict = tracking_results[camera_id]
        for local_id in representative_nodes[camera_id]:
            serial = representative_nodes[camera_id][local_id]["representative_node"]["serial"]
            if "GlobalOfflineID" in tracking_dict[serial]:
                global_id = tracking_dict[serial]["GlobalOfflineID"]
                assigned_tracks.append((global_id,camera_id,local_id,serial))
            else:
                unassigned_tracks.append((camera_id,local_id))

    target_list = [(camera_id,serial) for global_id,camera_id,local_id,serial in assigned_tracks]    
    feature_stack = create_mcpt_feature_stack(tracking_results,target_list)
    feature_stack_T = feature_stack.T
    feature_stack_norm = np.linalg.norm(feature_stack, axis=1)
    global_ids = [global_id for global_id,camera_id,local_id,serial in assigned_tracks]

    for k,(camera_id,local_id) in enumerate(unassigned_tracks):
        npy_path = representative_nodes[camera_id][local_id]["representative_node"]["npy_path"]        
        feature = np.load(npy_path)
        cos_sims = np.dot(feature,feature_stack_T)/ (np.linalg.norm(feature)*feature_stack_norm)
        
        if assign_all_tracklet == False:
            max_sim = np.max(cos_sims)
            if max_sim < sim_th:
                continue

        similar_indices = list(np.where(cos_sims >= sim_th)[0])
        if len(similar_indices) == 0:
            continue 
        
        tmp_global_ids = [global_id for i,global_id in enumerate(global_ids) if  i in similar_indices]
        global_id = mode(tmp_global_ids).mode
        
        counter += 1
        serials = representative_nodes[camera_id][local_id]["all_serials"]
        for serial in serials:
            tracking_results[camera_id][serial]["GlobalOfflineID"] = int(global_id)
    print(f"{counter} tracklets are reassigned")
    return tracking_results

def global_id_reassignment(tracking_results, representative_nodes,scene_id,**kwargs):
    # perform delete_small_global_id() and assign_global_id() for reassigning unclustered tracklets
    epsilon = kwargs.get("epsilon_mcpt",0.3)
    representative_selection_method = kwargs.get("representative_selection_method","centrality")
    delete_gid_th = kwargs.get("delete_gid_th",10000)
    assign_all_tracklet = kwargs.get("assign_all_tracklet",True)
    sim_th = kwargs.get("sim_th",0.8)
    delete_few_camera_cluter = kwargs.get('delete_few_camera_cluter',False)
    
    unique_global_ids = get_unique_global_ids(tracking_results,representative_nodes)

    global_serial_dict = get_serials_each_global_id(tracking_results,representative_nodes,unique_global_ids)

    tracking_results, unique_global_ids = delete_small_global_id(tracking_results,representative_nodes,global_serial_dict,
                                                                 delete_gid_th = delete_gid_th,delete_few_camera_cluter=delete_few_camera_cluter)

    tracking_results = assign_global_id(tracking_results,representative_nodes,
                                        delete_gid_th=delete_gid_th, assign_all_tracklet=assign_all_tracklet,sim_th=sim_th)
    
    return tracking_results

def translate_world_coordinate(x, y, homography_matrix):
    # translate camera coordinate to world coordinate
    vector_xyz = np.array([x, y, 1]) # z=1
    vector_xyz_3d = np.dot(np.linalg.inv(homography_matrix), vector_xyz.T)
    return vector_xyz_3d[0] / vector_xyz_3d[2], vector_xyz_3d[1] / vector_xyz_3d[2]


def interpolate_tracklet(tracking_results,representative_nodes,**kwargs):
    # interpolate missing detections for each tracklet
    max_interpolate_interval = kwargs.get("max_interpolate_interval",150)
    frame_sampling_freq = kwargs.get("frame_sampling_freq",1)
    for camera_id in tracking_results:
        tracking_dict = tracking_results[camera_id]
        local_ids = [tracking_dict[serial]["OfflineID"] for serial in tracking_dict]
        unique_local_ids = sorted(list(set(local_ids)))
        if min(unique_local_ids) == -1: unique_local_ids.remove(-1)
        local_id_serial_dict = {local_id:[] for local_id in unique_local_ids} 
        [local_id_serial_dict[tracking_dict[serial]["OfflineID"]].append(serial) for serial in tracking_dict if tracking_dict[serial]["OfflineID"] != -1]
        local_id_frame_dict = {local_id:[] for local_id in unique_local_ids} 
        [local_id_frame_dict[tracking_dict[serial]["OfflineID"]].append(tracking_dict[serial]["Frame"]) for serial in tracking_dict if tracking_dict[serial]["OfflineID"] != -1]

        max_serial = int(max(tracking_dict.keys()))
        for local_id in unique_local_ids:
            frames, serials = zip(*sorted(zip(local_id_frame_dict[local_id], local_id_serial_dict[local_id])))
            missing_frames = []
            for frame,next_frame in zip(frames[:-1],frames[1:]):
                diff = next_frame - frame
                if diff > max_interpolate_interval: continue
                while diff > frame_sampling_freq:
                    diff -= frame_sampling_freq
                    missing_frame = next_frame - diff 
                    missing_frames.append(missing_frame) 
            if missing_frames==0: continue
            global_id = tracking_dict[serials[0]]["GlobalOfflineID"] if "GlobalOfflineID" in tracking_dict[serials[0]] else None
            
            coordinates = [list(tracking_dict[serial]["Coordinate"].values())+list(tracking_dict[serial]["WorldCoordinate"].values()) for serial in serials]
            interpolator = RegularGridInterpolator((np.array(frames),), np.array(coordinates), method='linear') 
            for frame in missing_frames:
                x1,y1,x2,y2,w_x,w_y = interpolator([frame])[0]
                (x1, y1, x2, y2), (w_x,w_y) = map(int, [x1, y1, x2, y2]),map(float,[w_x,w_y])
                max_serial += 1
                if global_id != None:
                    tracking_dict[str(max_serial).zfill(8)] = {"Frame": frame, "Coordinate": {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}, "WorldCoordinate": {'x': w_x, 'y': w_y}, "OfflineID": local_id, "GlobalOfflineID": global_id}
                else:
                    tracking_dict[str(max_serial).zfill(8)] = {"Frame": frame, "Coordinate": {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}, "WorldCoordinate": {'x': w_x, 'y': w_y}, "OfflineID": local_id}
    return tracking_results  


def find_highest_centrality_node(tracking_dict, serials, **kwargs):
    # find highest centrality node from each tracklet
    epsilon = kwargs.get('epsilon_mcpt', 0.3)
    stack_max_size = kwargs.get('stack_max_size', 2000)
    image_size = kwargs.get('image_size', (1920,1080))
    aspect_th = kwargs.get('aspect_th', 1.6)

    pos_list = [list(tracking_dict[serial]["Coordinate"].values()) for serial in serials]
    pos_list = np.array(pos_list)
    aspects = (pos_list[:,3]-pos_list[:,1])/(pos_list[:,2]-pos_list[:,0])
    pos_list[:, 2] = image_size[0] - pos_list[:, 2]
    pos_list[:, 3] = image_size[1] - pos_list[:, 3]
    edge_distances = np.min(pos_list,axis = 1)
    new_serials = []
    for i, (serial,aspect, edge_distance) in enumerate(zip(serials,aspects,edge_distances)):
        if (aspect >= aspect_th): # (edge_distance <= 1) and 
            new_serials.append(serial)
    if len(new_serials) == 0:
        serial,feature = None,None
        pass
    elif len(new_serials) == 1 or len(new_serials)== 2:
        serial = new_serials[0]
        feature = np.load(tracking_dict[serial]["NpyPath"])
    else:
        freq =1
        while len(new_serials)//freq > stack_max_size:
            freq += 1
        for n, serial in enumerate(new_serials):
            if n % freq != 0: continue
            feature = np.load(tracking_dict[serial]["NpyPath"])
            if n== 0: 
                feature_stack = np.empty((0,len(feature.flatten())))
            feature_stack  = np.append(feature_stack , feature.reshape(1, -1), axis=0)
        similarity_matrix = cosine_similarity(feature_stack)
        similarity_matrix = np.where(similarity_matrix < 1-epsilon, 0, similarity_matrix)
        centralities = np.sum(similarity_matrix,axis=0)
        idx_max = np.argmax(centralities)
        serial = new_serials[idx_max*freq]
        feature = feature_stack[idx_max]
    return new_serials, serial, feature 

def minimize_similarity_by_sc_overlap(representative_nodes,matrix,tracking_results,clusters,camera_dict,**kwargs):
    # minimize similarity if tracklets are overlapping in SCPT results
    matrix_type = kwargs.get('matrix_type', "similarity")
    if matrix_type == "similarity":
        replace_value = -1
    elif matrix_type == "distance":
        replace_value = np.max(matrix[matrix<np.inf])

    for camera_id in representative_nodes:
        tracking_dict = tracking_results[int(camera_id)]
        indices = camera_dict[camera_id]["indices"]
        unique_local_ids = camera_dict[camera_id]["unique_local_ids"]
        local_ids_frame_dict = {local_id:[] for local_id in unique_local_ids}

        [local_ids_frame_dict[tracking_dict[serial]["OfflineID"]].append(tracking_dict[serial]["Frame"]) for serial in tracking_dict if tracking_dict[serial]["OfflineID"] in unique_local_ids]
        for index1 in range(len(indices)-1):
            local_id1 = unique_local_ids[index1]
            id1_frames = local_ids_frame_dict[local_id1]
            id1_index = indices[index1]
            for index2 in range(index1+1,len(indices)):   
                local_id2 = unique_local_ids[index2]
                id2_frames = local_ids_frame_dict[local_id2]
                common_frames = set(id1_frames).intersection(set(id2_frames))
                if len(common_frames) == 0: continue
                id2_index = indices[index2]
                matrix[id1_index,id2_index] = replace_value
                matrix[id2_index,id1_index] = replace_value     
    return matrix

def replace_negative_value_by_wcoordinate(similarity_matrix,distance_matrix,**kwargs):
    # replace multiple elements of the similarity matrix with a negative value based on the world coordinate
    distance_th = kwargs.get('distance_th', 7)
    replace_value = kwargs.get('replace_value', -10)
    print("replace_negative_value")
    print("distance_th:",distance_th)
    distance_matrix = np.where(distance_matrix == np.inf, 0, distance_matrix)
    similarity_matrix = np.where(distance_matrix > distance_th, replace_value, similarity_matrix)
    return similarity_matrix

def maximize_similarity_by_wcoordinate(similarity_matrix,distance_matrix,**kwargs):
    # replace multiple elements of the similarity matrix with 1 based on the world coordinate
    max_distance_th = kwargs.get('max_distance_th', 0.5)
    replace_value = kwargs.get('replace_value', 1)
    print("maximize_similarity_by_wcoordinate")
    similarity_matrix = np.where(distance_matrix < max_distance_th, replace_value, similarity_matrix)
    return similarity_matrix

def replace_similarity(representative_nodes,similarity_matrix,tracking_results,clusters,**kwargs):
    # replace multiple elements of the similarity matrix with another value
    distance_th = kwargs.get('distance_th', 10)
    check_sc_overlap = kwargs.get('check_sc_overlap', False)
    replace_similarity_by_wcoordinate = kwargs.get('replace_similarity_by_wcoordinate', False)
    distance_type = kwargs.get('distance_type', "min")
    short_track_th = kwargs.get("short_track_th",0) 
    keypoint_condition_th = kwargs.get("keypoint_condition_th",2)
    replace_value = kwargs.get('replace_value', -10)
    representative_selection_method = kwargs.get('representative_selection_method', 'keypoint')
    
    if check_sc_overlap:
        camera_dict = create_camera_dict(representative_nodes,short_track_th = short_track_th,keypoint_condition_th=keypoint_condition_th,
                                        representative_selection_method=representative_selection_method)
        similarity_matrix = minimize_similarity_by_sc_overlap(representative_nodes,similarity_matrix,tracking_results,clusters,camera_dict, matrix_type = "similarity")
    if replace_similarity_by_wcoordinate:
        min_distance_matrix,max_distance_matrix,mean_distance_matrix = create_distance_matrix(representative_nodes,tracking_results, distance_type = distance_type,short_track_th =short_track_th, keypoint_condition_th = keypoint_condition_th,representative_selection_method =representative_selection_method)
        similarity_matrix = maximize_similarity_by_wcoordinate(similarity_matrix, mean_distance_matrix)
        similarity_matrix = replace_negative_value_by_wcoordinate(similarity_matrix, min_distance_matrix, distance_th=distance_th,replace_value=replace_value)
        
    return similarity_matrix

def measure_euclidean_distance(id1_pos_list,id2_pos_list):
    points1 = np.array(id1_pos_list)
    points2 = np.array(id2_pos_list)
    diff = points1-points2
    euclid_distances = np.sqrt(np.sum(diff**2, axis=1))
    return euclid_distances

def create_distance_matrix(representative_nodes,tracking_results, **kwargs):
    # create a Euclidean distance matrix showing the Euclidean distance between each tracklet
    
    distance_type =  kwargs.get('distance_type', "max") #distance_type  min or max or mean 
    image_size = kwargs.get('image_size', (1920,1080))
    short_track_th = kwargs.get('short_track_th', 0)
    representative_selection_method = kwargs.get("representative_selection_method","keypoint") 
    keypoint_condition_th = kwargs.get("keypoint_condition_th",2)
    print("distance_type:",distance_type)
    camera_dict = create_camera_dict(representative_nodes,short_track_th = short_track_th,
                                   keypoint_condition_th=keypoint_condition_th, representative_selection_method=representative_selection_method)
    shape = np.sum([len(camera_dict[camera_id]["indices"]) for camera_id in camera_dict])
    max_distance_matrix  = np.full((shape, shape), np.inf, dtype=np.float16)
    mean_distance_matrix  = np.full((shape, shape), np.inf, dtype=np.float16)
    min_distance_matrix  = np.full((shape, shape), np.inf, dtype=np.float16)

    index_serials_dict = {index:[] for index in range(len(max_distance_matrix))}
    index_frames_dict = {index:[] for index in range(len(max_distance_matrix))}
    index_wpos_list_dict = {index:[] for index in range(len(max_distance_matrix))}
    
    for camera_id in representative_nodes:
        tracking_dict = tracking_results[int(camera_id)]
        indices = camera_dict[camera_id]["indices"]
        unique_local_ids = camera_dict[camera_id]["unique_local_ids"]
        local_ids_serials_dict = {local_id:[] for local_id in unique_local_ids}
        [local_ids_serials_dict[tracking_dict[serial]["OfflineID"]].append(serial) for serial in tracking_dict if tracking_dict[serial]["OfflineID"] in unique_local_ids]
        
        for tmp_index in range(len(indices)):
            local_id = unique_local_ids[tmp_index]
            serials = local_ids_serials_dict[local_id]
            frames = [tracking_dict[serial]["Frame"] for serial in serials]
            wpos_list = [list(tracking_dict[serial]["WorldCoordinate"].values()) for serial in serials]
            index = indices[tmp_index]
            index_serials_dict[index] += serials
            index_frames_dict[index] += frames
            index_wpos_list_dict[index] += wpos_list

    for id1_index in range(len(max_distance_matrix)-1):
        id1_frames = index_frames_dict[id1_index]
        id1_wpos_list = index_wpos_list_dict[id1_index]
        if id1_frames == []:
            continue

        for id2_index in range(id1_index+1,len(max_distance_matrix)):   
            id2_frames = index_frames_dict[id2_index]
            if id2_frames == []:
                continue
            common_frames = set(id1_frames).intersection(set(id2_frames))
            if len(common_frames) < 1: continue
            id2_wpos_list = index_wpos_list_dict[id2_index]
            id1_lap_indices = [i for i,id1_frame in enumerate(id1_frames) if id1_frame in common_frames]
            id2_lap_indices = [i for i,id2_frame in enumerate(id2_frames) if id2_frame in common_frames]
            id1_lap_wpos_list = [id1_wpos_list[id1_lap_index] for id1_lap_index in id1_lap_indices]
            id2_lap_wpos_list = [id2_wpos_list[id2_lap_index] for id2_lap_index in id2_lap_indices]

            euclid_distances = measure_euclidean_distance(id1_lap_wpos_list,id2_lap_wpos_list)
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

def delete_small_global_id(tracking_results,representative_nodes,global_serial_dict,**kwargs):
    # delete global id that contains only a little serials from tracking_results
    delete_gid_th = kwargs.get('delete_gid_th',10000)
    delete_few_camera_cluter = kwargs.get('delete_few_camera_cluter',False)
    print("delete_gid_th:",delete_gid_th)
    print("delete_few_camera_cluter:",delete_few_camera_cluter)
    delete_global_ids = []
    save_global_ids = []    

    for global_id in global_serial_dict:
        serial_counter = 0
        camera_ids=[]
        for camera_id in global_serial_dict[global_id]:
            if global_serial_dict[global_id][camera_id] != []:
                camera_ids.append(camera_id)
            for local_id,serial in global_serial_dict[global_id][camera_id]:                
                tmp_all_serials = representative_nodes[camera_id][local_id]["all_serials"] 
                serial_counter += len(tmp_all_serials)

        if serial_counter < delete_gid_th:
            delete_global_ids.append(global_id)
            continue
        if delete_few_camera_cluter:
            if len(set(camera_ids)) < 3:
                delete_global_ids.append(global_id)
                continue
        save_global_ids.append(global_id)

    for camera_id in tracking_results:
        tracking_dict = tracking_results[camera_id]
        for serial in tracking_dict:
            tmp_dict = tracking_dict[serial]
            if "GlobalOfflineID" in tmp_dict:
                global_id = tmp_dict["GlobalOfflineID"]
                if global_id in delete_global_ids:
                    del tmp_dict["GlobalOfflineID"]
    unique_global_ids = sorted(list(set(save_global_ids)))

    return tracking_results, unique_global_ids

def measure_world_coordinate(scene_id,tracking_results, **kwargs):
    # measur world coordinates in each node
    mean_world_coordinate_th = kwargs.get("mean_world_coordinate_th",2)
    model = kwargs.get("model","mmpose_hrnet")

    for camera_id in tracking_results:
        tracking_dict = tracking_results[camera_id]       
        with open(f"Original/scene_{str(scene_id).zfill(3)}/camera_{str(camera_id).zfill(4)}/calibration.json") as f:
            calibration_json = json.load(f)
        homography_matrix = np.array(calibration_json['homography matrix'])
        for serial in tracking_dict:
            value = tracking_dict[serial]
            x1,y1,x2,y2 = value["Coordinate"].values()
            x,y =  (x2+x1)/2,y2
            bbox_w_c = translate_world_coordinate(x,y, homography_matrix)
            value["WoorldCoordinate"] = {"x":bbox_w_c[0],"y":bbox_w_c[1]}

    for camera_id in tracking_results:
        tracking_dict = tracking_results[camera_id]
        for serial in tracking_dict:
            value = tracking_dict[serial]
    return tracking_results

def eval_noise_level(keypoints):
    # evaluate noise level in images based on pose estimation
    xs,ys,scores = zip(*keypoints)
    th = 0.75
    indices = [i for i,score in enumerate(scores) if score > th]
    condition = 0
    if len(indices)==2:
        if min(indices) <= 4:
            condition = 0 
        else:
            condition = 2
    if len(indices)==1:
        condition = 3
    if len(indices)==0:
        condition =4
    return condition

def remove_noise_images(scene_id,tracking_results,**kwargs):
    # remove noise images based on pose estimation
    model = kwargs.get("model","mmpose_hrnet")

    del_serials = {camera_id:[] for camera_id in tracking_results}

    for camera_id in tracking_results:
        tracking_dict = tracking_results[camera_id]
        for serial in tracking_dict:
            value = tracking_dict[serial]
            if "GlobalOfflineID" not in value:
                del_serials[camera_id].append(serial)
            
    for camera_id in tracking_results:
        tracking_dict = tracking_results[camera_id]
        for serial in del_serials[camera_id]:
            del tracking_dict[serial]

    for camera_id in tracking_results:
        tracking_dict = tracking_results[camera_id]
        keypoints_results = pose.PoseKeypoints(f"Pose/scene_{str(scene_id).zfill(3)}/camera_{str(camera_id).zfill(4)}/camera_{str(camera_id).zfill(4)}_out_keypoint.json")
        keypoints_results.assign_serial_from_tracking_dict(tracking_dict=tracking_dict)
        del_serials = []
        for serial in tracking_dict:
            value = tracking_dict[serial]
            
            kp = keypoints_results.get_keypoints(serial)
            if kp == None:
                del_serials.append(serial)
                continue
            keypoints = kp['Keypoints']
            condition = eval_noise_level(keypoints)
            coordinate = list(value["Coordinate"].values())
            w,h = coordinate[2]-coordinate[0],coordinate[3]-coordinate[1]
            if  w/h > 3 or  h/w > 5:
                del_serials.append(serial)
                continue
            if condition >= 2:
                if condition==2 and min(w,h) < 100:
                    continue
                del_serials.append(serial)
        for serial in del_serials:
            del tracking_dict[serial]

        local_ids = [tracking_dict[serial]["OfflineID"] for serial in tracking_dict]
        unique_local_ids = sorted(set(local_ids))
        if -1 in unique_local_ids:
            unique_local_ids.remove(-1)
        local_id_serials_dict = {local_id:[] for local_id in unique_local_ids}
        [local_id_serials_dict[local_id].append(serial) for local_id,serial in zip(local_ids,tracking_dict)]
        local_id_frames_dict = {local_id:[] for local_id in unique_local_ids}
        [local_id_frames_dict[local_id].append(tracking_dict[serial]["Frame"]) for local_id,serial in zip(local_ids,tracking_dict)]

        del_serials = []
        for local_id in local_id_serials_dict:
            if local_id == -1:
                continue
            frames, serials = zip(*sorted(zip(local_id_frames_dict[local_id], local_id_serials_dict[local_id])))
            for i in range(len(frames[:-1])):
                if i == 0:
                    continue
                past_frame = frames[i-1]
                frame = frames[i]
                future_frame = frames[i+1]
                if (frame - past_frame >30) and (future_frame - frame > 30):
                    del_serials.append(serials[i])

    return tracking_results


def delete_distant_persons(tracking_results,**kwargs):
    # delete the node that has long distances to other nodes with the same global id
 
    gid_serials = {}

    for camera_id in tracking_results:
        tracking_dict = tracking_results[camera_id]
        for serial in tracking_dict:
            value = tracking_dict[serial]
            gid = value["GlobalOfflineID"]
            gid_serials[gid] = []
    for camera_id in tracking_results:
        tracking_dict = tracking_results[camera_id]
        for serial in tracking_dict:
            value = tracking_dict[serial]
            gid = value["GlobalOfflineID"]
            frame = value["Frame"]
            gid_serials[gid].append((camera_id,serial,frame))
    delete_list= []
    for gid in gid_serials:
        value = gid_serials[gid]
        camera_ids,serials,frames = zip(*value)
        frames, serials,camera_ids = zip(*sorted(zip(frames, serials, camera_ids)))
        
        current_frame = frames[0]
        current_serial = serials[0]
        current_camera_ids = camera_ids[0]
        tmp_frames = []
        tmp_serials = []
        tmp_camera_ids = []
        for frame,serial,camera_id in zip(frames,serials,camera_ids):
            if frame !=current_frame:
                
                if len(tmp_frames) >=2:
                    world_coordinates = [] 
                    for tmp_camera_id,tmp_serial in zip(tmp_camera_ids,tmp_serials):
                        world_coordinate = tuple(tracking_results[tmp_camera_id][tmp_serial]["WorldCoordinate"].values())
                        world_coordinates.append(world_coordinate)
                    world_coordinates = np.array(world_coordinates)                   
                    distance_matrix = squareform(pdist(world_coordinates, 'euclidean'))
                    if len(distance_matrix)>2:
                        if np.max(distance_matrix) >7:
                            sum_row = np.sum(distance_matrix,axis=0)
                            argmax = np.argmax(sum_row)
                            delete_list.append((tmp_camera_ids[argmax],tmp_serials[argmax]))
                                                        
                current_frame = frame
                current_serial = serial
                current_camera_id = camera_id
                tmp_frames = [frame]
                tmp_serials = [serial]
                tmp_camera_ids = [camera_id]
            else:
                tmp_frames.append(frame)
                tmp_serials.append(serial)
                tmp_camera_ids.append(camera_id)

    for camera_id,serial in delete_list:
        del tracking_results[camera_id][serial]
    return tracking_results
