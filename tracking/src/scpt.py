import os
import numpy as np
import sys
import itertools

from sklearn.cluster import DBSCAN 
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations, permutations, product, chain
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.interpolate import RegularGridInterpolator
from collections import Counter, defaultdict, deque

from utils import TrackingFunctions, ParameterManager, ClusteringFunctions, ArrayCalculator
from parameters import CommonParameters, SCPTParameters, SCPTPostprocessParameters

class OfflineSingleCameraTracker(TrackingFunctions,ParameterManager,CommonParameters,SCPTParameters):
    def __init__(self, tracking_dict,identifiability_results=None, common_params=None,tracking_params=None):
        TrackingFunctions.__init__(self)
        ParameterManager.__init__(self)
        CommonParameters.__init__(self)
        SCPTParameters.__init__(self)
        if common_params:
            self.update_params(common_params)
        if tracking_params:
            self.update_params(tracking_params)

        self.tracking_dict = tracking_dict
        if self.is_exclude_low_identifiable_img or self.is_minimize_noisy_img_similarity:
            self.identifiability_results = identifiability_results
        else:
            del identifiability_results 

    def run_tracking(self):
        if self.is_print_parameters:
            print("=== start scpt ===")
            print("**scpt main parameters**")
            print("epsilon:",self.epsilon)
            print("frame_period:",self.frame_period)
            print("is_optimize_similarity_by_iou:",self.is_optimize_similarity_by_iou)
            print("is_run_overlap_suppression:",self.is_run_overlap_suppression,"\n")
        
        last_frame = self.get_max_value_of_dict(self.tracking_dict, "Frame")
        time_section_serial_dict = {timesection:[] for timesection in range(last_frame//self.frame_period+1)}
        for serial,value in self.tracking_dict.items():
            time_section = value["Frame"] // self.frame_period
            if self.is_exclude_low_identifiable_img and not self.check_is_use(serial):
                continue            
            time_section_serial_dict[time_section].append(serial) 
        
        max_localid = -1
        for time_section,serials in time_section_serial_dict.items(): 
            if len(serials) == 0: continue
            clusters = self.tracking_by_clustering(self.tracking_dict,serials)            
            clusters = [cluster+max_localid+1 if cluster != -1 else -i for i,cluster in enumerate(clusters)]

            max_localid = max(max_localid, max(clusters))

            if time_section > 0:
                past_serials = time_section_serial_dict[time_section-1]
                clusters = self.associate_cluster_between_period(clusters, serials, past_serials) if len(past_serials) > 0 else clusters

            for serial,cluster in zip(serials,clusters):
                self.tracking_dict[serial]["LocalID"] = int(cluster)

        # # We have tracking results in TrackingDict, yet will gather results for debugging. Could be deleted.
        self.tracking_dict = self.reassign_numerical_value(self.tracking_dict,"LocalID")
        return self.tracking_dict

    def tracking_by_clustering(self,tracking_dict,serials):
        if len(serials) ==1:
            clusters = [0]    
        else:
            setattr(self, "similarity_matrix", self.create_similarity_matrix(serials))
            self.similarity_matrix = np.where(self.similarity_matrix < (1-self.epsilon), 0, self.similarity_matrix)
            
            setattr(self, "distance_matrix",  1 - self.similarity_matrix)
            if self.clustering_method == "agglomerative":
                clusters = ClusteringFunctions.agglomerative_clustering(self.distance_matrix,self.epsilon) #min(clusters)=1
            elif self.clustering_method == "dbscan":
                clusters = ClusteringFunctions.dbscan(self.distance_matrix,min_samples=self.min_samples,epsion=self.epsilon)

            if self.is_run_overlap_suppression  == True:
                clusters = self.reclustering_overlap_cluster(serials,clusters)
        clusters = self.reassign_sequential_ids(clusters)

        return clusters 

    def create_similarity_matrix(self, serials):
        # create a similarity matrix from features
        similarity_matrix= self.similarity_matrix_by_npypaths([self.tracking_dict[serial]["NpyPath"] for serial in serials])
        similarity_matrix = self.optimize_similarity_matrix(similarity_matrix,serials) 
        return similarity_matrix

    def optimize_similarity_matrix(self,similarity_matrix,serials):
        if self.is_minimize_noisy_img_similarity:
            similarity_matrix = self.minimize_noisy_img_similarity(similarity_matrix,serials) #noizyな画像間の類似度を下げる
        if self.is_use_motion:
            frames, bboxes = zip(*[(value["Frame"],tuple(value["Coordinate"].values())) for i,serial in enumerate(serials) if (value := self.tracking_dict[serial])])
            motion_feature_processor = MotionFeatureProcessor(frames,bboxes,iou_th=self.iou_th, ratio_test_th=self.ratio_test_th, len_past_frame=self.len_past_frame)
            del frames,bboxes

            if self.is_set_speed_limit:            
                distances, elapsed_frames = motion_feature_processor.get_motion_information() 
                similarity_matrix =  self.considering_speed_limit(similarity_matrix,serials,distances, elapsed_frames) #Replace bbox pairs with a large amount of movement per unit time with 0.

        return similarity_matrix

    def fill_number_to_none(self,lst):
        # fill the missing value in sequential number list
        used_nums = [num for num in lst if num is not None]
        unused_nums = [num for num in range(len(lst)) if num not in used_nums]
        for i in range(len(lst)):
            if lst[i] is None:
                lst[i] = unused_nums.pop(0)
        return lst

    def create_centrality_matrix(self,clusters, frames):
        # translate the similarity matrix between each node into the centrality matrix between each cluster
        unique_clusters = sorted(list(set(clusters)))
        if -1 in unique_clusters:
            unique_clusters.remove(-1)
        len_unique_clusters = len(unique_clusters)

        centrality_matrix =  np.ones((len_unique_clusters ,len_unique_clusters))*-1 
        np.fill_diagonal(centrality_matrix, 0)

        cluster_frames_dict = self.make_dict(clusters,frames) 
        cluster_set_frames_dict = {cluster:set(cluster_frames_dict[cluster]) for cluster in unique_clusters}
        cluster_indices_dict = self.make_dict(clusters,range(len(clusters)))

        for i, cluster1 in enumerate(unique_clusters):
            cluster1_set_frames = cluster_set_frames_dict[cluster1]
            cluster1_indices = cluster_indices_dict[cluster1] #indices of similarity_matrix
            for j in range(i + 1, len(unique_clusters)):
                cluster2 = unique_clusters[j]
                cluster2_set_frames = cluster_set_frames_dict[cluster2]
                if cluster1_set_frames.intersection(cluster2_set_frames): continue
                cluster2_indices = cluster_indices_dict[cluster2]
                similarities = self.similarity_matrix[np.ix_(cluster1_indices, cluster2_indices)]
                centrality =  np.sum(similarities[similarities > (1 - self.epsilon)])
                centrality_matrix[i,j] = centrality
                centrality_matrix[j,i] = centrality
        return centrality_matrix

    def get_neighbor_indices(self,similarity_matrix):
        """
        """
        each_max_similarities  = np.max(similarity_matrix,axis=0)
        neighbor_indices = np.where(each_max_similarities > (1-self.epsilon))[0]
        
        sorted_indices = np.argsort(each_max_similarities[neighbor_indices])[::-1]
        neighbor_indices = neighbor_indices[sorted_indices]
        if self.num_candidates > 1 and len(neighbor_indices) > self.num_candidates:
            neighbor_indices = neighbor_indices[:self.num_candidates]
        neighbor_indices = neighbor_indices.tolist()
        return neighbor_indices

    def unique_nested_list(self, nested_list):
        unique_list = [list(x) for x in set(tuple(l) for l in nested_list)]
        return unique_list

    def get_candidates_indices_list(self,subcluster_indices_list,overlap_indices_list,):
        # get candidates of the assignment problem
        method = "high_identifiable"
        if method == "high_identifiable":
            self.num_candidates = 0

        if self.num_candidates > 0 and len(overlap_indices_list) < self.num_candidates:
            candidates_indices_list = overlap_indices_list 
        else:
            flatten_subcluster_indices = list(chain.from_iterable(subcluster_indices_list))
            extracted_similarity_matrix = self.similarity_matrix[flatten_subcluster_indices]
            
            neighbor_indices = self.get_neighbor_indices(extracted_similarity_matrix)
            candidates_indices_list = []
            for neighbor_index, overlap_indices in product(neighbor_indices, overlap_indices_list):
                if neighbor_index not in overlap_indices: continue
                candidates_indices_list.append(overlap_indices)                    
                for overlap_index in overlap_indices:
                    try: 
                        neighbor_indices.remove(overlap_index)                        
                    except: 
                        pass
        candidates_indices_list = self.unique_nested_list(candidates_indices_list)

        if len(candidates_indices_list) >2 and method == "high_identifiable":
            tmp_index = self.get_dissimilar_index(candidates_indices_list)
            candidates_indices_list = [candidates_indices_list[tmp_index]]            
 
        return candidates_indices_list

    def get_dissimilar_index(self,overlap_indices_list): 
        #determines the initial index for the assignment problem.
        distances = [min(self.distance_matrix[i, j] for i, j in combinations(overlap_indices, 2)) for overlap_indices in overlap_indices_list]
        return np.argmax(distances)

    def bipartite_matching(self,new_key,centrality_dict,centrality_matrix,overlap_indices):
        # bipartite matching between unclustered overlap nodes and clustered overlap nodes 

        sum_centrality = 0
        subcluster_indices = [None]*len(overlap_indices)
        th = 1-self.epsilon
        while np.max(centrality_matrix) > th:
            max_index = np.argmax(centrality_matrix)
            row_index, col_index = np.unravel_index(max_index, centrality_matrix.shape)
            centrality = centrality_matrix[row_index, col_index]
            sum_centrality += centrality
            subcluster_indices[row_index] = col_index
            centrality_matrix[row_index,:]=0
            centrality_matrix[:,col_index]=0
        centrality_dict[new_key] = {"overlap_indices":overlap_indices,"indices":subcluster_indices,"centrality":sum_centrality} 
        
        return centrality_dict

    def separate_into_subcluster(self,clusters, overlap_indices_list, distance_matrix):
        # overlap nodes are separated into subclusters
        max_overlap = max([len(i) for i in overlap_indices_list]) #the number of overlap in the same frame
        dissimilar_index = self.get_dissimilar_index(overlap_indices_list) #index of overlap_indices_list
        dissimilar_node_indices = overlap_indices_list.pop(dissimilar_index) 
        
        subcluster_indices_list = [[] for _ in range(max_overlap)]
        for i, dissimilar_node_index in enumerate(dissimilar_node_indices):
            subcluster_indices_list[i].append(dissimilar_node_index)
        
        # separte overlap nodes into several groups 
        while len(overlap_indices_list) != 0:    
            centrality_dict = {}
            max_centrality = 0
            candidates_indices_list = self.get_candidates_indices_list(subcluster_indices_list,overlap_indices_list)

            for i,overlap_indices in enumerate(candidates_indices_list): 
                centrality_matrix = np.zeros((len(overlap_indices),len(subcluster_indices_list))) #can not use create_centrality_matrix
                for j, overlap_index in enumerate(overlap_indices):
                    tmp_similarity_matrix = self.similarity_matrix[overlap_index]
                    for k, subcluster_indices in enumerate(subcluster_indices_list):
                        similarities = tmp_similarity_matrix[subcluster_indices]
                        centrality =  np.sum(similarities[similarities > (1 - self.epsilon)]) 
                        centrality_matrix[j,k] = centrality
                centrality_dict = self.bipartite_matching(i,centrality_dict,centrality_matrix,overlap_indices)
            
            max_centrality = 0 if centrality_dict == {} else np.max(map(lambda value: value["centrality"], centrality_dict.values()))

            if max_centrality == 0:
                max_index = self.get_dissimilar_index(overlap_indices_list)
                max_subcluster_indices = list(range(max_overlap))
                overlap_indices = overlap_indices_list[max_index]
            else:
                max_index = max(centrality_dict, key=lambda k: centrality_dict[k]["centrality"])
                max_subcluster_indices = list(centrality_dict[max_index]["indices"]) 
                if None in max_subcluster_indices:
                    max_subcluster_indices = self.fill_number_to_none(max_subcluster_indices)
                overlap_indices = centrality_dict[max_index]["overlap_indices"]

            for max_subcluster_index,overlap_index in zip(max_subcluster_indices,overlap_indices):
                subcluster_indices_list[max_subcluster_index].append(overlap_index)
            overlap_indices_list.remove(overlap_indices)
        
        # assign cluster ID
        for subcluster_indices in subcluster_indices_list:
            if len(subcluster_indices) == 1: 
                clusters[subcluster_indices[0]] = np.max(clusters)+1 # assign unique clusterID
            else:
                sub_clusters = ClusteringFunctions.agglomerative_clustering(distance_matrix[np.ix_(subcluster_indices, subcluster_indices)],epsilon=self.epsilon)
                sub_clusters = [sub_cluster+max(clusters) for sub_cluster in sub_clusters]            
                for sub_cluster,subcluster_index in zip(sub_clusters,subcluster_indices):
                    clusters[subcluster_index] = sub_cluster
        return clusters

    def overlap_suppression_clustering(self,frames,nonoverlap_indices,overlap_indices_list):
        clusters = [-1]*len(frames) 
        
        # clustering for non-overlapping nodes
        if nonoverlap_indices:            
            nonoverlap_clusters = ClusteringFunctions.agglomerative_clustering(self.distance_matrix[np.ix_(nonoverlap_indices,nonoverlap_indices)], epsilon=self.epsilon) if len(nonoverlap_indices) > 1 else [0]
            for target_index, cluster in zip(nonoverlap_indices, nonoverlap_clusters):
                clusters[target_index] = cluster
        
        # clustering for overlapping nodes         
        clusters = self.separate_into_subcluster(clusters, overlap_indices_list, self.distance_matrix)
        centrality_matrix = self.create_centrality_matrix(clusters,frames)

        # merging for subcluster
        clusters = self.associate_cluster(clusters,centrality_matrix, epsilon=self.epsilon, cost_function=1)
        return clusters

    def divide_overlap_or_nonoverlap(self,cluster_frames,cluster_indices):
        frame_indices_dict = self.make_dict(cluster_frames,cluster_indices)
        overlap_indices_list = [indices for indices in frame_indices_dict.values() if len(indices) > 1] 
        flattened_overlap_indices = list(chain.from_iterable(overlap_indices_list))
        nonoverlap_indices = [index for index in cluster_indices if index not in set(flattened_overlap_indices)]
        return overlap_indices_list, nonoverlap_indices

    def reclustering_overlap_cluster(self,serials,clusters): 
        frames = [self.tracking_dict[serial]["Frame"] for serial in serials]
        cluster_frames_dict = self.make_dict(clusters,frames)
        cluster_indices_dict = self.make_dict(clusters,range(len(clusters)))       
        for cluster,cluster_frames in cluster_frames_dict.items(): 
            if len(list(set(cluster_frames))) == len(cluster_frames):continue #This cluster doesn't have overlap
            cluster_indices = cluster_indices_dict[cluster]
            #divide overlap/nonoverlap
            overlap_indices_list, nonovelap_indices = self.divide_overlap_or_nonoverlap(cluster_frames,cluster_indices)
            
            tmp_clusters = self.overlap_suppression_clustering(frames,nonovelap_indices,overlap_indices_list)  

            max_cluster_id = np.max(clusters) 
            for index,tmp_cluster in enumerate(tmp_clusters):
                if clusters[index] != cluster: continue 
                clusters[index] = max_cluster_id + tmp_cluster + 1
        return clusters

    def considering_speed_limit(self,similarity_matrix,serials, distances, elapsed_frames,min_speed_limit=50):
        """
        単位時間当たりのbboxの移動速度がmin_speed_limit速い場合、類似度を0に置換する
        """
        replace_value = 0
        sigmoid_coef = 0.1
        exp_value = 2.8

        with np.errstate(divide='ignore', invalid='ignore'):
            speeds = np.true_divide(distances, elapsed_frames)
            speeds[np.isinf(speeds)] = 0 
        
        speed_coefficient = 1/(ArrayCalculator.sigmoid(sigmoid_coef*elapsed_frames)**exp_value)
        similarity_matrix = np.where(speeds>(min_speed_limit*speed_coefficient),replace_value, similarity_matrix)
        return similarity_matrix

    def noise_checker(self,result):
        if result["intersect_ratio"] < 0.2:
            if result["score"] < 0.6 or result["condition"] > 4:
                return True
        return False

    def minimize_noisy_img_similarity(self,similarity_matrix,serials):
        """
        Poseのconfidenceが低いもの同士の類似度を0に置換する
        """
        lst = []
        for i,j in itertools.combinations(range(len(serials)),2):
            serial1, serial2 = serials[i], serials[j]
            result1, result2 = self.identifiability_results[serial1], self.identifiability_results[serial2]
            is_noise1, is_noise2 = self.noise_checker(result1), self.noise_checker(result2)
            value = 0 if is_noise1 and is_noise2 else 1
            lst.append(value)
        mask = squareform(np.array(lst))
        np.fill_diagonal(mask, 1)
        similarity_matrix *=mask
        return similarity_matrix
    
    def check_is_use(self,serial):
        """
        対象serialのデータをTrackingに使用するか判定する
        """
        result = self.identifiability_results[serial]
        is_noise = self.noise_checker(result)
        is_use = False if is_noise else True
        return is_use

    def reassign_numerical_value(self,tracking_dict, key):
        """
        辞書のvalueを連番に振り直す
        """
        old_values = [value[key] for serial,value in tracking_dict.items()]
        new_values_dict = {key:i for i,key in enumerate(set(old_values)) if key != -1}
        new_values_dict[-1] = -1
        
        for serial,value in tracking_dict.items():
            old_value = value[key]
            tracking_dict[serial][key] = new_values_dict[old_value]
        return tracking_dict

    def associate_cluster_between_period(self,clusters,serials,past_serials,**params):
        # associate clusters between adjacent time periods
        frames = [self.tracking_dict[serial]["Frame"] for serial in serials]        
        past_frames,local_ids = zip(*[(tracking_info["Frame"], tracking_info["LocalID"]) for serial in past_serials if (tracking_info := self.tracking_dict.get(serial))])

        all_serials = past_serials + serials
        all_clusters = list(local_ids) + clusters
        all_frames = list(past_frames)  + frames
        
        self.delete_variable("distance_matrix")
        setattr(self, "similarity_matrix", self.create_similarity_matrix(all_serials))
       
        centrality_matrix = self.create_centrality_matrix(all_clusters, all_frames) #similarity_matrix,
        self.delete_variable("similarity_matrix")
        np.fill_diagonal(centrality_matrix, 0)
        
        all_clusters = self.associate_cluster(all_clusters,centrality_matrix,self.epsilon)      
        new_clusters = all_clusters[len(local_ids):]

        return new_clusters
        
    ######################################################################################################
    ######################################################################################################

class MotionFeatureProcessor(TrackingFunctions):
    def __init__(self,frames:list,bboxes:tuple,iou_th=0.9,ratio_test_th=0.6,len_past_frame=4):
        TrackingFunctions.__init__(self)
        self.frames = np.array(frames)
        self.bboxes = np.array(bboxes)
        self.iou_th = iou_th
        self.ratio_test_th = ratio_test_th
        self.len_past_frame = len_past_frame
  
    def get_motion_information(self):
        """
        対象serialの移動距離とbboxの出現時刻の差分を調べる
        """
        frames = self.frames
        bboxes = self.bboxes
        center_bottoms = np.column_stack(((bboxes[:, 0] + bboxes[:, 2]) / 2, bboxes[:, 3]))

        distances = np.sqrt(((center_bottoms[:, np.newaxis, :] - center_bottoms[np.newaxis, :, :])**2).sum(axis=2))
        elapsed_frames = np.abs(frames[:, np.newaxis] - frames)
        return distances.astype(np.float16), elapsed_frames.astype(np.int16)

class SCPTPostProcessor(TrackingFunctions,ParameterManager,CommonParameters,SCPTPostprocessParameters):
    def __init__(self, tracking_dict, params=None):
        TrackingFunctions.__init__(self)
        ParameterManager.__init__(self)
        CommonParameters.__init__(self)
        SCPTPostprocessParameters.__init__(self)

        if params:
            self.update_params(params)

        self.tracking_dict = tracking_dict

    def run_scpt_postprocess(self,):
        if self.is_print_parameters:
            print("=== start scpt_postprocess ===")
            print("**post process contents**")
            print("concat_high_iou_tracklet:",self.is_concat_high_iou_tracklet)
            print("sequential_nms:",self.is_run_sequential_nms)
            print("separate_warp:",self.is_run_separate_warp)
            print("exclude_short_track:",self.is_run_exclude_short_track)
            print("exclude_motionless_track:",self.is_run_exclude_motionless_track,"\n")    

        if self.is_concat_high_iou_tracklet:
            self.concat_high_iou_tracklet()
        if self.is_run_sequential_nms:
            self.sequential_non_maximum_suppression() 
        if self.is_run_separate_warp:
            self.separate_warp_tracklet()
        if self.is_run_exclude_short_track:
            self.exclude_short_tracklet()
        if self.is_run_exclude_motionless_track:
            self.exclude_motionless_tracklet()

        return self.tracking_dict

    def concat_high_iou_tracklet(self):
        len_past_frame = 10
        iou_th = 0.80
        self.assign_sequential_negativeids()

        #pre:欠損が生じる直前
        #post: 欠損が生じた直後
        local_ids,frames = zip(*[(value["LocalID"], value["Frame"]) for serial,value in self.tracking_dict.items()])
        localid_frames_dict = self.make_dict(local_ids,frames)
        localid_serials_dict = self.make_dict(local_ids,self.tracking_dict.keys())

        pre_serials,post_serials =  self.get_pre_and_post_serials()
        
        pre_frames, pre_localids = zip(*[(value["Frame"], value["LocalID"]) for serial in pre_serials if (value := self.tracking_dict[serial])])
        post_frames, post_localids = zip(*[(value["Frame"], value["LocalID"]) for serial in post_serials if (value := self.tracking_dict[serial])])
        max_frame = max(max(pre_frames),max(post_frames))
        min_frame = min(min(pre_frames),min(post_frames))
        preframe_serials_dict = self.make_dict(pre_frames,pre_serials) 
        postframe_serials_dict = self.make_dict(post_frames,post_serials)

        past_coordinates_list = deque(maxlen=len_past_frame)
        past_serials_list = deque(maxlen=len_past_frame)
        for frame in range(min_frame, max_frame):
            tmp_post_serials = postframe_serials_dict.get(frame)
            if tmp_post_serials is None:
                continue

            post_coordinates = np.array([(tuple(self.tracking_dict[serial]["Coordinate"].values())) for serial in tmp_post_serials])
            ious_list = self._compute_iou_with_past_frames(post_coordinates,past_coordinates_list)
           
            for i in range(len(ious_list)):
                ious = ious_list[-(i+1)]
                past_serials = past_serials_list[-(i+1)]
                max_value = np.max(ious)
                
                while max_value > iou_th:
                    max_index = np.unravel_index(np.argmax(ious), ious.shape)
                    max_value = ious[max_index]
                    ious[max_index] = 0
                    post_index,past_index = max_index 

                    post_serial = tmp_post_serials[post_index] 
                    past_serial = past_serials[past_index]
                    
                    post_localid = self.tracking_dict[post_serial]["LocalID"]
                    past_localid = self.tracking_dict[past_serial]["LocalID"]

                    tmp_post_frames = localid_frames_dict.get(post_localid)
                    tmp_past_frames = localid_frames_dict.get(past_localid)
                    if tmp_post_frames is None or tmp_past_frames is None:
                        continue                    
                    if set(tmp_post_frames).intersection(set(tmp_past_frames)):
                        continue
                    
                    old_id = min(post_localid,past_localid)
                    new_id = max(post_localid,past_localid)

                    for serial in localid_serials_dict[old_id]:
                        self.tracking_dict[serial]["LocalID"] = new_id 

                    localid_frames_dict[new_id] = tmp_post_frames + tmp_past_frames
                    localid_serials_dict[new_id] = localid_serials_dict[post_localid] + localid_serials_dict[past_localid]
                    del localid_frames_dict[old_id]
                    del localid_serials_dict[old_id]       

            tmp_pre_serials = preframe_serials_dict.get(frame)
            if tmp_pre_serials is None:
                past_serials_list.append([])
                past_coordinates_list.append(np.array([]))
            else:
                past_serials_list.append(tmp_pre_serials)
                past_coordinates_list.append(np.array([(tuple(self.tracking_dict[serial]["Coordinate"].values())) for serial in tmp_pre_serials]))
            
        self.assign_negativeids2default()
        return

    def _compute_iou_with_past_frames(self,pre_coordinates,past_coordinates_list):
        ious_list=[]
        for past_coordinates in past_coordinates_list:
            if len(past_coordinates) == 0:
                ious = np.array([0])
            else:
                ious = ArrayCalculator.compute_iou(pre_coordinates,past_coordinates)
            ious_list.append(ious)
        return ious_list

    def get_pre_and_post_serials(self):
        serials,local_ids,frames = zip(*[(serial,value["LocalID"],value["Frame"]) for serial,value in self.tracking_dict.items()])
        localid_dict = self.make_dict(local_ids,[(serial,frame) for (serial,frame) in zip(serials,frames)])

        pre_serials=[]
        post_serials=[]
        for local_id,(value) in localid_dict.items():
            tmp_serials,tmp_frames = zip(*value)
            pre_missing_serials, post_missing_serials = self.get_serials_before_after_missing(tmp_frames,tmp_serials)
            pre_serials.extend(pre_missing_serials)
            post_serials.extend(post_missing_serials)

        return pre_serials,post_serials

    def get_serials_before_after_missing(self,frames,serials):
        """
        同一IDが割り振られたシリアルとそれに対応するフレームのリストからフレームに欠損が生じる直前、直後のserialを取得する
        """
        frames,serials = map(list, zip(*sorted(zip(frames, serials))))

        post_missing_serials = self.get_post_missing_serials(frames,serials)
        pre_missing_serials = self.get_post_missing_serials(frames[::-1],serials[::-1])
        return pre_missing_serials, post_missing_serials

    def get_post_missing_serials(self,frames, serials):
        """
        フレームに欠損が生じた直後のserialを取得する
        ※invertを入力すれば、欠損直前を取得する
        """
        frame_interval = 1
        post_missing_serials=[serials[0]]
        if len(serials) == 0:
            return post_missing_serials
        
        array_frames = np.array(frames)
        differences = np.abs(array_frames[1:]-array_frames[:-1])
        for diff, frame, serial in zip(differences,frames[1:],serials[1:]):
            if diff > frame_interval:
                post_missing_serials.append(serial)

        return post_missing_serials

    def assign_sequential_negativeids(self):
        """
        LocalIDが-1になっているものを負の連番に振り直す
        """
        new_id = -2
        for serial, value in self.tracking_dict.items():
            if value["LocalID"] == -1:
                value["LocalID"] = new_id
                new_id -=1

    def assign_negativeids2default(self):
        """
        負の連番になっているLocalIDを―1に振り直す
        """
        for serial, value in self.tracking_dict.items():
            if value["LocalID"] < 0:
                value["LocalID"] = -1

    def exclude_motionless_tracklet(self, **params):
        # exclude tracklet from tracking_dict
        serials,local_ids = zip(*[(serial,value["LocalID"]) for serial,value in self.tracking_dict.items() if value["LocalID"] != -1])
        local_id_serials_dict = self.make_dict(local_ids, serials)

        for local_id,serials in local_id_serials_dict.items():

            bboxes = [tuple(self.tracking_dict[serial]["Coordinate"].values()) for serial in serials]
            bboxes = np.array(bboxes)
            center_bottoms = np.column_stack(((bboxes[:, 0] + bboxes[:, 2]) / 2, bboxes[:, 3]))
            x_max, y_max = np.max(center_bottoms,axis=0)
            x_min, y_min = np.min(center_bottoms,axis=0)

            if (x_max-x_min < self.stop_track_th) and (y_max-y_min < self.stop_track_th):
                for serial in serials:
                    self.tracking_dict[serial]["LocalID"] = -1

    def exclude_short_tracklet(self, **params):
        # exclude tracklet that contains only a little serials from tracking_dict
        if self.print_parameters:
            print("**exclude short tracklet parameters**")
            print(f"short_tracklet_th:{self.short_tracklet_th}")

        serials,local_ids = zip(*[(serial,value["LocalID"]) for serial,value in self.tracking_dict.items() if value["LocalID"] != -1])
        
        localid_serials_dict = self.make_dict(local_ids, serials)
        for local_id,serials in localid_serials_dict.items():
            if len(serials) > self.short_tracklet_th:
                continue
            for serial in serials:
                self.tracking_dict[serial]["LocalID"] = -1

    def sequential_non_maximum_suppression(self, **params):
        #Sequential NMS is perfomed in this function.
        #Sequential NMS calculates the overlap coefficient both temporally and spatially. 

        if self.is_print_parameters:
            print("**snms parameters**")
            print(f"temporally_snms_th:{self.temporally_snms_th}")
            print(f"spatially_snms_th:{self.temporally_snms_th}")
            print(f"merge_nonoverlap:{self.merge_nonoverlap}")
        
        local_ids,frames,serials = zip(*[(value["LocalID"],value["Frame"],serial) for serial,value in self.tracking_dict.items() if value["LocalID"] != -1])
    
        local_id_serials_dict = self.make_dict(local_ids, serials)
        local_id_frames_dict = self.make_dict(local_ids, frames)
        
        for local_id1, local_id2 in combinations(list(local_id_serials_dict.keys()),2):
            id1_frames = local_id_frames_dict[local_id1]
            id2_frames = local_id_frames_dict[local_id2]
            overlap_frames = set(id1_frames).intersection(set(id2_frames))

            if len(id1_frames) < len(id2_frames):
                (local_id1,local_id2) = (local_id2,local_id1)
                (id1_frames,id2_frames) = (id2_frames,id1_frames)

            if max(len(overlap_frames)/len(id1_frames),len(overlap_frames)/len(id2_frames)) < self.temporally_snms_th: 
                continue 
            
            id1_serials = local_id_serials_dict[local_id1]
            id2_serials = local_id_serials_dict[local_id2]
            
            id1_frames,id1_serials = zip(*sorted(zip(id1_frames,id1_serials)))
            id2_frames,id2_serials = zip(*sorted(zip(id2_frames,id2_serials)))

            id1_lap_bboxes = [list(self.tracking_dict[serial]["Coordinate"].values()) for (frame,serial) in zip(id1_frames,id1_serials) if frame in overlap_frames] 
            id2_lap_bboxes = [list(self.tracking_dict[serial]["Coordinate"].values()) for (frame,serial) in zip(id2_frames,id2_serials) if frame in overlap_frames] 
            overlap_coefficients = []
            for id1_lap_bbox,id2_lap_bbox in zip(id1_lap_bboxes,id2_lap_bboxes):
                overlap_coefficient= self.get_overlap_coefficient(id1_lap_coordinate,id2_lap_coordinate)
                overlap_coefficients.append(overlap_coefficient)

            if np.mean(overlap_coefficients) < self.spatially_snms_th: 
                continue

            if self.merge_nonoverlap:
                for id2_serial,id2_frame in zip(id2_serials,id2_frames):
                    if id2_frame in overlap_frames:
                        self.tracking_dict[id2_serial]["LocalID"] = -1
                    else:
                        self.tracking_dict[id2_serial]["LocalID"] =  local_id1
                        local_id_frames_dict[local_id1].append(id2_frame)
                        local_id_serials_dict[local_id1].append(id2_serial)
                        local_id_frames_dict[local_id2].remove(id2_frame)
                        local_id_serials_dict[local_id2].remove(id2_serial)
            else:
                for id2_serial in id2_serials:
                    self.tracking_dict[id2_serial]["LocalID"] = -1

    def separate_warp_tracklet(self, **kwargs):
        # separate warp tracklets based on motion feature. 
        local_ids,serials = zip(*[(value["LocalID"],serial) for serial,value in self.tracking_dict.items() if value["LocalID"] != -1])
        local_id_serials_dict = self.make_dict(local_ids,serials)
        unique_local_ids = sorted(list(local_id_serials_dict.keys()))
        max_local_id = max(unique_local_ids)
        
        while len(unique_local_ids) > 0:
            local_id = unique_local_ids.pop(0)
            serials = local_id_serials_dict[local_id] 
            if len(serials) <= 2: 
                continue    
            frames = [self.tracking_dict[serial]["Frame"] for serial in serials]
            if len(frames) != len(set(frames)):
                raise ValueError(f"local_id{local_id} contains overlap")

            frames, serials = zip(*sorted(zip(frames, serials))) #sort by frame

            bboxes = np.array([tuple(self.tracking_dict[serial]["Coordinate"].values()) for serial in serials])
            center_bottoms = np.column_stack(((bboxes[:, 0] + bboxes[:, 2]) / 2, bboxes[:, 3]))

            split_index1 = self.get_warp_index_method1(frames,center_bottoms)
            split_index2 = self.get_warp_index_method2(frames,center_bottoms)
            split_index = self.get_min_lst_with_none([split_index1,split_index2])
            
            if split_index != None:
                split_serials = serials[split_index:]
                max_local_id += 1
                unique_local_ids.append(max_local_id)
                local_id_serials_dict[max_local_id] = split_serials
                for serial in split_serials:
                    self.tracking_dict[serial]["LocalID"] = max_local_id

    def get_warp_index_method1(self,frames,trajectory,**kwargs):
        """
        Kalmanフィルタの要領でワープの瞬間を捉える
        """
        # get index when occur the warp 
        alpha = 0.5

        split_index = None    
        interpolator = RegularGridInterpolator((np.array(frames),), np.array(trajectory), method='linear') 
        
        interpolaterd_frames = [i for i in range(min(frames),max(frames)+1)]
        interpolaterd_trajectory = []
        for i,frame in enumerate(interpolaterd_frames):
            coordinate = interpolator([frame])[0]
            interpolaterd_trajectory.append(tuple(coordinate))
        
        x_list, y_list = zip(*interpolaterd_trajectory)
        delta_x = [x_list[i+1]-x  for i,x in enumerate(x_list[:-1])]
        delta_y = [y_list[i+1]-y  for i,y in enumerate(y_list[:-1])]
        interpolaterd_trajectory = np.array(interpolaterd_trajectory)
        last_frame = max(frames)

        for t,frame in enumerate(interpolaterd_frames):
            if t == 1:
                weighted_cumsum = np.array([delta_x[t-1],delta_y[t-1]])
            if t > 1:
                weighted_cumsum = alpha*weighted_cumsum+(1-alpha)*np.array([delta_x[t-1],delta_y[t-1]])
                if frame not in frames:
                    continue
                current_position = interpolaterd_trajectory[t]
                past_position = interpolaterd_trajectory[t-1]
                pred_current_position = current_position + weighted_cumsum
                distance = np.sqrt(np.square(current_position[0] - pred_current_position[0])+np.square(current_position[1] - pred_current_position[1]))
                if distance > self.warp_th:
                    break      
                last_frame = frame  
        if last_frame != max(frames):
            split_index = frames.index(last_frame)
        return split_index

    def get_warp_index_method2(self,frames,trajectory,**kwargs): 
        """
        長期オクルージョン解消直後の座標と
        (1)長期オクルージョン直前の座標間の距離と(2)直前までのTrackletとの最短距離を求め
        (2)/(1)が小さいものをワープと見なす。
        ⇒直線的軌跡のエラーは低減、Uターンする軌跡が誤識別するリスク増加
        """
        interval = 20
        frame_intervals = np.diff(frames)
        indices = np.where(frame_intervals > interval)[0]
        split_index = None
        
        for index in indices:
            position_t1 = np.array(trajectory[index])
            position_t2 = np.array(trajectory[index+1])
            position_ut1 = np.array(trajectory[:index])

            interval_distance = np.linalg.norm(position_t1 - position_t2)
            if index > 0:
                min_distance = np.min(np.linalg.norm(position_ut1 - position_t2, axis=1))
            else:
                min_distance = np.linalg.norm(position_t1 - position_t2)
            
            if interval_distance < 50:
                continue        
            if min_distance/interval_distance < 0.3:
                split_index = index
                break
        return split_index
















