import os
import numpy as np
import sys
from sklearn.cluster import DBSCAN 
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations, permutations, product, chain
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.interpolate import RegularGridInterpolator
from collections import Counter


def create_centrality_matrix(clusters, similarity_matrix,frames,**kwargs):
    # translate the similarity matrix between each node into the centrality matrix between each cluster
    remove_noise_cluster = kwargs.get('remove_noise_cluster', True)
    epsilon = kwargs.get('epsilon', 0.3)

    unique_clusters = sorted(list(set(clusters)))
    if remove_noise_cluster:
        if -1 in unique_clusters: 
            unique_clusters.remove(-1)

    centrality_matrix =  np.ones((len(unique_clusters),len(unique_clusters)))*-1 
    np.fill_diagonal(centrality_matrix, 0)

    cluster_frames_dict = {cluster:[] for cluster in unique_clusters}
    if remove_noise_cluster:
        [cluster_frames_dict[cluster].append(frame) for frame,cluster in zip(frames,clusters) if cluster != -1]
    else:
        [cluster_frames_dict[cluster].append(frame) for frame,cluster in zip(frames,clusters)]

    for i in range(len(unique_clusters)):
        cluster1 = unique_clusters[i]
        cluster1_frames = cluster_frames_dict[cluster1]
        cluster1_indices = [k for k,cluster in enumerate(clusters) if cluster ==cluster1] #indices of similarity_matrix
        for j in range(i+1,len(unique_clusters)):
            cluster2 = unique_clusters[j]
            cluster2_frames = cluster_frames_dict[cluster2]
            common_frames = set(cluster1_frames).intersection(set(cluster2_frames))
            if len(common_frames) > 0: continue
            cluster2_indices = [k for k,cluster in enumerate(clusters) if cluster ==cluster2]
            similarities = similarity_matrix[np.ix_(cluster1_indices, cluster2_indices)]
            centrality =  np.sum(similarities[similarities > (1 - epsilon)])
            centrality_matrix[i,j] = centrality
            centrality_matrix[j,i] = centrality
    return centrality_matrix

def associate_cluster(clusters,centrality_matrix,**kwargs):
    # perform hierarchical clustering that targets clusters.
    epsilon = kwargs.get('epsilon', 0.3)
    remove_noise_cluster = kwargs.get('remove_noise_cluster', True)
    cost_function = kwargs.get('cost_function', 1)
    minimize = kwargs.get("minimize",True)
    """
    cost_function:1 ⇒ single linkage like
    cost_function:2 ⇒ average linkage like
    """
    np.fill_diagonal(centrality_matrix, 0)
    clusters = np.array(clusters)
    unique_clusters = np.sort(np.unique(clusters)) 
    if remove_noise_cluster:
        if -1 in unique_clusters: 
            unique_clusters = unique_clusters[unique_clusters != -1]

    if cost_function == 1:
        pass
    elif cost_function == 2:
        count = Counter(clusters)
        if remove_noise_cluster:
            if -1 in count.keys():
                del count[-1]
    centrality = np.max(centrality_matrix)

    th = 1 - epsilon 
    while centrality > th:
        if cost_function == 1:
            max_index = np.argmax(centrality_matrix)
        elif cost_function == 2:
            len_element_matrix = np.outer(list(count.values()),list(count.values())) 
            averaged_centrality_matrix = np.multiply(centrality_matrix,1/len_element_matrix)
            np.fill_diagonal(averaged_centrality_matrix, 0)
            max_index = np.argmax(averaged_centrality_matrix)

        cluster1_index, cluster2_index = np.unravel_index(max_index, centrality_matrix.shape)
        cluster1 = unique_clusters[cluster1_index]
        cluster2 = unique_clusters[cluster2_index]
        if cost_function == 1 or cost_function == 3:
            centrality = centrality_matrix[cluster1_index, cluster2_index]
        elif cost_function == 2:
            centrality = averaged_centrality_matrix[cluster1_index, cluster2_index]
            
        if centrality > th:
            target_row = centrality_matrix[[cluster1_index,cluster2_index],:]
            sum_row = np.sum(target_row,axis=0)
            if minimize:
                mask = np.min(target_row, axis=0)
                sum_row = np.where(mask < 0, -1, sum_row)
            centrality_matrix[:, cluster1_index] = sum_row
            centrality_matrix[cluster1_index,:] = sum_row

            next_indices = np.arange(len(unique_clusters))             
            next_indices = next_indices[next_indices != cluster2_index]
            centrality_matrix = centrality_matrix[np.ix_(next_indices,next_indices)]
            np.fill_diagonal(centrality_matrix, 0)
            clusters = np.where(clusters == cluster2, cluster1, clusters)
            unique_clusters = unique_clusters[unique_clusters != cluster2]

            if cost_function == 2:
                count[cluster1] += count[cluster2]
                del count[cluster2]
        else:
            break
    return clusters


def get_initial_index(distance_matrix,overlap_indices_list): 
    # determines the initial index for the assignment problem.
    distances = [] 
    for overlap_indices in overlap_indices_list:
        min_distance = 2
        for index1,index2 in combinations(overlap_indices,2): #
            distance = distance_matrix[index1,index2]
            min_distance = distance if distance < min_distance else min_distance
        distances.append(min_distance)     
    max_index = np.argmax(distances) 
    return max_index


def fill_none(lst):
    # fill "None" to the missing value in sequential number list
    used_nums = [num for num in lst if num is not None]
    unused_nums = [num for num in range(len(lst)) if num not in used_nums]
    for i in range(len(lst)):
        if lst[i] is None:
            lst[i] = unused_nums.pop(0)
    return lst

def get_candidates_indices_list(similarity_matrix,subcluster_indices_list,overlap_indices_list,epsilon,**kwargs):
    # get candidates of the assignment problem
    num_candidates = kwargs.get('num_candidates', 10)

    if len(overlap_indices_list) < num_candidates:
        candidates_indices_list = overlap_indices_list 
    else:
        np.fill_diagonal(similarity_matrix, 0)
        flatten_subcluster_indices = list(chain.from_iterable(subcluster_indices_list))
        tmp_similarity_matrix = similarity_matrix[flatten_subcluster_indices]
        
        max_similarities =np.max(tmp_similarity_matrix,axis=0)
        neighbor_indices = np.where(max_similarities > (1-epsilon))[0]
        sorted_indices = np.argsort(max_similarities[neighbor_indices])[::-1]
        neighbor_indices = neighbor_indices[sorted_indices]
    
        if len(neighbor_indices) > num_candidates:
            neighbor_indices = neighbor_indices[:num_candidates]
        neighbor_indices = neighbor_indices.tolist() 

        candidates_indices_list = []
        for neighbor_index in neighbor_indices:
            for overlap_indices in overlap_indices_list:
                if neighbor_index not in overlap_indices: continue
                candidates_indices_list.append(overlap_indices)
                for overlap_index in overlap_indices:
                    try:
                        neighbor_indices.remove(overlap_index)                        
                    except:
                        pass
    return candidates_indices_list

def agglomerative_clustering(distance_matrix,**kwargs): 
    # perform agglomerative hierarchical clustering
    epsilon = kwargs.get('epsilon', 0.3)
    metric = kwargs.get('metric','cosine')
    np.fill_diagonal(distance_matrix, 0) 
    linked = linkage(squareform(distance_matrix), method='single', metric=metric)
    clusters = list(fcluster(linked, epsilon, criterion='distance')) # min(clusters)=1
    return clusters

def bipartite_matching(new_key,centrality_dict,centrality_matrix,overlap_indices,**kwargs):
    # bipartite matching between unclustered overlap nodes and clustered overlap nodes 
    epsilon = kwargs.get('epsilon', 0.3)

    sum_centrality = 0
    subcluster_indices = [None]*len(overlap_indices)
    th = 1-epsilon
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


def separate_into_subcluster(tmp_clusters, overlap_indices_list, distance_matrix,**kwargs):
    # overlap nodes are separated into subclusters
    epsilon = kwargs.get('epsilon', 0.3)
    matching_algo_th = kwargs.get('matching_algo_th', 0)   
    debug = kwargs.get('debug', False)

    max_overlap = max([len(i) for i in overlap_indices_list]) #the number of overlap in the same frame
    initial_index = get_initial_index(distance_matrix,overlap_indices_list) #index of overlap_indices_list
    initial_node_indices = overlap_indices_list[initial_index] 
    del overlap_indices_list[initial_index]

    subcluster_indices_list =  [[] for _ in range(max_overlap)] 
    [subcluster_indices_list[i].append(initial_node_index) for i,initial_node_index in enumerate(initial_node_indices)]

    similarity_matrix = 1-distance_matrix
    np.fill_diagonal(similarity_matrix, 0) 
    
    # separte overlap nodes into several groups 
    while len(overlap_indices_list) != 0:    
        centrality_dict = {}
        max_centrality = 0

        candidates_indices_list = get_candidates_indices_list(similarity_matrix,subcluster_indices_list,overlap_indices_list,epsilon)  
        for i,overlap_indices in enumerate(candidates_indices_list): 
            centrality_matrix = np.zeros((len(overlap_indices),len(subcluster_indices_list))) #can not use create_centrality_matrix
            for j, overlap_index in enumerate(overlap_indices):
                tmp_similarity_matrix = similarity_matrix[overlap_index]
                for k, subcluster_indices in enumerate(subcluster_indices_list):
                    similarities = tmp_similarity_matrix[subcluster_indices]
                    centrality =  np.sum(similarities[similarities > (1 - epsilon)]) 
                    centrality_matrix[j,k] = centrality
            
            centrality_dict = bipartite_matching(i,centrality_dict,centrality_matrix,overlap_indices,epsilon=epsilon)
        
        max_centrality = 0 if centrality_dict == {} else np.max([value["centrality"] for value in centrality_dict.values()]) 

        if max_centrality == 0:
            max_index = get_initial_index(distance_matrix,overlap_indices_list)
            max_subcluster_indices = list(range(max_overlap))
            overlap_indices = overlap_indices_list[max_index]
        else:
            max_index = [key for key,value in zip(centrality_dict,centrality_dict.values()) if value["centrality"]==max_centrality][0] 
            max_subcluster_indices = list(centrality_dict[max_index]["indices"]) 
            if None in max_subcluster_indices:
                max_subcluster_indices = fill_none(max_subcluster_indices)
            overlap_indices = centrality_dict[max_index]["overlap_indices"]
        [subcluster_indices_list[max_subcluster_index].append(overlap_index) for max_subcluster_index,overlap_index in zip(max_subcluster_indices,overlap_indices)]    
        overlap_indices_list.remove(overlap_indices)
    
    # assign cluster ID
    for subcluster_indices in subcluster_indices_list:
        if len(subcluster_indices) == 1: 
            tmp_clusters[subcluster_indices[0]] = np.max(tmp_clusters)+1
        else:
            sub_clusters = agglomerative_clustering(distance_matrix[np.ix_(subcluster_indices, subcluster_indices)],epsilon=epsilon)
            sub_clusters = [sub_cluster+max(tmp_clusters) for sub_cluster in sub_clusters]            
            for sub_cluster,sub_cluster_index in zip(sub_clusters,subcluster_indices):
                tmp_clusters[sub_cluster_index] = sub_cluster
    return tmp_clusters

def overlap_suppression_clustering(distance_matrix,frames,nonoverlap_indices,overlap_indices_list,**kwargs): #overlap_indices_list,
    epsilon = kwargs.get('epsilon', 0.3)
    debug = kwargs.get('debug', False)
    clusters = [-1]*len(frames) 
    
    # clustering for non-overlapping nodes
    if nonoverlap_indices != []:
        if len(nonoverlap_indices) > 1:
            nonoverlap_clusters = agglomerative_clustering(distance_matrix[np.ix_(nonoverlap_indices,nonoverlap_indices)], epsilon=epsilon)
        else:
            nonoverlap_clusters = [0]
        for k,target_index in enumerate(nonoverlap_indices):
            clusters[target_index] = nonoverlap_clusters[k]  

    # clustering for overlapping nodes         
    clusters = separate_into_subcluster(clusters, overlap_indices_list, distance_matrix,epsilon=epsilon,debug=debug)

    similarity_matrix = 1 - distance_matrix
    centrality_matrix = create_centrality_matrix(clusters,similarity_matrix,frames,epsilon=epsilon)

    # merging for subcluster
    clusters = associate_cluster(clusters,centrality_matrix, epsilon=epsilon)

    return clusters

def divide_overlap_or_nonoverlap(cluster_frames,cluster_indices):

    frame_indices_dict = {frame:[] for frame in sorted(list(set(cluster_frames)))}
    [frame_indices_dict[frame].append(index) for index,frame in zip(cluster_indices,cluster_frames)]
    overlap_indices_list = [indices for indices in frame_indices_dict.values() if len(indices) > 1] 
    flattened_overlap_indices = sum(overlap_indices_list, []) 
    nonoverlap_indices = [index for index in cluster_indices if index not in flattened_overlap_indices]

    return overlap_indices_list, nonoverlap_indices

def reclustering_overlap_cluster(distance_matrix,tracking_dict,serials,clusters,**kwargs): 
    epsilon = kwargs.get('epsilon', 0.3)
    debug = kwargs.get('debug', False)

    frames = [tracking_dict[serial]["Frame"] for serial in serials]
    
    cluster_frame_dict = {cluster:[] for cluster in set(clusters)} #20240418 add set()
    [cluster_frame_dict[cluster].append(frame) for cluster,frame in zip(clusters,frames)]
    cluster_indices_dict = {cluster:[] for cluster in set(clusters)} #20240418 add set()
    [cluster_indices_dict[cluster].append(i) for i,cluster in enumerate(clusters)]

    for cluster in cluster_frame_dict: 
        cluster_frames = cluster_frame_dict[cluster]
        if len(list(set(cluster_frames))) == len(cluster_frames):continue 
        cluster_indices = cluster_indices_dict[cluster]

        #divide overlap/nonoverlap
        overlap_indices_list, nonovelap_indices = divide_overlap_or_nonoverlap(cluster_frames,cluster_indices)
    
        tmp_clusters = overlap_suppression_clustering(distance_matrix,frames,nonovelap_indices,overlap_indices_list,epsilon=epsilon,debug=debug)  

        max_cluster_id = np.max(clusters) 
        for index,tmp_cluster in enumerate(tmp_clusters):
            if clusters[index] != cluster: continue 
            clusters[index] = max_cluster_id + tmp_cluster + 1
    return clusters 

def create_similarity_matrix_scpt(serials, tracking_dict, epsilon):
    # create a similarity matrix from features
    for n,serial in enumerate(serials):
        feature = np.load(tracking_dict[serial]["NpyPath"])
        if n==0: feature_stack = np.empty((0,len(feature.flatten())))
        feature_stack  = np.append(feature_stack , feature.reshape(1,-1) , axis=0)
    similarity_matrix = cosine_similarity(feature_stack)
    similarity_matrix = similarity_matrix.astype(np.float16)
    
    similarity_matrix = np.where(similarity_matrix < (1-epsilon),0,similarity_matrix)
    return similarity_matrix


def tracking_by_clustering(tracking_dict,serials,**kwargs):
    min_samples = kwargs.get('min_samples', 4)
    epsilon = kwargs.get('epsilon_scpt', 0.3)
    overlap_suppression = kwargs.get('overlap_suppression', True)
    debug = kwargs.get('debug', False)
    clustering_method = kwargs.get('clustering_method', "agglomerative")

    if len(serials) ==1:
        clusters = [0]    
    else:
        similarity_matrix = create_similarity_matrix_scpt(serials,tracking_dict,epsilon)
        
        np.fill_diagonal(similarity_matrix, 1)
        distance_matrix = 1 - similarity_matrix
        if clustering_method == "agglomerative":
            clusters = agglomerative_clustering(distance_matrix,epsion=epsilon) #min(clusters)=1
        
        elif clustering_method == "dbscan":
            dbscan = DBSCAN(eps=epsilon,min_samples=min_samples,metric="precomputed")
            clusters = dbscan.fit_predict(distance_matrix)
            coreindices = dbscan.core_sample_indices_
            clusters = [cluster if cluster != -1 else -i  for i,cluster in enumerate(clusters)]
            unique_clusters = list(set(clusters))
            new_clusterid_dict = {key:i for i,key in enumerate(unique_clusters)} 
            clusters = [new_clusterid_dict[old_cluster] for old_cluster in clusters]
            
        if overlap_suppression == True:    
            clusters = reclustering_overlap_cluster(distance_matrix,tracking_dict,serials,clusters,epsilon=epsilon,debug=debug)
    
    unique_clusters = list(set(clusters))
    new_clusterid_dict = {key:i for i,key in enumerate(unique_clusters)} 
    clusters = [new_clusterid_dict[old_cluster] for old_cluster in clusters]

    return clusters 

def associate_cluster_between_period(tracking_dict,clusters,serials,past_serials,**kwargs):
    # associate clusters between adjacent time periods
    epsilon = kwargs.get('epsilon_scpt', 0.3)
    frames = [tracking_dict[serial]["Frame"] for serial in serials] 
    past_frames = [tracking_dict[serial]["Frame"] for serial in past_serials] 
    offline_ids = [tracking_dict[serial]["OfflineID"] for serial in past_serials] 
    
    unique_offline_ids = list(set(offline_ids))
    unique_clusters = list(set(clusters))

    all_serials = past_serials + serials
    all_clusters = offline_ids + clusters
    all_unique_clusters = sorted(unique_offline_ids + unique_clusters)
    all_frames = past_frames  + frames

    similarity_matrix = create_similarity_matrix_scpt(all_serials, tracking_dict,epsilon)

    centrality_matrix = create_centrality_matrix(all_clusters, similarity_matrix,all_frames,epsilon=epsilon)
    del similarity_matrix
    np.fill_diagonal(centrality_matrix, 0)
    
    all_clusters = associate_cluster(all_clusters,centrality_matrix,epsilon=epsilon)

    for serial,cluster in zip(all_serials,all_clusters):
        tracking_dict[serial]["OfflineID"] = int(cluster)
    return tracking_dict

def get_overlap_coefficient(rectangle1, rectangle2):
    # meaure spatially overlap_coefficient
    overlap_width = min(rectangle1[2], rectangle2[2]) - max(rectangle1[0], rectangle2[0])
    overlap_height = min(rectangle1[3], rectangle2[3]) - max(rectangle1[1], rectangle2[1])
    overlap_area = max(overlap_width, 0) * max(overlap_height, 0)
    rectangle1_area = (rectangle1[2] - rectangle1[0]) * (rectangle1[3] - rectangle1[1])
    rectangle2_area = (rectangle2[2] - rectangle2[0]) * (rectangle2[3] - rectangle2[1])
    #iou = overlap_area / (rectangle1_area + rectangle2_area - overlap_area)
    overlap_coefficient = overlap_area / min(rectangle1_area,rectangle2_area)
    return overlap_coefficient

def sequential_non_maximum_suppression(tracking_dict,**kwargs):
    #Sequential NMS is perfomed in this function.
    #Sequential NMS calculates the overlap coefficient both temporally and spatially. 
    temporally_snms_th = kwargs.get('temporally_snms_th', 0.6)
    spatially_snms_th = kwargs.get('spatially_snms_th', 0.6)
    remove_noise_cluster = kwargs.get('remove_noise_cluster', True)
    merge_nonoverlap = kwargs.get('merge_nonoverlap', True)

    offline_ids = [tracking_dict[serial]["OfflineID"] for serial in tracking_dict.keys()]
    unique_offline_ids = sorted(list(set(offline_ids)))
    if remove_noise_cluster:
        if min(unique_offline_ids) == -1: 
            unique_offline_ids.remove(-1)

    offline_id_serial_dict = {offline_id:[] for offline_id in unique_offline_ids}
    [offline_id_serial_dict[tracking_dict[serial]["OfflineID"]].append(serial) for serial in tracking_dict.keys() if tracking_dict[serial]["OfflineID"] != -1]
    offline_id_frame_dict = {offline_id:[] for offline_id in unique_offline_ids}
    [offline_id_frame_dict[tracking_dict[serial]["OfflineID"]].append(tracking_dict[serial]["Frame"]) for serial in tracking_dict.keys() if tracking_dict[serial]["OfflineID"] != -1]
        
    for offline_id1, offline_id2 in combinations(unique_offline_ids,2):

        id1_frames = offline_id_frame_dict[offline_id1]
        id2_frames = offline_id_frame_dict[offline_id2]
        overlap_frames = set(id1_frames).intersection(set(id2_frames))

        if len(id1_frames) < len(id2_frames):
            (offline_id1,offline_id2) = (offline_id2,offline_id1)
            (id1_frames,id2_frames) = (id2_frames,id1_frames)

        if max(len(overlap_frames)/len(id1_frames),len(overlap_frames)/len(id2_frames)) <temporally_snms_th: continue 
        
        id1_serials = offline_id_serial_dict[offline_id1]
        id2_serials = offline_id_serial_dict[offline_id2]

        id1_lap_pos_list = [list(tracking_dict[serial]["Coordinate"].values()) for n,(frame,serial) in enumerate(zip(id1_frames,id1_serials)) if (frame in overlap_frames)] 
        id2_lap_pos_list = [list(tracking_dict[serial]["Coordinate"].values()) for n,(frame,serial) in enumerate(zip(id2_frames,id2_serials)) if (frame in overlap_frames)] 

        overlap_coefficients = []
        for id1_lap_pos,id2_lap_pos in zip(id1_lap_pos_list,id2_lap_pos_list):
            overlap_coefficient= get_overlap_coefficient(id1_lap_pos,id2_lap_pos)
            overlap_coefficients.append(overlap_coefficient)

        if np.mean(overlap_coefficients) < spatially_snms_th: continue

        if merge_nonoverlap:
            for id2_serial,id2_frame in zip(id2_serials,id2_frames):
                if id2_frame in overlap_frames:
                    tracking_dict[id2_serial]["OfflineID"] = -1
                else:
                    tracking_dict[id2_serial]["OfflineID"] =  offline_id1
                    offline_id_frame_dict[offline_id1].append(id2_frame)
                    offline_id_serial_dict[offline_id1].append(id2_serial)
                    offline_id_frame_dict[offline_id2].remove(id2_frame)
                    offline_id_serial_dict[offline_id2].remove(id2_serial)
        else:
            for noise_serial in noise_serials:
                tracking_dict[noise_serial]["OfflineID"] = -1

    return tracking_dict

def get_warp_index(frames,trajectory,**kwargs):
    # get index when occur the warp 
    alpha = kwargs.get('alpha', 0.5)
    warp_th = kwargs.get('warp_th', 50)

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
            if distance > warp_th:
                break      
            last_frame = frame  
    if last_frame != max(frames):
        split_index = frames.index(last_frame)
    return split_index

def separate_warp_tracklet(tracking_dict,**kwargs):
    # separate warp tracklets based on motion feature. 
    remove_noise_cluster = kwargs.get('remove_noise_cluster', True)
    warp_th = kwargs.get('warp_th', 50)

    offline_ids = [tracking_dict[serial]["OfflineID"] for serial in tracking_dict.keys()]
    unique_offline_ids = sorted(list(set(offline_ids)))
    if remove_noise_cluster:
        if min(unique_offline_ids) == -1: 
            unique_offline_ids.remove(-1)

    offline_id_serial_dict = {offline_id:[] for offline_id in unique_offline_ids} 
    [offline_id_serial_dict[tracking_dict[serial]["OfflineID"]].append(serial) for serial in tracking_dict if tracking_dict[serial]["OfflineID"] != -1]

    max_offline_id = max(unique_offline_ids)

    while len(unique_offline_ids) > 0:
        offline_id = unique_offline_ids.pop(0)
        serials = offline_id_serial_dict[offline_id] 
        if len(serials) <= 2: 
            continue    
        frames = [tracking_dict[serial]["Frame"] for serial in serials]
        if len(frames) != len(set(frames)):
            print(f"offline_id{offline_id} contains overlap")
            continue
        frames, serials = zip(*sorted(zip(frames, serials))) #sort by frame
        pos_list = [tracking_dict[serial]["Coordinate"] for serial in serials]
        trajectory = [((pos["x1"]+pos["x2"])/2,pos["y2"]) for pos in pos_list]
        split_index = get_warp_index(frames,trajectory,warp_th=warp_th)
        
        if split_index != None:
            split_serials = serials[split_index:]
            max_offline_id += 1
            unique_offline_ids.append(max_offline_id)
            offline_id_serial_dict[max_offline_id] = split_serials
            for serial in split_serials:
                tracking_dict[serial]["OfflineID"] = max_offline_id
    return tracking_dict

def exclude_short_tracklet(tracking_dict,**kwargs):
    # exclude tracklet that contains only a little serials from tracking_dict
    short_tracklet_th = kwargs.get('short_tracklet_th', 5)

    offline_ids = [tracking_dict[serial]["OfflineID"] for serial in tracking_dict]
    unique_offline_ids = sorted(list(set(offline_ids)))
    if min(unique_offline_ids) == -1: unique_offline_ids.remove(-1)

    offline_id_serial_dict = {offlineID:[] for offlineID in unique_offline_ids} #OnlineIDからserialを検索するDict
    [offline_id_serial_dict[tracking_dict[serial]["OfflineID"]].append(serial) for serial in tracking_dict if tracking_dict[serial]["OfflineID"] != -1]

    for offline_id in unique_offline_ids:
        serials = offline_id_serial_dict[offline_id]
        if len(serials) <= short_tracklet_th:
            for serial in serials:
                tracking_dict[serial]["OfflineID"] = -1
    return tracking_dict

def exclude_motionless_tracklet(tracking_dict,**kwargs):
    # exclude tracklet from tracking_dict
    stop_track_th = kwargs.get('stop_track_th', 25)

    offline_ids = [tracking_dict[serial]["OfflineID"] for serial in tracking_dict]
    unique_offline_ids = sorted(list(set(offline_ids)))
    if min(unique_offline_ids) == -1: unique_offline_ids.remove(-1)

    offline_id_serial_dict = {offlineID:[] for offlineID in unique_offline_ids} #OnlineIDからserialを検索するDict
    [offline_id_serial_dict[tracking_dict[serial]["OfflineID"]].append(serial) for serial in tracking_dict if tracking_dict[serial]["OfflineID"] != -1]

    for offline_id in unique_offline_ids:
        serials = offline_id_serial_dict[offline_id]
        pos_list = [tracking_dict[serial]["Coordinate"] for serial in serials] 
        x_pos_list = [(pos["x1"]+pos["x2"])/2 for pos in pos_list]
        y_pos_list = [pos["y2"] for pos in pos_list]
        x_min = np.min(x_pos_list)
        x_max = np.max(x_pos_list)
        y_min = np.min(y_pos_list)
        y_max = np.max(y_pos_list)
        if (x_max-x_min < stop_track_th) and (y_max-y_min < stop_track_th):
            for serial in serials:
                tracking_dict[serial]["OfflineID"] = -1

    return tracking_dict
