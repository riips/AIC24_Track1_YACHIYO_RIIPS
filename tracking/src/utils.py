from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import random
import cv2
import shutil

import os
import numpy as np
import json
import glob
import sys

from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations, permutations, product, chain
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform

# import pose
from parameters import CommonParameters


class ArrayCalculator:
    """
    ■Todo
    一部関数の実行元がArrayになってないかも
    """
    @staticmethod
    def translate_world_coordinate(x, y, homography_matrix):
        # translate camera coordinate to world coordinate
        vector_xyz = np.array([x, y, 1]) # z=1
        vector_xyz_3d = np.dot(np.linalg.inv(homography_matrix), vector_xyz.T)
        return vector_xyz_3d[0] / vector_xyz_3d[2], vector_xyz_3d[1] / vector_xyz_3d[2]

    @staticmethod
    def measure_aspect_ratios(self,coordinates):
        x1, y1, x2, y2 = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], coordinates[:, 3] 
        return (y2-y1)/(x2-x1)

    @staticmethod
    def run_ratio_test(matrix:np.ndarray,axis=0):
        if axis not in [0, 1]:
            raise ValueError("axis must be 0 or 1")
        max_values = np.max(matrix, axis=axis)
        len_row, len_col = matrix.shape

        if axis == 0:
            second_values = np.partition(matrix, -2, axis=axis)[-2, :] if len_row > 1 else np.zeros(len_col)
        else:
            second_values =  np.partition(matrix, -2, axis=axis)[:, -2] if len_col > 1 else np.zeros(len_row)

        ratio_test_results = np.divide(second_values, max_values, out=np.ones(len(max_values)), where=max_values != 0)
        return max_values, ratio_test_results

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def measure_euclidean_distances(list1,list2):
        points1 = np.array(list1)
        points2 = np.array(list2)
        diff = points1-points2
        return np.sqrt(np.sum(diff**2, axis=1))

    @staticmethod
    def _compute_intersection_area(bboxes1, bboxes2, mode):
        # Calculate the intersection areas and bounding box areas
        x1_1, y1_1, x2_1, y2_1 = bboxes1[:, 0], bboxes1[:, 1], bboxes1[:, 2], bboxes1[:, 3]
        x1_2, y1_2, x2_2, y2_2 = bboxes2[:, 0], bboxes2[:, 1], bboxes2[:, 2], bboxes2[:, 3]

        inter_x1 = np.maximum(x1_1[:, None] if mode == "all_combinations" else x1_1, x1_2)
        inter_y1 = np.maximum(y1_1[:, None] if mode == "all_combinations" else y1_1, y1_2)
        inter_x2 = np.minimum(x2_1[:, None] if mode == "all_combinations" else x2_1, x2_2)
        inter_y2 = np.minimum(y2_1[:, None] if mode == "all_combinations" else y2_1, y2_2)

        inter_width = np.maximum(0, inter_x2 - inter_x1)
        inter_height = np.maximum(0, inter_y2 - inter_y1)
        intersection = inter_width * inter_height

        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

        return intersection, area1, area2

    @staticmethod
    def compute_iou(bboxes1, bboxes2, mode="all_combinations"): 
        intersection, area1, area2 = ArrayCalculator._compute_intersection_area(bboxes1, bboxes2, mode)
        iou = intersection / (area1[:, None] + area2 - intersection) if mode == "all_combinations" else intersection / (area1 + area2 - intersection)
        return iou

    @staticmethod
    def compute_overlap_coefficient(bboxes1, bboxes2, mode="elementwise"):
        intersection, area1, area2 = ArrayCalculator._compute_intersection_area(bboxes1, bboxes2, mode)
        overlap_coefficient = intersection / np.minimum(area1[:, None], area2) if mode == "all_combinations" else intersection / np.minimum(area1, area2)
        return overlap_coefficient
     
class JSONHandler:
    @staticmethod
    def save_json(dict_, json_path):
        os.makedirs(os.path.dirname(json_path),exist_ok=True)
        try:
            with open(json_path, mode='w') as f:
                json.dump(dict_, f)
        except OSError as e:
            raise Exception(f"Error creating directory or writing file: {e}")

    @staticmethod
    def open_json(json_path):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"File not found: {json_path}")
        with open(json_path) as f:
            json_file = json.load(f)
        return json_file
    
    @staticmethod
    def concat_jsons(json_dir, startwith="",endwith=""):
        json_paths = sorted(glob.glob(os.path.join(json_dir,f"{startwith}*{endwith}.json")))
        jsons = {}
        for json_path in json_paths:
            json_ = JSONHandler.open_json(os.path.join(json_path))
            jsons.update(json_)
        return jsons

class ParameterManager:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def update_params(self, params_dict):
        for key, value in params_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ReferenceError(f"The parameter '{key}' has not been set.")

class DirectoryStructure(CommonParameters,ParameterManager):
    def __init__(self,common_params=None):
        CommonParameters.__init__(self)
        ParameterManager.__init__(self)
        if common_params:
            self.update_params(common_params)

    def _get_frame_dir(self):
        dir_ = os.path.join("Frames",f"Scene{str(self.scene_id).zfill(self.scene_digit)}",f"Camera{str(self.camera_id).zfill(self.camera_digit)}")
        return dir_

    def _get_feature_dir(self,):
        dir_ = os.path.join("EmbedFeature",f"{self.emb_model}",f"Scene{str(self.scene_id).zfill(self.scene_digit)}",f"Camera{str(self.camera_id).zfill(self.camera_digit)}")
        return dir_

    def _get_result_dir(self,):
        return

    def _get_pose_dir(self,):
        dir_ = os.path.join("Poses",f"{self.pose_model}",f"{self.pose_format}",f"Scene{str(self.scene_id).zfill(self.scene_digit)}",f"Camera{str(self.camera_id).zfill(self.camera_digit)}")
        return dir_

    def _get_pose_path(self,):
        return os.path.join(self._get_pose_dir(),"keypoint.json")

    def get_pose_result(self):
        return JSONHandler.open_json(self._get_pose_path())
    
    def get_scpt_path(self,):
        return


class ClusteringFunctions:
    @staticmethod
    def agglomerative_clustering(distance_matrix,epsilon,metric="cosine"):  
        # perform agglomerative hierarchical clustering
        np.fill_diagonal(distance_matrix, 0) 
        linked = linkage(squareform(distance_matrix), method='single', metric=metric)
        return list(fcluster(linked, epsilon, criterion='distance')) # min(clusters)=1

    @staticmethod
    def dbscan(distance_matrix,min_samples=5,epsilon=0.2):
        dbscan = DBSCAN(eps=epsilon,min_samples=min_samples,metric="precomputed")
        clusters = dbscan.fit_predict(distance_matrix)
        # coreindices = dbscan.core_sample_indices_
        #assign a unique ID to each noise point
        clusters = [cluster if cluster != -1 else -i  for i,cluster in enumerate(clusters)]
        return clusters #self.reassign_sequential_ids(

class TrackingFunctions():
    def delete_variable(self, var_name):
        if hasattr(self, var_name):
            delattr(self, var_name)
        else:
            # print(f"'{var_name}' does not exist.")
            pass

    def loop(self, loop_list, function):
        list_ = []
        for loop_element in loop_list:
            value = function(loop_element)
            list_.append(value)
        return list_
    
    def make_dict(self,keys,values):
        new_dict = defaultdict(list)
        for key, value in zip(keys, values):
            if key is None:
                continue
            new_dict[key].append(value)
        return dict(new_dict)

    def get_max_value_of_dict(self, dictionary, key):
        # get max value of any key from nested dictionary
        max_value = float('-inf')  
        for k, v in dictionary.items():
            if isinstance(v, dict):
                max_value = max(max_value, self.get_max_value_of_dict(v, key))
            elif k == key:  
                max_value = max(max_value, v)
        return max_value

    def get_min_lst_with_none(self,values):
        filtered_values = list(filter(lambda x: x is not None, values))
        return min(filtered_values) if filtered_values else None    

    def reassign_sequential_ids(self,id_list):
        unique_ids = list(set(id_list))
        new_id_dict = {key:i for i,key in enumerate(unique_ids)} 
        return [new_id_dict[old_id] for old_id in id_list]
 
    def load_npy_from_npypath(self,npy_path):
        return np.load(npy_path).reshape(1, -1)

    def make_feature_stack(self, npy_paths):
        feature_list = self.loop(npy_paths,self.load_npy_from_npypath)
        return np.vstack(feature_list)

    def similarity_matrix_by_npypaths(self,npy_paths):
        # create a similarity matrix from features
        feature_stack = self.make_feature_stack(npy_paths)
        similarity_matrix = cosine_similarity(feature_stack).astype(np.float16)
        return similarity_matrix
    
    def associate_cluster(self,clusters,centrality_matrix,epsilon,**kwargs):#, **parameters
        # perform hierarchical clustering that targets clusters.
        remove_noise_cluster = kwargs.get('remove_noise_cluster', True) #self.parameters["remove_noise_cluster"] #
        cost_function = kwargs.get('cost_function', 1) #self.parameters["cost_function"] #
        minimize = kwargs.get("minimize",True) #self.parameters["minimize"] 
        """
        cost_function:1 ⇒ single linkage like
        cost_function:2 ⇒ average linkage like
        """
        np.fill_diagonal(centrality_matrix, 0)
        clusters = np.array(clusters)
        unique_clusters = np.sort(np.unique(clusters)) 

        if remove_noise_cluster and -1 in unique_clusters:
            unique_clusters = unique_clusters[unique_clusters != -1]

        if cost_function == 2:
            count = Counter(clusters)
            if remove_noise_cluster and -1 in count.keys():
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
            
            if cost_function == 1:
                centrality = centrality_matrix[cluster1_index, cluster2_index]
            elif cost_function == 2:
                centrality = averaged_centrality_matrix[cluster1_index, cluster2_index]  
            
            if centrality < th:
                break
            
            target_row = centrality_matrix[[cluster1_index,cluster2_index],:]
            sum_row = np.sum(target_row,axis=0)
            if minimize:
                sum_row = np.where(np.min(target_row, axis=0) < 0, -1, sum_row)
            centrality_matrix[:, cluster1_index] = sum_row
            centrality_matrix[cluster1_index,:] = sum_row

            next_indices = np.arange(len(unique_clusters))             
            next_indices = next_indices[next_indices != cluster2_index]
            centrality_matrix = centrality_matrix[np.ix_(next_indices,next_indices)] 
            np.fill_diagonal(centrality_matrix, 0)

            cluster1 = unique_clusters[cluster1_index]
            cluster2 = unique_clusters[cluster2_index] 
            clusters = np.where(clusters == cluster2, cluster1, clusters)
            unique_clusters = unique_clusters[unique_clusters != cluster2]

            if cost_function == 2:
                count[cluster1] += count[cluster2]
                del count[cluster2]
            
        return clusters


class DetectedObjects(CommonParameters,ParameterManager):
    """
    Represents whole detected objects to track.
    Object dict is built by frame_id as a key and its entity contains a list of all Detected objects of the frame. 
    """
    def __init__(self,params={}):
        super().__init__(**params)

        self.num_objects = 0
        self.objects = {}
        self._objects_registered = {}
        self.camera_projection_matrix = None
        self.homography_matrix = None

    def __str__(self):
        return f"DetectedObjects: scene_id:{self.scene_id}, camera_id:{self.camera_id}, num_objects:{self.num_objects}"

    def load_from_directory(self, feature_root, calibration_path="Calibration"):
        if not os.path.isdir(feature_root):
            raise Exception(f'There is no directory to read from. {feature_root}')
        npys = sorted(glob.glob(os.path.join(feature_root, "**/*.npy"), recursive=True))
        calibration_path = os.path.join(calibration_path, f"scene_{self.scene_id:03d}", f"camera_{self.camera_id:04d}.json")
        self.load_calibration(calibration_path)
        for f in npys:
            self.add_object_from_image_path(f)

    def add_object(self, frame_id, coordinate, world_coordinate, confidence, feature_path, image_path=None):
        if isinstance(frame_id, str):
            frame_id = int(frame_id)

        # Check if coordinate is reasonable
        if coordinate.x1 >= coordinate.x2 or coordinate.y1 >= coordinate.y2:
            print(f"Unnatural coordinate found in frame {frame_id}: {coordinate}")
            return

        detected_obj = DetectedObject(object_id=self.num_objects, frame_id=frame_id, coordinate=coordinate, worldcoordinate=world_coordinate,
                                      confidence=confidence, feature_path=feature_path)
        key = f"{coordinate.x1}_{coordinate.y1}_{coordinate.x2}_{coordinate.y2}"
        if frame_id in self.objects:
            if not key in self._objects_registered[frame_id]:
                objects_per_frame = self.objects[frame_id].append(detected_obj)
                self._objects_registered[frame_id].append(key)
            else:
                print(f"Duplicate coord found in frame {frame_id}: {coordinate}")
                return
        else:
            objects_per_frame = self.objects[frame_id] = [detected_obj]
            self._objects_registered[frame_id] = [key]
        self.num_objects += 1

    def add_object_from_image_path(self, feature_path, image_path=None, calibration_path="Calibration"):
        file_path = os.path.basename(feature_path)
        if file_path.startswith("feature_"):
            _, frame_id, serial_no, x1, x2, y1, y2, conf = os.path.splitext(file_path)[0].split("_")
            conf = conf if len(conf) == 1 else conf[0]+"."+conf[1:]
        else:
            serial_no, frame_id, x1, x2, y1, y2 = os.path.splitext(file_path)[0].split("_")
            x1, x2, y1, y2 = int(x1.replace("x","")), int(x2), int(y1.replace("y","")), int(y2)
            conf = 0.98765 # Dummy
        World_coordinate = None
        if self.homography_matrix is not None:
            w_x, w_y = self.convert_coordinates_2world((int(float(x1)) + int(float(x2))) / 2, int(float(y2)))
            World_coordinate = WorldCoordinate(w_x, w_y)

        self.add_object(frame_id=int(frame_id), coordinate=Coordinate(x1, y1, x2, y2), world_coordinate=World_coordinate,
                        confidence=float(conf), feature_path=feature_path, image_path=image_path)

    def get_objects_of_frames(self, start_frame, end_frame):
        if start_frame > self.num_frames() or end_frame > self.num_frames():
            return None
        object_dict = {}
        for frame_id in range(start_frame, end_frame):
            if frame_id in self.objects:
                object_dict[frame_id] = self[frame_id]
            #else:
            #    print(f"There is no such frame in the DetectedObjects, will be ignored. frame_id: {frame_id}")
        return object_dict

    def get_object_ids_of_frames(self, start_frame, end_frame):
        """
        Returns a list of detected object IDs that appeared within the specified frame window.
        """
        if start_frame > self.num_frames() or end_frame > self.num_frames():
            return None
        object_ids = []
        for frame_id in range(start_frame, end_frame):
            if frame_id in self.objects:
                for det in self[frame_id]:
                    object_ids.append(det.object_id)
        return sorted(object_ids)

    def __getitem__(self, frame_id):
        if frame_id in self.objects:
            return self.objects[frame_id]
        else:
            return None

    def num_frames(self):
        """
        Returns number of frames that currently holding.
        """
        return len(self.objects)

    def last_frame_id(self):
        """
        Returns the last frame id.
        """
        return max(self.objects.keys())

    def to_trackingdict(self):
        """
        Compatibility function to convert detections in TrackingDict format.
        """
        track_dict = {}
        for frame_id, detected_objects in self.objects.items():
            """
            ■メモ
            detected_objectsに同一フレームのbbox情報が記録されている
            """
            for detected_object in detected_objects:

                serial_no = detected_object.object_id
                coordinate = json.loads(detected_object.coordinate.__str__())
                if detected_object.worldcoordinate.__str__() != "None":
                    world_coordinate = json.loads(detected_object.worldcoordinate.__str__())
                else:
                    world_coordinate = None
                new_object = { "Frame": frame_id, "NpyPath": detected_object.feature_path,
                                "Coordinate": coordinate, "WorldCoordinate": world_coordinate,  "LocalID": -1 }
                track_dict[serial_no] = new_object
        return track_dict

    def load_calibration(self, calib_path):
        if os.path.isfile(calib_path):
            with open(calib_path, 'r') as file:
                data = json.load(file)
                self.camera_projection_matrix = np.array(data["camera projection matrix"])
                self.homography_matrix =  np.array(data["homography matrix"])
        else:
            print(f'\033[33mwarning\033[0m : not found Calibration File.')
            print(f'\033[33mwarning\033[0m : world coordinate calculations are ignored.')

    def convert_coordinates_2world(self, x, y):
        vector_xyz = np.array([x, y, 1]) # z=1
        vector_xyz_3d = np.dot(np.linalg.inv(self.homography_matrix), vector_xyz.T)
        return vector_xyz_3d[0] / vector_xyz_3d[2], vector_xyz_3d[1] / vector_xyz_3d[2]
        
class DetectedObject:
    """
    Represents individual detected object to track.
    """
    def __init__(self, object_id, frame_id, coordinate, confidence, worldcoordinate, feature_path, image_path=None):
        self.object_id = f"{object_id:08d}" # AKA serial number
        self.frame_id = frame_id
        self.feature_path = feature_path
        self.confidence = confidence
        self.image_path = image_path
        if isinstance(coordinate, Coordinate):
            self.coordinate = coordinate
        elif isinstance(coordinate, (list, tuple)) and len(coordinate) == 4:
            self.coordinate = Coordinate(*coordinate)
        else:
            raise Exception(f"Unknown coordinate format: {coordinate}")

        if isinstance(worldcoordinate, WorldCoordinate):
            self.worldcoordinate = worldcoordinate
        elif isinstance(worldcoordinate, (list, tuple)) and len(worldcoordinate) == 4:
            self.worldcoordinate = WorldCoordinate(*worldcoordinate)
        else:
            self.worldcoordinate = None

class Coordinate:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = int(float(x1))
        self.y1 = int(float(y1))
        self.x2 = int(float(x2))
        self.y2 = int(float(y2))

    def __str__(self):
        return(f'{{"x1":{self.x1}, "y1":{self.y1}, "x2":{self.x2}, "y2":{self.y2}}}')

class WorldCoordinate:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
    def __str__(self):
        return(f'{{"x":{self.x}, "y":{self.y}}}')


def get_camera_ids(scene_id, json_f="config/scene_2_camera_id_file.json"):
    with open(json_f) as f:
        scene2camera = json.load(f)
    camera_ids = []
    for scene_camera in scene2camera:
        if scene_camera["scene_name"] == f"scene_{scene_id:03d}":
            camera_ids = scene_camera["camera_ids"]
            break
    return camera_ids

def get_scene_id(camera_id, json_f="config/scene_2_camera_id_file.json"):
    with open(json_f) as f:
        scene2camera = json.load(f)
    for scene_camera in scene2camera:
        if camera_id in scene_camera["camera_ids"]:
            return int(scene_camera["scene_name"][6:])
    return -1




