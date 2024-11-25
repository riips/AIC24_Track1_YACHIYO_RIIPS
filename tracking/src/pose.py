import os
import numpy as np
import json
import cv2
import sys
from collections import defaultdict #, deque,Counter, 

from scipy.spatial import ConvexHull, QhullError  
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

from utils import ParameterManager, TrackingFunctions, JSONHandler, DirectoryStructure
from parameters import CommonParameters,PreprocessParameters

class UnifiedPoseFormat:
    def __init__(self, format_type="Coco"):
        self.format_type = format_type
     
        self.index2part = self._get_index2part(self.format_type)
        self.part2index = self._get_part2index(self.index2part)
        self.connection_info = self._get_connection_info(self.format_type)

    def _get_index2part(self,format_type: str):
        match format_type:
            case "Coco":
              index2part = {
                  0: ["nose", "mid"],
                  1: ["eye", "left"],
                  2: ["eye", "right"],
                  3: ["ear", "left"],
                  4: ["ear", "right"],
                  5: ["shoulder", "left"],
                  6: ["shoulder", "right"],
                  7: ["elbow", "left"],
                  8: ["elbow", "right"],
                  9: ["wrist", "left"],
                  10: ["wrist", "right"],
                  11: ["hip", "left"],
                  12: ["hip", "right"],
                  13: ["knee", "left"],
                  14: ["knee", "right"],
                  15: ["ankle", "left"],
                  16: ["ankle", "right"]}
            case "CrowdPose": 
              index2part = {
                  0: ["shoulder", "left"],
                  1: ["shoulder", "right"],
                  2: ["elbow", "left"],
                  3: ["elbow", "right"],
                  4: ["wrist", "left"],
                  5: ["wrist", "right"],
                  6: ["hip", "left"],
                  7: ["hip", "right"],
                  8: ["knee", "left"],
                  9: ["knee", "right"],
                  10: ["ankle", "left"],
                  11: ["ankle", "right"],
                  12: ["nose", "mid"],
                  13: ["neck", "mid"]}
            case _:
              raise ValueError(f"Unsupported format: {format_type}")
        return index2part

    def _get_part2index(self,index2part):

        part2index = defaultdict(list)
        for index, (body_part,position) in index2part.items():
            part2index[body_part].append(index)

        body_regions = {"head":["nose","eye","ear"],
                        "torso":["shoulder","hip"],
                        "leg":["knee","ankle"],
                        "arm":["elbow","wrist"]}
        
        for body_region, body_parts in body_regions.items():
            for body_part in body_parts:
                part2index[body_region].extend(part2index[body_part])

        for body_region in ["left","right"]:
            for index, (body_part,position) in index2part.items():
                if position in [body_region,"mid"]:
                    part2index[body_region].append(index)

        return dict(part2index)

    def _get_connection_info(self,format_type: str):
        """
        keypointのjointとなるindexを取得する
        """
        match format_type:
            case "Coco":
              connection_info = {
                  0: [1,2],
                  1: [0,2,3],
                  2: [0,1,4],
                  3: [1,5],
                  4: [2,6],
                  5: [3,6,7,11],
                  6: [4,5,8,12],
                  7: [5,9],
                  8: [6,10],
                  9: [7],
                  10: [8],
                  11: [5,12,13],
                  12: [6,11,14],
                  13: [11,15],
                  14: [12,16],
                  15: [13],
                  16: [14]}
            case "CrowdPose":
                connection_info = {
                  0: [2,6,13],
                  1: [3,7,13],
                  2: [0,4],
                  3: [1,5],
                  4: [2],
                  5: [3],
                  6: [0,7,8],
                  7: [1,6,9],
                  8: [6,10],
                  9: [7,11],
                  10: [8],
                  11: [9],
                  12: [13],
                  13: [0,1,12]}
            case _:
                raise ValueError(f"Unsupported format: {format_type}")
        return connection_info

    def _get_value(self,dict_,key):
        value = dict_.get(key)
        if value is None:
            raise KeyError(f"Key {key} not found in dictionary")
        return value
    
    def get_indices(self,body_part):
        return self._get_value(self.part2index,body_part)

    def get_body_part(self,index):
        return self._get_value(self.index2part,index)

    def get_connection(self,index):
        return self._get_value(self.connection_info,index)

class PolygonProcessor:
    @staticmethod
    def check_is_inside(test_points, polygon_points):
        """
        test_pointsがポリゴン内に存在するか否かをTrue or Falseで返す。
        """
        try:
            hull = ConvexHull(polygon_points)
            sorted_polygon_points = [polygon_points[i] for i in hull.vertices]
        except QhullError:
            #polygon_pointsによってポリゴンが生成できない場合
            return None
        polygon = Polygon(sorted_polygon_points)
        is_inside_list = [polygon.covers(Point(point)) for point in test_points]
        return is_inside_list
    
    @staticmethod
    def calculate_intersect_ratio(query_bbox, gallery_bboxes):
        """
        gallery_bboxesをconcatした上で、queryとのintersect_ratioを測る
        """
        query_box = box(*query_bbox)
        gallery_boxes = [box(*bbox) for bbox in gallery_bboxes]
        gallery_union = unary_union(gallery_boxes)
        intersect = query_box.intersection(gallery_union).area
        query_area = query_box.area
        return intersect/query_area

class PoseAdaptor(CommonParameters,ParameterManager):
    def __init__(self, tracking_dict, common_params): 
        CommonParameters.__init__(self)
        ParameterManager.__init__(self)
        if common_params:
            self.update_params(common_params)

        pose_results = DirectoryStructure(common_params).get_pose_result()
        self.serial_dict = {}
        self.assign_serial_from_tracking_dict(tracking_dict=tracking_dict,keypoints=pose_results)

    def get_keypoints(self, serial:str):
        """
        This must be called after assign_serial_from_tracking_dict() was called,
        as it builds a dictionary with "serial" number as keys.
        
        serial: zero-filled 8-digit string
        """
        serial = f"{str(serial).zfill(self.bbox_digit)}"

        if len(self.serial_dict) == 0:
            raise Exception(f"Serial based dictionary is not built yet.")
        
        return self.serial_dict.get(serial)

    def _build_serial_dict(self, keypoints=None):
        """
        This must be called after assign_serial_from_tracking_dict() was called,
        as it builds a dictionary with "serial" number as keys.
        """
        if len(keypoints) == 0:
            return None
        serial_dict = {}
        if keypoints == None:
            keypoints = self.keypoints
        for i, frame in enumerate(keypoints):
            detections = keypoints[frame]
            for det in detections:
                if "serial" in det: 
                    serial = det["serial"]
                    if serial in serial_dict:
                        print(f"DUP in serial numbers!!")
                    else:
                        serial_dict[serial] = {"bbox": det["bbox"], "Keypoints": det["keypoints"]}
        self.serial_dict = serial_dict

        return self.serial_dict

    def assign_serial_from_tracking_dict(self, tracking_dict, keypoints=None):
        """
        tracking_dict: dictionary of tracking_dict or path to tracking_dict json file.
        """
        if keypoints == None:
            keypoints = self.keypoints

        tracking_coord = {}
        for serial,value in tracking_dict.items():
            td_coord = value["Coordinate"]
            td_frame = value["Frame"]
            key = f"{td_frame}_{td_coord['x1']}_{td_coord['y1']}_{td_coord['x2']}_{td_coord['y2']}"
            if key in tracking_coord:
                continue #raise Exception(f"DUP! {key}")
            tracking_coord[key] = serial
        for frame, detections in keypoints.items(): 
            for det in detections:
                bbox = det["bbox"]
                key = f"{int(frame)}_{int(bbox[0])}_{int(bbox[1])}_{int(bbox[2])}_{int(bbox[3])}"
                if key in tracking_coord:
                    det["serial"] = tracking_coord[key]
                else:
                    #print(f"No tracking found for bbox: {key}, {bbox}")
                    pass

        # Build dict with serial as key
        return self._build_serial_dict(keypoints=keypoints)

class IdentifiabilityEvaluator(TrackingFunctions,CommonParameters,PreprocessParameters,ParameterManager):
    def __init__(self,tracking_dict,common_params={}):
        CommonParameters.__init__(self)
        PreprocessParameters.__init__(self)
        ParameterManager.__init__(self)

        if common_params:
            self.update_params(common_params)
        self.tracking_dict = tracking_dict
        self.unified_pose_format = UnifiedPoseFormat(self.pose_format)
        self.keypoints_results = PoseAdaptor(tracking_dict, common_params)

    def eval_identifiabilities(self,tracking_dict):
        """
        各bboxの識別性を評価する
        """

        results = {}
        frames, all_serials = zip(*[(value["Frame"],serial) for serial,value in tracking_dict.items()])
        frame_serials_dict = self.make_dict(frames,all_serials)
        del frames, all_serials
        for frame,serials in frame_serials_dict.items():
            for i, serial in enumerate(serials):
                other_serials = [tmp_serial for tmp_serial in serials if serial != tmp_serial]
                results[serial] = self._eval_identifiability(serial,other_serials) 
        return results

    def _eval_identifiability(self,serial,other_serials,**kwargs):
        # evaluate identifiability from pose estimation results        
        kp = self.keypoints_results.get_keypoints(serial)
        if kp == None:
            condition, intersect_ratio, score, area = 4, 1 , 0, 0
            return {"intersect_ratio":intersect_ratio,"condition":condition,"score":score}
        
        x1,y1,x2,y2,bbox_confidence = kp["bbox"]
        intersect_ratio = self._eval_intersect_with_other_bboxes(x1,y1,x2,y2,other_serials)
        
        kp = self.minimize_occluded_kp(kp,other_serials) 
        
        condition,score = self.eval_keypoints(kp["Keypoints"])  
        area = int((x2-x1)*(y2-y1))
        score = score if type(score) == int else float(score) #.astype(np.float16)

        return {"intersect_ratio":intersect_ratio,"condition":condition,"score":score} #,"area":int(area)

    def _eval_intersect_with_other_bboxes(self,x1,y1,x2,y2,other_serials):
        """
        他のbboxとのintersect_ratioを計算する
        """
        other_bboxes = []
        for serial in other_serials:
            x1_,y1_,x2_,y2_ = self.tracking_dict[serial]["Coordinate"].values() 
            if y2_ > y2:
                other_bboxes.append((x1_,y1_,x2_,y2_))
        return PolygonProcessor.calculate_intersect_ratio((x1,y1,x2,y2),other_bboxes)

    def eval_keypoints(self,keypoints):
        """
        keypointによって識別性を評価する
        """

        x_list, y_list, scores = zip(*keypoints)
        scores = list(scores)
        if np.min(scores) >= self.keypoint_th:
            return 0, np.mean(scores)
        if max(scores) < self.keypoint_th:
            return 5, 0
        if not self.verify_pose_consistency(y_list):
            return 5, 0
            
        scores = self.check_hand_selfocclusion(x_list, y_list, scores)
        if not self.check_head_occlusion(y_list):
            scores = [max(score,self.keypoint_th) if i in head_indices else score for i,score in enumerate(scores) if (head_indices := self.unified_pose_format.get_indices("head"))]
        
        if np.min(scores) >= self.keypoint_th:
            score = np.mean(scores)
            condition = 1
        else:
            right_scores = [scores[idx] for idx in self.unified_pose_format.get_indices("right")]
            left_scores = [scores[idx] for idx in self.unified_pose_format.get_indices("left")]

            target_scores = left_scores if np.min(left_scores) > np.min(right_scores) else right_scores
            min_score = np.min(target_scores)
            score = np.mean(target_scores)
            if min_score >= self.keypoint_th:
                condition = 2
            else:
                face_scores = [scores[idx] for idx in self.unified_pose_format.get_indices("head")]
                count = sum(tmp_score >= self.keypoint_th for tmp_score in target_scores)
                #face_count = sum(tmp_score >= keypoint_th for tmp_score in face_scores)
                if count/len(target_scores) > 0.5 : #or face_count>2
                    condition = 3
                else: 
                    condition = 4
        return condition, score

    def check_head_occlusion(self, y_list):
        """
        顔のパーツの位置が肩より下、もしくは顔-肩と肩-尻距離の比によって顔のオクルージョンを判定する
        """

        face_y = max([y_list[idx] for idx in self.unified_pose_format.get_indices("head")])
        shoulder_y = min([y_list[idx] for idx in self.unified_pose_format.get_indices("shoulder")])
        hip_y = min([y_list[idx] for idx in self.unified_pose_format.get_indices("hip")])
        if (shoulder_y - face_y == 0) or (shoulder_y - hip_y == 0):
            is_occlusion = True
            return is_occlusion

        is_occlusion = True if face_y > shoulder_y or np.abs((shoulder_y - face_y)/(shoulder_y-hip_y)) < 0.2 else False
        return is_occlusion
    
    def check_hand_selfocclusion(self,x_list, y_list, scores):
        """
        胴体の内部に存在する腕のscoreを置換する
        torso_pointsでポリゴンが作れない場合はis_inside_listがNoneになる。
        ⇒Poseの推定エラーなのでscoreを0に置換する
        """

        torso_points = [(x_list[index],y_list[index]) for index in self.unified_pose_format.get_indices("torso")]
        arm_indices = self.unified_pose_format.get_indices("arm")
        arm_points = [(x_list[index],y_list[index]) for index in arm_indices]  

        is_inside_list = PolygonProcessor.check_is_inside(arm_points,torso_points)

        if is_inside_list is None:
            return len(scores)*[0]
        for index, is_inside in zip(arm_indices,is_inside_list):
            scores[index] = max(scores[index], self.keypoint_th) if is_inside else scores[index]
        return scores

    def verify_pose_consistency(self,y_list):
        """
        基本的にface座標は肩座標より小さく、肩座標は腰座標より小さい。
        ⇒その一貫性を確かめる
        """
        face_y = max([y_list[idx] for idx in self.unified_pose_format.get_indices("head")])
        shoulder_y = min([y_list[idx] for idx in self.unified_pose_format.get_indices("shoulder")])
        waist_y = min([y_list[idx] for idx in self.unified_pose_format.get_indices("hip")])

        if face_y >= shoulder_y or shoulder_y >= waist_y:
            return False
        else:
            return True
    
    def minimize_occluded_kp(self,kp,other_serials,**kwargs):
        """
        オクルージョンが発生しているにも関わらず、confidenceが上がることがある
        ⇒そのkeypointのconfidenceを0に置換する
        """
        determination_method = kwargs.get('determination_method', "keypoint")

        def _bbox2polygon(x1,y1,x2,y2):
            return [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]

        def check_bbox_overlap(bbox1, bbox2):
            x1_min, y1_min, x1_max, y1_max = bbox1
            x2_min, y2_min, x2_max, y2_max = bbox2
            return not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)

        x1,y1,x2,y2,bbox_confidence = kp["bbox"]
        x_list, y_list, scores = zip(*kp["Keypoints"])
        points = [(x,y) for x,y in zip(x_list,y_list)]

        for serial in other_serials:
            if determination_method == "bbox":
                x1_,y1_,x2_,y2_ = self.tracking_dict[serial]["Coordinate"].values()
                polygon_points = self._bbox2polygon(x1_,y1_,x2_,y2_)
            elif determination_method == "keypoint":
                other_kp = self.keypoints_results.get_keypoints(serial)
                x1_,y1_,x2_,y2_,bbox_confidence_ = other_kp["bbox"]
                polygon_points = [(x,y) for x,y,score in other_kp["Keypoints"]]
            else:
                raise ValueError("determination method")

            if y2_ < y2: #target bbox is not occluded
                continue
            if not check_bbox_overlap((x1,y1,x2,y2),(x1_,y1_,x2_,y2_)):
                continue

            is_inside_list = PolygonProcessor.check_is_inside(points, polygon_points)
            for i, is_inside in enumerate(is_inside_list):
                if is_inside:
                    kp["Keypoints"][i][2]=0 #0:x, 1:y, 2:confidence
        return kp



