from dataclasses import dataclass,field
import os 

@dataclass
class CommonParameters:
    image_size:tuple = (1920,1080)
    fps:int = 30
    scene_id:int = 12
    
    camera_id:int = 1 #dammy parameter
    camera_ids:list = field(default_factory=lambda:[1])
    
    det_model:str = "YoloX"
    emb_model:str = "osnet_x1_0"
    pose_model:str = "mmpose_hrnet"
    pose_format:str = "Coco" #Coco or CrowdPose

    scene_digit:int = 2
    camera_digit:int = 3
    bbox_digit:int = 8
    frame_digit:int = 6
    
    is_print_parameters:bool = False

@dataclass
class PreprocessParameters:
    keypoint_th:float = 0.75
    keypoint_condition_th:float = 1

@dataclass
class SCPTParameters:
    epsilon:float = 0.1
    metric:str = "cosine"
    is_exclude_low_identifiable_img:bool = True
    is_minimize_noisy_img_similarity:bool = False

    frame_period:int = 90

    is_run_overlap_suppression:bool = True
    num_candidates:int = 10
    clustering_method:str = "agglomerative" #agglomerative or dbsacn
    min_samples:int = 4 #dbscan parameter

    is_use_motion:bool = True
    is_optimize_similarity_by_iou:bool = True
    iou_th = 0.60
    ratio_test_th = 0.9
    len_past_frame:int = 4

    is_set_speed_limit:bool = True

    is_debug:bool = False

@dataclass
class SCPTPostprocessParameters:
    is_concat_high_iou_tracklet:bool = True
    iou_th:float = 0.80
    len_past_frame:int = 5


    is_run_sequential_nms:bool = True
    temporally_snms_th:float = 0.6
    spatially_snms_th:float = 0.6
    merge_nonoverlap:bool = True

    is_run_separate_warp:bool = True
    warp_th:int = 40
    alpha:float = 0.5

    is_run_exclude_short_track:bool = False
    short_tracklet_th:int = 120

    is_run_exclude_motionless_track:bool = False        
    stop_track_th:int = 25

@dataclass
class MCPTParameters:
    epsilon:float = 0.4
    appearance_based_tracking:bool = True
    keypoint_th:float = 0.75
    keypoint_condition_th:float = 1
    
    is_use_motion_feature:bool = False
    check_sc_overlap:bool = False
    distance_type:str = "max" #max or mean or min
    replace_similarity_by_wcoordinate:bool = False
    distance_th:int = 5
    replace_value: float = -10

    representative_selection_method:str = "keypoint" #keypoint or centrality
    aspect_th:float =0.5 
    representative_criterion:str = "min_intersect"

    resample_representative:bool = False
    save_representative_json:bool = True
    # return_keypoint:bool = False
    
@dataclass
class MCPTPostprocessParameters:
    is_reassign_global_id:bool = True
    reassign_criterion:str = "mode"
    # short_track_th:int = 120
    delete_gid_th:int = 6000 
    assign_all_tracklet:bool = False
    sim_th:float = 0.75

    is_delete_few_camera_cluster:bool = False 
    min_camera_appearance:int = 1
    
    measure_wcoordinate:bool = False

    remove_noise_image:bool = True

    delete_distant_person:bool = True
    distance_th:int = 8

    interpolate_track:bool = True
    frame_sampling_freq:int = 1
    max_interpolate_interval:int = 15    

